from dataclasses import dataclass
from typing import Any
import re

from pydantic import BaseModel, ConfigDict, Field

from app.llm.model_router import (
    ModelRouter,
    create_default_model_router,
)


class LogicalVerificationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    missing_information: list[str] = Field(
        default_factory=list,
    )

    suggested_revision: str | None = None


@dataclass(frozen=True)
class LogicalVerificationResult:
    relevant: bool
    confidence: float
    reason: str
    missing_information: list[str]
    suggested_revision: str | None


class LogicalVerifier:

    def __init__(
        self,
        model_router: ModelRouter | None = None,
    ):
        self.model_router = (
            model_router
            or create_default_model_router()
        )

    def verify(
        self,
        question: str,
        objective: str,
        sql: str,
        schema_context: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        pagination_complete: bool = False,
        attempt_number: int = 1,
    ) -> LogicalVerificationResult:

        question_text = question.lower().strip()
        objective_text = objective.lower().strip()
        sql_text = sql.lower().strip()

        result_columns = {
            str(column).lower()
            for row in rows
            for column in row.keys()
        }

        # The step objective is the primary semantic contract. The
        # original question remains broader context and is used by the
        # strict semantic judge to ensure that an intermediate step has
        # not silently changed the requested scope.
        direct_step = any(
            phrase in objective_text
            for phrase in (
                "directly answer",
                "directly retrieve",
                "answer the original user question",
                "answer the requested question",
            )
        )

        intent_text = (
            question_text
            if direct_step or not objective_text
            else objective_text
        )

        asks_count = self._asks_for_count(intent_text)
        asks_unique = self._asks_for_unique(intent_text)
        asks_comparison = self._asks_for_comparison(intent_text)
        asks_top_results = self._asks_for_top_results(intent_text)

        ranking_validation = self._validate_ranking(
            intent_text,
            sql_text,
        )

        if not ranking_validation["valid"]:
            return LogicalVerificationResult(
                relevant=False,
                confidence=0.0,
                reason=ranking_validation["reason"],
                missing_information=[
                    ranking_validation["missing"]
                ],
                suggested_revision=ranking_validation["revision"],
            )

        question_tokens = set(
            self._tokens(
                intent_text
            )
        )

        scope_tokens = set(
            self._tokens(
                question_text
                + " "
                + objective_text
            )
        )

        relationship_terms = {
            "ortholog",
            "orthologs",
            "relationship",
            "relationships",
            "mapping",
            "mappings",
            "mapped",
            "association",
            "associations",
            "linked",
            "reference",
        }

        asks_relationship = bool(
            question_tokens & relationship_terms
        )

        schema_text = self._schema_text(
            schema_context
        ).lower()

        sql_tokens = set(self._tokens(sql_text))
        schema_tokens = set(self._tokens(schema_text))
        result_tokens = set(
            self._tokens(" ".join(result_columns))
        )

        sql_has_select = sql_text.startswith("select")
        has_result_columns = bool(result_columns)

        sql_has_count = (
            "count(" in sql_text
            or "count (" in sql_text
        )

        sql_has_distinct = "distinct" in sql_text
        sql_has_group_by = "group by" in sql_text
        sql_has_order_by = "order by" in sql_text
        sql_has_join = " join " in f" {sql_text} "

        # An empty result can be semantically valid, but a relationship
        # aggregate still needs scope validation. Do not bypass the
        # strict semantic path merely because there are no rows.
        if not rows and not (
            asks_relationship and asks_count
        ):
            return LogicalVerificationResult(
                relevant=True,
                confidence=0.70,
                reason=(
                    "The query executed successfully but returned "
                    "no rows. The empty result may represent a "
                    "valid negative result."
                ),
                missing_information=[],
                suggested_revision=None,
            )

        entity_terms = {
            "gene",
            "genes",
            "human",
            "symbol",
            "pathway",
            "pathways",
            "id",
        }

        entity_hits = len(
            question_tokens
            & entity_terms
            & (
                schema_tokens
                | sql_tokens
                | result_tokens
            )
        )

        relationship_hits = len(
            question_tokens
            & relationship_terms
            & (
                schema_tokens
                | sql_tokens
                | result_tokens
            )
        )

        aggregation_match = asks_count and sql_has_count
        distinct_match = asks_unique and sql_has_distinct

        relationship_match = (
            asks_relationship
            and (
                relationship_hits > 0
                or sql_has_join
                or bool(question_tokens & {"ortholog", "orthologs"})
                and "ortholog" in sql_tokens
            )
        )

        comparison_match = (
            asks_comparison
            and (
                sql_has_group_by
                or sql_has_order_by
                or sql_has_join
            )
        )

        ranking_match = (
            asks_top_results
            and ranking_validation["valid"]
            and sql_has_order_by
        )

        schema_sql_overlap = schema_tokens & sql_tokens
        schema_overlap_score = min(
            0.15,
            0.03 * len(schema_sql_overlap),
        )

        result_question_match = (
            self._result_matches_question(
                question_tokens,
                result_tokens,
            )
        )

        score = 0.0

        if sql_has_select:
            score += 0.15

        if has_result_columns:
            score += 0.15

        if entity_hits > 0:
            score += min(
                0.15,
                0.05 * entity_hits,
            )

        if relationship_hits > 0:
            score += min(
                0.10,
                0.05 * relationship_hits,
            )

        score += schema_overlap_score

        if result_question_match:
            score += 0.10

        if aggregation_match:
            score += 0.15

        if distinct_match:
            score += 0.10

        if relationship_match:
            score += 0.10

        if comparison_match:
            score += 0.10

        if ranking_match:
            score += 0.15

        deterministic_confidence = min(
            0.99,
            round(score, 2),
        )

        # ==================================================
        # STRICT SEMANTIC CHECK
        #
        # Relationship + aggregate questions are intentionally
        # NOT accepted by token overlap / COUNT detection alone.
        #
        # This prevents a query from counting a narrower or
        # differently-scoped relationship representation and
        # presenting it as the answer to the original question.
        #
        # This is generic: no table, column, entity, tenant, or
        # database-specific knowledge is embedded here.
        # ==================================================

        strict_relationship_aggregate = (
            asks_relationship
            and asks_count
        )

        if strict_relationship_aggregate:
            return self._strict_relationship_aggregate_check(
                question=question,
                objective=objective,
                sql=sql,
                schema_context=schema_context,
                rows=rows,
                deterministic_confidence=deterministic_confidence,
                pagination_complete=pagination_complete,
                attempt_number=attempt_number,
            )

        # ==================================================
        # Bounded collection completeness
        #
        # Security keeps SELECTs bounded. For a list/collection
        # objective, a full first page is not by itself proof that
        # the complete population was returned. InvestigationEngine
        # may establish completeness through deterministic pagination.
        # ==================================================

        collection_request = self._is_collection_request(intent_text)
        limit_match = re.search(
            r"\blimit\s+(\d+)\b",
            sql_text,
            flags=re.IGNORECASE,
        )

        if (
            collection_request
            and limit_match
            and len(rows) >= int(limit_match.group(1))
            and not pagination_complete
        ):
            return LogicalVerificationResult(
                relevant=False,
                confidence=0.0,
                reason=(
                    "The investigation step requests a collection, but "
                    "the executed query returned a full bounded page "
                    "without establishing that the complete requested "
                    "population was retrieved."
                ),
                missing_information=[
                    "Complete collection coverage beyond the SQL LIMIT."
                ],
                suggested_revision=(
                    "Use a deterministic ORDER BY and allow the "
                    "investigation engine to paginate the bounded result "
                    "until the final page is reached."
                ),
            )

        # ==================================================
        # Strong deterministic evidence
        #
        # These shortcuts remain valid for non-ambiguous intents.
        # ==================================================

        count_entity_signal = (
            any(
                token in sql_tokens
                for token in {
                    "gene",
                    "genes",
                    "gene_id",
                    "symbol",
                    "pathway",
                    "pathways",
                    "user",
                    "users",
                    "customer",
                    "customers",
                    "record",
                    "records",
                    "transaction",
                    "transactions",
                    "id",
                }
            )
            or any(
                token in schema_tokens
                for token in {
                    "gene",
                    "genes",
                    "gene_id",
                    "pathway",
                    "pathways",
                    "user",
                    "customer",
                    "record",
                    "transaction",
                    "id",
                }
            )
        )

        direct_count_answer = (
            sql_has_select
            and has_result_columns
            and aggregation_match
            and (
                entity_hits > 0
                or count_entity_signal
            )
        )

        direct_relationship_answer = (
            sql_has_select
            and has_result_columns
            and relationship_match
            and entity_hits > 0
        )

        direct_comparison_answer = (
            sql_has_select
            and has_result_columns
            and comparison_match
            and entity_hits > 0
        )

        direct_ranking_answer = (
            sql_has_select
            and has_result_columns
            and ranking_match
            and entity_hits > 0
        )

        if direct_ranking_answer:
            return LogicalVerificationResult(
                relevant=True,
                confidence=deterministic_confidence,
                reason=(
                    "The executed SELECT directly matches the "
                    "requested ranking operation, including the "
                    "required ORDER BY direction."
                ),
                missing_information=[],
                suggested_revision=None,
            )

        if direct_count_answer:
            return LogicalVerificationResult(
                relevant=True,
                confidence=deterministic_confidence,
                reason=(
                    "The executed SELECT directly matches the "
                    "requested count operation and uses relevant "
                    "database entities."
                ),
                missing_information=[],
                suggested_revision=None,
            )

        if direct_relationship_answer:
            return LogicalVerificationResult(
                relevant=True,
                confidence=deterministic_confidence,
                reason=(
                    "The executed SELECT returns relationship data "
                    "relevant to the investigation objective and "
                    "contains the requested entities."
                ),
                missing_information=[],
                suggested_revision=None,
            )

        if direct_comparison_answer:
            return LogicalVerificationResult(
                relevant=True,
                confidence=deterministic_confidence,
                reason=(
                    "The executed SELECT contains comparison "
                    "structure and relevant evidence matching "
                    "the investigation objective."
                ),
                missing_information=[],
                suggested_revision=None,
            )

        if (
            sql_has_select
            and has_result_columns
            and entity_hits > 0
            and deterministic_confidence >= 0.65
        ):
            return LogicalVerificationResult(
                relevant=True,
                confidence=deterministic_confidence,
                reason=(
                    "The executed SELECT returned data with "
                    "multiple evidence signals matching the "
                    "investigation objective."
                ),
                missing_information=[],
                suggested_revision=None,
            )

        return self._general_llm_check(
            question=question,
            objective=objective,
            sql=sql,
            schema_context=schema_context,
            rows=rows,
            result_columns=result_columns,
            deterministic_confidence=deterministic_confidence,
            attempt_number=attempt_number,
        )

    # ======================================================
    # Strict semantic verification
    # ======================================================

    def _strict_relationship_aggregate_check(
        self,
        question: str,
        objective: str,
        sql: str,
        schema_context: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        deterministic_confidence: float,
        pagination_complete: bool = False,
        attempt_number: int = 1,
    ) -> LogicalVerificationResult:

        system_instruction = """
You are the strict semantic verifier for DB-Sentinel.

Your job is to determine whether an executed SQL result is
semantically equivalent to what the investigation question asks.

The SQL has already passed deterministic security validation.
Do not evaluate SQL safety.

Do not decide based on keyword overlap, successful execution,
non-empty results, table-name similarity, or the presence of
COUNT alone.

For aggregate relationship questions, verify ALL of the following:

1. TARGET ENTITY LEVEL
   Identify what real-world entity is being counted or aggregated.
   The SQL must count the same entity level requested by the question.

2. RELATIONSHIP SEMANTICS
   The SQL must represent the relationship requested by the
   question, not merely a related fact.

3. SCOPE COMPLETENESS
   The SQL must cover the requested population. Reject a query if
   it silently restricts the population to a narrower subset
   through an unrelated table, join, filter, path, category,
   status, or other condition that the question did not request.

4. REPRESENTATION EQUIVALENCE
   A database may contain multiple ways to represent related
   information. Accept an alternative representation only when
   the supplied schema, SQL, and evidence support that it is
   semantically equivalent to the requested relationship.

5. FILTER JUSTIFICATION
   Every substantive filter affecting the counted population must
   be justified by the question/objective or by a necessary part
   of representing the requested relationship.

6. AGGREGATE CORRECTNESS
   COUNT, COUNT DISTINCT, GROUP BY, and other aggregation choices
   must measure the requested entity without silently changing the
   meaning through duplication or narrowing.

7. RESULT EVIDENCE
   The returned columns and rows must support the exact claim that
   the investigation step is supposed to establish.

Important:
A query can be syntactically valid and still be semantically wrong.

Example of a generic failure:
If a question asks for the number of entities having relationship R,
a query that counts entities from a separate subset S where R happens
to be present is not equivalent unless the question explicitly asks
for entities in S.

When the query is too narrow, too broad, or represents a different
relationship, return relevant=false and explain the semantic gap.
Do not assume equivalence merely because the SQL returns a plausible
number.

If the evidence is insufficient to establish equivalence, return
relevant=false rather than guessing.

Return structured output only.
"""

        user_prompt = f"""
Logical verification attempt {attempt_number}.

Investigation question:
{question}

Investigation objective:
{objective}

Executed SQL:
{sql}

Schema context:
{schema_context}

Returned result columns:
{sorted(
    {
        str(column).lower()
        for row in rows
        for column in row.keys()
    }
)}

Returned rows:
{rows[:20]}

Pagination completeness:
{pagination_complete}

Determine whether this exact SQL result answers the requested
question at the requested semantic scope.

Pay particular attention to:
- what entity is counted,
- which relationship is represented,
- whether the population is complete,
- whether another table or predicate silently narrows the scope,
- whether the aggregation changes the requested meaning.

Do not infer equivalence from matching words alone.
"""

        decision = self.model_router.generate(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_model=LogicalVerificationDecision,
            complexity="fast",
            classification_text=question,
            attempt_number=attempt_number,
        )

        llm_confidence = max(
            0.0,
            min(
                1.0,
                float(decision.confidence),
            ),
        )

        # Strict checks deliberately require a stronger semantic
        # threshold than ordinary relevance checks. Ambiguity should
        # trigger regeneration rather than silently becoming evidence.
        strict_acceptance_threshold = 0.72

        if decision.relevant and (
            llm_confidence >= strict_acceptance_threshold
        ):
            final_confidence = round(
                min(
                    0.95,
                    deterministic_confidence * 0.35
                    + llm_confidence * 0.65,
                ),
                2,
            )

            return LogicalVerificationResult(
                relevant=True,
                confidence=final_confidence,
                reason=(
                    "The SQL result passed strict semantic "
                    "verification for the requested relationship "
                    "aggregate, including entity level and population "
                    "scope."
                    + (
                        f" {decision.reason}"
                        if decision.reason
                        else ""
                    )
                ),
                missing_information=[],
                suggested_revision=None,
            )

        missing_information = list(
            decision.missing_information
        )

        if not decision.relevant:
            reason = decision.reason
        else:
            reason = (
                "The semantic verifier could not establish the "
                "requested relationship and aggregate with sufficient "
                "confidence. The query must be regenerated or "
                "investigated further rather than accepted as evidence."
            )

        if not missing_information:
            missing_information = [
                (
                    "A stronger semantic proof is required that the "
                    "counted population exactly matches the scope "
                    "requested by the question."
                )
            ]

        suggested_revision = decision.suggested_revision

        if not suggested_revision:
            suggested_revision = (
                "Regenerate the query from the schema using the "
                "database representation that directly establishes "
                "the requested relationship at the requested entity "
                "scope, without unrelated narrowing predicates."
            )

        final_confidence = round(
            min(
                0.60,
                deterministic_confidence * 0.35
                + llm_confidence * 0.65,
            ),
            2,
        )

        return LogicalVerificationResult(
            relevant=False,
            confidence=final_confidence,
            reason=reason,
            missing_information=missing_information,
            suggested_revision=suggested_revision,
        )

    # ======================================================
    # General LLM relevance check
    # ======================================================

    def _general_llm_check(
        self,
        question: str,
        objective: str,
        sql: str,
        schema_context: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        result_columns: set[str],
        deterministic_confidence: float,
        attempt_number: int = 1,
    ) -> LogicalVerificationResult:

        system_instruction = """
You are the logical relevance verifier for DB-Sentinel.

Determine whether the executed SQL result actually provides
information needed by the investigation step.

Security validation has already been completed.
Semantic verification has already been completed.

Evaluate only question-to-result relevance.

Do not invent facts.
Do not treat successful SQL execution as proof of relevance.
Do not treat non-empty results as proof of relevance.
A valid empty result may be relevant.

IMPORTANT RANKING RULE:

If the question asks for:
- most
- highest
- top
- largest
- maximum
- rank
- ranking
- least
- lowest
- smallest
- minimum

the SQL MUST establish an ordering of the relevant metric.

A LIMIT without ORDER BY does NOT establish a ranking.

For:
- most
- highest
- top
- largest
- maximum
- rank
- ranking

the ranking metric must use DESC.

For:
- least
- lowest
- smallest
- minimum

the ranking metric must use ASC.

If the question explicitly asks for "top N",
the SQL should use LIMIT N.

Return structured output only.

IMPORTANT:
Your confidence value is only an auxiliary assessment.
DB-Sentinel independently calculates the final confidence.
Do not treat your confidence value as the final system confidence.
"""

        user_prompt = f"""
Logical verification attempt {attempt_number}.

Question:
{question}

Objective:
{objective}

SQL:
{sql}

Schema:
{schema_context}

Result columns:
{sorted(result_columns)}

Returned rows:
{rows[:20]}

Determine whether the result is relevant.
"""

        decision = self.model_router.generate(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_model=LogicalVerificationDecision,
            complexity="fast",
            classification_text=question,
            attempt_number=attempt_number,
        )

        llm_confidence = max(
            0.0,
            min(
                1.0,
                float(decision.confidence),
            ),
        )

        if decision.relevant:
            final_confidence = round(
                min(
                    0.85,
                    deterministic_confidence * 0.70
                    + llm_confidence * 0.30,
                ),
                2,
            )
        else:
            final_confidence = round(
                min(
                    0.50,
                    deterministic_confidence * 0.70
                    + llm_confidence * 0.30,
                ),
                2,
            )

        return LogicalVerificationResult(
            relevant=decision.relevant,
            confidence=final_confidence,
            reason=decision.reason,
            missing_information=decision.missing_information,
            suggested_revision=decision.suggested_revision,
        )

    # ======================================================
    # Ranking validation
    # ======================================================

    @staticmethod
    def _validate_ranking(
        question: str,
        sql: str,
    ) -> dict[str, Any]:

        question = question.lower()
        sql = sql.lower()

        asks_most = any(
            term in question
            for term in {
                "most",
                "highest",
                "largest",
                "maximum",
                "top",
                "rank",
                "ranking",
            }
        )

        asks_least = any(
            term in question
            for term in {
                "least",
                "lowest",
                "smallest",
                "minimum",
            }
        )

        if not asks_most and not asks_least:
            return {
                "valid": True,
                "reason": "",
                "missing": "",
                "revision": "",
            }

        order_match = re.search(
            r"\border\s+by\b(.+?)(?:\blimit\b|$)",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not order_match:
            direction = (
                "DESC"
                if asks_most
                else "ASC"
            )

            return {
                "valid": False,
                "reason": (
                    "The question asks for a ranking, but the SQL "
                    "has no ORDER BY clause. LIMIT alone cannot "
                    "establish which rows are the most or highest."
                ),
                "missing": (
                    "ORDER BY is required to establish the "
                    "requested ranking."
                ),
                "revision": (
                    f"Add ORDER BY <ranking_metric> {direction}"
                ),
            }

        order_clause = order_match.group(1).strip()
        normalized_order = order_clause.lower()

        expected_direction = (
            "DESC"
            if asks_most
            else "ASC"
        )

        opposite_direction = (
            "ASC"
            if expected_direction == "DESC"
            else "DESC"
        )

        has_explicit_direction = (
            re.search(
                r"\b(desc|asc)\b",
                normalized_order,
            )
            is not None
        )

        if not has_explicit_direction:
            return {
                "valid": False,
                "reason": (
                    "The ranking query contains ORDER BY but does "
                    "not explicitly specify the required "
                    f"{expected_direction} direction."
                ),
                "missing": (
                    f"Explicit {expected_direction} ordering is required."
                ),
                "revision": (
                    f"Use ORDER BY <ranking_metric> {expected_direction}"
                ),
            }

        if (
            re.search(
                rf"\b{opposite_direction.lower()}\b",
                normalized_order,
            )
            and not re.search(
                rf"\b{expected_direction.lower()}\b",
                normalized_order,
            )
        ):
            return {
                "valid": False,
                "reason": (
                    f"The question requests the "
                    f"{'highest/most' if asks_most else 'lowest/least'} "
                    f"values, but the SQL orders the metric in the "
                    "opposite direction."
                ),
                "missing": (
                    f"ORDER BY must use {expected_direction} for "
                    "this ranking request."
                ),
                "revision": (
                    f"Change the query to ORDER BY "
                    f"<ranking_metric> {expected_direction}"
                ),
            }

        top_n_match = re.search(
            r"\btop\s+(\d+)\b",
            question,
            flags=re.IGNORECASE,
        )

        if top_n_match:
            requested_n = int(top_n_match.group(1))

            limit_match = re.search(
                r"\blimit\s+(\d+)\b",
                sql,
                flags=re.IGNORECASE,
            )

            if not limit_match:
                return {
                    "valid": False,
                    "reason": (
                        f"The question asks for the top {requested_n} "
                        "results, but the SQL has no LIMIT."
                    ),
                    "missing": (
                        f"LIMIT {requested_n} is required for the "
                        "requested Top-N result."
                    ),
                    "revision": f"Add LIMIT {requested_n}",
                }

            actual_limit = int(limit_match.group(1))

            if actual_limit != requested_n:
                return {
                    "valid": False,
                    "reason": (
                        f"The question asks for the top {requested_n} "
                        f"results, but the SQL uses LIMIT {actual_limit}."
                    ),
                    "missing": (
                        f"LIMIT {requested_n} is required."
                    ),
                    "revision": (
                        f"Change the query to LIMIT {requested_n}"
                    ),
                }

        return {
            "valid": True,
            "reason": "",
            "missing": "",
            "revision": "",
        }

    # ======================================================
    # Intent helpers
    # ======================================================

    @staticmethod
    def _asks_for_count(
        question: str,
    ) -> bool:

        return bool(
            re.search(
                r"\b(?:how many|number of|count(?: of)?|total number|total count)\b",
                question or "",
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _asks_for_unique(
        question: str,
    ) -> bool:

        unique_terms = {
            "unique",
            "distinct",
            "different",
            "each",
            "individual",
        }

        return any(
            term in question
            for term in unique_terms
        )

    @staticmethod
    def _asks_for_comparison(
        question: str,
    ) -> bool:

        comparison_terms = {
            "compare",
            "comparison",
            "versus",
            "vs",
            "difference",
            "differences",
            "across",
        }

        return any(
            term in question
            for term in comparison_terms
        )

    @staticmethod
    def _asks_for_top_results(
        question: str,
    ) -> bool:

        return bool(
            re.search(
                r"\b(?:top|highest|most|largest|maximum|rank|ranking|least|lowest|smallest|minimum)\b",
                question or "",
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_collection_request(
        text: str,
    ) -> bool:
        normalized = (text or "").lower()
        if re.search(
            r"\b(?:top|highest|most|largest|maximum|least|lowest|smallest|minimum|rank|ranking)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return False

        return bool(
            re.search(
                r"\b(?:which|list|show|return|find|identify)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            and re.search(
                r"\b(?:all|each|every|records?|genes?|pathways?|entities|items|results)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    # ======================================================
    # Token helpers
    # ======================================================

    @staticmethod
    def _tokens(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"[a-z0-9_]+",
            text.lower(),
        )

    @staticmethod
    def _result_matches_question(
        question_tokens: set[str],
        result_tokens: set[str],
    ) -> bool:

        meaningful = {
            token
            for token in question_tokens
            if len(token) >= 4
        }

        if not meaningful:
            return False

        overlap = meaningful & result_tokens

        return len(overlap) >= 1

    # ======================================================
    # Schema extraction
    # ======================================================

    @staticmethod
    def _schema_text(
        schema_context: list[dict[str, Any]],
    ) -> str:

        parts = []

        for table in schema_context:

            parts.append(
                str(
                    table.get(
                        "schema",
                        "",
                    )
                )
            )

            parts.append(
                str(
                    table.get(
                        "table",
                        "",
                    )
                )
            )

            for column in table.get(
                "columns",
                [],
            ):

                if isinstance(column, dict):
                    parts.append(
                        str(
                            column.get(
                                "name",
                                column.get(
                                    "column",
                                    "",
                                ),
                            )
                        )
                    )
                else:
                    parts.append(str(column))

            for foreign_key in table.get(
                "foreign_keys",
                [],
            ):
                parts.append(str(foreign_key))

        return " ".join(parts)
