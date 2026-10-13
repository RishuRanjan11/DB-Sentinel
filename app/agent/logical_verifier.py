from dataclasses import dataclass, field
from datetime import date, datetime
import json
from typing import Any
import re

from pydantic import BaseModel, ConfigDict, Field

from app.agent.evidence import Evidence
from app.agent.question_normalizer import QuestionNormalizer
from app.agent.query_intent import AnalyticalQueryIntent
from app.agent.insight_generator import InsightGenerator
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
    requested_periods: list[int] = field(default_factory=list)
    observed_periods: list[int] = field(default_factory=list)
    unavailable_periods: list[int] = field(default_factory=list)


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
        query_language: str = "sql",
        query_intent: AnalyticalQueryIntent | dict[str, Any] | None = None,
    ) -> LogicalVerificationResult:

        question_text = question.lower().strip()
        objective_text = objective.lower().strip()
        sql_text = sql.lower().strip()
        if query_language == "mongodb":
            return self._verify_mongodb(
                question=question,
                objective=objective,
                query=sql,
                schema_context=schema_context,
                rows=rows,
                pagination_complete=pagination_complete,
                attempt_number=attempt_number,
                query_intent=query_intent,
            )

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
        intent_data = (
            query_intent.model_dump(mode="json")
            if isinstance(query_intent, AnalyticalQueryIntent)
            else query_intent or {}
        )
        intent_ranking = intent_data.get("kind") == "ranking"
        intent_metrics = intent_data.get("metrics", [])
        intent_metric_names = {
            str(value).lower()
            for metric in intent_metrics
            if isinstance(metric, dict)
            for value in (metric.get("alias"), metric.get("field"))
            if value
        } if isinstance(intent_metrics, list) else set()
        intent_sorts = intent_data.get("sort", [])
        intent_metric_sort = next(
            (
                item
                for item in intent_sorts
                if isinstance(item, dict)
                and item.get("direction") in {"asc", "desc"}
                and (
                    not intent_metric_names
                    or str(item.get("field", "")).lower()
                    in intent_metric_names
                )
            ),
            None,
        )

        ranking_validation = self._validate_ranking(
            f"{question_text} {objective_text}",
            sql_text,
            dialect={
                "sqlserver": "tsql",
                "mssql": "tsql",
                "sql": "tsql",
                "sql server": "tsql",
                "microsoft sql server": "tsql",
                "mysql": "mysql",
                "postgresql": "postgres",
            }.get(query_language, "postgres"),
            intent_ranking=intent_ranking,
            intent_direction=(
                intent_metric_sort.get("direction")
                if intent_metric_sort
                else None
            ),
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
        scalar_count_match = (
            "single aggregate result" in objective_text
            and asks_count
            and has_result_columns
            and len(rows) == 1
            and re.search(
                r"\b(?:count|sum)\s*\(",
                sql_text,
            ) is not None
            and not sql_has_group_by
            and re.search(r"\bover\s*\(", sql_text) is None
        )

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
            (
                r"\btop\s*\(?\s*(\d+)"
                if query_language in {
                    "tsql",
                    "sqlserver",
                    "mssql",
                    "sql",
                    "sql server",
                    "microsoft sql server",
                }
                else r"\blimit\s+(\d+)\b"
            ),
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
            and (
                scalar_count_match
                or (
                    aggregation_match
                    and (
                        entity_hits > 0
                        or count_entity_signal
                    )
                )
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
                confidence=max(
                    deterministic_confidence,
                    0.75 if scalar_count_match else 0.0,
                ),
                reason=(
                    "The executed SELECT returns one ungrouped scalar "
                    "aggregate for the requested count."
                    if scalar_count_match
                    else
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

    def _verify_mongodb(
        self,
        question: str,
        objective: str,
        query: str,
        schema_context: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        pagination_complete: bool,
        attempt_number: int,
        query_intent: AnalyticalQueryIntent | dict[str, Any] | None,
    ) -> LogicalVerificationResult:
        try:
            payload = json.loads(query)
        except json.JSONDecodeError as error:
            return LogicalVerificationResult(
                relevant=False,
                confidence=0.0,
                reason=f"MongoDB aggregation JSON is invalid: {error}",
                missing_information=["A valid MongoDB aggregation pipeline."],
                suggested_revision="Regenerate a valid aggregation pipeline.",
            )
        pipeline = payload.get("pipeline") if isinstance(payload, dict) else None
        if not isinstance(pipeline, list) or not pipeline:
            return LogicalVerificationResult(
                relevant=False,
                confidence=0.0,
                reason="MongoDB aggregation pipeline is missing.",
                missing_information=["A MongoDB aggregation pipeline."],
                suggested_revision="Return a bounded aggregation pipeline.",
            )

        relative_period_issue = self._relative_period_pipeline_issue(
            question,
            pipeline,
            schema_context,
        )
        if relative_period_issue:
            return LogicalVerificationResult(
                relevant=False,
                confidence=0.0,
                reason=relative_period_issue,
                missing_information=[
                    "Raw metric values grouped by the latest available years."
                ],
                suggested_revision=(
                    "Do not add a rolling date filter or use $$NOW. Group "
                    "the raw metric by calendar year from a date field in "
                    "the live schema (or by its existing year field), sort "
                    "the grouped years newest first, and return one row per "
                    f"latest year with a final limit of "
                    f"{QuestionNormalizer.relative_comparison_year_count(question)}."
                ),
            )

        ranking_issue = self._mongo_ranking_pipeline_issue(
            question,
            objective,
            pipeline,
            query_intent=query_intent,
        )
        if ranking_issue:
            return LogicalVerificationResult(
                relevant=False,
                confidence=0.0,
                reason=ranking_issue,
                missing_information=[
                    "Results ordered by the requested aggregate metric."
                ],
                suggested_revision=(
                    "Sort first by the requested grouped metric in the "
                    "requested direction, then use the grouped entity as "
                    "the stable secondary key."
                ),
            )

        period_text = f"{question} {objective}"
        requested_years = self._explicit_years(period_text)
        requires_period_rows = (
            len(requested_years) >= 2
            and bool(
                re.search(
                    r"\b(?:compare|comparison|versus|vs\.?|against|"
                    r"change|changed|difference|delta|trend|growth|grow|"
                    r"decline|declined|increase|"
                    r"increased|decrease|decreased|percentage|percent)\b",
                    period_text,
                    re.IGNORECASE,
                )
            )
        )
        if requires_period_rows and rows:
            observed_years = sorted({
                year
                for row in rows
                if (
                    year := self._period_row_year(
                        row,
                        requested_years,
                    )
                ) is not None
            })
            if not observed_years:
                return LogicalVerificationResult(
                    relevant=False,
                    confidence=0.0,
                    reason=(
                        "The aggregation collapsed the requested yearly "
                        "values into a summary row. Return one row per "
                        "requested year with its raw metric value; "
                        "calculate the difference or percentage only "
                        "after verifying those yearly results."
                    ),
                    missing_information=[
                        f"The raw metric value for year {year}."
                        for year in requested_years
                    ],
                    suggested_revision=(
                        "Group the requested raw metric by year, return "
                        "the raw metric value and year in separate rows, "
                        "and do not collapse them into a "
                        "percentage/difference row."
                    ),
                )
            if (
                set(observed_years) != set(requested_years)
                and not self._covers_requested_years(
                    pipeline,
                    requested_years,
                )
            ):
                return LogicalVerificationResult(
                    relevant=False,
                    confidence=0.0,
                    reason=(
                        "The grouped result does not include every "
                        "requested year and the aggregation does not "
                        "demonstrate coverage of both requested periods."
                    ),
                    missing_information=[
                        f"A verified result for year {year}."
                        for year in requested_years
                        if year not in observed_years
                    ],
                    suggested_revision=(
                        "Filter across the complete requested year range, "
                        "group by year, and return the raw metric for each "
                        "available requested year."
                    ),
                )

        top_n = re.search(r"\btop\s+(\d+)\b", question, re.I)
        limit = (
            pipeline[-1].get("$limit")
            if isinstance(pipeline[-1], dict)
            else None
        )
        sort_specs = [
            stage["$sort"]
            for stage in pipeline
            if isinstance(stage, dict) and isinstance(stage.get("$sort"), dict)
        ]
        ranking_requested = bool(
            re.search(
                r"\b(?:most|highest|largest|maximum|top|rank|ranking|least|lowest|smallest|minimum)\b",
                question,
                re.I,
            )
        ) or (
            isinstance(query_intent, AnalyticalQueryIntent)
            and query_intent.kind == "ranking"
        ) or (
            isinstance(query_intent, dict)
            and query_intent.get("kind") == "ranking"
        )
        if ranking_requested:
            if not sort_specs:
                return LogicalVerificationResult(
                    relevant=False,
                    confidence=0.0,
                    reason="The requested MongoDB ranking has no $sort stage.",
                    missing_information=["A metric-based $sort stage."],
                    suggested_revision="Add a deterministic metric sort.",
                )
            if len(sort_specs[-1]) < 2:
                return LogicalVerificationResult(
                    relevant=False,
                    confidence=0.0,
                    reason="Ranking requires a stable secondary sort key.",
                    missing_information=["A deterministic tie-break sort key."],
                    suggested_revision=(
                        "Add the ranked entity's identifier as a secondary "
                        "$sort key."
                    ),
                )
            if top_n and limit != int(top_n.group(1)):
                return LogicalVerificationResult(
                    relevant=False,
                    confidence=0.0,
                    reason="The final MongoDB $limit does not match requested Top-N.",
                    missing_information=["The requested number of ranked rows."],
                    suggested_revision=(
                        f"Set the final $limit to {top_n.group(1)}."
                    ),
                )
        if (
            self._is_collection_request(question)
            and isinstance(limit, int)
            and len(rows) >= limit
            and not pagination_complete
        ):
            return LogicalVerificationResult(
                relevant=False,
                confidence=0.0,
                reason=(
                    "The MongoDB collection query reached its cap without "
                    "establishing complete coverage."
                ),
                missing_information=["Complete collection pagination."],
                suggested_revision=(
                    "Add a deterministic $sort including _id so the "
                    "investigation can paginate safely."
                ),
            )

        if pagination_complete and InsightGenerator._comparative_metric_insight(
            question,
            rows,
        ) is not None:
            return LogicalVerificationResult(
                relevant=True,
                confidence=0.95,
                reason=(
                    "The complete grouped result contains both requested "
                    "measures for each entity; the relative high/low "
                    "comparison can be calculated deterministically from "
                    "these verified rows."
                ),
                missing_information=[],
                suggested_revision=None,
            )

        trend_evidence = Evidence(
            step_id=1,
            question=question,
            objective=objective,
            sql=query,
            rows=rows,
            schema_context=schema_context,
            logical_reason="",
            confidence=1.0,
            pagination_complete=pagination_complete,
            total_rows=len(rows),
        )
        if InsightGenerator._deterministic_entity_trends(
            f"{question} {objective}",
            [trend_evidence],
        ) is not None:
            return LogicalVerificationResult(
                relevant=True,
                confidence=0.95,
                reason=(
                    "The verified annual metric values are grouped by entity "
                    "and year, and the requested consistent trends can be "
                    "calculated deterministically from those rows."
                ),
                missing_information=[],
                suggested_revision=None,
            )

        decision = self.model_router.generate(
            system_instruction="""
You are DB-Sentinel's semantic verifier for MongoDB aggregation results.
Decide whether the exact executed pipeline and returned evidence answer
the original question and objective. Verify entity level, relationship
semantics, filters, population scope, aggregation behavior, and result
columns. Do not infer correctness from successful execution or keyword
overlap. Treat uncertainty as not relevant. The aggregation has already
passed deterministic security checks.
""",
            user_prompt=f"""
Question:
{question}

Investigation objective:
{objective}

Executed aggregation:
{query}

Live schema:
{schema_context}

Returned columns:
{sorted({str(key).lower() for row in rows for key in row})}

Returned rows:
{rows[:20]}

Pagination complete:
{pagination_complete}

Does this exact result answer the requested question at the requested scope?
""",
            response_model=LogicalVerificationDecision,
            complexity="fast",
            classification_text=question,
            attempt_number=attempt_number,
        )
        requested_years = self._explicit_years(question)
        track_period_coverage = (
            len(requested_years) >= 2
            and bool(
                re.search(
                    r"\b(?:compare|comparison|versus|vs\.?|against|"
                    r"change|changed|difference|delta|trend|growth|grow|"
                    r"decline|declined|increase|increased|decrease|"
                    r"decreased|percentage|percent)\b",
                    f"{question} {objective}",
                    re.IGNORECASE,
                )
            )
        )
        tracked_years = requested_years if track_period_coverage else []
        observed_years = self._observed_years(rows, tracked_years)
        unavailable_years = sorted(set(tracked_years) - set(observed_years))
        if (
            unavailable_years
            and self._covers_requested_years(
                pipeline,
                requested_years,
            )
            and (
                decision.relevant
                or self._is_missing_requested_year_decision(decision)
            )
        ):
            return LogicalVerificationResult(
                relevant=True,
                confidence=0.80,
                reason=(
                    "The aggregation covered every requested year and "
                    "grouped by year, but returned matching data only for "
                    f"{observed_years or 'none'}; no result was returned for "
                    f"{unavailable_years}. The comparison cannot determine "
                    "an increase or decrease without data for every year."
                ),
                missing_information=[],
                suggested_revision=None,
                requested_periods=requested_years,
                observed_periods=observed_years,
                unavailable_periods=unavailable_years,
            )
        if not decision.relevant or decision.confidence < 0.72:
            return LogicalVerificationResult(
                relevant=False,
                confidence=decision.confidence,
                reason=decision.reason,
                missing_information=decision.missing_information,
                suggested_revision=decision.suggested_revision,
            )
        return LogicalVerificationResult(
            relevant=True,
            confidence=round(min(0.95, float(decision.confidence)), 2),
            reason=decision.reason,
            missing_information=[],
            suggested_revision=None,
            requested_periods=tracked_years,
            observed_periods=observed_years,
            unavailable_periods=unavailable_years,
        )

    @classmethod
    def _relative_period_pipeline_issue(
        cls,
        question: str,
        pipeline: list[dict[str, Any]],
        schema_context: list[dict[str, Any]],
    ) -> str | None:
        duration = QuestionNormalizer.relative_comparison_year_count(question)
        if (
            duration is None
            or QuestionNormalizer.relative_calendar_years(question) is not None
        ):
            return None

        group_index = next(
            (
                index
                for index, stage in enumerate(pipeline)
                if isinstance(stage, dict) and "$group" in stage
            ),
            None,
        )
        if group_index is None:
            return (
                "An unanchored relative-year comparison must group results "
                "by calendar year; the pipeline does not contain a $group."
            )
        if any("$limit" in stage for stage in pipeline[:group_index]):
            return (
                "The pipeline limits source records before grouping by year. "
                "Remove that pre-group limit so the newest-year totals use "
                "the complete matching dataset."
            )

        group_key = pipeline[group_index]["$group"].get("_id")
        group_fields = cls._mongo_field_references(group_key)
        derived_year_fields = {
            str(field).lower()
            for stage in pipeline[:group_index]
            if isinstance(stage, dict)
            for operator in ("$addFields", "$set")
            for fields in [stage.get(operator)]
            if isinstance(fields, dict)
            for field, expression in fields.items()
            if cls._mongo_contains_operator(expression, "$year")
        }
        year_fields = {
            str(column.get("name", column.get("column", ""))).lower()
            for table in schema_context
            for column in table.get("columns", [])
            if isinstance(column, dict)
            and re.search(
                r"\b(?:year|date|time)\b",
                str(column.get("name", column.get("column", ""))),
                re.IGNORECASE,
            )
        }
        year_fields.update(derived_year_fields)
        schema_date_fields = {
            str(column.get("name", column.get("column", ""))).lower()
            for table in schema_context
            for column in table.get("columns", [])
            if isinstance(column, dict)
            and re.search(
                r"\bdate\b",
                str(column.get("data_type", column.get("type", ""))),
                re.IGNORECASE,
            )
        }
        valid_year_group = (
            cls._mongo_contains_operator(group_key, "$year")
            or any(
                field.lstrip("$").lower() in year_fields
                for field in group_fields
            )
        )
        if not valid_year_group:
            return (
                "The relative-year aggregation does not group by a year "
                "or by a date/year field present in the live schema."
            )

        for stage in pipeline:
            if not isinstance(stage, dict):
                continue
            if cls._mongo_contains_operator(
                stage,
                "$dateSubtract",
            ) or cls._mongo_contains_text(stage, "$$now"):
                return (
                    "The pipeline uses a current-date rolling window for "
                    "an unanchored relative-year request. Select the latest "
                    "calendar years present in stored data instead."
                )
            match_spec = stage.get("$match")
            if match_spec is not None and cls._mongo_match_uses_fields(
                match_spec,
                schema_date_fields,
            ):
                return (
                    "The pipeline filters stored records by a date field "
                    "before finding the latest available years. Remove the "
                    "unrequested date-window filter and select the latest "
                    "years from the data."
                )

        year_sort_fields = cls._mongo_year_sort_fields(
            group_key,
            pipeline[group_index + 1:],
        )
        descending_year_sort = next(
            (
                stage
                for index, stage in enumerate(pipeline)
                if index > group_index
                and isinstance(stage, dict)
                and isinstance(stage.get("$sort"), dict)
                and stage["$sort"]
            ),
            None,
        )
        if (
            descending_year_sort is None
            or next(iter(descending_year_sort["$sort"])).lower()
            not in year_sort_fields
            or next(iter(descending_year_sort["$sort"].values())) != -1
        ):
            return (
                "After grouping by year, the pipeline must sort primarily "
                "by the grouped year descending before applying the limit. "
                "Sorting by the metric first selects the wrong years."
            )

        final_limit = (
            pipeline[-1].get("$limit")
            if isinstance(pipeline[-1], dict)
            else None
        )
        composite_year_group = (
            isinstance(group_key, dict)
            and len(group_key) > 1
            and any(
                re.search(r"year", str(group_name), re.IGNORECASE)
                or cls._mongo_contains_operator(expression, "$year")
                or (
                    isinstance(expression, str)
                    and expression.lstrip("$").lower()
                    in derived_year_fields
                )
                for group_name, expression in group_key.items()
            )
        )
        if composite_year_group:
            if final_limit == 100:
                return None
            return (
                "A location-by-year result must use the standard bounded "
                "result limit, not the number of years, so all entity/year "
                "rows can be collected and paginated."
            )
        if final_limit != duration:
            return (
                "The pipeline must return exactly one row for each of the "
                f"{duration} latest grouped years using a final $limit of "
                f"{duration}."
            )

        return None

    @classmethod
    def _mongo_ranking_pipeline_issue(
        cls,
        question: str,
        objective: str,
        pipeline: list[dict[str, Any]],
        query_intent: AnalyticalQueryIntent | dict[str, Any] | None = None,
    ) -> str | None:
        relative_duration = (
            QuestionNormalizer.relative_comparison_year_count(question)
        )
        if relative_duration is not None:
            for index, stage in enumerate(pipeline):
                if not isinstance(stage, dict) or not isinstance(
                    stage.get("$group"), dict
                ):
                    continue
                group_key = stage["$group"].get("_id")
                if cls._mongo_year_sort_fields(
                    group_key,
                    pipeline[index + 1:],
                ):
                    return None

        intent_data = (
            query_intent.model_dump(mode="json")
            if isinstance(query_intent, AnalyticalQueryIntent)
            else query_intent or {}
        )
        request_text = f"{question} {objective}"
        if (
            intent_data.get("kind") != "ranking"
            and not re.search(
            r"\b(?:most|highest|largest|maximum|max|top|rank|ranking|"
            r"least|lowest|smallest|minimum|min)\b",
            request_text,
            re.IGNORECASE,
            )
        ):
            return None

        group_indexes = [
            index
            for index, stage in enumerate(pipeline)
            if isinstance(stage, dict)
            and isinstance(stage.get("$group"), dict)
        ]
        if len(group_indexes) != 1:
            return None

        group_index = group_indexes[0]
        group = pipeline[group_index]["$group"]
        metric_fields = [
            str(name)
            for name, expression in group.items()
            if name != "_id" and cls._mongo_contains_aggregate(expression)
        ]
        if not metric_fields:
            return None
        metric_aliases = set(metric_fields)
        for stage in pipeline[group_index + 1:]:
            projection = (
                stage.get("$project")
                if isinstance(stage, dict)
                else None
            )
            if not isinstance(projection, dict):
                continue
            projected_aliases = {
                str(name)
                for name, expression in projection.items()
                if isinstance(expression, str)
                and expression[1:] in metric_aliases
            }
            if projected_aliases:
                metric_aliases = projected_aliases

        sort_stages = [
            stage["$sort"]
            for stage in pipeline[group_index + 1:]
            if isinstance(stage, dict)
            and isinstance(stage.get("$sort"), dict)
            and stage["$sort"]
        ]
        if not sort_stages:
            return "The grouped ranking has no sort on its aggregate metric."

        intent_metrics = intent_data.get("metrics", [])
        intent_metric_names = {
            str(value).lower()
            for metric in intent_metrics
            if isinstance(metric, dict)
            for value in (metric.get("alias"), metric.get("field"))
            if value
        } if isinstance(intent_metrics, list) else set()
        intent_sorts = intent_data.get("sort", [])
        intent_metric_sort = next(
            (
                item
                for item in intent_sorts
                if isinstance(item, dict)
                and item.get("direction") in {"asc", "desc"}
                and (
                    not intent_metric_names
                    or str(item.get("field", "")).lower()
                    in intent_metric_names
                )
            ),
            None,
        )
        if intent_metric_sort:
            metric_aliases.add(str(intent_metric_sort.get("field", "")).lower())
        direction_text = (
            question
            if re.search(
                r"\b(?:least|lowest|smallest|minimum|min|most|highest|"
                r"largest|maximum|max)\b",
                question,
                re.IGNORECASE,
            )
            else objective
        )
        if intent_metric_sort:
            expected_direction = (
                -1 if intent_metric_sort["direction"] == "desc" else 1
            )
        else:
            descending = not bool(
                re.search(
                    r"\b(?:least|lowest|smallest|minimum|min)\b",
                    direction_text,
                    re.IGNORECASE,
                )
            )
            expected_direction = -1 if descending else 1
        primary_field, primary_direction = next(
            iter(sort_stages[-1].items())
        )
        if (
            primary_field not in metric_aliases
            or primary_direction != expected_direction
        ):
            return (
                "The grouped ranking must sort primarily by its requested "
                "aggregate metric in the requested direction, not by the "
                "entity key."
            )
        if len(sort_stages[-1]) < 2:
            return (
                "The grouped ranking must include the entity key as a "
                "stable secondary sort."
            )
        return None

    @classmethod
    def _mongo_contains_aggregate(cls, value: Any) -> bool:
        aggregate_operators = {
            "$sum",
            "$avg",
            "$min",
            "$max",
            "$first",
            "$last",
            "$count",
            "$stddevpop",
            "$stddevsamp",
        }
        if isinstance(value, dict):
            return any(
                str(key).lower() in aggregate_operators
                or cls._mongo_contains_aggregate(nested)
                for key, nested in value.items()
            )
        if isinstance(value, list):
            return any(cls._mongo_contains_aggregate(item) for item in value)
        return False

    @classmethod
    def _mongo_year_sort_fields(
        cls,
        group_key: Any,
        trailing_stages: list[dict[str, Any]],
    ) -> set[str]:
        if isinstance(group_key, str) and re.search(
            r"\byear\b",
            group_key,
            re.IGNORECASE,
        ):
            fields = {"_id"}
        elif isinstance(group_key, dict):
            if "$year" in group_key:
                fields = {"_id"}
            else:
                year_keys = [
                    str(group_name)
                    for group_name, expression in group_key.items()
                    if re.search(r"year", str(group_name), re.IGNORECASE)
                    or cls._mongo_contains_operator(expression, "$year")
                ]
                if len(year_keys) == 1:
                    fields = {f"_id.{year_keys[0]}".lower()}
                elif year_keys:
                    return set()
                else:
                    return set()
        else:
            return set()

        current_fields = set(fields)
        for stage in trailing_stages:
            projection = stage.get("$project") if isinstance(stage, dict) else None
            if not isinstance(projection, dict):
                continue
            aliases = {
                str(name).lower()
                for name, value in projection.items()
                if isinstance(value, str)
                and value.lower()
                in {f"${field}" for field in current_fields}
                or (
                    isinstance(value, dict)
                    and cls._mongo_contains_operator(value, "$year")
                )
            }
            if aliases:
                current_fields = aliases
            elif (
                projection.get("_id", 1) == 0
                and any(
                    field == "_id" or field.startswith("_id.")
                    for field in current_fields
                )
            ):
                current_fields = set()
        return current_fields

    @classmethod
    def _mongo_field_references(cls, value: Any) -> set[str]:
        if isinstance(value, str):
            return {value} if value.startswith("$") else set()
        if isinstance(value, dict):
            return {
                field
                for nested in value.values()
                for field in cls._mongo_field_references(nested)
            }
        if isinstance(value, list):
            return {
                field
                for nested in value
                for field in cls._mongo_field_references(nested)
            }
        return set()

    @classmethod
    def _mongo_contains_operator(cls, value: Any, operator: str) -> bool:
        if isinstance(value, dict):
            return operator in value or any(
                cls._mongo_contains_operator(nested, operator)
                for nested in value.values()
            )
        if isinstance(value, list):
            return any(
                cls._mongo_contains_operator(nested, operator)
                for nested in value
            )
        return False

    @classmethod
    def _mongo_contains_text(cls, value: Any, text: str) -> bool:
        if isinstance(value, str):
            return text in value.lower()
        if isinstance(value, dict):
            return any(
                cls._mongo_contains_text(nested, text)
                for nested in value.values()
            )
        if isinstance(value, list):
            return any(
                cls._mongo_contains_text(nested, text)
                for nested in value
            )
        return False

    @classmethod
    def _mongo_match_uses_fields(
        cls,
        value: Any,
        date_fields: set[str],
    ) -> bool:
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized_key = str(key).lower()
                if not normalized_key.startswith("$") and any(
                    normalized_key == field
                    or normalized_key.startswith(f"{field}.")
                    or field.startswith(f"{normalized_key}.")
                    for field in date_fields
                ):
                    return True
                if cls._mongo_match_uses_fields(nested, date_fields):
                    return True
        elif isinstance(value, list):
            return any(
                cls._mongo_match_uses_fields(nested, date_fields)
                for nested in value
            )
        return False

    @staticmethod
    def _explicit_years(question: str) -> list[int]:
        return sorted({
            int(year)
            for year in re.findall(r"\b(?:19|20)\d{2}\b", question or "")
        })

    @staticmethod
    def _period_row_year(
        row: dict[str, Any],
        requested_years: list[int],
    ) -> int | None:
        year = row.get("year")
        if year is None:
            year = row.get("_id")
        if isinstance(year, bool):
            return None
        try:
            parsed = int(year)
        except (TypeError, ValueError):
            return None
        return parsed if parsed in requested_years else None

    @classmethod
    def _observed_years(
        cls,
        rows: list[dict[str, Any]],
        requested_years: list[int],
    ) -> list[int]:
        requested = set(requested_years)
        observed: set[int] = set()

        def collect(value: Any) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, int) and value in requested:
                observed.add(value)
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        for row in rows:
            collect(row.get("_id"))
        return sorted(observed)

    @staticmethod
    def _is_missing_requested_year_decision(
        decision: LogicalVerificationDecision,
    ) -> bool:
        text = " ".join([
            decision.reason,
            *decision.missing_information,
        ]).lower()
        return (
            "year" in text
            and bool(
                re.search(
                    r"\b(?:missing|not returned|not present|"
                    r"not included|does not include|only returned|"
                    r"absent|unavailable)\b",
                    text,
                )
            )
            and not bool(
                re.search(
                    r"\b(?:incorrect|wrong|invalid|rather than|"
                    r"instead of|does not calculate|fails to calculate)\b",
                    text,
                )
            )
        )

    @classmethod
    def _covers_requested_years(
        cls,
        pipeline: list[dict[str, Any]],
        requested_years: list[int],
    ) -> bool:
        if len(requested_years) < 2:
            return False

        required_start = date(min(requested_years), 1, 1)
        required_end = date(max(requested_years) + 1, 1, 1)

        def contains_year_expression(value: Any) -> bool:
            if isinstance(value, dict):
                return (
                    "$year" in value
                    or any(
                        contains_year_expression(child)
                        for child in value.values()
                    )
                )
            if isinstance(value, list):
                return any(contains_year_expression(child) for child in value)
            return False

        group_indices = [
            index
            for index, stage in enumerate(pipeline)
            if isinstance(stage, dict)
            and isinstance(stage.get("$group"), dict)
            and contains_year_expression(stage["$group"].get("_id"))
        ]
        match_indices = [
            index
            for index, stage in enumerate(pipeline)
            if isinstance(stage, dict) and "$match" in stage
        ]
        if (
            len(group_indices) != 1
            or len(match_indices) != 1
            or match_indices[0] >= group_indices[0]
            or any(
                isinstance(stage, dict)
                and ("$limit" in stage or "$skip" in stage)
                for stage in pipeline[:group_indices[0]]
            )
        ):
            return False

        match = pipeline[match_indices[0]]["$match"]
        if len(match) != 1:
            return False
        condition = next(iter(match.values()))
        if not isinstance(condition, dict) or not {
            "$gte",
            "$lt",
        }.issubset(condition):
            return False
        start = cls._extended_json_date(condition["$gte"])
        end = cls._extended_json_date(condition["$lt"])
        return (
            start is not None
            and end is not None
            and start <= required_start
            and end >= required_end
        )

    @staticmethod
    def _extended_json_date(value: Any) -> date | None:
        if isinstance(value, dict) and isinstance(value.get("$date"), str):
            value = value["$date"]
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).date()
        except ValueError:
            return None

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

IMPORTANT COLLECTION PAGINATION RULE:
For a request to list, show, find, or return all matching records,
LIMIT 100 may be the required per-page fetch bound. This verifier is
checking whether the returned page is semantically relevant; it is not
deciding whether the entire collection has been returned. Do not reject
an otherwise relevant collection query merely because it has LIMIT.
The investigation engine separately paginates ordered pages and only
accepts the collection after pagination establishes completeness.

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

For a collection request, treat LIMIT as a per-page fetch bound.
Pagination completeness is checked separately by the investigation
engine after this relevance check.

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
        dialect: str = "postgres",
        intent_ranking: bool = False,
        intent_direction: str | None = None,
    ) -> dict[str, Any]:

        question = question.lower()
        sql = sql.lower()

        asks_most = any(
            re.search(
                rf"\b{re.escape(term)}\b",
                question,
            )
            is not None
            for term in {
                "most",
                "more",
                "highest",
                "largest",
                "maximum",
                "top",
                "rank",
                "ranking",
            }
        )

        asks_least = any(
            re.search(
                rf"\b{re.escape(term)}\b",
                question,
            )
            is not None
            for term in {
                "least",
                "less",
                "lowest",
                "smallest",
                "minimum",
            }
        )
        if intent_ranking:
            asks_most = intent_direction == "desc"
            asks_least = intent_direction == "asc"

        if not asks_most and not asks_least:
            return {
                "valid": True,
                "reason": "",
                "missing": "",
                "revision": "",
            }

        order_match = re.search(
            r"\border\s+by\b(.+?)(?:\blimit\b|\boffset\b|$)",
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

        if "," not in order_clause:
            return {
                "valid": False,
                "reason": (
                    "The ranking has no deterministic secondary ordering "
                    "for tied metric values."
                ),
                "missing": "A stable entity tie-breaker in ORDER BY.",
                "revision": (
                    f"Add a stable entity tie-breaker after "
                    f"{expected_direction}, such as <entity_name> ASC."
                ),
            }

        top_n_match = re.search(
            r"\b(?:top|first|which)\s+(\d+)\b",
            question,
            flags=re.IGNORECASE,
        )
        all_extreme_ties = bool(
            re.search(
                r"\b(?:all|every)\s+(?:entities|genes|items|records|rows)?\s*"
                r"(?:that\s+are\s+)?tied\b|\ball\s+ties\b",
                question,
                flags=re.IGNORECASE,
            )
        )
        implicit_top_one = bool(
            not top_n_match
            and not all_extreme_ties
            and re.search(r"\bwhich\b", question)
            and re.search(
                r"\b(?:most|highest|largest|maximum|max|least|lowest|"
                r"smallest|minimum)\b",
                question,
            )
            and not re.search(
                r"\b(?:in each|across|for each|per)\b",
                question,
            )
        )

        requested_n = (
            int(top_n_match.group(1))
            if top_n_match
            else 1 if implicit_top_one
            else None
        )
        if requested_n is not None:

            limit_match = re.search(
                (
                    r"\btop\s*\(?\s*(\d+)"
                    if dialect == "tsql"
                    else r"\blimit\s+(\d+)\b"
                ),
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

        elif all_extreme_ties:
            limit_match = re.search(
                (
                    r"\btop\s*\(?\s*(\d+)"
                    if dialect == "tsql"
                    else r"\blimit\s+(\d+)\b"
                ),
                sql,
                flags=re.IGNORECASE,
            )
            if not limit_match or int(limit_match.group(1)) != 100:
                return {
                    "valid": False,
                    "reason": (
                        "The question requests every entity tied at the "
                        "extreme. Keep the source query within the 100-row "
                        "evidence cap."
                    ),
                    "missing": "Bounded all-ties query (LIMIT 100).",
                    "revision": (
                        "Use LIMIT 100 and retain the ranking order."
                    ),
                }

        elif (
            re.search(r"\bwhich\b", question)
            and re.search(r"\b(?:in each|across|for each|per)\b", question)
            and re.search(r"\b(?:most|highest|largest|maximum|least|lowest|smallest|minimum)\b", question)
        ):
            limit_match = re.search(
                (
                    r"\btop\s*\(?\s*(\d+)"
                    if dialect == "tsql"
                    else r"\blimit\s+(\d+)\b"
                ),
                sql,
                flags=re.IGNORECASE,
            )
            if not limit_match or int(limit_match.group(1)) != 100:
                return {
                    "valid": False,
                    "reason": (
                        "The question requests a ranked comparison across "
                        "the grouped entities without specifying a Top-N. "
                        "Return the complete ranked result within the "
                        "system row cap using LIMIT 100."
                    ),
                    "missing": "Complete ranked group coverage (LIMIT 100).",
                    "revision": (
                        "Use LIMIT 100 and retain the requested ranking order."
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
                r"\b(?:top|higher|highest|more|most|larger|largest|maximum|"
                r"rank|ranking|less|least|lower|lowest|smaller|smallest|minimum)\b",
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
                r"\b(?:all|each|every|unique|distinct|records?|genes?|pathways?|entities|items|results)\b",
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
