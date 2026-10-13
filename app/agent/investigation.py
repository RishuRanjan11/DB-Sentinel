import hashlib
import re
import uuid

from app.agent.planner import QuestionPlanner
from app.agent.question_normalizer import QuestionNormalizer
from app.agent.sql_pipeline import SQLPipeline
from app.agent.logical_verifier import LogicalVerifier
from app.agent.query_repair import QueryRepairAnalyzer
from app.agent.evidence import Evidence, EvidenceStore
from app.agent.insight_generator import InsightGenerator
from app.agent.insight_verifier import InsightVerifier
from app.database.manager import DatabaseManager
from app.database.schema_retriever import get_relevant_schema
from app.llm.model_router import create_default_model_router
from app.security.policies import RequestPolicy
from app.security.permissions import DatabasePermissions
from app.core.trace import logger
from app.analysis.intent import AnalysisIntent, AnalysisIntentClassifier
from app.analysis.explorer import DatabaseExplorer
from app.reporting.artifacts import build_artifacts
from app.database.data_dictionary import DatabaseDictionary


def _expected_top_n(question: str) -> int | None:
    explicit = re.search(
        r"\b(?:top|first|which)\s+(\d+)\b",
        question or "",
        flags=re.IGNORECASE,
    )
    if explicit:
        return int(explicit.group(1))
    if re.search(
        r"\b(?:all|every)\s+(?:entities|genes|items|records|rows)?\s*"
        r"(?:that\s+are\s+)?tied\b|\ball\s+ties\b",
        question or "",
        flags=re.IGNORECASE,
    ):
        return None
    if re.search(
        r"\b(?:across|among|in each|per)\b",
        question or "",
        flags=re.IGNORECASE,
    ):
        return None
    if re.search(
        r"\bwhich\b.*\b(?:more|most|highest|largest|maximum|max|"
        r"least|lowest|smallest|minimum)\b",
        question or "",
        flags=re.IGNORECASE,
    ):
        return 1
    return None


class InvestigationEngine:

    MAX_QUERY_ATTEMPTS = 3

    def __init__(
        self,
        database_manager: DatabaseManager,
        model_router=None,
    ):
        self.database_manager = database_manager
        self.request_policy = RequestPolicy()
        self.database_permissions = DatabasePermissions()

        self.model_router = (
            model_router
            or create_default_model_router()
        )

        self.planner = QuestionPlanner(
            model_router=self.model_router
        )
        self.question_normalizer = QuestionNormalizer(
            model_router=self.model_router
        )

        self.logical_verifier = LogicalVerifier(
            model_router=self.model_router
        )

        self.insight_generator = InsightGenerator(
            model_router=self.model_router
        )

        self.insight_verifier = InsightVerifier(
            model_router=self.model_router
        )

        self.intent_classifier = AnalysisIntentClassifier()
        self.database_explorer = DatabaseExplorer(
            model_router=self.model_router
        )

        self.sql_pipeline = SQLPipeline(
            database_manager=database_manager,
            model_router=self.model_router,
        )

    @staticmethod
    def _failed_required_steps(step_results: list[dict]) -> list[dict]:
        return [
            step
            for step in step_results
            if (
                step.get("status") != "verified"
                and not (
                    step.get("status") == "skipped"
                    and step.get("skip_reason") == "empty_dependency"
                )
                and not step.get("optional", False)
            )
        ]

    @staticmethod
    def _asks_for_all_extreme_ties(
        question: str,
        objective: str,
    ) -> bool:
        text = f"{question or ''} {objective or ''}"
        return bool(
            re.search(
                r"\b(?:maximum|minimum|highest|lowest|max|min)\b",
                text,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(?:all|every)\s+(?:entities|genes|items|records|rows)?\s*"
                r"(?:that\s+are\s+)?tied\b|\ball\s+ties\b",
                text,
                re.IGNORECASE,
            )
        )

    def investigate(
        self,
        connection_id: str,
        question: str,
        organization_id: str | None = None,
        conversation_history: list[dict] | None = None,
        data_dictionary: DatabaseDictionary | None = None,
    ) -> dict:
        investigation_id = uuid.uuid4().hex[:12]
        question_hash = hashlib.sha256(
            question.encode("utf-8")
        ).hexdigest()[:16]
        logger.info(
            "INVESTIGATION START id=%s organization=%s connection=%s "
            "question_sha256=%s dictionary=%s",
            investigation_id,
            organization_id or "unspecified",
            connection_id,
            question_hash,
            bool(data_dictionary),
        )

        # ==================================================
        # 0. TENANT / DATABASE BOUNDARY
        # ==================================================

        connection = self.database_manager.get_connection(
            connection_id
        )

        if (
            organization_id is not None
            and connection.organization_id != organization_id
        ):
            raise PermissionError(
                f"Database connection '{connection_id}' does not belong "
                f"to organization '{organization_id}'."
            )

        # ==================================================
        # 1. SECURITY POLICY
        # ==================================================

        policy_decision = self.request_policy.evaluate(
            question
        )

        permission_decision = (
            self.database_permissions.check(
                policy_decision.operation
            )
        )

        if (
            not policy_decision.allowed
            or not permission_decision.allowed
        ):

            reason = (
                policy_decision.reason
                if not policy_decision.allowed
                else permission_decision.reason
            )

            logger.warning(
                "INVESTIGATION BLOCKED id=%s operation=%s reason=%s",
                investigation_id,
                policy_decision.operation,
                reason,
            )

            return {
                "status": "blocked",
                "question": question,
                "plan": None,
                "steps": [],
                "evidence": [],
                "insight": None,
                "insight_verification": None,
                "reason": reason,
                "security": {
                    "allowed": False,
                    "operation": (
                        policy_decision.operation
                    ),
                    "reason": reason,
                },
            }

        normalized_question = self.question_normalizer.normalize(
            question,
            conversation_history=conversation_history,
        )
        normalized_policy = self.request_policy.evaluate(
            normalized_question
        )
        normalized_permission = self.database_permissions.check(
            normalized_policy.operation
        )
        if not normalized_policy.allowed or not normalized_permission.allowed:
            reason = (
                normalized_policy.reason
                if not normalized_policy.allowed
                else normalized_permission.reason
            )
            logger.warning(
                "NORMALIZED QUESTION BLOCKED id=%s operation=%s reason=%s",
                investigation_id,
                normalized_policy.operation,
                reason,
            )
            return {
                "status": "blocked",
                "question": question,
                "normalized_question": normalized_question,
                "plan": None,
                "steps": [],
                "evidence": [],
                "insight": None,
                "insight_verification": None,
                "reason": reason,
                "security": {
                    "allowed": False,
                    "operation": normalized_policy.operation,
                    "reason": reason,
                },
            }

        # ==================================================
        # 2. DATABASE EXPLORATION
        # ==================================================

        intent = self.intent_classifier.classify(normalized_question)
        logger.info(
            "INVESTIGATION CLASSIFIED id=%s intent=%s",
            investigation_id,
            intent.value,
        )

        if intent == AnalysisIntent.DATABASE_EXPLORATION:
            adapter = self.database_manager.get(
                connection_id,
                organization_id=organization_id,
            )
            overview = self.database_explorer.explore(
                question=normalized_question,
                adapter=adapter,
                data_dictionary=data_dictionary,
            )
            logger.info(
                "EXPLORATION COMPLETE id=%s schemas=%s tables=%s "
                "relationships=%s dictionary=%s status=verified",
                investigation_id,
                len(overview.schemas),
                len(overview.tables),
                len(overview.relationships),
                bool(data_dictionary),
            )
            evidence = [{
                "type": "database_metadata",
                "schemas": overview.schemas,
                "tables": overview.tables,
                "relationships": overview.relationships,
            }]
            dashboard = {
                "title": question,
                "widgets": [{
                    "type": "schema_overview",
                    "title": "Database Structure",
                    "data": {
                        "schemas": overview.schemas,
                        "tables": overview.tables,
                        "relationships": overview.relationships,
                    },
                    "source_step_ids": [],
                }],
            }
            report = {
                "title": question,
                "executive_summary": overview.summary,
                "findings": [],
                "recommendations": [],
                "uncertainties": overview.limitations,
                "evidence": evidence,
            }
            return {
                "status": "verified",
                "question": question,
                "normalized_question": normalized_question,
                "intent": intent.value,
                "plan": None,
                "steps": [],
                "evidence": evidence,
                "insight": overview.model_dump(),
                "insight_verification": {
                    "verified": True,
                    "confidence": 1.0,
                    "supported_findings": [],
                    "unsupported_findings": [],
                    "warnings": overview.limitations,
                },
                "dashboard": dashboard,
                "report": report,
            }

        # ==================================================
        # 3. CREATE INVESTIGATION PLAN
        # ==================================================

        plan = self.planner.create_plan(
            normalized_question,
            conversation_history=conversation_history,
            authoritative_question=question,
        )
        logger.info(
            "INVESTIGATION PLAN id=%s steps=%s",
            investigation_id,
            len(plan.sub_questions),
        )

        logger.debug(
            "INVESTIGATION PLAN steps=%s",
            [
                {
                    "id": step.id,
                    "question": step.question,
                    "objective": step.objective,
                    "required_information": step.required_information,
                    "depends_on": step.depends_on,
                }
                for step in plan.sub_questions
            ],
        )

        evidence_store = EvidenceStore()

        step_results = []
        empty_result_steps: set[int] = set()

        # ==================================================
        # 3. EXECUTE SUB-QUESTIONS
        # ==================================================

        for sub_question in plan.sub_questions:
            logger.info(
                "STEP START id=%s step=%s dependencies=%s",
                investigation_id,
                sub_question.id,
                sub_question.depends_on,
            )

            original_question = (
                sub_question.question
            )

            # IMPORTANT:
            #
            # This never changes during retries.
            #
            # Verification feedback is guidance for SQL
            # generation. It must never rewrite the user's
            # actual question because doing so can accidentally
            # introduce ranking/comparison intent.
            current_question = (
                original_question
            )

            feedback = None
            query_intent = None

            # Keep every generated SQL attempt so the generator
            # can explicitly avoid repeating failed SQL.
            previous_sqls: list[str] = []

            step_result = {
                "id": sub_question.id,
                "question": original_question,
                "objective": (
                    sub_question.objective
                ),
                "required_information": list(
                    sub_question.required_information
                ),
                "depends_on": list(
                    sub_question.depends_on
                ),
                "optional": sub_question.optional,
                "status": "failed",
                "attempts": [],
            }

            successful_evidence = None

            # ==================================================
            # DEPENDENCY CHECK
            # ==================================================

            empty_dependencies = [
                dependency_id
                for dependency_id in sub_question.depends_on
                if dependency_id in empty_result_steps
            ]

            if empty_dependencies:
                step_result["status"] = "skipped"
                step_result["skip_reason"] = "empty_dependency"
                step_result["reason"] = (
                    "Skipped because prerequisite step(s) "
                    f"{empty_dependencies} returned no matching records. "
                    "Running this dependent query without those records "
                    "could broaden the requested scope."
                )
                empty_result_steps.add(sub_question.id)
                step_results.append(step_result)
                logger.info(
                    "STEP SKIPPED_EMPTY_DEPENDENCY id=%s step=%s "
                    "dependencies=%s",
                    investigation_id,
                    sub_question.id,
                    empty_dependencies,
                )
                continue

            missing_dependencies = [
                dependency_id
                for dependency_id in sub_question.depends_on
                if evidence_store.get(dependency_id) is None
            ]

            if missing_dependencies:

                step_result["status"] = "blocked"

                step_result["reason"] = (
                    "Prerequisite investigation steps were not "
                    f"verified: {missing_dependencies}"
                )

                step_results.append(
                    step_result
                )

                continue

            dependency_evidence = [
                evidence_store.get(
                    dependency_id
                )
                for dependency_id in sub_question.depends_on
            ]

            dependency_evidence = [
                item
                for item in dependency_evidence
                if item is not None
            ]

            empty_evidence = [
                item
                for item in dependency_evidence
                if not item.rows and item.total_rows == 0
            ]

            if empty_evidence:
                empty_dependency_ids = [
                    item.step_id for item in empty_evidence
                ]
                step_result["status"] = "skipped"
                step_result["skip_reason"] = "empty_dependency"
                step_result["reason"] = (
                    "Skipped because verified prerequisite step(s) "
                    f"{empty_dependency_ids} returned no matching records. "
                    "Running this dependent query without those records "
                    "could broaden the requested scope."
                )
                empty_result_steps.add(sub_question.id)
                step_results.append(step_result)
                logger.info(
                    "STEP SKIPPED_EMPTY_DEPENDENCY id=%s step=%s "
                    "dependencies=%s",
                    investigation_id,
                    sub_question.id,
                    empty_dependency_ids,
                )
                continue

            dependency_context = [
                {
                    "step_id": item.step_id,
                    "question": item.question,
                    "objective": item.objective,
                    "rows": item.rows[:100],
                }
                for item in dependency_evidence
            ]

            # ==================================================
            # BOUNDED RETRIES
            # ==================================================

            for attempt in range(
                1,
                self.MAX_QUERY_ATTEMPTS + 1,
            ):

                logger.debug(
                    "STEP %s ATTEMPT %s/%s question=%s",
                    sub_question.id,
                    attempt,
                    self.MAX_QUERY_ATTEMPTS,
                    original_question,
                )

                adapter = self.database_manager.get(
                    connection_id,
                    organization_id=organization_id,
                )

                schema_context = get_relevant_schema(
                    adapter=adapter,
                    required_information=(
                        sub_question.required_information
                    ),
                    data_dictionary=data_dictionary,
                )
                logger.info(
                    "STEP SCHEMA id=%s step=%s attempt=%s tables=%s dictionary=%s",
                    investigation_id,
                    sub_question.id,
                    attempt,
                    len(schema_context),
                    bool(data_dictionary),
                )

                if not schema_context:
                    logger.warning(
                        "STEP NO_RELEVANT_SCHEMA id=%s step=%s attempt=%s",
                        investigation_id,
                        sub_question.id,
                        attempt,
                    )

                    attempt_result = {
                        "attempt": attempt,
                        "question": current_question,
                        "sql": None,
                        "allowed": False,
                        "executed": False,
                        "verified": False,
                        "reason": (
                            "No relevant database schema "
                            "was found."
                        ),
                    }

                    step_result[
                        "attempts"
                    ].append(
                        attempt_result
                    )

                    feedback = (
                        "No relevant schema was found "
                        "for the requested information."
                    )

                    continue

                # ==================================================
                # GENERATE → VALIDATE → EXECUTE → VERIFY
                # ==================================================

                sql_question = current_question
                if self._asks_for_all_extreme_ties(
                    question,
                    sub_question.objective,
                ):
                    sql_question += (
                        "\nReturn all entities tied at the requested "
                        "maximum or minimum metric, not an arbitrary "
                        "single row. Return one row per requested entity, "
                        "aggregating relationship rows per entity when "
                        "needed. Order first by the requested metric "
                        "(DESC for a maximum, ASC for a minimum), then by "
                        "the entity identifier as a deterministic "
                        "tie-breaker. Keep the query bounded with LIMIT "
                        "100; the database will establish the exact tie "
                        "count."
                    )
                scalar_count_request = (
                    "single aggregate result"
                    in sub_question.objective.lower()
                )
                if scalar_count_request:
                    sql_question += (
                        "\nReturn one scalar COUNT or SUM result for the "
                        "requested filters, not raw records or grouped rows."
                    )

                relative_duration = (
                    QuestionNormalizer.relative_comparison_year_count(
                        original_question
                    )
                )
                if (
                    relative_duration
                    and QuestionNormalizer.relative_calendar_years(
                        original_question
                    ) is None
                ):
                    sql_question += (
                        "\nRequired relative-period contract: "
                        f"Return one raw metric row for each of the "
                        f"{relative_duration} latest distinct calendar "
                        "years that actually exist in the stored data. "
                        "Do not translate this into a rolling date window, "
                        "use the current date, or invent a year cutoff. "
                        "Group or derive the year from the live schema's "
                        "actual year/date field, sort years newest first, "
                        f"and limit the grouped year rows to "
                        f"{relative_duration}. Use only field paths shown "
                        "in the supplied live schema."
                    )

                pipeline_result = (
                    self.sql_pipeline
                    .generate_validate_and_execute(
                        connection_id=connection_id,
                        organization_id=organization_id,
                        question=sql_question,
                        required_information=(
                            sub_question.required_information
                        ),
                        schema_context=schema_context,
                        feedback=feedback,
                        evidence_context=dependency_context,
                        previous_sqls=previous_sqls,
                        attempt_number=attempt,
                        query_intent=query_intent,
                    )
                )
                if pipeline_result.get("query_intent") is not None:
                    query_intent = pipeline_result["query_intent"]
                sql = pipeline_result.get("sql")
                sql_hash = (
                    hashlib.sha256(str(sql).encode("utf-8")).hexdigest()[:16]
                    if sql
                    else "none"
                )

                generated_sql = pipeline_result.get(
                    "sql"
                )

                if generated_sql:

                    normalized_sql = " ".join(
                        str(
                            generated_sql
                        ).lower().split()
                    )

                    previous_normalized = {
                        " ".join(
                            item.lower().split()
                        )
                        for item in previous_sqls
                    }

                    if (
                        normalized_sql
                        not in previous_normalized
                    ):

                        previous_sqls.append(
                            str(
                                generated_sql
                            )
                        )

                attempt_result = {
                    "attempt": attempt,
                    "question": current_question,
                    "sql": pipeline_result.get(
                        "sql"
                    ),
                    "allowed": pipeline_result.get(
                        "allowed",
                        False,
                    ),
                    "executed": pipeline_result.get(
                        "executed",
                        False,
                    ),
                    "verified": pipeline_result.get(
                        "verified",
                        False,
                    ),
                    "row_count": pipeline_result.get(
                        "row_count",
                        0,
                    ),
                    "reason": pipeline_result.get(
                        "reason"
                    ),
                    "model_route": pipeline_result.get(
                        "model_route",
                        {},
                    ),
                    "pagination": pipeline_result.get(
                        "pagination",
                        {},
                    ),
                }

                step_result[
                    "attempts"
                ].append(
                    attempt_result
                )
                step_result["reason"] = pipeline_result.get("reason")
                logger.info(
                    "STEP ATTEMPT id=%s step=%s attempt=%s sql_sha256=%s "
                    "allowed=%s executed=%s verified=%s rows=%s reason=%s "
                    "sql=%s",
                    investigation_id,
                    sub_question.id,
                    attempt,
                    sql_hash,
                    pipeline_result.get("allowed", False),
                    pipeline_result.get("executed", False),
                    pipeline_result.get("verified", False),
                    pipeline_result.get("row_count", 0),
                    pipeline_result.get("reason"),
                    " ".join(str(sql or "").split()),
                )

                # ==================================================
                # SECURITY / SCHEMA FAILURE
                # ==================================================

                if not pipeline_result.get(
                    "allowed",
                    False,
                ):

                    logger.debug(
                        "STEP %s ATTEMPT %s SECURITY/SCHEMA FAILURE: %s",
                        sub_question.id,
                        attempt,
                        pipeline_result.get(
                            "reason"
                        ),
                    )
                    feedback = QueryRepairAnalyzer.create_feedback(
                        question=(
                            f"{original_question} "
                            f"(current step: {current_question})"
                        ),
                        query_language=pipeline_result.get(
                            "query_language",
                            "SQL",
                        ),
                        failure_stage="security/schema validation",
                        reason=str(pipeline_result.get("reason") or ""),
                        previous_feedback=feedback,
                    )

                    continue

                # ==================================================
                # EXECUTION FAILURE
                # ==================================================

                if not pipeline_result.get(
                    "executed",
                    False,
                ):

                    logger.debug(
                        "STEP %s ATTEMPT %s DATABASE FAILURE: %s",
                        sub_question.id,
                        attempt,
                        pipeline_result.get(
                            "reason"
                        ),
                    )
                    feedback = QueryRepairAnalyzer.create_feedback(
                        question=(
                            f"{original_question} "
                            f"(current step: {current_question})"
                        ),
                        query_language=pipeline_result.get(
                            "query_language",
                            "SQL",
                        ),
                        failure_stage="query execution",
                        reason=str(pipeline_result.get("reason") or ""),
                        previous_feedback=feedback,
                    )

                    continue

                # ==================================================
                # DETERMINISTIC RESULT VERIFICATION FAILURE
                # ==================================================

                if not pipeline_result.get(
                    "verified",
                    False,
                ):

                    verification = (
                        pipeline_result.get(
                            "verification"
                        )
                        or {}
                    )
                    semantic = (
                        pipeline_result.get(
                            "semantic_verification"
                        )
                        or {}
                    )

                    logger.debug(
                        "STEP %s ATTEMPT %s RESULT VERIFICATION FAILURE "
                        "result_verified=%s result_warnings=%s "
                        "semantic_verified=%s semantic_errors=%s "
                        "semantic_warnings=%s",
                        sub_question.id,
                        attempt,
                        verification.get(
                            "verified",
                            False,
                        ),
                        verification.get(
                            "warnings",
                            [],
                        ),
                        semantic.get(
                            "verified",
                            False,
                        ),
                        semantic.get(
                            "errors",
                            [],
                        ),
                        semantic.get(
                            "warnings",
                            [],
                        ),
                    )

                    feedback = QueryRepairAnalyzer.create_feedback(
                        question=(
                            f"{original_question} "
                            f"(current step: {current_question})"
                        ),
                        query_language=pipeline_result.get(
                            "query_language",
                            "SQL",
                        ),
                        failure_stage="result verification",
                        reason=str(pipeline_result.get("reason") or ""),
                        diagnostics={
                            "result verification": verification,
                            "semantic verification": semantic,
                        },
                        previous_feedback=feedback,
                    )

                    continue

                # ==================================================
                # LOGICAL VERIFICATION
                # ==================================================

                logical_result = (
                    self.logical_verifier.verify(
                        question=original_question,
                        objective=(
                            sub_question.objective
                        ),
                        sql=pipeline_result.get(
                            "source_sql",
                            pipeline_result["sql"],
                        ),
                        schema_context=schema_context,
                        rows=pipeline_result["rows"],
                        # This call checks whether the initial bounded rows
                        # answer the question. Full-list completeness is
                        # established separately below, before evidence is
                        # accepted.
                        pagination_complete=True,
                        attempt_number=attempt,
                        query_language=pipeline_result.get(
                            "query_language",
                            connection.database_type.lower(),
                        ),
                        query_intent=pipeline_result.get("query_intent"),
                    )
                )

                logger.debug(
                    "LOGICAL VERIFICATION step=%s relevant=%s "
                    "confidence=%.2f reason=%s",
                    sub_question.id,
                    logical_result.relevant,
                    logical_result.confidence,
                    logical_result.reason,
                )
                logger.info(
                    "STEP LOGICAL_VERIFICATION id=%s step=%s attempt=%s "
                    "relevant=%s confidence=%.3f reason=%s",
                    investigation_id,
                    sub_question.id,
                    attempt,
                    logical_result.relevant,
                    logical_result.confidence,
                    logical_result.reason,
                )

                attempt_result[
                    "logical_model_route"
                ] = dict(
                    getattr(
                        self.model_router,
                        "last_route_metadata",
                        {},
                    )
                    or {}
                )

                attempt_result[
                    "logical_verification"
                ] = {
                    "relevant": (
                        logical_result.relevant
                    ),
                    "confidence": (
                        logical_result.confidence
                    ),
                    "reason": (
                        logical_result.reason
                    ),
                    "missing_information": (
                        logical_result
                        .missing_information
                    ),
                    "suggested_revision": (
                        logical_result
                        .suggested_revision
                    ),
                }

                # ==================================================
                # VERIFIED EVIDENCE
                # ==================================================

                if logical_result.relevant:
                    collection_scan = pipeline_result.get(
                        "collection_scan"
                    )
                    if (
                        collection_scan
                        and collection_scan.get("complete")
                    ):
                        pipeline_result["pagination"] = {
                            "complete": True,
                            "pages": 1,
                            "total_rows": collection_scan["total_rows"],
                            "reason": (
                                "The database counted the complete "
                                "collection in one statement and returned "
                                "only the bounded preview."
                            ),
                            "mode": collection_scan["mode"],
                        }
                        pipeline_result["row_count"] = (
                            collection_scan["total_rows"]
                        )
                        attempt_result["pagination"] = (
                            pipeline_result["pagination"]
                        )
                        logger.info(
                            "STEP PAGINATION id=%s step=%s attempt=%s "
                            "complete=true pages=1 total_rows=%s "
                            "mode=single_query",
                            investigation_id,
                            sub_question.id,
                            attempt,
                            collection_scan["total_rows"],
                        )
                    elif pipeline_result.get("rows"):
                        pagination = self.sql_pipeline.collect_additional_pages(
                            connection_id=connection_id,
                            organization_id=organization_id,
                            sql=pipeline_result["sql"],
                            schema_context=schema_context,
                            initial_rows=pipeline_result["rows"],
                            expected_top_n=(
                                _expected_top_n(
                                    question
                                    + " "
                                    + sub_question.objective
                                )
                                if sub_question.id
                                == plan.sub_questions[-1].id
                                else None
                            ),
                        )
                        pipeline_result["rows"] = pagination["rows"]
                        pipeline_result["row_count"] = pagination.get(
                            "total_rows",
                            len(pagination["rows"]),
                        )
                        pipeline_result["pagination"] = {
                            "complete": pagination["complete"],
                            "pages": pagination["pages"],
                            "total_rows": pipeline_result["row_count"],
                            "reason": pagination["reason"],
                        }
                        attempt_result["pagination"] = (
                            pipeline_result["pagination"]
                        )
                        logger.info(
                            "STEP PAGINATION id=%s step=%s attempt=%s "
                            "complete=%s pages=%s total_rows=%s reason=%s",
                            investigation_id,
                            sub_question.id,
                            attempt,
                            pagination["complete"],
                            pagination["pages"],
                            pipeline_result["row_count"],
                            pagination["reason"],
                        )

                        if not pagination["complete"]:
                            pipeline_result["verified"] = False
                            pipeline_result["reason"] = (
                                "The query returned a bounded collection "
                                "but pagination could not establish "
                                f"completeness: {pagination['reason']}"
                            )
                            step_result["status"] = "failed"
                            step_result["reason"] = pipeline_result["reason"]
                            if scalar_count_request:
                                feedback = QueryRepairAnalyzer.create_feedback(
                                    question=current_question,
                                    query_language=pipeline_result.get(
                                        "query_language",
                                        "SQL",
                                    ),
                                    failure_stage="pagination",
                                    reason=pagination["reason"],
                                    diagnostics={
                                        "required result shape": (
                                            "one ungrouped COUNT or SUM "
                                            "aggregate preserving all "
                                            "requested filters"
                                        )
                                    },
                                    previous_feedback=feedback,
                                )
                            else:
                                feedback = QueryRepairAnalyzer.create_feedback(
                                    question=current_question,
                                    query_language=pipeline_result.get(
                                        "query_language",
                                        "SQL",
                                    ),
                                    failure_stage="pagination",
                                    reason=pagination["reason"],
                                    previous_feedback=feedback,
                                )
                            continue

                    successful_evidence = Evidence(
                        step_id=sub_question.id,
                        question=original_question,
                        objective=(
                            sub_question.objective
                        ),
                        sql=pipeline_result["sql"],
                        rows=pipeline_result["rows"],
                        schema_context=schema_context,
                        logical_reason=(
                            logical_result.reason
                        ),
                        confidence=(
                            logical_result.confidence
                        ),
                        depends_on=list(
                            sub_question.depends_on
                        ),
                        pagination_complete=bool(
                            (
                                pipeline_result.get("pagination")
                                or {}
                            ).get("complete", False)
                        ),
                        total_rows=(
                            pipeline_result.get("pagination") or {}
                        ).get(
                            "total_rows",
                            len(pipeline_result["rows"]),
                        ),
                        all_extreme_ties=(
                            (pipeline_result.get("collection_scan") or {})
                            .get("mode")
                            == "single_query_extreme_ties"
                        ),
                        ranking_metric=(
                            (pipeline_result.get("collection_scan") or {})
                            .get("ranking_metric")
                        ),
                        requested_periods=list(
                            logical_result.requested_periods
                        ),
                        observed_periods=list(
                            logical_result.observed_periods
                        ),
                        unavailable_periods=list(
                            logical_result.unavailable_periods
                        ),
                    )

                    evidence_store.add(
                        successful_evidence
                    )

                    step_result[
                        "status"
                    ] = "verified"

                    step_result[
                        "final_question"
                    ] = current_question

                    step_result[
                        "verification_question"
                    ] = original_question

                    step_result[
                        "evidence_confidence"
                    ] = (
                        logical_result.confidence
                    )

                    logger.debug(
                        "STEP %s VERIFIED on attempt %s",
                        sub_question.id,
                        attempt,
                    )
                    logger.info(
                        "STEP VERIFIED id=%s step=%s attempt=%s confidence=%.3f",
                        investigation_id,
                        sub_question.id,
                        attempt,
                        logical_result.confidence,
                    )

                    break

                # ==================================================
                # LOGICAL FAILURE → RETRY
                # ==================================================

                step_result["logical_failure_reason"] = logical_result.reason
                step_result["reason"] = logical_result.reason
                attempt_result["reason"] = logical_result.reason

                logger.debug(
                    "STEP %s ATTEMPT %s LOGICAL FAILURE: %s",
                    sub_question.id,
                    attempt,
                    logical_result.reason,
                )

                feedback = QueryRepairAnalyzer.create_feedback(
                    question=(
                        f"{original_question} "
                        f"(current step: {current_question})"
                    ),
                    query_language=pipeline_result.get(
                        "query_language",
                        "SQL",
                    ),
                    failure_stage="logical verification",
                    reason=logical_result.reason,
                    diagnostics={
                        "missing information": (
                            logical_result.missing_information
                        ),
                        "required revision": (
                            logical_result.suggested_revision
                        ),
                    },
                    previous_feedback=feedback,
                )

            # ==================================================
            # FINAL STEP STATUS
            # ==================================================

            if successful_evidence is None:

                step_result[
                    "status"
                ] = "failed"
                step_result["reason"] = (
                    step_result.get("logical_failure_reason")
                    or step_result.get("reason")
                )

                step_result[
                    "final_question"
                ] = current_question

                step_result[
                    "verification_question"
                ] = original_question

                logger.debug(
                    "STEP %s FAILED after %s attempts",
                    sub_question.id,
                    self.MAX_QUERY_ATTEMPTS,
                )
                logger.warning(
                    "STEP FAILED id=%s step=%s attempts=%s",
                    investigation_id,
                    sub_question.id,
                    len(step_result["attempts"]),
                )

            step_results.append(
                step_result
            )

        # ==================================================
        # 4. COLLECT VERIFIED EVIDENCE
        # ==================================================

        evidence = evidence_store.all()

        failed_steps = self._failed_required_steps(step_results)

        if not evidence:
            failed_reasons = [
                str(
                    step.get("reason")
                    or (step.get("attempts") or [{}])[-1].get("reason")
                    or "No verified evidence was produced."
                )
                for step in failed_steps
            ]
            logger.warning(
                "INVESTIGATION FAILED id=%s reason=no_verified_evidence",
                investigation_id,
            )

            return {
                "status": "failed",
                "question": question,
                "normalized_question": normalized_question,
                "plan": plan.model_dump(),
                "steps": step_results,
                "evidence": [],
                "insight": None,
                "insight_verification": None,
                "reason": (
                    "No verified evidence could be collected for the "
                    "investigation."
                    + (
                        " Last failure: " + " | ".join(failed_reasons)
                        if failed_reasons
                        else ""
                    )
                ),
            }

        if failed_steps:
            logger.warning(
                "INVESTIGATION FAILED id=%s reason=unverified_steps "
                "failed_steps=%s evidence=%s",
                investigation_id,
                [step.get("id") for step in failed_steps],
                len(evidence),
            )

            return {
                "status": "failed",
                "question": question,
                "normalized_question": normalized_question,
                "plan": plan.model_dump(),
                "steps": step_results,
                "evidence": [
                    {
                        "step_id": item.step_id,
                        "question": item.question,
                        "objective": item.objective,
                        "sql": item.sql,
                        "rows": item.rows,
                        "confidence": item.confidence,
                        "logical_reason": item.logical_reason,
                        "depends_on": item.depends_on,
                        "pagination_complete": item.pagination_complete,
                        "total_rows": item.total_rows,
                        "all_extreme_ties": item.all_extreme_ties,
                        "ranking_metric": item.ranking_metric,
                        "requested_periods": item.requested_periods,
                        "observed_periods": item.observed_periods,
                        "unavailable_periods": item.unavailable_periods,
                    }
                    for item in evidence
                ],
                "insight": None,
                "insight_verification": None,
                "reason": (
                    "The investigation did not verify every "
                    "planned analytical step; final insight "
                    "generation was blocked."
                ),
            }

        # ==================================================
        # 5. GENERATE INSIGHT
        # ==================================================

        insight = self.insight_generator.generate(
            question=question,
            evidence=evidence,
            interpretation_question=normalized_question,
        )
        optional_failures = [
            step
            for step in step_results
            if (
                step.get("optional", False)
                and step.get("status") != "verified"
            )
        ]
        for step in optional_failures:
            attempts = step.get("attempts") or []
            reason = (
                step.get("reason")
                or (
                    attempts[-1].get("reason")
                    if attempts and isinstance(attempts[-1], dict)
                    else None
                )
                or "the database could not verify this supplemental analysis"
            )
            insight.uncertainties.append(
                f"Supplemental analysis not verified "
                f"({step.get('question', 'additional analysis')}): {reason}"
            )
        logger.info(
            "INSIGHT GENERATED id=%s evidence=%s",
            investigation_id,
            len(evidence),
        )

        # ==================================================
        # 6. VERIFY INSIGHT
        # ==================================================

        insight_verification = (
            self.insight_verifier.verify(
                question=question,
                insight=insight,
                evidence=evidence,
                interpretation_question=normalized_question,
            )
        )

        logger.debug(
            "INSIGHT VERIFICATION verified=%s "
            "confidence=%.2f supported=%s "
            "unsupported=%s warnings=%s",
            insight_verification.verified,
            insight_verification.confidence,
            insight_verification.supported_findings,
            insight_verification.unsupported_findings,
            insight_verification.warnings,
        )

        # ==================================================
        # 7. FINAL STATUS
        # ==================================================

        if insight_verification.verified:

            status = "verified"

        else:

            status = (
                "insight_requires_review"
            )
        logger.info(
            "INVESTIGATION COMPLETE id=%s status=%s steps=%s evidence=%s "
            "insight_verified=%s confidence=%.3f",
            investigation_id,
            status,
            len(step_results),
            len(evidence),
            insight_verification.verified,
            insight_verification.confidence,
        )

        evidence_payload = [
            {
                "step_id": item.step_id,
                "question": item.question,
                "objective": item.objective,
                "sql": item.sql,
                "rows": item.rows,
                "confidence": item.confidence,
                "logical_reason": item.logical_reason,
                "depends_on": item.depends_on,
                "pagination_complete": item.pagination_complete,
                "total_rows": item.total_rows,
                "all_extreme_ties": item.all_extreme_ties,
                "ranking_metric": item.ranking_metric,
                "requested_periods": item.requested_periods,
                "observed_periods": item.observed_periods,
                "unavailable_periods": item.unavailable_periods,
            }
            for item in evidence
        ]
        dashboard, report = build_artifacts(
            question=question,
            insight=insight.model_dump(),
            evidence=evidence_payload,
        )

        return {
            "status": status,
            "question": question,
            "normalized_question": normalized_question,
            "plan": plan.model_dump(),
            "steps": step_results,
            "evidence": evidence_payload,
            "insight": insight.model_dump(),
            "insight_verification": (
                insight_verification.model_dump()
            ),
            "dashboard": dashboard.model_dump(),
            "report": report.model_dump(),
        }


def main():
    from app.main import run

    run()


if __name__ == "__main__":
    main()
