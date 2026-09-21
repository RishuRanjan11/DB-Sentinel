from app.agent.planner import QuestionPlanner
from app.agent.sql_pipeline import SQLPipeline
from app.agent.logical_verifier import LogicalVerifier
from app.agent.evidence import Evidence, EvidenceStore
from app.agent.insight_generator import InsightGenerator
from app.agent.insight_verifier import InsightVerifier
from app.database.manager import DatabaseManager
from app.database.schema_retriever import get_relevant_schema
from app.llm.model_router import create_default_model_router
from app.security.policies import RequestPolicy
from app.security.permissions import DatabasePermissions
from app.core.trace import logger


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

        self.logical_verifier = LogicalVerifier(
            model_router=self.model_router
        )

        self.insight_generator = InsightGenerator(
            model_router=self.model_router
        )

        self.insight_verifier = InsightVerifier(
            model_router=self.model_router
        )

        self.sql_pipeline = SQLPipeline(
            database_manager=database_manager,
            model_router=self.model_router,
        )

    def investigate(
        self,
        connection_id: str,
        question: str,
    ) -> dict:

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
                "REQUEST BLOCKED operation=%s reason=%s",
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

        # ==================================================
        # 2. CREATE INVESTIGATION PLAN
        # ==================================================

        plan = self.planner.create_plan(
            question
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

        # ==================================================
        # 3. EXECUTE SUB-QUESTIONS
        # ==================================================

        for sub_question in plan.sub_questions:

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
                "status": "failed",
                "attempts": [],
            }

            successful_evidence = None

            # ==================================================
            # DEPENDENCY CHECK
            # ==================================================

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
                    connection_id
                )

                schema_context = get_relevant_schema(
                    adapter=adapter,
                    required_information=(
                        sub_question.required_information
                    ),
                )

                if not schema_context:

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

                pipeline_result = (
                    self.sql_pipeline
                    .generate_validate_and_execute(
                        connection_id=connection_id,
                        question=current_question,
                        required_information=(
                            sub_question.required_information
                        ),
                        schema_context=schema_context,
                        feedback=feedback,
                        evidence_context=dependency_context,
                        previous_sqls=previous_sqls,
                        attempt_number=attempt,
                    )
                )

                # A bounded SELECT is a security boundary, not a semantic
                # assertion that the user only wants the first page. For
                # collection-style questions, complete the result through
                # deterministic pagination when the generated query has a
                # stable ORDER BY and the first page is full.
                pagination_failure = False
                pagination_reason = None

                if (
                    pipeline_result.get("executed")
                    and pipeline_result.get("verified")
                    and pipeline_result.get("rows")
                ):
                    pagination = self.sql_pipeline.collect_additional_pages(
                        connection_id=connection_id,
                        sql=pipeline_result["sql"],
                        schema_context=schema_context,
                        initial_rows=pipeline_result["rows"],
                    )

                    pipeline_result["rows"] = pagination["rows"]
                    pipeline_result["row_count"] = len(
                        pagination["rows"]
                    )
                    pipeline_result["pagination"] = {
                        "complete": pagination["complete"],
                        "pages": pagination["pages"],
                        "reason": pagination["reason"],
                    }

                    if not pagination["complete"]:
                        pagination_failure = True
                        pagination_reason = pagination["reason"]
                        pipeline_result["verified"] = False
                        pipeline_result["reason"] = (
                            "The query returned a bounded collection but "
                            "pagination could not establish completeness: "
                            f"{pagination_reason}"
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

                    feedback = (
                        "The generated SQL was rejected by the "
                        "deterministic security/schema validator.\n\n"
                        f"Exact reason: "
                        f"{pipeline_result.get('reason')}\n\n"
                        "Repair only the failing construct. "
                        "Use the exact supplied schema columns "
                        "and choose a materially different strategy "
                        "if the previous SQL is listed as a failed "
                        "attempt."
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

                    feedback = (
                        "The generated SQL reached PostgreSQL "
                        "but failed to execute.\n\n"
                        f"Exact database error: "
                        f"{pipeline_result.get('reason')}\n\n"
                        "Correct the failing SQL construct while "
                        "preserving the original question. "
                        "Do not invent columns; use only the "
                        "supplied schema."
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

                    # Pagination is a separate completeness gate from
                    # structural/semantic result verification. Do not
                    # misreport it as a verifier failure with empty
                    # error/warning lists. This distinction is important
                    # for collection questions such as "which ...".
                    if pagination_failure:
                        logger.debug(
                            "STEP %s ATTEMPT %s COLLECTION COMPLETENESS "
                            "FAILURE: %s",
                            sub_question.id,
                            attempt,
                            pagination_reason,
                        )

                        feedback = (
                            "The query executed and passed deterministic "
                            "result verification, but the requested "
                            "collection is not yet proven complete.\n\n"
                            f"Pagination result: {pagination_reason}\n\n"
                            "If the query uses LIMIT for a collection, "
                            "use a deterministic ORDER BY on a real column "
                            "from the supplied schema so the bounded result "
                            "can be paginated safely. Do not invent a "
                            "column and do not change the original question. "
                            "If the query is already ordered, preserve that "
                            "ordering and make the pagination strategy "
                            "deterministic."
                        )

                        continue

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

                    feedback = (
                        "The query executed but deterministic "
                        "result verification failed.\n\n"
                        f"Result verification: "
                        f"{verification}\n"
                        f"Semantic verification: "
                        f"{semantic}\n\n"
                        "Change the SQL only enough to satisfy "
                        "the failed verification while preserving "
                        "the original investigation intent."
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
                        sql=pipeline_result["sql"],
                        schema_context=schema_context,
                        rows=pipeline_result["rows"],
                        pagination_complete=bool(
                            (
                                pipeline_result.get(
                                    "pagination"
                                )
                                or {}
                            ).get(
                                "complete",
                                False,
                            )
                        ),
                        attempt_number=attempt,
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

                    break

                # ==================================================
                # LOGICAL FAILURE → RETRY
                # ==================================================

                logger.debug(
                    "STEP %s ATTEMPT %s LOGICAL FAILURE: %s",
                    sub_question.id,
                    attempt,
                    logical_result.reason,
                )

                logical_feedback = (
                    "The query executed successfully, but the "
                    "returned data is not sufficient to answer "
                    "this investigation step.\n\n"
                    f"Logical verification reason:\n"
                    f"{logical_result.reason}\n\n"
                    f"Missing information:\n"
                    f"{logical_result.missing_information}\n"
                )

                if logical_result.suggested_revision:

                    logical_feedback += (
                        "Required revision suggested by the "
                        "logical verifier:\n"
                        f"{logical_result.suggested_revision}\n"
                    )

                # Preserve constraints learned from earlier attempts.
                # A later logical failure must not erase an earlier
                # security/schema requirement such as LIMIT.
                prior_feedback = feedback

                feedback = (
                    (
                        "Previous attempt constraints/failures that "
                        "must still be satisfied:\n"
                        f"{prior_feedback}\n\n"
                    )
                    if prior_feedback
                    else ""
                ) + logical_feedback + (
                    "\nThe next SQL must use a materially different "
                    "strategy from the rejected SQL when the verifier "
                    "identified a semantic scope problem. Do not repeat "
                    "the rejected table, join, filter, or aggregation "
                    "strategy merely with cosmetic changes.\n\n"
                    "Keep the exact original step question unchanged. "
                    "Do not introduce ranking, comparison, or another "
                    "analytical requirement that is not present in it. "
                    "All previous deterministic security/schema "
                    "requirements remain mandatory."
                )

            # ==================================================
            # FINAL STEP STATUS
            # ==================================================

            if successful_evidence is None:

                step_result[
                    "status"
                ] = "failed"

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

            step_results.append(
                step_result
            )

        # ==================================================
        # 4. COLLECT VERIFIED EVIDENCE
        # ==================================================

        evidence = evidence_store.all()

        failed_steps = [
            step
            for step in step_results
            if step.get("status") != "verified"
        ]

        if not evidence:

            return {
                "status": "failed",
                "question": question,
                "plan": plan.model_dump(),
                "steps": step_results,
                "evidence": [],
                "insight": None,
                "insight_verification": None,
                "reason": (
                    "No verified evidence could be "
                    "collected for the investigation."
                ),
            }

        if failed_steps:

            return {
                "status": "failed",
                "question": question,
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
        )

        # ==================================================
        # 6. VERIFY INSIGHT
        # ==================================================

        insight_verification = (
            self.insight_verifier.verify(
                question=question,
                insight=insight,
                evidence=evidence,
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

        return {
            "status": status,
            "question": question,
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
                    "logical_reason": (
                        item.logical_reason
                    ),
                    "depends_on": item.depends_on,
                }
                for item in evidence
            ],
            "insight": insight.model_dump(),
            "insight_verification": (
                insight_verification.model_dump()
            ),
        }


def main():
    import sys

    from app.database.connection import DatabaseConnection

    connection = DatabaseConnection(
        connection_id="flybase",
        organization_id="local",
        workspace_id="default",
        name="FlyBase",
        database_type="postgresql",
        host="chado.flybase.org",
        port=5432,
        database_name="flybase",
        username="flybase",
        password=None,
        ssl_enabled=False,
    )

    database_manager = DatabaseManager()

    database_manager.register(
        connection
    )

    engine = InvestigationEngine(
        database_manager=database_manager
    )

    question = " ".join(
        sys.argv[1:]
    ).strip() or (
        "How many FlyBase genes have human orthologs?"
    )

    print()
    print("DB-SENTINEL")
    print("───────────")
    print("Investigation engine started")
    print(
        f"Question: {question}"
    )

    try:

        result = engine.investigate(
            connection_id="flybase",
            question=question,
        )

        print()
        print("Investigation Plan")
        print("──────────────────")
        plan = result.get("plan") or {}

        for planned_step in plan.get("sub_questions", []):
            print(
                f"Step {planned_step.get('id')}: "
                f"{planned_step.get('question')}"
            )
            print(
                f"  Objective: {planned_step.get('objective')}"
            )
            print(
                f"  Required: {planned_step.get('required_information', [])}"
            )
            print(
                f"  Depends on: {planned_step.get('depends_on', [])}"
            )

        print()
        print("Result")
        print("──────")
        print(
            f"Status: {result['status']}"
        )

        print(
            "Evidence: "
            f"{len(result.get('evidence', []))} "
            "verified step(s)"
        )

        for step in result.get(
            "steps",
            [],
        ):

            attempts = step.get(
                "attempts",
                [],
            )

            attempt_word = (
                "attempt"
                if len(attempts) == 1
                else "attempts"
            )

            print(
                f"Step {step.get('id')}: "
                f"{step.get('status')} "
                f"({len(attempts)} {attempt_word})"
            )

            for attempt in attempts:
                route = attempt.get("model_route") or {}
                logical_route = attempt.get("logical_model_route") or {}
                if route:
                    print(
                        f"  SQL model: "
                        f"{route.get('provider')}/{route.get('model')} "
                        f"(attempt {route.get('attempt_number')})"
                    )
                if logical_route:
                    print(
                        f"  Logical verifier model: "
                        f"{logical_route.get('provider')}/{logical_route.get('model')} "
                        f"(attempt {logical_route.get('attempt_number')})"
                    )


        verification = (
            result.get(
                "insight_verification"
            )
        )

        if verification:

            state = (
                "passed"
                if verification.get(
                    "verified"
                )
                else "review required"
            )

            print(
                "Insight verification: "
                f"{state} "
                f"({verification.get('confidence', 0):.2f})"
            )

            warnings = (
                verification.get(
                    "warnings"
                )
                or []
            )

            if warnings:

                print(
                    f"Warnings: {len(warnings)}"
                )

        if result.get(
            "insight"
        ):

            insight = result[
                "insight"
            ]

            print()
            print("Insight")
            print("───────")

            print(
                insight.get(
                    "summary",
                    "",
                )
            )

            for finding in insight.get(
                "findings",
                [],
            ):

                print(
                    f"• {finding}"
                )

        print()
        print(
            "Detailed execution trace: "
            "logs/db-sentinel.log"
        )

    finally:

        database_manager.close_all()


if __name__ == "__main__":
    main()