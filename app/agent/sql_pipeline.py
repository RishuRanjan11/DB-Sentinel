from app.agent.result_verifier import ResultVerifier
import re
from app.agent.sql_generator import SQLGenerator
from app.agent.semantic_verifier import SemanticResultVerifier
from app.database.executor import SecureQueryExecutor
from app.database.manager import DatabaseManager
from app.database.schema_retriever import get_relevant_schema
from app.security.sql_validator import SQLSecurityValidator


class SQLPipeline:

    def __init__(
        self,
        database_manager: DatabaseManager,
        model_router=None,
    ):
        self.database_manager = database_manager
        self.model_router = model_router

        self.generator = SQLGenerator(
            model_router=model_router
        )

        self.result_verifier = ResultVerifier()
        self.semantic_verifier = SemanticResultVerifier()

    def generate_validate_and_execute(
        self,
        connection_id: str,
        question: str,
        required_information: list[str],
        complexity: str | None = None,
        schema_context: list[dict] | None = None,
        feedback: str | None = None,
        evidence_context: list[dict] | None = None,
        previous_sqls: list[str] | None = None,
        attempt_number: int = 1,
    ) -> dict:

        adapter = self.database_manager.get(
            connection_id
        )

        if schema_context is None:

            schema_context = get_relevant_schema(
                adapter=adapter,
                required_information=required_information,
            )

        if not schema_context:

            return self._failure(
                question=question,
                reason=(
                    "No relevant database schema was found "
                    "for the requested information."
                ),
                schema_context=[],
            )

        generated = self.generator.generate(
            question=question,
            schema_context=schema_context,
            complexity=complexity,
            feedback=feedback,
            evidence_context=evidence_context,
            previous_sqls=previous_sqls,
            attempt_number=attempt_number,
        )

        generation_route = dict(
            getattr(
                self.model_router,
                "last_route_metadata",
                {},
            )
            or {}
        )

        sql = generated.sql.strip()

        normalized_sql = (
            " ".join(
                sql.lower().split()
            )
        )

        previous_normalized = {
            " ".join(
                item.lower().split()
            )
            for item in (
                previous_sqls or []
            )
            if item
        }

        if normalized_sql in previous_normalized:

            return {
                "question": question,
                "sql": generated.sql,
                "tables_used": generated.tables_used,
                "columns_used": generated.columns_used,
                "allowed": False,
                "executed": False,
                "verified": False,
                "rows": [],
                "row_count": 0,
                "reason": (
                    "The generated SQL is identical to a previous "
                    "failed attempt. Generate a materially "
                    "different SQL strategy."
                ),
                "verification": None,
                "semantic_verification": None,
                "schema_context": schema_context,
                "model_route": generation_route,
                "pagination": {
                    "complete": False,
                    "pages": 0,
                },
            }

        allowed_tables = {
            (
                str(table["schema"]).lower(),
                str(table["table"]).lower(),
            )
            for table in schema_context
        }

        sensitive_tokens = (
            "password",
            "passwd",
            "secret",
            "token",
            "api_key",
            "private_key",
            "access_key",
            "credential",
            "ssn",
            "social_security",
            "credit_card",
            "card_number",
            "security_answer",
            "private_note",
        )

        blocked_columns = set()

        for table in schema_context:

            for column in table.get(
                "columns",
                [],
            ):

                if isinstance(
                    column,
                    dict,
                ):

                    column_name = str(
                        column.get(
                            "name",
                            column.get(
                                "column",
                                "",
                            ),
                        )
                    ).lower()

                else:

                    column_name = str(
                        column
                    ).lower()

                if any(
                    token in column_name
                    for token in sensitive_tokens
                ):

                    blocked_columns.add(
                        (
                            table["schema"],
                            table["table"],
                            column_name,
                        )
                    )

        validator = SQLSecurityValidator(
            allowed_tables=allowed_tables,
            blocked_columns=blocked_columns,
            schema_context=schema_context,
        )

        validation = validator.validate(
            sql
        )

        if not validation.allowed:

            return {
                "question": question,
                "sql": generated.sql,
                "tables_used": generated.tables_used,
                "columns_used": generated.columns_used,
                "allowed": False,
                "executed": False,
                "verified": False,
                "rows": [],
                "row_count": 0,
                "reason": validation.reason,
                "verification": None,
                "semantic_verification": None,
                "schema_context": schema_context,
                "model_route": generation_route,
                "pagination": {
                    "complete": False,
                    "pages": 0,
                },
            }

        executor = SecureQueryExecutor(
            adapter=adapter,
            validator=validator,
        )

        execution = executor.execute(
            sql=sql
        )

        if not execution.success:

            return {
                "question": question,
                "sql": generated.sql,
                "tables_used": generated.tables_used,
                "columns_used": generated.columns_used,
                "allowed": True,
                "executed": False,
                "verified": False,
                "rows": [],
                "row_count": 0,
                "reason": execution.error,
                "verification": None,
                "semantic_verification": None,
                "schema_context": schema_context,
                "model_route": generation_route,
                "pagination": {
                    "complete": False,
                    "pages": 0,
                },
            }

        verification = self.result_verifier.verify(
            adapter=adapter,
            sql=sql,
            rows=execution.rows,
        )

        semantic_verification = (
            self.semantic_verifier.verify(
                adapter=adapter,
                sql=sql,
                rows=execution.rows,
                schema_context=schema_context,
            )
        )

        # Structural/result verification is a hard gate.
        #
        # SemanticResultVerifier may emit warnings for bounded,
        # valid result pages (for example LIMIT-based list retrieval).
        # Warnings must not turn an otherwise executable/structurally
        # valid query into a failed attempt. Actual semantic errors
        # remain a hard failure.
        verified = (
            verification.verified
            and not semantic_verification.errors
        )

        verification_reasons = []

        if not verification.verified:

            verification_reasons.extend(
                verification.warnings
            )

        if not semantic_verification.verified:

            verification_reasons.extend(
                semantic_verification.errors
            )

            verification_reasons.extend(
                semantic_verification.warnings
            )

        return {
            "question": question,
            "sql": generated.sql,
            "tables_used": generated.tables_used,
            "columns_used": generated.columns_used,
            "allowed": True,
            "executed": True,
            "verified": verified,
            "rows": execution.rows,
            "row_count": execution.row_count,
            "reason": (
                "Query executed and all deterministic result "
                "checks passed."
                if verified
                else (
                    "Query executed but deterministic result "
                    "verification failed: "
                    + " | ".join(
                        verification_reasons
                    )
                )
            ),
            "verification": {
                "verified": verification.verified,
                "checks": verification.checks,
                "warnings": verification.warnings,
            },
            "semantic_verification": {
                "verified": (
                    semantic_verification.verified
                ),
                "checks": (
                    semantic_verification.checks
                ),
                "warnings": (
                    semantic_verification.warnings
                ),
                "errors": (
                    semantic_verification.errors
                ),
            },
            "schema_context": schema_context,
            "model_route": generation_route,
            "pagination": {
                "complete": False,
                "pages": 1,
            },
        }

    def collect_additional_pages(
        self,
        connection_id: str,
        sql: str,
        schema_context: list[dict],
        initial_rows: list[dict],
        page_size: int = 100,
        max_pages: int = 1000,
    ) -> dict:
        """
        Complete a bounded collection query without removing the
        deterministic SQL LIMIT safety boundary.

        This is intentionally generic. It only paginates an already
        validated SELECT that:
        - has an explicit numeric LIMIT,
        - is not an aggregate/grouped query,
        - has a stable ORDER BY,
        - returned a full page.

        The same validated SQL strategy is reused with OFFSET; no
        database-specific table/column knowledge is introduced.
        """
        sql_text = str(sql or "").strip()
        normalized = " ".join(sql_text.lower().split())

        limit_match = re.search(
            r"\blimit\s+(\d+)\b",
            normalized,
            flags=re.IGNORECASE,
        )

        if not limit_match:
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": "The query has no explicit numeric LIMIT.",
            }

        requested_limit = int(limit_match.group(1))

        if requested_limit <= 0:
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": "The query LIMIT is not a positive page size.",
            }

        if requested_limit != page_size:
            page_size = requested_limit

        if len(initial_rows) < page_size:
            return {
                "rows": list(initial_rows),
                "complete": True,
                "pages": 1,
                "reason": "The first bounded page is smaller than the page size.",
            }

        if re.search(r"\bcount\s*\(", normalized):
            return {
                "rows": list(initial_rows),
                "complete": True,
                "pages": 1,
                "reason": "Aggregate COUNT query does not require pagination.",
            }

        if re.search(r"\bgroup\s+by\b", normalized):
            return {
                "rows": list(initial_rows),
                "complete": True,
                "pages": 1,
                "reason": "Grouped query is treated as a bounded analytical result.",
            }

        if not re.search(r"\border\s+by\b", normalized):
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": (
                    "A complete multi-page collection requires an explicit "
                    "ORDER BY so pagination is deterministic."
                ),
            }

        adapter = self.database_manager.get(connection_id)

        allowed_tables = {
            (
                str(table["schema"]).lower(),
                str(table["table"]).lower(),
            )
            for table in schema_context
        }

        sensitive_tokens = (
            "password",
            "passwd",
            "secret",
            "token",
            "api_key",
            "private_key",
            "access_key",
            "credential",
            "ssn",
            "social_security",
            "credit_card",
            "card_number",
            "security_answer",
            "private_note",
        )

        blocked_columns = set()

        for table in schema_context:
            for column in table.get("columns", []):
                if isinstance(column, dict):
                    column_name = str(
                        column.get(
                            "name",
                            column.get("column", ""),
                        )
                    ).lower()
                else:
                    column_name = str(column).lower()

                if any(
                    token in column_name
                    for token in sensitive_tokens
                ):
                    blocked_columns.add(
                        (
                            table["schema"],
                            table["table"],
                            column_name,
                        )
                    )

        validator = SQLSecurityValidator(
            allowed_tables=allowed_tables,
            blocked_columns=blocked_columns,
            schema_context=schema_context,
        )

        executor = SecureQueryExecutor(
            adapter=adapter,
            validator=validator,
        )

        all_rows = list(initial_rows)
        pages = 1
        offset = page_size

        while pages < max_pages:
            page_sql = self._with_offset(
                sql_text,
                offset=offset,
            )

            validation = validator.validate(page_sql)

            if not validation.allowed:
                return {
                    "rows": all_rows,
                    "complete": False,
                    "pages": pages,
                    "reason": (
                        "A pagination page was rejected by the "
                        f"deterministic SQL validator: {validation.reason}"
                    ),
                }

            execution = executor.execute(
                sql=page_sql
            )

            if not execution.success:
                return {
                    "rows": all_rows,
                    "complete": False,
                    "pages": pages,
                    "reason": (
                        "A pagination page failed during execution: "
                        f"{execution.error}"
                    ),
                }

            page_rows = execution.rows
            pages += 1
            all_rows.extend(page_rows)

            if len(page_rows) < page_size:
                return {
                    "rows": all_rows,
                    "complete": True,
                    "pages": pages,
                    "reason": (
                        "Pagination reached the final page."
                    ),
                }

            offset += page_size

        return {
            "rows": all_rows,
            "complete": False,
            "pages": pages,
            "reason": (
                "Pagination reached the configured maximum page count "
                "before proving completeness."
            ),
        }

    @staticmethod
    def _with_offset(
        sql: str,
        offset: int,
    ) -> str:
        statement = sql.strip().rstrip(";").strip()

        # If the original query already has OFFSET, replace it so every
        # page is deterministic relative to the original LIMIT.
        if re.search(
            r"\boffset\s+\d+\b",
            statement,
            flags=re.IGNORECASE,
        ):
            statement = re.sub(
                r"\boffset\s+\d+\b",
                f"OFFSET {offset}",
                statement,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            statement = f"{statement} OFFSET {offset}"

        return statement

    @staticmethod
    def _failure(
        question: str,
        reason: str,
        schema_context: list[dict],
    ) -> dict:

        return {
            "question": question,
            "sql": None,
            "tables_used": [],
            "columns_used": [],
            "allowed": False,
            "executed": False,
            "verified": False,
            "rows": [],
            "row_count": 0,
            "reason": reason,
            "verification": None,
            "semantic_verification": None,
            "schema_context": schema_context,
            "model_route": {},
            "pagination": {
                "complete": False,
                "pages": 0,
            },
        }