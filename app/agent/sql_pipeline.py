from app.agent.result_verifier import ResultVerifier
import json
import re
import sqlglot
from sqlglot import exp
from app.core.trace import logger
from app.agent.sql_generator import SQLGenerator
from app.agent.semantic_verifier import SemanticResultVerifier
from app.agent.mongo_generator import (
    MetricFormulaEnforcementError,
    MongoQueryGenerator,
)
from app.agent.query_intent import (
    AnalyticalQueryIntent,
    QueryIntentPlanner,
)
from app.database.executor import SecureQueryExecutor
from app.database.manager import DatabaseManager
from app.database.schema_retriever import get_relevant_schema
from app.security.sql_validator import SQLSecurityValidator
from app.security.mongodb_validator import MongoQuerySecurityValidator


class SQLPipeline:
    MAX_EVIDENCE_ROWS = 100
    COLLECTION_TOTAL_ALIAS = "__db_sentinel_total_rows"

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
        self.mongo_generator = MongoQueryGenerator(
            model_router=model_router
        )
        self.query_intent_planner = QueryIntentPlanner(
            model_router=model_router
        )

        self.result_verifier = ResultVerifier()
        self.semantic_verifier = SemanticResultVerifier()

    def _audit_context(
        self,
        connection_id: str,
        organization_id: str | None,
    ) -> dict[str, str]:
        connection = self.database_manager.get_connection(connection_id)
        return {
            "organization_id": organization_id or connection.organization_id,
            "workspace_id": connection.workspace_id,
            "connection_id": connection_id,
        }

    def generate_validate_and_execute(
        self,
        connection_id: str,
        question: str,
        required_information: list[str],
        organization_id: str | None = None,
        complexity: str | None = None,
        schema_context: list[dict] | None = None,
        feedback: str | None = None,
        evidence_context: list[dict] | None = None,
        previous_sqls: list[str] | None = None,
        attempt_number: int = 1,
        query_intent: AnalyticalQueryIntent | dict | None = None,
    ) -> dict:
        adapter = self.database_manager.get(
            connection_id,
            organization_id=organization_id,
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
                    "No relevant database schema was found for the "
                    "requested information."
                ),
                schema_context=[],
            )

        intent_question = question.splitlines()[0].strip()
        if query_intent is None or (feedback and attempt_number > 1):
            intent = self.query_intent_planner.create_intent(
                question=intent_question,
                required_information=required_information,
                schema_context=schema_context,
                feedback=feedback,
                attempt_number=attempt_number,
                previous_intent=query_intent,
            )
        elif isinstance(query_intent, AnalyticalQueryIntent):
            intent = query_intent
        else:
            intent = AnalyticalQueryIntent.model_validate(query_intent)

        invalid_references = (
            self.query_intent_planner.invalid_schema_references(
                intent,
                schema_context,
            )
        )
        if invalid_references:
            result = {
                "question": question,
                "sql": None,
                "tables_used": [],
                "columns_used": [],
                "allowed": False,
                "executed": False,
                "verified": False,
                "rows": [],
                "row_count": 0,
                "reason": (
                    "The analytical intent references fields absent from "
                    "the supplied live schema: "
                    + ", ".join(invalid_references)
                ),
                "verification": None,
                "semantic_verification": None,
                "schema_context": schema_context,
                "model_route": {},
                "pagination": {"complete": False, "pages": 0},
                "query_language": (
                    "mongodb"
                    if self.database_manager.get_connection(
                        connection_id
                    ).database_type.lower() == "mongodb"
                    else "sql"
                ),
                "query_intent": None,
            }
            return result

        result = self._generate_validate_and_execute_with_intent(
            connection_id=connection_id,
            question=question,
            required_information=required_information,
            organization_id=organization_id,
            complexity=complexity,
            schema_context=schema_context,
            feedback=feedback,
            evidence_context=evidence_context,
            previous_sqls=previous_sqls,
            attempt_number=attempt_number,
            query_intent=intent,
        )
        result["query_intent"] = intent.model_dump(mode="json")
        return result

    def _generate_validate_and_execute_with_intent(
        self,
        connection_id: str,
        question: str,
        required_information: list[str],
        organization_id: str | None = None,
        complexity: str | None = None,
        schema_context: list[dict] | None = None,
        feedback: str | None = None,
        evidence_context: list[dict] | None = None,
        previous_sqls: list[str] | None = None,
        attempt_number: int = 1,
        query_intent: AnalyticalQueryIntent | None = None,
    ) -> dict:

        adapter = self.database_manager.get(
            connection_id,
            organization_id=organization_id,
        )
        connection = self.database_manager.get_connection(connection_id)

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

        if connection.database_type.lower() == "mongodb":
            return self._generate_validate_and_execute_mongo(
                connection_id=connection_id,
                question=question,
                schema_context=schema_context,
                organization_id=organization_id,
                complexity=complexity,
                feedback=feedback,
                evidence_context=evidence_context,
                previous_queries=previous_sqls,
                attempt_number=attempt_number,
                query_intent=query_intent,
            )

        dialect = {
            "postgresql": "postgres",
            "postgres": "postgres",
            "mysql": "mysql",
            "sqlserver": "tsql",
            "sql server": "tsql",
            "microsoft sql server": "tsql",
            "mssql": "tsql",
            "sql": "tsql",
        }.get(connection.database_type.lower())
        if dialect is None:
            return self._failure(
                question=question,
                reason=(
                    f"No relational SQL dialect is configured for "
                    f"database type '{connection.database_type}'."
                ),
                schema_context=schema_context,
            )

        generated = self.generator.generate(
            question=question,
            schema_context=schema_context,
            complexity=complexity,
            feedback=feedback,
            evidence_context=evidence_context,
            previous_sqls=previous_sqls,
            attempt_number=attempt_number,
            dialect=dialect,
            query_intent=query_intent,
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
        formula_issue = self._metric_formula_issue(
            sql,
            dialect,
            query_intent,
        )
        if formula_issue:
            return {
                "question": question,
                "sql": sql,
                "tables_used": generated.tables_used,
                "columns_used": generated.columns_used,
                "allowed": False,
                "executed": False,
                "verified": False,
                "rows": [],
                "row_count": 0,
                "reason": formula_issue,
                "verification": None,
                "semantic_verification": None,
                "schema_context": schema_context,
                "model_route": generation_route,
                "query_language": dialect,
                "pagination": {"complete": False, "pages": 0},
            }
        execution_sql = sql
        collection_scan_enabled = False
        extreme_ties_requested = self._is_implicit_extreme_tie_request(
            question
        )
        if extreme_ties_requested:
            execution_sql = (
                self._build_single_query_extreme_tie_scan(
                    sql,
                    dialect,
                )
                or ""
            )
            if not execution_sql:
                return {
                    "question": question,
                    "sql": sql,
                    "tables_used": generated.tables_used,
                    "columns_used": generated.columns_used,
                    "allowed": False,
                    "executed": False,
                    "verified": False,
                    "rows": [],
                    "row_count": 0,
                    "reason": (
                        "The query could not be safely transformed to "
                        "return every entity tied at the requested extreme. "
                        "No partial top-one answer was accepted."
                    ),
                    "verification": None,
                    "semantic_verification": None,
                    "schema_context": schema_context,
                    "model_route": generation_route,
                    "query_language": dialect,
                    "pagination": {
                        "complete": False,
                        "pages": 0,
                    },
                }
            collection_scan_enabled = True
        elif self._is_complete_collection_request(question):
            execution_sql = self._build_single_query_collection_scan(
                sql,
                dialect,
            ) or sql
            collection_scan_enabled = execution_sql != sql

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
                "query_language": dialect,
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
            dialect=dialect,
        )

        validation = validator.validate(execution_sql)

        if not validation.allowed:

            return {
                "question": question,
                "sql": execution_sql,
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
                "query_language": dialect,
                "pagination": {
                    "complete": False,
                    "pages": 0,
                },
            }

        executor = SecureQueryExecutor(
            adapter=adapter,
            validator=validator,
            audit_context=self._audit_context(
                connection_id,
                organization_id,
            ),
        )

        execution = executor.execute(
            sql=execution_sql
        )

        if not execution.success:

            return {
                "question": question,
                "sql": execution_sql,
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

        rows = execution.rows
        collection_scan = None
        if collection_scan_enabled:
            rows, collection_scan = self._extract_collection_scan_total(
                execution.rows
            )
            if collection_scan is None:
                return {
                    "question": question,
                    "sql": execution_sql,
                    "source_sql": sql,
                    "tables_used": generated.tables_used,
                    "columns_used": generated.columns_used,
                    "allowed": True,
                    "executed": True,
                    "verified": False,
                    "rows": rows,
                    "row_count": len(rows),
                    "reason": (
                        "The single-query collection scan did not return "
                        "a valid total-row count."
                    ),
                    "verification": None,
                    "semantic_verification": None,
                    "schema_context": schema_context,
                    "model_route": generation_route,
                    "pagination": {
                        "complete": False,
                        "pages": 1,
                    },
                    "collection_scan": {
                        "complete": False,
                        "mode": "single_query",
                    },
                    "query_language": dialect,
                }
            if extreme_ties_requested:
                collection_scan["mode"] = "single_query_extreme_ties"
                collection_scan["ranking_metric"] = (
                    self._ranking_metric_from_sql(sql, dialect)
                )

        verification = self.result_verifier.verify(
            adapter=adapter,
            sql=execution_sql,
            rows=rows,
        )

        semantic_verification = (
            self.semantic_verifier.verify(
                adapter=adapter,
                sql=sql,
                rows=rows,
                schema_context=schema_context,
                dialect=dialect,
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
            "sql": execution_sql,
            "source_sql": sql,
            "tables_used": generated.tables_used,
            "columns_used": generated.columns_used,
            "allowed": True,
            "executed": True,
            "verified": verified,
            "rows": rows,
            "row_count": (
                collection_scan["total_rows"]
                if collection_scan is not None
                else execution.row_count
            ),
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
                "complete": collection_scan is not None,
                "pages": 1,
                **(
                    {"total_rows": collection_scan["total_rows"]}
                    if collection_scan is not None
                    else {}
                ),
            },
            "collection_scan": collection_scan,
            "query_language": dialect,
        }

    @staticmethod
    def _is_complete_collection_request(question: str) -> bool:
        text = str(question or "")
        if re.search(
            r"\b(?:top|highest|most|largest|maximum|least|lowest|smallest|minimum|rank|ranking)\b",
            text,
            re.IGNORECASE,
        ):
            return False
        return bool(
            re.search(
                r"\b(?:which|list|show|return|find|identify|name)\b",
                text,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(?:all|each|every|unique|distinct)\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_implicit_extreme_tie_request(question: str) -> bool:
        text = str(question or "")
        explicit_top_n = re.search(
            r"\b(?:top|first)\s+\d+\b",
            text,
            re.IGNORECASE,
        )
        return bool(
            not explicit_top_n
            and re.search(
                r"\b(?:which|who)\b",
                text,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(?:most|highest|largest|maximum|max|least|lowest|smallest|minimum|min)\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _ranking_metric_from_sql(sql: str, dialect: str) -> str | None:
        try:
            query = sqlglot.parse_one(sql, read=dialect)
        except sqlglot.errors.ParseError:
            return None
        order = query.args.get("order") if isinstance(query, exp.Select) else None
        if order is None or not order.expressions:
            return None
        first = order.expressions[0]
        return (
            first.this.name
            if isinstance(first, exp.Ordered)
            and isinstance(first.this, exp.Column)
            else None
        )

    @classmethod
    def _build_single_query_extreme_tie_scan(
        cls,
        sql: str,
        dialect: str,
    ) -> str | None:
        try:
            query = sqlglot.parse_one(sql, read=dialect)
        except sqlglot.errors.ParseError:
            return None
        if not isinstance(query, exp.Select):
            return None

        order = query.args.get("order")
        if (
            query.args.get("limit") is None
            or query.args.get("offset") is not None
            or order is None
            or query.args.get("locks")
        ):
            return None

        ordered_expressions = order.expressions
        if not ordered_expressions or any(
            not isinstance(item, exp.Ordered)
            or not isinstance(item.this, exp.Column)
            for item in ordered_expressions
        ):
            return None
        metric_order = ordered_expressions[0]
        descending = bool(metric_order.args.get("desc"))

        output_names: dict[str, str] = {}
        source_names: dict[str, str | None] = {}
        for projection in query.expressions:
            if isinstance(projection, exp.Star):
                return None
            output_name = str(projection.alias_or_name or "")
            if not output_name or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*",
                output_name,
            ):
                return None
            normalized_output = output_name.lower()
            if normalized_output in {
                cls.COLLECTION_TOTAL_ALIAS,
                "__db_sentinel_extreme_value",
            } or normalized_output in output_names:
                return None
            output_names[normalized_output] = output_name

            expression = (
                projection.this
                if isinstance(projection, exp.Alias)
                else projection
            )
            if isinstance(expression, exp.Column):
                source_name = expression.name.lower()
                if source_name in source_names:
                    source_names[source_name] = None
                else:
                    source_names[source_name] = output_name

        def resolve_output(column: exp.Column) -> str | None:
            return output_names.get(column.name.lower()) or source_names.get(
                column.name.lower()
            )

        metric_output = resolve_output(metric_order.this)
        if metric_output is None:
            return None

        output_columns = list(output_names.values())
        if not output_columns:
            return None

        sort_columns: list[str] = []
        for ordering in ordered_expressions:
            output_name = resolve_output(ordering.this)
            if output_name is None:
                return None
            direction = " DESC" if ordering.args.get("desc") else " ASC"
            nulls_first = ordering.args.get("nulls_first")
            nulls_last = ordering.args.get("nulls_last")
            nulls_order = (
                " NULLS FIRST"
                if nulls_first
                else " NULLS LAST"
                if nulls_last
                else ""
            )
            sort_columns.append(
                "__db_sentinel_ties."
                f"{exp.to_identifier(output_name, quoted=True).sql(dialect=dialect)}"
                f"{direction}{nulls_order}"
            )

        inner_query = query.copy()
        inner_query.set("limit", None)
        inner_query.set("offset", None)
        inner_query.set("order", None)
        inner_sql = inner_query.sql(dialect=dialect)
        base_alias = exp.to_identifier(
            "__db_sentinel_base",
            quoted=True,
        ).sql(dialect=dialect)
        metrics_alias = exp.to_identifier(
            "__db_sentinel_metrics",
            quoted=True,
        ).sql(dialect=dialect)
        ties_alias = exp.to_identifier(
            "__db_sentinel_ties",
            quoted=True,
        ).sql(dialect=dialect)
        metric_identifier = exp.to_identifier(
            metric_output,
            quoted=True,
        ).sql(dialect=dialect)
        extreme_function = "MAX" if descending else "MIN"
        extreme_alias = exp.to_identifier(
            "__db_sentinel_extreme_value",
            quoted=True,
        ).sql(dialect=dialect)
        selected_columns = ", ".join(
            f"{ties_alias}."
            f"{exp.to_identifier(name, quoted=True).sql(dialect=dialect)}"
            for name in output_columns
        )
        wrapped_sql = (
            f"SELECT {selected_columns}, COUNT(*) OVER() AS "
            f"{cls.COLLECTION_TOTAL_ALIAS} "
            f"FROM (SELECT {metrics_alias}.* FROM ("
            f"SELECT {base_alias}.*, "
            f"{extreme_function}({base_alias}.{metric_identifier}) "
            f"OVER() AS {extreme_alias} "
            f"FROM ({inner_sql}) AS {base_alias}"
            f") AS {metrics_alias} "
            f"WHERE {metrics_alias}.{extreme_alias} = "
            f"{metrics_alias}.{metric_identifier}"
            f") AS {ties_alias} "
            f"ORDER BY {', '.join(sort_columns)} "
            f"LIMIT {cls.MAX_EVIDENCE_ROWS}"
        )
        try:
            parsed_wrapper = sqlglot.parse_one(
                wrapped_sql,
                read=dialect,
            )
        except sqlglot.errors.ParseError:
            return None
        if not isinstance(parsed_wrapper, exp.Select):
            return None
        return parsed_wrapper.sql(dialect=dialect)

    @classmethod
    def _build_single_query_collection_scan(
        cls,
        sql: str,
        dialect: str,
    ) -> str | None:
        try:
            query = sqlglot.parse_one(sql, read=dialect)
        except sqlglot.errors.ParseError:
            return None
        if not isinstance(query, exp.Select):
            return None

        limit = query.args.get("limit")
        order = query.args.get("order")
        if (
            limit is None
            or query.args.get("offset") is not None
            or order is None
            or query.args.get("locks")
        ):
            return None

        projection_names: dict[str, str] = {}
        source_names: dict[str, str | None] = {}
        for projection in query.expressions:
            if isinstance(projection, exp.Star):
                return None
            output_name = str(projection.alias_or_name or "")
            if not output_name or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*",
                output_name,
            ):
                return None
            normalized_output = output_name.lower()
            if normalized_output == cls.COLLECTION_TOTAL_ALIAS:
                return None
            if normalized_output in projection_names:
                return None
            projection_names[normalized_output] = output_name

            expression = (
                projection.this
                if isinstance(projection, exp.Alias)
                else projection
            )
            if isinstance(expression, exp.Column):
                source_name = expression.name.lower()
                if source_name in source_names:
                    source_names[source_name] = None
                else:
                    source_names[source_name] = output_name

        sort_columns: list[str] = []
        for ordering in order.expressions:
            if not isinstance(ordering, exp.Ordered) or not isinstance(
                ordering.this,
                exp.Column,
            ):
                return None
            source_name = ordering.this.name.lower()
            output_name = projection_names.get(source_name)
            if output_name is None:
                output_name = source_names.get(source_name)
            if output_name is None:
                return None
            direction = " DESC" if ordering.args.get("desc") else " ASC"
            nulls_first = ordering.args.get("nulls_first")
            nulls_last = ordering.args.get("nulls_last")
            nulls_order = (
                " NULLS FIRST"
                if nulls_first
                else " NULLS LAST"
                if nulls_last
                else ""
            )
            sort_columns.append(
                f"__db_sentinel_collection."
                f"{exp.to_identifier(output_name).sql(dialect=dialect)}"
                f"{direction}{nulls_order}"
            )
        if not sort_columns:
            return None

        inner_query = query.copy()
        inner_query.set("limit", None)
        inner_query.set("offset", None)
        inner_query.set("order", None)
        inner_sql = inner_query.sql(dialect=dialect)
        wrapped_sql = (
            f"SELECT __db_sentinel_collection.*, "
            f"COUNT(*) OVER() AS {cls.COLLECTION_TOTAL_ALIAS} "
            f"FROM ({inner_sql}) AS __db_sentinel_collection "
            f"ORDER BY {', '.join(sort_columns)} LIMIT "
            f"{cls.MAX_EVIDENCE_ROWS}"
        )
        try:
            parsed_wrapper = sqlglot.parse_one(
                wrapped_sql,
                read=dialect,
            )
        except sqlglot.errors.ParseError:
            return None
        if not isinstance(parsed_wrapper, exp.Select):
            return None
        return parsed_wrapper.sql(dialect=dialect)

    @classmethod
    def _extract_collection_scan_total(
        cls,
        rows: list[dict],
    ) -> tuple[list[dict], dict | None]:
        if not rows:
            return [], {
                "complete": True,
                "mode": "single_query",
                "total_rows": 0,
            }

        count_key = next(
            (
                key
                for key in rows[0]
                if str(key).lower() == cls.COLLECTION_TOTAL_ALIAS
            ),
            None,
        )
        if count_key is None:
            return [dict(row) for row in rows], None
        try:
            total_rows = int(rows[0][count_key])
        except (TypeError, ValueError, OverflowError):
            return [dict(row) for row in rows], None
        if total_rows < len(rows):
            return [dict(row) for row in rows], None
        for row in rows:
            if row.get(count_key) != rows[0][count_key]:
                return [dict(row) for row in rows], None
        preview_rows = [
            {
                key: value
                for key, value in row.items()
                if str(key).lower() != cls.COLLECTION_TOTAL_ALIAS
            }
            for row in rows
        ]
        return preview_rows, {
            "complete": True,
            "mode": "single_query",
            "total_rows": total_rows,
        }

    @staticmethod
    def _metric_formula_issue(
        sql: str,
        dialect: str,
        query_intent: AnalyticalQueryIntent | dict | None,
    ) -> str | None:
        intent_data = (
            query_intent.model_dump(mode="json")
            if isinstance(query_intent, AnalyticalQueryIntent)
            else query_intent or {}
        )
        metrics = intent_data.get("metrics", [])
        formulas = [
            metric
            for metric in metrics
            if isinstance(metric, dict)
            and isinstance(metric.get("formula"), dict)
        ] if isinstance(metrics, list) else []
        if not formulas:
            return None

        try:
            statement = sqlglot.parse_one(sql, read=dialect)
        except sqlglot.errors.ParseError as error:
            return f"Could not verify the requested SQL metric formula: {error}"
        if not isinstance(statement, exp.Query):
            return "The query cannot express the requested metric formula."
        select = statement if isinstance(statement, exp.Select) else statement.find(exp.Select)
        if select is None:
            return "The query cannot express the requested metric formula."

        aggregate_types = {
            "sum": exp.Sum,
            "average": exp.Avg,
            "minimum": exp.Min,
            "maximum": exp.Max,
        }
        formula_types = {
            "multiply": exp.Mul,
            "add": exp.Add,
            "subtract": exp.Sub,
            "divide": exp.Div,
        }

        def column_matches(column: exp.Column, expected: str) -> bool:
            parts = [
                part.name.casefold()
                for part in column.parts
                if part.name
            ]
            expected_parts = [
                part.casefold()
                for part in expected.split(".")
                if part
            ]
            return bool(parts) and (
                parts[-len(expected_parts):] == expected_parts
            )

        def formula_matches(expression, formula: dict) -> bool:
            formula_type = formula_types.get(formula.get("operator"))
            fields = formula.get("fields")
            if (
                formula_type is None
                or not isinstance(fields, list)
                or not fields
                or not all(isinstance(field, str) and field for field in fields)
            ):
                return False
            if formula.get("operator") in {"subtract", "divide"} and len(fields) != 2:
                return False

            def flatten(node):
                if isinstance(node, formula_type) and formula.get("operator") in {
                    "multiply",
                    "add",
                }:
                    return flatten(node.this) + flatten(node.expression)
                return [node]

            operands = flatten(expression)
            if len(operands) != len(fields) or not all(
                isinstance(operand, exp.Column)
                for operand in operands
            ):
                return False
            if formula.get("operator") in {"multiply", "add"}:
                def match_commutative(index: int, remaining: list[str]) -> bool:
                    if index == len(operands):
                        return not remaining
                    return any(
                        column_matches(operands[index], expected)
                        and match_commutative(
                            index + 1,
                            remaining[:position] + remaining[position + 1:],
                        )
                        for position, expected in enumerate(remaining)
                    )

                return match_commutative(0, list(fields))
            return all(
                column_matches(operand, expected)
                for operand, expected in zip(operands, fields)
            )

        for metric in formulas:
            alias = str(metric.get("alias", "")).casefold()
            accumulator_type = aggregate_types.get(metric.get("operation"))
            formula = metric["formula"]
            projections = [
                projection
                for projection in select.expressions
                if projection.alias_or_name.casefold() == alias
            ]
            if not alias or accumulator_type is None or len(projections) != 1:
                return (
                    f"The generated SQL does not expose exactly one requested "
                    f"metric alias for formula '{alias}'."
                )
            aggregates = list(projections[0].find_all(accumulator_type))
            if (
                len(aggregates) != 1
                or not formula_matches(aggregates[0].this, formula)
            ):
                return (
                    f"The generated SQL does not apply the canonical formula "
                    f"to metric '{alias}' before aggregation."
                )
        return None

    def _generate_validate_and_execute_mongo(
        self,
        connection_id: str,
        question: str,
        schema_context: list[dict],
        organization_id: str | None,
        complexity: str | None,
        feedback: str | None,
        evidence_context: list[dict] | None,
        previous_queries: list[str] | None,
        attempt_number: int,
        query_intent: AnalyticalQueryIntent | None,
    ) -> dict:
        try:
            generated = self.mongo_generator.generate(
                question=question,
                schema_context=schema_context,
                complexity=complexity,
                feedback=feedback,
                evidence_context=evidence_context,
                previous_queries=previous_queries,
                attempt_number=attempt_number,
                query_intent=query_intent,
            )
        except MetricFormulaEnforcementError as error:
            logger.warning(
                "MongoDB metric formula enforcement failed: %s",
                error,
            )
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
                "reason": str(error),
                "verification": None,
                "semantic_verification": None,
                "schema_context": schema_context,
                "model_route": dict(
                    getattr(self.model_router, "last_route_metadata", {}) or {}
                ),
                "query_language": "mongodb",
                "pagination": {"complete": False, "pages": 0},
            }
        query_text = self.mongo_generator.serialize(generated)
        normalized_query = " ".join(query_text.lower().split())
        previous_normalized = {
            " ".join(item.lower().split())
            for item in (previous_queries or [])
            if item
        }
        generation_route = dict(
            getattr(self.model_router, "last_route_metadata", {}) or {}
        )
        if normalized_query in previous_normalized:
            return {
                "question": question,
                "sql": query_text,
                "tables_used": [generated.collection],
                "columns_used": generated.fields_used,
                "allowed": False,
                "executed": False,
                "verified": False,
                "rows": [],
                "row_count": 0,
                "reason": (
                    "The generated MongoDB pipeline repeats a previous "
                    "failed attempt."
                ),
                "verification": None,
                "semantic_verification": None,
                "schema_context": schema_context,
                "model_route": generation_route,
                "query_language": "mongodb",
                "pagination": {"complete": False, "pages": 0},
            }
        allowed_collections = {
            (
                str(table["schema"]),
                str(table["table"]),
            )
            for table in schema_context
        }
        validator = MongoQuerySecurityValidator(
            allowed_collections=allowed_collections,
            schema_context=schema_context,
        )
        validation = validator.validate(query_text)
        if not validation.allowed:
            return {
                "question": question,
                "sql": query_text,
                "tables_used": [generated.collection],
                "columns_used": generated.fields_used,
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
                "query_language": "mongodb",
                "pagination": {"complete": False, "pages": 0},
            }

        adapter = self.database_manager.get(
            connection_id,
            organization_id=organization_id,
        )
        execution = SecureQueryExecutor(
            adapter=adapter,
            validator=validator,
            audit_context=self._audit_context(
                connection_id,
                organization_id,
            ),
        ).execute(query_text)
        if not execution.success:
            return {
                "question": question,
                "sql": query_text,
                "tables_used": [generated.collection],
                "columns_used": generated.fields_used,
                "allowed": execution.security_allowed,
                "executed": False,
                "verified": False,
                "rows": [],
                "row_count": 0,
                "reason": execution.error,
                "verification": None,
                "semantic_verification": None,
                "schema_context": schema_context,
                "model_route": generation_route,
                "query_language": "mongodb",
                "pagination": {"complete": False, "pages": 0},
            }

        result_verification = self.result_verifier.verify(
            adapter=adapter,
            sql=query_text,
            rows=execution.rows,
        )
        return {
            "question": question,
            "sql": query_text,
            "tables_used": [generated.collection],
            "columns_used": generated.fields_used,
            "allowed": True,
            "executed": True,
            "verified": result_verification.verified,
            "rows": execution.rows,
            "row_count": execution.row_count,
            "reason": (
                "MongoDB aggregation executed and structural result "
                "checks passed."
                if result_verification.verified
                else (
                    "MongoDB aggregation result verification failed: "
                    + " | ".join(result_verification.warnings)
                )
            ),
            "verification": {
                "verified": result_verification.verified,
                "checks": result_verification.checks,
                "warnings": result_verification.warnings,
            },
            "semantic_verification": {
                "verified": True,
                "checks": [
                    "MongoDB aggregation passed deterministic pipeline "
                    "and collection validation."
                ],
                "warnings": [],
                "errors": [],
            },
            "schema_context": schema_context,
            "model_route": generation_route,
            "query_language": "mongodb",
            "pagination": {"complete": False, "pages": 1},
        }

    def collect_additional_pages(
        self,
        connection_id: str,
        sql: str,
        schema_context: list[dict],
        initial_rows: list[dict],
        organization_id: str | None = None,
        page_size: int = 100,
        expected_top_n: int | None = None,
    ) -> dict:
        """
        Complete a collection query by fetching bounded pages until the
        final page, without removing the per-query SQL LIMIT boundary.

        This is intentionally generic. It only paginates an already
        validated SELECT that:
        - has an explicit numeric LIMIT,
        - has a deterministic ORDER BY,
        - returned a full page.

        The same validated SQL strategy is reused with OFFSET; no
        database-specific table/column knowledge is introduced.
        """
        sql_text = str(sql or "").strip()
        normalized = " ".join(sql_text.lower().split())
        manager = getattr(self, "database_manager", None)
        connection = (
            manager.get_connection(connection_id)
            if manager is not None
            else None
        )
        database_type = (
            connection.database_type.lower()
            if connection is not None
            else "postgres"
        )
        dialect = {
            "postgresql": "postgres",
            "postgres": "postgres",
            "mysql": "mysql",
            "sqlserver": "tsql",
            "sql server": "tsql",
            "microsoft sql server": "tsql",
            "mssql": "tsql",
            "sql": "tsql",
        }.get(database_type, "postgres")
        if database_type == "mongodb":
            return self._collect_mongo_pages(
                connection_id=connection_id,
                query_text=sql_text,
                schema_context=schema_context,
                initial_rows=initial_rows,
                organization_id=organization_id,
                expected_top_n=expected_top_n,
            )

        limit_pattern = (
            r"\btop\s*\(?\s*(\d+)\s*\)?"
            if dialect == "tsql"
            else r"\blimit\s+(\d+)\b"
        )
        limit_match = re.search(limit_pattern, normalized, flags=re.IGNORECASE)

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

        page_size = min(
            requested_limit,
            SecureQueryExecutor.MAX_ROWS,
        )

        if self._is_ungrouped_aggregate(sql_text, dialect):
            return {
                "rows": list(initial_rows),
                "complete": True,
                "pages": 1,
                "reason": (
                    "The ungrouped aggregate query returns a single "
                    "aggregate result and does not require pagination."
                ),
            }

        if (
            expected_top_n is not None
            and expected_top_n <= len(initial_rows)
            and re.search(r"\border\s+by\b", normalized)
        ):
            return {
                "rows": list(initial_rows[:expected_top_n]),
                "complete": True,
                "pages": 1,
                "reason": (
                    "The requested ordered Top-N is present in the "
                    "bounded result."
                ),
            }

        if len(initial_rows) < page_size:
            return {
                "rows": list(initial_rows),
                "complete": True,
                "pages": 1,
                "reason": "The first bounded page is smaller than the page size.",
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

        if not self._has_stable_pagination_order(
            sql=sql_text,
            schema_context=schema_context,
            dialect=dialect,
        ):
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": (
                    "A complete multi-page collection requires an ORDER BY "
                    "that uniquely identifies rows or groups."
                ),
            }

        if manager is None:
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": "Database manager is unavailable for pagination.",
            }

        adapter = manager.get(
            connection_id,
            organization_id=organization_id,
        )

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
            dialect=dialect,
        )

        executor = SecureQueryExecutor(
            adapter=adapter,
            validator=validator,
            audit_context=self._audit_context(
                connection_id,
                self.database_manager.get_connection(
                    connection_id
                ).organization_id,
            ),
        )

        all_rows = list(initial_rows[:self.MAX_EVIDENCE_ROWS])
        total_rows = len(initial_rows)
        pages = 1
        offset = page_size

        while True:
            page_sql = self._with_offset(
                sql_text,
                offset=offset,
                page_size=page_size,
                dialect=dialect,
            )

            validation = validator.validate(page_sql)

            if not validation.allowed:
                return {
                    "rows": all_rows,
                    "total_rows": total_rows,
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
                    "total_rows": total_rows,
                    "complete": False,
                    "pages": pages,
                    "reason": (
                        "A pagination page failed during execution: "
                        f"{execution.error}"
                    ),
                }

            page_rows = execution.rows
            pages += 1
            total_rows += len(page_rows)
            remaining_preview_rows = (
                self.MAX_EVIDENCE_ROWS - len(all_rows)
            )
            if remaining_preview_rows > 0:
                all_rows.extend(page_rows[:remaining_preview_rows])

            if len(page_rows) < page_size:
                return {
                    "rows": all_rows,
                    "total_rows": total_rows,
                    "complete": True,
                    "pages": pages,
                    "reason": (
                        "Pagination reached the final page."
                    ),
                }

            offset += page_size

    @classmethod
    def _has_stable_pagination_order(
        cls,
        sql: str,
        schema_context: list[dict],
        dialect: str,
    ) -> bool:
        try:
            query = sqlglot.parse_one(sql, read=dialect)
        except sqlglot.errors.ParseError:
            return False
        if not isinstance(query, exp.Select):
            return False

        order = query.args.get("order")
        if order is None:
            return False

        ordered_expressions = {
            cls._normalized_expression(item.this, dialect)
            for item in order.expressions
            if isinstance(item, exp.Ordered)
        }
        if not ordered_expressions:
            return False

        projection_aliases = {
            projection.alias.lower(): cls._normalized_expression(
                projection.this,
                dialect,
            )
            for projection in query.expressions
            if isinstance(projection, exp.Alias)
        }
        ordered_expressions.update(
            projection_aliases[expression.name.lower()]
            for item in order.expressions
            if isinstance(item, exp.Ordered)
            for expression in [item.this]
            if (
                isinstance(expression, exp.Column)
                and not expression.table
                and expression.name.lower() in projection_aliases
            )
        )

        if query.args.get("group"):
            group = query.args["group"]
            group_expressions = {
                cls._normalized_expression(item, dialect)
                for item in group.expressions
            }
            return bool(group_expressions) and group_expressions.issubset(
                ordered_expressions
            )

        primary_keys = {
            key.lower()
            for table in schema_context
            for key in cls._primary_key_names(table)
        }
        primary_key_ordered = any(
            isinstance(node, exp.Column)
            and node.name.lower() in primary_keys
            for item in order.expressions
            for node in item.walk()
        )
        if query.args.get("distinct"):
            distinct_expressions = {
                cls._normalized_expression(item, dialect)
                for item in query.expressions
            }
            ordered_distinct_columns = {
                item.name.lower()
                for expression in order.expressions
                if isinstance(expression, exp.Ordered)
                for item in expression.this.walk()
                if isinstance(item, exp.Column)
            }
            selected_primary_key = any(
                isinstance(projection, exp.Column)
                and projection.name.lower() in primary_keys
                for projection in query.expressions
            )
            return bool(
                distinct_expressions
                and (
                    distinct_expressions.issubset(ordered_expressions)
                    or (
                        selected_primary_key
                        and primary_key_ordered
                    )
                )
            )

        return primary_key_ordered

    @staticmethod
    def _normalized_expression(expression: exp.Expression, dialect: str) -> str:
        if isinstance(expression, exp.Alias):
            expression = expression.this
        return expression.sql(dialect=dialect, normalize=True).lower()

    @staticmethod
    def _is_ungrouped_aggregate(sql: str, dialect: str) -> bool:
        try:
            query = sqlglot.parse_one(sql, read=dialect)
        except sqlglot.errors.ParseError:
            return False
        if not isinstance(query, exp.Select) or query.args.get("group"):
            return False
        for projection in query.expressions:
            for node in projection.walk():
                if (
                    isinstance(node, exp.AggFunc)
                    and node.find_ancestor(exp.Select) is query
                    and node.find_ancestor(exp.Window) is None
                ):
                    return True
        return False

    def _collect_mongo_pages(
        self,
        connection_id: str,
        query_text: str,
        schema_context: list[dict],
        initial_rows: list[dict],
        organization_id: str | None,
        expected_top_n: int | None,
    ) -> dict:
        try:
            payload = json.loads(query_text)
        except json.JSONDecodeError as error:
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": f"MongoDB query JSON could not be paginated: {error}",
            }

        pipeline = payload.get("pipeline") if isinstance(payload, dict) else None
        if not isinstance(pipeline, list) or not pipeline:
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": "MongoDB query has no aggregation pipeline.",
            }
        limit = pipeline[-1].get("$limit") if isinstance(pipeline[-1], dict) else None
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": "MongoDB query has no final numeric $limit.",
            }
        limit = min(limit, SecureQueryExecutor.MAX_ROWS)
        group_indices = [
            index
            for index, stage in enumerate(pipeline)
            if isinstance(stage, dict) and "$group" in stage
        ]
        if group_indices:
            final_group_index = group_indices[-1]
            final_group = pipeline[final_group_index]["$group"]
            trailing_stages = pipeline[final_group_index + 1:-1]
            single_result_tail = {
                "$project",
                "$addFields",
                "$set",
                "$unset",
                "$sort",
                "$limit",
                "$skip",
                "$replaceRoot",
                "$replaceWith",
            }
            if (
                isinstance(final_group, dict)
                and final_group.get("_id") is None
                and len(initial_rows) <= 1
                and all(
                    isinstance(stage, dict)
                    and len(stage) == 1
                    and next(iter(stage)) in single_result_tail
                    for stage in trailing_stages
                )
            ):
                return {
                    "rows": list(initial_rows),
                    "total_rows": len(initial_rows),
                    "complete": True,
                    "pages": 1,
                    "reason": (
                        "The global MongoDB aggregation produces at most "
                        "one result row."
                    ),
                }
        if len(initial_rows) < limit:
            return {
                "rows": list(initial_rows),
                "complete": True,
                "pages": 1,
                "reason": "The first bounded result is smaller than its limit.",
            }

        if any("$count" in stage for stage in pipeline):
            if expected_top_n == len(initial_rows) == limit:
                return {
                    "rows": list(initial_rows),
                    "complete": True,
                    "pages": 1,
                    "reason": "The requested MongoDB Top-N result is complete.",
                }
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": (
                    "A MongoDB $count result should contain at most one row."
                ),
            }

        sort_indices = [
            index
            for index, stage in enumerate(pipeline)
            if isinstance(stage, dict) and "$sort" in stage
        ]
        sort_stages = [
            stage.get("$sort")
            for stage in pipeline
            if isinstance(stage, dict) and "$sort" in stage
        ]
        if not sort_stages or not isinstance(sort_stages[-1], dict):
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": (
                    "Complete MongoDB collection pagination requires a "
                    "deterministic $sort."
                ),
            }
        if group_indices and sort_indices[-1] < group_indices[-1]:
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": (
                    "Complete grouped MongoDB pagination requires a "
                    "$sort after the final $group stage."
                ),
            }
        if "_id" not in sort_stages[-1]:
            return {
                "rows": list(initial_rows),
                "complete": False,
                "pages": 1,
                "reason": (
                    "Complete MongoDB collection pagination requires _id "
                    "as the final stable sort key."
                ),
            }

        # Pagination must reapply the same deterministic validator used
        # for the original aggregation and retain the connection's tenant.
        adapter = self.database_manager.get(
            connection_id,
            organization_id=organization_id,
        )
        allowed_collections = {
            (str(table["schema"]), str(table["table"]))
            for table in schema_context
        }
        validator = MongoQuerySecurityValidator(
            allowed_collections=allowed_collections,
            schema_context=schema_context,
        )
        executor = SecureQueryExecutor(
            adapter=adapter,
            validator=validator,
            audit_context=self._audit_context(
                connection_id,
                organization_id,
            ),
        )
        rows = list(initial_rows[:self.MAX_EVIDENCE_ROWS])
        total_rows = len(initial_rows)
        offset = limit
        pages = 1
        while True:
            page_pipeline = [
                stage
                for stage in pipeline[:-1]
                if not (isinstance(stage, dict) and "$skip" in stage)
            ]
            page_pipeline.append({"$skip": offset})
            page_pipeline.append({"$limit": limit})
            page_query = json.dumps(
                {
                    "collection": payload["collection"],
                    "pipeline": page_pipeline,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            execution = executor.execute(page_query)
            if not execution.success:
                return {
                    "rows": rows,
                    "total_rows": total_rows,
                    "complete": False,
                    "pages": pages,
                    "reason": (
                        "A MongoDB pagination query was rejected or failed: "
                        f"{execution.error}"
                    ),
                }
            page_rows = execution.rows
            pages += 1
            total_rows += len(page_rows)
            remaining_preview_rows = (
                self.MAX_EVIDENCE_ROWS - len(rows)
            )
            if remaining_preview_rows > 0:
                rows.extend(page_rows[:remaining_preview_rows])
            if len(page_rows) < limit:
                return {
                    "rows": rows,
                    "total_rows": total_rows,
                    "complete": True,
                    "pages": pages,
                    "reason": "MongoDB pagination reached the final page.",
                }
            offset += limit

    @staticmethod
    def _primary_key_names(table: dict) -> list[str]:
        keys = table.get(
            "primary_keys",
            table.get("primary_key", []),
        )
        if isinstance(keys, (str, dict)):
            keys = [keys]

        names = []
        for key in keys:
            if isinstance(key, dict):
                name = key.get("column", key.get("name"))
            else:
                name = key
            if name:
                names.append(str(name))
        return names

    @staticmethod
    def _with_offset(
        sql: str,
        offset: int,
        page_size: int = 100,
        dialect: str = "postgres",
    ) -> str:
        statement = sql.strip().rstrip(";").strip()

        if dialect == "tsql":
            statement = re.sub(
                r"\btop\s*\(?\s*\d+\s*\)?\s*",
                "",
                statement,
                count=1,
                flags=re.IGNORECASE,
            )
            statement = re.sub(
                r"\boffset\s+\d+\s+rows"
                r"(?:\s+fetch\s+next\s+\d+\s+rows\s+only)?",
                "",
                statement,
                count=1,
                flags=re.IGNORECASE,
            ).rstrip()
            return (
                f"{statement} OFFSET {offset} ROWS "
                f"FETCH NEXT {page_size} ROWS ONLY"
            )

        statement = re.sub(
            r"\blimit\s+\d+\b",
            f"LIMIT {page_size}",
            statement,
            count=1,
            flags=re.IGNORECASE,
        )

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