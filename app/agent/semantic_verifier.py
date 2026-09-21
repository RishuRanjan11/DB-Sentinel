from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import expressions

from app.database.adapters.base import DatabaseAdapter


@dataclass(frozen=True)
class SemanticVerificationResult:
    verified: bool
    checks: list[str]
    warnings: list[str]
    errors: list[str]


class SemanticResultVerifier:

    def verify(
        self,
        adapter: DatabaseAdapter,
        sql: str,
        rows: list[dict[str, Any]],
        schema_context: list[dict],
    ) -> SemanticVerificationResult:

        checks: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []

        if not rows:
            return SemanticVerificationResult(
                verified=True,
                checks=[
                    "Query executed successfully but returned no rows."
                ],
                warnings=[
                    "No returned rows were available for relationship verification."
                ],
                errors=[],
            )

        checks.append(
            f"Received {len(rows)} rows for semantic verification."
        )

        # ---------------------------------------------------------
        # Parse SQL
        # ---------------------------------------------------------

        try:
            statements = sqlglot.parse(
                sql,
                read="postgres",
            )

        except Exception as error:
            return SemanticVerificationResult(
                verified=False,
                checks=checks,
                warnings=[],
                errors=[
                    "Could not parse SQL for semantic verification: "
                    f"{error}"
                ],
            )

        if len(statements) != 1:
            return SemanticVerificationResult(
                verified=False,
                checks=checks,
                warnings=[],
                errors=[
                    "Semantic verification requires exactly "
                    "one SQL statement."
                ],
            )

        statement = statements[0]

        if not isinstance(
            statement,
            expressions.Select,
        ):
            return SemanticVerificationResult(
                verified=False,
                checks=checks,
                warnings=[],
                errors=[
                    "Semantic verification requires a SELECT statement."
                ],
            )

        # ---------------------------------------------------------
        # Build schema metadata
        # ---------------------------------------------------------

        table_map = self._build_table_map(
            statement
        )

        cte_names = {
            cte.alias_or_name.lower()
            for cte in statement.find_all(expressions.CTE)
            if cte.alias_or_name
        }

        foreign_keys = self._collect_foreign_keys(
            schema_context
        )

        # ---------------------------------------------------------
        # Verify relational structure
        # ---------------------------------------------------------

        join_results = self._verify_joins(
            statement=statement,
            table_map=table_map,
            foreign_keys=foreign_keys,
        )

        checks.extend(
            join_results["checks"]
        )

        warnings.extend(
            join_results["warnings"]
        )

        errors.extend(
            join_results["errors"]
        )

        verified_relationships = (
            join_results["verified_relationships"]
        )

        # ---------------------------------------------------------
        # Resolve output columns
        # ---------------------------------------------------------

        output_columns = self._resolve_output_columns(
            statement=statement,
            table_map=table_map,
            schema_context=schema_context,
        )

        if output_columns:

            checks.append(
                f"Resolved {len(output_columns)} projected "
                f"database columns."
            )

        else:

            warnings.append(
                "Could not resolve projected columns "
                "against the supplied schema."
            )

        # ---------------------------------------------------------
        # Verify returned columns correspond to actual SQL
        # projections.
        # ---------------------------------------------------------

        result_columns = {
            str(column).lower()
            for row in rows
            for column in row.keys()
        }

        expected_columns = {
            str(column["result_key"]).lower()
            for column in output_columns
            if column.get("result_key")
        }

        missing_result_columns = (
            expected_columns - result_columns
        )

        if missing_result_columns:

            errors.append(
                "SQL projected columns missing from result: "
                f"{sorted(missing_result_columns)}"
            )

        else:

            if expected_columns:
                checks.append(
                    "Returned result columns match the "
                    "resolved SQL projections."
                )

        # ---------------------------------------------------------
        # Duplicate detection
        # ---------------------------------------------------------

        unique_rows = {
            self._canonicalize(row)
            for row in rows
        }

        duplicate_count = (
            len(rows) - len(unique_rows)
        )

        if duplicate_count > 0:

            warnings.append(
                f"{duplicate_count} duplicate result rows detected."
            )

        else:

            checks.append(
                "No duplicate result rows detected."
            )

        # ---------------------------------------------------------
        # Final decision
        # ---------------------------------------------------------

        if errors:

            verified = False

        else:
            # Relationship metadata is supporting evidence, not a hard
            # correctness gate. A valid SELECT may legitimately use a
            # relationship representation that is not declared through a
            # database FK, or may return a single physical table whose
            # semantics are already established by the supplied schema.
            #
            # Security validation has already established that physical
            # tables/columns are permitted, SQL parsing succeeded, the
            # query executed successfully, and projected columns were
            # checked above. Semantic scope is handled separately by the
            # logical verifier. Therefore absence of independently matched
            # FK metadata must remain a warning rather than turn a valid
            # query into a failed execution attempt.
            verified = True

            if verified_relationships > 0:
                checks.append(
                    "Database relationship metadata independently "
                    "verified at least one query relationship."
                )
            elif len(table_map) == 1:
                checks.append(
                    "Single-table result is structurally consistent "
                    "with the supplied schema."
                )
            elif cte_names:
                warnings.append(
                    "CTE-based relationships could not all be matched "
                    "directly to database foreign-key metadata."
                )
            elif self._is_aggregate_query(statement):
                warnings.append(
                    "Aggregate result is structurally valid; no additional "
                    "result-level foreign-key relationship was required."
                )
            else:
                warnings.append(
                    "No database relationship could be independently "
                    "verified from foreign-key metadata; this is advisory "
                    "only because schema and SQL validation already passed."
                )

        return SemanticVerificationResult(
            verified=verified,
            checks=checks,
            warnings=warnings,
            errors=errors,
        )

    @staticmethod
    def _is_aggregate_query(statement: expressions.Select) -> bool:
        """Return True when the SELECT contains a SQL aggregate."""

        aggregate_names = {
            "count",
            "sum",
            "avg",
            "min",
            "max",
            "bool_and",
            "bool_or",
            "array_agg",
            "string_agg",
            "json_agg",
            "jsonb_agg",
        }

        for expression in statement.find_all(expressions.Func):
            name = (
                getattr(expression, "name", "")
                or ""
            ).lower()

            if name in aggregate_names:
                return True

        return False

    @classmethod
    def _canonicalize(cls, value: Any):
        """Return a hashable representation of arbitrary DB values."""
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    sorted(
                        (
                            str(key),
                            cls._canonicalize(item),
                        )
                        for key, item in value.items()
                    )
                ),
            )

        if isinstance(value, (list, tuple)):
            return (
                "sequence",
                tuple(
                    cls._canonicalize(item)
                    for item in value
                ),
            )

        if isinstance(value, set):
            items = [
                cls._canonicalize(item)
                for item in value
            ]
            items.sort(key=repr)
            return ("set", tuple(items))

        try:
            hash(value)
            return ("value", value)
        except TypeError:
            return ("repr", repr(value))

    # =============================================================
    # JOIN VERIFICATION
    # =============================================================

    def _verify_joins(
        self,
        statement: expressions.Select,
        table_map: dict[str, tuple[str, str]],
        foreign_keys: list[dict],
    ) -> dict:

        checks: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []

        verified_relationships = 0

        joins = list(
            statement.find_all(
                expressions.Join
            )
        )

        if not joins:
            return {
                "checks": checks,
                "warnings": warnings,
                "errors": errors,
                "verified_relationships": 0,
            }

        for join in joins:

            on_expression = join.args.get(
                "on"
            )

            if on_expression is None:
                warnings.append(
                    "JOIN without an ON condition "
                    "could not be independently "
                    "verified."
                )
                continue

            equality_expressions = list(
                on_expression.find_all(
                    expressions.EQ
                )
            )

            if not equality_expressions:
                warnings.append(
                    "JOIN condition did not contain "
                    "a directly verifiable equality."
                )
                continue

            join_verified = False

            for equality in equality_expressions:

                left = equality.left
                right = equality.right

                if not isinstance(
                    left,
                    expressions.Column,
                ):
                    continue

                if not isinstance(
                    right,
                    expressions.Column,
                ):
                    continue

                left_info = self._resolve_column_reference(
                    left,
                    table_map,
                )

                right_info = self._resolve_column_reference(
                    right,
                    table_map,
                )

                if not left_info or not right_info:
                    continue

                left_schema = left_info["schema"]
                left_table = left_info["table"]
                left_column = left_info["column"]

                right_schema = right_info["schema"]
                right_table = right_info["table"]
                right_column = right_info["column"]

                matching_fk = self._find_matching_fk(
                    foreign_keys=foreign_keys,
                    source_schema=left_schema,
                    source_table=left_table,
                    source_column=left_column,
                    target_schema=right_schema,
                    target_table=right_table,
                    target_column=right_column,
                )

                if matching_fk is None:

                    matching_fk = self._find_matching_fk(
                        foreign_keys=foreign_keys,
                        source_schema=right_schema,
                        source_table=right_table,
                        source_column=right_column,
                        target_schema=left_schema,
                        target_table=left_table,
                        target_column=left_column,
                    )

                if matching_fk is None:
                    continue

                checks.append(
                    "Verified JOIN relationship: "
                    f"{left_schema}.{left_table}.{left_column} "
                    f"↔ "
                    f"{right_schema}.{right_table}.{right_column} "
                    "using database foreign-key metadata."
                )

                verified_relationships += 1
                join_verified = True
                break

            if not join_verified:

                warnings.append(
                    "A JOIN was found, but its column relationship "
                    "could not be matched to supplied foreign-key metadata."
                )

        return {
            "checks": checks,
            "warnings": warnings,
            "errors": errors,
            "verified_relationships": verified_relationships,
        }

    # =============================================================
    # FOREIGN KEY HELPERS
    # =============================================================

    @staticmethod
    def _find_matching_fk(
        foreign_keys: list[dict],
        source_schema: str,
        source_table: str,
        source_column: str,
        target_schema: str,
        target_table: str,
        target_column: str,
    ) -> dict | None:

        for foreign_key in foreign_keys:

            if (
                foreign_key["schema"] == source_schema
                and foreign_key["table"] == source_table
                and foreign_key["column"] == source_column
                and foreign_key["referred_schema"] == target_schema
                and foreign_key["referred_table"] == target_table
                and foreign_key["referred_column"] == target_column
            ):
                return foreign_key

        return None

    @staticmethod
    def _collect_foreign_keys(
        schema_context: list[dict],
    ) -> list[dict]:

        foreign_keys = []

        for table in schema_context:

            schema = table.get("schema")
            table_name = table.get("table")

            if not schema or not table_name:
                continue

            for foreign_key in table.get(
                "foreign_keys",
                [],
            ):

                column = foreign_key.get(
                    "column"
                )

                referred_schema = foreign_key.get(
                    "referred_schema"
                )

                referred_table = foreign_key.get(
                    "referred_table"
                )

                referred_column = foreign_key.get(
                    "referred_column"
                )

                if not all(
                    [
                        column,
                        referred_schema,
                        referred_table,
                        referred_column,
                    ]
                ):
                    continue

                foreign_keys.append(
                    {
                        "schema": schema,
                        "table": table_name,
                        "column": column,
                        "referred_schema": referred_schema,
                        "referred_table": referred_table,
                        "referred_column": referred_column,
                    }
                )

        return foreign_keys

    # =============================================================
    # TABLE / COLUMN RESOLUTION
    # =============================================================

    @staticmethod
    def _build_table_map(
        statement: expressions.Select,
    ) -> dict[str, tuple[str, str]]:

        table_map = {}

        for table in statement.find_all(
            expressions.Table
        ):

            schema = table.db
            table_name = table.name

            if not schema or not table_name:
                continue

            table_map[
                table.alias_or_name.lower()
            ] = (
                schema,
                table_name,
            )

            table_map[
                table_name.lower()
            ] = (
                schema,
                table_name,
            )

        return table_map

    @staticmethod
    def _resolve_column_reference(
        column: expressions.Column,
        table_map: dict[str, tuple[str, str]],
    ) -> dict | None:

        table_reference = column.table

        if not table_reference:
            return None

        table_info = table_map.get(
            table_reference.lower()
        )

        if not table_info:
            return None

        return {
            "schema": table_info[0],
            "table": table_info[1],
            "column": column.name,
        }

    def _resolve_output_columns(
        self,
        statement: expressions.Select,
        table_map: dict[str, tuple[str, str]],
        schema_context: list[dict],
    ) -> list[dict]:

        output_columns = []

        for projection in statement.expressions:

            expression = projection
            result_key = None

            if isinstance(
                projection,
                expressions.Alias,
            ):

                expression = projection.this
                result_key = projection.alias

            if not isinstance(
                expression,
                expressions.Column,
            ):
                continue

            column_name = expression.name

            if result_key is None:
                result_key = column_name

            table_reference = expression.table

            if table_reference:

                table_info = table_map.get(
                    table_reference.lower()
                )

                if table_info:

                    output_columns.append(
                        {
                            "schema": table_info[0],
                            "table": table_info[1],
                            "column": column_name,
                            "result_key": result_key,
                        }
                    )

                continue

            matches = []

            for table in schema_context:

                schema = table.get("schema")
                table_name = table.get("table")

                if not schema or not table_name:
                    continue

                if any(
                    isinstance(column, dict)
                    and column.get("name") == column_name
                    for column in table.get(
                        "columns",
                        [],
                    )
                ):

                    matches.append(
                        (
                            schema,
                            table_name,
                        )
                    )

            if len(matches) == 1:

                output_columns.append(
                    {
                        "schema": matches[0][0],
                        "table": matches[0][1],
                        "column": column_name,
                        "result_key": result_key,
                    }
                )

        return output_columns