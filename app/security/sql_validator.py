from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import expressions


@dataclass(frozen=True)
class SQLValidationResult:
    allowed: bool
    reason: str


class SQLSecurityValidator:
    """
    Deterministic security and schema boundary for generated SQL.

    The LLM proposes SQL. This class decides whether the SQL is allowed.
    No database, schema, table, or column is hardcoded here.
    """

    MAX_LIMIT = 100

    def __init__(
        self,
        allowed_tables: set[tuple[str, str]],
        blocked_columns: set[tuple[str, str, str]] | None = None,
        schema_context: list[dict[str, Any]] | None = None,
    ):
        self.allowed_tables = {
            (
                str(schema).lower(),
                str(table).lower(),
            )
            for schema, table in allowed_tables
        }

        self.blocked_columns = {
            (
                str(schema).lower(),
                str(table).lower(),
                str(column).lower(),
            )
            for schema, table, column in (blocked_columns or set())
        }

        self.blocked_column_names = {
            column
            for _, _, column in self.blocked_columns
        }

        self.schema_columns = self._build_schema_columns(
            schema_context or []
        )

    @staticmethod
    def _build_schema_columns(
        schema_context: list[dict[str, Any]],
    ) -> dict[tuple[str, str], set[str]]:
        result: dict[tuple[str, str], set[str]] = {}

        for table in schema_context:
            schema = str(
                table.get("schema", "")
            ).lower()

            table_name = str(
                table.get("table", "")
            ).lower()

            if not schema or not table_name:
                continue

            columns: set[str] = set()

            for column in table.get("columns", []) or []:
                if isinstance(column, dict):
                    name = column.get(
                        "name",
                        column.get("column", ""),
                    )
                else:
                    name = column

                if name:
                    columns.add(
                        str(name).lower()
                    )

            result[
                (schema, table_name)
            ] = columns

        return result

    @staticmethod
    def _cte_output_columns(
        statement,
    ) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}

        for cte in statement.find_all(
            expressions.CTE
        ):
            name = (
                cte.alias_or_name or ""
            ).lower()

            if not name:
                continue

            output: set[str] = set()
            query = cte.this

            select = None

            if isinstance(
                query,
                expressions.Select,
            ):
                select = query

            elif query is not None:
                select = query.find(
                    expressions.Select
                )

            if select is not None:

                for projection in select.expressions:

                    alias = getattr(
                        projection,
                        "alias",
                        None,
                    )

                    if alias:
                        output.add(
                            str(alias).lower()
                        )
                        continue

                    if isinstance(
                        projection,
                        expressions.Column,
                    ):
                        output.add(
                            projection.name.lower()
                        )
                        continue

                    output_name = getattr(
                        projection,
                        "output_name",
                        None,
                    )

                    if output_name:
                        output.add(
                            str(output_name).lower()
                        )

            alias_node = cte.args.get(
                "alias"
            )

            if alias_node is not None:

                for column in (
                    alias_node.args.get(
                        "columns"
                    )
                    or []
                ):

                    name_value = getattr(
                        column,
                        "name",
                        None,
                    )

                    if name_value:
                        output.add(
                            str(name_value).lower()
                        )

            result[name] = output

        return result

    @staticmethod
    def _select_aliases(
        statement,
    ) -> set[str]:

        aliases: set[str] = set()

        for select in statement.find_all(
            expressions.Select
        ):

            for projection in select.expressions:

                alias = getattr(
                    projection,
                    "alias",
                    None,
                )

                if alias:
                    aliases.add(
                        str(alias).lower()
                    )

        return aliases

    @staticmethod
    def _derived_relation_aliases(
        statement,
    ) -> set[str]:
        """
        Return aliases created by derived/lateral relations.

        These are not physical database tables and therefore cannot
        be validated against schema_context as physical tables.
        """

        aliases: set[str] = set()

        for node in statement.walk():

            alias = getattr(
                node,
                "alias_or_name",
                None,
            )

            if (
                alias
                and not isinstance(
                    node,
                    expressions.Table,
                )
            ):
                aliases.add(
                    str(alias).lower()
                )

        return aliases

    def validate(
        self,
        sql: str,
    ) -> SQLValidationResult:

        if not sql or not sql.strip():

            return SQLValidationResult(
                False,
                "SQL query is empty.",
            )

        try:

            statements = sqlglot.parse(
                sql,
                read="postgres",
            )

        except Exception as error:

            return SQLValidationResult(
                False,
                f"SQL parsing failed: {error}",
            )

        if len(statements) != 1:

            return SQLValidationResult(
                False,
                "Multiple SQL statements are not allowed.",
            )

        statement = statements[0]

        if not isinstance(
            statement,
            expressions.Select,
        ):

            return SQLValidationResult(
                False,
                "Only SELECT statements are allowed.",
            )

        forbidden_expressions = (
            expressions.Insert,
            expressions.Update,
            expressions.Delete,
            expressions.Create,
            expressions.Drop,
            expressions.Alter,
            expressions.TruncateTable,
            expressions.Grant,
            expressions.Revoke,
        )

        for expression_type in forbidden_expressions:

            if statement.find(
                expression_type
            ) is not None:

                return SQLValidationResult(
                    False,
                    (
                        "Forbidden SQL operation detected: "
                        f"{expression_type.__name__}"
                    ),
                )

        tables = list(
            statement.find_all(
                expressions.Table
            )
        )

        cte_names = {
            cte.alias_or_name.lower()
            for cte in statement.find_all(
                expressions.CTE
            )
            if cte.alias_or_name
        }

        physical_tables = [
            table
            for table in tables
            if table.name.lower()
            not in cte_names
        ]

        if not physical_tables:

            return SQLValidationResult(
                False,
                "Query does not reference any physical table.",
            )

        for table in physical_tables:

            schema = table.db

            if not schema:

                return SQLValidationResult(
                    False,
                    (
                        f"Table '{table.name}' is not "
                        "schema-qualified."
                    ),
                )

            table_key = (
                schema.lower(),
                table.name.lower(),
            )

            if table_key not in self.allowed_tables:

                return SQLValidationResult(
                    False,
                    (
                        f"Table '{schema}.{table.name}' "
                        "is not permitted for this investigation."
                    ),
                )

        relation_map: dict[
            str,
            tuple[str, str],
        ] = {}

        for table in physical_tables:

            relation_map[
                table.alias_or_name.lower()
            ] = (
                table.db.lower(),
                table.name.lower(),
            )

        cte_columns = (
            self._cte_output_columns(
                statement
            )
        )

        select_aliases = (
            self._select_aliases(
                statement
            )
        )

        derived_aliases = (
            self._derived_relation_aliases(
                statement
            )
        )

        for column in statement.find_all(
            expressions.Column
        ):

            column_name = (
                column.name or ""
            ).lower()

            qualifier = (
                column.table or ""
            ).lower()

            if (
                not column_name
                or column_name == "*"
            ):
                continue

            if (
                not qualifier
                and column_name
                in self.blocked_column_names
            ):

                return SQLValidationResult(
                    False,
                    (
                        f"Sensitive column '{column_name}' "
                        "is not permitted by the data-access policy."
                    ),
                )

            if qualifier:

                if qualifier in cte_columns:

                    available = cte_columns[
                        qualifier
                    ]

                    if (
                        available
                        and column_name
                        not in available
                    ):

                        return SQLValidationResult(
                            False,
                            (
                                f"Column '{qualifier}.{column_name}' "
                                f"does not exist in CTE '{qualifier}'. "
                                f"Available CTE columns: "
                                f"{sorted(available)}"
                            ),
                        )

                    continue

                if (
                    qualifier in derived_aliases
                    and qualifier not in relation_map
                ):
                    # Derived/lateral relation columns are validated
                    # by PostgreSQL itself. They are not physical
                    # database columns.
                    continue

                table_key = relation_map.get(
                    qualifier
                )

                if table_key is None:

                    return SQLValidationResult(
                        False,
                        (
                            f"Column reference "
                            f"'{qualifier}.{column_name}' "
                            "uses an unknown table alias "
                            "or relation."
                        ),
                    )

                schema, table_name = table_key

                columns = self.schema_columns.get(
                    (schema, table_name),
                    set(),
                )

                if (
                    columns
                    and column_name
                    not in columns
                ):

                    return SQLValidationResult(
                        False,
                        (
                            f"Column "
                            f"'{schema}.{table_name}.{column_name}' "
                            "does not exist in the supplied schema. "
                            f"Available columns: "
                            f"{sorted(columns)}"
                        ),
                    )

                if (
                    schema,
                    table_name,
                    column_name,
                ) in self.blocked_columns:

                    return SQLValidationResult(
                        False,
                        (
                            f"Sensitive column "
                            f"'{schema}.{table_name}."
                            f"{column_name}' is not permitted "
                            "by the data-access policy."
                        ),
                    )

                continue

            if column_name in select_aliases:
                continue

            cte_matches = [
                name
                for name, columns
                in cte_columns.items()
                if column_name in columns
            ]

            physical_matches = [
                (
                    schema,
                    table_name,
                )
                for schema, table_name
                in relation_map.values()
                if column_name
                in self.schema_columns.get(
                    (schema, table_name),
                    set(),
                )
            ]

            if (
                not cte_matches
                and not physical_matches
            ):

                if all(
                    self.schema_columns.get(
                        (schema, table_name)
                    )
                    for schema, table_name
                    in relation_map.values()
                ):

                    return SQLValidationResult(
                        False,
                        (
                            f"Column '{column_name}' "
                            "does not exist in the supplied schema."
                        ),
                    )

            if (
                len(cte_matches)
                + len(physical_matches)
                > 1
            ):

                return SQLValidationResult(
                    False,
                    (
                        f"Column '{column_name}' is ambiguous "
                        f"across relations: "
                        f"{cte_matches + physical_matches}. "
                        "Qualify the column explicitly."
                    ),
                )

            if physical_matches:

                schema, table_name = (
                    physical_matches[0]
                )

                if (
                    schema,
                    table_name,
                    column_name,
                ) in self.blocked_columns:

                    return SQLValidationResult(
                        False,
                        (
                            f"Sensitive column "
                            f"'{schema}.{table_name}."
                            f"{column_name}' is not permitted "
                            "by the data-access policy."
                        ),
                    )

        limit = statement.args.get(
            "limit"
        )

        if limit is None:

            return SQLValidationResult(
                False,
                "LIMIT clause is required.",
            )

        limit_expression = (
            limit.expression
        )

        if not isinstance(
            limit_expression,
            expressions.Literal,
        ):

            return SQLValidationResult(
                False,
                "LIMIT must use a numeric literal.",
            )

        if not limit_expression.is_int:

            return SQLValidationResult(
                False,
                "LIMIT must be an integer.",
            )

        try:

            limit_value = int(
                limit_expression.this
            )

        except (
            ValueError,
            TypeError,
        ):

            return SQLValidationResult(
                False,
                "LIMIT must be a numeric value.",
            )

        if limit_value <= 0:

            return SQLValidationResult(
                False,
                "LIMIT must be greater than zero.",
            )

        if limit_value > self.MAX_LIMIT:

            return SQLValidationResult(
                False,
                (
                    f"LIMIT {limit_value} exceeds "
                    f"maximum allowed LIMIT "
                    f"of {self.MAX_LIMIT}."
                ),
            )

        return SQLValidationResult(
            True,
            "SQL passed security and schema validation.",
        )