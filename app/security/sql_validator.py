import sqlglot
from sqlglot import expressions


class SQLValidationResult:
    def __init__(
        self,
        allowed: bool,
        reason: str,
    ):
        self.allowed = allowed
        self.reason = reason

    def __repr__(self):
        return (
            f"SQLValidationResult("
            f"allowed={self.allowed}, "
            f"reason='{self.reason}')"
        )


class SQLSecurityValidator:

    # Initial DB-Sentinel policy.
    # We are running against a public read-only database.
    ALLOWED_SCHEMAS = {
        "gene",
        "dataclass",
        "dataclass_relationship",
        "gene_group",
        "humanhealth",
    }

    MAX_LIMIT = 100

    FORBIDDEN_KEYWORDS = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
    }

    def validate(self, sql: str) -> SQLValidationResult:

        if not sql or not sql.strip():
            return SQLValidationResult(
                False,
                "SQL query is empty."
            )

        # --------------------------------------------------
        # 1. Parse SQL
        # --------------------------------------------------

        try:
            statements = sqlglot.parse(
                sql,
                read="postgres"
            )
        except Exception as error:
            return SQLValidationResult(
                False,
                f"SQL parsing failed: {error}"
            )

        if len(statements) != 1:
            return SQLValidationResult(
                False,
                "Multiple SQL statements are not allowed."
            )

        statement = statements[0]

        # --------------------------------------------------
        # 2. Only SELECT is allowed
        # --------------------------------------------------

        if not isinstance(
            statement,
            expressions.Select
        ):
            return SQLValidationResult(
                False,
                "Only SELECT statements are allowed."
            )

        # --------------------------------------------------
        # 3. Check forbidden operations
        # --------------------------------------------------

        sql_upper = sql.upper()

        for keyword in self.FORBIDDEN_KEYWORDS:

            if keyword in sql_upper:
                return SQLValidationResult(
                    False,
                    f"Forbidden SQL operation detected: "
                    f"{keyword}"
                )

        # --------------------------------------------------
        # 4. Check referenced tables
        # --------------------------------------------------

        tables = statement.find_all(
            expressions.Table
        )

        for table in tables:

            schema = table.db

            # If no schema is specified, reject it.
            # We want explicit schema-qualified queries.
            if not schema:
                return SQLValidationResult(
                    False,
                    f"Table '{table.name}' "
                    f"is not schema-qualified."
                )

            if schema not in self.ALLOWED_SCHEMAS:
                return SQLValidationResult(
                    False,
                    f"Schema '{schema}' "
                    f"is not allowed."
                )

        # --------------------------------------------------
        # 5. LIMIT must exist
        # --------------------------------------------------

        limit = statement.args.get("limit")

        if limit is None:
            return SQLValidationResult(
                False,
                "LIMIT clause is required."
            )

        # --------------------------------------------------
        # 6. LIMIT must be numeric
        # --------------------------------------------------

        limit_expression = limit.expression

        try:
            limit_value = int(
                limit_expression.this
            )
        except (ValueError, TypeError, AttributeError):
            return SQLValidationResult(
                False,
                "LIMIT must be a numeric value."
            )

        # --------------------------------------------------
        # 7. LIMIT must not exceed maximum
        # --------------------------------------------------

        if limit_value > self.MAX_LIMIT:

            return SQLValidationResult(
                False,
                f"LIMIT {limit_value} exceeds "
                f"maximum allowed LIMIT "
                f"of {self.MAX_LIMIT}."
            )

        # --------------------------------------------------
        # 8. Query passed all checks
        # --------------------------------------------------

        return SQLValidationResult(
            True,
            "SQL passed security validation."
        )


if __name__ == "__main__":

    validator = SQLSecurityValidator()

    safe_sql = """
    SELECT
        g.id AS gene_id,
        g.symbol AS gene_symbol,
        o.symbol AS ortholog_symbol
    FROM dataclass.genev2 AS g
    JOIN dataclass_relationship.gene_human_orthologv2 AS r
        ON g.id = r.gene_id
    JOIN dataclass.orthologv2 AS o
        ON o.id = r.ortholog_id
    LIMIT 100;
    """

    dangerous_sql = """
    SELECT *
    FROM dataclass.genev2
    LIMIT 10000;
    """

    result = validator.validate(dangerous_sql)

    print(result)