from app.agent.sql_pipeline import SQLPipeline
from app.database.connection import DatabaseConnection
from app.database.manager import DatabaseManager


SEPARATOR = "=" * 70
SUB_SEPARATOR = "-" * 70


def print_section(title: str):
    print()
    print(f"      {title}")
    print(f"      {SUB_SEPARATOR}")


def print_table(headers: list[str], rows: list[list[str]]):
    widths = []

    for i, header in enumerate(headers):
        values = [str(row[i]) for row in rows]
        widths.append(
            max(
                len(str(header)),
                *(len(value) for value in values),
            )
        )

    header_line = "  ".join(
        str(header).ljust(widths[i])
        for i, header in enumerate(headers)
    )

    separator = "  ".join(
        "-" * widths[i]
        for i in range(len(headers))
    )

    print(f"      {header_line}")
    print(f"      {separator}")

    for row in rows:
        print(
            "      "
            + "  ".join(
                str(value).ljust(widths[i])
                for i, value in enumerate(row)
            )
        )


def main():

    print()
    print(SEPARATOR)
    print("DB-Sentinel — SQL Investigation")
    print(SEPARATOR)

    # =========================================================
    # 1. Database Connection
    # =========================================================

    print()
    print("[1/7] Creating database connection...")

    connection = DatabaseConnection(
        connection_id="dev-flybase",
        organization_id="development-org",
        workspace_id="development-workspace",
        name="FlyBase Development",
        database_type="postgresql",
        host="chado.flybase.org",
        port=5432,
        database_name="flybase",
        username="flybase",
        password=None,
        ssl_enabled=False,
    )

    print("      ✓ PostgreSQL connection configured")

    print_table(
        ["Property", "Value"],
        [
            ["Database", connection.name],
            ["Type", connection.database_type],
            ["Host", connection.host],
            ["Port", connection.port],
            ["Database Name", connection.database_name],
            ["SSL", connection.ssl_enabled],
        ],
    )

    # =========================================================
    # 2. Database Manager
    # =========================================================

    print()
    print("[2/7] Creating database manager...")

    manager = DatabaseManager()

    print("      ✓ Database manager initialized")

    # =========================================================
    # 3. Register Adapter
    # =========================================================

    print()
    print("[3/7] Registering database adapter...")

    manager.register(connection)

    print("      ✓ PostgreSQL adapter registered")

    # =========================================================
    # 4. SQL Pipeline
    # =========================================================

    print()
    print("[4/7] Creating SQL pipeline...")

    pipeline = SQLPipeline(
        database_manager=manager
    )

    print("      ✓ SQL pipeline initialized")

    # =========================================================
    # 5. Generate / Validate / Execute
    # =========================================================

    print()
    print("[5/7] Generating and validating SQL...")

    question = (
        "Which FlyBase genes have human orthologs?"
    )

    result = pipeline.generate_validate_and_execute(
        connection_id="dev-flybase",
        question=question,
        required_information=[
            "FlyBase gene",
            "human ortholog",
        ],
    )

    # ---------------------------------------------------------
    # Question
    # ---------------------------------------------------------

    print_section("Question")

    print(f"      {result['question']}")

    # ---------------------------------------------------------
    # Complexity
    # ---------------------------------------------------------

    print_section("Query Complexity")

    complexity = result.get(
        "complexity",
        result.get("required_complexity", "fast"),
    )

    print(f"      {complexity}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print_section("Model")

    model = result.get(
        "model",
        result.get("model_used", "Not reported"),
    )

    print(f"      {model}")

    # ---------------------------------------------------------
    # Generated SQL
    # ---------------------------------------------------------

    print_section("Generated SQL")

    sql = result.get("sql", "")

    for line in sql.strip().splitlines():
        print(f"      {line}")

    # ---------------------------------------------------------
    # Tables Used
    # ---------------------------------------------------------

    print_section("Tables Used")

    tables_used = result.get("tables_used", [])

    if tables_used:
        print_table(
            ["#", "Schema", "Table"],
            [
                [
                    index,
                    table.split(".", 1)[0]
                    if "." in table
                    else "-",
                    table.split(".", 1)[1]
                    if "." in table
                    else table,
                ]
                for index, table in enumerate(
                    tables_used,
                    start=1,
                )
            ],
        )
    else:
        print("      None")

    # ---------------------------------------------------------
    # Columns Used
    # ---------------------------------------------------------

    print_section("Columns Used")

    columns_used = result.get("columns_used", [])

    if columns_used:
        print_table(
            ["#", "Schema", "Table", "Column"],
            [
                [
                    index,
                    parts[0] if len(parts) == 3 else "-",
                    parts[1] if len(parts) == 3 else "-",
                    parts[2] if len(parts) == 3 else column,
                ]
                for index, column in enumerate(
                    columns_used,
                    start=1,
                )
                for parts in [column.split(".", 2)]
            ],
        )
    else:
        print("      None")

    # ---------------------------------------------------------
    # Security
    # ---------------------------------------------------------

    print_section("Security Validation")

    allowed = result.get("allowed", False)
    reason = result.get("reason", "")

    print(
        f"      {'✓ ALLOWED' if allowed else '✗ REJECTED'}"
    )
    print(f"      Reason: {reason}")

    # ---------------------------------------------------------
    # Execution
    # ---------------------------------------------------------

    print_section("Execution")

    executed = result.get("executed", False)
    row_count = result.get("row_count", 0)

    print(
        f"      {'✓ SUCCESS' if executed else '✗ FAILED'}"
    )
    print(f"      Rows returned: {row_count}")

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print_section("Results")

    rows = result.get("rows", [])

    if rows:

        columns = list(rows[0].keys())

        display_rows = []

        for row in rows[:10]:
            display_rows.append(
                [row.get(column, "") for column in columns]
            )

        print_table(
            columns,
            display_rows,
        )

        print()
        print(
            f"      Showing {min(10, len(rows))} "
            f"of {len(rows)} rows"
        )

    else:
        print("      No rows returned.")

    # ---------------------------------------------------------
    # Schema Context
    # ---------------------------------------------------------

    print_section("Schema Context")

    schema_context = result.get(
        "schema_context",
        [],
    )

    if schema_context:

        for table in schema_context:

            schema = table.get(
                "schema",
                "-",
            )

            table_name = table.get(
                "table",
                "-",
            )

            print()
            print(
                f"      {schema}.{table_name}"
            )

            columns = table.get(
                "columns",
                [],
            )

            if columns:

                column_rows = [
                    [
                        column.get("name", "-"),
                        column.get("type", "-"),
                        column.get("nullable", "-"),
                    ]
                    for column in columns
                ]

                print_table(
                    [
                        "Column",
                        "Type",
                        "Nullable",
                    ],
                    column_rows,
                )

            primary_key = table.get(
                "primary_key",
                [],
            )

            if primary_key:
                print(
                    f"      Primary Key: "
                    f"{', '.join(primary_key)}"
                )

            foreign_keys = table.get(
                "foreign_keys",
                [],
            )

            if foreign_keys:

                print("      Foreign Keys:")

                fk_rows = [
                    [
                        fk.get("column", "-"),
                        (
                            f"{fk.get('referred_schema', '-')}"
                            f"."
                            f"{fk.get('referred_table', '-')}"
                        ),
                        fk.get(
                            "referred_column",
                            "-",
                        ),
                    ]
                    for fk in foreign_keys
                ]

                print_table(
                    [
                        "Column",
                        "References",
                        "Referenced Column",
                    ],
                    fk_rows,
                )

    else:
        print("      No schema context returned.")

    # ---------------------------------------------------------
    # Semantic Verification
    # ---------------------------------------------------------

    print_section("Semantic Verification")

    semantic = result.get(
        "semantic_verification",
        {},
    )

    semantic_verified = semantic.get(
        "verified",
        False,
    )

    print(
        f"      "
        f"{'✓ VERIFIED' if semantic_verified else '✗ NOT VERIFIED'}"
    )

    checks = semantic.get(
        "checks",
        [],
    )

    if checks:

        for check in checks:
            print(f"      ✓ {check}")

    warnings = semantic.get(
        "warnings",
        [],
    )

    if warnings:

        for warning in warnings:
            print(f"      ! {warning}")

    errors = semantic.get(
        "errors",
        [],
    )

    if errors:

        for error in errors:
            print(f"      ✗ {error}")

    # =========================================================
    # 6. Close Connections
    # =========================================================

    print()
    print("[6/7] Closing connections...")

    manager.close_all()

    print("      ✓ Connections closed")

    # =========================================================
    # 7. Done
    # =========================================================

    print()
    print("[7/7] Done")

    print()
    print(SEPARATOR)
    print("Investigation completed successfully.")
    print(SEPARATOR)
    print()


if __name__ == "__main__":
    main()