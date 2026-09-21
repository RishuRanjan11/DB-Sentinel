from app.database.connection import DatabaseConnection
from app.database.manager import DatabaseManager
from app.security.sql_validator import SQLSecurityValidator


def main():

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

    manager = DatabaseManager()
    manager.register(connection)

    adapter = manager.get("dev-flybase")

    # ---------------------------------------------------------
    # Use only tables that the investigation would have
    # legitimately retrieved.
    # ---------------------------------------------------------

    allowed_tables = {
        (
            "dataclass",
            "genev2",
        ),
        (
            "dataclass_relationship",
            "gene_human_orthologv2",
        ),
    }

    validator = SQLSecurityValidator(
        allowed_tables=allowed_tables
    )

    tests = [
        (
            "Valid SELECT",
            """
            SELECT id
            FROM dataclass.genev2
            LIMIT 10
            """,
            True,
        ),
        (
            "Missing LIMIT",
            """
            SELECT id
            FROM dataclass.genev2
            """,
            False,
        ),
        (
            "LIMIT above maximum",
            """
            SELECT id
            FROM dataclass.genev2
            LIMIT 1000
            """,
            False,
        ),
        (
            "UPDATE",
            """
            UPDATE dataclass.genev2
            SET name = 'x'
            LIMIT 1
            """,
            False,
        ),
        (
            "DELETE",
            """
            DELETE FROM dataclass.genev2
            WHERE id = 'FBgn0000008'
            """,
            False,
        ),
        (
            "DROP",
            """
            DROP TABLE dataclass.genev2
            """,
            False,
        ),
        (
            "INSERT",
            """
            INSERT INTO dataclass.genev2
            VALUES ('x')
            """,
            False,
        ),
        (
            "ALTER",
            """
            ALTER TABLE dataclass.genev2
            ADD COLUMN malicious text
            """,
            False,
        ),
        (
            "Multiple statements",
            """
            SELECT id
            FROM dataclass.genev2
            LIMIT 10;

            SELECT symbol
            FROM dataclass.genev2
            LIMIT 10
            """,
            False,
        ),
        (
            "Unqualified table",
            """
            SELECT id
            FROM genev2
            LIMIT 10
            """,
            False,
        ),
        (
            "Unauthorized table",
            """
            SELECT humanhealth_id
            FROM public.humanhealth
            LIMIT 10
            """,
            False,
        ),
        (
            "Invented table",
            """
            SELECT id
            FROM dataclass.fake_gene_table
            LIMIT 10
            """,
            False,
        ),
        (
            "LIMIT zero",
            """
            SELECT id
            FROM dataclass.genev2
            LIMIT 0
            """,
            False,
        ),
        (
            "LIMIT negative",
            """
            SELECT id
            FROM dataclass.genev2
            LIMIT -1
            """,
            False,
        ),
    ]

    passed = 0
    failed = 0

    print()
    print("=" * 70)
    print("DB-Sentinel Security Validation Tests")
    print("=" * 70)

    for name, sql, expected in tests:

        result = validator.validate(sql)

        success = (
            result.allowed == expected
        )

        if success:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print()
        print(f"[{status}] {name}")
        print(f"  Expected allowed : {expected}")
        print(f"  Actual allowed   : {result.allowed}")
        print(f"  Reason           : {result.reason}")

    print()
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 70)

    manager.close_all()

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()