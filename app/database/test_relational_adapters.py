from app.database.adapters.sqlalchemy_relational import (
    SQLAlchemyRelationalAdapter,
)
from app.database.connection import DatabaseConnection
from app.database.manager import DatabaseManager
from app.security.sql_validator import SQLSecurityValidator


def connection(database_type):
    return DatabaseConnection(
        connection_id=database_type,
        organization_id="test",
        workspace_id="test",
        name=database_type,
        database_type=database_type,
        host="localhost",
        port=1433,
        database_name="sample",
        username="reader",
        password="not-used-in-test",
    )


def test_relational_adapter_selects_database_dialects():
    mysql = SQLAlchemyRelationalAdapter(connection("mysql"))
    sqlserver = SQLAlchemyRelationalAdapter(connection("sqlserver"))

    assert mysql.drivername == "mysql+pymysql"
    assert mysql.sqlglot_dialect == "mysql"
    assert sqlserver.drivername == "mssql+pyodbc"
    assert sqlserver.sqlglot_dialect == "tsql"


def test_manager_registers_postgresql_mysql_sqlserver_and_mongodb():
    manager = DatabaseManager()

    assert set(manager.adapter_registry.supported_types()) >= {
        "postgresql",
        "mysql",
        "sql",
        "mssql",
        "sqlserver",
        "mongodb",
    }


def test_sql_validator_accepts_mysql_and_sql_server_select_syntax():
    mysql = SQLSecurityValidator(
        allowed_tables={("sample", "people")},
        schema_context=[{
            "schema": "sample",
            "table": "people",
            "columns": [{"name": "id"}, {"name": "name"}],
        }],
        dialect="mysql",
    )
    sqlserver = SQLSecurityValidator(
        allowed_tables={("dbo", "people")},
        schema_context=[{
            "schema": "dbo",
            "table": "people",
            "columns": [{"name": "id"}, {"name": "name"}],
        }],
        dialect="tsql",
    )

    assert mysql.validate(
        "SELECT id, name FROM sample.people ORDER BY id LIMIT 10"
    ).allowed
    assert sqlserver.validate(
        "SELECT TOP 10 id, name FROM dbo.people ORDER BY id"
    ).allowed
    assert not sqlserver.validate(
        "SELECT id INTO dbo.copy_people FROM dbo.people"
    ).allowed


def test_mysql_read_only_execution_commits_session_settings_before_query():
    class Transaction:
        def __init__(self, events):
            self.events = events

        def commit(self):
            self.events.append("transaction_commit")

        def rollback(self):
            self.events.append("transaction_rollback")

    class Result:
        def mappings(self):
            return self

        def fetchmany(self, size):
            assert size == 101
            return [{"id": 1}]

    class Connection:
        def __init__(self):
            self.events = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement, parameters=None):
            statement_text = str(statement)
            self.events.append(statement_text)
            if statement_text.lstrip().upper().startswith("SELECT"):
                return Result()
            return None

        def commit(self):
            self.events.append("session_commit")

        def begin(self):
            assert self.events[-1] == "session_commit"
            self.events.append("begin")
            return Transaction(self.events)

    class Engine:
        def __init__(self):
            self.connection = Connection()

        def connect(self):
            return self.connection

    adapter = SQLAlchemyRelationalAdapter(connection("mysql"))
    engine = Engine()
    adapter.engine = engine

    rows = adapter.execute_read_only("SELECT id FROM sample.people")

    assert rows == [{"id": 1}]
    assert engine.connection.events[-1] == "transaction_commit"
    assert engine.connection.events.count("session_commit") == 2
