from __future__ import annotations

from typing import Any

import sqlglot
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from app.database.adapters.base import DatabaseAdapter
from app.database.connection import DatabaseConnection
from app.core.trace import logger


class SQLAlchemyRelationalAdapter(DatabaseAdapter):
    DIALECTS = {
        "mysql": ("mysql+pymysql", "mysql"),
        "mssql": ("mssql+pyodbc", "tsql"),
        "sqlserver": ("mssql+pyodbc", "tsql"),
        "sql server": ("mssql+pyodbc", "tsql"),
        "microsoft sql server": ("mssql+pyodbc", "tsql"),
        "sql": ("mssql+pyodbc", "tsql"),
    }
    SYSTEM_SCHEMAS = {
        "information_schema",
        "sys",
    }

    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
        try:
            self.drivername, self.sqlglot_dialect = self.DIALECTS[
                connection.database_type.lower()
            ]
        except KeyError as error:
            raise ValueError(
                f"Unsupported relational database type: "
                f"{connection.database_type}"
            ) from error
        self.engine: Engine | None = None

    def connect(self) -> None:
        if self.engine is not None:
            return
        if not self.connection.host or not self.connection.database_name:
            raise ValueError("Database host and database name are required.")
        if not self.connection.username:
            raise ValueError("Database username is required.")

        options = self.connection.options or {}
        normalized_options = {
            str(key).lower(): value
            for key, value in options.items()
        }
        query: dict[str, str] = {
            str(key): str(value)
            for key, value in options.items()
            if str(key).lower()
            not in {
                "encrypt",
                "trustservercertificate",
                "trust_server_certificate",
            }
        }
        connect_args: dict[str, Any] = {"connect_timeout": 10}

        if self.drivername == "mysql+pymysql":
            if self.connection.ssl_enabled:
                connect_args["ssl"] = {"check_hostname": True}
        else:
            query.setdefault(
                "driver",
                str(
                    normalized_options.get(
                        "driver",
                        "ODBC Driver 18 for SQL Server",
                    )
                ),
            )
            query["Encrypt"] = "yes" if self.connection.ssl_enabled else "no"
            query["TrustServerCertificate"] = str(
                normalized_options.get(
                    "trustservercertificate",
                    normalized_options.get(
                        "trust_server_certificate",
                        "no",
                    ),
                )
            )
            connect_args.pop("connect_timeout", None)
            connect_args["timeout"] = 10

        url = URL.create(
            drivername=self.drivername,
            username=self.connection.username,
            password=self.connection.password,
            host=self.connection.host,
            port=self.connection.port or (
                3306 if self.drivername == "mysql+pymysql" else 1433
            ),
            database=self.connection.database_name,
            query=query,
        )
        self.engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_timeout=10,
            pool_size=2,
            max_overflow=0,
            connect_args=connect_args,
        )

    def test_connection(self) -> bool:
        self.connect()
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as error:
            logger.warning(
                "Database connection test failed for type=%s: %s",
                self.connection.database_type,
                error,
            )
            return False

    def execute(
        self,
        query: Any,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict]:
        return self.execute_read_only(query, parameters)

    def execute_read_only(
        self,
        query: Any,
        parameters: dict[str, Any] | None = None,
        statement_timeout_ms: int = 10_000,
        max_rows: int = 100,
    ) -> list[dict]:
        query_text = str(query)
        self._require_select(query_text)
        if statement_timeout_ms <= 0:
            raise ValueError("statement_timeout_ms must be greater than zero.")
        if max_rows <= 0 or max_rows > 100:
            raise ValueError("max_rows must be between 1 and 100.")
        self.connect()
        with self.engine.connect() as connection:
            if self.drivername == "mysql+pymysql":
                connection.execute(
                    text("SET SESSION TRANSACTION READ ONLY")
                )
                connection.commit()
                connection.execute(
                    text(
                        "SET SESSION max_execution_time = "
                        ":timeout_ms"
                    ),
                    {"timeout_ms": statement_timeout_ms},
                )
                connection.commit()
            else:
                driver_connection = (
                    connection.connection.driver_connection
                )
                driver_connection.timeout = max(
                    1,
                    (statement_timeout_ms + 999) // 1000,
                )
            transaction = connection.begin()
            try:
                result = connection.execute(
                    text(query_text),
                    parameters or {},
                )
                rows = [
                    dict(row)
                    for row in result.mappings().fetchmany(max_rows + 1)
                ]
                if len(rows) > max_rows:
                    raise RuntimeError(
                        f"Query returned more than the maximum allowed "
                        f"{max_rows} rows."
                    )
                transaction.commit()
                return rows
            except Exception:
                transaction.rollback()
                raise

    def get_schemas(self) -> list[str]:
        self.connect()
        if self.drivername == "mysql+pymysql":
            return [self.connection.database_name]
        schemas = inspect(self.engine).get_schema_names()
        return sorted(
            schema
            for schema in schemas
            if schema.lower() not in self.SYSTEM_SCHEMAS
        )

    def get_tables(self, schema: str | None = None) -> list[str]:
        self.connect()
        if self.drivername == "mysql+pymysql":
            schema = self.connection.database_name
        inspector = inspect(self.engine)
        if schema:
            return inspector.get_table_names(schema=schema)
        return [
            f"{schema_name}.{table}"
            for schema_name in self.get_schemas()
            for table in inspector.get_table_names(schema=schema_name)
        ]

    def get_schema(self, schema: str, table: str) -> dict:
        self.connect()
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table, schema=schema)
        primary_key = inspector.get_pk_constraint(
            table,
            schema=schema,
        ).get("constrained_columns") or []
        foreign_keys = [
            {
                "column": source_column,
                "referred_schema": key.get("referred_schema") or schema,
                "referred_table": key.get("referred_table"),
                "referred_column": target_column,
            }
            for key in inspector.get_foreign_keys(table, schema=schema)
            for source_column, target_column in zip(
                key.get("constrained_columns") or [],
                key.get("referred_columns") or [],
            )
        ]
        return {
            "schema": schema,
            "table": table,
            "columns": [
                {
                    "name": column["name"],
                    "data_type": str(column["type"]),
                    "nullable": bool(column.get("nullable", True)),
                }
                for column in columns
            ],
            "primary_key": primary_key,
            "foreign_keys": foreign_keys,
        }

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

    def _require_select(self, query: str) -> None:
        try:
            statements = sqlglot.parse(
                query,
                read=self.sqlglot_dialect,
            )
        except sqlglot.errors.ParseError as error:
            raise ValueError(f"SQL parsing failed: {error}") from error
        if (
            len(statements) != 1
            or not isinstance(statements[0], sqlglot.exp.Select)
        ):
            raise ValueError("Only one read-only SELECT statement is allowed.")
