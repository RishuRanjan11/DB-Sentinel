from threading import Lock
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL

from app.database.connection import DatabaseConnection
from app.database.adapters.base import DatabaseAdapter
from app.core.trace import logger


class PostgreSQLAdapter(DatabaseAdapter):

    def __init__(self, connection: DatabaseConnection):
        if connection.database_type.lower() != "postgresql":
            raise ValueError(
                "PostgreSQLAdapter requires "
                "database_type='postgresql'."
            )

        self.connection = connection
        self.engine: Engine | None = None

        # Metadata cache.
        #
        # Instead of executing 3 metadata queries for every table,
        # PostgreSQL metadata is loaded in bulk once and reused.
        self._schema_cache: dict[tuple[str, str], dict] = {}
        self._schema_cache_loaded = False
        self._schema_cache_lock = Lock()

    def connect(self) -> None:
        if self.engine is not None:
            return

        if not self.connection.host:
            raise ValueError("Database host is required.")

        if not self.connection.database_name:
            raise ValueError("Database name is required.")

        if not self.connection.username:
            raise ValueError("Database username is required.")

        database_url = URL.create(
            drivername="postgresql+psycopg2",
            username=self.connection.username,
            password=self.connection.password,
            host=self.connection.host,
            port=self.connection.port or 5432,
            database=self.connection.database_name,
        )

        connect_args = {
            "connect_timeout": 10,
        }

        if self.connection.ssl_enabled:
            connect_args["sslmode"] = "require"

        self.engine = create_engine(
            database_url,
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

        except Exception as error:
            logger.warning("Database connection test failed: %s", error)
            return False

    def execute(
        self,
        query: Any,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict]:

        self.connect()

        with self.engine.connect() as connection:
            result = connection.execute(
                text(query),
                parameters or {},
            )

            return [
                dict(row)
                for row in result.mappings().all()
            ]

    def get_schemas(self) -> list[str]:
        self.connect()

        query = text("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN (
                'information_schema',
                'pg_catalog'
            )
            ORDER BY schema_name
        """)

        with self.engine.connect() as connection:
            result = connection.execute(query)

            return [
                row[0]
                for row in result
            ]

    def get_tables(
        self,
        schema: str | None = None,
    ) -> list[str]:

        self.connect()

        if schema:
            query = text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)

            parameters = {
                "schema": schema,
            }

        else:
            query = text("""
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN (
                    'information_schema',
                    'pg_catalog'
                )
                AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name
            """)

            parameters = {}

        with self.engine.connect() as connection:
            result = connection.execute(
                query,
                parameters,
            )

            if schema:
                return [
                    row[0]
                    for row in result
                ]

            return [
                f"{row[0]}.{row[1]}"
                for row in result
            ]

    def _load_schema_cache(self) -> None:
        """
        Load PostgreSQL schema metadata in bulk.

        Previously:
            N tables × 3 metadata queries

        Now:
            3 metadata queries total.

        This is especially important for remote databases such as
        FlyBase where repeatedly querying pg_catalog can cause the
        server to terminate the connection.
        """

        if self._schema_cache_loaded:
            return

        with self._schema_cache_lock:

            if self._schema_cache_loaded:
                return

            self.connect()

            columns_query = text("""
                SELECT
                    n.nspname AS schema_name,
                    c.relname AS table_name,
                    a.attname AS column_name,
                    pg_catalog.format_type(
                        a.atttypid,
                        a.atttypmod
                    ) AS data_type,
                    NOT a.attnotnull AS nullable
                FROM pg_catalog.pg_attribute a
                JOIN pg_catalog.pg_class c
                    ON c.oid = a.attrelid
                JOIN pg_catalog.pg_namespace n
                    ON n.oid = c.relnamespace
                WHERE a.attnum > 0
                  AND NOT a.attisdropped
                  AND c.relkind IN ('r', 'p')
                  AND n.nspname NOT IN (
                      'information_schema',
                      'pg_catalog'
                  )
                ORDER BY
                    n.nspname,
                    c.relname,
                    a.attnum
            """)

            primary_key_query = text("""
                SELECT
                    n.nspname AS schema_name,
                    c.relname AS table_name,
                    a.attname AS column_name,
                    array_position(
                        i.indkey,
                        a.attnum
                    ) AS column_position
                FROM pg_catalog.pg_index i
                JOIN pg_catalog.pg_class c
                    ON c.oid = i.indrelid
                JOIN pg_catalog.pg_namespace n
                    ON n.oid = c.relnamespace
                JOIN pg_catalog.pg_attribute a
                    ON a.attrelid = i.indrelid
                   AND a.attnum = ANY(i.indkey)
                WHERE i.indisprimary
                  AND n.nspname NOT IN (
                      'information_schema',
                      'pg_catalog'
                  )
                ORDER BY
                    n.nspname,
                    c.relname,
                    column_position
            """)

            foreign_key_query = text("""
                SELECT
                    source_namespace.nspname
                        AS schema_name,

                    source_class.relname
                        AS table_name,

                    source_attr.attname
                        AS column_name,

                    target_namespace.nspname
                        AS referred_schema,

                    target_class.relname
                        AS referred_table,

                    target_attr.attname
                        AS referred_column,

                    source_key.ordinality
                        AS column_position

                FROM pg_catalog.pg_constraint constraint_info

                JOIN pg_catalog.pg_class source_class
                    ON source_class.oid =
                       constraint_info.conrelid

                JOIN pg_catalog.pg_namespace source_namespace
                    ON source_namespace.oid =
                       source_class.relnamespace

                JOIN pg_catalog.pg_class target_class
                    ON target_class.oid =
                       constraint_info.confrelid

                JOIN pg_catalog.pg_namespace target_namespace
                    ON target_namespace.oid =
                       target_class.relnamespace

                CROSS JOIN LATERAL unnest(
                    constraint_info.conkey
                ) WITH ORDINALITY AS source_key(
                    attnum,
                    ordinality
                )

                CROSS JOIN LATERAL unnest(
                    constraint_info.confkey
                ) WITH ORDINALITY AS target_key(
                    attnum,
                    ordinality
                )

                JOIN pg_catalog.pg_attribute source_attr
                    ON source_attr.attrelid =
                       source_class.oid
                   AND source_attr.attnum =
                       source_key.attnum

                JOIN pg_catalog.pg_attribute target_attr
                    ON target_attr.attrelid =
                       target_class.oid
                   AND target_attr.attnum =
                       target_key.attnum
                   AND target_key.ordinality =
                       source_key.ordinality

                WHERE constraint_info.contype = 'f'
                  AND source_namespace.nspname NOT IN (
                      'information_schema',
                      'pg_catalog'
                  )

                ORDER BY
                    source_namespace.nspname,
                    source_class.relname,
                    source_key.ordinality
            """)

            with self.engine.connect() as connection:

                columns_result = connection.execute(
                    columns_query
                ).mappings().all()

                primary_key_result = connection.execute(
                    primary_key_query
                ).mappings().all()

                foreign_key_result = connection.execute(
                    foreign_key_query
                ).mappings().all()

            cache: dict[tuple[str, str], dict] = {}

            # ---------------------------------------------------------
            # Columns
            # ---------------------------------------------------------

            for row in columns_result:

                key = (
                    row["schema_name"],
                    row["table_name"],
                )

                if key not in cache:
                    cache[key] = {
                        "schema": row["schema_name"],
                        "table": row["table_name"],
                        "columns": [],
                        "primary_key": [],
                        "foreign_keys": [],
                    }

                cache[key]["columns"].append(
                    {
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["nullable"],
                    }
                )

            # ---------------------------------------------------------
            # Primary keys
            # ---------------------------------------------------------

            for row in primary_key_result:

                key = (
                    row["schema_name"],
                    row["table_name"],
                )

                if key not in cache:
                    cache[key] = {
                        "schema": row["schema_name"],
                        "table": row["table_name"],
                        "columns": [],
                        "primary_key": [],
                        "foreign_keys": [],
                    }

                cache[key]["primary_key"].append(
                    row["column_name"]
                )

            # ---------------------------------------------------------
            # Foreign keys
            # ---------------------------------------------------------

            for row in foreign_key_result:

                key = (
                    row["schema_name"],
                    row["table_name"],
                )

                if key not in cache:
                    cache[key] = {
                        "schema": row["schema_name"],
                        "table": row["table_name"],
                        "columns": [],
                        "primary_key": [],
                        "foreign_keys": [],
                    }

                cache[key]["foreign_keys"].append(
                    {
                        "column": row["column_name"],
                        "referred_schema": row["referred_schema"],
                        "referred_table": row["referred_table"],
                        "referred_column": row["referred_column"],
                    }
                )

            self._schema_cache = cache
            self._schema_cache_loaded = True

    def get_schema(
        self,
        schema: str,
        table: str,
    ) -> dict:

        self._load_schema_cache()

        key = (schema, table)

        details = self._schema_cache.get(key)

        if details is None:
            raise ValueError(
                f"Table '{schema}.{table}' was not found."
            )

        # Return a copy so callers cannot mutate the cache.
        return {
            "schema": details["schema"],
            "table": details["table"],
            "columns": [
                dict(column)
                for column in details["columns"]
            ],
            "primary_key": list(
                details["primary_key"]
            ),
            "foreign_keys": [
                dict(foreign_key)
                for foreign_key in details["foreign_keys"]
            ],
        }

    def refresh_schema_cache(self) -> None:
        """
        Explicitly refresh metadata.

        Useful later for SaaS environments where customer schemas
        can change while DB-Sentinel is running.
        """

        with self._schema_cache_lock:
            self._schema_cache.clear()
            self._schema_cache_loaded = False

        self._load_schema_cache()

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

        with self._schema_cache_lock:
            self._schema_cache.clear()
            self._schema_cache_loaded = False

    def execute_read_only(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        statement_timeout_ms: int = 10_000,
        max_rows: int = 100,
    ) -> list[dict]:

        if statement_timeout_ms <= 0:
            raise ValueError(
                "statement_timeout_ms must be greater than zero."
            )

        if max_rows <= 0:
            raise ValueError(
                "max_rows must be greater than zero."
            )

        if max_rows > 100:
            raise ValueError(
                "max_rows cannot exceed 100."
            )

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if self.engine is None:
            self.connect()

        with self.engine.connect() as connection:

            transaction = connection.begin()

            try:

                connection.execute(
                    text("SET TRANSACTION READ ONLY")
                )

                connection.execute(
                    text(
                        "SET LOCAL statement_timeout = "
                        ":timeout_ms"
                    ),
                    {
                        "timeout_ms": statement_timeout_ms
                    },
                )

                result = connection.execute(
                    text(query),
                    parameters or {},
                )

                rows = [
                    dict(row._mapping)
                    for row in result.fetchmany(
                        max_rows + 1
                    )
                ]

                if len(rows) > max_rows:
                    raise RuntimeError(
                        f"Query returned more than the maximum "
                        f"allowed {max_rows} rows."
                    )

                transaction.commit()

                return rows

            except Exception:
                transaction.rollback()
                raise