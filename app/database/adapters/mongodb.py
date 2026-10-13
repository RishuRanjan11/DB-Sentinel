from __future__ import annotations

import json
from itertools import islice
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from bson import json_util
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from app.core.trace import logger
from app.database.adapters.base import DatabaseAdapter
from app.database.connection import DatabaseConnection
from app.security.mongodb_validator import MongoQuerySecurityValidator


class MongoDBAdapter(DatabaseAdapter):
    SYSTEM_COLLECTION_PREFIX = "system."

    def __init__(self, connection: DatabaseConnection):
        if connection.database_type.lower() != "mongodb":
            raise ValueError(
                "MongoDBAdapter requires database_type='mongodb'."
            )
        if not connection.database_name:
            raise ValueError("MongoDB database name is required.")
        self.connection = connection
        self.client: MongoClient | None = None
        self.database = None

    def connect(self) -> None:
        if self.client is not None:
            return
        options = self.connection.options or {}
        host = self.connection.host or ""
        if not host:
            raise ValueError("MongoDB host or URI is required.")
        timeout = int(options.get("server_selection_timeout_ms", 10_000))
        if host.startswith(("mongodb://", "mongodb+srv://")):
            kwargs: dict[str, Any] = {
                "serverSelectionTimeoutMS": timeout,
                "connectTimeoutMS": timeout,
            }
            self.client = MongoClient(
                host,
                **kwargs,
            )
        else:
            kwargs: dict[str, Any] = {
                "host": host,
                "port": self.connection.port or 27017,
                "tls": self.connection.ssl_enabled,
                "serverSelectionTimeoutMS": timeout,
                "connectTimeoutMS": timeout,
                "authSource": options.get(
                    "auth_source",
                    "admin",
                ),
            }
            if self.connection.username:
                kwargs["username"] = self.connection.username
            if self.connection.password:
                kwargs["password"] = self.connection.password
            self.client = MongoClient(**kwargs)
        self.database = self.client[self.connection.database_name]

    def test_connection(self) -> bool:
        self.connect()
        try:
            self.client.admin.command("ping")
            return True
        except PyMongoError as error:
            logger.warning(
                "MongoDB connection test failed: %s",
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
        if statement_timeout_ms <= 0:
            raise ValueError("statement_timeout_ms must be greater than zero.")
        if max_rows <= 0 or max_rows > 100:
            raise ValueError("max_rows must be between 1 and 100.")
        if isinstance(query, str):
            try:
                query = json_util.loads(query)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    "MongoDB query must be valid JSON or Extended JSON."
                ) from error
        elif isinstance(query, dict):
            query = json_util.loads(json.dumps(query))
        if not isinstance(query, dict):
            raise ValueError(
                "MongoDB query must contain a collection and pipeline."
            )
        collection_name = query.get("collection")
        pipeline = query.get("pipeline")
        if (
            not isinstance(collection_name, str)
            or not collection_name
            or not isinstance(pipeline, list)
            or not all(isinstance(stage, dict) for stage in pipeline)
        ):
            raise ValueError(
                "MongoDB query must contain a collection and pipeline."
            )
        if (
            not pipeline
            or not isinstance(pipeline[-1].get("$limit"), int)
            or isinstance(pipeline[-1].get("$limit"), bool)
            or not 1 <= pipeline[-1]["$limit"] <= max_rows
        ):
            raise ValueError(
                f"MongoDB pipeline must end with a $limit between 1 and {max_rows}."
            )
        self.connect()
        allowed_collections = set(self.get_tables(self.connection.database_name))
        if collection_name not in allowed_collections:
            raise ValueError(
                f"MongoDB collection '{collection_name}' does not exist "
                "in the connected database."
            )
        self._validate_read_only_stages(
            pipeline,
            allowed_collections,
        )
        collection = self.database[collection_name]
        cursor = collection.aggregate(
            pipeline,
            maxTimeMS=statement_timeout_ms,
            allowDiskUse=False,
        )
        rows = [
            json.loads(json_util.dumps(row))
            for row in islice(cursor, max_rows + 1)
        ]
        if len(rows) > max_rows:
            raise RuntimeError(
                f"Query returned more than the maximum allowed {max_rows} rows."
            )
        return rows

    @classmethod
    def _validate_read_only_stages(
        cls,
        pipeline: list[dict],
        allowed_collections: set[str],
    ) -> None:
        for stage in pipeline:
            if len(stage) != 1:
                raise ValueError(
                    "Every MongoDB aggregation stage must have one operator."
                )
            operator, specification = next(iter(stage.items()))
            if operator not in MongoQuerySecurityValidator.ALLOWED_STAGES:
                raise ValueError(
                    f"MongoDB aggregation stage '{operator}' is not permitted."
                )
            if operator == "$lookup":
                if not isinstance(specification, dict):
                    raise ValueError("$lookup must be an object.")
                foreign_collection = specification.get("from")
                if foreign_collection not in allowed_collections:
                    raise ValueError(
                        f"$lookup collection '{foreign_collection}' does "
                        "not exist in the connected database."
                    )
                if "db" in specification:
                    raise ValueError(
                        "Cross-database $lookup is not permitted."
                    )
            nested_pipeline = (
                specification.get("pipeline")
                if isinstance(specification, dict)
                else None
            )
            if isinstance(nested_pipeline, list):
                cls._validate_read_only_stages(
                    nested_pipeline,
                    allowed_collections,
                )

    def get_schemas(self) -> list[str]:
        if not self.connection.database_name:
            raise ValueError("MongoDB database name is required.")
        return [self.connection.database_name]

    def get_tables(self, schema: str | None = None) -> list[str]:
        self.connect()
        if schema and schema != self.connection.database_name:
            return []
        return sorted(
            name
            for name in self.database.list_collection_names()
            if not name.startswith(self.SYSTEM_COLLECTION_PREFIX)
        )

    def get_schema(self, schema: str, table: str) -> dict:
        self.connect()
        if schema != self.connection.database_name:
            raise ValueError(f"Unknown MongoDB database '{schema}'.")
        if table not in self.get_tables(schema):
            raise ValueError(f"Unknown MongoDB collection '{table}'.")
        collection = self.database[table]
        sampled_documents = collection.find(max_time_ms=10_000).limit(50)
        field_types: dict[str, set[str]] = {}
        for document in sampled_documents:
            self._collect_field_types(document, "", field_types, depth=0)
        columns = [
            {
                "name": name,
                "data_type": ", ".join(sorted(types)),
                "nullable": True,
            }
            for name, types in sorted(field_types.items())
        ]
        return {
            "schema": schema,
            "table": table,
            "columns": columns,
            "primary_key": ["_id"],
            "foreign_keys": [],
        }

    @classmethod
    def _collect_field_types(
        cls,
        value: Any,
        path: str,
        field_types: dict[str, set[str]],
        depth: int,
        record: bool = True,
    ) -> None:
        if depth > 6 or len(field_types) >= 1000:
            return

        if record and path:
            field_types.setdefault(path, set()).add(cls._bson_type(value))

        if isinstance(value, dict):
            for key, nested in value.items():
                nested_path = f"{path}.{key}" if path else str(key)
                cls._collect_field_types(
                    nested,
                    nested_path,
                    field_types,
                    depth + 1,
                )
        elif isinstance(value, list):
            for nested in value[:20]:
                if isinstance(nested, (dict, list)):
                    cls._collect_field_types(
                        nested,
                        path,
                        field_types,
                        depth + 1,
                        record=False,
                    )

    @staticmethod
    def _bson_type(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, datetime):
            return "date"
        if isinstance(value, date):
            return "date"
        if isinstance(value, int):
            return "int"
        if isinstance(value, (float, Decimal)):
            return "double" if isinstance(value, float) else "decimal"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        if value.__class__.__name__ == "Decimal128":
            return "decimal"
        if value.__class__.__name__ == "ObjectId":
            return "objectId"
        return value.__class__.__name__

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
            self.database = None
