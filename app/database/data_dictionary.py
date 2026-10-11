from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.database.adapters.base import DatabaseAdapter


MAX_DICTIONARY_BYTES = 2 * 1024 * 1024
MAX_DESCRIPTION_LENGTH = 2_000


@dataclass(frozen=True)
class DatabaseDictionary:
    description: str
    schema_descriptions: dict[str, str]
    table_descriptions: dict[tuple[str, str], str]
    column_descriptions: dict[tuple[str, str, str], str]

    @classmethod
    def load(
        cls,
        path: str | Path,
        adapter: DatabaseAdapter,
    ) -> DatabaseDictionary:
        dictionary_path = Path(path).expanduser()
        if dictionary_path.stat().st_size > MAX_DICTIONARY_BYTES:
            raise ValueError("Data dictionary files must be 2 MiB or smaller.")

        try:
            payload = json.loads(dictionary_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read data dictionary JSON: {error}") from error

        if not isinstance(payload, dict):
            raise ValueError("Data dictionary root must be a JSON object.")
        if payload.get("version") != 1:
            raise ValueError("Data dictionary must set version to 1.")

        description = cls._description(payload.get("description", ""), "description")
        schemas = adapter.get_schemas()
        actual_tables = cls._database_tables(adapter, schemas)
        actual_schemas = {schema.lower() for schema in schemas}
        schema_descriptions: dict[str, str] = {}
        table_descriptions: dict[tuple[str, str], str] = {}
        column_descriptions: dict[tuple[str, str, str], str] = {}
        unknown: list[str] = []
        schemas = payload.get("schemas", [])
        if not isinstance(schemas, list):
            raise ValueError("'schemas' must be a JSON array.")

        for schema_record in schemas:
            if not isinstance(schema_record, dict):
                raise ValueError("Each schema entry must be a JSON object.")
            schema_name = cls._required_name(schema_record, "name", "schema")
            schema_key = schema_name.lower()
            if schema_key not in actual_schemas:
                unknown.append(schema_name)
                continue
            schema_description = cls._description(
                schema_record.get("description", ""),
                f"{schema_name} description",
            )
            if schema_description:
                schema_descriptions[schema_key] = schema_description
            tables = schema_record.get("tables", [])
            if not isinstance(tables, list):
                raise ValueError(f"Tables for schema '{schema_name}' must be a JSON array.")

            for table_record in tables:
                if not isinstance(table_record, dict):
                    raise ValueError("Each table entry must be a JSON object.")
                table_name = cls._required_name(table_record, "name", "table")
                table_key = (schema_key, table_name.lower())
                metadata = actual_tables.get(table_key)
                if metadata is None:
                    unknown.append(f"{schema_name}.{table_name}")
                    continue

                table_description = cls._description(
                    table_record.get("description", ""),
                    f"{schema_name}.{table_name} description",
                )
                if table_description:
                    table_descriptions[table_key] = table_description

                columns = table_record.get("columns", [])
                if not isinstance(columns, list):
                    raise ValueError(
                        f"Columns for '{schema_name}.{table_name}' must be a JSON array."
                    )
                actual_columns = {
                    str(column.get("name", column.get("column", ""))).lower()
                    if isinstance(column, dict)
                    else str(column).lower()
                    for column in metadata.get("columns", [])
                }
                for column_record in columns:
                    if not isinstance(column_record, dict):
                        raise ValueError(
                            f"Column entries for '{schema_name}.{table_name}' must be objects."
                        )
                    column_name = cls._required_name(
                        column_record,
                        "name",
                        f"column in {schema_name}.{table_name}",
                    )
                    if column_name.lower() not in actual_columns:
                        unknown.append(
                            f"{schema_name}.{table_name}.{column_name}"
                        )
                        continue
                    column_description = cls._description(
                        column_record.get("description", ""),
                        f"{schema_name}.{table_name}.{column_name} description",
                    )
                    if column_description:
                        column_descriptions[
                            (schema_key, table_name.lower(), column_name.lower())
                        ] = column_description

        if unknown:
            preview = ", ".join(unknown[:20])
            suffix = " ..." if len(unknown) > 20 else ""
            raise ValueError(
                "Data dictionary references unknown database objects: "
                f"{preview}{suffix}"
            )
        if (
            not description
            and not schema_descriptions
            and not table_descriptions
            and not column_descriptions
        ):
            raise ValueError("Data dictionary contains no descriptions.")

        return cls(
            description=description,
            schema_descriptions=schema_descriptions,
            table_descriptions=table_descriptions,
            column_descriptions=column_descriptions,
        )

    @staticmethod
    def _database_tables(
        adapter: DatabaseAdapter,
        schemas: list[str],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        result = {}
        for schema in schemas:
            for table in adapter.get_tables(schema=schema):
                details = adapter.get_schema(schema=schema, table=table)
                result[(schema.lower(), table.lower())] = details
        return result

    @staticmethod
    def _required_name(record: dict, field: str, label: str) -> str:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Every {label} must have a non-empty '{field}'.")
        return value.strip()

    @staticmethod
    def _description(value: Any, label: str) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a string.")
        description = value.strip()
        if len(description) > MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"{label} must be at most {MAX_DESCRIPTION_LENGTH} characters."
            )
        return description

    def enrich_schema_context(
        self,
        schema_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        enriched = []
        for table in schema_context:
            item = dict(table)
            schema = str(item.get("schema", "")).lower()
            table_name = str(item.get("table", "")).lower()
            schema_description = self.schema_descriptions.get(schema)
            if schema_description:
                item["dictionary_schema_description"] = schema_description
            table_description = self.table_descriptions.get(
                (schema, table_name)
            )
            if table_description:
                item["dictionary_description"] = table_description

            columns = []
            for column in item.get("columns", []):
                if not isinstance(column, dict):
                    columns.append(column)
                    continue
                column_item = dict(column)
                column_name = str(
                    column_item.get("name", column_item.get("column", ""))
                ).lower()
                column_description = self.column_descriptions.get(
                    (schema, table_name, column_name)
                )
                if column_description:
                    column_item["dictionary_description"] = column_description
                columns.append(column_item)
            item["columns"] = columns
            enriched.append(item)
        return enriched

    def description_for_prompt(self) -> str:
        table_keys = set(self.table_descriptions)
        table_keys.update(
            (schema, table)
            for schema, table, _ in self.column_descriptions
        )
        records = {
            "database_description": self.description,
            "schemas": self.schema_descriptions,
            "tables": [
                {
                    "schema": schema,
                    "table": table,
                    "description": self.table_descriptions.get(
                        (schema, table), ""
                    ),
                    "columns": {
                        column: column_description
                        for (column_schema, column_table, column), column_description
                        in self.column_descriptions.items()
                        if column_schema == schema and column_table == table
                    },
                }
                for schema, table in sorted(table_keys)
            ],
        }
        return json.dumps(records, ensure_ascii=True, sort_keys=True)

    @property
    def documented_table_count(self) -> int:
        return len(self.table_descriptions)

    @property
    def documented_column_count(self) -> int:
        return len(self.column_descriptions)
