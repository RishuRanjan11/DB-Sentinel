from app.database.adapters.base import DatabaseAdapter


def find_relevant_tables(
    adapter: DatabaseAdapter,
    keyword: str,
    schema_name: str | None = None,
) -> list[dict]:
    """Find tables/columns through the active database adapter."""
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return []

    schemas = [schema_name] if schema_name else adapter.get_schemas()
    matches: list[dict] = []

    for schema in schemas:
        for table in adapter.get_tables(schema=schema):
            details = adapter.get_schema(schema=schema, table=table)
            if keyword in table.lower():
                matches.append({"schema": schema, "table": table, "matched_by": "table_name"})
                continue

            columns = details.get("columns", [])
            matching_columns = []
            for column in columns:
                name = column.get("name", column.get("column", "")) if isinstance(column, dict) else str(column)
                if keyword in str(name).lower():
                    matching_columns.append(str(name))

            if matching_columns:
                matches.append({
                    "schema": schema,
                    "table": table,
                    "matched_by": "column_name",
                    "columns": matching_columns,
                })

    return matches
