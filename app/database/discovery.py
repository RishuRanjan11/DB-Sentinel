from sqlalchemy import inspect

from app.database.connection import engine


def find_relevant_tables(
    keyword: str,
    schema_name: str | None = None
):
    inspector = inspect(engine)

    schemas = (
        [schema_name]
        if schema_name
        else inspector.get_schema_names()
    )

    keyword = keyword.lower()
    matches = []

    for schema in schemas:
        tables = inspector.get_table_names(schema=schema)

        for table in tables:
            table_match = keyword in table.lower()

            if table_match:
                matches.append({
                    "schema": schema,
                    "table": table,
                    "matched_by": "table_name"
                })
                continue

            # Only inspect columns if table name didn't match
            columns = inspector.get_columns(
                table,
                schema=schema
            )

            matching_columns = [
                column["name"]
                for column in columns
                if keyword in column["name"].lower()
            ]

            if matching_columns:
                matches.append({
                    "schema": schema,
                    "table": table,
                    "matched_by": "column_name",
                    "columns": matching_columns
                })

    return matches


if __name__ == "__main__":
    keyword = "ortholog"

    print(
        f"Searching tables and columns for '{keyword}'...\n",
        flush=True
    )

    matches = find_relevant_tables(keyword)

    if not matches:
        print("No matches found.")
    else:
        for match in matches:
            print(
                f"{match['schema']}.{match['table']} "
                f"→ matched by {match['matched_by']}"
            )

            if "columns" in match:
                print(
                    f"   Columns: {', '.join(match['columns'])}"
                )