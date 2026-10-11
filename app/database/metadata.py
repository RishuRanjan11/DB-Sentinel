from app.database.adapters.base import DatabaseAdapter
from app.models.schema import ColumnInfo, ForeignKeyInfo, TableSchema, SchemaContext


def get_table_metadata(adapter: DatabaseAdapter, schema_name: str, table_name: str) -> TableSchema:
    details = adapter.get_schema(schema=schema_name, table=table_name)
    columns = [
        ColumnInfo(
            name=str(column.get("name", column.get("column", ""))),
            data_type=str(column.get("data_type", "unknown")),
            nullable=bool(column.get("nullable", True)),
        )
        for column in details.get("columns", [])
        if isinstance(column, dict)
    ]

    foreign_keys = [
        ForeignKeyInfo(
            column=str(fk.get("column", fk.get("source_column", ""))),
            referred_schema=str(fk.get("referred_schema", fk.get("target_schema", ""))),
            referred_table=str(fk.get("referred_table", fk.get("target_table", ""))),
            referred_column=str(fk.get("referred_column", fk.get("target_column", ""))),
        )
        for fk in details.get("foreign_keys", [])
        if isinstance(fk, dict)
    ]

    return TableSchema(
        schema_name=schema_name,
        table_name=table_name,
        columns=columns,
        primary_key=list(details.get("primary_key", details.get("primary_keys", []))),
        foreign_keys=foreign_keys,
    )


def get_schema_context(adapter: DatabaseAdapter) -> SchemaContext:
    tables = []
    for schema in adapter.get_schemas():
        for table in adapter.get_tables(schema=schema):
            tables.append(get_table_metadata(adapter, schema, table))
    return SchemaContext(tables=tables)
