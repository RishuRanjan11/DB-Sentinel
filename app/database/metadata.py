from sqlalchemy import inspect

from app.database.connection import engine
from app.models.schema import (
    ColumnInfo,
    ForeignKeyInfo,
    TableSchema,
    SchemaContext,
)


def get_table_metadata(schema_name: str, table_name: str) -> TableSchema:
    inspector = inspect(engine)

    columns = inspector.get_columns(
        table_name,
        schema=schema_name
    )

    primary_key = inspector.get_pk_constraint(
        table_name,
        schema=schema_name
    )

    foreign_keys = inspector.get_foreign_keys(
        table_name,
        schema=schema_name
    )

    column_info = [
        ColumnInfo(
            name=column["name"],
            data_type=str(column["type"]),
            nullable=column["nullable"],
        )
        for column in columns
    ]

    foreign_key_info = [
        ForeignKeyInfo(
            column=fk["constrained_columns"][0],
            referred_schema=fk["referred_schema"],
            referred_table=fk["referred_table"],
            referred_column=fk["referred_columns"][0],
        )
        for fk in foreign_keys
        if len(fk["constrained_columns"]) == 1
        and len(fk["referred_columns"]) == 1
    ]

    return TableSchema(
        schema_name=schema_name,
        table_name=table_name,
        columns=column_info,
        primary_key=primary_key.get("constrained_columns", []),
        foreign_keys=foreign_key_info,
    )


if __name__ == "__main__":
    table = get_table_metadata("gene", "gene")

    context = SchemaContext(
        tables=[table]
    )

    print(context.model_dump_json(indent=2))