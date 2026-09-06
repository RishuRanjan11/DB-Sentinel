from pydantic import BaseModel
from typing import List, Optional


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool


class ForeignKeyInfo(BaseModel):
    column: str
    referred_schema: str
    referred_table: str
    referred_column: str


class TableSchema(BaseModel):
    schema_name: str
    table_name: str
    columns: List[ColumnInfo]
    primary_key: List[str]
    foreign_keys: List[ForeignKeyInfo]


class SchemaContext(BaseModel):
    tables: List[TableSchema]