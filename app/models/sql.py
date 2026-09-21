from pydantic import BaseModel, ConfigDict


class GeneratedSQL(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    sql: str
    tables_used: list[str]
    columns_used: list[str]