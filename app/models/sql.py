from pydantic import BaseModel
from typing import List


class GeneratedSQL(BaseModel):
    sql: str
    tables_used: List[str]
    columns_used: List[str]