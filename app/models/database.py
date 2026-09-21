from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DatabaseType(str, Enum):

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"


class DatabaseConnectionConfig(BaseModel):

    id: str = Field(
        ...,
        min_length=1,
    )

    name: str = Field(
        ...,
        min_length=1,
    )

    database_type: DatabaseType

    host: str

    port: int

    database_name: str

    username: str

    password: str | None = None

    ssl_enabled: bool = True

    created_at: datetime | None = None