from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatabaseConnection:
    """
    Represents a database connection configuration.

    This object is database-engine agnostic.
    """

    connection_id: str

    organization_id: str
    workspace_id: str

    name: str
    database_type: str

    host: str | None = None
    port: int | None = None
    database_name: str | None = None
    username: str | None = None
    password: str | None = None

    ssl_enabled: bool = True

    options: dict[str, Any] | None = None