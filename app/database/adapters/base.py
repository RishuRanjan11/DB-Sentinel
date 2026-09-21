from abc import ABC, abstractmethod
from typing import Any


class DatabaseAdapter(ABC):

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        ...

    @abstractmethod
    def execute(
        self,
        query: Any,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict]:
        ...

    @abstractmethod
    def execute_read_only(
        self,
        query: Any,
        parameters: dict[str, Any] | None = None,
        statement_timeout_ms: int = 10_000,
        max_rows: int = 100,
    ) -> list[dict]:
        ...

    @abstractmethod
    def get_schemas(self) -> list[str]:
        ...

    @abstractmethod
    def get_tables(
        self,
        schema: str | None = None,
    ) -> list[str]:
        ...

    @abstractmethod
    def get_schema(
        self,
        schema: str,
        table: str,
    ) -> dict:
        ...

    @abstractmethod
    def close(self) -> None:
        ...