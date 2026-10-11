from collections.abc import Callable

from app.database.adapters.base import DatabaseAdapter
from app.database.connection import DatabaseConnection


AdapterFactory = Callable[[DatabaseConnection], DatabaseAdapter]


class DatabaseAdapterRegistry:
    """Registry for pluggable database adapters."""

    def __init__(self):
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, database_type: str, factory: AdapterFactory) -> None:
        key = database_type.strip().lower()
        if not key:
            raise ValueError("Database type cannot be empty.")
        self._factories[key] = factory

    def create(self, connection: DatabaseConnection) -> DatabaseAdapter:
        key = connection.database_type.strip().lower()
        factory = self._factories.get(key)
        if factory is None:
            raise ValueError(f"Unsupported database type: {connection.database_type}")
        return factory(connection)

    def supported_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
