from collections import defaultdict

from app.database.connection import DatabaseConnection
from app.database.adapters.base import DatabaseAdapter
from app.database.adapters.factory import DatabaseAdapterRegistry
from app.database.adapters.postgresql import PostgreSQLAdapter


class DatabaseManager:
    """Tenant-aware registry of database connections.

    One organization may own any number of database connections. Connection
    identifiers are globally unique, while all organization-scoped access is
    explicitly checked before an adapter is returned.
    """

    def __init__(self, adapter_registry: DatabaseAdapterRegistry | None = None):
        self.adapter_registry = adapter_registry or DatabaseAdapterRegistry()
        if "postgresql" not in self.adapter_registry.supported_types():
            self.adapter_registry.register("postgresql", PostgreSQLAdapter)

        self._adapters: dict[str, DatabaseAdapter] = {}
        self._connections: dict[str, DatabaseConnection] = {}
        self._organization_connections: dict[str, set[str]] = defaultdict(set)

    def register(self, connection: DatabaseConnection) -> DatabaseAdapter:
        if not connection.organization_id:
            raise ValueError("organization_id is required for every database connection.")
        if not connection.connection_id:
            raise ValueError("connection_id is required.")
        if connection.connection_id in self._connections:
            raise ValueError(f"Database connection '{connection.connection_id}' already exists.")

        adapter = self.adapter_registry.create(connection)
        self._connections[connection.connection_id] = connection
        self._adapters[connection.connection_id] = adapter
        self._organization_connections[connection.organization_id].add(connection.connection_id)
        return adapter

    def register_adapter_type(self, database_type: str, factory) -> None:
        self.adapter_registry.register(database_type, factory)

    def get(self, connection_id: str, organization_id: str | None = None) -> DatabaseAdapter:
        """Return an adapter.

        New tenant-aware callers should always pass organization_id. The
        optional form is retained for existing local tests/CLI compatibility.
        """
        adapter = self._adapters.get(connection_id)
        if adapter is None:
            raise ValueError(f"Database connection '{connection_id}' not found.")

        if organization_id is not None:
            connection = self._connections[connection_id]
            if connection.organization_id != organization_id:
                raise PermissionError(
                    f"Database connection '{connection_id}' does not belong to organization "
                    f"'{organization_id}'."
                )
        return adapter

    def get_for_organization(self, organization_id: str, connection_id: str) -> DatabaseAdapter:
        return self.get(connection_id, organization_id=organization_id)

    def get_connection(self, connection_id: str) -> DatabaseConnection:
        try:
            return self._connections[connection_id]
        except KeyError as error:
            raise ValueError(f"Database connection '{connection_id}' not found.") from error

    def list_connections(self, organization_id: str) -> list[DatabaseConnection]:
        connection_ids = sorted(self._organization_connections.get(organization_id, set()))
        return [self._connections[item] for item in connection_ids]

    def list_connection_ids(self, organization_id: str) -> list[str]:
        return sorted(self._organization_connections.get(organization_id, set()))

    def remove(self, connection_id: str, organization_id: str | None = None) -> None:
        adapter = self.get(connection_id, organization_id=organization_id)
        connection = self._connections[connection_id]
        adapter.close()
        self._adapters.pop(connection_id, None)
        self._connections.pop(connection_id, None)
        self._organization_connections[connection.organization_id].discard(connection_id)
        if not self._organization_connections[connection.organization_id]:
            self._organization_connections.pop(connection.organization_id, None)

    def close_organization(self, organization_id: str) -> None:
        for connection_id in list(self._organization_connections.get(organization_id, set())):
            self.remove(connection_id, organization_id=organization_id)

    def close_all(self) -> None:
        for adapter in list(self._adapters.values()):
            adapter.close()
        self._adapters.clear()
        self._connections.clear()
        self._organization_connections.clear()
