from app.database.adapters.base import DatabaseAdapter
from app.database.adapters.factory import DatabaseAdapterRegistry
from app.database.connection import DatabaseConnection
from app.database.manager import DatabaseManager


class FakeAdapter(DatabaseAdapter):
    def __init__(self, connection):
        self.connection = connection

    def connect(self):
        pass

    def test_connection(self):
        return True

    def execute(self, query, parameters=None):
        return []

    def execute_read_only(self, query, parameters=None, statement_timeout_ms=10000, max_rows=100):
        return []

    def get_schemas(self):
        return []

    def get_tables(self, schema=None):
        return []

    def get_schema(self, schema, table):
        return {}

    def close(self):
        pass


def test_one_org_can_have_multiple_databases_without_cross_org_access():
    registry = DatabaseAdapterRegistry()
    registry.register("fake", FakeAdapter)
    manager = DatabaseManager(adapter_registry=registry)

    manager.register(DatabaseConnection("a1", "org-a", "w", "A1", "fake"))
    manager.register(DatabaseConnection("a2", "org-a", "w", "A2", "fake"))
    manager.register(DatabaseConnection("b1", "org-b", "w", "B1", "fake"))

    assert manager.list_connection_ids("org-a") == ["a1", "a2"]
    assert manager.list_connection_ids("org-b") == ["b1"]

    try:
        manager.get_for_organization("org-b", "a1")
    except PermissionError:
        pass
    else:
        raise AssertionError("Cross-organization database access was allowed")

    manager.close_all()
