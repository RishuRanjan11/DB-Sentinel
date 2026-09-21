from app.database.connection import DatabaseConnection
from app.database.adapters.base import DatabaseAdapter
from app.database.adapters.postgresql import PostgreSQLAdapter


class DatabaseManager:

    def __init__(self):

        self._adapters: dict[
            str,
            DatabaseAdapter,
        ] = {}

    # --------------------------------------------------
    # Register connection
    # --------------------------------------------------

    def register(
        self,
        connection: DatabaseConnection,
    ) -> DatabaseAdapter:

        if connection.connection_id in self._adapters:
            raise ValueError(
                f"Database connection "
                f"'{connection.connection_id}' "
                f"already exists."
            )

        adapter = self._create_adapter(
            connection
        )

        self._adapters[
            connection.connection_id
        ] = adapter

        return adapter

    # --------------------------------------------------
    # Create adapter
    # --------------------------------------------------

    def _create_adapter(
        self,
        connection: DatabaseConnection,
    ) -> DatabaseAdapter:

        database_type = (
            connection.database_type.lower()
        )

        if database_type == "postgresql":

            return PostgreSQLAdapter(
                connection
            )

        raise ValueError(
            f"Unsupported database type: "
            f"{connection.database_type}"
        )

    # --------------------------------------------------
    # Get adapter
    # --------------------------------------------------

    def get(
        self,
        connection_id: str,
    ) -> DatabaseAdapter:

        adapter = self._adapters.get(
            connection_id
        )

        if adapter is None:

            raise ValueError(
                f"Database connection "
                f"'{connection_id}' not found."
            )

        return adapter

    # --------------------------------------------------
    # Remove connection
    # --------------------------------------------------

    def remove(
        self,
        connection_id: str,
    ) -> None:

        adapter = self._adapters.pop(
            connection_id,
            None,
        )

        if adapter is not None:
            adapter.close()

    # --------------------------------------------------
    # Close all
    # --------------------------------------------------

    def close_all(self) -> None:

        for adapter in self._adapters.values():
            adapter.close()

        self._adapters.clear()