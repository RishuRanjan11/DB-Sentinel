from app.database.connection import DatabaseConnection
from app.database.manager import DatabaseManager


def main():

    print("1. Creating connection configuration...")

    flybase = DatabaseConnection(
        connection_id="dev-flybase",
        organization_id="development-org",
        workspace_id="development-workspace",
        name="FlyBase Development",
        database_type="postgresql",
        host="chado.flybase.org",
        port=5432,
        database_name="flybase",
        username="flybase",
        password=None,
        ssl_enabled=False,
    )

    print("2. Creating DatabaseManager...")

    manager = DatabaseManager()

    print("3. Registering adapter...")

    adapter = manager.register(
        flybase
    )

    print("4. Testing database connection...")

    connected = adapter.test_connection()

    print(
        "5. Connection successful:",
        connected
    )

    print("6. Closing connection...")

    manager.close_all()

    print("7. Done.")


if __name__ == "__main__":
    main()