from __future__ import annotations

import getpass
import os
import uuid

from dotenv import load_dotenv

from app.agent.investigation import InvestigationEngine
from app.conversation.analyst import ConversationalAnalyst
from app.conversation.session import ConversationSession
from app.database.connection import DatabaseConnection
from app.database.data_dictionary import DatabaseDictionary
from app.database.manager import DatabaseManager
from app.core.console import configure_console, section, safe_text

load_dotenv()


def prompt_required(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value:
            return value
        print("This value is required.")


def create_database(manager: DatabaseManager, organization_id: str) -> str:
    print()
    section("Connect Database")

    env_defaults = {
        "name": os.getenv("DB_NAME") or os.getenv("FLYBASE_DB_NAME") or "FlyBase",
        "database_type": os.getenv("DB_TYPE") or "postgresql",
        "workspace_id": os.getenv("DB_WORKSPACE_ID") or "default",
        "host": os.getenv("FLYBASE_DB_HOST") or os.getenv("DB_HOST") or None,
        "port": os.getenv("FLYBASE_DB_PORT") or os.getenv("DB_PORT") or "5432",
        "database_name": os.getenv("FLYBASE_DB_NAME") or os.getenv("DB_NAME") or None,
        "username": os.getenv("FLYBASE_DB_USER") or os.getenv("DB_USER") or None,
        "password": os.getenv("FLYBASE_DB_PASSWORD") or os.getenv("DB_PASSWORD") or None,
        "ssl_enabled": os.getenv("DB_SSL_ENABLED", "yes").lower() in {
            "y",
            "yes",
            "true",
            "1",
        },
    }

    name = prompt_required("Database name", env_defaults["name"])
    database_type = prompt_required("Database type", env_defaults["database_type"]).lower()
    connection_id = prompt_required(
        "Connection ID",
        f"{organization_id}-{uuid.uuid4().hex[:8]}",
    )
    workspace_id = prompt_required("Workspace ID", env_defaults["workspace_id"])
    host = prompt_required("Host", env_defaults["host"])
    port = int(prompt_required("Port", env_defaults["port"]))
    database_name = prompt_required("Database/schema name", env_defaults["database_name"])
    username = prompt_required("Username", env_defaults["username"])
    password = os.getenv("FLYBASE_DB_PASSWORD") or os.getenv("DB_PASSWORD")
    if password is None:
        password = getpass.getpass("Password (leave empty if none): ")
    ssl_enabled = prompt_required(
        "SSL enabled",
        "yes" if env_defaults["ssl_enabled"] else "no",
    ).lower() in {
        "y",
        "yes",
        "true",
        "1",
    }

    connection = DatabaseConnection(
        connection_id=connection_id,
        organization_id=organization_id,
        workspace_id=workspace_id,
        name=name,
        database_type=database_type,
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        password=password or None,
        ssl_enabled=ssl_enabled,
    )

    manager.register(connection)
    adapter = manager.get_for_organization(
        organization_id,
        connection_id,
    )

    print("Testing database connection...")
    if not adapter.test_connection():
        manager.remove(connection_id, organization_id=organization_id)
        raise RuntimeError("Database connection test failed.")

    print(f"Connected: {name} ({database_type})")
    return connection_id


def print_result(result: dict) -> None:
    print()
    section("Result")
    print(f"Status: {result.get('status', 'unknown')}")

    if result.get("reason"):
        print(f"Reason: {result['reason']}")

    insight = result.get("insight")
    if insight:
        print()
        print("Insight")
        print(safe_text("───────"))
        print(insight.get("summary", ""))
        for finding in insight.get("findings", []):
            print(f"{safe_text('•')} {finding}")

    dashboard = result.get("dashboard")
    if dashboard:
        print()
        print("Dashboard artifact generated.")
        print(f"Title: {dashboard.get('title', '')}")
        print(f"Widgets: {len(dashboard.get('widgets', []))}")

    report = result.get("report")
    if report:
        print("Report artifact generated.")

    verification = result.get("insight_verification")
    if verification:
        state = "passed" if verification.get("verified") else "review required"
        print(
            f"Insight verification: {state} "
            f"({verification.get('confidence', 0):.2f})"
        )

    evidence = result.get("evidence", [])
    print(f"Evidence: {len(evidence)} item(s)")
    displayed = 0
    total_rows = sum(
        len(item.get("rows") or [])
        for item in evidence
        if isinstance(item, dict)
    )
    for item in evidence:
        rows = item.get("rows") or []
        if not rows:
            continue
        remaining = 10 - displayed
        if remaining <= 0:
            break
        print()
        print(
            f"Evidence rows (step {item.get('step_id', '-')}, "
            f"{len(rows)} row(s)):"
        )
        for row in rows[:remaining]:
            print(safe_text(str(row)))
            displayed += 1
    if total_rows > displayed:
        print(
            f"Showing {displayed} of {total_rows} evidence rows; "
            "the complete result remains in the returned evidence/report."
        )


def run() -> None:
    configure_console()
    print()
    print("DB-SENTINEL")
    print(safe_text("───────────"))
    print("Conversational Autonomous Data Analyst")

    organization_id = prompt_required("Organization ID", "local")

    manager = DatabaseManager()
    session: ConversationSession | None = None
    data_dictionaries: dict[str, DatabaseDictionary] = {}

    try:
        while True:
            connections = manager.list_connections(organization_id)

            print()
            section(f"Organization: {organization_id}")

            if connections:
                print("Connected databases:")
                for index, connection in enumerate(connections, start=1):
                    print(
                        f"  {index}. {connection.name} "
                        f"[{connection.database_type}] "
                        f"({connection.connection_id})"
                    )
            else:
                print("No databases connected yet.")

            print()
            print("Commands:")
            print("  connect  - connect another database")
            print("  use      - select a connected database")
            print("  list     - list connected databases")
            print("  quit     - exit")

            command = input("\nDB-Sentinel> ").strip()

            if command.lower() in {"quit", "exit", "q"}:
                break

            if command.lower() == "connect":
                try:
                    connection_id = create_database(manager, organization_id)
                    if session is None:
                        session = ConversationSession(
                            conversation_id=uuid.uuid4().hex,
                            organization_id=organization_id,
                            database_connection_ids=tuple(
                                manager.list_connection_ids(organization_id)
                            ),
                        )
                    else:
                        session.database_connection_ids = tuple(
                            manager.list_connection_ids(organization_id)
                        )
                    print(f"Connection ready: {connection_id}")
                except Exception as error:
                    print(f"Connection failed: {error}")
                continue

            if command.lower() == "list":
                continue

            if command.lower() == "use":
                connections = manager.list_connections(organization_id)
                if not connections:
                    print("No databases connected. Use 'connect' first.")
                    continue

                selection = input("Select database number or connection ID: ").strip()
                selected = next(
                    (
                        item
                        for item in connections
                        if item.connection_id == selection
                        or item.name == selection
                        or str(connections.index(item) + 1) == selection
                    ),
                    None,
                )

                if selected is None:
                    print("Database not found.")
                    continue

                if selected.connection_id not in data_dictionaries:
                    dictionary_path = os.getenv(
                        "DB_SENTINEL_DATA_DICTIONARY"
                    )
                    if dictionary_path:
                        try:
                            data_dictionaries[selected.connection_id] = (
                                DatabaseDictionary.load(
                                    dictionary_path,
                                    manager.get_for_organization(
                                        organization_id,
                                        selected.connection_id,
                                    ),
                                )
                            )
                            print(
                                "Loaded data dictionary: "
                                f"{data_dictionaries[selected.connection_id].documented_table_count} "
                                "table description(s), "
                                f"{data_dictionaries[selected.connection_id].documented_column_count} "
                                "column description(s)."
                            )
                        except (OSError, ValueError) as error:
                            print(f"Data dictionary was not loaded: {error}")

                if session is None:
                    session = ConversationSession(
                        conversation_id=uuid.uuid4().hex,
                        organization_id=organization_id,
                        database_connection_ids=tuple(
                            manager.list_connection_ids(organization_id)
                        ),
                    )
                else:
                    session.database_connection_ids = tuple(
                        manager.list_connection_ids(organization_id)
                    )

                analyst = ConversationalAnalyst(
                    InvestigationEngine(manager)
                )

                print()
                print(f"Using database: {selected.name}")
                print(
                    "Ask questions continuously. Type /dictionary to load a "
                    "reference, /dictionary clear to remove it, or /back to "
                    "choose another database."
                )

                while True:
                    question = input("\nYou> ").strip()

                    normalized_question = question.lower()

                    # Session commands are deterministic application controls.
                    # Never send them to the LLM/investigation engine.
                    if normalized_question in {"/back", "/use", "back"}:
                        break
                    if normalized_question in {"/quit", "/exit", "quit", "exit", "q"}:
                        return
                    if normalized_question == "/dictionary":
                        dictionary_path = input(
                            "Data dictionary JSON path: "
                        ).strip()
                        if not dictionary_path:
                            print("No data dictionary loaded.")
                            continue
                        try:
                            dictionary = DatabaseDictionary.load(
                                dictionary_path,
                                manager.get_for_organization(
                                    organization_id,
                                    selected.connection_id,
                                ),
                            )
                        except (OSError, ValueError) as error:
                            print(f"Data dictionary was not loaded: {error}")
                            continue
                        data_dictionaries[selected.connection_id] = dictionary
                        print(
                            "Data dictionary loaded as unverified reference: "
                            f"{dictionary.documented_table_count} table description(s), "
                            f"{dictionary.documented_column_count} column description(s). "
                            "It does not replace live database metadata or evidence."
                        )
                        continue
                    if normalized_question == "/dictionary clear":
                        data_dictionaries.pop(selected.connection_id, None)
                        print("Data dictionary reference cleared.")
                        continue
                    if not question:
                        continue

                    try:
                        result = analyst.ask(
                            session=session,
                            question=question,
                            connection_id=selected.connection_id,
                            data_dictionary=data_dictionaries.get(
                                selected.connection_id
                            ),
                        )
                        print_result(result)
                    except Exception as error:
                        print(f"Analysis failed: {error}")
                continue

            print("Unknown command. Use connect, use, list, or quit.")

    finally:
        manager.close_all()


if __name__ == "__main__":
    run()
