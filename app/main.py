from __future__ import annotations

import getpass
import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from dotenv import dotenv_values, load_dotenv
from sqlalchemy.engine import URL, make_url

from app.agent.investigation import InvestigationEngine
from app.conversation.analyst import ConversationalAnalyst
from app.conversation.session import ConversationSession
from app.database.connection import DatabaseConnection
from app.database.manager import DatabaseManager
from app.core.console import configure_console, section, safe_text
from app.core.trace import logger


def _environment_file_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    configured_path = os.getenv("DB_SENTINEL_ENV_FILE")
    if not configured_path:
        return project_root / ".env"
    env_file = Path(configured_path).expanduser()
    return env_file if env_file.is_absolute() else project_root / env_file


load_dotenv(dotenv_path=_environment_file_path())


def prompt_required(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if not value and default is not None:
            return default
        if value:
            return value
        print("This value is required.")


def prompt_port(default: str) -> int:
    while True:
        value = prompt_required("Port", default)
        try:
            port = int(value)
        except ValueError:
            print("Port must be a positive integer.")
            continue
        if port > 0:
            return port
        print("Port must be a positive integer.")


@contextmanager
def analysis_progress():
    stop = threading.Event()
    frames = ("|", "/", "-", "\\")

    def animate() -> None:
        index = 0
        while not stop.wait(0.15):
            sys.stdout.write(
                f"\rAnalyzing your question... {frames[index % len(frames)]}"
            )
            sys.stdout.flush()
            index += 1

    print("Analyzing your question... |", end="", flush=True)
    worker = threading.Thread(target=animate, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        print("\rAnalysis finished.                         ", flush=True)


def _connection_from_url(
    connection_url: str,
    connection_id: str,
    organization_id: str,
    workspace_id: str,
    name: str,
    database_name: str | None = None,
) -> DatabaseConnection:
    connection_url = connection_url.strip()
    if re.match(r"^mongodb(?:\+srv)?://", connection_url, re.IGNORECASE):
        try:
            parsed = urlsplit(connection_url)
            parsed_hostname = parsed.hostname
        except ValueError as error:
            raise ValueError(
                "MongoDB URI is malformed. URL-encode reserved password "
                "characters such as @ and #."
            ) from error
        uri_database_name = unquote(
            parsed.path.lstrip("/").split("/", 1)[0]
        )
        if parsed.fragment:
            raise ValueError(
                "MongoDB URI contains a fragment, usually because # in the "
                "password was not URL-encoded as %23. URL-encode the entire "
                "password and include /<database-name> after the host."
            )
        if not parsed_hostname:
            raise ValueError(
                "MongoDB URI host is missing or malformed. URL-encode "
                "reserved password characters such as @ and #."
            )
        database_name = uri_database_name or (database_name or "").strip()
        if not database_name:
            raise ValueError(
                "MongoDB database name is required. Provide it in the URI "
                "path or enter it when prompted."
            )
        normalized_uri = connection_url
        if not uri_database_name:
            normalized_uri = parsed._replace(
                path="/" + quote(database_name, safe="")
            ).geturl()
        query = {
            key.lower(): values[-1]
            for key, values in parse_qs(parsed.query).items()
            if values
        }
        ssl_enabled = query.get(
            "tls",
            query.get("ssl", "true"),
        ).lower() not in {"false", "0", "no"}
        return DatabaseConnection(
            connection_id=connection_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            name=name,
            database_type="mongodb",
            host=normalized_uri,
            database_name=database_name,
            ssl_enabled=ssl_enabled,
        )

    normalized_url = connection_url.strip()
    if normalized_url.lower().startswith("postgres://"):
        normalized_url = "postgresql://" + normalized_url[len("postgres://"):]
    elif normalized_url.lower().startswith("sqlserver://"):
        normalized_url = "mssql://" + normalized_url[len("sqlserver://"):]
    scheme_separator = normalized_url.find("://")
    if scheme_separator >= 0:
        authority_start = scheme_separator + 3
        authority_end = min(
            (
                position
                for delimiter in "/?#"
                if (position := normalized_url.find(delimiter, authority_start))
                >= 0
            ),
            default=len(normalized_url),
        )
        authority = normalized_url[authority_start:authority_end]
        username_password, separator, host = authority.rpartition("@")
        if separator and "@" in username_password:
            normalized_url = (
                normalized_url[:authority_start]
                + username_password.replace("@", "%40")
                + "@"
                + host
                + normalized_url[authority_end:]
            )
    try:
        parsed = make_url(normalized_url)
    except Exception as error:
        raise ValueError("Database URL is invalid or malformed.") from error

    scheme = parsed.drivername.lower()
    if scheme.startswith("postgresql"):
        database_type = "postgresql"
    elif scheme.startswith("mysql"):
        database_type = "mysql"
    elif scheme.startswith("mssql"):
        database_type = "sqlserver"
    else:
        raise ValueError(
            "Database URL scheme must be PostgreSQL, MySQL, SQL Server, "
            "or MongoDB."
        )
    if not parsed.host or not parsed.database:
        raise ValueError(
            "Database URL must include a host and database name."
        )

    query = dict(parsed.query)
    normalized_query = {
        str(key).lower(): str(value).lower()
        for key, value in query.items()
    }
    ssl_enabled = not (
        normalized_query.get("sslmode") == "disable"
        or normalized_query.get("encrypt") in {"no", "false", "0"}
        or normalized_query.get("ssl") in {"false", "0", "no"}
        or normalized_query.get("ssl_disabled") in {"true", "1", "yes"}
    )
    return DatabaseConnection(
        connection_id=connection_id,
        organization_id=organization_id,
        workspace_id=workspace_id,
        name=name,
        database_type=database_type,
        host=parsed.host,
        port=parsed.port,
        database_name=parsed.database,
        username=parsed.username,
        password=parsed.password,
        ssl_enabled=ssl_enabled,
        options=query,
    )


def _connection_url(connection: DatabaseConnection) -> str:
    if connection.database_type.lower() == "mongodb":
        if not connection.host:
            raise ValueError("MongoDB connection URI is missing.")
        if connection.host.startswith(("mongodb://", "mongodb+srv://")):
            return connection.host
        auth = ""
        if connection.username:
            auth = quote(connection.username, safe="")
            if connection.password:
                auth += ":" + quote(connection.password, safe="")
            auth += "@"
        port = f":{connection.port}" if connection.port else ""
        query = f"?tls={'true' if connection.ssl_enabled else 'false'}"
        return (
            f"mongodb://{auth}{connection.host}{port}/"
            f"{quote(connection.database_name or '', safe='')}{query}"
        )
    drivername = {
        "postgres": "postgresql+psycopg2",
        "postgresql": "postgresql+psycopg2",
        "mysql": "mysql+pymysql",
        "sqlserver": "mssql+pyodbc",
        "sql server": "mssql+pyodbc",
        "microsoft sql server": "mssql+pyodbc",
        "mssql": "mssql+pyodbc",
        "sql": "mssql+pyodbc",
    }.get(connection.database_type.lower())
    if drivername is None:
        raise ValueError(
            f"Cannot persist unsupported database type "
            f"'{connection.database_type}'."
        )
    query = {
        str(key): str(value)
        for key, value in (connection.options or {}).items()
    }
    database_type = connection.database_type.lower()
    if database_type in {"postgres", "postgresql"}:
        query["sslmode"] = "require" if connection.ssl_enabled else "disable"
    elif database_type == "mysql" and not connection.ssl_enabled:
        query["ssl_disabled"] = "true"
    elif database_type in {
        "sqlserver",
        "sql server",
        "microsoft sql server",
        "mssql",
        "sql",
    }:
        query["Encrypt"] = "yes" if connection.ssl_enabled else "no"
    return URL.create(
        drivername=drivername,
        username=connection.username,
        password=connection.password,
        host=connection.host,
        port=connection.port,
        database=connection.database_name,
        query=query,
    ).render_as_string(hide_password=False)


def _saved_connection_profiles() -> list[dict[str, str]]:
    file_values = dotenv_values(_environment_file_path())
    raw_profiles = file_values.get("DB_SENTINEL_CONNECTIONS_JSON")
    if raw_profiles is None:
        raw_profiles = os.getenv("DB_SENTINEL_CONNECTIONS_JSON", "")
    if not raw_profiles:
        return []
    try:
        profiles = json.loads(raw_profiles)
    except json.JSONDecodeError as error:
        raise ValueError(
            "DB_SENTINEL_CONNECTIONS_JSON in the local .env is invalid JSON."
        ) from error
    if not isinstance(profiles, list):
        raise ValueError(
            "DB_SENTINEL_CONNECTIONS_JSON must contain a JSON array."
        )
    for profile in profiles:
        if not isinstance(profile, dict) or not all(
            isinstance(profile.get(field), str) and profile[field]
            for field in (
                "connection_id",
                "organization_id",
                "workspace_id",
                "name",
                "url",
            )
        ):
            raise ValueError(
                "Each saved database profile must contain non-empty "
                "connection_id, organization_id, workspace_id, name, and url."
            )
    return profiles


def _persist_connection(connection: DatabaseConnection) -> None:
    profiles = [
        profile
        for profile in _saved_connection_profiles()
        if not (
            profile["connection_id"] == connection.connection_id
            and profile["organization_id"] == connection.organization_id
        )
    ]
    profiles.append({
        "connection_id": connection.connection_id,
        "organization_id": connection.organization_id,
        "workspace_id": connection.workspace_id,
        "name": connection.name,
        "url": _connection_url(connection),
    })
    _write_saved_connection_profiles(profiles)


def _write_saved_connection_profiles(
    profiles: list[dict[str, str]],
) -> None:
    from dotenv import set_key

    serialized = json.dumps(profiles, ensure_ascii=True, separators=(",", ":"))
    env_file = _environment_file_path()
    set_key(
        str(env_file),
        "DB_SENTINEL_CONNECTIONS_JSON",
        serialized,
        quote_mode="always",
    )
    os.environ["DB_SENTINEL_CONNECTIONS_JSON"] = serialized


def _remove_persisted_connection(
    connection_id: str,
    organization_id: str,
) -> bool:
    profiles = _saved_connection_profiles()
    retained = [
        profile
        for profile in profiles
        if not (
            profile["connection_id"] == connection_id
            and profile["organization_id"] == organization_id
        )
    ]
    if len(retained) == len(profiles):
        return False
    _write_saved_connection_profiles(retained)
    return True


def delete_saved_connection(
    manager: DatabaseManager,
    organization_id: str,
    connection_id: str,
) -> bool:
    connection = manager.get_connection(connection_id)
    if connection.organization_id != organization_id:
        raise PermissionError(
            f"Database connection '{connection_id}' does not belong "
            f"to organization '{organization_id}'."
        )
    _remove_persisted_connection(connection_id, organization_id)
    manager.remove(connection_id, organization_id=organization_id)
    logger.info(
        "CONNECTION PROFILE DELETED organization_id=%s workspace_id=%s "
        "connection_id=%s",
        organization_id,
        connection.workspace_id,
        connection_id,
    )
    return True


def _restore_connection_profiles(
    manager: DatabaseManager,
    organization_id: str,
    profiles: list[dict[str, str]] | None = None,
) -> list[str]:
    restored = []
    for profile in (
        profiles
        if profiles is not None
        else _saved_connection_profiles()
    ):
        if profile["organization_id"] != organization_id:
            continue
        connection = _connection_from_url(
            connection_url=profile["url"],
            connection_id=profile["connection_id"],
            organization_id=profile["organization_id"],
            workspace_id=profile["workspace_id"],
            name=profile["name"],
        )
        manager.register(connection)
        restored.append(connection.name)
    return restored


def _register_and_test(
    manager: DatabaseManager,
    connection: DatabaseConnection,
) -> str:
    manager.register(connection)
    adapter = manager.get_for_organization(
        connection.organization_id,
        connection.connection_id,
    )
    try:
        connected = adapter.test_connection()
    except Exception:
        manager.remove(
            connection.connection_id,
            organization_id=connection.organization_id,
        )
        raise
    if not connected:
        manager.remove(
            connection.connection_id,
            organization_id=connection.organization_id,
        )
        detail = getattr(adapter, "last_connection_error", None)
        if detail:
            password = connection.password
            if password:
                detail = detail.replace(password, "[REDACTED]")
                detail = detail.replace(
                    quote(password, safe=""),
                    "[REDACTED]",
                )
            raise RuntimeError(
                f"Database connection test failed: {detail}"
            )
        raise RuntimeError("Database connection test failed; see the log for details.")
    profile_saved = False
    try:
        _persist_connection(connection)
        profile_saved = True
    except (OSError, ValueError) as error:
        logger.error(
            "Connected database profile could not be saved: %s",
            error,
        )
        print(
            "Connected, but this database profile could not be saved "
            f"for the next start: {error}"
        )
    print(f"Connected: {connection.name} ({connection.database_type})")
    if profile_saved:
        print("Connection profile saved locally for the next start.")
    return connection.connection_id


def create_database(manager: DatabaseManager, organization_id: str) -> str:
    configured_url = (
        os.getenv("DB_SENTINEL_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("DB_URL")
    )
    default_mode = "url" if configured_url else "fields"
    mode = input(
        f"Connection input (fields/url) [{default_mode}]: "
    ).strip().lower() or default_mode
    if mode not in {"fields", "url", "uri"}:
        raise ValueError("Connection input must be 'fields' or 'url'.")
    if mode in {"url", "uri"}:
        values = {
            "url": configured_url,
            "database_name": None,
            "connection_id": f"{organization_id}-{uuid.uuid4().hex[:8]}",
            "workspace_id": "default",
            "name": "Database",
        }
        if not values["url"]:
            values["url"] = getpass.getpass(
                "Database URL/URI (hidden input): "
            ).strip()
        if not values["url"]:
            raise ValueError(
                "Provide a database URL or set "
                "DB_SENTINEL_DATABASE_URL, DATABASE_URL, or DB_URL."
            )
        values["connection_id"] = prompt_required(
            "Connection ID",
            values["connection_id"],
        )
        values["workspace_id"] = prompt_required(
            "Workspace ID",
            values["workspace_id"],
        )
        values["name"] = prompt_required(
            "Connection display name",
            values["name"],
        )
        fields = [
            ("url", "Database URL/URI"),
            ("connection_id", "Connection ID"),
            ("workspace_id", "Workspace ID"),
            ("name", "Connection display name"),
        ]
        while True:
            is_mongodb_uri = False
            mongodb_uri_has_database = False
            mongodb_uri_fragment = False
            try:
                is_mongodb_uri = bool(
                    re.match(
                        r"^mongodb(?:\+srv)?://",
                        values["url"],
                        re.IGNORECASE,
                    )
                )
                if is_mongodb_uri:
                    try:
                        mongodb_uri = urlsplit(values["url"])
                        mongodb_uri_has_database = bool(
                            mongodb_uri.path.strip("/")
                        )
                        mongodb_uri_fragment = bool(mongodb_uri.fragment)
                    except ValueError:
                        pass
                    if (
                        not mongodb_uri_has_database
                        and not values["database_name"]
                        and not mongodb_uri_fragment
                    ):
                        values["database_name"] = prompt_required(
                            "MongoDB database name"
                        )
                connection = _connection_from_url(
                    connection_url=values["url"],
                    connection_id=values["connection_id"],
                    organization_id=organization_id,
                    workspace_id=values["workspace_id"],
                    name=values["name"],
                    database_name=values["database_name"],
                )
                print("Testing database connection...")
                return _register_and_test(manager, connection)
            except Exception as error:
                print(f"Connection failed: {error}")
                print("Edit a field and retry, or type 'cancel'.")
                editable_fields = list(fields)
                if is_mongodb_uri and not mongodb_uri_has_database:
                    editable_fields.append(
                        ("database_name", "MongoDB database name")
                    )
                for index, (key, label) in enumerate(
                    editable_fields,
                    start=1,
                ):
                    shown_value = (
                        "[hidden]" if key == "url" and values[key] else values[key]
                    )
                    print(f"  {index}. {label}: {shown_value or '(unset)'}")
                selection = input("Field number to edit [cancel]: ").strip()
                if not selection:
                    raise
                try:
                    field_key, label = editable_fields[int(selection) - 1]
                except (ValueError, IndexError):
                    if selection.lower() == "cancel":
                        raise
                    print("Choose a listed field number or cancel.")
                    continue
                if field_key == "url":
                    values[field_key] = getpass.getpass(
                        f"{label} (hidden input): "
                    ).strip()
                    values["database_name"] = None
                else:
                    values[field_key] = prompt_required(label, values[field_key])
    return _create_database_from_fields(manager, organization_id)


def _create_database_from_fields(
    manager: DatabaseManager,
    organization_id: str,
) -> str:
    print()
    section("Connect Database")

    env_defaults = {
        "name": os.getenv("DB_DISPLAY_NAME") or "Database",
        "database_type": os.getenv("DB_TYPE") or "postgresql",
        "workspace_id": os.getenv("DB_WORKSPACE_ID") or "default",
        "host": os.getenv("DB_HOST") or None,
        "port": os.getenv("DB_PORT") or "5432",
        "database_name": os.getenv("DB_DATABASE_NAME") or os.getenv("DB_NAME") or None,
        "username": os.getenv("DB_USER") or None,
        "password": os.getenv("DB_PASSWORD"),
        "ssl_enabled": os.getenv("DB_SSL_ENABLED", "yes").lower() in {
            "y",
            "yes",
            "true",
            "1",
        },
    }

    name = prompt_required("Connection display name", env_defaults["name"])
    database_type = prompt_required("Database type", env_defaults["database_type"]).lower()
    default_ports = {
        "postgresql": "5432",
        "postgres": "5432",
        "mysql": "3306",
        "sql": "1433",
        "sqlserver": "1433",
        "sql server": "1433",
        "mssql": "1433",
        "mongodb": "27017",
    }
    connection_id = prompt_required(
        "Connection ID",
        f"{organization_id}-{uuid.uuid4().hex[:8]}",
    )
    workspace_id = prompt_required("Workspace ID", env_defaults["workspace_id"])
    host = prompt_required("Host", env_defaults["host"])
    port = prompt_port(
        default_ports.get(
            database_type,
            env_defaults["port"],
        )
    )
    database_name = prompt_required("Database/schema name", env_defaults["database_name"])
    if database_type == "mongodb":
        username = (
            input("Username (optional): ").strip()
            or env_defaults["username"]
            or None
        )
    else:
        username = prompt_required("Username", env_defaults["username"])
    password = os.getenv("DB_PASSWORD")
    if password is None and username:
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

    fields = (
        ("name", "Connection display name"),
        ("database_type", "Database type"),
        ("connection_id", "Connection ID"),
        ("workspace_id", "Workspace ID"),
        ("host", "Host"),
        ("port", "Port"),
        ("database_name", "Database/schema name"),
        ("username", "Username"),
        ("password", "Password"),
        ("ssl_enabled", "SSL enabled"),
    )
    while True:
        try:
            print("Testing database connection...")
            return _register_and_test(manager, connection)
        except Exception as error:
            print(f"Connection failed: {error}")
            print("Edit a field and retry, or type 'cancel'.")
            for index, (field_name, label) in enumerate(fields, start=1):
                value = getattr(connection, field_name)
                shown = "[hidden]" if field_name == "password" and value else value
                print(f"  {index}. {label}: {shown or '(unset)'}")
            selection = input("Field number to edit [cancel]: ").strip()
            if not selection:
                raise
            try:
                field_name, label = fields[int(selection) - 1]
            except (ValueError, IndexError):
                if selection.lower() == "cancel":
                    raise
                print("Choose a listed field number or cancel.")
                continue
            if field_name == "password":
                value = getpass.getpass("Password (leave empty if none): ")
            else:
                value = prompt_required(
                    label,
                    str(getattr(connection, field_name) or ""),
                )
            if field_name == "port":
                try:
                    value = int(value)
                except ValueError:
                    print("Port must be a positive integer.")
                    continue
                if value <= 0:
                    print("Port must be a positive integer.")
                    continue
            if field_name == "ssl_enabled":
                value = str(value).lower() in {"y", "yes", "true", "1"}
            if field_name in {"username", "password"} and not value:
                value = None
            connection = replace(connection, **{field_name: value})


def print_result(result: dict) -> None:
    print()
    section("Result")
    print(f"Status: {result.get('status', 'unknown')}")

    if result.get("reason"):
        print(f"Reason: {result['reason']}")

    normalized_question = result.get("normalized_question")
    if normalized_question and normalized_question != result.get("question"):
        print(f"Interpreted question: {safe_text(normalized_question)}")

    insight = result.get("insight")
    if insight:
        print()
        print("Insight")
        print(safe_text("───────"))
        print(insight.get("summary", ""))
        for finding in insight.get("findings", []):
            print(f"{safe_text('•')} {finding}")
        recommendations = insight.get("recommendations", [])
        if recommendations:
            print()
            print("Recommended actions")
            for recommendation in recommendations:
                print(f"{safe_text('•')} {recommendation}")
        uncertainties = insight.get("uncertainties", [])
        if uncertainties:
            print()
            print("Limitations")
            for uncertainty in uncertainties:
                print(f"{safe_text('•')} {uncertainty}")

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
    max_display_rows = 100
    preview_rows = sum(
        len(item.get("rows") or [])
        for item in evidence
        if isinstance(item, dict)
    )
    scanned_rows = sum(
        item.get(
            "total_rows",
            len(item.get("rows") or []),
        )
        for item in evidence
        if isinstance(item, dict)
    )
    for item in evidence:
        rows = item.get("rows") or []
        if item.get("type") == "database_metadata" and re.search(
            r"\b(?:columns?|fields?)\b",
            str(result.get("question", "")),
            flags=re.IGNORECASE,
        ):
            print()
            print("Database columns (live metadata)")
            for table in item.get("tables", []):
                table_name = (
                    f"{table.get('schema', '')}.{table.get('table', '')}"
                ).strip(".")
                columns = table.get("columns", [])
                print(f"{table_name} — {len(columns)} column(s)")
                for column in columns:
                    if isinstance(column, dict):
                        name = column.get("name", column.get("column", ""))
                    else:
                        name = column
                    print(f"  {safe_text(str(name))}")
            continue
        if not rows:
            continue
        remaining = max_display_rows - displayed
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
    if scanned_rows > preview_rows:
        if displayed >= max_display_rows:
            print(
                f"Output limit reached: showing {displayed} of "
                f"{scanned_rows} matching rows (maximum "
                f"{max_display_rows} rows displayed)."
            )
        else:
            print(
                f"Showing {displayed} of {preview_rows} retained preview "
                f"rows; the complete scan covered {scanned_rows} result rows."
            )
    elif preview_rows > displayed:
        print(
            f"Showing {displayed} of {preview_rows} evidence rows; "
            "the complete result remains in the returned evidence/report."
        )


def run() -> None:
    configure_console()
    print()
    print("DB-SENTINEL")
    print(safe_text("───────────"))
    print("Conversational Autonomous Data Analyst")

    try:
        saved_profiles = _saved_connection_profiles()
    except ValueError as error:
        saved_profiles = []
        logger.error("Saved database profiles could not be read: %s", error)
        print(f"Saved database profiles could not be read: {error}")
    saved_organizations = {
        profile["organization_id"]
        for profile in saved_profiles
    }
    default_organization = (
        next(iter(saved_organizations))
        if len(saved_organizations) == 1
        else "local"
    )
    organization_id = prompt_required(
        "Organization ID",
        default_organization,
    )
    if saved_profiles and organization_id not in saved_organizations:
        logger.warning(
            "No saved database profiles exist for selected organization=%s; "
            "saved organizations=%s",
            organization_id,
            sorted(saved_organizations),
        )
        print(
            f"No saved database profiles exist for '{organization_id}'. "
            "Saved profiles are available for: "
            f"{', '.join(sorted(saved_organizations))}. "
            "Restart and enter one of those organization IDs to restore them."
        )

    manager = DatabaseManager()
    session: ConversationSession | None = None

    try:
        restored = _restore_connection_profiles(
            manager,
            organization_id,
            saved_profiles,
        )
        if restored:
            print(
                "Restored saved database connection(s): "
                + ", ".join(restored)
            )
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
            print("  delete   - remove a saved connection from this organization")
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

            if command.lower() == "delete":
                if not connections:
                    print("No database connections to delete.")
                    continue
                selection = input(
                    "Select database number or connection ID to delete: "
                ).strip()
                selected = next(
                    (
                        item
                        for index, item in enumerate(connections, start=1)
                        if item.connection_id == selection
                        or item.name == selection
                        or str(index) == selection
                    ),
                    None,
                )
                if selected is None:
                    print("Database not found.")
                    continue
                confirmation = input(
                    f"Remove saved connection '{selected.name}' "
                    f"({selected.connection_id})? [y/N]: "
                ).strip().lower()
                if confirmation not in {"y", "yes"}:
                    print("Connection was not removed.")
                    continue
                try:
                    delete_saved_connection(
                        manager,
                        organization_id,
                        selected.connection_id,
                    )
                    if session is not None:
                        session.database_connection_ids = tuple(
                            manager.list_connection_ids(organization_id)
                        )
                    print(
                        "Saved connection removed. This does not delete or "
                        "modify data in the remote database."
                    )
                except Exception as error:
                    print(f"Could not remove saved connection: {error}")
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
                    "Ask questions continuously or type /back to choose "
                    "another database."
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
                    if not question:
                        continue

                    try:
                        with analysis_progress():
                            result = analyst.ask(
                                session=session,
                                question=question,
                                connection_id=selected.connection_id,
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
