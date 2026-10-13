import json
from pathlib import Path

import pytest
from dotenv import dotenv_values

from app.database.connection import DatabaseConnection
from app.database.adapters.postgresql import PostgreSQLAdapter
from app.main import (
    _connection_from_url,
    _environment_file_path,
    _remove_persisted_connection,
    _persist_connection,
    _restore_connection_profiles,
    _saved_connection_profiles,
    create_database,
    delete_saved_connection,
)
from app.database.manager import DatabaseManager


def connection_from(url):
    return _connection_from_url(
        connection_url=url,
        connection_id="test",
        organization_id="org",
        workspace_id="workspace",
        name="Test database",
    )


def test_postgres_url_parses_credentials_and_ssl_without_logging_url():
    connection = connection_from(
        "postgresql://reader:p%40ss@db.example.test:5432/analytics"
        "?sslmode=require"
    )

    assert connection.database_type == "postgresql"
    assert connection.host == "db.example.test"
    assert connection.port == 5432
    assert connection.database_name == "analytics"
    assert connection.username == "reader"
    assert connection.password == "p@ss"
    assert connection.ssl_enabled


def test_postgres_url_accepts_at_sign_in_unescaped_password():
    connection = connection_from(
        "postgresql://reader:part1@part2@db.example.test:5432/analytics"
    )

    assert connection.host == "db.example.test"
    assert connection.username == "reader"
    assert connection.password == "part1@part2"


def test_postgresql_adapter_hides_supabase_managed_schemas_only_for_supabase():
    supabase = PostgreSQLAdapter(
        connection_from(
            "postgresql://reader:pass@aws-0-ap-southeast-2.pooler.supabase.com/"
            "postgres"
        )
    )
    other_postgres = PostgreSQLAdapter(
        connection_from("postgresql://reader:pass@db.example.test/analytics")
    )

    assert "vault" in supabase._managed_schema_filter("schema_name")
    assert supabase._managed_schema_filter("schema_name").startswith(" AND ")
    assert other_postgres._managed_schema_filter("schema_name") == ""


@pytest.mark.parametrize(
    ("url", "database_type", "ssl_enabled"),
    [
        ("postgres://reader:pw@db.example.test/reporting?sslmode=disable", "postgresql", False),
        ("mysql+pymysql://reader:pw@db.example.test:3307/reporting", "mysql", True),
        ("mssql://reader:pw@db.example.test/reporting?driver=ODBC+Driver+18+for+SQL+Server", "sqlserver", True),
        ("sqlserver://reader:pw@db.example.test/reporting?Encrypt=no", "sqlserver", False),
    ],
)
def test_relational_connection_url_schemes(url, database_type, ssl_enabled):
    connection = connection_from(url)

    assert connection.database_type == database_type
    assert connection.database_name == "reporting"
    assert connection.ssl_enabled is ssl_enabled


def test_mongodb_uri_is_kept_intact_and_database_is_extracted():
    url = "mongodb+srv://reader:p%40ss@cluster.example.test/analytics?tls=true"
    connection = connection_from(url)

    assert connection.database_type == "mongodb"
    assert connection.host == url
    assert connection.database_name == "analytics"
    assert connection.ssl_enabled


def test_mongodb_uri_parses_url_encoded_reserved_password_characters():
    url = (
        "mongodb+srv://reader:p%40ss%23word@cluster.example.test/"
        "sample_analytics?appName=DBSentinel"
    )
    connection = connection_from(url)

    assert connection.host == url
    assert connection.database_name == "sample_analytics"


def test_mongodb_uri_without_path_uses_prompted_database_name_and_normalizes_uri():
    url = (
        "mongodb+srv://reader:secret@cluster.example.test/"
        "?appName=DBSentinel"
    )
    connection = _connection_from_url(
        connection_url=url,
        connection_id="test",
        organization_id="org",
        workspace_id="workspace",
        name="Test database",
        database_name="sample_analytics",
    )

    assert connection.database_name == "sample_analytics"
    assert connection.host == (
        "mongodb+srv://reader:secret@cluster.example.test/"
        "sample_analytics?appName=DBSentinel"
    )


def test_mongodb_uri_explains_unencoded_password_fragment():
    with pytest.raises(ValueError, match="password was not URL-encoded as %23"):
        connection_from(
            "mongodb+srv://reader:p%23ss#word@cluster.example.test/"
            "sample_analytics?appName=DBSentinel"
        )


def test_mongodb_uri_explains_missing_database_path():
    with pytest.raises(ValueError, match="database name is required"):
        connection_from(
            "mongodb+srv://reader:secret@cluster.example.test/"
            "?appName=DBSentinel"
        )


def test_url_mode_prompts_for_mongodb_database_when_uri_has_no_path(monkeypatch):
    monkeypatch.delenv("DB_SENTINEL_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_URL", raising=False)
    inputs = iter([
        "url",
        "",
        "",
        "",
        "sample_analytics",
    ])
    captured = {}
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "app.main.getpass.getpass",
        lambda _: (
            "mongodb+srv://reader:secret@cluster.example.test/"
            "?appName=DBSentinel"
        ),
    )
    monkeypatch.setattr(
        "app.main._register_and_test",
        lambda _manager, connection: (
            captured.setdefault("connection", connection)
            and connection.connection_id
        ),
    )

    connection_id = create_database(DatabaseManager(), "org")

    assert connection_id.startswith("org-")
    assert captured["connection"].database_name == "sample_analytics"
    assert captured["connection"].host.endswith(
        "/sample_analytics?appName=DBSentinel"
    )


def test_connection_url_requires_host_and_database():
    with pytest.raises(ValueError, match="host and database name"):
        connection_from("postgresql://reader@db.example.test")


def test_configured_url_does_not_prompt_for_the_url_or_dictionary(monkeypatch):
    monkeypatch.setenv(
        "DB_SENTINEL_DATABASE_URL",
        "postgresql://reader:p%40ss@db.example.test:5432/postgres",
    )
    inputs = iter(["url", "", "", ""])
    captured = {}
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "app.main.getpass.getpass",
        lambda _: pytest.fail("Configured URL should not prompt for secret input."),
    )
    monkeypatch.setattr(
        "app.main._register_and_test",
        lambda _manager, connection: captured.setdefault("connection", connection)
        and connection.connection_id,
    )

    connection_id = create_database(DatabaseManager(), "org")

    assert connection_id.startswith("org-")
    assert captured["connection"].database_name == "postgres"
    assert captured["connection"].name == "Database"


def test_failed_url_connection_can_edit_only_the_bad_field(monkeypatch):
    monkeypatch.setenv(
        "DB_SENTINEL_DATABASE_URL",
        "postgresql://reader:p%40ss@wrong.example.test:5432/analytics",
    )
    inputs = iter(["url", "", "", "", "1"])
    urls = iter([
        "postgresql://reader:p%40ss@correct.example.test:5432/analytics",
    ])
    attempts = []
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("app.main.getpass.getpass", lambda _: next(urls))

    def register(_manager, connection):
        attempts.append(connection.host)
        if connection.host == "wrong.example.test":
            raise RuntimeError("could not connect")
        return connection.connection_id

    monkeypatch.setattr("app.main._register_and_test", register)

    result = create_database(DatabaseManager(), "org")

    assert result.startswith("org-")
    assert attempts == ["wrong.example.test", "correct.example.test"]


def test_failed_field_connection_can_edit_only_the_bad_field(monkeypatch):
    monkeypatch.delenv("DB_DISPLAY_NAME", raising=False)
    monkeypatch.delenv("DB_TYPE", raising=False)
    monkeypatch.delenv("DB_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.delenv("DB_DATABASE_NAME", raising=False)
    monkeypatch.delenv("DB_NAME", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    monkeypatch.delenv("DB_SENTINEL_DATABASE_URL", raising=False)
    inputs = iter([
        "fields", "", "", "", "", "wrong.example.test", "", "analytics", "reader",
        "", "5", "correct.example.test",
    ])
    passwords = iter(["secret"])
    attempts = []
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("app.main.getpass.getpass", lambda _: next(passwords))

    def register(_manager, connection):
        attempts.append(connection.host)
        if connection.host == "wrong.example.test":
            raise RuntimeError("could not connect")
        return connection.connection_id

    monkeypatch.setattr("app.main._register_and_test", register)

    result = create_database(DatabaseManager(), "org")

    assert result.startswith("org-")
    assert attempts == ["wrong.example.test", "correct.example.test"]


def test_postgresql_connection_failure_is_available_without_exposing_password(
    caplog,
    monkeypatch,
):
    connection = DatabaseConnection(
        connection_id="test",
        organization_id="org",
        workspace_id="workspace",
        name="Test",
        database_type="postgresql",
        host="db.example.test",
        database_name="analytics",
        username="reader",
        password="secret",
    )
    adapter = PostgreSQLAdapter(connection)

    def fail_connect():
        raise RuntimeError("authentication failed for secret")

    monkeypatch.setattr(adapter, "connect", fail_connect)

    assert not adapter.test_connection()
    assert "authentication failed for [REDACTED]" in adapter.last_connection_error
    assert "secret" not in caplog.text


def test_database_connection_is_persisted_and_restored_from_local_env(
    tmp_path,
    monkeypatch,
):
    env_path = tmp_path / ".env"
    monkeypatch.setenv("DB_SENTINEL_ENV_FILE", str(env_path))
    monkeypatch.delenv("DB_SENTINEL_CONNECTIONS_JSON", raising=False)
    connection = _connection_from_url(
        "postgresql://reader:p%40ss@db.example.test:5432/analytics",
        "analytics",
        "org",
        "workspace",
        "Analytics",
    )

    _persist_connection(connection)
    profiles = _saved_connection_profiles()
    assert len(profiles) == 1
    assert profiles[0]["name"] == "Analytics"
    assert "p%40ss" in profiles[0]["url"]
    assert env_path.exists()

    manager = DatabaseManager()
    restored = _restore_connection_profiles(manager, "org")
    assert restored == ["Analytics"]
    assert manager.get_connection("analytics").database_name == "analytics"
    manager.close_all()


def test_saving_profiles_preserves_other_connections(
    tmp_path,
    monkeypatch,
):
    env_path = tmp_path / ".env"
    monkeypatch.setenv("DB_SENTINEL_ENV_FILE", str(env_path))
    monkeypatch.delenv("DB_SENTINEL_CONNECTIONS_JSON", raising=False)
    first = _connection_from_url(
        "postgresql://reader:p%40ss@db.example.test:5432/first",
        "first",
        "org",
        "team-a",
        "First",
    )
    second = _connection_from_url(
        "postgresql://reader:p%40ss@db.example.test:5432/second",
        "second",
        "org",
        "team-b",
        "Second",
    )

    _persist_connection(first)
    _persist_connection(second)

    profiles = _saved_connection_profiles()
    assert {profile["connection_id"] for profile in profiles} == {
        "first",
        "second",
    }
    saved_value = dotenv_values(env_path)["DB_SENTINEL_CONNECTIONS_JSON"]
    assert {
        profile["connection_id"] for profile in json.loads(saved_value)
    } == {"first", "second"}


def test_saved_profiles_are_restored_from_file_not_stale_environment(
    tmp_path,
    monkeypatch,
):
    env_path = tmp_path / ".env"
    monkeypatch.setenv("DB_SENTINEL_ENV_FILE", str(env_path))
    monkeypatch.delenv("DB_SENTINEL_CONNECTIONS_JSON", raising=False)
    connection = _connection_from_url(
        "mongodb+srv://user:password@example.test/analytics",
        "mongo",
        "org",
        "workspace",
        "MongoDB",
    )
    _persist_connection(connection)

    monkeypatch.setenv(
        "DB_SENTINEL_CONNECTIONS_JSON",
        json.dumps([{"connection_id": "stale"}]),
    )

    profiles = _saved_connection_profiles()

    assert [profile["connection_id"] for profile in profiles] == ["mongo"]
    manager = DatabaseManager()
    assert _restore_connection_profiles(manager, "org", profiles) == ["MongoDB"]
    manager.close_all()


def test_environment_profile_path_is_shared_for_relative_override(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DB_SENTINEL_ENV_FILE", "config/.env")

    assert _environment_file_path() == (
        Path(__file__).resolve().parents[1] / "config" / ".env"
    )


def test_delete_saved_connection_removes_only_selected_profile(
    tmp_path,
    monkeypatch,
):
    env_path = tmp_path / ".env"
    monkeypatch.setenv("DB_SENTINEL_ENV_FILE", str(env_path))
    monkeypatch.delenv("DB_SENTINEL_CONNECTIONS_JSON", raising=False)
    first = _connection_from_url(
        "postgresql://reader:p%40ss@db.example.test:5432/first",
        "first",
        "org",
        "team-a",
        "First",
    )
    second = _connection_from_url(
        "postgresql://reader:p%40ss@db.example.test:5432/second",
        "second",
        "other-org",
        "team-b",
        "Second",
    )
    _persist_connection(first)
    _persist_connection(second)
    manager = DatabaseManager()
    manager.register(first)
    manager.register(second)

    assert delete_saved_connection(manager, "org", "first")
    assert manager.list_connection_ids("org") == []
    assert [p["connection_id"] for p in _saved_connection_profiles()] == [
        "second"
    ]
    assert not _remove_persisted_connection("missing", "org")
    manager.close_all()
