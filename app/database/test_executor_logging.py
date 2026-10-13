from app.database.executor import SecureQueryExecutor


def test_query_audit_log_includes_executed_sql(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.database.executor.logger.info",
        lambda *args: calls.append(args),
    )

    SecureQueryExecutor._audit(
        event="QUERY_EXECUTED",
        sql="SELECT id\nFROM public.people\nLIMIT 10",
        reason="Query executed successfully.",
        audit_context={
            "organization_id": "org-1",
            "workspace_id": "team-2",
            "connection_id": "db-3",
        },
    )

    assert len(calls) == 1
    assert "organization_id=%s workspace_id=%s connection_id=%s" in calls[0][0]
    assert calls[0][1:4] == ("QUERY_EXECUTED", "org-1", "team-2")
    assert calls[0][4] == "db-3"
    assert calls[0][-1] == "SELECT id FROM public.people LIMIT 10"
