from app.agent.query_repair import QueryRepairAnalyzer


def test_execution_error_feedback_diagnoses_unknown_column_without_bypassing_schema():
    feedback = QueryRepairAnalyzer.create_feedback(
        question="Compare annual sales by region.",
        query_language="postgres",
        failure_stage="query execution",
        reason='column "region_name" does not exist',
    )

    assert "Failure category: query syntax, identifier, or type error." in feedback
    assert "exact live-schema identifiers" not in feedback
    assert "reported database error and live schema" in feedback
    assert "Compare annual sales by region." in feedback
    assert "Do not weaken security" in feedback


def test_duplicate_query_error_requests_a_materially_different_repair():
    feedback = QueryRepairAnalyzer.create_feedback(
        question="List all genes.",
        query_language="SQL",
        failure_stage="security/schema validation",
        reason="The generated SQL is identical to a previous failed attempt.",
    )

    assert "Failure category: repeated query." in feedback
    assert "actual construction" in feedback


def test_transient_execution_error_does_not_encourage_semantic_changes():
    feedback = QueryRepairAnalyzer.create_feedback(
        question="Show TB totals by year.",
        query_language="mongodb",
        failure_stage="query execution",
        reason="Database connection temporarily unavailable.",
        previous_feedback="Return one row per requested year.",
    )

    assert "Failure category: database or transient execution failure." in feedback
    assert "do not alter the requested semantics" in feedback
    assert "Earlier repair constraints that still apply:" in feedback
    assert "Return one row per requested year." in feedback
