from unittest.mock import patch

from app.database.connection import DatabaseConnection
from app.main import (
    _persist_connection,
    analysis_progress,
    print_result,
    run,
)


def test_print_result_displays_all_live_column_names(capsys):
    print_result({
        "status": "verified",
        "question": "Tell me all columns name in this database",
        "insight": {"summary": "Live database metadata was inspected."},
        "evidence": [{
            "type": "database_metadata",
            "tables": [{
                "schema": "public",
                "table": "tb_data",
                "columns": [
                    {"name": "country", "data_type": "text"},
                    {"name": "year", "data_type": "integer"},
                ],
            }],
        }],
    })

    output = capsys.readouterr().out
    assert "public.tb_data — 2 column(s)" in output
    assert "  country" in output
    assert "  year" in output


def test_print_result_shows_up_to_100_rows_and_reports_output_limit(capsys):
    print_result({
        "status": "verified",
        "question": "Name all unique genes.",
        "insight": {
            "summary": (
                "The complete scan found 2000 matching results. "
                "A bounded preview is retained as evidence."
            ),
            "findings": [],
        },
        "evidence": [{
            "step_id": 1,
            "rows": [{"name": f"gene-{index}"} for index in range(105)],
            "total_rows": 2000,
        }],
    })

    output = capsys.readouterr().out
    assert "2000 matching results" in output
    assert "gene-99" in output
    assert "gene-100" not in output
    assert "Output limit reached: showing 100 of 2000 matching rows" in output


def test_print_result_displays_recommendations_and_limitations(capsys):
    print_result({
        "status": "verified",
        "question": "Compare the metric over time.",
        "insight": {
            "summary": "The verified results differ by period.",
            "findings": ["The later period has a lower measured value."],
            "recommendations": ["Review the affected segment."],
            "uncertainties": ["A geographic breakdown was unavailable."],
        },
        "evidence": [],
    })

    output = capsys.readouterr().out
    assert "Recommended actions" in output
    assert "Review the affected segment." in output
    assert "Limitations" in output
    assert "A geographic breakdown was unavailable." in output


def test_print_result_displays_grammar_corrected_interpretation(capsys):
    print_result({
        "status": "verified",
        "question": "Total amount change in sales from 2015 to 2017",
        "normalized_question": (
            "What was the total change in sales from 2015 to 2017?"
        ),
        "insight": {"summary": "Sales increased by 46."},
        "evidence": [],
    })

    output = capsys.readouterr().out
    assert (
        "Interpreted question: What was the total change in sales "
        "from 2015 to 2017?"
    ) in output


def test_analysis_progress_shows_and_clears_waiting_message(capsys):
    with analysis_progress():
        print("work completed")

    output = capsys.readouterr().out
    assert "Analyzing your question..." in output
    assert "Analysis finished." in output


def test_startup_explains_when_saved_profiles_belong_to_another_org(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("DB_SENTINEL_ENV_FILE", str(tmp_path / ".env"))
    monkeypatch.delenv("DB_SENTINEL_CONNECTIONS_JSON", raising=False)
    _persist_connection(DatabaseConnection(
        connection_id="mongo",
        organization_id="google",
        workspace_id="default",
        name="MongoDB",
        database_type="mongodb",
        host="mongodb+srv://user:password@example.test/analytics",
        database_name="analytics",
    ))

    with patch("builtins.input", side_effect=["local", "quit"]):
        run()

    output = capsys.readouterr().out
    assert "No saved database profiles exist for 'local'." in output
    assert "Saved profiles are available for: google." in output
