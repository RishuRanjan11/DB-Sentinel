from app.agent.evidence import Evidence
from app.agent.insight_generator import GeneratedInsight
from app.agent.insight_verifier import InsightVerifier
from app.agent.logical_verifier import LogicalVerifier
from app.agent.sql_pipeline import SQLPipeline
from app.security.policies import RequestPolicy


def test_unspecified_ranked_group_question_requires_complete_capped_result():
    question = (
        "Which pathways contain the most FlyBase genes with human "
        "orthologs, and how many are in each?"
    )

    limited = LogicalVerifier._validate_ranking(
        question,
        "SELECT pathway, count(DISTINCT gene_id) AS gene_count "
        "FROM public.membership GROUP BY pathway "
        "ORDER BY gene_count DESC, pathway ASC LIMIT 5",
    )
    complete = LogicalVerifier._validate_ranking(
        question,
        "SELECT pathway, count(DISTINCT gene_id) AS gene_count "
        "FROM public.membership GROUP BY pathway "
        "ORDER BY gene_count DESC, pathway ASC LIMIT 100",
    )

    assert not limited["valid"]
    assert complete["valid"]


def test_explicit_top_n_must_match_limit():
    result = LogicalVerifier._validate_ranking(
        "Compare results and identify the top 5 pathways.",
        "SELECT pathway, count(*) AS n FROM public.membership "
        "GROUP BY pathway ORDER BY n DESC, pathway ASC LIMIT 5",
    )
    assert result["valid"]

    wrong_limit = LogicalVerifier._validate_ranking(
        "Compare results and identify the top 5 pathways.",
        "SELECT pathway, count(*) AS n FROM public.membership "
        "GROUP BY pathway ORDER BY n DESC, pathway ASC LIMIT 100",
    )
    assert not wrong_limit["valid"]

    no_tie_breaker = LogicalVerifier._validate_ranking(
        "Identify the top 5 pathways.",
        "SELECT pathway, count(*) AS n FROM public.membership "
        "GROUP BY pathway ORDER BY n DESC LIMIT 5",
    )
    assert not no_tie_breaker["valid"]


def test_completed_pagination_does_not_report_incomplete_limit_warning():
    evidence = Evidence(
        step_id=1,
        question="Which genes are present?",
        objective="Return the requested gene records.",
        sql="SELECT gene_id FROM public.gene ORDER BY gene_id LIMIT 100",
        rows=[{"gene_id": 1}],
        schema_context=[],
        logical_reason="Verified.",
        confidence=0.9,
        pagination_complete=True,
    )

    decision = InsightVerifier().verify(
        question="Which genes are present?",
        insight=GeneratedInsight(
            summary="The complete ordered result is supported.",
            findings=[],
        ),
        evidence=[evidence],
    )

    assert not any("completeness is limited" in warning for warning in decision.warnings)


def test_grouped_pagination_only_claims_requested_top_n_complete():
    pipeline = SQLPipeline.__new__(SQLPipeline)
    sql = (
        "SELECT pathway, count(*) AS gene_count "
        "FROM public.membership GROUP BY pathway "
        "ORDER BY gene_count DESC, pathway ASC LIMIT 5"
    )
    rows = [{"pathway": f"pathway-{index}", "gene_count": 10 - index}
            for index in range(5)]

    requested_top_n = pipeline.collect_additional_pages(
        connection_id="unused",
        sql=sql,
        schema_context=[],
        initial_rows=rows,
        expected_top_n=5,
    )
    unbounded_ranking = pipeline.collect_additional_pages(
        connection_id="unused",
        sql=sql,
        schema_context=[],
        initial_rows=rows,
    )

    assert requested_top_n["complete"]
    assert not unbounded_ranking["complete"]

def test_collection_pagination_requires_primary_key_tiebreaker():
    pipeline = SQLPipeline.__new__(SQLPipeline)
    result = pipeline.collect_additional_pages(
        connection_id="unused",
        sql=(
            "SELECT id, symbol FROM public.gene "
            "ORDER BY symbol LIMIT 5"
        ),
        schema_context=[{
            "schema": "public",
            "table": "gene",
            "primary_key": ["id"],
        }],
        initial_rows=[
            {"id": index, "symbol": f"gene-{index}"}
            for index in range(5)
        ],
    )

    assert not result["complete"]
    assert "primary-key tie-breaker" in result["reason"]


def test_destructive_ddl_and_sensitive_requests_are_blocked():
    policy = RequestPolicy()
    assert policy.evaluate("Delete all gene rows").allowed is False
    assert policy.evaluate("Alter the gene table").allowed is False
    assert policy.evaluate("Show private sensitive information").allowed is False
