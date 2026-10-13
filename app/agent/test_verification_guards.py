import json

import pytest

from app.agent.evidence import Evidence
from app.agent.insight_generator import GeneratedInsight
from app.agent.insight_verifier import InsightVerifier
from app.agent.logical_verifier import (
    LogicalVerificationDecision,
    LogicalVerifier,
)
from app.agent.planner import (
    InvestigationPlan,
    QuestionPlanner,
    SubQuestion,
)
from app.agent.sql_pipeline import SQLPipeline
from app.security.policies import RequestPolicy
from app.agent.investigation import (
    InvestigationEngine,
    _expected_top_n,
)


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


def test_stable_pagination_supports_distinct_and_group_keys():
    pipeline = SQLPipeline.__new__(SQLPipeline)

    assert pipeline._has_stable_pagination_order(
        sql=(
            "SELECT DISTINCT name FROM public.gene "
            "ORDER BY name ASC LIMIT 100"
        ),
        schema_context=[{
            "schema": "public",
            "table": "gene",
            "columns": [{"name": "name"}],
        }],
        dialect="postgres",
    )
    assert pipeline._has_stable_pagination_order(
        sql=(
            "SELECT country, year, SUM(cases) AS total "
            "FROM public.observations GROUP BY country, year "
            "ORDER BY country ASC, year ASC LIMIT 100"
        ),
        schema_context=[],
        dialect="postgres",
    )
    assert not pipeline._has_stable_pagination_order(
        sql=(
            "SELECT country, year, SUM(cases) AS total "
            "FROM public.observations GROUP BY country, year "
            "ORDER BY total DESC LIMIT 100"
        ),
        schema_context=[],
        dialect="postgres",
    )

    assert pipeline._has_stable_pagination_order(
        sql=(
            "SELECT DISTINCT id, symbol, name FROM dataclass.genev2 "
            "ORDER BY id LIMIT 100"
        ),
        schema_context=[{
            "schema": "dataclass",
            "table": "genev2",
            "primary_key": ["id"],
            "columns": [
                {"name": "id"},
                {"name": "symbol"},
                {"name": "name"},
            ],
        }],
        dialect="postgres",
    )


def test_pagination_fetches_every_page_without_a_fixed_page_cap(monkeypatch):
    from types import SimpleNamespace
    import app.agent.sql_pipeline as sql_pipeline_module

    values = [f"gene-{index:04d}" for index in range(2001)]

    class FakeManager:
        def get_connection(self, connection_id):
            return SimpleNamespace(
                database_type="postgresql",
                organization_id="org",
                workspace_id="workspace",
            )

        def get(self, connection_id, organization_id=None):
            return object()

    class FakeValidator:
        def __init__(self, **kwargs):
            pass

        def validate(self, sql):
            return SimpleNamespace(allowed=True, reason="")

    class FakeExecutor:
        MAX_ROWS = 100

        def __init__(self, **kwargs):
            pass

        def execute(self, sql):
            import re

            offset = int(re.search(r"\bOFFSET\s+(\d+)", sql, re.I).group(1))
            page = values[offset:offset + 2]
            return SimpleNamespace(
                success=True,
                rows=[{"name": value} for value in page],
                error=None,
            )

    monkeypatch.setattr(
        sql_pipeline_module,
        "SQLSecurityValidator",
        FakeValidator,
    )
    monkeypatch.setattr(
        sql_pipeline_module,
        "SecureQueryExecutor",
        FakeExecutor,
    )

    pipeline = SQLPipeline.__new__(SQLPipeline)
    pipeline.database_manager = FakeManager()
    first_page = [{"name": value} for value in values[:2]]
    result = pipeline.collect_additional_pages(
        connection_id="test",
        organization_id="org",
        sql=(
            "SELECT DISTINCT name FROM public.gene "
            "ORDER BY name ASC LIMIT 2"
        ),
        schema_context=[{
            "schema": "public",
            "table": "gene",
            "columns": [{"name": "name"}],
        }],
        initial_rows=first_page,
        page_size=2,
    )

    assert result["complete"]
    assert result["pages"] == 1001
    assert len(result["rows"]) == 100
    assert result["total_rows"] == len(values)
    assert [row["name"] for row in result["rows"]] == values[:100]


def test_complete_large_collection_reports_scanned_count_and_bounded_preview():
    from app.agent.insight_generator import InsightGenerator

    evidence = Evidence(
        step_id=1,
        question="Name all unique genes.",
        objective="Return all unique gene names.",
        sql=(
            "SELECT DISTINCT name FROM public.gene "
            "ORDER BY name LIMIT 100"
        ),
        rows=[{"name": f"gene-{index}"} for index in range(100)],
        schema_context=[],
        logical_reason="The full ordered result was scanned.",
        confidence=0.95,
        pagination_complete=True,
        total_rows=40905,
    )

    insight = InsightGenerator._large_complete_collection(
        "Name all unique genes.",
        [evidence],
    )

    assert insight is not None
    assert "40905 matching results" in insight.summary
    assert "bounded preview is retained" in insight.summary


def test_scanned_result_count_is_valid_evidence_for_collection_summary():
    evidence = Evidence(
        step_id=1,
        question="Name all unique genes.",
        objective="Return all unique gene names.",
        sql="SELECT DISTINCT name FROM public.gene ORDER BY name LIMIT 100",
        rows=[{"name": f"gene-{index}"} for index in range(100)],
        schema_context=[],
        logical_reason="The full ordered result was scanned.",
        confidence=0.95,
        pagination_complete=True,
        total_rows=40905,
    )

    decision = InsightVerifier().verify(
        question="Name all unique genes.",
        insight=GeneratedInsight(
            summary=(
                "The complete scan found 40905 matching results. "
                "A preview of 100 rows is retained as evidence."
            ),
            findings=[],
        ),
        evidence=[evidence],
    )

    assert decision.verified
    assert not decision.unsupported_findings


def test_collection_relevance_verifier_treats_limit_as_page_bound():
    from app.agent.logical_verifier import LogicalVerificationDecision

    class CollectionRouter:
        def __init__(self):
            self.system_instruction = ""
            self.user_prompt = ""

        def generate(self, **kwargs):
            self.system_instruction = kwargs["system_instruction"]
            self.user_prompt = kwargs["user_prompt"]
            return LogicalVerificationDecision(
                relevant=True,
                confidence=0.9,
                reason="The page contains the requested gene records.",
            )

    router = CollectionRouter()
    decision = LogicalVerifier(model_router=router).verify(
        question="List all unique genes.",
        objective="Return all unique gene names.",
        sql=(
            "SELECT DISTINCT id, symbol, name FROM dataclass.genev2 "
            "ORDER BY id LIMIT 100"
        ),
        schema_context=[{
            "schema": "dataclass",
            "table": "genev2",
            "primary_key": ["id"],
            "columns": [
                {"name": "id"},
                {"name": "symbol"},
                {"name": "name"},
            ],
        }],
        rows=[{"id": 1, "symbol": "gene-1", "name": "Gene 1"}],
        pagination_complete=True,
    )

    assert decision.relevant
    assert "per-page fetch bound" in router.system_instruction
    assert "checked separately" in router.user_prompt


def test_full_collection_query_returns_one_bounded_page_with_total_count():
    from app.agent.sql_pipeline import SQLPipeline
    from app.security.sql_validator import SQLSecurityValidator

    query = SQLPipeline._build_single_query_collection_scan(
        "SELECT DISTINCT id, symbol, name FROM dataclass.genev2 "
        "ORDER BY id ASC LIMIT 100",
        "postgres",
    )

    assert query is not None
    assert "COUNT(*) OVER" in query
    assert query.endswith("LIMIT 100")
    assert "FROM (SELECT DISTINCT id, symbol, name FROM dataclass.genev2)" in query
    assert "ORDER BY __db_sentinel_collection.id ASC" in query

    validator = SQLSecurityValidator(
        allowed_tables={("dataclass", "genev2")},
        schema_context=[{
            "schema": "dataclass",
            "table": "genev2",
            "primary_key": ["id"],
            "columns": [
                {"name": "id"},
                {"name": "symbol"},
                {"name": "name"},
            ],
        }],
    )
    assert validator.validate(query).allowed


def test_single_query_collection_scan_extracts_count_and_only_preview_rows():
    from app.agent.sql_pipeline import SQLPipeline

    source_rows = [
        {
            "id": index,
            "name": f"gene-{index}",
            SQLPipeline.COLLECTION_TOTAL_ALIAS: 2000,
        }
        for index in range(100)
    ]

    preview, scan = SQLPipeline._extract_collection_scan_total(source_rows)

    assert len(preview) == 100
    assert preview[0] == {"id": 0, "name": "gene-0"}
    assert SQLPipeline.COLLECTION_TOTAL_ALIAS not in preview[0]
    assert scan == {
        "complete": True,
        "mode": "single_query",
        "total_rows": 2000,
    }


def test_implicit_maximum_query_returns_all_ties_with_bounded_preview():
    from app.agent.sql_pipeline import SQLPipeline
    from app.security.sql_validator import SQLSecurityValidator

    source_sql = (
        "SELECT gene_id, MAX(diopt_score) AS diopt_score "
        "FROM dataclass_relationship.gene_human_orthologv2 "
        "GROUP BY gene_id "
        "ORDER BY diopt_score DESC, gene_id ASC LIMIT 100"
    )
    tie_question = (
        "Which genes have maximum relevance with human ortholog? "
        "Return all entities tied at the maximum requested metric. "
        "Order by the requested metric DESC then gene identifier ASC. "
        "Keep the query bounded with LIMIT 100."
    )
    assert _expected_top_n(tie_question) is None
    assert SQLPipeline._is_implicit_extreme_tie_request(tie_question)
    assert LogicalVerifier._validate_ranking(
        tie_question,
        source_sql,
    )["valid"]

    query = SQLPipeline._build_single_query_extreme_tie_scan(
        source_sql,
        "postgres",
    )

    assert query is not None
    assert "MAX(" in query
    assert "COUNT(*) OVER" in query
    assert "LIMIT 100" in query

    validator = SQLSecurityValidator(
        allowed_tables={
            ("dataclass_relationship", "gene_human_orthologv2")
        },
        schema_context=[{
            "schema": "dataclass_relationship",
            "table": "gene_human_orthologv2",
            "columns": [
                {"name": "gene_id"},
                {"name": "diopt_score"},
            ],
        }],
        dialect="postgres",
    )
    assert validator.validate(query).allowed


def test_all_ties_summary_reports_exact_count_and_verifies_bounded_preview():
    from app.agent.insight_generator import InsightGenerator

    evidence = Evidence(
        step_id=1,
        question="Which genes have maximum relevance with human ortholog?",
        objective=(
            "Return all entities tied at the maximum requested metric."
        ),
        sql=(
            "SELECT gene_id, MAX(diopt_score) AS max_diopt_score "
            "FROM dataclass_relationship.gene_human_orthologv2 "
            "GROUP BY gene_id ORDER BY max_diopt_score DESC, gene_id ASC "
            "LIMIT 100"
        ),
        rows=[
            {"gene_id": f"FBgn{index:07d}", "max_diopt_score": 14}
            for index in range(100)
        ],
        schema_context=[],
        logical_reason="The query and ranking were verified.",
        confidence=0.98,
        pagination_complete=True,
        total_rows=2222,
        all_extreme_ties=True,
        ranking_metric="max_diopt_score",
    )

    insight = InsightGenerator._deterministic_extreme_ties(
        evidence.question,
        [evidence],
    )

    assert insight is not None
    assert "maximum diopt score is 14" in insight.summary.lower()
    assert "2222 entities" in insight.summary
    assert "Showing 100 of 2222" in insight.summary

    decision = InsightVerifier().verify(
        question=evidence.question,
        insight=insight,
        evidence=[evidence],
    )

    assert decision.verified
    assert not decision.unsupported_findings


def test_empty_extreme_tie_scan_reports_no_determinable_extreme():
    from app.agent.insight_generator import InsightGenerator

    evidence = Evidence(
        step_id=1,
        question="Which genes have maximum relevance?",
        objective="Return every entity tied at the maximum.",
        sql="SELECT gene_id, score FROM public.gene_score LIMIT 100",
        rows=[],
        schema_context=[],
        logical_reason="The complete tie scan returned no rows.",
        confidence=0.98,
        pagination_complete=True,
        total_rows=0,
        all_extreme_ties=True,
        ranking_metric="max_score",
    )

    insight = InsightGenerator._deterministic_extreme_ties(
        evidence.question,
        [evidence],
    )
    assert insight is not None
    assert "no entities with a measurable score" in insight.summary.lower()

    decision = InsightVerifier().verify(
        question=evidence.question,
        insight=insight,
        evidence=[evidence],
    )
    assert decision.verified


def test_explicit_top_n_phrase_completes_ordered_non_grouped_query():
    question = (
        "Which 5 countries had the highest pulmonary TB cases in 2023?"
    )
    sql = (
        "SELECT country, cases FROM public.tb_dr_surveillance "
        "ORDER BY cases DESC, country ASC LIMIT 5"
    )
    rows = [
        {"country": f"country-{index}", "cases": 100 - index}
        for index in range(5)
    ]

    assert _expected_top_n(question) == 5
    assert LogicalVerifier._validate_ranking(question, sql)["valid"]

    result = SQLPipeline.__new__(SQLPipeline).collect_additional_pages(
        connection_id="unused",
        sql=sql,
        schema_context=[],
        initial_rows=rows,
        expected_top_n=_expected_top_n(question),
    )

    assert result["complete"]
    assert result["rows"] == rows
    assert result["pages"] == 1


def test_direct_top_n_case_ranking_uses_single_analysis_step():
    assert QuestionPlanner._is_direct_ranking_lookup(
        "Which 5 countries had the highest pulmonary TB cases in 2023?"
    )
    assert QuestionPlanner._is_direct_ranking_lookup(
        "Which region generates minimum sales?"
    )
    assert QuestionPlanner._is_direct_ranking_lookup(
        "Which department has the lowest average handling time?"
    )

def test_contrastive_multi_measure_ranking_is_one_complete_plan_step():
    question = (
        "Which products sell the most units but generate relatively low revenue?"
    )

    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError("Compound ranking requests are planned directly.")

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(question)

    assert QuestionPlanner._is_direct_ranking_lookup(question)
    assert len(plan.sub_questions) == 1
    objective = plan.sub_questions[0].objective.lower()
    assert "multi-measure contrast" in objective
    assert "every measure explicitly requested" in objective
    assert "median" in objective
    assert "arbitrary result limit" in objective

def test_contrast_without_high_low_measure_language_keeps_single_metric_plan():
    question = "Which regions generated the most sales but exclude archived records?"

    plan = QuestionPlanner._direct_ranking_plan(question, question)

    assert "multi-measure contrast" not in plan.sub_questions[0].objective.lower()

@pytest.mark.parametrize(
    ("question", "expected_direction", "expected_tie_rule"),
    [
        (
            "Which region generates maximum sales?",
            "highest to lowest",
            "tied at the requested maximum",
        ),
        (
            "Which region generates minimum sales?",
            "lowest to highest",
            "tied at the requested minimum",
        ),
    ],
)
def test_direct_grouped_ranking_plan_preserves_requested_direction(
    question,
    expected_direction,
    expected_tie_rule,
):
    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "A direct metric ranking should not invoke plan generation."
            )

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(question)

    assert len(plan.sub_questions) == 1
    assert expected_direction in plan.sub_questions[0].objective
    assert expected_tie_rule in plan.sub_questions[0].objective


def test_simple_period_scoped_count_uses_one_required_fast_step():
    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError("A direct count should not need planner generation.")

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(
        "How many cases are in India in 2022?"
    )

    assert len(plan.sub_questions) == 1
    assert plan.sub_questions[0].optional is False
    assert "single aggregate result" in plan.sub_questions[0].objective


def test_scalar_count_aggregate_is_verified_without_ranking_model_inference():
    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError("A verified scalar count needs no LLM guess.")

    result = LogicalVerifier(model_router=UnusedRouter()).verify(
        question="How many cases are in India in 2022?",
        objective=(
            "Return the requested count for the user's stated filters "
            "as a single aggregate result. Do not add comparisons or "
            "segment analyses unless the user asks for them."
        ),
        sql=(
            "SELECT SUM(COALESCE(new_cases, 0) + COALESCE(return_cases, 0)) "
            "AS total_cases FROM public.observations "
            "WHERE country = 'India' AND year = 2022 LIMIT 1"
        ),
        schema_context=[],
        rows=[{"total_cases": 100}],
        pagination_complete=True,
    )

    assert result.relevant
    assert result.confidence >= 0.75
    assert "ungrouped scalar aggregate" in result.reason
    assert _expected_top_n("How many cases are in India in 2022?") is None


def test_count_breakdowns_and_comparisons_still_use_analytical_planning():
    assert not QuestionPlanner._is_simple_count_lookup(
        "How many cases are there per country?"
    )
    assert not QuestionPlanner._is_simple_count_lookup(
        "How many cases were there in 2022 compared with 2021?"
    )


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
    assert "uniquely identifies rows or groups" in result["reason"]


def test_singular_comparative_question_uses_ordered_top_one():
    question = "Which country is more affected?"
    assert _expected_top_n(question) == 1

    ranking = LogicalVerifier._validate_ranking(
        question,
        "SELECT country, SUM(cases) AS total_cases "
        "FROM public.tb_data GROUP BY country "
        "ORDER BY total_cases DESC, country ASC LIMIT 1",
    )
    assert ranking["valid"]

    pipeline = SQLPipeline.__new__(SQLPipeline)
    complete = pipeline.collect_additional_pages(
        connection_id="unused",
        sql=(
            "SELECT country, SUM(cases) AS total_cases "
            "FROM public.tb_data GROUP BY country "
            "ORDER BY total_cases DESC, country ASC LIMIT 1"
        ),
        schema_context=[],
        initial_rows=[{"country": "Example", "total_cases": 10}],
        expected_top_n=_expected_top_n(question),
    )
    assert complete["complete"]


def test_implicit_maximum_question_requires_only_the_ordered_first_result():
    question = "Which country have maximum number of cases?"
    sql = (
        "SELECT country, SUM(value) AS total_cases "
        "FROM public.observations GROUP BY country "
        "ORDER BY total_cases DESC, country ASC LIMIT 1"
    )

    assert _expected_top_n(question) == 1
    assert LogicalVerifier._validate_ranking(question, sql)["valid"]
    assert not LogicalVerifier._validate_ranking(
        question,
        sql.replace("LIMIT 1", "LIMIT 100"),
    )["valid"]


def test_relevance_ranking_question_is_not_split_into_dependent_steps():
    class RankingRouter:
        def generate(self, **kwargs):
            return InvestigationPlan(
                original_question=(
                    "Which genes have maximum relevence with human ortholog"
                ),
                goal="Find genes with maximum human-ortholog relevance.",
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question="Find genes with human orthologs.",
                        objective="Retrieve genes and ortholog metrics.",
                    ),
                    SubQuestion(
                        id=2,
                        question="Rank genes by relevance.",
                        objective="Return the genes with maximum relevance.",
                        depends_on=[1],
                    ),
                ],
            )

    question = "Which genes have maximum relevence with human ortholog"
    plan = QuestionPlanner(model_router=RankingRouter()).create_plan(question)

    assert len(plan.sub_questions) == 1
    assert plan.sub_questions[0].question == question
    assert not plan.sub_questions[0].depends_on
    assert "highest to lowest" in plan.sub_questions[0].objective
    assert "all entities tied" in plan.sub_questions[0].objective.lower()


def test_ungrouped_aggregate_is_complete_without_row_pagination():
    pipeline = SQLPipeline.__new__(SQLPipeline)
    result = pipeline.collect_additional_pages(
        connection_id="unused",
        sql=(
            "SELECT SUM(first_value + second_value) AS total "
            "FROM public.observations WHERE country = 'Example' LIMIT 1"
        ),
        schema_context=[],
        initial_rows=[{"total": 42}],
    )

    assert result["complete"]
    assert result["rows"] == [{"total": 42}]
    assert result["pages"] == 1


def test_window_aggregate_still_requires_collection_pagination():
    pipeline = SQLPipeline.__new__(SQLPipeline)
    result = pipeline.collect_additional_pages(
        connection_id="unused",
        sql=(
            "SELECT id, SUM(value) OVER (ORDER BY id) AS running_total "
            "FROM public.observations ORDER BY id LIMIT 5"
        ),
        schema_context=[{
            "schema": "public",
            "table": "observations",
            "primary_key": ["id"],
        }],
        initial_rows=[{"id": index, "running_total": index} for index in range(5)],
    )

    assert not result["complete"]


def test_period_scoped_lookup_preserves_optional_comparison_steps():
    class PlannedRouter:
        def generate(self, **kwargs):
            assert kwargs["response_model"] is InvestigationPlan
            return InvestigationPlan(
                original_question="What are the sales in August?",
                goal="Answer the metric question with comparison context.",
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question="Measure the requested metric in August.",
                        objective="Return the requested period's metric.",
                    ),
                    SubQuestion(
                        id=2,
                        question="Compare the same metric in the preceding period.",
                        objective="Measure the change against the prior period.",
                        depends_on=[1],
                        optional=True,
                    ),
                    SubQuestion(
                        id=3,
                        question="Identify contributing dimensions in the data.",
                        objective="Compare available segment-level metrics.",
                        depends_on=[1, 2],
                        optional=True,
                    ),
                ],
            )

    plan = QuestionPlanner(model_router=PlannedRouter()).create_plan(
        "Compare sales in August with the preceding comparable period."
    )

    assert [step.optional for step in plan.sub_questions] == [
        False,
        True,
        True,
    ]
    assert plan.sub_questions[2].depends_on == [1, 2]
    assert "preceding comparable period" in plan.sub_questions[2].question


def test_explicit_year_comparison_uses_one_schema_grounded_step():
    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "An explicit two-year comparison should be normalized directly."
            )

    question = (
        "Compare the sales of year 2017 and 2018 and tell if it increased "
        "or decreased."
    )
    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(question)

    assert len(plan.sub_questions) == 1
    assert not plan.sub_questions[0].optional
    assert not plan.sub_questions[0].depends_on
    assert "2017" in plan.sub_questions[0].objective
    assert "2018" in plan.sub_questions[0].objective


def test_change_wording_with_explicit_years_uses_one_required_step():
    question = "Total amount change in the sales volume from 2015 to 2017"

    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError("A two-year change should use one direct step.")

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(question)

    assert len(plan.sub_questions) == 1
    assert plan.sub_questions[0].optional is False
    assert plan.sub_questions[0].depends_on == []
    assert "2015" in plan.sub_questions[0].objective
    assert "2017" in plan.sub_questions[0].objective


def test_growth_over_three_years_from_anchor_uses_inclusive_year_rows():
    from app.agent.question_normalizer import QuestionNormalizer

    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "A relative year range should be planned deterministically."
            )

    question = "what was the growth over the last 3 years from 2017"
    years = QuestionNormalizer.relative_calendar_years(question)
    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(
        question,
        authoritative_question=question,
    )

    assert years == [2015, 2016, 2017]
    assert len(plan.sub_questions) == 1
    assert plan.sub_questions[0].depends_on == []
    assert all(
        f"Year {year}" in plan.sub_questions[0].required_information
        for year in years
    )
    assert "raw value per calendar year" in plan.sub_questions[0].objective


def test_normalizer_expands_relative_year_range_and_preserves_duration():
    from app.agent.question_normalizer import (
        NormalizedQuestion,
        QuestionNormalizer,
    )

    class NormalizingRouter:
        def generate(self, **kwargs):
            return NormalizedQuestion(
                normalized_question=(
                    "What was the growth over the last three years "
                    "from 2017?"
                )
            )

    normalized = QuestionNormalizer(
        model_router=NormalizingRouter()
    ).normalize("what was the growth over the last 3 years from 2017")

    assert "2015, 2016, and 2017 inclusive" in normalized
    assert QuestionNormalizer.relative_calendar_years(normalized) is None


def test_unsafe_normalization_falls_back_to_original_instead_of_blocking():
    from app.agent.question_normalizer import (
        NormalizedQuestion,
        QuestionNormalizer,
    )

    class DroppingRouter:
        def generate(self, **kwargs):
            return NormalizedQuestion(
                normalized_question="What was the change in sales?"
            )

    original = "What was the change in sales from 2015 to 2017?"
    normalized = QuestionNormalizer(model_router=DroppingRouter()).normalize(
        original
    )

    assert normalized == original


def test_normalized_question_does_not_replace_original_planning_constraints():
    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError("The original question is a direct comparison.")

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(
        "What was the total sales change from 2015 to 2017?",
        authoritative_question=(
            "Total amount change in the sales volume from 2015 to 2017"
        ),
    )

    assert plan.original_question == (
        "Total amount change in the sales volume from 2015 to 2017"
    )
    assert len(plan.sub_questions) == 1
    assert "2015" in plan.sub_questions[0].objective
    assert "2017" in plan.sub_questions[0].objective


def test_question_normalizer_corrects_grammar_without_changing_constraints():
    from app.agent.question_normalizer import (
        NormalizedQuestion,
        QuestionNormalizer,
    )

    class NormalizingRouter:
        def generate(self, **kwargs):
            assert kwargs["response_model"] is NormalizedQuestion
            assert "Total amount change" in kwargs["user_prompt"]
            return NormalizedQuestion(
                normalized_question=(
                    "What was the total change in sales volume from "
                    "2015 to 2017?"
                )
            )

    normalized = QuestionNormalizer(
        model_router=NormalizingRouter()
    ).normalize(
        "Total amount change in the sales volume from 2015 to 2017"
    )

    assert normalized == (
        "What was the total change in sales volume from 2015 to 2017?"
    )


def test_question_normalizer_falls_back_when_rewrite_drops_numeric_constraints():
    from app.agent.question_normalizer import (
        NormalizedQuestion,
        QuestionNormalizer,
    )

    class DroppingRouter:
        def generate(self, **kwargs):
            return NormalizedQuestion(
                normalized_question="What was the change in sales volume?"
            )

    original = "What was the total change in sales volume from 2015 to 2017?"
    normalized = QuestionNormalizer(
        model_router=DroppingRouter()
    ).normalize(original)

    assert normalized == original


def test_total_sales_change_reports_direction_and_absolute_difference():
    from app.agent.insight_generator import InsightGenerator

    question = "Total amount chnage in the sales volume from 2015 to 2017"
    interpretation = (
        "What was the total change in sales volume from 2015 to 2017?"
    )
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective="Return the requested metric for each requested year.",
            sql='{"collection":"sales","pipeline":[]}',
            rows=[
                {"_id": 2015, "totalSales": 1006},
                {"_id": 2017, "totalSales": 1052},
            ],
            schema_context=[],
            logical_reason="Verified yearly totals.",
            confidence=0.95,
        )
    ]

    insight = InsightGenerator().generate(
        question,
        evidence,
        interpretation_question=interpretation,
    )
    verification = InsightVerifier().verify(
        question,
        insight,
        evidence,
        interpretation_question=interpretation,
    )

    assert "increased" in insight.summary
    assert "1006" in insight.summary
    assert "1052" in insight.summary
    assert "46" in insight.summary
    assert verification.verified


def test_percentage_change_is_derived_from_verified_yearly_sales():
    from app.agent.insight_generator import InsightGenerator

    question = (
        "Tell me the increase or decrease in sales from 2015 to 2017 "
        "in percentage"
    )
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective=(
                "Return one raw requested-metric value and year label "
                "for each requested period."
            ),
            sql='{"collection":"sales","pipeline":[]}',
            rows=[
                {"_id": 2015, "totalSales": 1006},
                {"_id": 2017, "totalSales": 1052},
            ],
            schema_context=[],
            logical_reason="Verified yearly sales totals.",
            confidence=0.95,
        )
    ]

    insight = InsightGenerator().generate(question, evidence)
    verification = InsightVerifier().verify(question, insight, evidence)

    assert "increased" in insight.summary
    assert "4.57%" in insight.summary
    assert "1006" in insight.summary
    assert "1052" in insight.summary
    assert verification.verified


def test_percentage_change_with_zero_baseline_is_reported_as_undefined():
    from app.agent.insight_generator import InsightGenerator

    question = "What is the percentage change in sales from 2015 to 2017?"
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective="Return the raw sales metric for each requested year.",
            sql='{"collection":"sales","pipeline":[]}',
            rows=[
                {"_id": 2015, "totalSales": 0},
                {"_id": 2017, "totalSales": 10},
            ],
            schema_context=[],
            logical_reason="Verified yearly sales totals.",
            confidence=0.95,
        )
    ]

    insight = InsightGenerator().generate(question, evidence)
    verification = InsightVerifier().verify(question, insight, evidence)

    assert "increased" in insight.summary
    assert "cannot be calculated" in insight.summary
    assert "%" not in insight.summary
    assert verification.verified


def test_three_year_growth_summary_uses_every_verified_year():
    from app.agent.insight_generator import InsightGenerator

    question = (
        "What was the growth over the last 3 years from 2017 "
        "(calendar years 2015, 2016, and 2017 inclusive)?"
    )
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective="Return raw yearly sales values.",
            sql='{"collection":"sales","pipeline":[]}',
            rows=[
                {"_id": 2015, "totalSales": 1006},
                {"_id": 2016, "totalSales": 1010},
                {"_id": 2017, "totalSales": 1052},
            ],
            schema_context=[],
            logical_reason="Verified annual sales totals.",
            confidence=0.95,
        )
    ]

    insight = InsightGenerator().generate(question, evidence)
    verification = InsightVerifier().verify(question, insight, evidence)

    assert "increased" in insight.summary
    assert "46" in insight.summary
    assert "4.57%" in insight.summary
    assert "increased in every observed year" in insight.summary
    assert all(str(year) in insight.summary for year in (2015, 2016, 2017))
    assert verification.verified


def test_mongodb_three_year_growth_requires_annual_metric_rows():
    class RelevantRouter:
        def generate(self, **kwargs):
            return LogicalVerificationDecision(
                relevant=True,
                confidence=0.95,
                reason="The requested annual sales values are returned.",
            )

    years = [2015, 2016, 2017]
    query = (
        '{"collection":"sales","pipeline":['
        '{"$match":{"saleDate":{"$gte":{"$date":"2015-01-01T00:00:00Z"},'
        '"$lt":{"$date":"2018-01-01T00:00:00Z"}}}},'
        '{"$group":{"_id":{"$year":"$saleDate"},"totalSales":{"$sum":1}}},'
        '{"$sort":{"_id":1}},{"$limit":100}]}'
    )
    question = (
        "What was growth over the last 3 years from 2017 "
        "(calendar years 2015, 2016, and 2017 inclusive)?"
    )

    result = LogicalVerifier(model_router=RelevantRouter()).verify(
        question=question,
        objective=(
            "Return one raw value per year for years 2015, 2016, and 2017."
        ),
        sql=query,
        schema_context=[],
        rows=[
            {"_id": 2015, "totalSales": 1006},
            {"_id": 2016, "totalSales": 1010},
            {"_id": 2017, "totalSales": 1052},
        ],
        query_language="mongodb",
    )

    assert result.relevant
    assert result.requested_periods == years
    assert result.observed_periods == years
    assert result.unavailable_periods == []

def test_complete_mongodb_multi_measure_result_supports_relative_comparison():
    question = (
        "Which products sell the most units but generate relatively low revenue?"
    )

    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "Complete grouped metrics are checked deterministically."
            )

    pipeline = [
        {
            "$group": {
                "_id": "$product",
                "total_units_sold": {"$sum": "$quantity"},
                "total_revenue": {"$sum": "$amount"},
            }
        },
        {"$sort": {"total_units_sold": -1, "_id": 1}},
        {"$limit": 100},
    ]
    rows = [
        {"_id": "A", "total_units_sold": 100, "total_revenue": 200},
        {"_id": "B", "total_units_sold": 80, "total_revenue": 400},
        {"_id": "C", "total_units_sold": 20, "total_revenue": 300},
    ]

    result = LogicalVerifier(model_router=UnusedRouter()).verify(
        question=question,
        objective=(
            "Return both requested measures for every entity so the relative "
            "comparison can be computed from complete grouped evidence."
        ),
        sql=json.dumps({"collection": "sales", "pipeline": pipeline}),
        schema_context=[],
        rows=rows,
        pagination_complete=True,
        query_language="mongodb",
    )

    assert result.relevant
    assert "relative high/low comparison" in result.reason


def test_reason_for_decline_is_classified_as_why_analysis():
    assert QuestionPlanner._asks_why(
        "What were the reasons sales decreased?"
    )
    assert QuestionPlanner._asks_why(
        "Which factors contributed to the sales decline?"
    )


def test_insight_verifier_rejects_unsupported_causal_explanation():
    question = "Why did sales decrease?"
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective="Compare sales totals over time.",
            sql="SELECT year, total_sales FROM sales",
            rows=[
                {"year": 2016, "total_sales": 100},
                {"year": 2017, "total_sales": 80},
            ],
            schema_context=[],
            logical_reason="Verified sales values.",
            confidence=0.95,
        )
    ]
    insight = GeneratedInsight(
        summary="Sales fell because the Austin store underperformed.",
        findings=[],
    )

    verification = InsightVerifier().verify(question, insight, evidence)

    assert not verification.verified
    assert any("Causal language" in finding for finding in verification.unsupported_findings)


def test_period_comparison_rejects_collapsed_percentage_only_mongodb_result():
    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "Collapsed comparison output should fail deterministically."
            )

    query = (
        '{"collection":"sales","pipeline":['
        '{"$group":{"_id":{"$year":"$saleDate"},"sales":{"$sum":1}}},'
        '{"$group":{"_id":null,"percentageChange":{"$sum":"$sales"}}},'
        '{"$limit":1}]}'
    )
    result = LogicalVerifier(model_router=UnusedRouter()).verify(
        question=(
            "What is the percentage increase or decrease in sales "
            "from 2015 to 2017?"
        ),
        objective=(
            "Return one result row per year with raw requested metric "
            "values for 2015 and 2017."
        ),
        sql=query,
        schema_context=[],
        rows=[{"_id": None, "percentageChange": 4.57}],
        query_language="mongodb",
    )

    assert not result.relevant
    assert "collapsed" in result.reason
    assert "raw metric value" in result.suggested_revision


def test_period_comparison_accepts_individual_year_rows():
    class RelevantRouter:
        def generate(self, **kwargs):
            return LogicalVerificationDecision(
                relevant=True,
                confidence=0.95,
                reason="Each requested year and raw metric are returned.",
            )

    query = (
        '{"collection":"sales","pipeline":['
        '{"$match":{"saleDate":{"$gte":{"$date":"2015-01-01T00:00:00Z"},'
        '"$lt":{"$date":"2018-01-01T00:00:00Z"}}}},'
        '{"$group":{"_id":{"$year":"$saleDate"},"totalSales":{"$sum":1}}},'
        '{"$sort":{"_id":1}},{"$limit":100}]}'
    )
    result = LogicalVerifier(model_router=RelevantRouter()).verify(
        question="What is the percentage change in sales from 2015 to 2017?",
        objective=(
            "Return the requested metric grouped by each explicitly "
            "requested year."
        ),
        sql=query,
        schema_context=[],
        rows=[
            {"_id": 2015, "totalSales": 1006},
            {"_id": 2017, "totalSales": 1052},
        ],
        query_language="mongodb",
    )

    assert result.relevant
    assert result.observed_periods == [2015, 2017]
    assert result.unavailable_periods == []


def test_mongodb_missing_requested_year_is_partial_verified_evidence():
    from app.agent.logical_verifier import LogicalVerificationDecision

    class MissingPeriodRouter:
        def generate(self, **kwargs):
            return LogicalVerificationDecision(
                relevant=False,
                confidence=0.9,
                reason=(
                    "The aggregation only returned data for 2017; "
                    "the requested year 2018 is missing."
                ),
                missing_information=["A result for year 2018."],
            )

    query = (
        '{"collection":"sales","pipeline":['
        '{"$match":{"saleDate":{"$gte":{"$date":"2017-01-01T00:00:00Z"},'
        '"$lt":{"$date":"2019-01-01T00:00:00Z"}}}},'
        '{"$group":{"_id":{"$year":"$saleDate"},'
        '"totalSales":{"$sum":{"$multiply":["$items.price",'
        '"$items.quantity"]}}}},'
        '{"$sort":{"_id":1}},{"$limit":100}]}'
    )
    result = LogicalVerifier(model_router=MissingPeriodRouter()).verify(
        question="Compare sales in 2017 and 2018.",
        objective="Compare the requested metric for both years.",
        sql=query,
        schema_context=[],
        rows=[{"_id": 2017, "totalSales": 2185852.90}],
        query_language="mongodb",
    )

    assert result.relevant
    assert result.requested_periods == [2017, 2018]
    assert result.observed_periods == [2017]
    assert result.unavailable_periods == [2018]
    assert "cannot determine an increase or decrease" in result.reason


def test_mongodb_single_year_count_is_not_marked_as_period_comparison():
    class RelevantRouter:
        def generate(self, **kwargs):
            return LogicalVerificationDecision(
                relevant=True,
                confidence=0.95,
                reason="The requested customer count is returned.",
            )

    query = (
        '{"collection":"sales","pipeline":['
        '{"$match":{"saleDate":{"$gte":{"$date":"2017-01-01T00:00:00Z"},'
        '"$lt":{"$date":"2018-01-01T00:00:00Z"}}}},'
        '{"$group":{"_id":null,"unique_customers":'
        '{"$addToSet":"$customer.email"}}},'
        '{"$project":{"_id":0,"customer_count":'
        '{"$size":"$unique_customers"}}},{"$limit":1}]}'
    )
    result = LogicalVerifier(model_router=RelevantRouter()).verify(
        question="How many customers bought printer paper in 2017?",
        objective="Return the requested count for the user's stated filters.",
        sql=query,
        schema_context=[],
        rows=[{"customer_count": 488}],
        query_language="mongodb",
    )

    assert result.relevant
    assert result.requested_periods == []
    assert result.observed_periods == []
    assert result.unavailable_periods == []


def test_missing_year_override_requires_full_requested_range():
    from app.agent.logical_verifier import LogicalVerificationDecision

    class MissingPeriodRouter:
        def generate(self, **kwargs):
            return LogicalVerificationDecision(
                relevant=False,
                confidence=0.9,
                reason="The requested year 2018 is missing.",
                missing_information=["A result for 2018."],
            )

    query = (
        '{"collection":"sales","pipeline":['
        '{"$match":{"saleDate":{"$gte":{"$date":"2017-01-01T00:00:00Z"},'
        '"$lt":{"$date":"2018-01-01T00:00:00Z"}}}},'
        '{"$group":{"_id":{"$year":"$saleDate"},'
        '"totalSales":{"$sum":"$amount"}}},{"$limit":100}]}'
    )
    result = LogicalVerifier(model_router=MissingPeriodRouter()).verify(
        question="Compare sales in 2017 and 2018.",
        objective="Compare the requested metric for both years.",
        sql=query,
        schema_context=[],
        rows=[{"_id": 2017, "totalSales": 50}],
        query_language="mongodb",
    )

    assert not result.relevant
    assert not result.unavailable_periods


def test_missing_year_comparison_has_deterministic_grounded_summary():
    from app.agent.evidence import Evidence
    from app.agent.insight_generator import InsightGenerator

    evidence = Evidence(
        step_id=1,
        question="Compare sales in 2017 and 2018.",
        objective="Compare the requested metric for both years.",
        sql='{"collection":"sales","pipeline":[]}',
        rows=[{"_id": 2017, "totalSales": 100}],
        schema_context=[],
        logical_reason="Verified the requested period range.",
        confidence=0.8,
        requested_periods=[2017, 2018],
        observed_periods=[2017],
        unavailable_periods=[2018],
    )

    insight = InsightGenerator().generate(
        "Compare sales in 2017 and 2018.",
        [evidence],
    )
    verification = InsightVerifier().verify(
        question="Compare sales in 2017 and 2018.",
        insight=insight,
        evidence=[evidence],
    )

    assert "cannot be determined" in insight.summary
    assert "2018" in insight.summary
    assert "increased or decreased" in " ".join(insight.findings)
    assert verification.verified


def test_mongodb_decimal_comparison_generates_concise_grounded_insight():
    from app.agent.insight_generator import InsightGenerator

    question = (
        "Compare the sales of year 2017 and 2016 and tell if it "
        "increased or decreased."
    )
    rows = [
        {
            "_id": 2016,
            "totalSalesVolume": {"$numberDecimal": "1934819.86"},
        },
        {
            "_id": 2017,
            "totalSalesVolume": {"$numberDecimal": "2185852.90"},
        },
    ]
    evidence = [
        Evidence(
            step_id=step_id,
            question=question,
            objective="Compare yearly sales.",
            sql='{"collection":"sales","pipeline":[]}',
            rows=rows,
            schema_context=[],
            logical_reason="Verified yearly totals.",
            confidence=0.95,
        )
        for step_id in range(1, 4)
    ]
    evidence.append(
        Evidence(
            step_id=4,
            question=question,
            objective="Return sales grouped by store and year.",
            sql='{"collection":"sales","pipeline":[]}',
            rows=[
                {
                    "_id": {"storeLocation": "Austin", "year": 2016},
                    "totalSales": {"$numberDecimal": "79018.23"},
                },
                {
                    "_id": {"storeLocation": "Austin", "year": 2017},
                    "totalSales": {"$numberDecimal": "95630.52"},
                },
            ],
            schema_context=[],
            logical_reason="Verified store-level breakdown.",
            confidence=0.95,
        )
    )

    insight = InsightGenerator().generate(question, evidence)
    verification = InsightVerifier().verify(question, insight, evidence)

    assert "increased" in insight.summary
    assert "1934819.86" in insight.summary
    assert "2185852.9" in insight.summary
    assert verification.verified
    assert "Austin" not in insight.summary


def test_period_comparison_verifier_rejects_wrong_direction():
    question = "Compare sales in 2016 and 2017."
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective="Compare yearly sales.",
            sql='{"collection":"sales","pipeline":[]}',
            rows=[
                {
                    "_id": 2016,
                    "totalSalesVolume": {"$numberDecimal": "1934819.86"},
                },
                {
                    "_id": 2017,
                    "totalSalesVolume": {"$numberDecimal": "2185852.90"},
                },
            ],
            schema_context=[],
            logical_reason="Verified yearly totals.",
            confidence=0.95,
        )
    ]
    incorrect = GeneratedInsight(
        summary=(
            "Sales decreased from 2016 to 2017: 1934819.86 in 2016 "
            "and 2185852.9 in 2017."
        ),
        findings=[],
    )

    decision = InsightVerifier().verify(question, incorrect, evidence)

    assert not decision.verified
    assert any("direction" in item for item in decision.unsupported_findings)


def test_failed_optional_analysis_does_not_fail_required_answer():
    failed = InvestigationEngine._failed_required_steps([
        {"id": 1, "status": "verified", "optional": False},
        {"id": 2, "status": "failed", "optional": True},
        {"id": 3, "status": "blocked", "optional": True},
    ])

    assert failed == []

    required_failure = InvestigationEngine._failed_required_steps([
        {"id": 1, "status": "failed", "optional": False},
        {"id": 2, "status": "failed", "optional": True},
    ])
    assert [step["id"] for step in required_failure] == [1]


def test_steps_skipped_for_verified_empty_dependencies_do_not_fail_answer():
    failed = InvestigationEngine._failed_required_steps([
        {"id": 1, "status": "verified", "optional": False},
        {
            "id": 2,
            "status": "skipped",
            "skip_reason": "empty_dependency",
            "optional": False,
        },
        {
            "id": 3,
            "status": "skipped",
            "skip_reason": "empty_dependency",
            "optional": False,
        },
    ])

    assert failed == []

    unexpected_skip = InvestigationEngine._failed_required_steps([
        {
            "id": 1,
            "status": "skipped",
            "skip_reason": "not_applicable",
            "optional": False,
        },
    ])
    assert [step["id"] for step in unexpected_skip] == [1]


def test_why_plan_sanitization_does_not_insert_domain_specific_content():
    plan = InvestigationPlan(
        original_question="Why did the service response time increase?",
        goal="Investigate response time.",
        sub_questions=[
            SubQuestion(
                id=1,
                question="Measure the service response time.",
                objective="Establish the response time baseline.",
            ),
            SubQuestion(
                id=2,
                question="Identify the factor that causes the increase.",
                objective="Find evidence for the cause.",
                depends_on=[1],
            ),
        ],
    )

    sanitized = QuestionPlanner._sanitize_why_plan(
        plan.original_question,
        plan,
    )
    combined = " ".join(
        step.question + " " + step.objective
        for step in sanitized.sub_questions
    ).lower()

    assert "pathway" not in combined
    assert "ortholog" not in combined
    assert sanitized.sub_questions[1].optional


def test_why_plan_keeps_unrequested_breakdowns_supplemental():
    question = "Why are sales decreasing in the last two years?"
    plan = InvestigationPlan(
        original_question=question,
        goal="Investigate the sales decrease.",
        sub_questions=[
            SubQuestion(
                id=1,
                question="Measure total sales for the recent years.",
                objective="Establish the sales baseline trend.",
                optional=True,
            ),
            SubQuestion(
                id=2,
                question=(
                    "Break down sales by product category, region, "
                    "and customer segment."
                ),
                objective=(
                    "Identify dimensions associated with the decrease."
                ),
                depends_on=[1],
            ),
        ],
    )

    sanitized = QuestionPlanner._sanitize_why_plan(question, plan)

    assert sanitized.sub_questions[0].optional is False
    assert sanitized.sub_questions[1].optional is True


def test_why_plan_adds_optional_schema_grounded_breakdown():
    question = "Why did response time increase?"
    plan = InvestigationPlan(
        original_question=question,
        goal="Explain the response-time increase.",
        sub_questions=[
            SubQuestion(
                id=1,
                question="Measure response time for the requested period.",
                objective="Establish the response-time baseline.",
            ),
        ],
    )

    sanitized = QuestionPlanner._sanitize_why_plan(question, plan)
    baseline, breakdown = sanitized.sub_questions

    assert baseline.optional is False
    assert breakdown.optional is True
    assert breakdown.depends_on == [baseline.id]
    assert "live schema" in breakdown.objective


def test_why_plan_preserves_explicitly_requested_breakdowns_as_required():
    question = "Why are sales decreasing by region?"
    plan = InvestigationPlan(
        original_question=question,
        goal="Investigate the regional sales decrease.",
        sub_questions=[
            SubQuestion(
                id=1,
                question="Measure sales by region over time.",
                objective="Establish the regional sales baseline.",
            ),
            SubQuestion(
                id=2,
                question="Compare regional sales totals.",
                objective="Identify regions associated with the decrease.",
                depends_on=[1],
            ),
        ],
    )

    sanitized = QuestionPlanner._sanitize_why_plan(question, plan)

    assert all(not step.optional for step in sanitized.sub_questions)


def test_why_fallback_never_invents_domain_specific_explanations():
    from app.agent.insight_generator import InsightGenerator

    generator = InsightGenerator.__new__(InsightGenerator)
    evidence = [
        Evidence(
            step_id=1,
            question="Why did response time increase?",
            objective="Measure average response time.",
            sql="SELECT average_response_time FROM service_metrics",
            rows=[{"average_response_time": 42}],
            schema_context=[],
            logical_reason="Verified baseline.",
            confidence=0.95,
        ),
        Evidence(
            step_id=2,
            question="Compare response time by service.",
            objective="Compare the measured response time across services.",
            sql="SELECT service, average_response_time FROM service_metrics",
            rows=[{"service": "alpha", "average_response_time": 42}],
            schema_context=[],
            logical_reason="Verified comparison.",
            confidence=0.95,
        ),
    ]

    fallback = generator._safe_fallback(
        "Why did response time increase?",
        evidence,
    )

    combined = " ".join(
        [fallback.summary, *fallback.findings, *fallback.uncertainties]
    ).lower()
    assert "pathway" not in combined
    assert "ortholog" not in combined
    assert "could not be verified" in fallback.summary


def test_report_preserves_evidence_based_recommendations():
    from app.reporting.artifacts import build_artifacts

    _, report = build_artifacts(
        question="Compare results across segments.",
        insight={
            "summary": "The verified segments differ.",
            "findings": ["Segment A has lower sales than the comparison period."],
            "recommendations": [
                "Review the affected segment before changing the allocation."
            ],
            "uncertainties": [],
        },
        evidence=[],
    )

    assert report.recommendations == [
        "Review the affected segment before changing the allocation."
    ]


def test_destructive_ddl_and_sensitive_requests_are_blocked():
    policy = RequestPolicy()
    assert policy.evaluate("Delete all gene rows").allowed is False
    assert policy.evaluate("Alter the gene table").allowed is False
    assert policy.evaluate("Show private sensitive information").allowed is False
