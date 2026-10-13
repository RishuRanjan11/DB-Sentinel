import json

import pytest

from app.agent.evidence import Evidence
from app.agent.insight_generator import GeneratedInsight, InsightGenerator
from app.agent.insight_verifier import InsightVerifier
from app.agent.logical_verifier import (
    LogicalVerificationDecision,
    LogicalVerifier,
)
from app.agent.planner import QuestionPlanner
from app.agent.question_normalizer import QuestionNormalizer
from app.analysis.intent import AnalysisIntent, AnalysisIntentClassifier
from app.security.policies import RequestPolicy


@pytest.mark.parametrize(
    "question",
    [
        "Which FlyBase genes have no human ortholog?",
        "Compare confirmed TB cases in India between 2021 and 2022.",
        "How did unknown TB cases change across countries?",
        "Which MongoDB product category had the largest sales decline?",
        "Why did sales fall, and which regions were associated with the decline?",
        "What percentage of customers made repeat purchases?",
    ],
)
def test_analyst_question_examples_are_classified_as_data_analysis(question):
    assert (
        AnalysisIntentClassifier.classify(question)
        == AnalysisIntent.DATA_ANALYSIS
    )


@pytest.mark.parametrize(
    ("question", "operation"),
    [
        ("Insert a new gene record.", "INSERT"),
        ("Update the customer record.", "UPDATE"),
        ("Delete TB case records.", "DELETE"),
        ("Drop the staging table.", "DROP"),
    ],
)
def test_new_crud_write_requests_are_rejected(question, operation):
    decision = RequestPolicy().evaluate(question)

    assert not decision.allowed
    assert decision.operation == operation


@pytest.mark.parametrize(
    "question",
    [
        "List genes without a human ortholog.",
        "Show confirmed TB cases by country for 2022.",
        "Compare sales by product category.",
        "Count customers with repeat purchases.",
    ],
)
def test_analyst_read_requests_remain_allowed(question):
    decision = RequestPolicy().evaluate(question)

    assert decision.allowed
    assert decision.operation == "READ"


def test_tb_year_comparison_plan_keeps_both_periods_and_requested_metric():
    question = (
        "Compare confirmed TB cases in India between 2021 and 2022 "
        "and quantify the percentage change."
    )

    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError("Explicit year comparisons are planned directly.")

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(question)

    assert len(plan.sub_questions) == 1
    step = plan.sub_questions[0]
    assert not step.optional
    assert "2021" in step.objective
    assert "2022" in step.objective
    assert "raw requested metric value" in step.objective
    assert "confirmed TB cases in India" in step.question


def test_unanchored_last_two_years_plan_uses_latest_years_in_database():
    question = (
        "How much positive or negative growth has happened in the last 2 years?"
    )

    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "An unanchored relative period should be planned directly."
            )

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(question)

    assert len(plan.sub_questions) == 1
    step = plan.sub_questions[0]
    assert not step.optional
    assert step.depends_on == []
    assert "2 most recent distinct calendar years" in step.objective
    assert "current system date" in step.objective
    assert "do not filter to" in step.objective
    assert not any(
        str(year) in step.objective
        for year in range(2020, 2031)
    )


def test_growth_from_last_year_plan_compares_two_available_years():
    question = "What was the growth from last year?"

    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "An unanchored relative comparison should be planned directly."
            )

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(question)

    assert len(plan.sub_questions) == 1
    assert "2 most recent distinct calendar years" in plan.sub_questions[0].objective


def test_relative_growth_plan_preserves_requested_entity_breakdown():
    question = (
        "Which locations have shown consistent growth or declining sales "
        "over the last three years?"
    )

    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "Relative entity trends are planned directly."
            )

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(question)
    objective = plan.sub_questions[0].objective.lower()

    assert "group by that dimension and calendar year" in objective
    assert "one row per entity/year" in objective
    assert "do not limit the result to the number of years" in objective


def test_relative_comparison_year_count_keeps_both_years_for_last_year_growth():
    assert (
        QuestionNormalizer.relative_comparison_year_count(
            "What was the growth from last year?"
        )
        == 2
    )
    assert (
        QuestionNormalizer.relative_comparison_year_count(
            "Show sales from the last 3 years."
        )
        == 3
    )


def test_decline_driver_plan_requires_requested_category_breakdown():
    question = (
        "Which product categories contributed most to the sales decline "
        "from 2022 to 2023, and why?"
    )

    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "The requested period comparison should be planned directly."
            )

    plan = QuestionPlanner(model_router=UnusedRouter()).create_plan(
        question
    )

    baseline = next(
        step
        for step in plan.sub_questions
        if "raw requested metric value" in step.objective
    )
    category_breakdown = next(
        step
        for step in plan.sub_questions
        if "product category" in step.question.lower()
    )

    assert not baseline.optional
    assert not category_breakdown.optional
    assert category_breakdown.depends_on == [baseline.id]
    assert "without asserting" in category_breakdown.objective


def test_tb_trend_summary_uses_every_verified_year():
    question = "How did reported TB cases change from 2020 through 2022?"
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective=(
                "Return reported TB cases separately for 2020, 2021, and 2022."
            ),
            sql="SELECT year, reported_cases FROM tb_reports",
            rows=[
                {"year": 2020, "reported_cases": 120},
                {"year": 2021, "reported_cases": 100},
                {"year": 2022, "reported_cases": 80},
            ],
            schema_context=[],
            logical_reason="All requested annual values were verified.",
            confidence=0.98,
        )
    ]

    insight = InsightGenerator._deterministic_period_comparison(
        question,
        evidence,
    )

    assert insight is not None
    assert "decreased from 2020 to 2022" in insight.summary
    assert "120 in 2020" in insight.summary
    assert "100 in 2021" in insight.summary
    assert "80 in 2022" in insight.summary
    assert "decreased in every observed year" in insight.summary


def test_entity_trend_summary_classifies_each_group_from_all_requested_years():
    question = (
        "Which locations have shown consistent growth or declining sales "
        "over the last three years?"
    )
    rows = [
        {"_id": {"storeLocation": "North", "year": 2022}, "total_sales": 100},
        {"_id": {"storeLocation": "North", "year": 2023}, "total_sales": 120},
        {"_id": {"storeLocation": "North", "year": 2024}, "total_sales": 140},
        {"_id": {"storeLocation": "South", "year": 2022}, "total_sales": 200},
        {"_id": {"storeLocation": "South", "year": 2023}, "total_sales": 180},
        {"_id": {"storeLocation": "South", "year": 2024}, "total_sales": 160},
        {"_id": {"storeLocation": "West", "year": 2022}, "total_sales": 90},
        {"_id": {"storeLocation": "West", "year": 2023}, "total_sales": 110},
        {"_id": {"storeLocation": "West", "year": 2024}, "total_sales": 105},
    ]
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective=(
                "Group sales by each requested location and calendar year."
            ),
            sql="MongoDB aggregation grouped by location and year",
            rows=rows,
            schema_context=[],
            logical_reason="Complete grouped annual location values returned.",
            confidence=0.95,
            pagination_complete=True,
            total_rows=len(rows),
        )
    ]

    insight = InsightGenerator._deterministic_entity_trends(question, evidence)
    assert insight is not None
    verification = InsightVerifier().verify(question, insight, evidence)

    assert insight.findings == [
        "North: total sales increased consistently (2022: 100; 2023: 120; 2024: 140).",
        "South: total sales declined consistently (2022: 200; 2023: 180; 2024: 160).",
    ]
    assert "both" not in insight.summary
    assert verification.verified


def test_entity_trend_summary_states_when_no_entity_consistently_declines():
    question = (
        "Which locations have shown consistent growth or declining sales "
        "over the last three years?"
    )
    rows = [
        {"_id": {"storeLocation": "North", "year": 2022}, "total_sales": 100},
        {"_id": {"storeLocation": "North", "year": 2023}, "total_sales": 120},
        {"_id": {"storeLocation": "North", "year": 2024}, "total_sales": 140},
    ]
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective="Group sales by each requested location and calendar year.",
            sql="MongoDB aggregation grouped by location and year",
            rows=rows,
            schema_context=[],
            logical_reason="Complete grouped annual location values returned.",
            confidence=0.95,
            pagination_complete=True,
            total_rows=len(rows),
        )
    ]

    insight = InsightGenerator._deterministic_entity_trends(question, evidence)
    assert insight is not None
    verification = InsightVerifier().verify(question, insight, evidence)

    assert insight.findings == [
        "North: total sales increased consistently "
        "(2022: 100; 2023: 120; 2024: 140)."
    ]
    assert "none shows a consistent decline" in insight.summary
    assert verification.verified


def test_unanchored_growth_summary_uses_latest_years_in_evidence():
    question = (
        "How much positive or negative growth has happened in the last 2 years?"
    )
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective=(
                "Return sales volume for the two most recent distinct "
                "calendar years present in the database."
            ),
            sql="MongoDB aggregation grouped by sale year",
            rows=[
                {"_id": 2016, "totalSalesVolume": 100},
                {"_id": 2017, "totalSalesVolume": 125},
            ],
            schema_context=[],
            logical_reason="Both most recent annual values were verified.",
            confidence=0.98,
        )
    ]

    insight = InsightGenerator._deterministic_period_comparison(
        question,
        evidence,
    )

    assert insight is not None
    assert "increased from 2016 to 2017" in insight.summary
    assert "100 in 2016" in insight.summary
    assert "125 in 2017" in insight.summary
    assert "absolute change was 25" in insight.summary


def test_unanchored_growth_does_not_summarize_if_fewer_years_are_available():
    question = "How much growth happened in the last 2 years?"
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective="Return sales for the two latest years available.",
            sql="SELECT year, sales FROM sales",
            rows=[{"year": 2017, "sales": 125}],
            schema_context=[],
            logical_reason="Only one annual value was returned.",
            confidence=0.8,
        )
    ]

    assert InsightGenerator._deterministic_period_comparison(
        question,
        evidence,
    ) is None


def test_mongodb_verifier_rejects_rolling_current_date_for_relative_years():
    question = (
        "How much positive or negative growth has happened in the last 2 years?"
    )
    pipeline = [
        {
            "$match": {
                "saleDate": {
                    "$gte": {
                        "$dateSubtract": {
                            "startDate": "$$NOW",
                            "unit": "year",
                            "amount": 2,
                        }
                    }
                }
            }
        },
        {
            "$group": {
                "_id": {"$year": "$saleDate"},
                "total": {"$sum": "$total"},
            }
        },
        {"$sort": {"_id": -1}},
        {"$limit": 2},
    ]
    schema = [{
        "table": "sales",
        "columns": [
            {"name": "saleDate", "data_type": "date"},
            {"name": "total", "data_type": "decimal"},
        ],
    }]

    issue = LogicalVerifier._relative_period_pipeline_issue(
        question,
        pipeline,
        schema,
    )

    assert issue is not None
    assert "current-date rolling window" in issue


def test_mongodb_verifier_accepts_latest_stored_year_aggregation():
    question = (
        "How much positive or negative growth has happened in the last 2 years?"
    )
    pipeline = [
        {
            "$group": {
                "_id": {"$year": "$saleDate"},
                "total": {"$sum": "$total"},
            }
        },
        {"$sort": {"_id": -1}},
        {"$limit": 2},
    ]
    schema = [{
        "table": "sales",
        "columns": [
            {"name": "saleDate", "data_type": "date"},
            {"name": "total", "data_type": "decimal"},
        ],
    }]

    assert LogicalVerifier._relative_period_pipeline_issue(
        question,
        pipeline,
        schema,
    ) is None


def test_mongodb_verifier_rejects_metric_sort_before_year_sort():
    question = "What was the growth from last year?"
    pipeline = [
        {
            "$group": {
                "_id": {"$year": "$saleDate"},
                "total": {"$sum": "$total"},
            }
        },
        {"$sort": {"total": 1, "_id": -1}},
        {"$limit": 2},
    ]
    schema = [{
        "table": "sales",
        "columns": [
            {"name": "saleDate", "data_type": "date"},
            {"name": "total", "data_type": "decimal"},
        ],
    }]

    issue = LogicalVerifier._relative_period_pipeline_issue(
        question,
        pipeline,
        schema,
    )

    assert issue is not None
    assert "sort primarily by the grouped year descending" in issue


def test_mongodb_verifier_accepts_projected_year_as_primary_sort():
    question = "What was the growth from last year?"
    pipeline = [
        {
            "$group": {
                "_id": {"$year": "$saleDate"},
                "total": {"$sum": "$total"},
            }
        },
        {"$project": {"year": "$_id", "total": 1}},
        {"$sort": {"year": -1, "total": 1}},
        {"$limit": 2},
    ]
    schema = [{
        "table": "sales",
        "columns": [
            {"name": "saleDate", "data_type": "date"},
            {"name": "total", "data_type": "decimal"},
        ],
    }]

    assert LogicalVerifier._relative_period_pipeline_issue(
        question,
        pipeline,
        schema,
    ) is None


def test_mongodb_verifier_accepts_real_nested_year_group_projection():
    question = "What is the growth compared to last year?"
    pipeline = [
        {
            "$group": {
                "_id": {"year": {"$year": "$saleDate"}},
                "count": {"$sum": 1},
            }
        },
        {
            "$project": {
                "_id": 0,
                "count": 1,
                "year": "$_id.year",
            }
        },
        {"$sort": {"year": -1}},
        {"$limit": 2},
    ]
    schema = [{
        "table": "sales",
        "columns": [
            {"name": "saleDate", "data_type": "date"},
        ],
    }]

    assert LogicalVerifier._relative_period_pipeline_issue(
        question,
        pipeline,
        schema,
    ) is None


def test_mongodb_verifier_accepts_composite_entity_and_derived_year_groups():
    question = (
        "Which locations have shown consistent growth or declining sales "
        "over the last three years?"
    )
    pipeline = [
        {"$addFields": {"year": {"$year": "$saleDate"}}},
        {
            "$group": {
                "_id": {
                    "storeLocation": "$storeLocation",
                    "year": "$year",
                },
                "total_sales": {"$sum": "$total"},
            }
        },
        {"$sort": {"_id.year": -1, "_id": 1}},
        {"$limit": 100},
    ]
    schema = [{
        "table": "sales",
        "columns": [
            {"name": "saleDate", "data_type": "date"},
            {"name": "storeLocation", "data_type": "string"},
            {"name": "total", "data_type": "decimal"},
        ],
    }]

    assert LogicalVerifier._relative_period_pipeline_issue(
        question,
        pipeline,
        schema,
    ) is None


def test_mongodb_verifier_rejects_composite_year_group_limited_to_year_count():
    question = (
        "Which locations have shown consistent growth or declining sales "
        "over the last three years?"
    )
    pipeline = [
        {"$addFields": {"year": {"$year": "$saleDate"}}},
        {
            "$group": {
                "_id": {
                    "storeLocation": "$storeLocation",
                    "year": "$year",
                },
                "total_sales": {"$sum": "$total"},
            }
        },
        {"$sort": {"_id.year": -1, "_id": 1}},
        {"$limit": 3},
    ]
    schema = [{
        "table": "sales",
        "columns": [
            {"name": "saleDate", "data_type": "date"},
            {"name": "storeLocation", "data_type": "string"},
            {"name": "total", "data_type": "decimal"},
        ],
    }]

    issue = LogicalVerifier._relative_period_pipeline_issue(
        question,
        pipeline,
        schema,
    )

    assert issue is not None
    assert "not the number of years" in issue


def test_mongodb_year_trend_is_not_revalidated_as_metric_ranking():
    question = (
        "Which locations have shown consistent growth or declining sales "
        "over the last three years?"
    )
    pipeline = [
        {
            "$group": {
                "_id": {
                    "storeLocation": "$storeLocation",
                    "year": {"$year": "$saleDate"},
                },
                "total_sales": {"$sum": "$total"},
            }
        },
        {"$sort": {"_id.year": -1, "_id": 1}},
        {"$limit": 100},
    ]

    assert LogicalVerifier._mongo_ranking_pipeline_issue(
        question,
        "Rank locations by sales.",
        pipeline,
        query_intent={"kind": "ranking"},
    ) is None


def test_mongodb_logical_verification_accepts_the_observed_nested_year_pipeline():
    class AcceptingRouter:
        def generate(self, **kwargs):
            return LogicalVerificationDecision(
                relevant=True,
                confidence=0.95,
                reason="Annual sales counts for the latest years were returned.",
            )

    question = "What is the growth compared to last year?"
    pipeline = [
        {
            "$group": {
                "_id": {"year": {"$year": "$saleDate"}},
                "count": {"$sum": 1},
            }
        },
        {
            "$project": {
                "_id": 0,
                "count": 1,
                "year": "$_id.year",
            }
        },
        {"$sort": {"year": -1}},
        {"$limit": 2},
    ]
    verifier = LogicalVerifier(model_router=AcceptingRouter())

    result = verifier.verify(
        question=question,
        objective=(
            "Return sales counts for the two latest distinct calendar years."
        ),
        sql=json.dumps({"collection": "sales", "pipeline": pipeline}),
        schema_context=[{
            "table": "sales",
            "columns": [{"name": "saleDate", "data_type": "date"}],
        }],
        rows=[
            {"year": 2017, "count": 25},
            {"year": 2016, "count": 20},
        ],
        query_language="mongodb",
    )

    assert result.relevant
    assert result.confidence == 0.95


def test_mongodb_logical_verification_accepts_entity_year_rows_for_trend_analysis():
    class UnusedRouter:
        def generate(self, **kwargs):
            raise AssertionError(
                "Complete entity/year evidence supports deterministic analysis."
            )

    question = (
        "Which locations have shown consistent growth or decline in sales "
        "over the last three years? Compare each location's annual sales "
        "using the latest three years available in the data, and list only "
        "locations with a consistent year-over-year trend."
    )
    pipeline = [
        {"$unwind": {"path": "$items"}},
        {"$addFields": {"year": {"$year": "$saleDate"}}},
        {
            "$group": {
                "_id": {
                    "storeLocation": "$storeLocation",
                    "year": "$year",
                },
                "annual_sales": {
                    "$sum": {
                        "$multiply": [
                            "$items.price",
                            "$items.quantity",
                        ]
                    }
                },
            }
        },
        {
            "$project": {
                "storeLocation": "$_id.storeLocation",
                "year": "$_id.year",
                "annual_sales": 1,
                "_id": 1,
            }
        },
        {"$sort": {"year": -1, "_id": 1}},
        {"$limit": 100},
    ]
    rows = [
        {
            "_id": {"storeLocation": "North", "year": 2022},
            "storeLocation": "North",
            "year": 2022,
            "annual_sales": 100,
        },
        {
            "_id": {"storeLocation": "North", "year": 2023},
            "storeLocation": "North",
            "year": 2023,
            "annual_sales": 120,
        },
        {
            "_id": {"storeLocation": "North", "year": 2024},
            "storeLocation": "North",
            "year": 2024,
            "annual_sales": 140,
        },
        {
            "_id": {"storeLocation": "South", "year": 2022},
            "storeLocation": "South",
            "year": 2022,
            "annual_sales": 200,
        },
        {
            "_id": {"storeLocation": "South", "year": 2023},
            "storeLocation": "South",
            "year": 2023,
            "annual_sales": 180,
        },
        {
            "_id": {"storeLocation": "South", "year": 2024},
            "storeLocation": "South",
            "year": 2024,
            "annual_sales": 160,
        },
    ]

    result = LogicalVerifier(model_router=UnusedRouter()).verify(
        question=question,
        objective=(
            "Return one raw annual sales value per location and year for the "
            "three latest distinct years available; derive trends from values."
        ),
        sql=json.dumps({"collection": "sales", "pipeline": pipeline}),
        schema_context=[{
            "table": "sales",
            "columns": [
                {"name": "saleDate", "data_type": "date"},
                {"name": "storeLocation", "data_type": "string"},
                {"name": "items.price", "data_type": "decimal"},
                {"name": "items.quantity", "data_type": "integer"},
            ],
        }],
        rows=rows,
        pagination_complete=True,
        query_language="mongodb",
    )

    assert result.relevant
    assert result.confidence == 0.95
    assert "calculated deterministically" in result.reason


def test_mongodb_ranking_verifier_rejects_entity_first_metric_sort():
    issue = LogicalVerifier._mongo_ranking_pipeline_issue(
        "Which regions generated the maximum number of sales?",
        "Rank regions from highest to lowest by salesCount.",
        [
            {
                "$group": {
                    "_id": "$storeLocation",
                    "salesCount": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1, "salesCount": -1}},
            {"$limit": 100},
        ],
    )

    assert issue is not None
    assert "sort primarily by its requested aggregate metric" in issue


def test_mongodb_ranking_verifier_accepts_metric_first_sort():
    issue = LogicalVerifier._mongo_ranking_pipeline_issue(
        "Which regions generated the maximum number of sales?",
        "Rank regions from highest to lowest by salesCount.",
        [
            {
                "$group": {
                    "_id": "$storeLocation",
                    "salesCount": {"$sum": 1},
                }
            },
            {"$sort": {"salesCount": -1, "_id": 1}},
            {"$limit": 100},
        ],
    )

    assert issue is None


def test_maximum_grouped_ranking_insight_returns_only_maximum_ties():
    evidence = [
        Evidence(
            step_id=1,
            question="Which regions generated the maximum number of sales?",
            objective="Rank regions from highest to lowest by salesCount.",
            sql="MongoDB aggregation grouped by storeLocation",
            rows=[
                {"_id": "North", "salesCount": 12},
                {"_id": "West", "salesCount": 12},
                {"_id": "South", "salesCount": 9},
                {"_id": "East", "salesCount": 4},
            ],
            schema_context=[],
            logical_reason="Sorted by salesCount descending.",
            confidence=0.95,
        )
    ]

    insight = InsightGenerator._deterministic_ranking(
        "Which regions generated the maximum number of sales?",
        evidence,
    )

    assert insight is not None
    assert insight.summary == "The maximum sales count is 12."
    assert insight.findings == [
        "North: 12 sales count.",
        "West: 12 sales count.",
    ]

def test_contrastive_ranking_does_not_collapse_to_single_metric():
    question = (
        "Which products sell the most units but generate relatively low revenue?"
    )
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective=(
                "Compare unit volume and revenue for each product, preserving "
                "the requested contrast."
            ),
            sql="MongoDB aggregation grouped by product",
            rows=[
                {"product": "A", "total_units": 100, "total_revenue": 200},
                {"product": "B", "total_units": 80, "total_revenue": 400},
            ],
            schema_context=[],
            logical_reason="Both measures are returned per product.",
            confidence=0.95,
        )
    ]

    insight = InsightGenerator._deterministic_ranking(question, evidence)

    assert insight is None


def test_contrastive_metric_insight_uses_a_data_derived_relative_baseline():
    question = (
        "Which products sell the most units but generate relatively low revenue?"
    )
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective=(
                "Return unit volume and revenue for every product so their "
                "relative relationship can be evaluated."
            ),
            sql="MongoDB aggregation grouped by product",
            rows=[
                {"_id": "laptop", "total_units_sold": 6793, "total_revenue": 6775977.07},
                {"_id": "backpack", "total_units_sold": 6918, "total_revenue": 817374.10},
                {"_id": "pens", "total_units_sold": 13766, "total_revenue": 581843.27},
                {"_id": "binder", "total_units_sold": 25493, "total_revenue": 511644.57},
                {"_id": "notepad", "total_units_sold": 20727, "total_revenue": 463615.48},
                {"_id": "envelopes", "total_units_sold": 25078, "total_revenue": 376658.49},
                {"_id": "printer paper", "total_units_sold": 12092, "total_revenue": 367459.29},
            ],
            schema_context=[],
            logical_reason="Both requested measures are grouped by product.",
            confidence=0.95,
            pagination_complete=True,
        )
    ]

    insight = InsightGenerator._comparative_metric_insight(
        question,
        InsightGenerator._evidence_rows(evidence),
    )

    assert insight is not None
    assert "median of the returned entities" in insight.summary
    assert insight.findings == [
        "binder: 25493 total units sold; 511644.57 total revenue.",
        "envelopes: 25078 total units sold; 376658.49 total revenue.",
        "notepad: 20727 total units sold; 463615.48 total revenue.",
    ]
    verification = InsightVerifier().verify(question, insight, evidence)
    assert verification.verified
    assert not verification.unsupported_findings


@pytest.mark.parametrize(
    ("question", "expected_summary", "expected_finding"),
    [
        (
            "Which region generates maximum sales?",
            "The maximum total sales is 2921009.92.",
            "Denver: 2921009.92 total sales.",
        ),
        (
            "Which region generates minimum sales?",
            "The minimum total sales is 672885.17.",
            "San Diego: 672885.17 total sales.",
        ),
    ],
)
def test_extreme_sales_insight_handles_mongodb_decimal128_values(
    question,
    expected_summary,
    expected_finding,
):
    evidence = [
        Evidence(
            step_id=1,
            question=question,
            objective="Rank regions by total sales.",
            sql="MongoDB aggregation grouped by storeLocation",
            rows=[
                {
                    "_id": "Austin",
                    "total_sales": {"$numberDecimal": "1445603.11"},
                },
                {
                    "_id": "Denver",
                    "total_sales": {"$numberDecimal": "2921009.92"},
                },
                {
                    "_id": "London",
                    "total_sales": {"$numberDecimal": "1583066.79"},
                },
                {
                    "_id": "New York",
                    "total_sales": {"$numberDecimal": "1016059.59"},
                },
                {
                    "_id": "San Diego",
                    "total_sales": {"$numberDecimal": "672885.17"},
                },
                {
                    "_id": "Seattle",
                    "total_sales": {"$numberDecimal": "2255947.69"},
                },
            ],
            schema_context=[],
            logical_reason="Ranking is ordered by total_sales.",
            confidence=0.95,
        )
    ]

    insight = InsightGenerator._deterministic_ranking(question, evidence)

    assert insight is not None
    assert insight.summary == expected_summary
    assert insight.findings == [expected_finding]


def test_minimum_ranking_verifier_reads_mongodb_decimal128_rows():
    rows = [
        {"_id": "San Diego", "total_sales": {"$numberDecimal": "225134.90"}},
        {"_id": "New York", "total_sales": {"$numberDecimal": "311750.43"}},
        {"_id": "Austin", "total_sales": {"$numberDecimal": "436116.89"}},
        {"_id": "London", "total_sales": {"$numberDecimal": "501452.82"}},
        {"_id": "Seattle", "total_sales": {"$numberDecimal": "698159.52"}},
        {"_id": "Denver", "total_sales": {"$numberDecimal": "929942.08"}},
    ]
    insight = GeneratedInsight(
        summary="The minimum total sales is 225134.9.",
        findings=["San Diego: 225134.9 total sales."],
    )
    evidence = [
        Evidence(
            step_id=1,
            question="Which region generates minimum sales?",
            objective="Rank regions from lowest to highest by total sales.",
            sql="MongoDB aggregation grouped by storeLocation",
            rows=rows,
            schema_context=[],
            logical_reason="Sorted by total_sales ascending.",
            confidence=0.95,
            pagination_complete=True,
        )
    ]

    result = InsightVerifier().verify(
        question="Which region generates minimum sales?",
        insight=insight,
        evidence=evidence,
    )

    assert result.verified
    assert "Ranking metric verified from evidence column 'total_sales'." in (
        result.supported_findings
    )
    assert not result.unsupported_findings
