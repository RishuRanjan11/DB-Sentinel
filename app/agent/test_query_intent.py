import pytest
from pydantic import ValidationError

from app.agent.logical_verifier import LogicalVerifier
from app.agent.query_intent import (
    AnalyticalQueryIntent,
    QueryIntentPlanner,
)
from app.agent.sql_pipeline import SQLPipeline


def test_query_intent_represents_grouped_metric_ranking():
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "ranking",
        "metrics": [{
            "operation": "count",
            "field": None,
            "alias": "sales_count",
        }],
        "dimensions": ["storeLocation"],
        "filters": [],
        "periods": [],
        "sort": [
            {"field": "sales_count", "direction": "desc"},
            {"field": "storeLocation", "direction": "asc"},
        ],
        "limit": 100,
        "include_all_extreme_ties": True,
        "requires_causal_evidence": False,
        "result_requirements": [
            "Return each region tied at the maximum sales count."
        ],
    })

    assert intent.kind == "ranking"
    assert intent.metrics[0].operation == "count"
    assert intent.sort[0].direction == "desc"
    assert intent.include_all_extreme_ties


@pytest.mark.parametrize(
    "intent",
    [
        {
            "kind": "ranking",
            "metrics": [],
            "dimensions": ["region"],
            "sort": [],
        },
        {
            "kind": "trend",
            "metrics": [{"operation": "sum", "field": "sales", "alias": "total"}],
            "periods": [],
        },
    ],
)
def test_query_intent_rejects_incomplete_analytical_contracts(intent):
    with pytest.raises(ValidationError):
        AnalyticalQueryIntent.model_validate(intent)


def test_ranking_intent_requires_sort_on_requested_metric():
    with pytest.raises(ValidationError, match="sort by a requested metric"):
        AnalyticalQueryIntent.model_validate({
            "kind": "ranking",
            "metrics": [{
                "operation": "count",
                "field": None,
                "alias": "sales_count",
            }],
            "dimensions": ["storeLocation"],
            "sort": [{"field": "storeLocation", "direction": "asc"}],
        })


def test_query_intent_schema_guard_rejects_unavailable_field_references():
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "ranking",
        "metrics": [{
            "operation": "count",
            "field": None,
            "alias": "sales_count",
        }],
        "dimensions": ["storeLocation"],
        "filters": [{
            "field": "unknownRegion",
            "operator": "eq",
            "value": "West",
        }],
        "periods": [],
        "sort": [{"field": "sales_count", "direction": "desc"}],
        "limit": 100,
    })

    assert QueryIntentPlanner.invalid_schema_references(
        intent,
        [{
            "table": "sales",
            "columns": [{"name": "storeLocation", "data_type": "string"}],
        }],
    ) == ["unknownRegion"]


def test_query_intent_schema_guard_checks_sort_dimensions():
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "ranking",
        "metrics": [{
            "operation": "count",
            "field": None,
            "alias": "sales_count",
        }],
        "dimensions": ["storeLocation"],
        "filters": [],
        "periods": [],
        "sort": [
            {"field": "sales_count", "direction": "desc"},
            {"field": "missingDimension", "direction": "asc"},
        ],
    })

    assert QueryIntentPlanner.invalid_schema_references(
        intent,
        [{
            "table": "sales",
            "columns": [{"name": "storeLocation", "data_type": "string"}],
        }],
    ) == ["missingDimension"]


def test_query_intent_schema_guard_accepts_live_fields():
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "comparison",
        "metrics": [{
            "operation": "sum",
            "field": "amount",
            "alias": "total_amount",
        }],
        "dimensions": ["saleDate"],
        "filters": [{
            "field": "storeLocation",
            "operator": "eq",
            "value": "West",
        }],
        "periods": ["2022", "2023"],
        "sort": [{"field": "saleDate", "direction": "asc"}],
        "limit": 100,
    })

    assert QueryIntentPlanner.invalid_schema_references(
        intent,
        [{
            "table": "sales",
            "columns": [
                {"name": "amount", "data_type": "decimal"},
                {"name": "saleDate", "data_type": "date"},
                {"name": "storeLocation", "data_type": "string"},
            ],
        }],
    ) == []


def test_metric_formula_is_typed_and_schema_grounded():
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "aggregate",
        "metrics": [{
            "operation": "sum",
            "formula": {
                "operator": "multiply",
                "fields": ["items.unitPrice", "items.quantity"],
            },
            "alias": "revenue",
        }],
        "dimensions": ["region"],
    })

    assert QueryIntentPlanner.invalid_schema_references(
        intent,
        [{
            "table": "orders",
            "columns": [
                {"name": "items.unitPrice", "data_type": "decimal"},
                {"name": "items.quantity", "data_type": "integer"},
                {"name": "region", "data_type": "string"},
            ],
        }],
    ) == []
    assert QueryIntentPlanner.invalid_schema_references(
        intent,
        [{
            "table": "orders",
            "columns": [
                {"name": "items.unitPrice", "data_type": "decimal"},
                {"name": "region", "data_type": "string"},
            ],
        }],
    ) == ["items.quantity"]


def test_metric_formula_rejects_conflicting_or_non_numeric_fields():
    with pytest.raises(ValidationError, match="either a source field or a formula"):
        AnalyticalQueryIntent.model_validate({
            "kind": "aggregate",
            "metrics": [{
                "operation": "sum",
                "field": "unitPrice",
                "formula": {
                    "operator": "multiply",
                    "fields": ["unitPrice", "quantity"],
                },
                "alias": "revenue",
            }],
        })

    with pytest.raises(ValidationError, match="numeric aggregation"):
        AnalyticalQueryIntent.model_validate({
            "kind": "aggregate",
            "metrics": [{
                "operation": "count",
                "formula": {
                    "operator": "multiply",
                    "fields": ["unitPrice", "quantity"],
                },
                "alias": "revenue",
            }],
        })


def test_query_intent_planner_uses_structured_schema_grounded_response():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return AnalyticalQueryIntent.model_validate({
                "kind": "ranking",
                "metrics": [{
                    "operation": "count",
                    "field": None,
                    "alias": "sales_count",
                }],
                "dimensions": ["storeLocation"],
                "filters": [],
                "periods": [],
                "sort": [
                    {"field": "sales_count", "direction": "desc"},
                    {"field": "storeLocation", "direction": "asc"},
                ],
                "limit": 100,
                "include_all_extreme_ties": True,
                "result_requirements": [
                    "Return the regions tied at maximum sales count."
                ],
            })

    router = CapturingRouter()
    intent = QueryIntentPlanner(router).create_intent(
        question="Which region has the maximum number of sales?",
        required_information=["region and number of sales"],
        schema_context=[{
            "schema": "analytics",
            "table": "sales",
            "columns": [{"name": "storeLocation", "data_type": "string"}],
        }],
    )

    assert intent.kind == "ranking"
    assert router.kwargs["response_model"] is AnalyticalQueryIntent
    assert "storeLocation" in router.kwargs["user_prompt"]
    assert "unit price multiplied by quantity" in router.kwargs["system_instruction"]


def test_query_intent_repair_prompt_carries_previous_contract_and_feedback():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return AnalyticalQueryIntent.model_validate({
                "kind": "aggregate",
                "metrics": [{
                    "operation": "count",
                    "field": None,
                    "alias": "customer_count",
                }],
            })

    prior_intent = AnalyticalQueryIntent.model_validate({
        "kind": "aggregate",
        "metrics": [{
            "operation": "count",
            "field": None,
            "alias": "customer_count",
        }],
        "filters": [{
            "field": "saleDate",
            "operator": "gte",
            "value": "2017-01-01",
        }],
    })
    router = CapturingRouter()
    QueryIntentPlanner(router).create_intent(
        question="How many customers bought paper in 2017?",
        required_information=["customer count"],
        schema_context=[{
            "table": "sales",
            "columns": [{"name": "saleDate", "data_type": "date"}],
        }],
        feedback="The previous count grouped the result and pagination failed.",
        attempt_number=2,
        previous_intent=prior_intent,
    )

    assert "previous count grouped the result" in router.kwargs["user_prompt"]
    assert '"value": "2017-01-01"' in router.kwargs["user_prompt"]
    assert "preserve every other\nvalid measure" in router.kwargs["user_prompt"]


@pytest.mark.parametrize(
    ("question", "expected_kind", "expected_sorts"),
    [
        ("List regions by sales.", "aggregate", []),
        ("Which region generates maximum sales?", "ranking", 2),
        ("Which region generates minimum sales?", "ranking", 2),
        ("Sort the results by sales.", "ranking", 2),
    ],
)
def test_planner_does_not_promote_listing_to_ranking(
    question,
    expected_kind,
    expected_sorts,
):
    class RankingRouter:
        def generate(self, **kwargs):
            return AnalyticalQueryIntent.model_validate({
                "kind": "ranking",
                "metrics": [{
                    "operation": "sum",
                    "field": "amount",
                    "alias": "total_sales",
                }],
                "dimensions": ["region"],
                "filters": [],
                "periods": [],
                "sort": [
                    {"field": "total_sales", "direction": "desc"},
                    {"field": "region", "direction": "asc"},
                ],
                "limit": 100,
            })

    intent = QueryIntentPlanner(RankingRouter()).create_intent(
        question=question,
        required_information=["regions and sales"],
        schema_context=[{
            "table": "sales",
            "columns": [
                {"name": "region", "data_type": "string"},
                {"name": "amount", "data_type": "decimal"},
            ],
        }],
    )

    assert intent.kind == expected_kind
    if isinstance(expected_sorts, int):
        assert len(intent.sort) == expected_sorts
    else:
        assert intent.sort == expected_sorts


def test_sql_pipeline_rejects_intent_fields_outside_live_schema():
    class FakeDatabaseManager:
        def get(self, connection_id, organization_id=None):
            return object()

        def get_connection(self, connection_id):
            class Connection:
                database_type = "postgresql"

            return Connection()

    intent = AnalyticalQueryIntent.model_validate({
        "kind": "aggregate",
        "metrics": [{
            "operation": "sum",
            "field": "unavailableAmount",
            "alias": "total",
        }],
    })
    result = SQLPipeline(FakeDatabaseManager()).generate_validate_and_execute(
        connection_id="test-connection",
        question="What is the total?",
        required_information=["total"],
        schema_context=[{
            "schema": "analytics",
            "table": "sales",
            "columns": [{"name": "amount", "data_type": "decimal"}],
        }],
        query_intent=intent,
    )

    assert result["allowed"] is False
    assert result["executed"] is False
    assert result["rows"] == []
    assert "unavailableAmount" in result["reason"]
    assert result["query_intent"] is None


def test_logical_sql_ranking_direction_comes_from_intent():
    valid = LogicalVerifier._validate_ranking(
        "Order the regional sales totals.",
        (
            'SELECT region, COUNT(*) AS sales_count FROM sales '
            'GROUP BY region ORDER BY sales_count DESC, region ASC LIMIT 10'
        ),
        intent_ranking=True,
        intent_direction="desc",
    )
    invalid = LogicalVerifier._validate_ranking(
        "Order the regional sales totals.",
        (
            'SELECT region, COUNT(*) AS sales_count FROM sales '
            'GROUP BY region ORDER BY sales_count ASC, region ASC LIMIT 10'
        ),
        intent_ranking=True,
        intent_direction="desc",
    )

    assert valid["valid"] is True
    assert invalid["valid"] is False


def test_logical_mongo_ranking_uses_metric_sort_not_dimension_tiebreak():
    intent = {
        "kind": "ranking",
        "metrics": [{
            "operation": "count",
            "field": None,
            "alias": "sales_count",
        }],
        "dimensions": ["storeLocation"],
        "sort": [
            {"field": "storeLocation", "direction": "asc"},
            {"field": "sales_count", "direction": "desc"},
        ],
    }
    issue = LogicalVerifier._mongo_ranking_pipeline_issue(
        "Order regions by sales count",
        "Which region has the maximum number of sales?",
        [
            {"$group": {
                "_id": "$storeLocation",
                "count": {"$sum": 1},
            }},
            {"$sort": {"count": -1, "_id": 1}},
        ],
        query_intent=intent,
    )

    assert issue is None


def test_logical_mongo_listing_by_metric_is_not_treated_as_ranking():
    issue = LogicalVerifier._mongo_ranking_pipeline_issue(
        "List regions by sales.",
        "Directly retrieve regional sales totals.",
        [
            {"$group": {
                "_id": "$region",
                "total_sales": {"$sum": "$amount"},
            }},
            {"$sort": {"_id": 1, "total_sales": -1}},
        ],
        query_intent={
            "kind": "aggregate",
            "metrics": [{
                "operation": "sum",
                "field": "amount",
                "alias": "total_sales",
            }],
            "dimensions": ["region"],
            "sort": [],
        },
    )

    assert issue is None
