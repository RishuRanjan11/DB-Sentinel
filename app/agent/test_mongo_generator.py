import json

import pytest
from pydantic import ValidationError

from app.agent.query_intent import AnalyticalQueryIntent
from app.agent.mongo_generator import MongoQueryGenerator
from app.agent.mongo_generator import MetricFormulaEnforcementError
from app.agent.planner import QuestionPlanner
from app.agent.sql_generator import SQLGenerator
from app.agent.sql_pipeline import SQLPipeline
from app.models.mongo import GeneratedMongoQuery, GeneratedMongoStage
from app.models.sql import GeneratedSQL


def test_mongodb_generation_uses_structured_stage_fields_then_serializes_pipeline():
    query = GeneratedMongoQuery.model_validate({
        "collection": "customers",
        "pipeline": [
            {
                "operator": "$match",
                "specification_json": (
                    '{"birthdate":{"$gte":"1970-01-01",'
                    '"$lt":"1981-01-01"}}'
                ),
            },
            {
                "operator": "$limit",
                "specification_json": "100",
            },
        ],
        "fields_used": ["birthdate"],
    })

    serialized = json.loads(MongoQueryGenerator.serialize(query))

    assert serialized["pipeline"] == [
        {
            "$match": {
                "birthdate": {
                    "$gte": "1970-01-01",
                    "$lt": "1981-01-01",
                }
            }
        },
        {"$limit": 100},
    ]


def test_mongodb_stage_response_schema_has_named_operator_and_json_spec_fields():
    schema = GeneratedMongoQuery.model_json_schema()
    stage_schema = schema["$defs"]["GeneratedMongoStage"]

    assert set(stage_schema["properties"]) == {
        "operator",
        "specification_json",
    }
    assert stage_schema["properties"]["specification_json"]["type"] == "string"


def test_mongodb_generation_rejects_invalid_stage_specification_json():
    with pytest.raises(ValidationError, match="stage specification must be valid JSON"):
        GeneratedMongoQuery.model_validate({
            "collection": "customers",
            "pipeline": [
                {
                    "operator": "$match",
                    "specification_json": "{not-json}",
                }
            ],
            "fields_used": [],
        })


def test_date_range_is_not_mistaken_for_a_comparison_request():
    question = (
        "How many customers are born in between 1970 to 1980, "
        "list all of them and their products?"
    )

    assert QuestionPlanner._has_explicit_time_period(question)
    assert not QuestionPlanner._asks_for_period_comparison(question)


def test_mongodb_prompt_receives_only_supplied_live_schema():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return GeneratedMongoQuery(
                collection="records",
                pipeline=[
                    {
                        "operator": "$limit",
                        "specification_json": "10",
                    }
                ],
                fields_used=["record_date"],
            )

    router = CapturingRouter()
    schema = [{
        "schema": "active_database",
        "table": "records",
        "columns": [{"name": "record_date", "data_type": "date"}],
        "primary_key": ["_id"],
        "foreign_keys": [],
    }]

    MongoQueryGenerator(router).generate(
        question="List records in a date range.",
        schema_context=schema,
    )

    assert "active_database" in router.kwargs["user_prompt"]
    assert "record_date" in router.kwargs["user_prompt"]
    assert "records" in router.kwargs["user_prompt"]
    assert router.kwargs["response_model"] is GeneratedMongoQuery


def test_unanchored_year_growth_prompt_uses_stored_years_without_array_assumptions():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return GeneratedMongoQuery(
                collection="sales",
                pipeline=[
                    {
                        "operator": "$group",
                        "specification_json": (
                            '{"_id":{"$year":"$saleDate"},'
                            '"total":{"$sum":"$total"}}'
                        ),
                    },
                    {
                        "operator": "$sort",
                        "specification_json": '{"_id":-1}',
                    },
                    {
                        "operator": "$limit",
                        "specification_json": "2",
                    },
                ],
                fields_used=["saleDate", "total"],
            )

    router = CapturingRouter()
    MongoQueryGenerator(router).generate(
        question=(
            "How much positive or negative growth has happened in the last "
            "2 years?"
        ),
        schema_context=[{
            "schema": "analytics",
            "table": "sales",
            "columns": [
                {"name": "saleDate", "data_type": "date"},
                {"name": "total", "data_type": "decimal"},
                {"name": "items", "data_type": "array"},
            ],
        }],
    )

    prompt = router.kwargs["user_prompt"].lower()
    assert "latest 2 distinct calendar years present" in prompt
    assert "not a rolling date interval" in prompt
    assert "do not use $$now" in prompt
    assert "unlisted nested array paths or fields" in prompt
    assert "do not calculate growth inside the" in prompt
    assert "pipeline." in prompt


def test_relative_growth_repairs_metric_sort_and_returns_two_latest_years():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return GeneratedMongoQuery(
                collection="sales",
                pipeline=[
                    {
                        "operator": "$group",
                        "specification_json": (
                            '{"_id":{"$year":"$saleDate"},'
                            '"total":{"$sum":"$total"}}'
                        ),
                    },
                    {
                        "operator": "$project",
                        "specification_json": (
                            '{"year":"$_id","total":1,"_id":0}'
                        ),
                    },
                    {
                        "operator": "$sort",
                        "specification_json": '{"total":1,"year":-1}',
                    },
                    {
                        "operator": "$limit",
                        "specification_json": "1",
                    },
                ],
                fields_used=["saleDate", "total"],
            )

    query = MongoQueryGenerator(CapturingRouter()).generate(
        question="What was the growth from last year?",
        schema_context=[{
            "schema": "analytics",
            "table": "sales",
            "columns": [
                {"name": "saleDate", "data_type": "date"},
                {"name": "total", "data_type": "decimal"},
            ],
        }],
    )
    pipeline = json.loads(MongoQueryGenerator.serialize(query))["pipeline"]

    assert pipeline == [
        {
            "$group": {
                "_id": {"$year": "$saleDate"},
                "total": {"$sum": "$total"},
            }
        },
        {"$project": {"year": "$_id", "total": 1, "_id": 0}},
        {"$sort": {"year": -1}},
        {"$limit": 2},
    ]


def test_relative_growth_repairs_nested_year_group_projection_shape():
    class CapturingRouter:
        def generate(self, **kwargs):
            return GeneratedMongoQuery(
                collection="sales",
                pipeline=[
                    {
                        "operator": "$group",
                        "specification_json": (
                            '{"_id":{"year":{"$year":"$saleDate"}},'
                            '"count":{"$sum":1}}'
                        ),
                    },
                    {
                        "operator": "$project",
                        "specification_json": (
                            '{"_id":0,"count":1,"year":"$_id.year"}'
                        ),
                    },
                    {
                        "operator": "$sort",
                        "specification_json": '{"year":-1}',
                    },
                    {
                        "operator": "$limit",
                        "specification_json": "2",
                    },
                ],
                fields_used=["saleDate"],
            )

    query = MongoQueryGenerator(CapturingRouter()).generate(
        question="What is the growth compared to last year?",
        schema_context=[{
            "schema": "analytics",
            "table": "sales",
            "columns": [
                {"name": "saleDate", "data_type": "date"},
            ],
        }],
    )
    pipeline = json.loads(MongoQueryGenerator.serialize(query))["pipeline"]

    assert pipeline == [
        {
            "$group": {
                "_id": {"year": {"$year": "$saleDate"}},
                "count": {"$sum": 1},
            }
        },
        {"$project": {"_id": 0, "count": 1, "year": "$_id.year"}},
        {"$sort": {"year": -1}},
        {"$limit": 2},
    ]


def test_maximum_grouped_sales_repairs_entity_first_sort_dynamically():
    class CapturingRouter:
        def generate(self, **kwargs):
            return GeneratedMongoQuery(
                collection="sales",
                pipeline=[
                    {
                        "operator": "$group",
                        "specification_json": (
                            '{"_id":"$storeLocation",'
                            '"salesCount":{"$sum":1}}'
                        ),
                    },
                    {
                        "operator": "$sort",
                        "specification_json": (
                            '{"_id":1,"salesCount":-1}'
                        ),
                    },
                    {
                        "operator": "$limit",
                        "specification_json": "100",
                    },
                ],
                fields_used=["storeLocation"],
            )

    query = MongoQueryGenerator(CapturingRouter()).generate(
        question="Which regions generated the maximum number of sales?",
        schema_context=[{
            "schema": "analytics",
            "table": "sales",
            "columns": [{"name": "storeLocation", "data_type": "string"}],
        }],
    )
    pipeline = json.loads(MongoQueryGenerator.serialize(query))["pipeline"]

    assert pipeline[1] == {"$sort": {"salesCount": -1, "_id": 1}}
    assert pipeline[-1] == {"$limit": 100}


def test_maximum_direction_ignores_generic_tie_instruction_after_question():
    class CapturingRouter:
        def generate(self, **kwargs):
            return GeneratedMongoQuery(
                collection="sales",
                pipeline=[
                    {
                        "operator": "$group",
                        "specification_json": (
                            '{"_id":"$storeLocation",'
                            '"count":{"$sum":1}}'
                        ),
                    },
                    {
                        "operator": "$sort",
                        "specification_json": '{"_id":1,"count":1}',
                    },
                    {
                        "operator": "$limit",
                        "specification_json": "100",
                    },
                ],
                fields_used=["storeLocation"],
            )

    query = MongoQueryGenerator(CapturingRouter()).generate(
        question=(
            "Which region generates the maximum number of sales?"
            "\nReturn all entities tied at the maximum or minimum requested "
            "metric."
        ),
        schema_context=[{
            "schema": "analytics",
            "table": "sales",
            "columns": [{"name": "storeLocation", "data_type": "string"}],
        }],
    )
    pipeline = json.loads(MongoQueryGenerator.serialize(query))["pipeline"]

    assert pipeline[1] == {"$sort": {"count": -1, "_id": 1}}


def test_maximum_grouped_metric_is_repaired_after_projection_aliasing():
    class CapturingRouter:
        def generate(self, **kwargs):
            return GeneratedMongoQuery(
                collection="orders",
                pipeline=[
                    {
                        "operator": "$group",
                        "specification_json": (
                            '{"_id":"$region","orderCount":{"$sum":1}}'
                        ),
                    },
                    {
                        "operator": "$project",
                        "specification_json": (
                            '{"_id":0,"region":"$_id",'
                            '"count":"$orderCount"}'
                        ),
                    },
                    {
                        "operator": "$sort",
                        "specification_json": '{"region":1,"count":-1}',
                    },
                    {
                        "operator": "$limit",
                        "specification_json": "100",
                    },
                ],
                fields_used=["region"],
            )

    query = MongoQueryGenerator(CapturingRouter()).generate(
        question="Which region had the highest number of orders?",
        schema_context=[{
            "schema": "analytics",
            "table": "orders",
            "columns": [{"name": "region", "data_type": "string"}],
        }],
    )
    pipeline = json.loads(MongoQueryGenerator.serialize(query))["pipeline"]

    assert pipeline[2] == {"$sort": {"count": -1, "region": 1}}


def test_mongo_rank_repair_uses_intent_direction_not_question_word_heuristics():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return GeneratedMongoQuery(
                collection="sales",
                pipeline=[
                    {
                        "operator": "$group",
                        "specification_json": (
                            '{"_id":"$storeLocation",'
                            '"count":{"$sum":1}}'
                        ),
                    },
                    {
                        "operator": "$sort",
                        "specification_json": '{"_id":1,"count":1}',
                    },
                    {
                        "operator": "$limit",
                        "specification_json": "100",
                    },
                ],
                fields_used=["storeLocation"],
            )

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
            {"field": "storeLocation", "direction": "asc"},
            {"field": "sales_count", "direction": "desc"},
        ],
        "limit": 100,
    })
    router = CapturingRouter()
    query = MongoQueryGenerator(router).generate(
        question=(
            "Which region generates the maximum number of sales?\n"
            "Return all entities tied at the maximum or minimum requested metric."
        ),
        schema_context=[{
            "schema": "analytics",
            "table": "sales",
            "columns": [{"name": "storeLocation", "data_type": "string"}],
        }],
        query_intent=intent,
    )
    pipeline = json.loads(MongoQueryGenerator.serialize(query))["pipeline"]

    assert pipeline[1] == {"$sort": {"count": -1, "_id": 1}}
    assert "Canonical analytical query intent" in router.kwargs["user_prompt"]


def test_mongo_rank_repair_handles_nested_sales_metric_and_top_one_intent():
    class CapturingRouter:
        def generate(self, **kwargs):
            return GeneratedMongoQuery(
                collection="sales",
                pipeline=[
                    {
                        "operator": "$unwind",
                        "specification_json": '{"path":"$items"}',
                    },
                    {
                        "operator": "$group",
                        "specification_json": (
                            '{"_id":"$storeLocation","total_sales":'
                            '{"$sum":{"$multiply":'
                            '["$items.price","$items.quantity"]}}}'
                        ),
                    },
                    {
                        "operator": "$sort",
                        "specification_json": (
                            '{"_id":1,"total_sales":-1}'
                        ),
                    },
                    {
                        "operator": "$limit",
                        "specification_json": "1",
                    },
                ],
                fields_used=[
                    "items.price",
                    "items.quantity",
                    "storeLocation",
                ],
            )

    intent = AnalyticalQueryIntent.model_validate({
        "kind": "ranking",
        "metrics": [{
            "operation": "sum",
            "field": "items.price",
            "alias": "total_sales",
        }],
        "dimensions": ["storeLocation"],
        "filters": [],
        "periods": [],
        "sort": [
            {"field": "storeLocation", "direction": "asc"},
            {"field": "total_sales", "direction": "desc"},
        ],
        "limit": 1,
    })
    query = MongoQueryGenerator(CapturingRouter()).generate(
        question=(
            "Rank the regions by total sales in descending order and "
            "identify the top region."
        ),
        schema_context=[{
            "schema": "analytics",
            "table": "sales",
            "columns": [
                {"name": "storeLocation", "data_type": "string"},
                {"name": "items.price", "data_type": "decimal"},
                {"name": "items.quantity", "data_type": "integer"},
            ],
        }],
        query_intent=intent,
    )
    pipeline = json.loads(MongoQueryGenerator.serialize(query))["pipeline"]

    assert pipeline[2] == {"$sort": {"total_sales": -1, "_id": 1}}
    assert pipeline[3] == {"$limit": 1}


def test_mongo_generation_enforces_canonical_line_revenue_formula():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return GeneratedMongoQuery(
                collection="orders",
                pipeline=[
                    GeneratedMongoStage(
                        operator="$unwind",
                        specification_json='{"path":"$items"}',
                    ),
                    GeneratedMongoStage(
                        operator="$group",
                        specification_json=(
                            '{"_id":"$region","revenue":'
                            '{"$sum":"$items.unitPrice"}}'
                        ),
                    ),
                    GeneratedMongoStage(
                        operator="$limit",
                        specification_json="100",
                    ),
                ],
                fields_used=["items.unitPrice", "region"],
            )

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
    router = CapturingRouter()
    query = MongoQueryGenerator(router).generate(
        question="List regions by sales.",
        schema_context=[{
            "schema": "analytics",
            "table": "orders",
            "columns": [
                {"name": "region", "data_type": "string"},
                {"name": "items.unitPrice", "data_type": "decimal"},
                {"name": "items.quantity", "data_type": "integer"},
            ],
        }],
        query_intent=intent,
    )
    pipeline = json.loads(MongoQueryGenerator.serialize(query))["pipeline"]

    assert pipeline[1]["$group"]["revenue"] == {
        "$sum": {
            "$multiply": ["$items.unitPrice", "$items.quantity"],
        },
    }
    assert "formula" in router.kwargs["user_prompt"]


def test_mongo_formula_enforcement_fails_closed_without_group_stage():
    query = GeneratedMongoQuery(
        collection="orders",
        pipeline=[
            GeneratedMongoStage(
                operator="$limit",
                specification_json="100",
            ),
        ],
        fields_used=["unit_price", "quantity"],
    )
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "aggregate",
        "metrics": [{
            "operation": "sum",
            "formula": {
                "operator": "multiply",
                "fields": ["unit_price", "quantity"],
            },
            "alias": "revenue",
        }],
    })

    with pytest.raises(
        MetricFormulaEnforcementError,
        match="does not contain a \\$group stage",
    ):
        MongoQueryGenerator._enforce_metric_formulas(
            query,
            query_intent=intent,
        )


def test_relative_year_optimizer_keeps_all_entity_year_groups():
    query = GeneratedMongoQuery(
        collection="sales",
        pipeline=[
            GeneratedMongoStage(
                operator="$addFields",
                specification_json='{"year":{"$year":"$saleDate"}}',
            ),
            GeneratedMongoStage(
                operator="$group",
                specification_json=(
                    '{"_id":{"storeLocation":"$storeLocation","year":"$year"},'
                    '"total_sales":{"$sum":"$total"}}'
                ),
            ),
            GeneratedMongoStage(
                operator="$sort",
                specification_json='{"total_sales":-1}',
            ),
            GeneratedMongoStage(
                operator="$limit",
                specification_json="3",
            ),
        ],
        fields_used=["saleDate", "storeLocation", "total"],
    )

    repaired = MongoQueryGenerator._enforce_latest_year_order(query, 3)
    pipeline = json.loads(MongoQueryGenerator.serialize(repaired))["pipeline"]

    assert list(pipeline[-2]["$sort"].items()) == [
        ("_id.year", -1),
        ("_id", 1),
    ]
    assert pipeline[-1] == {"$limit": 100}


def test_relative_year_optimizer_preserves_group_id_for_projected_entity_sort():
    query = GeneratedMongoQuery(
        collection="sales",
        pipeline=[
            GeneratedMongoStage(
                operator="$addFields",
                specification_json='{"year":{"$year":"$saleDate"}}',
            ),
            GeneratedMongoStage(
                operator="$group",
                specification_json=(
                    '{"_id":{"storeLocation":"$storeLocation","year":"$year"},'
                    '"annual_sales":{"$sum":"$total"}}'
                ),
            ),
            GeneratedMongoStage(
                operator="$project",
                specification_json=(
                    '{"storeLocation":"$_id.storeLocation",'
                    '"year":"$_id.year","annual_sales":1,"_id":0}'
                ),
            ),
            GeneratedMongoStage(
                operator="$sort",
                specification_json='{"year":-1,"_id":1}',
            ),
            GeneratedMongoStage(
                operator="$limit",
                specification_json="100",
            ),
        ],
        fields_used=["saleDate", "storeLocation", "total"],
    )

    repaired = MongoQueryGenerator._enforce_latest_year_order(query, 3)
    pipeline = json.loads(MongoQueryGenerator.serialize(repaired))["pipeline"]

    assert pipeline[2]["$project"]["_id"] == 1
    assert list(pipeline[-2]["$sort"].items()) == [
        ("year", -1),
        ("_id", 1),
    ]
    assert pipeline[-1] == {"$limit": 100}


def test_metric_formula_enforcement_targets_earliest_relevant_group():
    query = GeneratedMongoQuery(
        collection="sales",
        pipeline=[
            GeneratedMongoStage(
                operator="$group",
                specification_json=(
                    '{"_id":{"location":"$location","year":"$year"},'
                    '"total_sales":{"$sum":"$items.price"}}'
                ),
            ),
            GeneratedMongoStage(
                operator="$group",
                specification_json=(
                    '{"_id":"$_id.location",'
                    '"annual_sales":{"$push":"$total_sales"}}'
                ),
            ),
            GeneratedMongoStage(
                operator="$limit",
                specification_json="100",
            ),
        ],
        fields_used=["items.price", "items.quantity", "location", "year"],
    )
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "trend",
        "periods": ["latest years"],
        "metrics": [{
            "operation": "sum",
            "formula": {
                "operator": "multiply",
                "fields": ["items.price", "items.quantity"],
            },
            "alias": "total_sales",
        }],
    })

    repaired = MongoQueryGenerator._enforce_metric_formulas(
        query,
        query_intent=intent,
    )
    pipeline = json.loads(MongoQueryGenerator.serialize(repaired))["pipeline"]

    assert pipeline[0]["$group"]["total_sales"] == {
        "$sum": {
            "$multiply": ["$items.price", "$items.quantity"],
        }
    }
    assert pipeline[1]["$group"]["annual_sales"] == {
        "$push": "$total_sales",
    }


def test_mongo_global_aggregate_is_complete_without_collection_sort():
    query = {
        "collection": "sales",
        "pipeline": [
            {"$match": {"items.name": "printer paper"}},
            {
                "$group": {
                    "_id": None,
                    "customer_emails": {"$addToSet": "$customer.email"},
                },
            },
            {
                "$project": {
                    "_id": 0,
                    "customer_count": {"$size": "$customer_emails"},
                },
            },
            {"$limit": 1},
        ],
    }

    result = SQLPipeline(None)._collect_mongo_pages(
        connection_id="unused",
        query_text=json.dumps(query),
        schema_context=[],
        initial_rows=[{"customer_count": 17}],
        organization_id=None,
        expected_top_n=None,
    )

    assert result == {
        "rows": [{"customer_count": 17}],
        "total_rows": 1,
        "complete": True,
        "pages": 1,
        "reason": "The global MongoDB aggregation produces at most one result row.",
    }


def test_mongo_global_aggregate_followed_by_unwind_is_not_assumed_complete():
    query = {
        "collection": "sales",
        "pipeline": [
            {"$group": {"_id": None, "customers": {"$addToSet": "$customer"}}},
            {"$unwind": "$customers"},
            {"$limit": 1},
        ],
    }

    result = SQLPipeline(None)._collect_mongo_pages(
        connection_id="unused",
        query_text=json.dumps(query),
        schema_context=[],
        initial_rows=[{"customers": "first"}],
        organization_id=None,
        expected_top_n=None,
    )

    assert not result["complete"]
    assert "deterministic $sort" in result["reason"]


def test_mongo_rank_repair_uses_minimum_metric_direction_from_intent():
    query = GeneratedMongoQuery(
        collection="sales",
        pipeline=[
            GeneratedMongoStage(
                operator="$unwind",
                specification_json='{"path":"$items"}',
            ),
            GeneratedMongoStage(
                operator="$group",
                specification_json=(
                    '{"_id":"$storeLocation","total_sales":'
                    '{"$sum":{"$multiply":'
                    '["$items.price","$items.quantity"]}}}'
                ),
            ),
            GeneratedMongoStage(
                operator="$sort",
                specification_json='{"_id":1,"total_sales":1}',
            ),
            GeneratedMongoStage(
                operator="$limit",
                specification_json="100",
            ),
        ],
        fields_used=["items.price", "items.quantity", "storeLocation"],
    )
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "ranking",
        "metrics": [{
            "operation": "sum",
            "field": "items.price",
            "alias": "total_sales",
        }],
        "dimensions": ["storeLocation"],
        "filters": [],
        "periods": [],
        "sort": [
            {"field": "storeLocation", "direction": "asc"},
            {"field": "total_sales", "direction": "asc"},
        ],
        "limit": 100,
        "include_all_extreme_ties": True,
    })

    repaired = MongoQueryGenerator._enforce_group_ranking(
        query,
        "Which region generates minimum sales?",
        query_intent=intent,
    )
    pipeline = json.loads(
        MongoQueryGenerator.serialize(repaired)
    )["pipeline"]

    assert pipeline[2] == {"$sort": {"total_sales": 1, "_id": 1}}
    assert list(pipeline[2]["$sort"]) == ["total_sales", "_id"]
    assert pipeline[3] == {"$limit": 100}


def test_sql_prompt_receives_only_supplied_live_schema():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return GeneratedSQL(
                sql=(
                    "SELECT record_date FROM active_schema.records "
                    "LIMIT 10"
                ),
                tables_used=["active_schema.records"],
                columns_used=["record_date"],
            )

    router = CapturingRouter()
    schema = [{
        "schema": "active_schema",
        "table": "records",
        "columns": [{"name": "record_date", "data_type": "date"}],
        "primary_key": ["record_id"],
        "foreign_keys": [],
    }]

    SQLGenerator(router).generate(
        question="List records in a date range.",
        schema_context=schema,
    )

    assert "active_schema.records" in router.kwargs["user_prompt"]
    assert "record_date" in router.kwargs["user_prompt"]
    assert router.kwargs["response_model"] is GeneratedSQL
    assert "active_schema.records" not in router.kwargs["system_instruction"]
    assert "FlyBase" not in router.kwargs["system_instruction"]

    SQLGenerator(router).generate(
        question="List records in a date range.",
        schema_context=schema,
        dialect="tsql",
    )
    assert "Target SQL dialect: tsql" in router.kwargs["system_instruction"]
    assert "use TOP (n)" in router.kwargs["system_instruction"]
    assert "PostgreSQL-only" not in router.kwargs["system_instruction"]


def test_sql_generation_receives_canonical_query_intent():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return GeneratedSQL(
                sql="SELECT record_date FROM active_schema.records LIMIT 10",
                tables_used=["active_schema.records"],
                columns_used=["record_date"],
            )

    router = CapturingRouter()
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "lookup",
        "metrics": [{
            "operation": "raw",
            "field": "record_date",
            "alias": "record_date",
        }],
        "dimensions": [],
        "filters": [],
        "periods": [],
        "sort": [],
        "limit": 10,
        "result_requirements": ["Return matching record dates."],
    })

    SQLGenerator(router).generate(
        question="List the record dates.",
        schema_context=[{
            "schema": "active_schema",
            "table": "records",
            "columns": [{"name": "record_date", "data_type": "date"}],
        }],
        query_intent=intent,
    )

    assert "Canonical analytical query intent" in router.kwargs["user_prompt"]
    assert '"field": "record_date"' in router.kwargs["user_prompt"]


def test_sql_generation_receives_metric_formula_contract():
    class CapturingRouter:
        def generate(self, **kwargs):
            self.kwargs = kwargs
            return GeneratedSQL(
                sql=(
                    "SELECT region, SUM(unit_price * quantity) AS revenue "
                    "FROM analytics.orders GROUP BY region LIMIT 100"
                ),
                tables_used=["analytics.orders"],
                columns_used=["region", "unit_price", "quantity"],
            )

    router = CapturingRouter()
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "aggregate",
        "metrics": [{
            "operation": "sum",
            "formula": {
                "operator": "multiply",
                "fields": ["unit_price", "quantity"],
            },
            "alias": "revenue",
        }],
        "dimensions": ["region"],
    })

    SQLGenerator(router).generate(
        question="List regions by sales.",
        schema_context=[{
            "schema": "analytics",
            "table": "orders",
            "columns": [
                {"name": "region", "data_type": "string"},
                {"name": "unit_price", "data_type": "decimal"},
                {"name": "quantity", "data_type": "integer"},
            ],
        }],
        query_intent=intent,
    )

    assert '"operator": "multiply"' in router.kwargs["user_prompt"]
    assert "When a metric includes a formula" in (
        router.kwargs["user_prompt"]
    )


def test_pipeline_replans_query_intent_after_failed_attempt(monkeypatch):
    class Manager:
        def get(self, connection_id, organization_id=None):
            return object()

    class CapturingPlanner:
        def __init__(self):
            self.calls = []

        def create_intent(self, **kwargs):
            self.calls.append(kwargs)
            return AnalyticalQueryIntent.model_validate({
                "kind": "aggregate",
                "metrics": [{
                    "operation": "count",
                    "field": None,
                    "alias": "customer_count",
                }],
            })

        @staticmethod
        def invalid_schema_references(intent, schema_context):
            return []

    pipeline = SQLPipeline(Manager())
    planner = CapturingPlanner()
    pipeline.query_intent_planner = planner
    monkeypatch.setattr(
        pipeline,
        "_generate_validate_and_execute_with_intent",
        lambda **kwargs: {"query_intent": kwargs["query_intent"].model_dump(mode="json")},
    )
    schema = [{
        "table": "sales",
        "columns": [{"name": "saleDate", "data_type": "date"}],
    }]
    first = pipeline.generate_validate_and_execute(
        connection_id="mongo",
        question="How many customers bought paper in 2017?",
        required_information=["customer count"],
        schema_context=schema,
    )
    pipeline.generate_validate_and_execute(
        connection_id="mongo",
        question="How many customers bought paper in 2017?",
        required_information=["customer count"],
        schema_context=schema,
        feedback="The prior pipeline could not establish completeness.",
        attempt_number=2,
        query_intent=first["query_intent"],
    )

    assert len(planner.calls) == 2
    assert planner.calls[1]["previous_intent"] == first["query_intent"]
    assert "could not establish completeness" in planner.calls[1]["feedback"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(unit_price * quantity) AS revenue FROM orders",
        "SELECT SUM(quantity * unit_price) AS revenue FROM orders",
        "SELECT SUM(o.unit_price * o.quantity) AS revenue FROM orders o",
    ],
)
def test_sql_formula_contract_accepts_equivalent_sum_expressions(sql):
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "aggregate",
        "metrics": [{
            "operation": "sum",
            "formula": {
                "operator": "multiply",
                "fields": ["unit_price", "quantity"],
            },
            "alias": "revenue",
        }],
    })

    assert SQLPipeline._metric_formula_issue(
        sql,
        "postgres",
        intent,
    ) is None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SUM(unit_price) AS revenue FROM orders",
        "SELECT SUM(unit_price * quantity) AS total FROM orders",
        "SELECT SUM(unit_price + quantity) AS revenue FROM orders",
    ],
)
def test_sql_formula_contract_rejects_incorrect_or_missing_expression(sql):
    intent = AnalyticalQueryIntent.model_validate({
        "kind": "aggregate",
        "metrics": [{
            "operation": "sum",
            "formula": {
                "operator": "multiply",
                "fields": ["unit_price", "quantity"],
            },
            "alias": "revenue",
        }],
    })

    assert SQLPipeline._metric_formula_issue(
        sql,
        "postgres",
        intent,
    ) is not None


def test_planner_removes_unrequested_period_analyses_from_any_plan():
    class PlannedRouter:
        def generate(self, **kwargs):
            return QuestionPlannerPlan.model_validate({
                "original_question": kwargs["user_prompt"],
                "goal": "Answer the customer request.",
                "sub_questions": [
                    {
                        "id": 1,
                        "question": "Find matching entities.",
                        "objective": "Apply the user's requested date range.",
                        "required_information": ["entity", "date"],
                        "depends_on": [],
                    },
                    {
                        "id": 2,
                        "question": "Find related items.",
                        "objective": "Return requested related records.",
                        "required_information": ["related items"],
                        "depends_on": [1],
                    },
                    {
                        "id": 3,
                        "question": "Count matching entities.",
                        "objective": "Return the requested count.",
                        "required_information": ["count"],
                        "depends_on": [1],
                    },
                    {
                        "id": 4,
                        "question": "Compare with the preceding period.",
                        "objective": "Measure period-to-period change.",
                        "required_information": [],
                        "depends_on": [1],
                        "optional": True,
                    },
                    {
                        "id": 5,
                        "question": "Compare segment differences.",
                        "objective": (
                            "Compare the same breakdown across "
                            "corresponding periods."
                        ),
                        "required_information": [],
                        "depends_on": [1, 4],
                        "optional": True,
                    },
                ],
            })

    from app.agent.planner import InvestigationPlan as QuestionPlannerPlan

    question = (
        "How many entities were created between 1970 and 1980, "
        "and list the related items?"
    )
    plan = QuestionPlanner(model_router=PlannedRouter()).create_plan(question)

    assert [step.id for step in plan.sub_questions] == [1, 2, 3]
    assert all(
        dependency in {1, 2, 3}
        for step in plan.sub_questions
        for dependency in step.depends_on
    )
