import json

import pytest

from app.security.mongodb_validator import MongoQuerySecurityValidator


def make_validator():
    schema = [{
        "schema": "analytics",
        "table": "people",
        "columns": [
            {"name": "_id", "data_type": "objectId"},
            {"name": "age", "data_type": "int"},
            {"name": "name", "data_type": "string"},
            {"name": "password_hash", "data_type": "string"},
        ],
    }, {
        "schema": "analytics",
        "table": "visits",
        "columns": [{"name": "person_id", "data_type": "objectId"}],
    }]
    return MongoQuerySecurityValidator(
        allowed_collections={
            ("analytics", "people"),
            ("analytics", "visits"),
        },
        schema_context=schema,
    )


def query(pipeline, collection="people"):
    return json.dumps({"collection": collection, "pipeline": pipeline})


def test_mongodb_validator_accepts_bounded_read_only_pipeline():
    result = make_validator().validate(query([
        {"$match": {"age": {"$gte": 18}}},
        {"$sort": {"age": -1, "_id": 1}},
        {"$limit": 20},
    ]))

    assert result.allowed


@pytest.mark.parametrize(
    ("pipeline", "reason"),
    [
        ([{"$match": {}}, {"$out": "leaked"}, {"$limit": 1}], "not permitted"),
        ([{"$match": {"password_hash": {"$exists": True}}}, {"$limit": 1}], "Sensitive field"),
        ([{"$match": {}}, {"$limit": 101}], "between 1 and 100"),
        ([{"$match": {}}], "top-level $limit"),
    ],
)
def test_mongodb_validator_rejects_unsafe_or_unbounded_pipeline(pipeline, reason):
    result = make_validator().validate(query(pipeline))

    assert not result.allowed
    assert reason in result.reason


def test_mongodb_validator_restricts_lookup_to_allowed_collections():
    allowed = make_validator().validate(query([
        {
            "$lookup": {
                "from": "visits",
                "localField": "_id",
                "foreignField": "person_id",
                "as": "visits",
            }
        },
        {"$limit": 10},
    ]))
    denied = make_validator().validate(query([
        {
            "$lookup": {
                "from": "private_collection",
                "localField": "_id",
                "foreignField": "person_id",
                "as": "matches",
            }
        },
        {"$limit": 10},
    ]))

    assert allowed.allowed
    assert not denied.allowed


def test_mongodb_validator_rejects_whole_document_projection():
    result = make_validator().validate(query([
        {"$project": {"record": "$$ROOT"}},
        {"$limit": 10},
    ]))

    assert not result.allowed


def test_mongodb_validator_blocks_server_side_javascript_expression():
    result = make_validator().validate(query([
        {
            "$project": {
                "computed": {
                    "$function": {
                        "body": "function() { return this.age; }",
                        "args": [],
                        "lang": "js",
                    }
                }
            }
        },
        {"$limit": 10},
    ]))

    assert not result.allowed
    assert "$function" in result.reason


@pytest.mark.parametrize(
    "stage",
    [
        {"$match": {"unknown_field": {"$exists": True}}},
        {"$sort": {"unknown_field": 1}},
        {"$project": {"unknown_field": 1}},
        {"$project": {"renamed": "$unknown_field"}},
    ],
)
def test_mongodb_validator_rejects_fields_not_in_live_schema(stage):
    result = make_validator().validate(
        query([stage, {"$limit": 10}])
    )

    assert not result.allowed
    assert "not present in the live collection schema" in result.reason


def test_mongodb_validator_accepts_pipeline_aliases_and_nested_match():
    result = make_validator().validate(query([
        {"$project": {"display_name": "$name", "age": 1}},
        {"$match": {"$or": [
            {"display_name": {"$exists": True}},
            {"age": {"$gte": 18}},
        ]}},
        {"$sort": {"display_name": 1, "_id": 1}},
        {"$limit": 10},
    ]))

    assert result.allowed


def test_mongodb_validator_preserves_fields_after_exclusion_projection():
    result = make_validator().validate(query([
        {"$project": {"age": 0}},
        {"$match": {"name": {"$exists": True}}},
        {"$limit": 10},
    ]))

    assert result.allowed


def test_mongodb_validator_rejects_unknown_lookup_join_fields():
    result = make_validator().validate(query([
        {
            "$lookup": {
                "from": "visits",
                "localField": "unknown_id",
                "foreignField": "person_id",
                "as": "visits",
            }
        },
        {"$limit": 10},
    ]))

    assert not result.allowed
    assert "localField" in result.reason
