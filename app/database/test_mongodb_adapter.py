import json
from datetime import datetime

import pytest
from bson import ObjectId

from app.database.adapters.mongodb import MongoDBAdapter
from app.database.connection import DatabaseConnection


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows
        self.aggregate_calls = []

    def find(self, *args, **kwargs):
        return FakeCursor(self.rows)

    def aggregate(self, pipeline, **kwargs):
        self.aggregate_calls.append((pipeline, kwargs))
        return iter(self.rows)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.limit_count = None

    def limit(self, count):
        self.limit_count = count
        return iter(self.rows[:count])


class FakeDatabase:
    def __init__(self, collection):
        self.collection = collection

    def list_collection_names(self):
        return ["people"]

    def __getitem__(self, name):
        assert name == "people"
        return self.collection


def make_adapter(rows):
    adapter = MongoDBAdapter(DatabaseConnection(
        connection_id="mongo",
        organization_id="test",
        workspace_id="test",
        name="mongo",
        database_type="mongodb",
        host="localhost",
        database_name="analytics",
    ))
    adapter.client = object()
    adapter.database = FakeDatabase(FakeCollection(rows))
    return adapter


def test_mongodb_adapter_returns_json_safe_extended_bson_values():
    adapter = make_adapter([{"_id": ObjectId("507f1f77bcf86cd799439011")}])

    rows = adapter.execute_read_only({
        "collection": "people",
        "pipeline": [{"$limit": 1}],
    })

    assert rows == [{"_id": {"$oid": "507f1f77bcf86cd799439011"}}]
    json.dumps(rows)


def test_mongodb_adapter_decodes_extended_json_dates_before_execution():
    adapter = make_adapter([])
    query = json.dumps({
        "collection": "people",
        "pipeline": [
            {
                "$match": {
                    "birthdate": {
                        "$gte": {"$date": "1970-01-01T00:00:00Z"},
                        "$lt": {"$date": "1981-01-01T00:00:00Z"},
                    }
                }
            },
            {"$limit": 1},
        ],
    })

    adapter.execute_read_only(query)

    executed_pipeline = adapter.database.collection.aggregate_calls[0][0]
    birthdate_filter = executed_pipeline[0]["$match"]["birthdate"]
    assert birthdate_filter["$gte"] == datetime(1970, 1, 1)
    assert birthdate_filter["$lt"] == datetime(1981, 1, 1)


def test_mongodb_adapter_rejects_write_stages_even_without_pipeline_validator():
    adapter = make_adapter([])

    with pytest.raises(ValueError, match="\\$out.*not permitted"):
        adapter.execute_read_only({
            "collection": "people",
            "pipeline": [
                {"$out": "leaked"},
                {"$limit": 1},
            ],
        })

    assert adapter.database.collection.aggregate_calls == []


def test_mongodb_schema_discovery_includes_nested_array_fields():
    from bson.decimal128 import Decimal128

    adapter = make_adapter([
        {
            "_id": 1,
            "saleDate": datetime(2017, 1, 1),
            "items": [
                {
                    "name": "item",
                    "price": Decimal128("12.50"),
                    "quantity": 2,
                }
            ],
        }
    ])

    schema = adapter.get_schema("analytics", "people")
    fields = {
        column["name"]: column["data_type"]
        for column in schema["columns"]
    }

    assert fields["saleDate"] == "date"
    assert fields["items"] == "array"
    assert fields["items.price"] == "decimal"
    assert fields["items.quantity"] == "int"
