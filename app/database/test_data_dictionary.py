import json

import pytest

from app.database.data_dictionary import DatabaseDictionary
from app.database.schema_retriever import get_relevant_schema


class FakeAdapter:
    def get_schemas(self):
        return ["public"]

    def get_tables(self, schema=None):
        return ["gene"]

    def get_schema(self, schema, table):
        return {
            "schema": schema,
            "table": table,
            "columns": [
                {"name": "gene_id", "type": "integer"},
                {"name": "symbol", "type": "text"},
            ],
            "primary_key": ["gene_id"],
            "foreign_keys": [],
        }


def write_dictionary(tmp_path, payload):
    path = tmp_path / "dictionary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dictionary_is_validated_and_enriches_real_schema(tmp_path):
    path = write_dictionary(
        tmp_path,
        {
            "version": 1,
            "description": "Reference for a biological database.",
            "schemas": [
                {
                    "name": "public",
                    "description": "Shared biological data.",
                    "tables": [
                        {
                            "name": "gene",
                            "description": "One record per gene.",
                            "columns": [
                                {
                                    "name": "symbol",
                                    "description": "The published gene symbol.",
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )

    dictionary = DatabaseDictionary.load(path, FakeAdapter())
    enriched = dictionary.enrich_schema_context(
        [FakeAdapter().get_schema("public", "gene")]
    )[0]

    assert enriched["dictionary_schema_description"] == "Shared biological data."
    assert enriched["dictionary_description"] == "One record per gene."
    assert enriched["columns"][1]["dictionary_description"] == (
        "The published gene symbol."
    )
    assert enriched["columns"][0]["name"] == "gene_id"
    assert dictionary.documented_table_count == 1
    assert dictionary.documented_column_count == 1


@pytest.mark.parametrize(
    ("schema", "table", "column"),
    [
        ("missing", "gene", None),
        ("public", "missing", None),
        ("public", "gene", "missing"),
    ],
)
def test_dictionary_rejects_objects_not_in_live_database(
    tmp_path,
    schema,
    table,
    column,
):
    table_entry = {"name": table, "description": "A table."}
    if column:
        table_entry["columns"] = [
            {"name": column, "description": "A column."}
        ]
    path = write_dictionary(
        tmp_path,
        {
            "version": 1,
            "schemas": [
                {"name": schema, "tables": [table_entry]}
            ],
        },
    )

    with pytest.raises(ValueError, match="unknown database objects"):
        DatabaseDictionary.load(path, FakeAdapter())


def test_dictionary_requires_a_supported_version_and_nonempty_content(tmp_path):
    path = write_dictionary(tmp_path, {"version": 2, "schemas": []})
    with pytest.raises(ValueError, match="version to 1"):
        DatabaseDictionary.load(path, FakeAdapter())

    path = write_dictionary(tmp_path, {"version": 1, "schemas": []})
    with pytest.raises(ValueError, match="contains no descriptions"):
        DatabaseDictionary.load(path, FakeAdapter())


def test_dictionary_descriptions_help_match_business_terms_to_real_tables(
    tmp_path,
):
    path = write_dictionary(
        tmp_path,
        {
            "version": 1,
            "schemas": [
                {
                    "name": "public",
                    "tables": [
                        {
                            "name": "gene",
                            "description": "FlyBase gene records and approved identifiers.",
                        }
                    ],
                }
            ],
        },
    )
    dictionary = DatabaseDictionary.load(path, FakeAdapter())

    result = get_relevant_schema(
        adapter=FakeAdapter(),
        required_information=["approved identifiers"],
        data_dictionary=dictionary,
    )

    assert [table["table"] for table in result] == ["gene"]
    assert result[0]["dictionary_description"].startswith("FlyBase gene")
