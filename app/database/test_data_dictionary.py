import json
import csv
from pathlib import Path

import pytest
from openpyxl import Workbook

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


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_tabular_dictionary_formats_are_validated_and_enrich_schema(
    tmp_path,
    suffix,
):
    path = tmp_path / f"dictionary{suffix}"
    headers = ["Schema Name", "Table Name", "Column Name", "Description"]
    rows = [
        ["public", "gene", "", "One record per gene."],
        ["public", "gene", "symbol", "The approved gene symbol."],
    ]

    if suffix == ".csv":
        import csv

        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(headers)
        for row in rows:
            sheet.append(row)
        workbook.save(path)

    dictionary = DatabaseDictionary.load(path, FakeAdapter())
    enriched = dictionary.enrich_schema_context(
        [FakeAdapter().get_schema("public", "gene")]
    )[0]

    assert enriched["dictionary_description"] == "One record per gene."
    assert enriched["columns"][1]["dictionary_description"] == (
        "The approved gene symbol."
    )


def test_tabular_dictionary_rejects_unknown_live_database_objects(tmp_path):
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "schema,table,column,description\n"
        "public,missing,,Not a real table.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown database objects"):
        DatabaseDictionary.load(path, FakeAdapter())


def test_tabular_dictionary_infers_unique_schema_and_rejects_duplicate_headers(
    tmp_path,
):
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "table,column,description\n"
        "gene,symbol,The approved gene symbol.\n",
        encoding="utf-8",
    )
    dictionary = DatabaseDictionary.load(path, FakeAdapter())
    enriched = dictionary.enrich_schema_context(
        [FakeAdapter().get_schema("public", "gene")]
    )[0]
    assert enriched["columns"][1]["dictionary_description"] == (
        "The approved gene symbol."
    )

    path.write_text(
        "schema,schema_name,table,description\n"
        "public,public,gene,A gene.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple columns for 'schema'"):
        DatabaseDictionary.load(path, FakeAdapter())


def test_who_variable_dictionary_maps_unique_live_columns(tmp_path):
    class TBAdapter(FakeAdapter):
        def get_schema(self, schema, table):
            return {
                **super().get_schema(schema, table),
                "columns": [
                    {"name": "budget_cpp_dstb", "data_type": "numeric"},
                ],
            }

    path = tmp_path / "tb_dictionary.csv"
    path.write_text(
        "variable_name,dataset,code_list,definition\n"
        '"budget_cpp_dstb",Budget,,Average cost per patient (USD)\n',
        encoding="utf-8",
    )

    dictionary = DatabaseDictionary.load(path, TBAdapter())
    enriched = dictionary.enrich_schema_context(
        [TBAdapter().get_schema("public", "gene")]
    )[0]

    assert dictionary.documented_column_count == 1
    assert enriched["columns"][0]["dictionary_description"] == (
        "Dataset: Budget. Average cost per patient (USD)"
    )


def test_who_variable_dictionary_includes_dataset_and_code_list(tmp_path):
    class TBAdapter(FakeAdapter):
        def get_schema(self, schema, table):
            return {
                **super().get_schema(schema, table),
                "columns": [
                    {"name": "is_reported", "data_type": "integer"},
                ],
            }

    path = tmp_path / "tb_dictionary.csv"
    path.write_text(
        "variable_name,dataset,code_list,definition\n"
        '"is_reported","Notifications","0=No; 1=Yes","Whether reported"\n',
        encoding="utf-8",
    )
    dictionary = DatabaseDictionary.load(path, TBAdapter())
    enriched = dictionary.enrich_schema_context(
        [TBAdapter().get_schema("public", "gene")]
    )[0]

    assert enriched["columns"][0]["dictionary_description"] == (
        "Dataset: Notifications. Whether reported. Code list: 0=No; 1=Yes"
    )


def test_who_variable_dictionary_rejects_ambiguous_live_column(tmp_path):
    class AmbiguousAdapter(FakeAdapter):
        def get_schemas(self):
            return ["public", "archive"]

        def get_tables(self, schema=None):
            return ["gene"] if schema in {"public", "archive"} else []

        def get_schema(self, schema, table):
            return {
                "schema": schema,
                "table": table,
                "columns": [{"name": "country", "data_type": "text"}],
            }

    path = tmp_path / "tb_dictionary.csv"
    path.write_text(
        "variable_name,dataset,definition\n"
        "country,Notification,Country name\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="column is ambiguous"):
        DatabaseDictionary.load(path, AmbiguousAdapter())


def test_supplied_tb_dictionary_file_loads_against_matching_live_metadata():
    path = (
        Path(__file__).resolve().parents[2]
        / "ExtraFiles"
        / "TB_data_dictionary_2026-01-18.csv"
    )
    if not path.exists():
        pytest.skip("Supplied TB data dictionary is not present.")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        variable_names = [
            row["variable_name"]
            for row in csv.DictReader(handle)
        ]

    class TBMetadataAdapter(FakeAdapter):
        def get_schema(self, schema, table):
            return {
                "schema": schema,
                "table": table,
                "columns": [
                    {"name": name, "data_type": "text"}
                    for name in variable_names
                ],
                "primary_key": [],
                "foreign_keys": [],
            }

    dictionary = DatabaseDictionary.load(path, TBMetadataAdapter())
    assert dictionary.documented_column_count == 698
