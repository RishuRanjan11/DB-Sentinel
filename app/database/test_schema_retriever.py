from app.database.schema_retriever import get_relevant_schema


class FakeAdapter:
    def get_schemas(self):
        return ["public"]

    def get_tables(self, schema=None):
        return ["gene", "pathway"]

    def get_schema(self, schema, table):
        if table == "gene":
            return {
                "schema": "public",
                "table": "gene",
                "columns": [
                    {"name": "gene_id", "data_type": "integer"},
                    {"name": "symbol", "data_type": "text"},
                ],
                "primary_key": ["gene_id"],
                "foreign_keys": [],
            }
        return {
            "schema": "public",
            "table": "pathway",
            "columns": [{"name": "pathway_id", "data_type": "integer"}],
            "primary_key": ["pathway_id"],
            "foreign_keys": [],
        }


def test_plural_question_matches_singular_table_name():
    result = get_relevant_schema(
        adapter=FakeAdapter(),
        required_information=["How many genes are there?"],
    )

    assert result
    assert result[0]["table"] == "gene"


def test_single_table_schema_is_available_for_unmatched_business_terms():
    class SingleTableAdapter(FakeAdapter):
        def get_tables(self, schema=None):
            return ["tb_dr_surveillance"]

        def get_schema(self, schema, table):
            return {
                "schema": schema,
                "table": table,
                "columns": [
                    {"name": "country", "data_type": "text"},
                    {"name": "pulm_labconf_new", "data_type": "integer"},
                ],
                "primary_key": [],
                "foreign_keys": [],
            }

    result = get_relevant_schema(
        adapter=SingleTableAdapter(),
        required_information=["How many cases does India have?"],
    )

    assert len(result) == 1
    assert result[0]["table"] == "tb_dr_surveillance"


def test_unmatched_business_terms_do_not_fall_back_across_multiple_tables():
    result = get_relevant_schema(
        adapter=FakeAdapter(),
        required_information=["How many cases does India have?"],
    )

    assert result == []
