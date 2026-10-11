from app.analysis.explorer import DatabaseExplorer, DatabaseOverview


class FakeAdapter:
    def get_schemas(self):
        return ["public"]

    def get_tables(self, schema=None):
        return ["gene"]

    def get_schema(self, schema, table):
        return {
            "columns": [{"name": "gene_id", "type": "integer", "nullable": False}],
            "primary_key": ["gene_id"],
            "foreign_keys": [],
        }


class FakeModelRouter:
    def generate(self, **kwargs):
        return DatabaseOverview(
            summary="A database overview.",
            schemas=["model-invented"],
            tables=[{}],
            relationships=[{}],
            limitations=[],
        )


def test_explorer_preserves_database_metadata_from_introspection():
    result = DatabaseExplorer(FakeModelRouter()).explore(
        question="What tables are available?",
        adapter=FakeAdapter(),
    )

    assert result.schemas == ["public"]
    assert result.tables == [
        {
            "schema": "public",
            "table": "gene",
            "columns": [{"name": "gene_id", "type": "integer", "nullable": False}],
            "primary_keys": ["gene_id"],
            "foreign_keys": [],
        }
    ]
    assert result.relationships == []
