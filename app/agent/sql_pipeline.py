from app.agent.sql_generator import SQLGenerator
from app.security.sql_validator import SQLSecurityValidator


class SQLPipeline:

    def __init__(self):
        self.generator = SQLGenerator()
        self.validator = SQLSecurityValidator()

    def generate_and_validate(
        self,
        question: str,
        schema_context: list[dict],
    ):

        # --------------------------------------------
        # 1. Ask Gemini to generate SQL
        # --------------------------------------------

        generated = self.generator.generate(
            question=question,
            schema_context=schema_context,
        )

        sql = generated.sql

        # --------------------------------------------
        # 2. Independently validate generated SQL
        # --------------------------------------------

        validation = self.validator.validate(sql)

        # --------------------------------------------
        # 3. Return both proposal and security result
        # --------------------------------------------

        return {
            "question": question,
            "sql": sql,
            "tables_used": generated.tables_used,
            "columns_used": generated.columns_used,
            "allowed": validation.allowed,
            "reason": validation.reason,
        }

if __name__ == "__main__":

    from app.database.schema_retriever import get_relevant_schema

    question = "Which FlyBase genes have human orthologs?"

    schema_context = get_relevant_schema(
        [
            "FlyBase gene",
            "human ortholog",
        ]
    )

    pipeline = SQLPipeline()

    result = pipeline.generate_and_validate(
        question=question,
        schema_context=schema_context,
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nGenerated SQL:")
    print(result["sql"])

    print("\nSecurity decision:")
    print(result["allowed"])

    print("\nReason:")
    print(result["reason"])