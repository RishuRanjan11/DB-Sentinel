import json

from google.genai import types

from app.agent.llm_client import get_gemini_client
from app.models.sql import GeneratedSQL


class SQLGenerator:

    def __init__(self):
        self.client = get_gemini_client()

    def generate(
        self,
        question: str,
        schema_context: list[dict],
    ) -> GeneratedSQL:

        schema_text = json.dumps(
            schema_context,
            indent=2,
            default=str
        )

        system_instruction = """
You are the SQL generation component of DB-Sentinel.

Your task is to generate PostgreSQL SQL that answers
the user's analytical question using ONLY the database
schema supplied to you.

STRICT RULES:

1. Generate exactly one SELECT statement.

2. Never generate:
   INSERT
   UPDATE
   DELETE
   DROP
   ALTER
   CREATE
   TRUNCATE
   GRANT
   REVOKE

3. Use ONLY tables present in the supplied schema.

4. Use ONLY columns present in the supplied schema.

5. Never invent tables or columns.

6. Use explicit schema-qualified table names.

7. Use the foreign-key relationships supplied in the schema
   when relationships are required.

8. Always include a LIMIT clause.

9. LIMIT must never exceed 100.

10. Prefer the smallest number of tables necessary to answer
    the question.

11. Do not use SELECT * unless absolutely necessary.

12. Return only the requested structured output.

IMPORTANT:

The generated SQL is only a proposal.

It will be independently parsed and validated by
DB-Sentinel's deterministic security layer before
execution.

Do not assume that your own SQL is trusted.
"""

        user_prompt = f"""
User question:

{question}

Available database schema:

{schema_text}

Generate a PostgreSQL SELECT query that answers
the user's question.
"""

        response = self.client.models.generate_content(
            model="gemini-3.7-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0,
                response_mime_type="application/json",
                response_schema=GeneratedSQL,
            ),
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        return GeneratedSQL.model_validate_json(
            response.text
        )

if __name__ == "__main__":

    from app.database.schema_retriever import get_relevant_schema

    question = "Which FlyBase genes have human orthologs?"

    required_information = [
        "FlyBase gene",
        "human ortholog",
    ]

    schema_context = get_relevant_schema(
        required_information
    )

    generator = SQLGenerator()

    result = generator.generate(
        question=question,
        schema_context=schema_context,
    )

    print("\nGenerated SQL:")
    print(result.sql)

    print("\nTables used:")
    for table in result.tables_used:
        print(table)

    print("\nColumns used:")
    for column in result.columns_used:
        print(column)