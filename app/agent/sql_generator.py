from app.llm.model_router import ModelRouter
from app.llm.model_router import create_default_model_router
from app.models.sql import GeneratedSQL


class SQLGenerator:

    def __init__(
        self,
        model_router: ModelRouter | None = None,
    ):
        self.model_router = (
            model_router
            or create_default_model_router()
        )

    def generate(
        self,
        question: str,
        schema_context: list[dict],
        complexity: str | None = None,
        feedback: str | None = None,
        evidence_context: list[dict] | None = None,
        previous_sqls: list[str] | None = None,
        attempt_number: int = 1,
    ) -> GeneratedSQL:

        if not schema_context:

            raise ValueError(
                "SQL generation requires real schema context."
            )

        schema_text = self._format_schema(
            schema_context
        )

        evidence_text = ""

        if evidence_context:

            evidence_text = f"""
Verified evidence from prerequisite investigation steps:

{evidence_context}

Use this evidence only as factual context. Do not invent IDs,
values, or relationships that are not present in the supplied
schema or verified evidence. Prefer a direct SQL join against
the supplied schema when the same result can be obtained without
materializing evidence values into SQL.

CRITICAL: prerequisite result rows are evidence, not a parameter
list. Do NOT copy returned IDs, names, or other row values into
IN (...), VALUES (...), CASE lists, or similar literal SQL unless
the ORIGINAL USER QUESTION explicitly requests those exact values.
When the same result can be derived from the database schema, use
the appropriate JOIN, EXISTS, aggregation, or relationship filter
instead. This keeps retries independent and prevents a verification
step from becoming hardcoded to the previous result sample.
"""

        previous_sql_text = ""

        if previous_sqls:

            previous_sql_text = f"""
Previous SQL attempts that failed:

{previous_sqls[-3:]}

IMPORTANT:
Do not repeat any of these SQL statements.

Use a materially different SQL strategy while preserving the
user's original intent.
"""

        feedback_text = ""

        if feedback:

            feedback_text = f"""
Previous attempt feedback:

{feedback}

Generate a corrected SQL query.
Do not repeat the previous mistake.
"""

        system_instruction = """
You are the SQL generation component of DB-Sentinel.

Your job is to generate ONE PostgreSQL SELECT query that answers
the user's question using ONLY the database schema supplied below.

IMPORTANT ARCHITECTURE RULE:

The database schema supplied to you is the ONLY source of truth.

You MUST NOT invent:
- tables
- schemas
- columns
- relationships
- aliases representing nonexistent tables

You MUST NOT assume conventional table names.

You MUST use the exact schema and table names supplied.

============================================================
CORE SQL RULES
============================================================

1. Generate exactly one SELECT statement.

2. Never generate:
   - INSERT
   - UPDATE
   - DELETE
   - DROP
   - ALTER
   - CREATE
   - TRUNCATE
   - GRANT
   - REVOKE
   - MERGE
   - UPSERT
   - transaction statements

3. Every referenced physical database table MUST be
   schema-qualified.

   Common table expressions (CTEs) are allowed and their
   query-local names do not need schema qualification.

4. Every referenced physical table MUST exist in the supplied
   schema.

   CTE references must refer only to CTEs defined in the same
   query.

5. Every referenced physical column MUST exist in the supplied
   schema.

6. Use supplied foreign-key relationships whenever a relationship
   between tables is required.

7. Do not invent joins.

8. Use the smallest number of tables necessary.

9. Do not use SELECT * unless genuinely necessary.

10. Return only the columns required to answer the question.

11. Always include LIMIT.

12. LIMIT must be between 1 and 100.

13. Never intentionally return more than 100 rows.

14. Prefer deterministic SQL.

15. Do not fabricate business logic that is not supported by the
    schema.

16. If the requested information cannot be answered using the
    supplied schema, generate the closest meaningful valid query.
    The deterministic validation layer will reject unsupported SQL.

17. If previous feedback is provided, correct the query accordingly.

18. If a previous attempt reports an unknown or invalid column,
    use ONLY the exact columns listed in the supplied schema.

    Never guess conventional columns such as:
    - id
    - name
    - created_at
    - primary_id

    unless they actually appear in the schema.

19. If a previous attempt reports a PostgreSQL syntax error,
    repair the exact SQL construct instead of merely changing
    aliases.

20. For PostgreSQL LATERAL table functions, use valid PostgreSQL
    syntax.

    For example:

        JOIN LATERAL (...) AS alias ON TRUE

    when an ON clause is required.

21. Never bypass security restrictions.

============================================================
COUNTING RULES
============================================================

22. When the question asks how many UNIQUE entities exist, use:

    COUNT(DISTINCT <entity_id>)

    rather than:

    COUNT(<entity_id>)

23. This applies especially to:
    - genes
    - users
    - customers
    - products
    - pathways
    - transactions
    - records
    - IDs
    - any entity identified by a primary key or foreign key

24. If joins can duplicate an entity, COUNT(DISTINCT ...) is REQUIRED.

25. When counting entities across a grouping dimension, count the
    distinct entity identifier within each group.

Generic pattern:

    SELECT
        <group_expression> AS <group_alias>,
        COUNT(DISTINCT <entity_identifier>) AS <metric_alias>
    FROM <schema-qualified-table>
    GROUP BY <group_expression>
    ORDER BY <metric_alias> DESC
    LIMIT 100;

26. Never use COUNT(*) for an entity-count question when the joined
    tables can contain duplicate entity rows.

27. If the schema identifies a primary key for the entity, prefer that
    primary key for COUNT(DISTINCT ...).

============================================================
NULL / ARRAY / JSON RULES
============================================================

28. When the question asks whether a JSON/JSONB array contains one
    or more elements, checking only:

        column IS NOT NULL

    is NOT sufficient.

29. For PostgreSQL JSONB arrays, prefer:

        column IS NOT NULL
        AND jsonb_array_length(column) > 0

    when the column is known to contain a JSONB array.

30. Do not count NULL or empty arrays as containing information.

31. When the question asks for entities "with human orthologs",
    "with mappings", "with relationships", or equivalent, make sure
    the relevant relationship value actually contains data rather
    than merely being non-NULL.

============================================================
RANKING RULES
============================================================

32. Words such as:
    - most
    - highest
    - largest
    - maximum
    - top
    - rank
    - ranking

    REQUIRE an ORDER BY using the relevant metric.

33. Words such as:
    - least
    - lowest
    - smallest
    - minimum

    REQUIRE an ORDER BY using the relevant metric in ASC order.

34. LIMIT alone does NOT establish ranking.

35. For "most", "highest", "largest", "maximum", or "top",
    use DESC.

36. For "least", "lowest", "smallest", or "minimum",
    use ASC.

37. When a question asks for "top N", use:

        ORDER BY <metric> DESC
        LIMIT N

38. When ranking aggregated entities, the ordering MUST use the
    calculated aggregate metric.

Generic pattern:

    SELECT
        <group_expression> AS <group_alias>,
        COUNT(DISTINCT <entity_identifier>) AS <metric_alias>
    FROM <schema-qualified-table>
    GROUP BY <group_expression>
    ORDER BY <metric_alias> DESC
    LIMIT 5;

39. Never claim a ranking without deterministic ORDER BY.

============================================================
FILTERING RULES
============================================================

40. Apply filters required by the user's question.

41. When filtering relationships represented by JSONB arrays,
    ensure the array is non-empty when the question requires
    the existence of at least one relationship.

42. Do not introduce arbitrary filters just to make a query return
    fewer rows.

43. Do not use placeholder/example IDs unless those IDs are actually
    present in the supplied evidence or explicitly supplied by the user.

============================================================
GROUPING RULES
============================================================

44. Every non-aggregated selected column must be appropriately grouped.

45. Aggregates must correspond directly to the user's requested
    measurement.

46. When grouping entities and counting related entities, use
    COUNT(DISTINCT related_entity_id) whenever duplicates are possible.

============================================================
OUTPUT RULES
============================================================

47. Return only a structured GeneratedSQL object containing:

    - sql
    - tables_used
    - columns_used

48. Do not return explanations outside the structured response.

The deterministic DB-Sentinel security layer will independently
validate the generated SQL before execution.
"""

        user_prompt = f"""
User question:

{question}

Available database schema:

{schema_text}

{evidence_text}

{feedback_text}

{previous_sql_text}

This is SQL generation attempt {attempt_number}.

If a previous attempt failed, repair the specific failure rather
than merely changing aliases or whitespace.

Before generating SQL, internally determine:

1. What entity is being requested?
2. Whether the question asks for a count.
3. Whether the count must be DISTINCT.
4. Whether the question requires ranking.
5. Whether DESC or ASC is required.
6. Whether JSON/JSONB existence requires checking for a non-empty
   array.
7. Which supplied foreign-key relationships are required.
8. Which exact columns should be returned.

Then generate the SQL.

Return a structured GeneratedSQL object containing:

- sql
- tables_used
- columns_used
"""

        return self.model_router.generate(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_model=GeneratedSQL,
            complexity=complexity,
            classification_text=question,
        )

    @staticmethod
    def _format_schema(
        schema_context: list[dict],
    ) -> str:

        sections = []

        for schema in schema_context:

            schema_name = schema.get(
                "schema",
                "unknown",
            )

            table_name = schema.get(
                "table",
                "unknown",
            )

            columns = schema.get(
                "columns",
                [],
            )

            primary_keys = schema.get(
                "primary_keys",
                [],
            )

            foreign_keys = schema.get(
                "foreign_keys",
                [],
            )

            lines = [
                f"SCHEMA: {schema_name}",
                f"TABLE: {schema_name}.{table_name}",
                "COLUMNS:",
            ]

            for column in columns:

                if isinstance(
                    column,
                    dict,
                ):

                    column_name = column.get(
                        "name",
                        column.get(
                            "column",
                            "unknown",
                        ),
                    )

                    data_type = column.get(
                        "type",
                        column.get(
                            "data_type",
                            "unknown",
                        ),
                    )

                    lines.append(
                        f"- {column_name} ({data_type})"
                    )

                else:

                    lines.append(
                        f"- {column}"
                    )

            if primary_keys:

                lines.append(
                    "PRIMARY KEYS:"
                )

                for key in primary_keys:

                    if isinstance(
                        key,
                        dict,
                    ):

                        lines.append(
                            f"- {key.get('column', key.get('name', key))}"
                        )

                    else:

                        lines.append(
                            f"- {key}"
                        )

            if foreign_keys:

                lines.append(
                    "FOREIGN KEYS:"
                )

                for foreign_key in foreign_keys:

                    if isinstance(
                        foreign_key,
                        dict,
                    ):

                        source_column = (
                            foreign_key.get(
                                "column",
                                foreign_key.get(
                                    "source_column",
                                    "unknown",
                                ),
                            )
                        )

                        target_schema = (
                            foreign_key.get(
                                "referred_schema",
                                foreign_key.get(
                                    "target_schema",
                                    "unknown",
                                ),
                            )
                        )

                        target_table = (
                            foreign_key.get(
                                "referred_table",
                                foreign_key.get(
                                    "target_table",
                                    "unknown",
                                ),
                            )
                        )

                        target_column = (
                            foreign_key.get(
                                "referred_column",
                                foreign_key.get(
                                    "target_column",
                                    "unknown",
                                ),
                            )
                        )

                        lines.append(
                            "- "
                            f"{source_column} -> "
                            f"{target_schema}."
                            f"{target_table}."
                            f"{target_column}"
                        )

                    else:

                        lines.append(
                            f"- {foreign_key}"
                        )

            sections.append(
                "\n".join(lines)
            )

        return "\n\n".join(sections)