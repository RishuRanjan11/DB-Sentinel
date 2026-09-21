import re

from app.database.adapters.base import DatabaseAdapter


def normalize(text: str) -> str:
    return (
        str(text)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9]+",
        normalize(text),
    )


def _column_name(column) -> str:
    if isinstance(column, dict):
        return str(
            column.get(
                "name",
                column.get(
                    "column",
                    "",
                ),
            )
        )

    return str(column)


def _foreign_key_text(foreign_key) -> str:
    if not isinstance(foreign_key, dict):
        return str(foreign_key)

    return " ".join(
        str(value)
        for key, value in foreign_key.items()
        if key in {
            "column",
            "source_column",
            "referred_column",
            "referred_table",
            "referred_schema",
            "target_column",
            "target_table",
            "target_schema",
        }
        and value
    )


def score_table(
    table_name: str,
    required_information: list[str],
    columns: list | None = None,
    foreign_keys: list | None = None,
) -> tuple[int, list[str]]:

    columns = columns or []
    foreign_keys = foreign_keys or []

    searchable_parts = [
        table_name,
        *[
            _column_name(column)
            for column in columns
        ],
        *[
            _foreign_key_text(foreign_key)
            for foreign_key in foreign_keys
        ],
    ]

    searchable_text = normalize(
        " ".join(searchable_parts)
    )

    searchable_tokens = set(
        tokenize(searchable_text)
    )

    table_tokens = set(
        tokenize(table_name)
    )

    score = 0
    matched_terms = []

    for information in required_information:

        info_text = normalize(
            information
        ).strip()

        if not info_text:
            continue

        words = tokenize(info_text)

        if not words:
            continue

        # Exact phrase.
        if info_text in searchable_text:
            score += 20
            matched_terms.append(information)
            continue

        # Individual meaningful terms.
        important_words = [
            word
            for word in words
            if len(word) > 2
        ]

        if not important_words:
            continue

        matches = [
            word
            for word in important_words
            if word in searchable_tokens
        ]

        if len(matches) == len(important_words):
            score += 15
            matched_terms.append(information)
            continue

        if len(matches) >= 2:
            score += min(
                12,
                len(matches) * 3,
            )
            matched_terms.append(information)
            continue

        # Single highly meaningful table/column term.
        if len(matches) == 1:
            score += 5
            matched_terms.append(information)

    return score, matched_terms


def find_relevant_tables(
    adapter: DatabaseAdapter,
    required_information: list[str],
    schemas: list[str] | None = None,
    max_results: int = 6,
):

    results = []

    if schemas is None:
        schemas = adapter.get_schemas()

    for schema in schemas:

        tables = adapter.get_tables(
            schema=schema
        )

        for table in tables:

            details = adapter.get_schema(
                schema=schema,
                table=table,
            )

            score, matched_terms = score_table(
                table_name=table,
                required_information=(
                    required_information
                ),
                columns=details.get(
                    "columns",
                    [],
                ),
                foreign_keys=details.get(
                    "foreign_keys",
                    [],
                ),
            )

            if score <= 0:
                continue

            results.append(
                {
                    "schema": schema,
                    "table": table,
                    "score": score,
                    "matched_terms": matched_terms,
                    "details": details,
                }
            )

    results.sort(
        key=lambda result: (
            -result["score"],
            result["schema"],
            result["table"],
        )
    )

    return results[:max_results]


def get_relevant_schema(
    adapter: DatabaseAdapter,
    required_information: list[str],
    schemas: list[str] | None = None,
    max_results: int = 6,
):

    candidates = find_relevant_tables(
        adapter=adapter,
        required_information=required_information,
        schemas=schemas,
        max_results=max_results,
    )

    schema_context = []

    added_tables: set[
        tuple[str, str]
    ] = set()

    for candidate in candidates:

        schema = candidate["schema"]
        table = candidate["table"]

        table_key = (
            schema,
            table,
        )

        if table_key in added_tables:
            continue

        details = candidate["details"]

        details = dict(details)

        details["score"] = (
            candidate["score"]
        )

        details["matched_terms"] = (
            candidate["matched_terms"]
        )

        schema_context.append(details)

        added_tables.add(table_key)

    # --------------------------------------------------
    # Include FK-related tables.
    # --------------------------------------------------

    index = 0

    while index < len(schema_context):

        details = schema_context[index]
        index += 1

        for foreign_key in details.get(
            "foreign_keys",
            [],
        ):

            if not isinstance(
                foreign_key,
                dict,
            ):
                continue

            referred_schema = (
                foreign_key.get(
                    "referred_schema"
                )
            )

            referred_table = (
                foreign_key.get(
                    "referred_table"
                )
            )

            if (
                not referred_schema
                or not referred_table
            ):
                continue

            referred_key = (
                referred_schema,
                referred_table,
            )

            if referred_key in added_tables:
                continue

            referred_details = (
                adapter.get_schema(
                    schema=referred_schema,
                    table=referred_table,
                )
            )

            schema_context.append(
                referred_details
            )

            added_tables.add(
                referred_key
            )

    return schema_context