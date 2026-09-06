from sqlalchemy import inspect

from app.database.connection import engine


SEARCH_SCHEMAS = [
    "gene",
    "dataclass",
    "dataclass_relationship",
    "gene_group",
    "humanhealth",
]


def normalize(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def score_table(
    table_name: str,
    required_information: list[str]
) -> tuple[int, list[str]]:

    table_text = normalize(table_name)

    score = 0
    matched_terms = []

    for information in required_information:

        info_text = normalize(information)
        words = info_text.split()

        if info_text in table_text:
            score += 20
            matched_terms.append(information)
            continue

        if all(word in table_text for word in words):
            score += 10
            matched_terms.append(information)
            continue

        important_words = [
            word
            for word in words
            if len(word) > 3
        ]

        matched_words = [
            word
            for word in important_words
            if word in table_text
        ]

        if matched_words:
            score += len(matched_words) * 2
            matched_terms.append(information)

    if "relationship" in table_name.lower():
        score += 2

    return score, matched_terms


def find_relevant_tables(
    required_information: list[str],
    max_results: int = 3
):
    inspector = inspect(engine)

    results = []

    for schema in SEARCH_SCHEMAS:

        tables = inspector.get_table_names(
            schema=schema
        )

        for table in tables:

            score, matched_terms = score_table(
                table,
                required_information
            )

            if score == 0:
                continue

            results.append({
                "schema": schema,
                "table": table,
                "score": score,
                "matched_terms": matched_terms,
            })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:max_results]


def get_table_details(
    inspector,
    schema: str,
    table: str,
    score: int | None = None,
    matched_terms: list[str] | None = None,
):
    columns = inspector.get_columns(
        table,
        schema=schema
    )

    primary_key = inspector.get_pk_constraint(
        table,
        schema=schema
    )

    foreign_keys = inspector.get_foreign_keys(
        table,
        schema=schema
    )

    return {
        "schema": schema,
        "table": table,
        "score": score,
        "matched_terms": matched_terms or [],
        "columns": [
            {
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": column["nullable"],
            }
            for column in columns
        ],
        "primary_key": primary_key.get(
            "constrained_columns",
            []
        ),
        "foreign_keys": [
            {
                "columns": fk["constrained_columns"],
                "referred_schema": fk["referred_schema"],
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"],
            }
            for fk in foreign_keys
        ],
    }


def get_relevant_schema(
    required_information: list[str],
    max_results: int = 3,
):
    inspector = inspect(engine)

    candidates = find_relevant_tables(
        required_information,
        max_results=max_results,
    )

    schema_context = []
    added_tables = set()

    for candidate in candidates:

        schema = candidate["schema"]
        table = candidate["table"]

        table_key = (schema, table)

        if table_key in added_tables:
            continue

        details = get_table_details(
            inspector,
            schema,
            table,
            candidate["score"],
            candidate["matched_terms"],
        )

        schema_context.append(details)
        added_tables.add(table_key)

        # Follow foreign keys to discover tables needed for joins.
        for fk in details["foreign_keys"]:

            referred_schema = fk["referred_schema"]
            referred_table = fk["referred_table"]

            referred_key = (
                referred_schema,
                referred_table,
            )

            if referred_key in added_tables:
                continue

            referred_details = get_table_details(
                inspector,
                referred_schema,
                referred_table,
            )

            schema_context.append(
                referred_details
            )

            added_tables.add(referred_key)

    return schema_context


if __name__ == "__main__":

    required_information = [
        "FlyBase gene",
        "human ortholog",
    ]

    print(
        "Building relationship-aware schema context...\n",
        flush=True,
    )

    context = get_relevant_schema(
        required_information
    )

    for table in context:

        print(
            f"\nTABLE: "
            f"{table['schema']}.{table['table']}"
        )

        print("COLUMNS:")

        for column in table["columns"]:

            print(
                f"  {column['name']} "
                f"| {column['type']} "
                f"| nullable={column['nullable']}"
            )

        print(
            f"PRIMARY KEY: "
            f"{table['primary_key']}"
        )

        print("FOREIGN KEYS:")

        for fk in table["foreign_keys"]:
            print(f"  {fk}")