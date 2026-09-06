from sqlalchemy import inspect

from app.database.connection import engine


SEARCH_SCHEMAS = [
    "gene",
    "dataclass",
    "dataclass_relationship",
    "gene_group",
    "humanhealth",
]


def search_schema(search_terms: list[str]):
    inspector = inspect(engine)

    results = []

    for schema in SEARCH_SCHEMAS:
        tables = inspector.get_table_names(schema=schema)

        for table in tables:
            table_lower = table.lower()

            matched_terms = []

            for term in search_terms:
                term_lower = term.lower()

                # Handle multi-word concepts
                words = term_lower.split()

                if all(word in table_lower for word in words):
                    matched_terms.append(term)

            if not matched_terms:
                continue

            # Calculate relevance score
            score = 0

            for term in matched_terms:
                term_lower = term.lower()

                if term_lower.replace(" ", "_") in table_lower:
                    score += 10
                elif all(
                    word in table_lower
                    for word in term_lower.split()
                ):
                    score += 5

            # Extra relevance for relationship tables
            if "relationship" in schema.lower():
                score += 2

            results.append({
                "schema": schema,
                "table": table,
                "score": score,
                "matched_terms": matched_terms,
            })

    # Highest relevance first
    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results


if __name__ == "__main__":

    terms = [
        "gene",
        "human ortholog",
        "ortholog",
    ]

    print("Searching schema...\n", flush=True)

    results = search_schema(terms)

    for result in results:
        print(
            f"{result['score']:>3} | "
            f"{result['schema']}.{result['table']} "
            f"| {result['matched_terms']}"
        )