from enum import Enum


class QueryComplexity(str, Enum):
    SIMPLE = "fast"
    COMPLEX = "pro"


def classify_question(question: str) -> str:
    """
    Deterministically classify a database question.

    Philosophy:
        Fast model -> straightforward database questions.
        Pro model  -> multi-step reasoning, joins, comparisons,
                      rankings, investigation, optimization, or
                      evidence-based explanations.

    The classifier intentionally errs toward COMPLEX when the
    question contains signals that require deeper reasoning.
    """

    if not question or not question.strip():
        return QueryComplexity.SIMPLE.value

    text = " ".join(question.lower().strip().split())

    score = 0

    # ---------------------------------------------------------
    # 1. Explicit multi-step / investigation requests
    # ---------------------------------------------------------

    strong_complex_patterns = [
        "if the first result",
        "investigate further",
        "investigate",
        "rerun",
        "re-run",
        "retry",
        "try again",
        "optimize the query",
        "optimize",
        "verify",
        "verified answer",
        "insufficient",
        "inconsistent",
        "check the result",
        "validate the result",
        "double check",
        "cross-check",
        "cross check",
        "step by step",
        "multiple questions",
        "subquestion",
        "sub-question",
    ]

    for pattern in strong_complex_patterns:
        if pattern in text:
            score += 3

    # ---------------------------------------------------------
    # 2. Comparison / analytical reasoning
    # ---------------------------------------------------------

    analytical_patterns = [
        "compare",
        "comparison",
        "compared with",
        "versus",
        "vs",
        "difference between",
        "differences between",
        "trend",
        "over time",
        "correlation",
        "correlated",
        "relationship between",
        "relationship among",
        "distribution",
        "variation",
        "variations",
        "anomaly",
        "anomalies",
        "outlier",
        "outliers",
        "analyze",
        "analysis",
        "analyse",
        "why",
        "explain why",
        "reason",
        "reasons",
        "cause",
        "causes",
        "driven by",
        "factor",
        "factors",
        "impact",
        "effect",
        "effects",
        "pattern",
        "patterns",
    ]

    for pattern in analytical_patterns:
        if pattern in text:
            score += 2

    # ---------------------------------------------------------
    # 3. Ranking / top-N questions
    # ---------------------------------------------------------

    ranking_patterns = [
        "top ",
        "top",
        "highest",
        "lowest",
        "most",
        "least",
        "maximum",
        "minimum",
        "largest",
        "smallest",
        "best",
        "worst",
        "rank",
        "ranking",
        "ranked",
    ]

    for pattern in ranking_patterns:
        if pattern in text:
            score += 2

    # ---------------------------------------------------------
    # 4. Multi-table / relationship signals
    # ---------------------------------------------------------

    relationship_patterns = [
        "join",
        "joins",
        "joined",
        "related to",
        "related with",
        "associated with",
        "linked to",
        "linked with",
        "mapped to",
        "mapped with",
        "belong to",
        "belonging to",
        "across tables",
        "multiple tables",
        "multiple datasets",
        "multiple sources",
        "between tables",
        "among tables",
        "from both",
        "from each",
        "for each",
        "per ",
    ]

    for pattern in relationship_patterns:
        if pattern in text:
            score += 2

    # ---------------------------------------------------------
    # 5. SQL / aggregation complexity
    # ---------------------------------------------------------

    sql_complexity_patterns = [
        "group by",
        "having",
        "subquery",
        "sub-query",
        "nested query",
        "nested queries",
        "recursive",
        "window function",
        "partition by",
        "distinct",
        "aggregate",
        "aggregation",
        "count by",
        "average by",
        "sum by",
        "median by",
    ]

    for pattern in sql_complexity_patterns:
        if pattern in text:
            score += 2

    # ---------------------------------------------------------
    # 6. Prediction / advanced analysis
    # ---------------------------------------------------------

    predictive_patterns = [
        "predict",
        "prediction",
        "forecast",
        "forecasting",
        "future",
        "estimate",
        "estimation",
        "probability",
        "classification",
        "machine learning",
    ]

    for pattern in predictive_patterns:
        if pattern in text:
            score += 3

    # ---------------------------------------------------------
    # 7. Multiple-question indicators
    # ---------------------------------------------------------

    question_indicators = [
        "?",
        " and ",
        " then ",
        " also ",
        " as well as ",
        "along with",
    ]

    indicator_count = sum(
        1 for pattern in question_indicators if pattern in text
    )

    if indicator_count >= 2:
        score += 2

    # ---------------------------------------------------------
    # 8. Long questions usually contain more reasoning
    # ---------------------------------------------------------

    word_count = len(text.split())

    if word_count > 40:
        score += 2
    elif word_count > 25:
        score += 1

    # ---------------------------------------------------------
    # 9. Direct simple-question protection
    #
    # These should normally stay on the fast model unless
    # another strong complexity signal exists.
    # ---------------------------------------------------------

    simple_patterns = [
        "how many",
        "what is the count",
        "count of",
        "list",
        "show me",
        "which",
        "find",
        "get",
    ]

    has_simple_pattern = any(
        pattern in text for pattern in simple_patterns
    )

    # A simple request containing a strong investigation/
    # analytical signal is still complex.
    if has_simple_pattern and score <= 1:
        return QueryComplexity.SIMPLE.value

    # ---------------------------------------------------------
    # Final decision
    # ---------------------------------------------------------

    if score >= 3:
        return QueryComplexity.COMPLEX.value

    return QueryComplexity.SIMPLE.value


def required_capabilities(
    complexity: str,
) -> set[str]:
    """
    Return capabilities required for SQL generation.

    Both fast and pro models currently need the same
    structured SQL-generation capabilities.
    """

    return {
        "sql_generation",
        "structured_output",
    }