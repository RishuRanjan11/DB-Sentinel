from app.analysis.intent import AnalysisIntent, AnalysisIntentClassifier


def test_database_exploration_intent_is_generic():
    questions = [
        "What does this database contain?",
        "What this database contains",
        "Explain this database.",
        "What tables are available?",
        "What tables and entities are available in this database?",
        "Which schemas are available and what is in each one?",
        "How are the tables related?",
        "What does each table represent?",
        "I am new to this database. Give me a beginner-friendly map of its structure.",
    ]
    for question in questions:
        assert AnalysisIntentClassifier.classify(question) == AnalysisIntent.DATABASE_EXPLORATION
