from enum import Enum
import re


class AnalysisIntent(str, Enum):
    DATABASE_EXPLORATION = "database_exploration"
    DATA_ANALYSIS = "data_analysis"


class AnalysisIntentClassifier:
    """Classify broad analytical capability without database-specific rules."""

    _EXPLORATION_PATTERNS = (
        r"\bwhat(?:\s+does)?\s+(?:this|the)\s+(?:database|dataset)\s+(?:contain|contains|have|include)\b",
        r"\bwhat\s+(?:is|are)\s+(?:in|inside)\s+(?:this|the)\s+(?:database|dataset)\b",
        r"\bwhat\s+(?:this|the)\s+(?:database|dataset)\s+(?:contains|contain|has|have|includes|include)\b",
        r"\b(?:explain|describe|understand|overview)\s+(?:this|the)\s+(?:database|dataset|schema)\b",
        r"\bwhat\s+(?:tables|entities|collections)\s+(?:are\s+)?(?:available|present|there)\b",
        r"\bwhat\s+(?:(?:tables?|entities|collections|schemas?)(?:\s*,?\s*(?:and|,)\s*(?:tables?|entities|collections|schemas?))*)\s+(?:are\s+)?(?:available|present|there)\b",
        r"\bwhich\s+(?:tables?|entities|collections|schemas?)\s+(?:are\s+)?(?:available|present|there)\b",
        r"\bhow\s+(?:are|is)\s+(?:the\s+)?(?:tables|entities|collections)\s+(?:related|connected)\b",
        r"\bwhat\s+does\s+(?:each|the)\s+(?:table|entity|collection)\s+(?:represent|mean)\b",
        r"\b(?:i(?:'m| am) new to|beginner(?:'s)? guide|beginner[- ]friendly)\b.*\b(?:database|dataset|schema|structure|map)\b",
        r"\b(?:map|structure|layout)\s+(?:of|for)\s+(?:this|the)\s+(?:database|dataset|schema)\b",
    )

    @classmethod
    def classify(cls, question: str) -> AnalysisIntent:
        text = " ".join((question or "").lower().split())
        if any(re.search(pattern, text) for pattern in cls._EXPLORATION_PATTERNS):
            return AnalysisIntent.DATABASE_EXPLORATION
        return AnalysisIntent.DATA_ANALYSIS
