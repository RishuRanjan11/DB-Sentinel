import re
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from app.llm.model_router import ModelRouter, create_default_model_router


class NormalizedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_question: str = Field(min_length=1)


class QuestionNormalizer:
    """Correct wording without changing the user's analytical constraints."""

    _NUMBER_PATTERN = re.compile(
        r"(?<![A-Za-z0-9_])\d+(?:[,.]\d+)*(?![A-Za-z0-9_])"
    )
    _QUOTED_PATTERN = re.compile(r"""(['"])(.*?)\1""")
    _NEGATION_PATTERN = re.compile(
        r"\b(?:not|no|never|without|excluding|exclude|except|only)\b",
        re.IGNORECASE,
    )
    _NUMBER_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    _RELATIVE_YEAR_PERIOD = re.compile(
        r"\b(?:last|past|previous|preceding)\s+"
        r"(?P<count>\d+|[a-z]+)\s+"
        r"(?:calendar\s+)?years?\s+"
        r"(?:from|ending(?:\s+in)?|through|up\s+to)\s+"
        r"(?P<end>(?:19|20)\d{2})\b",
        re.IGNORECASE,
    )
    _RELATIVE_YEAR_DURATION = re.compile(
        r"\b(?:last|past|previous|preceding)\s+"
        r"(?:(?P<count>\d+|[a-z]+)\s+)?"
        r"(?:calendar\s+)?years?\b",
        re.IGNORECASE,
    )

    def __init__(self, model_router: ModelRouter | None = None):
        self.model_router = model_router or create_default_model_router()

    @classmethod
    def _validate_preserved_constraints(
        cls,
        original: str,
        normalized: str,
    ) -> None:
        original_period = cls.relative_calendar_years(original)
        normalized_period = cls.relative_calendar_years(normalized)
        original_numbers = cls._numeric_constraints(original)
        normalized_numbers = cls._numeric_constraints(normalized)
        if original_numbers != normalized_numbers:
            raise ValueError(
                "Question normalization changed numeric constraints; "
                "the original question remains available for planning."
            )

        original_years = cls._explicit_years(original)
        normalized_years = cls._explicit_years(normalized)
        if not set(original_years) <= set(normalized_years):
            raise ValueError(
                "Question normalization removed an explicit year; "
                "the original question remains available for planning."
            )
        if original_period:
            if normalized_period and normalized_period != original_period:
                raise ValueError(
                    "Question normalization changed the relative calendar "
                    "period; the original question remains available for planning."
                )
            if not set(original_period) <= set(normalized_years):
                raise ValueError(
                    "Question normalization removed years implied by the "
                    "relative period; the original question remains "
                    "available for planning."
                )

        original_literals = [
            value.casefold()
            for _, value in cls._QUOTED_PATTERN.findall(original)
        ]
        normalized_literals = [
            value.casefold()
            for _, value in cls._QUOTED_PATTERN.findall(normalized)
        ]
        if not Counter(original_literals) <= Counter(normalized_literals):
            raise ValueError(
                "Question normalization changed a quoted filter value; "
                "the original question remains available for planning."
            )

        original_negations = Counter(
            match.group(0).casefold()
            for match in cls._NEGATION_PATTERN.finditer(original)
        )
        normalized_negations = Counter(
            match.group(0).casefold()
            for match in cls._NEGATION_PATTERN.finditer(normalized)
        )
        if original_negations != normalized_negations:
            raise ValueError(
                "Question normalization removed a negation or scope "
                "constraint; the original question remains available "
                "for planning."
            )

    @classmethod
    def _numeric_constraints(cls, question: str) -> Counter[str]:
        text = question
        relative_period = cls._RELATIVE_YEAR_PERIOD.search(text)
        if relative_period:
            text = (
                text[:relative_period.start()]
                + text[relative_period.end():]
            )
        years = set(cls._explicit_years(text))
        values = Counter(
            value.replace(",", "")
            for value in cls._NUMBER_PATTERN.findall(text)
            if not (
                len(value.replace(",", "")) == 4
                and value.replace(",", "").isdigit()
                and int(value.replace(",", "")) in years
            )
        )
        words = [
            str(value)
            for value in re.findall(r"\b[a-z]+\b", text, re.IGNORECASE)
            if value.lower() in cls._NUMBER_WORDS
        ]
        for word in words:
            values[str(cls._NUMBER_WORDS[word.lower()])] += 1
        return values

    @classmethod
    def _number_value(cls, value: str) -> int | None:
        if value.isdigit():
            return int(value)
        return cls._NUMBER_WORDS.get(value.lower())

    @classmethod
    def relative_calendar_years(cls, question: str) -> list[int] | None:
        match = cls._RELATIVE_YEAR_PERIOD.search(question or "")
        if not match:
            return None
        duration = cls._number_value(match.group("count") or "one")
        end_year = int(match.group("end"))
        if duration is None or not 1 <= duration <= 50:
            return None
        return list(range(end_year - duration + 1, end_year + 1))

    @classmethod
    def relative_calendar_duration(cls, question: str) -> int | None:
        match = cls._RELATIVE_YEAR_DURATION.search(question or "")
        if not match:
            return None
        duration = cls._number_value(match.group("count") or "one")
        if duration is None or not 1 <= duration <= 50:
            return None
        return duration

    @classmethod
    def relative_comparison_year_count(cls, question: str) -> int | None:
        duration = cls.relative_calendar_duration(question)
        if duration is None:
            return None
        requires_comparison = bool(
            re.search(
                r"\b(?:growth|grow|increased?|decreased?|change|changed|"
                r"compare|comparison|trend|positive|negative|higher|lower|"
                r"rise|rose|fall|fell|decline|declined)\b",
                question or "",
                re.IGNORECASE,
            )
        )
        return max(2, duration) if requires_comparison else duration

    @classmethod
    def _explicit_years(cls, question: str) -> list[int]:
        return sorted({
            int(year)
            for year in re.findall(r"\b(?:19|20)\d{2}\b", question or "")
        })

    @classmethod
    def _make_relative_period_explicit(
        cls,
        original: str,
        normalized: str,
    ) -> str:
        years = cls.relative_calendar_years(original)
        match = cls._RELATIVE_YEAR_PERIOD.search(original)
        if not years or not match:
            return normalized
        year_labels = ", ".join(map(str, years[:-1]))
        if len(years) > 1:
            year_labels += f", and {years[-1]}"
        canonical = f"calendar years {year_labels} inclusive"
        normalized_match = cls._RELATIVE_YEAR_PERIOD.search(normalized)
        if normalized_match:
            return (
                normalized[:normalized_match.start()]
                + canonical
                + normalized[normalized_match.end():]
            )
        return f"{normalized.rstrip(' .?!')} ({canonical})?"

    def normalize(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        original = (question or "").strip()
        if not original:
            raise ValueError("Question cannot be empty.")

        history = conversation_history or []
        history_text = "\n".join(
            f"Question: {item.get('question', '')}\n"
            f"Result summary: {item.get('result', {})}"
            for item in history[-5:]
        )
        result = self.model_router.generate(
            system_instruction=(
                "Rewrite the user's database-analysis question into clear, "
                "grammatically correct English. Correct spelling and "
                "grammar only. Preserve the user's exact intent, entities, "
                "metrics, filters, date ranges, years, quantities, ranking "
                "direction, comparisons, negation, and requested output. "
                "Do not answer the question, add analysis, infer missing "
                "details, or change its scope. Use conversation context "
                "only to resolve an explicit follow-up reference. Return "
                "one concise question."
            ),
            user_prompt=(
                f"Original user question:\n{original}\n\n"
                f"Recent conversation context:\n"
                f"{history_text or 'No previous conversation context.'}"
            ),
            response_model=NormalizedQuestion,
            complexity="fast",
            classification_text=original,
        )
        model_normalized = result.normalized_question.strip()
        if not model_normalized:
            raise ValueError("Question normalization returned an empty question.")
        normalized = self._make_relative_period_explicit(
            original,
            model_normalized,
        )
        try:
            self._validate_preserved_constraints(original, normalized)
        except ValueError as error:
            from app.core.trace import logger

            logger.warning(
                "Question normalization rejected; using original wording. "
                "reason=%s",
                error,
            )
            return original
        return normalized
