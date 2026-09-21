import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.evidence import Evidence
from app.agent.insight_generator import GeneratedInsight


class InsightVerificationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verified: bool
    confidence: float
    supported_findings: list[str] = Field(default_factory=list)
    unsupported_findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class InsightVerifier:
    """Deterministically verify an insight against verified evidence.

    No database/schema/domain names are embedded here. Verification uses only
    the returned evidence rows and the natural-language user question.
    """

    NUMBER_PATTERN = re.compile(
        r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])"
    )
    CAUSAL_PATTERNS = (
        r"\bcauses\b",
        r"\bcaused by\b",
        r"\bdrives\b",
        r"\bdriven by\b",
        r"\bleads to\b",
        r"\bresponsible for\b",
    )

    def __init__(self, model_router=None):
        self.model_router = model_router

    @staticmethod
    def _normalize(value: Any) -> str:
        return "" if value is None else str(value).strip().lower()

    @staticmethod
    def _normalize_number(value: Any) -> str:
        try:
            number = float(value)
            return str(int(number)) if number.is_integer() else f"{number:g}"
        except (TypeError, ValueError):
            return str(value).strip()

    @classmethod
    def _number_tokens(cls, text: str) -> list[str]:
        return [cls._normalize_number(v) for v in cls.NUMBER_PATTERN.findall(text)]

    @classmethod
    def _rows(cls, evidence: list[Evidence]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in evidence:
            for row in item.rows or []:
                if isinstance(row, dict):
                    rows.append({str(k): v for k, v in row.items()})
        return rows

    @classmethod
    def _numeric_columns(cls, rows: list[dict[str, Any]]) -> list[str]:
        result: set[str] = set()
        for row in rows:
            for key, value in row.items():
                if value is None:
                    continue
                try:
                    float(value)
                except (TypeError, ValueError):
                    continue
                result.add(str(key))
        return sorted(result)

    @classmethod
    def _string_columns(cls, rows: list[dict[str, Any]]) -> list[str]:
        result: set[str] = set()
        for row in rows:
            for key, value in row.items():
                if value is None:
                    continue
                try:
                    float(value)
                    continue
                except (TypeError, ValueError):
                    result.add(str(key))
        return sorted(result)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
            if len(token) > 1
        }

    @classmethod
    def _choose_entity_column(cls, question: str, rows: list[dict[str, Any]]) -> str | None:
        columns = cls._string_columns(rows)
        if not columns:
            return None
        q_tokens = cls._tokens(question)
        scored: list[tuple[float, str]] = []
        for column in columns:
            values = [str(row[column]) for row in rows if row.get(column) is not None]
            if not values:
                continue
            uniqueness = len(set(values)) / max(1, len(values))
            lexical = len(cls._tokens(column) & q_tokens)
            scored.append((uniqueness * 5 + lexical * 3, column))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1] if scored else None

    @classmethod
    def _choose_metric_column(cls, question: str, rows: list[dict[str, Any]]) -> str | None:
        columns = cls._numeric_columns(rows)
        if not columns:
            return None
        q_tokens = cls._tokens(question)
        asks_count = bool(re.search(r"\b(?:how many|number|count|counts)\b", question, re.I))
        scored: list[tuple[float, str]] = []
        for column in columns:
            tokens = cls._tokens(column)
            score = len(tokens & q_tokens) * 10
            if asks_count:
                score += sum(2 for token in tokens if token in {"count", "number", "total", "num"})
            scored.append((score, column))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1] if scored else None

    @classmethod
    def _row_facts(cls, question: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entity_column = cls._choose_entity_column(question, rows)
        numeric_columns = cls._numeric_columns(rows)
        facts = []
        for row in rows:
            entity = row.get(entity_column) if entity_column else None
            numbers: dict[str, str] = {}
            for column in numeric_columns:
                value = row.get(column)
                if value is None:
                    continue
                numbers[column] = cls._normalize_number(value)
            facts.append({
                "entity": None if entity is None else str(entity),
                "entity_column": entity_column,
                "numbers": numbers,
                "row": row,
            })
        return facts

    @classmethod
    def _evidence_number_set(cls, rows: list[dict[str, Any]]) -> set[str]:
        result: set[str] = set()
        for row in rows:
            for value in row.values():
                if value is None:
                    continue
                result.update(cls._number_tokens(str(value)))
        return result

    @classmethod
    def _verify_aggregate_numeric_claims(
        cls,
        claim_text: str,
        rows: list[dict[str, Any]],
    ) -> tuple[list[str], list[str], list[str]]:
        """Verify numeric claims for aggregate/single-row evidence."""
        numbers = cls._number_tokens(claim_text)
        if not numbers:
            return [], [], []
        evidence_numbers = cls._evidence_number_set(rows)
        unsupported = [
            f"{number} is not present in verified evidence"
            for number in numbers
            if number not in evidence_numbers
        ]
        return (
            [f"{number} is supported by verified evidence" for number in numbers if number in evidence_numbers],
            unsupported,
            [],
        )

    @classmethod
    def _claim_segments(cls, text: str) -> list[str]:
        if not text:
            return []
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = re.split(r"(?:\n+|[•▪◦]\s*|;\s*|(?<!\d)\.(?!\d))", text)
        return [p.strip(" \t-–—") for p in parts if p.strip(" \t-–—")]

    @classmethod
    def _explicit_entity_number_pairs(
        cls,
        claim_text: str,
        entities: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        """Associate numbers only with their local entity claim.

        This prevents a number belonging to one evidence row from being
        attributed to the preceding entity merely because both occur in a
        large character window.
        """
        pairs: list[tuple[str, str]] = []

        for segment in cls._claim_segments(claim_text):
            present = [
                fact for fact in entities
                if fact["entity"]
                and str(fact["entity"]).lower() in segment.lower()
            ]
            if not present:
                continue

            for fact in present:
                entity = str(fact["entity"])
                explicit_patterns = (
                    rf"{re.escape(entity)}\s*(?:->|→|=>|:|=)\s*(-?\d+(?:\.\d+)?)",
                    rf"{re.escape(entity)}\s+(?:is|are|has|have|contains|contain|with|"
                    rf"count(?:s)?|number(?:s)?|value(?:s)?|total(?:s)?|rank(?:ed)?|ranking)"
                    rf"\s+(?:about\s+|approximately\s+)?(-?\d+(?:\.\d+)?)",
                    rf"(-?\d+(?:\.\d+)?)\s+(?:for|of)\s+{re.escape(entity)}",
                    rf"{re.escape(entity)}\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)",
                )
                for pattern in explicit_patterns:
                    for match in re.finditer(pattern, segment, re.I):
                        pairs.append((entity, cls._normalize_number(match.group(1))))

            # If a segment contains one entity and no explicit relation,
            # accept only a number that is actually present in that row.
            if len(present) == 1:
                fact = present[0]
                entity = str(fact["entity"])
                already = {e.lower() for e, _ in pairs if e.lower() == entity.lower()}
                if not already:
                    allowed = set(fact["numbers"].values())
                    for number in cls._number_tokens(segment):
                        if number in allowed:
                            pairs.append((entity, number))

        seen: set[tuple[str, str]] = set()
        result: list[tuple[str, str]] = []
        for pair in pairs:
            key = (pair[0].lower(), pair[1])
            if key not in seen:
                seen.add(key)
                result.append(pair)
        return result

    @classmethod
    def _verify_entity_numeric_claims(
        cls,
        question: str,
        claim_text: str,
        rows: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        facts = [
            fact for fact in cls._row_facts(question, rows)
            if fact["entity"] and len(str(fact["entity"])) > 1
        ]
        if not facts:
            return [], []

        pairs = cls._explicit_entity_number_pairs(claim_text, facts)
        supported: list[str] = []
        unsupported: list[str] = []

        for entity, number in pairs:
            fact = next(
                (
                    item for item in facts
                    if str(item["entity"]).lower() == entity.lower()
                ),
                None,
            )
            if not fact:
                continue

            if number in set(fact["numbers"].values()):
                supported.append(f"{entity} → {number}")
            else:
                unsupported.append(
                    f"{entity} → {number} is not supported by that row"
                )

        return list(dict.fromkeys(supported)), list(dict.fromkeys(unsupported))

    @classmethod
    def _verify_ranking(
        cls,
        question: str,
        insight: GeneratedInsight,
        rows: list[dict[str, Any]],
    ) -> tuple[bool, list[str], list[str]]:
        if not re.search(
            r"\b(?:most|highest|largest|maximum|top|rank|ranking|least|lowest|smallest|minimum)\b",
            question or "",
            re.I,
        ):
            return True, [], []

        facts = cls._row_facts(question, rows)
        facts = [fact for fact in facts if fact["entity"] and fact["numbers"]]
        if not facts:
            return False, [], ["Ranking requested but evidence has no entity/metric rows."]

        metric_column = cls._choose_metric_column(question, rows)
        if not metric_column:
            return False, [], ["Ranking requested but no numeric ranking metric could be identified."]

        expected_desc = not bool(
            re.search(r"\b(?:least|lowest|smallest|minimum)\b", question or "", re.I)
        )
        expected = []
        for fact in facts:
            value = fact["numbers"].get(metric_column)
            if value is None:
                continue
            expected.append((fact["entity"], float(value)))
        expected.sort(key=lambda x: x[1], reverse=expected_desc)
        top_n = re.search(r"\btop\s+(\d+)\b", question or "", re.I)
        if top_n:
            expected = expected[: int(top_n.group(1))]

        claim_text = " ".join([insight.summary, *insight.findings])
        pairs = cls._explicit_entity_number_pairs(claim_text, facts)
        actual_map: dict[str, float] = {}

        for entity, number in pairs:
            fact = next(
                (
                    f for f in facts
                    if str(f["entity"]).lower() == entity.lower()
                ),
                None,
            )
            if not fact:
                continue
            if number in set(fact["numbers"].values()):
                actual_map.setdefault(entity.lower(), float(number))

        actual: list[tuple[str, float]] = []
        for entity, _ in expected:
            value = actual_map.get(entity.lower())
            if value is not None:
                actual.append((entity, value))

        errors: list[str] = []
        if len(actual) < len(expected):
            errors.append("The insight does not report every required ranked evidence row.")
            return False, [f"Expected ranking metric: {metric_column}"], errors

        expected_entities = [entity for entity, _ in expected]
        actual_entities = [entity for entity, _ in actual]
        if actual_entities != expected_entities:
            errors.append("The insight ranking order does not match the verified evidence order.")

        for (expected_entity, expected_value), (_, actual_value) in zip(expected, actual):
            if actual_value != expected_value:
                errors.append(
                    f"{expected_entity} has claimed metric {actual_value:g}, "
                    f"but verified evidence reports {expected_value:g}."
                )

        return not errors, [f"Ranking metric verified from evidence column '{metric_column}'."], errors

    @classmethod
    def _causal_claims(cls, text: str) -> bool:
        return any(re.search(pattern, text, re.I) for pattern in cls.CAUSAL_PATTERNS)

    def verify(
        self,
        question: str,
        insight: GeneratedInsight,
        evidence: list[Evidence],
    ) -> InsightVerificationDecision:
        rows = self._rows(evidence)
        claim_text = " ".join([insight.summary, *insight.findings])
        numeric_claim_context = bool(
            self._numeric_columns(rows)
            and re.search(
                r"\b(?:how many|number|count|counts|most|highest|largest|maximum|top|rank|ranking|least|lowest|smallest|minimum|compare|comparison|why|reason|explain)\b",
                question or "",
                re.I,
            )
        )

        if numeric_claim_context:
            supported, unsupported, warnings = self._verify_aggregate_numeric_claims(
                claim_text, rows
            )

            entity_supported, entity_unsupported = self._verify_entity_numeric_claims(
                question, claim_text, rows
            )
            supported.extend(entity_supported)
            unsupported.extend(entity_unsupported)
        else:
            supported, unsupported, warnings = [], [], []

        # Numbers that are part of the user's analytical constraint are not
        # database findings. For example, in "identify the top 5 pathways",
        # the 5 describes the requested result cardinality, not a metric that
        # must exist in the evidence. Likewise, ranking direction words and
        # other query constraints must not be treated as unsupported values.
        #
        # Keep this generic: derive excluded cardinality values solely from the
        # original question rather than from any database/domain knowledge.
        if unsupported:
            top_n_match = re.search(
                r"\btop\s+(\d+)\b",
                question or "",
                re.I,
            )
            query_constraint_numbers: set[str] = set()

            if top_n_match:
                query_constraint_numbers.add(
                    self._normalize_number(top_n_match.group(1))
                )

            if query_constraint_numbers:
                filtered_unsupported = []
                for item in unsupported:
                    match = re.match(
                        r"^(-?\d+(?:\.\d+)?)\s+is not present in verified evidence$",
                        item,
                    )
                    if match and self._normalize_number(match.group(1)) in query_constraint_numbers:
                        continue
                    filtered_unsupported.append(item)
                unsupported = filtered_unsupported

        ranking_valid, ranking_supported, ranking_errors = self._verify_ranking(
            question, insight, rows
        )
        supported.extend(ranking_supported)
        unsupported.extend(ranking_errors)

        if self._causal_claims(claim_text):
            if re.search(r"\b(?:why|reason|explain)\b", question or "", re.I):
                warnings.append(
                    "Causal language appears in a why/explanation answer; the evidence should be interpreted as association unless causal evidence is present."
                )
            else:
                unsupported.append("Unsupported causal claim.")

        for item in evidence:
            sql = (item.sql or "").lower()
            if re.search(r"\blimit\s+\d+\b", sql) and not re.search(
                r"\b(?:count|sum|avg|min|max)\s*\(", sql
            ):
                warnings.append(
                    f"Evidence step {item.step_id} uses LIMIT; completeness is limited to returned rows."
                )

        unsupported = list(dict.fromkeys(unsupported))
        warnings = list(dict.fromkeys(warnings))
        supported = list(dict.fromkeys(supported))

        if unsupported or not ranking_valid:
            return InsightVerificationDecision(
                verified=False,
                confidence=0.20,
                supported_findings=supported,
                unsupported_findings=unsupported or ["Ranking verification failed."],
                warnings=warnings,
            )

        return InsightVerificationDecision(
            verified=True,
            confidence=0.93 if warnings else 0.98,
            supported_findings=supported,
            unsupported_findings=[],
            warnings=warnings,
        )
