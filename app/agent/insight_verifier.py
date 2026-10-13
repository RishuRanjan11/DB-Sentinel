from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.evidence import Evidence
from app.agent.insight_generator import GeneratedInsight, InsightGenerator


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
        r"\bcaus(?:e|es|ed|ing)\b",
        r"\bcaused by\b",
        r"\bdriv(?:e|es|en|ing)\b",
        r"\bdriven by\b",
        r"\bleads to\b",
        r"\bresponsible for\b",
        r"\bdue to\b",
        r"\bbecause of\b",
        r"\b(?:fell|decreased|declined|increased|grew|dropped|rose|"
        r"changed|varied)\b.{0,60}\bbecause\b",
        r"\bresulted from\b",
        r"\bcontribut(?:e|es|ed|ing)\s+to\b",
    )

    def __init__(self, model_router=None):
        self.model_router = model_router

    @staticmethod
    def _normalize(value: Any) -> str:
        return "" if value is None else str(value).strip().lower()

    @staticmethod
    def _normalize_number(value: Any) -> str:
        return InsightGenerator._normalize_number(value)

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
                if InsightGenerator._numeric_value(value) is None:
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
                if InsightGenerator._numeric_value(value) is None:
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
        columns = [
            column
            for column in cls._string_columns(rows)
            if any(
                isinstance(row.get(column), str)
                and InsightGenerator._numeric_value(row[column]) is None
                for row in rows
            )
        ]
        if not columns:
            return None
        q_tokens = cls._tokens(question)
        scored: list[tuple[float, str]] = []
        for column in columns:
            values = [
                str(row[column])
                for row in rows
                if isinstance(row.get(column), str)
                and InsightGenerator._numeric_value(row[column]) is None
            ]
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
        scanned_result_count: int | None = None,
        retained_preview_count: int | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Verify numeric claims for aggregate/single-row evidence."""
        numbers = cls._number_tokens(claim_text)
        if not numbers:
            return [], [], []
        evidence_numbers = cls._evidence_number_set(rows)
        count_claim = bool(
            re.search(
                r"\b(?:complete scan|full scan|scanned|matching results|tied)\b",
                claim_text,
                re.IGNORECASE,
            )
        )
        if scanned_result_count is not None and count_claim:
            evidence_numbers.add(
                cls._normalize_number(scanned_result_count)
            )
        if (
            retained_preview_count is not None
            and re.search(r"\bpreview\b", claim_text, re.IGNORECASE)
        ):
            evidence_numbers.add(
                cls._normalize_number(retained_preview_count)
            )
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
        rows_complete: bool = True,
    ) -> tuple[bool, list[str], list[str]]:
        if not re.search(
            r"\b(?:most|highest|largest|maximum|top|rank|ranking|least|lowest|smallest|minimum)\b",
            question or "",
            re.I,
        ):
            return True, [], []

        comparative = InsightGenerator._comparative_metric_insight(
            question,
            rows,
        )
        if comparative is not None:
            if (
                insight.summary == comparative.summary
                and insight.findings == comparative.findings
            ):
                return (
                    True,
                    [
                        "The relative multi-measure comparison and entity/value "
                        "associations match the verified rows."
                    ],
                    [],
                )
            return (
                False,
                [],
                [
                    "The insight does not match the verified high-volume, "
                    "low-relative-value comparison."
                ],
            )

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
        top_n = re.search(r"\btop\s+(\d+)\b", question or "", re.I)
        implicit_extreme = bool(
            top_n is None
            and re.search(
                r"\b(?:maximum|max|highest|largest|minimum|min|lowest|smallest)\b",
                question or "",
                re.I,
            )
            and not re.search(r"\b(?:rank|ranking|top)\b", question or "", re.I)
        )
        if implicit_extreme and not rows_complete:
            return (
                False,
                [],
                ["Cannot verify an implicit extreme from incomplete result rows."],
            )

        expected: list[tuple[str, Decimal]] = []
        for fact in facts:
            value = fact["numbers"].get(metric_column)
            if value is None:
                continue
            expected.append((fact["entity"], Decimal(value)))
        expected.sort(key=lambda x: x[1], reverse=expected_desc)
        if implicit_extreme and expected:
            extreme_value = expected[0][1]
            expected = [
                item for item in expected if item[1] == extreme_value
            ]
        if top_n:
            expected = expected[: int(top_n.group(1))]

        claim_text = " ".join(
            [
                insight.summary,
                *insight.findings,
                *insight.recommendations,
            ]
        )
        pairs = cls._explicit_entity_number_pairs(claim_text, facts)
        actual_map: dict[str, Decimal] = {}

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
                actual_map.setdefault(entity.lower(), Decimal(number))

        actual: list[tuple[str, Decimal]] = []
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
    def _verify_extreme_tie_summary(
        cls,
        question: str,
        insight: GeneratedInsight,
        evidence: list[Evidence],
    ) -> tuple[bool, list[str], list[str]]:
        tie_evidence = [
            item
            for item in evidence
            if item.all_extreme_ties
        ]
        if len(tie_evidence) != 1:
            return (
                False,
                [],
                ["Extreme-tie answer requires exactly one complete tie scan."],
            )

        item = tie_evidence[0]
        if (
            not item.pagination_complete
            or item.total_rows is None
            or not item.ranking_metric
            or item.total_rows < len(item.rows)
        ):
            return (
                False,
                [],
                ["Extreme-tie evidence is incomplete or has an invalid count."],
            )

        if item.total_rows == 0 and not item.rows:
            claim_text = " ".join(
                [
                    insight.summary,
                    *insight.findings,
                    *insight.recommendations,
                ]
            ).lower()
            extreme_label = (
                "minimum"
                if re.search(
                    r"\b(?:least|lowest|smallest|minimum|min)\b",
                    question or "",
                    re.IGNORECASE,
                )
                else "maximum"
            )
            metric_words = re.sub(
                r"^(?:max|maximum|min|minimum)[ _]+",
                "",
                str(item.ranking_metric).replace("_", " "),
                flags=re.IGNORECASE,
            ).lower()
            if (
                "no entities" in claim_text
                and extreme_label in claim_text
                and "could not be determined" in claim_text
                and metric_words in claim_text
            ):
                return (
                    True,
                    ["The complete tie scan returned zero matching entities."],
                    [],
                )
            return (
                False,
                [],
                ["The summary does not correctly report the empty extreme-tie result."],
            )

        if not item.rows:
            return (
                False,
                [],
                ["Extreme-tie evidence has no rows despite a nonzero count."],
            )

        metric_column = next(
            (
                column
                for column in item.rows[0]
                if str(column).lower()
                == str(item.ranking_metric).lower()
            ),
            None,
        )
        if metric_column is None:
            return (
                False,
                [],
                ["The requested ranking metric is absent from tie evidence."],
            )

        metric_values: list[float] = []
        try:
            metric_values = [
                float(row[metric_column])
                for row in item.rows
            ]
        except (KeyError, TypeError, ValueError):
            return (
                False,
                [],
                ["Tie evidence contains a missing or nonnumeric metric value."],
            )
        if not metric_values or len(set(metric_values)) != 1:
            return (
                False,
                [],
                ["Returned rows do not all share the same extreme metric value."],
            )

        claim_text = " ".join(
            [
                insight.summary,
                *insight.findings,
                *insight.recommendations,
            ]
        )
        low_extreme = bool(
            re.search(
                r"\b(?:least|lowest|smallest|minimum|min)\b",
                question or "",
                re.IGNORECASE,
            )
        )
        extreme_label = "minimum" if low_extreme else "maximum"
        normalized_numbers = {
            cls._normalize_number(value)
            for value in cls._number_tokens(claim_text)
        }
        required_numbers = {
            cls._normalize_number(item.total_rows),
            cls._normalize_number(metric_values[0]),
        }
        if not required_numbers.issubset(normalized_numbers):
            return (
                False,
                [],
                ["The summary omits the exact tie count or extreme metric value."],
            )
        if extreme_label not in claim_text.lower():
            return (
                False,
                [],
                [f"The summary does not identify the {extreme_label} metric."],
            )
        metric_words = re.sub(
            r"^(?:max|maximum|min|minimum)[ _]+",
            "",
            str(item.ranking_metric).replace("_", " "),
            flags=re.IGNORECASE,
        ).lower()
        if metric_words not in claim_text.lower():
            return (
                False,
                [],
                ["The summary does not identify the verified ranking metric."],
            )

        return (
            True,
            [
                f"All {item.total_rows} rows tied at the verified "
                f"{extreme_label} {metric_words} value "
                f"{cls._normalize_number(metric_values[0])}."
            ],
            [],
        )

    @classmethod
    def _causal_claims(cls, text: str) -> bool:
        return any(re.search(pattern, text, re.I) for pattern in cls.CAUSAL_PATTERNS)

    def verify(
        self,
        question: str,
        insight: GeneratedInsight,
        evidence: list[Evidence],
        interpretation_question: str | None = None,
    ) -> InsightVerificationDecision:
        rows = self._rows(evidence)
        claim_text = " ".join([insight.summary, *insight.findings])
        analysis_question = " ".join(
            part
            for part in (question, interpretation_question or "")
            if part
        )
        entity_trend = InsightGenerator._deterministic_entity_trends(
            analysis_question,
            evidence,
        )
        if entity_trend is not None:
            if (
                insight.summary == entity_trend.summary
                and insight.findings == entity_trend.findings
            ):
                return InsightVerificationDecision(
                    verified=True,
                    confidence=0.98,
                    supported_findings=[
                        "Per-entity trend directions and annual values match "
                        "the complete verified evidence."
                    ],
                )
            return InsightVerificationDecision(
                verified=False,
                confidence=0.0,
                unsupported_findings=[
                    "The reported per-entity trend does not match the "
                    "verified annual values."
                ],
            )
        has_scanned_collection_count = any(
            item.pagination_complete
            and item.total_rows is not None
            and item.total_rows > len(item.rows)
            for item in evidence
        )
        numeric_claim_context = bool(
            (
                self._numeric_columns(rows)
                or has_scanned_collection_count
            )
            and re.search(
                r"\b(?:how many|number|count|counts|most|highest|largest|maximum|top|rank|ranking|least|lowest|smallest|minimum|compare|comparison|change|changed|difference|delta|trend|growth|decline|increase|decrease|why|reason|explain|name|list|show|find)\b",
                analysis_question,
                re.I,
            )
        )

        if numeric_claim_context:
            supported, unsupported, warnings = self._verify_aggregate_numeric_claims(
                claim_text,
                rows,
                scanned_result_count=(
                    sum(
                        item.total_rows
                        for item in evidence
                        if item.pagination_complete
                        and item.total_rows is not None
                    )
                    or None
                ),
                retained_preview_count=sum(
                    len(item.rows)
                    for item in evidence
                    if item.pagination_complete
                    and item.total_rows is not None
                ),
            )
            series = InsightGenerator._period_series_facts(
                analysis_question,
                evidence,
            )
            if series is not None:
                points, _ = series
                earlier_year, earlier_value = points[0]
                later_year, later_value = points[-1]
                delta = later_value - earlier_value
                change_value = InsightGenerator._normalize_number(
                    abs(delta)
                )
                derived_values = {change_value}
                percentage_requested = bool(
                    re.search(
                        r"\b(?:percentage|percent)\b|%",
                        analysis_question,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\b(?:growth|grew|grown)\b",
                        analysis_question,
                        re.IGNORECASE,
                    )
                )
                expected_percentage: str | None = None
                if earlier_value != 0:
                    expected_percentage = InsightGenerator._normalize_number(
                        (abs(delta) / abs(earlier_value) * Decimal("100")).quantize(
                            Decimal("0.01"),
                            rounding=ROUND_HALF_UP,
                        )
                    )
                    if percentage_requested:
                        derived_values.add(expected_percentage)
                unsupported = [
                    item
                    for item in unsupported
                    if not any(
                        item == f"{value} is not present in verified evidence"
                        for value in derived_values
                    )
                ]
                claim_numbers = set(self._number_tokens(claim_text))
                if change_value in claim_numbers:
                    supported.append(
                        f"{change_value} is the derived difference between "
                        "the verified requested-period values"
                    )
                if percentage_requested:
                    if expected_percentage is None:
                        if not re.search(
                            r"\b(?:cannot|unable to)\s+be\s+calculated\b",
                            claim_text,
                            re.IGNORECASE,
                        ):
                            unsupported.append(
                                "Percentage change cannot be calculated "
                                "because the baseline is zero."
                            )
                    elif (
                        expected_percentage not in claim_numbers
                        or f"{expected_percentage}%" not in claim_text
                    ):
                        unsupported.append(
                            "The percentage change does not match the "
                            "calculation from the verified period values."
                        )
                    else:
                        supported.append(
                            f"{expected_percentage}% is the verified "
                            "percentage change derived from both yearly values"
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
            query_constraint_numbers.update(
                self._normalize_number(year)
                for year in re.findall(
                    r"\b(?:19|20)\d{2}\b",
                    question or "",
                )
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

        if any(item.all_extreme_ties for item in evidence):
            ranking_valid, ranking_supported, ranking_errors = (
                self._verify_extreme_tie_summary(
                    question,
                    insight,
                    evidence,
                )
            )
        else:
            ranking_valid, ranking_supported, ranking_errors = (
                self._verify_ranking(
                    question,
                    insight,
                    rows,
                    rows_complete=all(
                        item.pagination_complete
                        for item in evidence
                    ),
                )
            )
        supported.extend(ranking_supported)
        unsupported.extend(ranking_errors)

        if self._causal_claims(claim_text):
            unsupported.append(
                "Causal language is unsupported; report the observed "
                "association or difference instead."
            )

        comparison = InsightGenerator._period_comparison_facts(
            question,
            evidence,
        )
        if comparison is not None:
            earlier_year, earlier_value, later_year, later_value, metric = comparison
            metric_label = re.sub(
                r"([a-z])([A-Z])",
                r"\1 \2",
                metric.replace("_", " "),
            ).lower()
            expected_direction = (
                "increased"
                if later_value > earlier_value
                else "decreased"
                if later_value < earlier_value
                else "did not change"
            )
            direction_pattern = (
                r"\b(?:increas(?:e|ed|ing)|higher|rose|up)\b"
                if expected_direction == "increased"
                else r"\b(?:decreas(?:e|ed|ing)|lower|fell|down)\b"
                if expected_direction == "decreased"
                else r"\b(?:did not change|unchanged|same)\b"
            )
            required_values = (
                str(earlier_year),
                InsightGenerator._normalize_number(earlier_value),
                str(later_year),
                InsightGenerator._normalize_number(later_value),
            )
            if not re.search(direction_pattern, claim_text, re.I):
                unsupported.append(
                    f"The comparison direction is not supported; evidence shows "
                    f"{metric_label} {expected_direction}."
                )
            if any(value not in claim_text for value in required_values):
                unsupported.append(
                    "The comparison must state both requested years and their verified values."
                )

        for item in evidence:
            sql = (item.sql or "").lower()
            if (
                not item.pagination_complete
                and re.search(r"\blimit\s+\d+\b", sql)
                and not re.search(
                r"\b(?:count|sum|avg|min|max)\s*\(", sql
                )
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
