import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.evidence import Evidence
from app.agent.question_normalizer import QuestionNormalizer
from app.llm.model_router import ModelRouter, create_default_model_router


class GeneratedInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    findings: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class InsightGenerator:
    """Turn verified evidence into a grounded user-facing insight.

    This component is intentionally database-agnostic. It never knows table,
    schema, column, entity, or domain names. Deterministic paths operate only
    on the structure and values already present in verified evidence.
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

    def __init__(self, model_router: ModelRouter | None = None):
        self.model_router = model_router or create_default_model_router()

    @staticmethod
    def _normalize_number(value: Any) -> str:
        try:
            number = InsightGenerator._numeric_value(value)
            if number is None:
                return str(value).strip()
            normalized = format(number, "f")
            return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized
        except (TypeError, ValueError, InvalidOperation):
            return str(value).strip()

    @staticmethod
    def _numeric_value(value: Any) -> Decimal | None:
        if isinstance(value, dict):
            for key in (
                "$numberDecimal",
                "$numberDouble",
                "$numberInt",
                "$numberLong",
            ):
                if key in value:
                    value = value[key]
                    break
            else:
                return None
        if isinstance(value, bool) or value is None:
            return None
        try:
            number = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None
        return number if number.is_finite() else None

    @classmethod
    def _row_year(cls, row: dict[str, Any]) -> int | None:
        identifier = row.get("_id")
        if isinstance(identifier, dict):
            identifier = identifier.get("year")
        if identifier is None:
            identifier = row.get("year")
        if isinstance(identifier, bool):
            return None
        try:
            year = int(identifier)
        except (TypeError, ValueError):
            return None
        return year if 1000 <= year <= 9999 else None

    @classmethod
    def _period_comparison_facts(
        cls,
        question: str,
        evidence: list[Evidence],
    ) -> tuple[int, Decimal, int, Decimal, str] | None:
        years = sorted({
            int(year)
            for year in re.findall(r"\b(?:19|20)\d{2}\b", question or "")
        })
        if len(years) != 2:
            return None
        series = cls._period_series_facts(question, evidence, years)
        if series is None:
            return None
        points, metric = series
        return points[0][0], points[0][1], points[-1][0], points[-1][1], metric

    @classmethod
    def _period_series_facts(
        cls,
        question: str,
        evidence: list[Evidence],
        years: list[int] | None = None,
    ) -> tuple[list[tuple[int, Decimal]], str] | None:
        years = years or sorted({
            int(year)
            for year in re.findall(r"\b(?:19|20)\d{2}\b", question or "")
        })
        if len(years) == 2 and re.search(
            r"\b(?:through|trend|year by year|each year|over the years?)\b",
            question or "",
            re.IGNORECASE,
        ):
            years = list(range(years[0], years[1] + 1))
        if len(years) < 2:
            years = next(
                (
                    sorted(set(item.requested_periods))
                    for item in evidence
                    if len(set(item.requested_periods)) >= 2
                ),
                years,
            )
        if len(years) < 2:
            evidence_years = sorted({
                int(year)
                for item in evidence
                for text in (item.question, item.objective)
                for year in re.findall(r"\b(?:19|20)\d{2}\b", text or "")
            })
            if len(evidence_years) >= 2:
                years = evidence_years
        if len(years) < 2:
            duration = QuestionNormalizer.relative_comparison_year_count(
                question
            )
            if duration:
                available_years = sorted({
                    year
                    for item in evidence
                    for row in (item.rows or [])
                    if isinstance(row, dict)
                    and (year := cls._row_year(row)) is not None
                })
                if len(available_years) >= duration:
                    years = available_years[-duration:]
        if len(years) < 2 or not re.search(
            r"\b(?:compare|comparison|versus|vs\.?|against|percentage|"
            r"percent|increase|"
            r"increased|increasing|decrease|decreased|decreasing|higher|"
            r"lower|change|changed|changing|difference|delta|trend|"
            r"growth|grow|decline|declined)\b",
            question or "",
            re.IGNORECASE,
        ):
            return None

        evidence_context = " ".join(
            f"{item.question} {item.objective}"
            for item in evidence
        )
        question_tokens = cls._tokens(
            f"{question} {evidence_context}"
        )
        candidates: list[
            tuple[int, list[tuple[int, Decimal]], str]
        ] = []
        for item in evidence:
            by_year: dict[int, dict[str, Decimal]] = {}
            duplicate_year = False
            for row in item.rows or []:
                if not isinstance(row, dict):
                    continue
                year = cls._row_year(row)
                if year not in years:
                    continue
                identifier = row.get("_id")
                if isinstance(identifier, dict) and set(identifier) != {"year"}:
                    continue
                metrics = {
                    str(column): number
                    for column, value in row.items()
                    if column not in {"_id", "year"}
                    and (number := cls._numeric_value(value)) is not None
                }
                if metrics:
                    if year in by_year:
                        duplicate_year = True
                        break
                    by_year[year] = metrics
            if duplicate_year or not all(year in by_year for year in years):
                continue

            shared_columns = set.intersection(
                *(set(by_year[year]) for year in years)
            )
            for column in shared_columns:
                normalized_column = re.sub(r"[^a-z0-9]", "", column.lower())
                tokens = cls._tokens(column)
                lexical_score = sum(
                    10
                    for token in question_tokens
                    if token in normalized_column and len(token) > 2
                )
                aggregate_score = sum(
                    2
                    for token in tokens
                    if token in {"total", "sum", "amount", "volume", "revenue"}
                )
                if lexical_score or aggregate_score:
                    candidates.append((
                        lexical_score + aggregate_score,
                        [
                            (year, by_year[year][column])
                            for year in years
                        ],
                        column,
                    ))

        if not candidates:
            return None
        _, points, metric = max(
            candidates,
            key=lambda candidate: (
                candidate[0],
                "total" in candidate[2].lower(),
                candidate[2].lower(),
            ),
        )
        return points, metric

    @classmethod
    def _deterministic_entity_trends(
        cls,
        question: str,
        evidence: list[Evidence],
    ) -> GeneratedInsight | None:
        duration = QuestionNormalizer.relative_comparison_year_count(question)
        if (
            duration is None
            or not re.search(
                r"\b(?:growth|grow|increas|decreas|declin|trend|rise|fall)\w*\b",
                question or "",
                re.IGNORECASE,
            )
            or not evidence
            or any(
                not item.pagination_complete
                or (
                    item.total_rows is not None
                    and item.total_rows > len(item.rows or [])
                )
                for item in evidence
            )
        ):
            return None

        rows = cls._evidence_rows(evidence)
        years = sorted({
            year
            for row in rows
            if (year := cls._row_year(row)) is not None
        })
        if len(years) < duration:
            return None
        years = years[-duration:]

        def normalized_tokens(text: str) -> set[str]:
            text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
            return {
                token[:-1] if token.endswith("s") and len(token) > 3 else token
                for token in cls._tokens(text.replace("_", " "))
            }

        question_tokens = normalized_tokens(
            " ".join(
                [question]
                + [f"{item.question} {item.objective}" for item in evidence]
            )
        )
        dimensions: set[str] = set()
        for row in rows:
            identifier = row.get("_id")
            if isinstance(identifier, dict):
                dimensions.update(
                    str(key)
                    for key, value in identifier.items()
                    if cls._row_year({"_id": {str(key): value}}) is None
                )
        if not dimensions:
            return None

        scored_dimensions = sorted(
            (
                len(question_tokens & normalized_tokens(dimension)),
                dimension,
            )
            for dimension in dimensions
        )
        best_dimension_score = scored_dimensions[-1][0]
        best_dimensions = [
            dimension
            for score, dimension in scored_dimensions
            if score == best_dimension_score
        ]
        if best_dimension_score == 0 or len(best_dimensions) != 1:
            return None
        entity_key = best_dimensions[0]

        numeric_columns = cls._numeric_columns(rows)
        metric_question_tokens = question_tokens - {
            "year",
            "date",
            "time",
        }
        metric_scores = sorted(
            (
                len(metric_question_tokens & normalized_tokens(column)),
                column,
            )
            for column in numeric_columns
        )
        best_metric_score = metric_scores[-1][0] if metric_scores else 0
        best_metrics = [
            column
            for score, column in metric_scores
            if score == best_metric_score
        ]
        if best_metric_score == 0 or len(best_metrics) != 1:
            return None
        metric_column = best_metrics[0]

        by_entity: dict[str, dict[int, Decimal]] = {}
        for row in rows:
            identifier = row.get("_id")
            if not isinstance(identifier, dict):
                continue
            year = cls._row_year(row)
            entity = identifier.get(entity_key)
            value = cls._numeric_value(row.get(metric_column))
            if year not in years or entity is None or value is None:
                continue
            values = by_entity.setdefault(str(entity), {})
            if year in values:
                return None
            values[year] = value

        metric_label = re.sub(
            r"([a-z])([A-Z])",
            r"\1 \2",
            metric_column.replace("_", " "),
        ).lower()
        findings: list[str] = []
        increasing_entities: list[str] = []
        declining_entities: list[str] = []
        for entity, values in sorted(
            by_entity.items(),
            key=lambda item: item[0].lower(),
        ):
            if not all(year in values for year in years):
                continue
            ordered_values = [values[year] for year in years]
            if all(
                later > earlier
                for earlier, later in zip(ordered_values, ordered_values[1:])
            ):
                direction = "increased consistently"
                increasing_entities.append(entity)
            elif all(
                later < earlier
                for earlier, later in zip(ordered_values, ordered_values[1:])
            ):
                direction = "declined consistently"
                declining_entities.append(entity)
            else:
                continue
            annual_values = "; ".join(
                f"{year}: {cls._normalize_number(values[year])}"
                for year in years
            )
            findings.append(
                f"{entity}: {metric_label} {direction} ({annual_values})."
            )

        first_year, last_year = years[0], years[-1]
        if increasing_entities and declining_entities:
            summary = (
                f"These entities show consistent year-over-year growth or "
                f"decline in {metric_label} from {first_year} through "
                f"{last_year}:"
            )
        elif increasing_entities:
            summary = (
                f"Some entities show consistent year-over-year growth in "
                f"{metric_label} from {first_year} through {last_year}; "
                "none shows a consistent decline."
            )
        elif declining_entities:
            summary = (
                f"Some entities show a consistent year-over-year decline in "
                f"{metric_label} from {first_year} through {last_year}; "
                "none shows consistent growth."
            )
        else:
            summary = (
                f"No entity shows strictly consistent year-over-year growth "
                f"or decline in {metric_label} from {first_year} through "
                f"{last_year}."
            )
        return GeneratedInsight(
            summary=summary,
            findings=findings,
            supporting_evidence=[
                "Each direction was calculated from the complete verified "
                "set of annual values for that entity."
            ],
            uncertainties=[],
        )

    @classmethod
    def _number_tokens(cls, text: str) -> list[str]:
        return [cls._normalize_number(v) for v in cls.NUMBER_PATTERN.findall(text)]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_]+", text.lower())
            if len(token) > 1
        }

    @classmethod
    def _evidence_rows(cls, evidence: list[Evidence]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in evidence:
            for row in item.rows or []:
                if isinstance(row, dict):
                    rows.append(dict(row))
        return rows

    @classmethod
    def _numeric_columns(cls, rows: list[dict[str, Any]]) -> list[str]:
        columns: set[str] = set()
        for row in rows:
            for key, value in row.items():
                if cls._numeric_value(value) is None:
                    continue
                columns.add(str(key))
        return sorted(columns)

    @classmethod
    def _candidate_entity_columns(cls, rows: list[dict[str, Any]], question: str) -> list[str]:
        if not rows:
            return []

        columns = list({str(k) for row in rows for k in row.keys()})
        question_tokens = cls._tokens(question)
        scored: list[tuple[float, str]] = []

        for column in columns:
            values = [row.get(column) for row in rows if row.get(column) is not None]
            if not values:
                continue

            numeric = sum(
                cls._numeric_value(value) is not None
                for value in values
            )
            if numeric == len(values):
                continue

            unique_ratio = len({str(v) for v in values}) / max(1, len(values))
            column_tokens = cls._tokens(column)
            lexical = len(question_tokens & column_tokens)
            score = unique_ratio * 5 + lexical * 3
            scored.append((score, column))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [column for _, column in scored]

    @classmethod
    def _choose_entity_column(cls, rows: list[dict[str, Any]], question: str) -> str | None:
        candidates = cls._candidate_entity_columns(rows, question)
        return candidates[0] if candidates else None

    @classmethod
    def _choose_metric_column(cls, rows: list[dict[str, Any]], question: str) -> str | None:
        numeric_columns = cls._numeric_columns(rows)
        if not numeric_columns:
            return None

        q_tokens = cls._tokens(question)
        asks_count = bool(re.search(r"\b(?:how many|number|count|counts)\b", question, re.I))
        scored: list[tuple[float, str]] = []

        for column in numeric_columns:
            tokens = cls._tokens(column)
            score = len(tokens & q_tokens) * 10
            if asks_count:
                score += sum(
                    2 for token in tokens if token in {"count", "number", "total", "num"}
                )
            score += min(5, len(tokens))
            scored.append((score, column))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][1]

    @classmethod
    def _is_aggregate_question(cls, question: str) -> bool:
        text = " ".join((question or "").lower().split())
        if not text:
            return False
        return bool(re.search(r"\b(?:how many|number of|count)\b", text)) and not bool(
            re.search(
                r"\b(?:compare|comparison|why|trend|correlation|relationship|top|most|highest|lowest|least|ranking|rank)\b",
                text,
            )
        )

    @classmethod
    def _single_aggregate_value(cls, evidence: list[Evidence]) -> tuple[Any, str] | None:
        candidates: list[tuple[Any, str]] = []
        for item in evidence:
            rows = item.rows or []
            if len(rows) != 1 or not isinstance(rows[0], dict):
                continue
            row = rows[0]
            numeric = []
            for column, value in row.items():
                if cls._numeric_value(value) is None:
                    continue
                numeric.append((value, str(column)))
            if len(numeric) == 1:
                candidates.extend(numeric)
            elif numeric:
                # Prefer a column whose name semantically indicates an aggregate.
                numeric.sort(
                    key=lambda pair: (
                        0 if re.search(r"\b(?:count|number|total|num|value|aggregate)\b", pair[1], re.I) else 1,
                        pair[1].lower(),
                    )
                )
                candidates.append(numeric[0])

        if not candidates:
            return None
        # One-step aggregate evidence is the normal case. If multiple verified
        # steps expose values, use the first only when all candidates agree.
        normalized = {cls._normalize_number(value) for value, _ in candidates}
        if len(normalized) == 1:
            return candidates[0]
        return candidates[0]

    @classmethod
    def _ranking_requested(cls, question: str) -> bool:
        return bool(
            re.search(
                r"\b(?:most|highest|largest|maximum|top|rank|ranking|least|lowest|smallest|minimum)\b",
                question or "",
                re.I,
            )
        )

    @classmethod
    def _requested_top_n(cls, question: str) -> int | None:
        match = re.search(r"\btop\s+(\d+)\b", question or "", re.I)
        return int(match.group(1)) if match else None

    @classmethod
    def _ranking_rows(
        cls,
        question: str,
        evidence: list[Evidence],
    ) -> tuple[list[tuple[str, Any]], str] | None:
        rows = cls._evidence_rows(evidence)
        if not rows:
            return None

        entity_column = cls._choose_entity_column(rows, question)
        metric_column = cls._choose_metric_column(rows, question)
        if not entity_column or not metric_column:
            return None

        values: list[tuple[str, Any]] = []
        for row in rows:
            entity = row.get(entity_column)
            value = row.get(metric_column)
            if entity is None or value is None:
                continue
            numeric_value = cls._numeric_value(value)
            if numeric_value is None:
                continue
            values.append((str(entity), value))

        if not values:
            return None

        descending = not bool(
            re.search(r"\b(?:least|lowest|smallest|minimum)\b", question or "", re.I)
        )
        values.sort(
            key=lambda item: cls._numeric_value(item[1]),
            reverse=descending,
        )
        top_n = cls._requested_top_n(question)
        if top_n is not None:
            values = values[:top_n]
        return values, metric_column

    @staticmethod
    def _high_low_contrast_clauses(question: str) -> tuple[str, str] | None:
        connector = re.search(
            r"\b(?:but|while|whereas|yet|although|despite)\b",
            question or "",
            re.IGNORECASE,
        )
        if connector is None:
            return None
        clauses = (
            question[:connector.start()],
            question[connector.end():],
        )
        high_pattern = (
            r"\b(?:most|highest|largest|maximum|max|more|greater|high|higher)\b"
        )
        low_pattern = (
            r"\b(?:low|lower|lowest|least|smallest|minimum|min|less|fewer)\b"
        )
        high_clause = next(
            (clause for clause in clauses if re.search(high_pattern, clause, re.I)),
            None,
        )
        low_clause = next(
            (clause for clause in clauses if re.search(low_pattern, clause, re.I)),
            None,
        )
        return (
            (high_clause, low_clause)
            if high_clause and low_clause and high_clause != low_clause
            else None
        )

    @classmethod
    def _comparative_metric_insight(
        cls,
        question: str,
        rows: list[dict[str, Any]],
    ) -> GeneratedInsight | None:
        """Summarize a high/low metric contrast using a disclosed median baseline."""
        clauses = cls._high_low_contrast_clauses(question)
        if clauses is None or not rows:
            return None
        high_clause, low_clause = clauses

        numeric_columns = cls._numeric_columns(rows)
        ignored_tokens = {
            "but", "while", "whereas", "yet", "although", "despite",
            "most", "highest", "largest", "maximum", "max", "more",
            "greater", "high", "higher", "low", "lower", "lowest",
            "least", "smallest", "minimum", "min", "less", "fewer",
            "which", "what", "who", "the", "a", "an", "of", "for",
            "in", "on", "by", "to", "from", "with", "and", "or",
            "have", "has", "had", "sell", "sells", "sold", "generate",
            "generates", "generated", "relatively", "comparatively",
        }

        def normalized_tokens(text: str) -> set[str]:
            result = set()
            readable_text = re.sub(
                r"([a-z])([A-Z])",
                r"\1 \2",
                text.replace("_", " "),
            )
            for token in cls._tokens(readable_text):
                if token in ignored_tokens:
                    continue
                result.add(token[:-1] if token.endswith("s") and len(token) > 3 else token)
            return result

        def choose_column(clause: str) -> str | None:
            clause_tokens = normalized_tokens(clause)
            candidates = sorted(
                (
                    len(clause_tokens & normalized_tokens(column)),
                    column,
                )
                for column in numeric_columns
            )
            if not candidates:
                return None
            best_score = candidates[-1][0]
            best = [column for score, column in candidates if score == best_score]
            if best_score == 0 or len(best) != 1:
                return None
            return best[0]

        primary_column = choose_column(high_clause)
        secondary_column = choose_column(low_clause)
        entity_column = cls._choose_entity_column(rows, question)
        if (
            primary_column is None
            or secondary_column is None
            or primary_column == secondary_column
            or entity_column is None
        ):
            return None

        comparable: list[tuple[str, Decimal, Decimal, Decimal]] = []
        for row in rows:
            entity = row.get(entity_column)
            primary = cls._numeric_value(row.get(primary_column))
            secondary = cls._numeric_value(row.get(secondary_column))
            if (
                entity is None
                or primary is None
                or primary <= 0
                or secondary is None
            ):
                continue
            comparable.append((str(entity), primary, secondary, secondary / primary))
        if not comparable:
            return None

        def median(values: list[Decimal]) -> Decimal:
            ordered = sorted(values)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[middle]
            return (ordered[middle - 1] + ordered[middle]) / Decimal(2)

        high_baseline = median([item[1] for item in comparable])
        relative_low_baseline = median([item[3] for item in comparable])
        selected = [
            item for item in comparable
            if item[1] > high_baseline and item[3] < relative_low_baseline
        ]
        selected.sort(key=lambda item: (-item[1], item[0].lower()))

        def display_column(column: str) -> str:
            return re.sub(r"([a-z])([A-Z])", r"\1 \2", column).replace(
                "_", " "
            ).lower()

        primary_label = display_column(primary_column)
        secondary_label = display_column(secondary_column)
        summary = (
            "Among the returned entities, "
            f"{'these entities have' if selected else 'no entity has'} "
            f"above-median {primary_label} and below-median "
            f"{secondary_label} per {primary_label}. The median of the "
            "returned entities is the comparison baseline."
        )
        findings = [
            f"{entity}: {cls._normalize_number(primary)} {primary_label}; "
            f"{cls._normalize_number(secondary)} {secondary_label}."
            for entity, primary, secondary, _ in selected
        ]
        return GeneratedInsight(
            summary=summary,
            findings=findings,
            supporting_evidence=[
                "The comparison uses the median across returned entities; "
                "each listed measure is taken from that entity's verified row."
            ],
            uncertainties=[],
        )

    @classmethod
    def _deterministic_aggregate(
        cls,
        question: str,
        evidence: list[Evidence],
    ) -> GeneratedInsight | None:
        if not cls._is_aggregate_question(question):
            return None
        result = cls._single_aggregate_value(evidence)
        if result is None:
            return None
        value, column = result
        value_text = cls._normalize_number(value)
        return GeneratedInsight(
            summary=f"The verified result is {value_text}.",
            findings=[f"The verified aggregate value for the requested question is {value_text}."],
            supporting_evidence=[
                f"Verified step {item.step_id}: {item.objective}"
                for item in evidence
                if item.rows
            ],
            uncertainties=[],
        )

    @classmethod
    def _deterministic_extreme_ties(
        cls,
        question: str,
        evidence: list[Evidence],
    ) -> GeneratedInsight | None:
        tie_evidence = [
            item
            for item in evidence
            if item.all_extreme_ties
            and item.pagination_complete
            and item.total_rows is not None
            and item.ranking_metric
        ]
        if len(tie_evidence) != 1:
            return None

        item = tie_evidence[0]
        rows = [
            row for row in item.rows
            if isinstance(row, dict)
        ]
        if not rows and item.total_rows == 0:
            metric_text = re.sub(
                r"^(?:max|maximum|min|minimum)[ _]+",
                "",
                str(item.ranking_metric).replace("_", " "),
                flags=re.IGNORECASE,
            )
            low_extreme = bool(
                re.search(
                    r"\b(?:least|lowest|smallest|minimum|min)\b",
                    question or "",
                    re.IGNORECASE,
                )
            )
            extreme_label = "minimum" if low_extreme else "maximum"
            return GeneratedInsight(
                summary=(
                    f"No entities with a measurable {metric_text} were "
                    f"returned, so the requested {extreme_label} could "
                    "not be determined."
                ),
                findings=[],
                supporting_evidence=[
                    f"Verified step {item.step_id}: the complete tie scan "
                    "returned zero rows."
                ],
                uncertainties=[],
            )
        metric_column = next(
            (
                column
                for column in rows[0]
                if str(column).lower()
                == str(item.ranking_metric).lower()
            ),
            None,
        ) if rows else None
        if metric_column is None or not rows:
            return None

        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row[metric_column]))
            except (KeyError, TypeError, ValueError):
                return None
        if not values or len(set(values)) != 1:
            return None

        metric_value = values[0]
        value_text = cls._normalize_number(metric_value)
        metric_text = re.sub(
            r"^(?:max|maximum|min|minimum)[ _]+",
            "",
            str(item.ranking_metric).replace("_", " "),
            flags=re.IGNORECASE,
        )
        low_extreme = bool(
            re.search(
                r"\b(?:least|lowest|smallest|minimum|min)\b",
                question or "",
                re.IGNORECASE,
            )
        )
        extreme_label = "minimum" if low_extreme else "maximum"
        total = int(item.total_rows)
        retained = len(rows)
        preview_note = (
            f"Showing {retained} of {total} tied entities; "
            f"the preview is capped at {retained} rows."
            if total > retained
            else f"All {retained} tied entities are shown."
        )
        return GeneratedInsight(
            summary=(
                f"The {extreme_label} {metric_text} is {value_text}. "
                f"{total} entities are tied at this {extreme_label}. "
                f"{preview_note}"
            ),
            findings=[],
            supporting_evidence=[
                f"Verified step {item.step_id}: the database query "
                f"returned the exact count of entities tied at the "
                f"{extreme_label}."
            ],
            uncertainties=[],
        )

    @classmethod
    def _deterministic_ranking(
        cls,
        question: str,
        evidence: list[Evidence],
    ) -> GeneratedInsight | None:
        if not cls._ranking_requested(question):
            return None
        if cls._high_low_contrast_clauses(question) is not None:
            return None
        selected = cls._ranking_rows(question, evidence)
        if selected is None:
            return None
        values, metric_column = selected
        metric_label = re.sub(
            r"([a-z])([A-Z])",
            r"\1 \2",
            metric_column,
        ).replace("_", " ").lower()
        top_n = cls._requested_top_n(question)
        implicit_extreme = bool(
            top_n is None
            and re.search(
                r"\b(?:maximum|max|highest|largest|minimum|min|lowest|"
                r"smallest)\b",
                question or "",
                re.IGNORECASE,
            )
            and not re.search(
                r"\b(?:rank|ranking|top)\b",
                question or "",
                re.IGNORECASE,
            )
        )
        if implicit_extreme and values:
            numeric_values = [
                (entity, cls._numeric_value(value))
                for entity, value in values
            ]
            target_value = max(
                value for _, value in numeric_values if value is not None
            )
            if re.search(
                r"\b(?:minimum|min|lowest|smallest)\b",
                question or "",
                re.IGNORECASE,
            ):
                target_value = min(
                    value for _, value in numeric_values if value is not None
                )
            values = [
                (entity, value)
                for entity, value in values
                if cls._numeric_value(value) == target_value
            ]
        scope = f"top {top_n} " if top_n is not None else ""
        if implicit_extreme and values:
            extreme = (
                "minimum"
                if re.search(
                    r"\b(?:minimum|min|lowest|smallest)\b",
                    question or "",
                    re.IGNORECASE,
                )
                else "maximum"
            )
            summary = (
                f"The {extreme} {metric_label} is "
                f"{cls._normalize_number(values[0][1])}."
            )
        else:
            summary = (
                f"The verified evidence ranks the {scope}requested "
                f"entities by {metric_label}."
            )
        findings = [
            f"{entity}: {cls._normalize_number(value)} {metric_label}."
            for entity, value in values
        ]
        return GeneratedInsight(
            summary=summary,
            findings=findings,
            supporting_evidence=[
                f"Verified step {item.step_id}: ranking derived from returned database rows."
                for item in evidence
                if item.rows
            ],
            uncertainties=[],
        )

    @staticmethod
    def _large_complete_collection(
        question: str,
        evidence: list[Evidence],
    ) -> GeneratedInsight | None:
        asks_for_collection = bool(
            re.search(
                r"\b(?:which|list|show|find|name|return)\b",
                question or "",
                re.I,
            )
        )
        asks_for_metric_or_ranking = bool(
            re.search(
                r"\b(?:how many|number of|count|compare|comparison|top|most|highest|largest|maximum|least|lowest|smallest|minimum|rank|ranking)\b",
                question or "",
                re.I,
            )
        )
        complete = bool(evidence) and all(
            item.pagination_complete for item in evidence
        )
        preview_rows = sum(len(item.rows) for item in evidence)
        scanned_rows = sum(
            item.total_rows
            if item.total_rows is not None
            else len(item.rows)
            for item in evidence
        )
        if (
            not asks_for_collection
            or asks_for_metric_or_ranking
            or not complete
            or scanned_rows <= preview_rows
            or len(evidence) != 1
        ):
            return None

        return GeneratedInsight(
            summary=(
                f"The complete scan found {scanned_rows} matching results. "
                "A bounded preview is retained as evidence; "
                "the remaining results were scanned but not kept in memory."
            ),
            findings=[],
            supporting_evidence=[
                f"Verified step {item.step_id}: {item.objective}"
                for item in evidence
            ],
            uncertainties=[],
        )

    def _build_evidence_text(self, evidence: list[Evidence]) -> list[dict[str, Any]]:
        return [
            {
                "step_id": item.step_id,
                "question": item.question,
                "objective": item.objective,
                "rows": item.rows[:100],
                "logical_verification": item.logical_reason,
                "confidence": item.confidence,
                "pagination_complete": item.pagination_complete,
                "total_rows": item.total_rows,
                "all_extreme_ties": item.all_extreme_ties,
                "ranking_metric": item.ranking_metric,
            }
            for item in evidence
        ]

    def _system_instruction(self) -> str:
        return """
You are DB-Sentinel's evidence-grounded insight generator.

Answer the ORIGINAL user question using ONLY the verified evidence.
Never invent database facts, entities, metrics, numbers, relationships,
or explanations. Do not assume any database schema.

Every numeric statement must be supported by a value in the evidence.
Every entity/value pairing must correspond to the same verified row.
Do not calculate new statistics unless the calculated value is explicitly
present in the evidence.

For ranking questions, preserve the verified ranking order and requested
Top-N scope.

When the question contrasts multiple measures, report the relevant measures
together for each entity and explain the comparison basis. Do not turn a
multi-measure contrast into a ranking by only one measure, and do not imply
that a total is a rate or per-unit value unless the evidence supports that
calculation.

For why/explanation, driver, or contributing-factor questions, synthesize
all relevant verified steps: state the measured outcome, compare it with
verified segment or period evidence when available, and identify only the
observed associations that the rows support. Treat optional breakdowns as
helpful evidence, not as proof of causation. If no relevant breakdown was
verified, say that the evidence does not identify a reason. Never present
correlation, a segment difference, or a before/after change as a cause.

For comparisons and trends, state the direction and the values for the
requested periods when those values are verified. Keep conclusions and
limitations specific to the returned evidence; do not repeat generic
limitations when a supported conclusion is available.

Respect LIMITs and do not claim completeness from a limited non-aggregate
result. Evidence explicitly marked pagination_complete is complete even
though the underlying page queries use LIMIT.

Return only the requested structured insight.

Include recommendations only when the verified evidence justifies a
specific, proportionate next action. Recommendations must be clearly
separated from findings, must not claim unsupported causes, and must not
introduce facts or numbers absent from the evidence. If the evidence does
not justify an action, return an empty recommendations list.
"""

    def _generate_once(
        self,
        question: str,
        evidence_text: list[dict[str, Any]],
        feedback: str | None = None,
    ) -> GeneratedInsight:
        prompt = f"""
Original question:
{question}

Verified evidence:
{evidence_text}
"""
        if feedback:
            prompt += f"\nPrevious validation problems:\n{feedback}\nCorrect the answer without inventing facts."
        return self.model_router.generate(
            system_instruction=self._system_instruction(),
            user_prompt=prompt,
            response_model=GeneratedInsight,
            complexity="fast",
            classification_text=question,
        )

    def _validate_grounding(
        self,
        insight: GeneratedInsight,
        evidence: list[Evidence],
    ) -> list[str]:
        rows = self._evidence_rows(evidence)
        evidence_numbers = {
            token
            for row in rows
            for value in row.values()
            for token in self._number_tokens(str(value))
        }
        generated_numbers = self._number_tokens(
            " ".join(
                [
                    insight.summary,
                    *insight.findings,
                    *insight.recommendations,
                ]
            )
        )
        problems = [
            f"Unsupported numeric value: {number}"
            for number in generated_numbers
            if number not in evidence_numbers
        ]
        text = " ".join(
            [insight.summary, *insight.findings, *insight.recommendations]
        ).lower()
        if any(re.search(pattern, text) for pattern in self.CAUSAL_PATTERNS):
            problems.append("Potentially unsupported causal language detected.")
        return list(dict.fromkeys(problems))

    def _safe_fallback(self, question: str, evidence: list[Evidence]) -> GeneratedInsight:
        aggregate = self._deterministic_aggregate(question, evidence)
        if aggregate:
            return aggregate
        missing_periods = self._deterministic_missing_period_comparison(evidence)
        if missing_periods:
            return missing_periods
        entity_trends = self._deterministic_entity_trends(question, evidence)
        if entity_trends:
            return entity_trends
        extreme_ties = self._deterministic_extreme_ties(
            question,
            evidence,
        )
        if extreme_ties:
            return extreme_ties
        comparative = self._comparative_metric_insight(
            question,
            self._evidence_rows(evidence),
        )
        if comparative:
            return comparative
        ranking = self._deterministic_ranking(question, evidence)
        if ranking:
            return ranking
        # If model synthesis fails, avoid substituting an explanation that
        # is not present in the verified evidence.
        if re.search(
            r"\b(?:why|reason|reasons|explain|explanation|cause|causes|"
            r"driver|drivers|factor|factors|contributor|contributors)\b",
            question or "",
            re.I,
        ):
            return GeneratedInsight(
                summary=(
                    "The requested explanation could not be verified from "
                    "the available evidence."
                ),
                findings=[
                    (
                        "No causal conclusion is supported by this "
                        "fallback summary."
                    )
                ],
                supporting_evidence=[
                    f"Verified step {item.step_id}: {item.objective}"
                    for item in evidence
                    if item.rows
                ],
                uncertainties=[
                    (
                        "A cause should only be stated when the verified "
                        "evidence directly supports it; associations alone "
                        "are not causal proof."
                    )
                ],
            )

        return GeneratedInsight(
            summary=(
                "The verified evidence does not support a more specific "
                "answer without introducing unsupported claims."
            ),
            findings=[],
            supporting_evidence=[
                f"Verified step {item.step_id}: {item.objective}"
                for item in evidence
                if item.rows
            ],
            uncertainties=[
                "The available verified evidence was insufficient for a more specific evidence-grounded statement."
            ],
        )

    @staticmethod
    def _deterministic_missing_period_comparison(
        evidence: list[Evidence],
    ) -> GeneratedInsight | None:
        partial = [
            item
            for item in evidence
            if item.requested_periods and item.unavailable_periods
        ]
        if len(partial) != 1:
            return None

        item = partial[0]
        missing = ", ".join(map(str, item.unavailable_periods))
        observed = ", ".join(map(str, item.observed_periods))
        if observed:
            summary = (
                "The requested comparison cannot be determined: the "
                f"query returned data for {observed}, but no matching "
                f"result for {missing}."
            )
            findings = [
                (
                    f"Results were returned for {observed}; no matching "
                    f"result was returned for {missing}."
                ),
                (
                    "Because not every requested period has a result, the "
                    "available evidence cannot establish whether the "
                    "metric increased or decreased."
                ),
            ]
        else:
            summary = (
                "The requested comparison cannot be determined because "
                "the query returned no matching results for the requested "
                f"periods: {missing}."
            )
            findings = [
                "No matching result was returned for any requested period.",
                (
                    "The available evidence cannot establish whether the "
                    "metric increased or decreased."
                ),
            ]

        return GeneratedInsight(
            summary=summary,
            findings=findings,
            supporting_evidence=[
                (
                    f"Verified step {item.step_id}: the aggregation "
                    "covered and grouped by all requested periods."
                )
            ],
            uncertainties=[
                "A comparison requires a matching result for every requested period."
            ],
        )

    @classmethod
    def _deterministic_period_comparison(
        cls,
        question: str,
        evidence: list[Evidence],
    ) -> GeneratedInsight | None:
        series = cls._period_series_facts(question, evidence)
        if series is None:
            return None
        points, metric = series
        earlier_year, earlier_value = points[0]
        later_year, later_value = points[-1]
        direction = "increased" if later_value > earlier_value else (
            "decreased" if later_value < earlier_value else "did not change"
        )
        change = later_value - earlier_value
        metric_label = re.sub(
            r"([a-z])([A-Z])",
            r"\1 \2",
            metric.replace("_", " "),
        ).lower()
        earlier_text = cls._normalize_number(earlier_value)
        later_text = cls._normalize_number(later_value)
        change_text = cls._normalize_number(abs(change))
        change_detail = (
            f"The absolute change was {change_text}."
            if change != 0
            else "The absolute change was 0."
        )
        percentage_requested = bool(
            re.search(
                r"\b(?:percentage|percent)\b|%",
                question,
                re.IGNORECASE,
            )
            or re.search(r"\b(?:growth|grew|grown)\b", question, re.IGNORECASE)
        )
        percentage_detail = ""
        if percentage_requested:
            if earlier_value == 0:
                percentage_detail = (
                    " Percentage change cannot be calculated because "
                    "the baseline value is zero."
                )
            else:
                percentage = (
                    abs(change) / abs(earlier_value) * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                percentage_text = cls._normalize_number(percentage)
                percentage_direction = (
                    "increase"
                    if change > 0
                    else "decrease"
                    if change < 0
                    else "change"
                )
                percentage_detail = (
                    f" The percentage {percentage_direction} was "
                    f"{percentage_text}%."
                )
        yearly_values = "; ".join(
            f"{value} in {year}"
            for year, value in points
        )
        annual_changes = [
            current - previous
            for (_, previous), (_, current) in zip(points, points[1:])
        ]
        if len(points) > 2:
            if all(delta > 0 for delta in annual_changes):
                trend_detail = " The metric increased in every observed year."
            elif all(delta < 0 for delta in annual_changes):
                trend_detail = " The metric decreased in every observed year."
            elif all(delta == 0 for delta in annual_changes):
                trend_detail = " The metric was unchanged in every observed year."
            else:
                trend_detail = " The metric fluctuated between observed years."
            yearly_detail = f" Annual values: {yearly_values}."
        else:
            trend_detail = ""
            yearly_detail = ""
        return GeneratedInsight(
            summary=(
                f"{metric_label.capitalize()} {direction} from "
                f"{earlier_year} to {later_year}: "
                f"{earlier_text} in {earlier_year} and {later_text} "
                f"in {later_year}. {change_detail}{percentage_detail}"
                f"{trend_detail}{yearly_detail}"
            ),
            findings=[
                (
                    f"{metric_label}: {earlier_text} in {earlier_year}; "
                    f"{later_text} in {later_year}. The value "
                    f"{direction} over this period."
                )
            ],
            supporting_evidence=[
                "Direction and values were compared from the verified "
                "database results for both requested years."
            ],
            uncertainties=[],
        )

    def generate(
        self,
        question: str,
        evidence: list[Evidence],
        interpretation_question: str | None = None,
    ) -> GeneratedInsight:
        # Deterministic answers are preferred whenever the verified result is
        # structurally sufficient. This prevents an LLM from changing a correct
        # aggregate/ranking into an unrelated narrative.
        analysis_question = " ".join(
            part
            for part in (question, interpretation_question or "")
            if part
        )
        missing_periods = self._deterministic_missing_period_comparison(evidence)
        if missing_periods:
            return missing_periods

        period_comparison = self._deterministic_period_comparison(
            analysis_question,
            evidence,
        )
        if period_comparison:
            return period_comparison

        entity_trends = self._deterministic_entity_trends(
            analysis_question,
            evidence,
        )
        if entity_trends:
            return entity_trends

        aggregate = self._deterministic_aggregate(question, evidence)
        if aggregate:
            return aggregate

        extreme_ties = self._deterministic_extreme_ties(
            question,
            evidence,
        )
        if extreme_ties:
            return extreme_ties

        comparative = self._comparative_metric_insight(
            analysis_question,
            self._evidence_rows(evidence),
        )
        if comparative:
            return comparative

        ranking = self._deterministic_ranking(question, evidence)
        if ranking:
            return ranking

        collection = self._large_complete_collection(question, evidence)
        if collection:
            return collection

        evidence_text = self._build_evidence_text(evidence)
        insight = self._generate_once(question, evidence_text)
        problems = self._validate_grounding(insight, evidence)
        if not problems:
            return insight

        corrected = self._generate_once(
            question,
            evidence_text,
            feedback="\n".join(f"- {problem}" for problem in problems),
        )
        if not self._validate_grounding(corrected, evidence):
            return corrected

        return self._safe_fallback(question, evidence)
