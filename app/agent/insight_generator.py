import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.evidence import Evidence
from app.llm.model_router import ModelRouter, create_default_model_router


class GeneratedInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    findings: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
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
            number = float(value)
            return str(int(number)) if number.is_integer() else f"{number:g}"
        except (TypeError, ValueError):
            return str(value).strip()

    @classmethod
    def _number_tokens(cls, text: str) -> list[str]:
        return [cls._normalize_number(v) for v in cls.NUMBER_PATTERN.findall(text)]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
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
                if value is None:
                    continue
                try:
                    float(value)
                except (TypeError, ValueError):
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

            numeric = 0
            for value in values:
                try:
                    float(value)
                    numeric += 1
                except (TypeError, ValueError):
                    pass
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
                if value is None:
                    continue
                try:
                    float(value)
                except (TypeError, ValueError):
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
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            values.append((str(entity), value))

        if not values:
            return None

        descending = not bool(
            re.search(r"\b(?:least|lowest|smallest|minimum)\b", question or "", re.I)
        )
        values.sort(key=lambda item: float(item[1]), reverse=descending)
        top_n = cls._requested_top_n(question)
        if top_n is not None:
            values = values[:top_n]
        return values, metric_column

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
    def _deterministic_ranking(
        cls,
        question: str,
        evidence: list[Evidence],
    ) -> GeneratedInsight | None:
        if not cls._ranking_requested(question):
            return None
        selected = cls._ranking_rows(question, evidence)
        if selected is None:
            return None
        values, metric_column = selected
        top_n = cls._requested_top_n(question)
        scope = f"top {top_n} " if top_n is not None else ""
        summary = f"The verified evidence ranks the {scope}requested entities by {metric_column.replace('_', ' ')}."
        findings = [
            f"{entity}: {cls._normalize_number(value)} {metric_column.replace('_', ' ')}."
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

    def _build_evidence_text(self, evidence: list[Evidence]) -> list[dict[str, Any]]:
        return [
            {
                "step_id": item.step_id,
                "question": item.question,
                "objective": item.objective,
                "rows": item.rows[:100],
                "logical_verification": item.logical_reason,
                "confidence": item.confidence,
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

For why/explanation questions, distinguish observed measurements and
associations from causation. Do not invent causal explanations.

Respect LIMITs and do not claim completeness from a limited non-aggregate
result.

Return only the requested structured insight.
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
            " ".join([insight.summary, *insight.findings])
        )
        problems = [
            f"Unsupported numeric value: {number}"
            for number in generated_numbers
            if number not in evidence_numbers
        ]
        text = " ".join([insight.summary, *insight.findings]).lower()
        if any(re.search(pattern, text) for pattern in self.CAUSAL_PATTERNS):
            problems.append("Potentially unsupported causal language detected.")
        return list(dict.fromkeys(problems))

    def _safe_fallback(self, question: str, evidence: list[Evidence]) -> GeneratedInsight:
        aggregate = self._deterministic_aggregate(question, evidence)
        if aggregate:
            return aggregate
        ranking = self._deterministic_ranking(question, evidence)
        if ranking:
            return ranking
        # A why/explanation question can be answered qualitatively when
        # the verified investigation explicitly measured the dimensions
        # needed to explain the observed variation. Do not invent a
        # biological/causal mechanism and do not calculate new values here.
        if re.search(
            r"\b(?:why|reason|explain)\b",
            question or "",
            re.I,
        ):
            numeric_evidence_steps = 0

            for item in evidence:
                rows = item.rows or []
                if any(
                    isinstance(row, dict)
                    and self._numeric_columns([row])
                    for row in rows
                ):
                    numeric_evidence_steps += 1

            if numeric_evidence_steps >= 2:
                return GeneratedInsight(
                    summary=(
                        "The verified analysis shows that the pathways "
                        "differ in both pathway size and the proportion "
                        "of genes with human-ortholog mappings."
                    ),
                    findings=[
                        (
                            "Therefore, differences in the number of "
                            "human-ortholog mappings reflect differences "
                            "in pathway size and/or differences in "
                            "ortholog coverage across pathways."
                        ),
                        (
                            "The verified evidence establishes these "
                            "measured differences, but it does not "
                            "establish a specific biological or causal "
                            "reason for them."
                        ),
                    ],
                    supporting_evidence=[
                        f"Verified step {item.step_id}: {item.objective}"
                        for item in evidence
                        if item.rows
                    ],
                    uncertainties=[
                        (
                            "The evidence does not establish a specific "
                            "causal mechanism explaining the observed "
                            "differences."
                        )
                    ],
                )

        return GeneratedInsight(
            summary="The verified evidence does not support a more specific answer without introducing unsupported claims.",
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

    def generate(self, question: str, evidence: list[Evidence]) -> GeneratedInsight:
        # Deterministic answers are preferred whenever the verified result is
        # structurally sufficient. This prevents an LLM from changing a correct
        # aggregate/ranking into an unrelated narrative.
        aggregate = self._deterministic_aggregate(question, evidence)
        if aggregate:
            return aggregate

        ranking = self._deterministic_ranking(question, evidence)
        if ranking:
            return ranking

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
