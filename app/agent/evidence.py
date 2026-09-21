from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    step_id: int
    question: str
    objective: str
    sql: str
    rows: list[dict[str, Any]]
    schema_context: list[dict[str, Any]]
    logical_reason: str
    confidence: float
    depends_on: list[int] = field(default_factory=list)


@dataclass
class EvidenceStore:

    _evidence: list[Evidence] = field(
        default_factory=list
    )

    def add(
        self,
        evidence: Evidence,
    ) -> None:

        self._evidence.append(
            evidence
        )

    def all(self) -> list[Evidence]:
        return list(self._evidence)

    def get(
        self,
        step_id: int,
    ) -> Evidence | None:

        for evidence in self._evidence:

            if evidence.step_id == step_id:
                return evidence

        return None

    def clear(self) -> None:
        self._evidence.clear()