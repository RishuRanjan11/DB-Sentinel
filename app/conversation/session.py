from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ConversationTurn:
    question: str
    result: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConversationSession:
    """In-memory conversation state scoped to exactly one organization."""

    conversation_id: str
    organization_id: str
    database_connection_ids: tuple[str, ...]
    turns: list[ConversationTurn] = field(default_factory=list)

    def add_turn(self, question: str, result: dict[str, Any]) -> None:
        self.turns.append(ConversationTurn(question=question, result=result))

    def history(self) -> list[dict[str, Any]]:
        return [
            {
                "question": turn.question,
                "result": turn.result,
                "created_at": turn.created_at.isoformat(),
            }
            for turn in self.turns
        ]
