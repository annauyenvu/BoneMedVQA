"""Session conversation state (no PHI storage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class ConversationTurn:
    question: str
    answer: str
    confidence: float
    abstained: bool
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    """In-memory multi-turn dialogue context for one session."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    turns: list[ConversationTurn] = field(default_factory=list)
    max_turns: int = 20

    def add(self, question: str, answer: str, confidence: float, abstained: bool, **meta: Any) -> None:
        self.turns.append(
            ConversationTurn(
                question=question,
                answer=answer,
                confidence=confidence,
                abstained=abstained,
                meta=meta,
            )
        )
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def history_text(self) -> str:
        lines = []
        for i, t in enumerate(self.turns, start=1):
            lines.append(f"Q{i}: {t.question}")
            lines.append(f"A{i}: {t.answer}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [
                {
                    "question": t.question,
                    "answer": t.answer,
                    "confidence": t.confidence,
                    "abstained": t.abstained,
                    "meta": t.meta,
                }
                for t in self.turns
            ],
        }
