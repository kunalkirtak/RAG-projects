"""
memory.py
---------
Lightweight in-memory conversation history for the RAG pipeline. Keeps the
last N turns and can format them for inclusion in prompts (query rewriting,
follow-up questions, etc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class Turn:
    """A single question/answer turn in a conversation."""

    query: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationMemory:
    """Fixed-size sliding-window conversation memory.

    Args:
        max_turns: Maximum number of turns to retain. Older turns are
                   dropped once this limit is exceeded.
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self._turns: List[Turn] = []

    def add_turn(self, query: str, answer: str, sources: List[Dict[str, Any]] | None = None) -> None:
        """Record a new conversation turn."""
        self._turns.append(Turn(query=query, answer=answer, sources=sources or []))
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def get_history(self) -> List[Turn]:
        """Return the retained turns, oldest first."""
        return list(self._turns)

    def format_for_prompt(self, max_turns: int | None = None) -> str:
        """Render recent history as plain text for inclusion in an LLM prompt."""
        turns = self._turns[-max_turns:] if max_turns else self._turns
        if not turns:
            return ""
        lines = []
        for turn in turns:
            lines.append(f"User: {turn.query}")
            lines.append(f"Assistant: {turn.answer}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Discard all stored turns."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
