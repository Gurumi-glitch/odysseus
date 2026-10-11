"""Write a turn's assistant row: ``metadata.agent_v2`` plus legacy projections.

``agent_v2`` is the runtime's replay source (ADR 0007); ``content`` and
``round_texts`` are projections legacy readers keep using. This ticket writes
the row once at turn end; per-round checkpointing comes in #57.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from core.models import ChatMessage
from src.chat_agent.routing import CHAT_AGENT


@dataclass
class TurnRecord:
    provider: str
    model: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    round_texts: List[str] = field(default_factory=list)

    def agent_v2(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "provider": self.provider,
            "model": self.model,
            "rounds": [{"blocks": [{"type": "text", "text": text}]} for text in self.round_texts],
        }

    def metadata(self) -> dict:
        return {
            "runtime": CHAT_AGENT,
            "agent_v2": self.agent_v2(),
            "round_texts": list(self.round_texts),
            "model": self.model,
        }

    def content(self) -> str:
        return "\n\n".join(text for text in self.round_texts if text)


def save_turn(sess, session_id: str, record: TurnRecord, *, incognito: bool) -> Optional[str]:
    """Persist the turn row and return its message id (None in Nobody mode)."""
    if incognito:
        # Relies on: Nobody-mode turns live only in this request-local
        # transcript, which build_chat_context reads back for the next turn.
        from routes.chat_helpers import _append_incognito_message
        _append_incognito_message(session_id, "assistant", record.content(), record.metadata())
        return None
    message = ChatMessage("assistant", record.content(), metadata=record.metadata())
    # Relies on: Session.add_message persists through the session manager,
    # which stamps the row id on metadata["_db_id"] and bumps sidebar recency.
    sess.add_message(message)
    return (message.metadata or {}).get("_db_id")
