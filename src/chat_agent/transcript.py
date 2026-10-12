"""Write a turn's assistant row: ``metadata.agent_v2`` plus legacy projections.

``agent_v2`` is the runtime's replay source (ADR 0007); ``content``,
``round_texts`` and ``tool_events`` are projections legacy readers keep using. This ticket writes
the row once at turn end; per-round checkpointing comes in #57.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.models import ChatMessage
# Relies on: each takes the answer text and tool events shaped like legacy
# tool_events ("tool", "output", success per tool_result_is_successful, plus
# note_id/note_title and calendar "events" when the tool returned them), and
# returns the text with the matching titles turned into #note-/#email-/#event-
# links, leaving lines that already hold such a link alone.
from src.agent_loop import (
    _linkify_calendar_titles_from_tool_events,
    _linkify_email_titles_from_tool_events,
    _linkify_note_titles_from_tool_events,
)
from src.chat_agent.routing import CHAT_AGENT


# Why a turn ended early (metadata.stop_reason; #65 renders it). A turn that
# ends because the model stopped calling tools has none.
STOP_MAX_ROUNDS = "max_rounds"
STOP_BUDGET = "budget"
STOP_INTERRUPTED = "interrupted"
STOP_REPEAT_GUARD = "repeat_guard"
STOP_ASK_EXPIRED = "ask_expired"


@dataclass
class TurnRecord:
    provider: str
    model: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Each round: {"blocks": [...]} plus "usage" when the provider reported it
    # (compaction reads it, #67). Block shapes: src/chat_agent/providers/__init__.py.
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    # The legacy tool_events projection, one entry per call (tools.ToolOutcome).
    tool_events: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    # Each run call's result fields merged under its tool event: the ids the
    # final text's note, email and calendar titles link to. Not persisted.
    link_sources: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def round_texts(self) -> List[str]:
        """Each round's text; the last one with its titles linked (what
        final_response shows). agent_v2 keeps the model's own words."""
        texts = [
            "".join(block["text"] for block in r["blocks"] if block["type"] == "text")
            for r in self.rounds
        ]
        if texts:
            texts[-1] = linkify(texts[-1], self.link_sources)
        return texts

    def agent_v2(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "provider": self.provider,
            "model": self.model,
            "rounds": list(self.rounds),
        }

    def metadata(self) -> dict:
        metadata = {
            "runtime": CHAT_AGENT,
            "agent_v2": self.agent_v2(),
            "round_texts": list(self.round_texts),
            "model": self.model,
        }
        if self.tool_events:
            metadata["tool_events"] = list(self.tool_events)
        if self.stop_reason:
            metadata["stop_reason"] = self.stop_reason
        return metadata

    def content(self) -> str:
        return "\n\n".join(text for text in self.round_texts if text)


def linkify(text: str, sources: List[Dict[str, Any]]) -> str:
    """Link note, email and calendar titles the turn's tools returned."""
    text = _linkify_note_titles_from_tool_events(text, sources)
    text = _linkify_email_titles_from_tool_events(text, sources)
    return _linkify_calendar_titles_from_tool_events(text, sources)


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
