"""Write a turn's assistant row: ``metadata.agent_v2`` plus legacy projections.

``agent_v2`` is the runtime's replay source (ADR 0007); ``content``,
``round_texts`` and ``tool_events`` are projections legacy readers keep using.
This ticket writes the row once at turn end; per-round checkpointing comes in
#57.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from core.models import ChatMessage
from src.chat_agent.providers.chat_completions import ToolCall
from src.chat_agent.routing import CHAT_AGENT
from src.chat_agent.tools import ToolOutcome


@dataclass
class Round:
    text: str = ""
    calls: List[ToolCall] = field(default_factory=list)
    outcomes: List[ToolOutcome] = field(default_factory=list)

    def blocks(self) -> List[dict]:
        blocks: List[dict] = [{"type": "text", "text": self.text}] if self.text else []
        blocks += [
            {"type": "tool_call", "call_id": c.call_id, "name": c.name, "arguments": c.arguments}
            for c in self.calls
        ]
        blocks += [
            {
                "type": "tool_result", "call_id": o.call_id, "ok": o.ok,
                "status": o.status, "exit_code": o.exit_code, "content": o.model_text,
            }
            for o in self.outcomes
        ]
        return blocks


@dataclass
class TurnRecord:
    provider: str
    model: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    rounds: List[Round] = field(default_factory=list)

    def agent_v2(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "provider": self.provider,
            "model": self.model,
            "rounds": [{"blocks": r.blocks()} for r in self.rounds],
        }

    def tool_events(self) -> List[dict]:
        events = []
        for number, round_ in enumerate(self.rounds, start=1):
            for outcome in round_.outcomes:
                event = {
                    "round": number,
                    "model": self.model,
                    "tool": outcome.tool,
                    "desc": outcome.desc,
                    "command": outcome.command,
                    "output": outcome.model_text,
                    "exit_code": outcome.exit_code,
                    "status": outcome.status,
                    "call_id": outcome.call_id,
                }
                if not outcome.ok:
                    event["error"] = True
                events.append(event)
        return events

    def round_texts(self) -> List[str]:
        return [r.text for r in self.rounds]

    def metadata(self) -> dict:
        metadata = {
            "runtime": CHAT_AGENT,
            "agent_v2": self.agent_v2(),
            "round_texts": self.round_texts(),
            "model": self.model,
        }
        tool_events = self.tool_events()
        if tool_events:
            metadata["tool_events"] = tool_events
        return metadata

    def content(self) -> str:
        return "\n\n".join(r.text for r in self.rounds if r.text)


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
