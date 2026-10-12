"""The turn record: ``metadata.agent_v2`` plus the legacy projections (ADR 0007).

``agent_v2`` is the runtime's only replay source; ``content``, ``round_texts``,
``tool_events``, ``thinking`` and the context projections are what legacy
readers keep using. The writer inserts the row ``in_progress`` at the first
round and rewrites it at each round boundary, so a cancel or restart keeps the
completed tool work. The record carries a keyed ``content_digest`` over the
persisted content and itself: a row replays from ``agent_v2`` only when the
digest verifies and nobody marked it ``edited``, so a rewritten or
client-forged record replays as plain text.
"""

from __future__ import annotations

import hmac
import json
import logging
import uuid
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional

from core.models import ChatMessage
from src.attachment_refs import persistable_message_content
from src.chat_agent import tools
from src.chat_agent.asks import RESOLVED_EXPIRED
from src.chat_agent.routing import CHAT_AGENT
# Relies on: _grant_key returns the persistent application key (the same bytes
# across restarts) or None when it is unavailable.
from src.tool_approval_scopes import _grant_key

logger = logging.getLogger(__name__)

CANCELLED_REASON = "Cancelled: the turn stopped before this call finished."
INTERRUPTED_REASON = "Cancelled: the server restarted before this call finished."
# ask_expired causes (spec #44, Stream events).
ASK_EXPIRED_CANCEL = "cancel"
ASK_EXPIRED_RESTART = "restart"


# Why a turn ended early (metadata.stop_reason; #65 renders it). A turn that
# ends because the model stopped calling tools has none.
STOP_MAX_ROUNDS = "max_rounds"
STOP_BUDGET = "budget"
# Stop, Finish and a server restart all interrupt the turn.
STOP_INTERRUPTED = "interrupted"
STOP_REPEAT_GUARD = "repeat_guard"
# A provider error or crash after the row was first checkpointed.
STOP_ERROR = "error"
STOP_ASK_EXPIRED = "ask_expired"


@dataclass
class TurnRecord:
    provider: str
    model: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Each round: {"blocks": [...]} plus "usage" when the provider reported it
    # (compaction reads it, #67) and "results" (tool-result envelopes) once its
    # calls ran. Block shapes: src/chat_agent/providers/__init__.py.
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    # The legacy tool_events projection, one entry per call (tools.ToolOutcome).
    tool_events: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    # Context the route resolved for this turn, projected as save_assistant_response
    # does: character_name, web_sources, rag_sources, memories_used.
    context: Dict[str, Any] = field(default_factory=dict)
    # The tool event of a call waiting on an ask: its card and no exit code
    # yet. Checkpointed so a restart can expire the ask (spec #44, Transcript).
    pending_ask: Optional[Dict[str, Any]] = None

    @property
    def round_texts(self) -> List[str]:
        return [
            "".join(block["text"] for block in r["blocks"] if block["type"] == "text")
            for r in self.rounds
        ]

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
        tool_events = self.tool_events + ([self.pending_ask] if self.pending_ask else [])
        if tool_events:
            metadata["tool_events"] = tool_events
        thinking = _thinking(self.rounds)
        if thinking:
            metadata["thinking"] = thinking
        metadata.update({key: value for key, value in self.context.items() if value})
        if self.stop_reason:
            metadata["stop_reason"] = self.stop_reason
        return metadata

    def content(self) -> str:
        # Relies on: persistable_message_content is what the session manager
        # stores for a string, so the digest is over the persisted content.
        return persistable_message_content("\n\n".join(text for text in self.round_texts if text))

    def close_open_calls(self, reason: str, ask_cause: str) -> None:
        """Give every call that has no result a ``cancelled`` one (exit 130).

        Every round must replay as complete call/result pairs; a call still
        waiting on an ask keeps its card, now resolved as expired.
        """
        if self.pending_ask:
            self.tool_events.append(self.pending_ask)
            self.pending_ask = None
        close_open_calls(self.rounds, self.tool_events, reason, ask_cause, self.model)


def close_open_calls(
    rounds: List[Dict[str, Any]], tool_events: List[Dict[str, Any]], reason: str, ask_cause: str, model: str,
) -> None:
    for number, turn_round in enumerate(rounds, start=1):
        results = turn_round.setdefault("results", [])
        answered = {result["call_id"] for result in results}
        for block in turn_round["blocks"]:
            if block["type"] != "tool_call" or block["call_id"] in answered:
                continue
            outcome = tools.ToolOutcome(
                block["call_id"], block["name"], block["arguments"],
                tools.STATUS_CANCELLED, tools.EXIT_CANCELLED, reason,
            )
            results.append(outcome.envelope())
            pending = next((e for e in tool_events if e.get("call_id") == block["call_id"]), None)
            if pending is None:
                tool_events.append(outcome.tool_event(number, model))
                continue
            pending.update(status=tools.STATUS_CANCELLED, exit_code=tools.EXIT_CANCELLED, error=True)
            ask = pending.get("ask_user")
            if isinstance(ask, dict) and not ask.get("resolved"):
                ask.update(resolved=RESOLVED_EXPIRED, expired_cause=ask_cause)
        if not results:
            turn_round.pop("results")


def _thinking(rounds: List[Dict[str, Any]]) -> str:
    """Readable reasoning for the legacy ``thinking`` projection: Chat
    Completions ``reasoning_content`` and xAI reasoning summaries."""
    parts = []
    for turn_round in rounds:
        for block in turn_round["blocks"]:
            if block["type"] != "reasoning":
                continue
            if block.get("reasoning_content"):
                parts.append(block["reasoning_content"])
            for summary in (block.get("item") or {}).get("summary") or ():
                if summary.get("text"):
                    parts.append(summary["text"])
    return "\n\n".join(parts)


def _digest(content: Any, agent_v2: Dict[str, Any]) -> Optional[str]:
    key = _grant_key()
    if key is None:
        return None
    unsigned = {k: v for k, v in agent_v2.items() if k != "content_digest"}
    payload = json.dumps(
        {"content": content, "agent_v2": unsigned}, sort_keys=True, separators=(",", ":"),
    )
    return hmac.new(key, payload.encode("utf-8"), sha256).hexdigest()


def _signed(content: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
    agent_v2 = metadata["agent_v2"]
    agent_v2.pop("content_digest", None)
    digest = _digest(content, agent_v2)
    if digest:
        agent_v2["content_digest"] = digest
    return metadata


def trusted_rounds(message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The rounds to replay for a history row, or None to replay it as text.

    Only a record this writer signed, over exactly the content now persisted,
    and never marked ``edited`` or still ``in_progress`` is trusted.
    """
    metadata = message.get("metadata") or {}
    agent_v2 = metadata.get("agent_v2")
    if not isinstance(agent_v2, dict) or metadata.get("edited") or metadata.get("in_progress"):
        return None
    claimed = agent_v2.get("content_digest")
    expected = _digest(message.get("content"), agent_v2)
    if not isinstance(claimed, str) or expected is None or not hmac.compare_digest(claimed, expected):
        return None
    return agent_v2.get("rounds") or None


class TurnWriter:
    """Checkpoints one turn's row: inserted at the first round, then rewritten."""

    def __init__(self, sess, session_id: str, record: TurnRecord, *, incognito: bool):
        self.sess = sess
        self.session_id = session_id
        self.record = record
        self.incognito = incognito
        self.message: Optional[ChatMessage] = None

    def checkpoint(self) -> None:
        # Nobody-mode turns live only in the request-local transcript, which
        # cannot survive a restart; they are written once, at finish.
        if not self.incognito:
            self._write(in_progress=True)

    def finish(self) -> Optional[str]:
        """Write the final row and return its message id (None in Nobody mode).

        A turn stopped mid-round keeps its completed calls; the rest get
        cancelled results.
        """
        self.record.close_open_calls(CANCELLED_REASON, ASK_EXPIRED_CANCEL)
        if self.incognito:
            # Relies on: build_chat_context reads Nobody-mode history back from
            # this request-local transcript on the next turn.
            from routes.chat_helpers import _append_incognito_message
            content = self.record.content()
            _append_incognito_message(
                self.session_id, "assistant", content,
                _signed(content, self._metadata(in_progress=False)),
            )
            return None
        self._write(in_progress=False)
        return (self.message.metadata or {}).get("_db_id")

    @property
    def started(self) -> bool:
        return self.message is not None

    def _metadata(self, *, in_progress: bool) -> Dict[str, Any]:
        metadata = self.record.metadata()
        if in_progress:
            metadata["in_progress"] = True
        elif self.record.stop_reason in (STOP_INTERRUPTED, STOP_ERROR):
            # The legacy marker the history renderer shows as stopped.
            metadata["stopped"] = True
        return metadata

    def _write(self, *, in_progress: bool) -> None:
        content = self.record.content()
        metadata = _signed(content, self._metadata(in_progress=in_progress))
        if self.message is None:
            self.message = ChatMessage("assistant", content, metadata=metadata)
            # Relies on: Session.add_message persists through the session
            # manager, which stamps the row id on metadata["_db_id"] and bumps
            # sidebar recency.
            self.sess.add_message(self.message)
            return
        current = self._find_row()
        kept = {k: current.metadata[k] for k in ("_db_id", "timestamp") if k in (current.metadata or {})}
        current.content = content
        current.metadata = {**metadata, **kept}
        self.message = current
        self.sess.update_message(current)

    def _find_row(self) -> ChatMessage:
        """The row now holding this turn: compaction and truncation re-create
        row ids, so it is found by turn id, falling back to the one inserted."""
        for entry in reversed(getattr(self.sess, "history", None) or ()):
            metadata = getattr(entry, "metadata", None)
            agent_v2 = metadata.get("agent_v2") if isinstance(metadata, dict) else None
            if isinstance(agent_v2, dict) and agent_v2.get("turn_id") == self.record.turn_id:
                return entry
        return self.message


def sweep_interrupted_turns() -> int:
    """Startup: rows a previous process left ``in_progress`` become interrupted.

    Calls without a result (including one waiting on an ask) get a cancelled
    result, so the row replays as complete pairs and its ask shows expired.
    Returns the number of rows swept.
    """
    from core.database import ChatMessage as DbChatMessage, SessionLocal

    db = SessionLocal()
    swept = 0
    try:
        rows = db.query(DbChatMessage).filter(
            DbChatMessage.role == "assistant",
            DbChatMessage.meta_data.like('%"in_progress"%'),
        ).all()
        for row in rows:
            try:
                metadata = json.loads(row.meta_data)
            except (TypeError, ValueError):
                continue
            agent_v2 = metadata.get("agent_v2") if isinstance(metadata, dict) else None
            if not metadata.get("in_progress") or not isinstance(agent_v2, dict):
                continue
            tool_events = metadata.setdefault("tool_events", [])
            close_open_calls(
                agent_v2.get("rounds") or [], tool_events, INTERRUPTED_REASON, ASK_EXPIRED_RESTART,
                agent_v2.get("model") or "",
            )
            if not tool_events:
                metadata.pop("tool_events")
            metadata.pop("in_progress")
            metadata["stop_reason"] = STOP_INTERRUPTED
            metadata["stopped"] = True
            row.meta_data = json.dumps(_signed(row.content, metadata))
            swept += 1
        db.commit()
    finally:
        db.close()
    if swept:
        logger.info("chat_agent: marked %d interrupted turn(s) left from the last run", swept)
    return swept
