"""Asks of the chat_agent runtime: a question to the user that the turn awaits.

An ask is a Future inside the detached run, keyed by session, run and ask id
(spec #44, Asks). The run is flagged ``awaiting_user`` while it waits, so the
background gate does not hold the instance for one open question. The only
answer path is ``POST /api/chat/ask/{sid}``; a chat_stream send would cancel
the waiting run.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from src import agent_runs

ASK_TIMEOUT_SECONDS = 30 * 60

PERMISSION = "permission"
QUESTION = "question"
REPEAT_GUARD = "repeat_guard"

ONCE = "once"
CHAT = "chat"
DENY = "deny"
ANSWER = "answer"
# Decisions each kind of ask accepts. A repeat-guard "once" lets the model go on.
DECISIONS: Dict[str, frozenset] = {
    PERMISSION: frozenset({ONCE, CHAT, DENY}),
    QUESTION: frozenset({ANSWER, DENY}),
    REPEAT_GUARD: frozenset({ONCE, DENY}),
}

BY_USER = "user"
EXPIRED_TIMEOUT = "timeout"


@dataclass(frozen=True)
class Answer:
    decision: str
    text: str = ""
    by: str = BY_USER


@dataclass
class _Pending:
    kind: str
    future: asyncio.Future


_PENDING: Dict[Tuple[str, str, str], _Pending] = {}
# "Allow for this chat", interim and in memory until the runtime store holds
# session rules (#62): (owner, session id) -> canonical tool names.
_SESSION_GRANTS: Dict[Tuple[str, str], Set[str]] = {}


def new_ask_id() -> str:
    return uuid.uuid4().hex


def _run_id(session_id: str) -> str:
    run = agent_runs.get_active_run(session_id)
    return run.run_id if run else ""


async def wait(session_id: str, ask_id: str, kind: str) -> Optional[Answer]:
    """Wait for the user's answer; None when the ask timed out."""
    # The run draining this turn is the session's active run: starting
    # another run cancels this one, so no ask outlives its run.
    run = agent_runs.get_active_run(session_id)
    key = (session_id, run.run_id if run else "", ask_id)
    pending = _Pending(kind, asyncio.get_running_loop().create_future())
    _PENDING[key] = pending
    if run is not None:
        run.awaiting_user = True
    try:
        return await asyncio.wait_for(pending.future, ASK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return None
    finally:
        _PENDING.pop(key, None)
        if run is not None:
            run.awaiting_user = False


def answer(session_id: str, ask_id: str, decision: str, text: str = "") -> bool:
    """Resolve an open ask of the session's live run. False when there is none
    or the decision does not fit its kind."""
    pending = _PENDING.get((session_id, _run_id(session_id), ask_id))
    if pending is None or pending.future.done():
        return False
    if decision not in DECISIONS[pending.kind]:
        return False
    pending.future.set_result(Answer(decision, text))
    return True


def grant_for_chat(owner: str, session_id: str, tool: str) -> None:
    _SESSION_GRANTS.setdefault((owner, session_id), set()).add(tool)


def granted_for_chat(owner: str, session_id: str, tool: str) -> bool:
    return tool in _SESSION_GRANTS.get((owner, session_id), ())


def permission_card(ask_id: str, session_id: str, tool: str, content: str) -> dict:
    """The legacy approval card (tool_approvals.PendingToolApproval.public_payload)
    with this runtime's decisions and the ask's own kind."""
    return {
        "kind": "tool_approval",
        "approval_id": ask_id,
        "ask_id": ask_id,
        "ask_kind": PERMISSION,
        "session_id": session_id,
        "question": "Allow this action?",
        "description": "The agent wants to run this action and needs your approval.",
        "options": [
            {"label": "Allow once", "value": ONCE,
             "description": "Run this action now; ask again next time."},
            {"label": "Allow for this chat", "value": CHAT,
             "description": "Run it and stop asking for this tool in this chat."},
            {"label": "Deny", "value": DENY, "description": "Do not run this action."},
        ],
        "action": {"tool": tool, "content": content},
    }


def question_card(ask_id: str, session_id: str, question: dict) -> dict:
    """The ask_user tool's card as today, with the ask's id and kind."""
    return {**question, "ask_id": ask_id, "ask_kind": QUESTION, "session_id": session_id}


def resolution_event(ask_id: str, reply: Optional[Answer]) -> dict:
    if reply is None:
        return {"type": "ask_expired", "ask_id": ask_id, "cause": EXPIRED_TIMEOUT}
    event = {"type": "ask_resolved", "ask_id": ask_id, "decision": reply.decision, "by": reply.by}
    if reply.text:
        event["reason"] = reply.text
    return event


def resolved_card(card: dict, reply: Optional[Answer]) -> dict:
    """The card as stored on its tool event, so history renders it read-only."""
    return {**card, "resolved": reply.decision if reply else "expired"}
