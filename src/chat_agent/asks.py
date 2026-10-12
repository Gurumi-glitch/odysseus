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
from typing import Dict, Optional, Tuple

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
ALL_DECISIONS = frozenset().union(*DECISIONS.values())

BY_USER = "user"
EXPIRED_TIMEOUT = "timeout"
# Stored as the card's ``resolved`` value when nobody answered.
RESOLVED_EXPIRED = "expired"


@dataclass(frozen=True)
class Answer:
    decision: str
    text: str = ""
    by: str = BY_USER


@dataclass
class _Pending:
    kind: str
    owner: str
    future: asyncio.Future


_PENDING: Dict[Tuple[str, str, str], _Pending] = {}


def owner_key(owner: str) -> str:
    # strip().lower() is how AuthManager keys usernames (same as store.py).
    return str(owner or "").strip().lower()


def _run_id(session_id: str) -> str:
    run = agent_runs.get_active_run(session_id)
    return run.run_id if run else ""


async def wait(session_id: str, owner: str, card: dict) -> Optional[Answer]:
    """Wait for the user's answer to this card; None when the ask timed out."""
    # The run draining this turn is the session's active run: starting
    # another run cancels this one, so no ask outlives its run.
    run = agent_runs.get_active_run(session_id)
    key = (session_id, run.run_id if run else "", card["ask_id"])
    pending = _Pending(card["ask_kind"], owner_key(owner), asyncio.get_running_loop().create_future())
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


def answer(session_id: str, owner: str, ask_id: str, decision: str, text: str = "") -> bool:
    """Resolve an open ask of the session's live run, asked of this owner.
    False when there is none or the decision does not fit its kind."""
    pending = _PENDING.get((session_id, _run_id(session_id), ask_id))
    if pending is None or pending.future.done() or pending.owner != owner_key(owner):
        return False
    if decision not in DECISIONS[pending.kind]:
        return False
    pending.future.set_result(Answer(decision, text))
    return True


def _approval_card(kind: str, session_id: str, question: str, description: str,
                   options: list, action: dict) -> dict:
    """The legacy approval card (tool_approvals.PendingToolApproval.public_payload)
    with this runtime's decisions and the ask's own kind."""
    ask_id = uuid.uuid4().hex
    return {
        "kind": "tool_approval", "approval_id": ask_id, "ask_id": ask_id, "ask_kind": kind,
        "session_id": session_id, "question": question, "description": description,
        "options": options, "action": action,
    }


# A scheduled task can deliver unattended, so its card spells out what it
# runs, when, on what trigger and where results go (#40).
_TASK_CARD_FIELDS = (
    ("task_id", "Task"), ("name", "Name"), ("stored_task", "Stored task"), ("prompt", "Prompt"),
    ("task_type", "Type"), ("action_name", "Built-in action"), ("schedule", "Schedule"),
    ("scheduled_time", "Time (UTC)"), ("scheduled_date", "Date"), ("weekdays", "Weekdays"),
    ("day_of_month", "Day of month"), ("cron_expression", "Cron"), ("trigger_type", "Trigger"),
    ("trigger_event", "Trigger event"), ("trigger_count", "Every N events"),
    ("output_target", "Delivers to"),
)


def permission_card(
    session_id: str, tool: str, content: str, rule: Optional[str], task: Optional[dict] = None,
) -> dict:
    """``rule``: what "Allow for this chat" would allow, None when the ask
    must be answered each time. ``task``: the manage_tasks fields to show."""
    description = "The agent wants to run this action and needs your approval."
    lines = [f"{label}: {task[key]}" for key, label in _TASK_CARD_FIELDS
             if task and task.get(key) not in (None, "")]
    if lines:
        description += "\n" + "\n".join(lines)
    options = [{"label": "Allow once", "value": ONCE, "description": "Run this action now; ask again next time."}]
    if rule is not None:
        options.append({"label": "Allow for this chat", "value": CHAT,
                        "description": f"Run it and allow {rule} for the rest of this chat."})
    options.append({"label": "Deny", "value": DENY, "description": "Do not run this action."})
    return _approval_card(
        PERMISSION, session_id, "Allow this action?", description, options, {"tool": tool, "content": content},
    )


def repeat_guard_card(session_id: str, tool: str, content: str, streak: int) -> dict:
    return _approval_card(
        REPEAT_GUARD, session_id, "The agent keeps repeating the same call. Continue?",
        f"This exact call was made {streak} times in a row.",
        [
            {"label": "Continue", "value": ONCE, "description": "Run the call and let the agent go on."},
            {"label": "Stop", "value": DENY, "description": "Stop this turn here."},
        ],
        {"tool": tool, "content": content},
    )


def question_card(session_id: str, question: dict) -> dict:
    """The ask_user tool's card as today, with the ask's id and kind."""
    return {**question, "ask_id": uuid.uuid4().hex, "ask_kind": QUESTION, "session_id": session_id}


def resolution_event(card: dict, reply: Optional[Answer]) -> dict:
    if reply is None:
        return {"type": "ask_expired", "ask_id": card["ask_id"], "cause": EXPIRED_TIMEOUT}
    event = {"type": "ask_resolved", "ask_id": card["ask_id"], "decision": reply.decision, "by": reply.by}
    if reply.text:
        event["reason"] = reply.text
    return event


def resolved_card(card: dict, reply: Optional[Answer]) -> dict:
    """The card as stored on its tool event, so history renders it read-only."""
    return {**card, "resolved": reply.decision if reply else RESOLVED_EXPIRED}
