"""Asks wait in place inside the turn (ticket #56), driven through the chat route.

The browser reads the turn's SSE stream; when an ``ask_user`` card arrives it
answers through ``POST /api/chat/ask/{sid}`` while the same stream stays open.
"""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.chat_routes as chat_routes
import src.tool_execution as tool_execution
import src.tool_security as tool_security
from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
from src import interactive_gate
from src.chat_agent import asks, permissions, store
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _sse_tool_calls,
    _switch_on,
)

ASK_PATH = "/api/chat/ask/{session_id}"


@pytest.fixture
def harness(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path)
    _switch_on()
    # The ask endpoint answers as whoever the request is signed in as.
    monkeypatch.setattr(chat_routes, "effective_user", lambda request: request.state.current_user)
    return harness


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)


@pytest.fixture
def workspace(tmp_path):
    # Process tools launch only inside a sealed workspace, as on legacy.
    path = tmp_path / "workspace"
    path.mkdir()
    return str(path)


@pytest.fixture
def executor(monkeypatch):
    ran = []

    async def run(block, *args, **kwargs):
        ran.append(block.content)
        return block.tool_type, {"output": "file.txt", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)
    return ran


def answer_request(user="alice", **headers):
    return SimpleNamespace(
        headers=headers,
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
        state=SimpleNamespace(current_user=user),
    )


async def post_answer(harness, request, ask_id, decision, text="", session_id="session-1"):
    return await harness.routes[ASK_PATH](
        request, session_id, {"ask_id": ask_id, "decision": decision, "text": text},
    )


async def send_and_answer(harness, request, on_ask):
    """Stream the turn; ``on_ask(card)`` answers each ask_user card. Returns the body."""
    response = await harness.endpoint(request)
    body = []
    async for chunk in response.body_iterator:
        body.append(chunk)
        for line in chunk.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                event = json.loads(line[6:])
                if event.get("type") == "ask_user":
                    await on_ask(event["data"])
    return "".join(body)


def answering(harness, *decisions):
    """An ``on_ask`` that answers the cards in order with these decisions."""
    queue = list(decisions)

    async def on_ask(card):
        await post_answer(harness, answer_request(), card["ask_id"], *queue.pop(0))
    return on_ask


def _events(body, kind):
    return [e for e in _data_events(body) if e.get("type") == kind]


def _tool_messages(payload):
    return [m["content"] for m in payload["messages"] if m["role"] == "tool"]


def _bash(command):
    return _sse_tool_calls(("bash", {"command": command}))


@pytest.mark.asyncio
async def test_bash_asks_and_allow_once_runs_it_in_the_same_turn(harness, admin, executor, workspace):
    harness.provider_rounds = [_bash("ls"), _sse_events(["You have file.txt."])]

    body = await send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace),
        answering(harness, ("once",)),
    )

    [ask] = _events(body, "ask_user")
    card = ask["data"]
    assert card["kind"] == "tool_approval"
    assert card["approval_id"] == card["ask_id"]
    assert card["ask_kind"] == "permission"
    assert [o["value"] for o in card["options"]] == ["once", "chat", "deny"]
    assert card["action"] == {"tool": "bash", "content": "ls"}
    assert _events(body, "ask_resolved") == [
        {"type": "ask_resolved", "ask_id": card["ask_id"], "decision": "once", "by": "user"},
    ]
    assert executor == ["ls"]
    [output] = _events(body, "tool_output")
    assert output["exit_code"] == 0
    _, second = harness.captured["provider_requests"]
    assert "file.txt" in _tool_messages(second)[0]
    [row] = harness.saved
    assert row.content == "You have file.txt."
    [tool_event] = row.metadata["tool_events"]
    assert tool_event["ask_user"]["resolved"] == "once"


@pytest.mark.asyncio
async def test_deny_gives_the_model_a_denial_with_the_reason_and_the_turn_continues(
    harness, admin, executor, workspace,
):
    harness.provider_rounds = [_bash("rm -rf build"), _sse_events(["OK, I left build alone."])]

    body = await send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace),
        answering(harness, ("deny", "keep the build")),
    )

    assert executor == []
    [resolved] = _events(body, "ask_resolved")
    assert (resolved["decision"], resolved["reason"]) == ("deny", "keep the build")
    [output] = _events(body, "tool_output")
    assert output["exit_code"] == 126
    _, second = harness.captured["provider_requests"]
    denial = _tool_messages(second)[0]
    assert "Denied by the user" in denial and "keep the build" in denial
    [row] = harness.saved
    assert row.content == "OK, I left build alone."
    assert "stop_reason" not in row.metadata
    assert row.metadata["tool_events"][0]["ask_user"]["resolved"] == "deny"


@pytest.mark.asyncio
async def test_allow_for_this_chat_allows_the_whole_shell_tool_in_this_chat_only(
    harness, admin, executor, workspace,
):
    harness.provider_rounds = [
        _bash("ls"), _bash("ls -la"), _bash("ls; rm -rf /tmp/x"), _sse_events(["Done."]),
    ]
    asked = []

    async def on_ask(card):
        asked.append(card["action"]["content"])
        decision = "chat" if len(asked) == 1 else "deny"
        await post_answer(harness, answer_request(), card["ask_id"], decision)

    body = await send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace), on_ask,
    )

    # Shell specifiers are not supported yet: the answer allows bash as a whole.
    assert asked == ["ls"]
    assert (_events(body, "ask_user")[0]["data"]["options"][1]["description"]
            == "Run it and allow bash for the rest of this chat.")
    assert executor == ["ls", "ls -la", "ls; rm -rf /tmp/x"]
    assert store.session_rules("alice", "session-1") == [permissions.Rule("bash").record()]
    assert store.session_rules("alice", "session-2") == []
    assert store.session_rules("bob", "session-1") == []


@pytest.mark.asyncio
async def test_the_ask_endpoint_refuses_loopback_bearer_other_owners_and_other_sessions(
    harness, admin, executor, workspace,
):
    harness.provider_rounds = [_bash("ls"), _sse_events(["Done."])]
    refusals = []

    async def on_ask(card):
        ask_id = card["ask_id"]
        for request in (
            answer_request(**{INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN}),
            SimpleNamespace(**{**vars(answer_request()), "state": SimpleNamespace(
                current_user="alice", api_token=True, api_token_owner="alice")}),
        ):
            with pytest.raises(HTTPException) as refused:
                await post_answer(harness, request, ask_id, "once")
            refusals.append(refused.value.status_code)
        for request, session_id in ((answer_request("bob"), "session-1"), (answer_request(), "session-2")):
            with pytest.raises(HTTPException) as refused:
                await post_answer(harness, request, ask_id, "once", session_id=session_id)
            refusals.append(refused.value.status_code)
        with pytest.raises(HTTPException) as refused:
            await post_answer(harness, answer_request(), ask_id, "answer")
        refusals.append(refused.value.status_code)
        await post_answer(harness, answer_request(), ask_id, "deny")

    body = await send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace), on_ask,
    )

    assert refusals == [403, 403, 409, 409, 409]
    assert executor == []
    assert _events(body, "ask_resolved")[0]["decision"] == "deny"


@pytest.mark.asyncio
async def test_an_unanswered_ask_expires_and_ends_the_turn_keeping_the_work(
    harness, admin, executor, workspace, monkeypatch,
):
    monkeypatch.setattr(asks, "ASK_TIMEOUT_SECONDS", 0.05)
    harness.provider_rounds = [
        _sse_tool_calls(("manage_notes", {"action": "list"})),
        _bash("ls"),
        _sse_events(["never reached"]),
    ]

    async def ignore(card):
        return None

    body = await send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace), ignore,
    )

    [card] = [e["data"] for e in _events(body, "ask_user")]
    assert _events(body, "ask_expired") == [
        {"type": "ask_expired", "ask_id": card["ask_id"], "cause": "timeout"},
    ]
    assert len(harness.captured["provider_requests"]) == 2
    [row] = harness.saved
    assert row.metadata["stop_reason"] == "ask_expired"
    first, second = row.metadata["tool_events"]
    assert first["exit_code"] == 0
    assert (second["exit_code"], second["ask_user"]["resolved"]) == (126, "expired")


@pytest.mark.asyncio
async def test_a_pending_ask_does_not_hold_background_work(harness, admin, workspace, monkeypatch):
    # The tab is hidden: no browser heartbeat, no foreground requests.
    monkeypatch.setattr(interactive_gate, "_LAST_BROWSER_ACTIVITY", 0.0)
    monkeypatch.setattr(interactive_gate, "_ACTIVE_REQUESTS", 0)
    monkeypatch.setattr(interactive_gate, "_LAST_ACTIVITY", 0.0)
    monkeypatch.setattr(chat_routes, "_active_streams", {"session-1": {"status": "streaming"}})
    harness.provider_rounds = [_bash("ls"), _sse_events(["Done."])]
    seen = {}

    async def run(block, *args, **kwargs):
        seen["while_running"] = interactive_gate.has_foreground_activity(time.monotonic())
        return block.tool_type, {"output": "", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)

    async def on_ask(card):
        seen["while_asking"] = interactive_gate.has_foreground_activity(time.monotonic())
        # A due scheduled task waits on this gate before it runs.
        seen["task_waited"] = await asyncio.wait_for(
            interactive_gate.wait_for_interactive_quiet("scheduled task"), timeout=2,
        )
        await post_answer(harness, answer_request(), card["ask_id"], "once")

    await send_and_answer(harness, _Request("agent", allow_bash="true", workspace=workspace), on_ask)

    assert seen == {"while_asking": False, "task_waited": False, "while_running": True}


@pytest.mark.asyncio
async def test_ask_user_question_waits_for_the_answer_in_the_same_turn(harness):
    harness.provider_rounds = [
        _sse_tool_calls(("ask_user", {
            "question": "Which list?", "options": [{"label": "Work"}, {"label": "Home"}],
        })),
        _sse_events(["Using Home."]),
    ]

    body = await send_and_answer(harness, _Request("agent"), answering(harness, ("answer", "Home")))

    [ask] = _events(body, "ask_user")
    card = ask["data"]
    assert (card["ask_kind"], card["question"]) == ("question", "Which list?")
    assert "kind" not in card
    _, second = harness.captured["provider_requests"]
    assert _tool_messages(second)[0] == "The user answered: Home"
    [row] = harness.saved
    assert row.metadata["tool_events"][0]["ask_user"]["resolved"] == "answer"
