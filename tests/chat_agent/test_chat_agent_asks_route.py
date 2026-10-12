"""Asks wait in place inside the turn (ticket #56), driven through the chat route.

The browser reads the turn's SSE stream; when an ``ask_user`` card arrives it
answers through ``POST /api/chat/ask/{sid}`` while the same stream stays open.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import src.tool_execution as tool_execution
import src.tool_security as tool_security
from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
from src import agent_runs
from src.chat_agent import asks
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _sse_events,
    _sse_tool_calls,
    _switch_on,
)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path)
    _switch_on()
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
def bash_runs(monkeypatch):
    ran = []

    async def executor(block, *args, **kwargs):
        ran.append(block.content)
        return "bash", {"output": "file.txt", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", executor)
    return ran


def _answer_request(**headers):
    request = SimpleNamespace(
        headers=headers,
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
        state=SimpleNamespace(current_user="alice"),
    )
    return request


async def _send_and_answer(harness, request, answers):
    """Stream the turn; answer each ``ask_user`` card with the next answer body."""
    answers = list(answers)
    response = await harness.endpoint(request)
    events = []
    async for chunk in response.body_iterator:
        for line in chunk.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            events.append(event)
            if event.get("type") == "ask_user":
                ask_id = event["data"]["ask_id"]
                await harness.routes["/api/chat/ask/{session_id}"](
                    _answer_request(), "session-1", {"ask_id": ask_id, **answers.pop(0)},
                )
    return events


def _tool_messages(payload):
    return [m for m in payload["messages"] if m["role"] == "tool"]


@pytest.mark.asyncio
async def test_bash_asks_and_allow_once_runs_it_in_the_same_turn(harness, admin, bash_runs, workspace):
    harness.provider_rounds = [
        _sse_tool_calls(("bash", {"command": "ls"})),
        _sse_events(["You have file.txt."]),
    ]

    events = await _send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace), [{"decision": "once"}],
    )

    [ask] = [e for e in events if e.get("type") == "ask_user"]
    card = ask["data"]
    assert card["kind"] == "tool_approval"
    assert card["approval_id"] == card["ask_id"]
    assert card["ask_kind"] == "permission"
    assert [o["value"] for o in card["options"]] == ["once", "chat", "deny"]
    assert card["action"]["tool"] == "bash"
    assert {"type": "ask_resolved", "ask_id": card["ask_id"], "decision": "once", "by": "user"} in events
    assert bash_runs == ["ls"], [e for e in events if e.get("type") == "tool_output"]
    [output] = [e for e in events if e.get("type") == "tool_output"]
    assert output["exit_code"] == 0
    _, second = harness.captured["provider_requests"]
    assert "file.txt" in _tool_messages(second)[0]["content"]
    [row] = harness.saved
    assert row.content == "You have file.txt."
    [tool_event] = row.metadata["tool_events"]
    assert tool_event["ask_user"]["resolved"] == "once"
