"""Turns nobody can answer never hang; deleting a chat stops its run (ticket #61).

Driven through POST /api/chat_stream with the scripted provider of the route
tests. A turn is non-interactive when the request says ``interactive: false``
(notes agent-solve, group panes, compare synthesis) or arrives on an API token.
"""

import asyncio
from types import SimpleNamespace

import pytest

import routes.session_routes as session_routes
from src import agent_runs
from tests.chat_agent.test_chat_agent_asks_route import (  # noqa: F401 (fixtures)
    _bash,
    _events,
    _tool_messages,
    admin,
    executor,
    harness,
    send_and_answer,
    workspace,
)
from tests.chat_agent.test_chat_agent_route import _EmptyDb, _Request, _sse_events, _sse_tool_calls

# Far below the 30-minute ask timeout: an answer that waits at all fails the test.
NO_WAIT_SECONDS = 5


async def _never_answer(card):
    return None


async def _send_without_answering(harness, request):
    return await asyncio.wait_for(
        send_and_answer(harness, request, _never_answer), NO_WAIT_SECONDS,
    )



@pytest.mark.asyncio
async def test_notes_agent_solve_bash_step_is_denied_at_once(harness, admin, executor, workspace):
    harness.provider_rounds = [_bash("ls"), _sse_events(["I could not run ls."])]

    body = await _send_without_answering(
        harness, _Request("agent", allow_bash="true", workspace=workspace, interactive="false"),
    )

    [card] = [e["data"] for e in _events(body, "ask_user")]
    [resolved] = _events(body, "ask_resolved")
    assert (resolved["ask_id"], resolved["decision"], resolved["by"]) == (card["ask_id"], "deny", "runtime")
    assert executor == []
    [output] = _events(body, "tool_output")
    assert output["exit_code"] == 126
    _, second = harness.captured["provider_requests"]
    assert "no interactive user" in _tool_messages(second)[0].lower()
    [row] = harness.saved
    assert row.content == "I could not run ls."
    assert "stop_reason" not in row.metadata
    assert "in_progress" not in row.metadata
    assert row.metadata["tool_events"][0]["ask_user"]["resolved"] == "deny"


@pytest.mark.asyncio
async def test_api_token_turn_denies_every_ask_at_once(harness):
    harness.provider_rounds = [
        _sse_tool_calls(("ask_user", {"question": "Which list?", "options": [{"label": "Work"}, {"label": "Home"}]})),
        _sse_events(["No one to ask."]),
    ]
    request = _Request("agent")
    request.state = SimpleNamespace(
        current_user="alice", api_token=True, api_token_owner="alice", api_token_scopes=["chat"],
    )

    body = await _send_without_answering(harness, request)

    assert _events(body, "ask_resolved")[0]["by"] == "runtime"
    _, second = harness.captured["provider_requests"]
    assert "no interactive user" in _tool_messages(second)[0].lower()
    assert _events(body, "final_response")[0]["content"] == "No one to ask."


# --- deleting the chat stops its run ------------------------------------------


@pytest.fixture
def delete_route(monkeypatch, harness):
    seen = {}

    def delete_session(session_id, **kwargs):
        run = agent_runs._RUNS.get(session_id)
        seen["run_done_at_delete"] = bool(run and run.task and run.task.done())
        seen["writes_at_delete"] = list(harness.writes)
        return True

    manager = SimpleNamespace(delete_session=delete_session)
    monkeypatch.setattr(session_routes, "_verify_session_owner", lambda *a, **k: None)
    monkeypatch.setattr(session_routes, "effective_user", lambda request: "alice")
    monkeypatch.setattr(session_routes, "SessionLocal", _EmptyDb)
    monkeypatch.setattr("routes.chat_helpers.remove_session_sft_trace_rows", lambda *a: None)
    router = session_routes.setup_session_routes(manager, {})
    endpoints = {(route.path, method): route.endpoint for route in router.routes for method in route.methods}
    return SimpleNamespace(endpoints=endpoints, seen=seen)


@pytest.fixture
def recorded_writes(harness):
    """Every save the turn makes, in order (insert or in-place update)."""
    harness.writes = []
    add = harness.session.add_message

    def add_message(message):
        harness.writes.append(("add", message.content))
        add(message)

    def update_message(message):
        harness.writes.append(("update", message.content))
        return True

    harness.session.add_message = add_message
    harness.session.update_message = update_message


@pytest.mark.asyncio
@pytest.mark.parametrize("path, method", [
    ("/api/session/{sid}", "DELETE"),
    ("/api/session/{sid}/delete", "POST"),
])
async def test_deleting_a_chat_during_a_pending_ask_stops_the_run(
    harness, recorded_writes, delete_route, admin, executor, workspace, path, method,
):
    harness.provider_rounds = [_bash("ls"), _sse_events(["never reached"])]

    async def delete_instead_of_answering(card):
        await delete_route.endpoints[(path, method)](_Request(), "session-1")

    await asyncio.wait_for(send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace), delete_instead_of_answering,
    ), NO_WAIT_SECONDS)

    assert delete_route.seen["run_done_at_delete"] is True
    assert harness.writes == delete_route.seen["writes_at_delete"]
    assert executor == []
    assert len(harness.captured["provider_requests"]) == 1
    assert agent_runs.get_status("session-1") == "stopped"


@pytest.mark.asyncio
async def test_bulk_delete_stops_each_chats_run(
    harness, recorded_writes, delete_route, admin, executor, workspace,
):
    harness.provider_rounds = [_bash("ls"), _sse_events(["never reached"])]

    async def delete_instead_of_answering(card):
        request = _Request()

        async def body():
            return {"ids": ["session-1"]}

        request.json = body
        await delete_route.endpoints[("/api/sessions/bulk-delete", "POST")](request)

    await asyncio.wait_for(send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace), delete_instead_of_answering,
    ), NO_WAIT_SECONDS)

    assert delete_route.seen["run_done_at_delete"] is True
    assert harness.writes == delete_route.seen["writes_at_delete"]
    assert executor == []
