"""Steering messages join the running turn (ticket #63), driven through the chat route.

The browser posts ``POST /api/chat/steer/{sid}`` while the turn's stream is
open; the text joins the turn at the next round boundary.
"""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import routes.chat_agent_routes as chat_agent_routes
import routes.chat_routes as chat_routes
import src.tool_execution as tool_execution
from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
from core.models import ChatMessage
from src import agent_runs
from src.chat_agent import classifier
from src.chat_agent.providers import chat_completions
from tests.chat_agent.test_chat_agent_asks_route import (  # noqa: F401 (fixtures)
    _bash,
    _events,
    admin,
    answer_request,
    harness,
    post_answer,
    send_and_answer,
    workspace,
)
from tests.chat_agent.test_chat_agent_route import _Request, _sse_events, _sse_tool_calls

STEER_PATH = "/api/chat/steer/{session_id}"


async def post_steer(harness, request, text, session_id="session-1"):
    return await harness.routes[STEER_PATH](request, session_id, {"text": text})


def _user_texts(payload):
    return [m["content"] for m in payload["messages"] if m["role"] == "user"]


@pytest.fixture
def steer_while_running(monkeypatch, harness):
    """The tool executor: each run call sends the next scripted steer first,
    as a user typing while the tool runs."""
    steers = []
    ran = []

    async def run(block, *args, **kwargs):
        ran.append(block.content)
        if steers:
            await post_steer(harness, answer_request(), steers.pop(0))
        return block.tool_type, {"output": "file.txt", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)
    return steers


async def _no_ask(card):
    raise AssertionError(f"unexpected ask: {card}")


@pytest.mark.asyncio
async def test_a_steer_sent_mid_turn_is_used_at_the_next_round_and_no_new_turn_starts(
    harness, steer_while_running,
):
    steer_while_running.append("Only list the notes from today.")
    harness.provider_rounds = [
        _sse_tool_calls(("manage_notes", {"action": "list"})),
        _sse_events(["Here are today's notes."]),
    ]

    body = await send_and_answer(harness, _Request("agent"), _no_ask)

    first, second = harness.captured["provider_requests"]
    assert "Only list the notes from today." not in _user_texts(first)
    assert _user_texts(second)[-1] == "Only list the notes from today."
    # The steer follows the round's tool result in the request.
    assert second["messages"][-2]["role"] == "tool"
    [row] = harness.saved
    first_round, last_round = row.metadata["agent_v2"]["rounds"]
    assert first_round["steers"] == [{"type": "text", "text": "Only list the notes from today."}]
    assert "steers" not in last_round
    assert row.metadata["steers"] == [{"round": 1, "text": "Only list the notes from today."}]
    assert row.content == "> ↳ Only list the notes from today.\n\nHere are today's notes."
    assert _events(body, "final_response")[0]["content"] == "Here are today's notes."


@pytest.fixture
def steer_during_request(monkeypatch, harness):
    """Steers to send while the provider answers request N (0-based): the
    user typing while the model streams."""
    plan = {}

    async def provider(request):
        index = len(harness.captured["provider_requests"])
        harness.captured["provider_requests"].append(json.loads(request.content))
        for text in plan.pop(index, ()):
            await post_steer(harness, answer_request(), text)
        return httpx.Response(
            200, content=harness.provider_rounds.pop(0), headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)
    return plan


@pytest.mark.asyncio
async def test_a_steer_sent_while_the_model_answers_without_tools_gets_one_more_round(
    harness, steer_during_request,
):
    steer_during_request[0] = ["Also say it in French."]
    harness.provider_rounds = [_sse_events(["Hello!"]), _sse_events(["Bonjour !"])]

    await send_and_answer(harness, _Request("agent"), _no_ask)

    _, second = harness.captured["provider_requests"]
    assert second["messages"][-2] == {"role": "assistant", "content": "Hello!"}
    assert second["messages"][-1] == {"role": "user", "content": "Also say it in French."}
    [row] = harness.saved
    assert row.content == "Hello!\n\n> ↳ Also say it in French.\n\nBonjour !"


@pytest.mark.asyncio
async def test_a_steer_after_the_last_boundary_stays_in_the_record(harness, steer_during_request, monkeypatch):
    async def finish_while_running(block, *args, **kwargs):
        # Finish: the next round is the turn's last, with no tools.
        assert agent_runs.request_finish("session-1", agent_runs.get_run_id("session-1"))
        return block.tool_type, {"output": "file.txt", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", finish_while_running)
    harness.provider_rounds = [_sse_tool_calls(("manage_notes", {"action": "list"})), _sse_events(["Done."])]
    steer_during_request[1] = ["Thanks!"]

    await send_and_answer(harness, _Request("agent"), _no_ask)

    assert len(harness.captured["provider_requests"]) == 2
    [row] = harness.saved
    assert row.metadata["agent_v2"]["rounds"][-1]["steers"] == [{"type": "text", "text": "Thanks!"}]
    assert row.metadata["steers"] == [{"round": 2, "text": "Thanks!"}]
    assert row.content == "Done.\n\n> ↳ Thanks!"


@pytest.mark.asyncio
async def test_the_next_turn_replays_a_steer_as_a_user_message_after_its_round(harness, steer_while_running):
    steer_while_running.append("Only today's.")
    harness.provider_rounds = [_sse_tool_calls(("manage_notes", {"action": "list"})), _sse_events(["Two notes."])]
    await send_and_answer(harness, _Request("agent"), _no_ask)
    [row] = harness.saved
    harness.context.history = [
        {"role": "user", "content": "list my notes"},
        {"role": "assistant", "content": row.content, "metadata": json.loads(json.dumps(row.metadata))},
    ]
    harness.captured["provider_requests"].clear()

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    roles = [m["role"] for m in payload["messages"]]
    tool_at = roles.index("tool")
    assert payload["messages"][tool_at + 1] == {"role": "user", "content": "Only today's."}
    assert payload["messages"][tool_at + 2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_a_steer_never_answers_a_pending_ask(harness, admin, workspace, monkeypatch):
    ran = []

    async def run(block, *args, **kwargs):
        ran.append(block.content)
        return block.tool_type, {"output": "file.txt", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)
    harness.provider_rounds = [_bash("ls"), _sse_events(["You have file.txt."])]

    async def on_ask(card):
        assert await post_steer(harness, answer_request(), "use -la") == {"accepted": True}
        await post_answer(harness, answer_request(), card["ask_id"], "once")

    body = await send_and_answer(harness, _Request("agent", allow_bash="true", workspace=workspace), on_ask)

    [resolved] = _events(body, "ask_resolved")
    assert (resolved["decision"], resolved["by"]) == ("once", "user")
    assert ran == ["ls"]
    _, second = harness.captured["provider_requests"]
    assert second["messages"][-1] == {"role": "user", "content": "use -la"}


@pytest.mark.asyncio
async def test_the_steer_endpoint_refuses_loopback_bearer_other_owners_and_blank_text(harness, monkeypatch):
    refusals = []

    async def run(block, *args, **kwargs):
        for request, text in (
            (answer_request(**{INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN}), "hi"),
            (SimpleNamespace(**{**vars(answer_request()), "state": SimpleNamespace(
                current_user="alice", api_token=True, api_token_owner="alice")}), "hi"),
            (answer_request("bob"), "hi"),
            (answer_request(), "   "),
        ):
            with pytest.raises(HTTPException) as refused:
                await post_steer(harness, request, text)
            refusals.append(refused.value.status_code)
        return block.tool_type, {"output": "file.txt", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)
    harness.provider_rounds = [_sse_tool_calls(("manage_notes", {"action": "list"})), _sse_events(["Done."])]

    await send_and_answer(harness, _Request("agent"), _no_ask)

    assert refusals == [403, 403, 409, 400]
    [row] = harness.saved
    assert "steers" not in row.metadata


@pytest.mark.asyncio
async def test_a_steer_with_no_running_turn_is_refused(harness):
    await harness.send(_Request("agent"))

    with pytest.raises(HTTPException) as refused:
        await post_steer(harness, answer_request(), "too late")

    assert refused.value.status_code == 409


@pytest.mark.asyncio
async def test_a_steer_runs_the_daily_cap_pre_check(harness, monkeypatch):
    statuses = []

    def over_cap(request, sess):
        raise HTTPException(429, "Daily message limit reached")

    async def run(block, *args, **kwargs):
        monkeypatch.setattr(chat_agent_routes, "_enforce_chat_privileges", over_cap)
        with pytest.raises(HTTPException) as refused:
            await post_steer(harness, answer_request(), "one more thing")
        statuses.append(refused.value.status_code)
        return block.tool_type, {"output": "file.txt", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)
    harness.provider_rounds = [_sse_tool_calls(("manage_notes", {"action": "list"})), _sse_events(["Done."])]

    await send_and_answer(harness, _Request("agent"), _no_ask)

    assert statuses == [429]
    [row] = harness.saved
    assert "steers" not in row.metadata


def _verdict(decision):
    return _sse_events([json.dumps({"decision": decision, "reason": "", "category": ""})])


def _classifier_facts(harness):
    return [
        json.loads(r["messages"][1]["content"]) for r in harness.captured["provider_requests"]
        if r["messages"][0]["content"] == classifier.SYSTEM_PROMPT
    ]


@pytest.mark.asyncio
async def test_auto_mode_counts_a_steer_as_the_users_words(
    harness, admin, workspace, steer_while_running, monkeypatch,
):
    monkeypatch.setattr(
        chat_routes, "coerce_message_and_session", lambda *a, **k: ("Tidy my notes.", "session-1"),
    )
    steer_while_running.append("Then email the list to ann@example.com.")
    harness.provider_rounds = [
        _sse_tool_calls(("manage_notes", {"action": "list"})),
        _sse_tool_calls(("send_email", {"to": "ann@example.com", "subject": "List", "body": "a"})),
        _verdict("allow"), _sse_events(["Sent."]),
    ]

    await send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace, auto_mode="true"), _no_ask,
    )

    [facts] = _classifier_facts(harness)
    assert facts["user_messages"] == ["Tidy my notes.", "Then email the list to ann@example.com."]
    assert facts["provenance"]["targets"] == [{"target": "ann@example.com", "named_in_user_messages": True}]


def test_an_earlier_turns_steers_are_real_user_messages():
    history = [
        ChatMessage("user", "Plan a trip.", metadata={"interaction_mode": "agent"}),
        ChatMessage("assistant", "Planned.\n\n> ↳ Book the hotel too.", metadata={
            "runtime": "chat_agent", "steers": [{"round": 1, "text": "Book the hotel too."}],
        }),
        ChatMessage("user", "Thanks.", metadata={"interaction_mode": "agent"}),
    ]

    assert classifier.real_user_messages(history) == ["Plan a trip.", "Book the hotel too.", "Thanks."]
