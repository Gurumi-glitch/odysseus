"""How a new-runtime turn stops (ticket #54), driven through the chat route.

Covers the repeat guard (reminders at 3/5/8, stop at 12), the stop reason the
caps write, and Finish (one more round with no tools).
"""

import pytest

import src.agent_runs as agent_runs
import src.settings as settings
import src.tool_execution as tool_execution
from tests.chat_agent.test_chat_agent_route import (
    _Request,
    _data_events,
    _sse_events,
    _sse_tool_calls,
)
from tests.chat_agent.test_chat_agent_asks_route import answering, send_and_answer
from tests.chat_agent.test_chat_agent_tools_route import admin, db, harness  # noqa: F401  (fixtures)

LIST_NOTES = ("manage_notes", {"action": "list"})


def _tool_results(payload):
    return [m["content"] for m in payload["messages"] if m["role"] == "tool"]


def _last_result(payload):
    return _tool_results(payload)[-1]


def _outputs(body):
    return [e for e in _data_events(body) if e.get("type") == "tool_output"]


@pytest.fixture
def executor(monkeypatch):
    """Replace the legacy executor's body; the test decides what each call does."""
    calls = []
    script = {"on_call": None}

    async def run(block, *args, **kwargs):
        calls.append(block.content)
        if script["on_call"] is not None:
            await script["on_call"](len(calls))
        return block.tool_type, {"output": f"result {len(calls)}", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)
    return calls, script


@pytest.mark.asyncio
async def test_repeated_call_gets_reminders_at_3_5_8_and_stops_at_12(harness, executor):
    calls, _ = executor
    harness.provider_rounds = [_sse_tool_calls(LIST_NOTES) for _ in range(13)]

    body = await send_and_answer(harness, _Request("agent"), answering(harness, ("deny",)))

    requests = harness.captured["provider_requests"]
    assert len(requests) == 12
    # Request n+1 carries the result of call n.
    results = {n: _last_result(requests[n]) for n in range(1, 12)}
    assert [n for n, text in results.items() if "<system-reminder>" in text] == list(range(3, 12))
    assert "Please carefully analyze the previous result" in results[3]
    assert "repeated_times: 5" in results[5]
    assert "Do not call any tool in your next response" in results[8]
    assert "Do not call any tool in your next response" in results[11]
    # At the 12th call the user is asked; Stop means it never runs.
    [ask] = [e["data"] for e in _data_events(body) if e.get("type") == "ask_user"]
    assert (ask["ask_kind"], ask["kind"]) == ("repeat_guard", "tool_approval")
    assert [o["value"] for o in ask["options"]] == ["once", "deny"]
    assert len(calls) == 11
    assert _outputs(body)[-1]["exit_code"] == 126
    [stop] = [e for e in _data_events(body) if e.get("type") == "loop_breaker_triggered"]
    assert stop["reason"] == "repeat_guard"
    assert stop["message"]
    [row] = harness.saved
    assert row.metadata["stop_reason"] == "repeat_guard"
    # The reminder reaches the model, not the tool card.
    assert "<system-reminder>" not in row.metadata["tool_events"][2]["output"]
    assert "<system-reminder>" in row.metadata["agent_v2"]["rounds"][2]["results"][0]["content"]


@pytest.mark.asyncio
async def test_continue_at_the_repeat_guard_ask_runs_the_call_and_the_turn_goes_on(harness, executor):
    calls, _ = executor
    harness.provider_rounds = [_sse_tool_calls(LIST_NOTES) for _ in range(12)] + [_sse_events(["Done."])]

    body = await send_and_answer(harness, _Request("agent"), answering(harness, ("once",)))

    assert len(calls) == 12
    assert not [e for e in _data_events(body) if e.get("type") == "loop_breaker_triggered"]
    [row] = harness.saved
    assert row.content.endswith("Done.")
    assert "stop_reason" not in row.metadata
    assert row.metadata["tool_events"][11]["ask_user"]["resolved"] == "once"


@pytest.mark.asyncio
async def test_a_different_call_resets_the_streak(harness, executor):
    other = ("manage_notes", {"action": "list", "limit": 5})
    harness.provider_rounds = [
        _sse_tool_calls(LIST_NOTES), _sse_tool_calls(LIST_NOTES), _sse_tool_calls(other),
        _sse_tool_calls(LIST_NOTES), _sse_tool_calls(LIST_NOTES), _sse_events(["Done."]),
    ]

    await harness.send(_Request("agent"))

    final = harness.captured["provider_requests"][-1]
    assert not [text for text in _tool_results(final) if "<system-reminder>" in text]
    [row] = harness.saved
    assert "stop_reason" not in row.metadata


@pytest.mark.asyncio
async def test_argument_key_order_does_not_hide_a_repeat(harness, executor):
    harness.provider_rounds = [
        _sse_tool_calls(("manage_notes", {"action": "list", "limit": 5})),
        _sse_tool_calls(("manage_notes", {"limit": 5, "action": "list"})),
        _sse_tool_calls(("manage_notes", {"action": "list", "limit": 5})),
        _sse_events(["Done."]),
    ]

    await harness.send(_Request("agent"))

    assert "<system-reminder>" in _last_result(harness.captured["provider_requests"][-1])


def _settings(monkeypatch, values):
    monkeypatch.setattr(settings, "get_setting", lambda key, default=None: values.get(key, default))


@pytest.mark.asyncio
async def test_round_cap_records_max_rounds(harness, monkeypatch, executor):
    _settings(monkeypatch, {"agent_max_rounds": 2})
    harness.provider_rounds = [_sse_tool_calls(LIST_NOTES) for _ in range(3)]

    _, body = await harness.send(_Request("agent"))

    assert {"type": "rounds_exhausted", "rounds": 2} in _data_events(body)
    [row] = harness.saved
    assert row.metadata["stop_reason"] == "max_rounds"


@pytest.mark.asyncio
async def test_tool_call_cap_records_budget(harness, monkeypatch, executor):
    _settings(monkeypatch, {"agent_max_tool_calls": 1})
    harness.provider_rounds = [_sse_tool_calls(LIST_NOTES, ("manage_notes", {"action": "list", "limit": 5}))]

    _, body = await harness.send(_Request("agent"))

    [budget] = [e for e in _data_events(body) if e.get("type") == "budget_exceeded"]
    assert (budget["kind"], budget["used"], budget["limit"]) == ("tool_calls", 1, 1)
    [row] = harness.saved
    assert row.metadata["stop_reason"] == "budget"


@pytest.mark.asyncio
async def test_finish_lets_the_current_tool_finish_then_answers_without_tools(harness, executor):
    calls, script = executor

    async def finish_on_first_call(n):
        if n == 1:
            assert agent_runs.request_finish("session-1", agent_runs.get_run_id("session-1"))

    script["on_call"] = finish_on_first_call
    harness.provider_rounds = [
        _sse_tool_calls(LIST_NOTES, ("manage_notes", {"action": "list", "limit": 5})),
        _sse_events(["Here is what I found."]),
        _sse_tool_calls(LIST_NOTES),
    ]

    _, body = await harness.send(_Request("agent"))

    assert len(calls) == 1
    assert [o["exit_code"] for o in _outputs(body)] == [0, 126]
    first, last = harness.captured["provider_requests"]
    assert "tools" in first
    assert "tools" not in last and "tool_choice" not in last
    assert len(_tool_results(last)) == 2
    [row] = harness.saved
    assert row.content == "Here is what I found."
    assert row.metadata["stop_reason"] == "interrupted"

