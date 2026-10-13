"""Temperature, max_tokens and the output limit on the new runtime (ticket #93, spec #89).

Route tests drive POST /api/chat_stream with scripted provider streams and
assert the outgoing payloads, the SSE outcome and the saved row.
"""

import json
from types import SimpleNamespace

import pytest

import routes.history.history_routes as history_routes
from src.chat_agent import loop
from src.chat_agent.providers import Endpoint, xai_responses
from src.generation_budget import context_safety_margin
from tests.chat_agent.test_chat_agent_route import _data_events, _Request, _sse_tool_calls, _switch_on
from tests.chat_agent.test_chat_agent_route import harness  # noqa: F401

MODEL = "x-ai/grok-4.1-fast"
THINKING_LIMIT = (
    f"Stopped: {MODEL} reached the output limit while thinking, before any answer. "
    "Raise Max tokens or lower the thinking level."
)
TOOL_CALL_LIMIT = (
    f"Stopped: {MODEL} reached the output limit in the middle of a tool call; the call was not run."
)
OUTPUT_LIMIT_NOTICE = "stopped at output limit"


def _stream(deltas, finish_reason=None):
    chunks = [{"choices": [{"delta": delta}]} for delta in deltas]
    if finish_reason:
        chunks.append({"choices": [{"delta": {}, "finish_reason": finish_reason}]})
    return b"".join(("data: " + json.dumps(chunk) + "\n\n").encode() for chunk in chunks) + b"data: [DONE]\n\n"


def _cut_tool_call():
    """A round whose tool-call arguments stop half way at the output limit."""
    return _stream([
        {"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                         "function": {"name": "no_such_tool", "arguments": '{"path": "/tm'}}]},
    ], finish_reason="length")


def _error(body):
    lines = body.splitlines()
    index = lines.index("event: error")
    return json.loads(lines[index + 1][len("data: "):])


def _blocks(row):
    return [block for turn_round in row.metadata["agent_v2"]["rounds"] for block in turn_round["blocks"]]


def _window(monkeypatch, window, measured):
    calls = []

    def known(url, model):
        calls.append((url, model))
        return window, measured

    monkeypatch.setattr(loop, "get_context_length_known", known)
    return calls


# --- temperature ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_unset_temperature_and_max_tokens_send_nothing(harness):
    _switch_on()
    harness.provider_rounds = [_sse_tool_calls(("no_such_tool", {})), _stream([{"content": "Done."}])]

    await harness.send(_Request("agent"))

    requests = harness.captured["provider_requests"]
    assert len(requests) == 2
    for payload in requests:
        assert "temperature" not in payload
        assert "max_tokens" not in payload and "max_completion_tokens" not in payload


@pytest.mark.asyncio
async def test_session_temperature_is_sent_on_every_round(harness):
    _switch_on()
    harness.session.temperature_override = 0.6
    harness.provider_rounds = [_sse_tool_calls(("no_such_tool", {})), _stream([{"content": "Done."}])]

    await harness.send(_Request("agent"))

    assert [p["temperature"] for p in harness.captured["provider_requests"]] == [0.6, 0.6]


def test_xai_payload_omits_unset_temperature_and_max_output_tokens():
    payload = xai_responses.build_payload(
        Endpoint("https://api.x.ai/v1", "grok-4.1-fast", None), [], cache_key="s", temperature=None, max_tokens=0,
    )
    assert "temperature" not in payload
    assert "max_output_tokens" not in payload


# --- max_tokens ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_max_tokens_is_clamped_to_the_room_left_in_a_measured_window(harness, monkeypatch):
    _switch_on()
    window = 32_768
    _window(monkeypatch, window, True)
    harness.session.max_tokens_override = 100_000
    harness.provider_rounds = [_sse_tool_calls(("no_such_tool", {})), _stream([{"content": "Done."}])]

    await harness.send(_Request("agent"))

    sent = [p["max_tokens"] for p in harness.captured["provider_requests"]]
    assert len(sent) == 2
    # Below the window less its safety margin, and the prompt grows by a round.
    assert all(0 < value < window - context_safety_margin(window) for value in sent)
    assert sent[1] < sent[0]


@pytest.mark.asyncio
async def test_set_max_tokens_that_fits_is_sent_as_is(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch, 32_768, True)
    harness.session.max_tokens_override = 64
    harness.provider_rounds = [_stream([{"content": "Done."}])]

    await harness.send(_Request("agent"))

    assert harness.captured["provider_requests"][0]["max_tokens"] == 64


@pytest.mark.asyncio
async def test_set_max_tokens_is_not_clamped_on_an_unmeasured_window(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch, 128_000, False)
    harness.session.max_tokens_override = 100_000
    harness.provider_rounds = [_stream([{"content": "Done."}])]

    await harness.send(_Request("agent"))

    assert harness.captured["provider_requests"][0]["max_tokens"] == 100_000


@pytest.mark.asyncio
async def test_unset_max_tokens_stays_unset_on_a_measured_window(harness, monkeypatch):
    # The window is measured every turn since #94 (prompt budget); an unset
    # max_tokens still sends nothing.
    _switch_on()
    _window(monkeypatch, 16_384, True)
    harness.provider_rounds = [_stream([{"content": "Done."}])]

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert "max_tokens" not in payload and "max_completion_tokens" not in payload


# --- finish_reason=length --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("deltas", [
    [{"reasoning_content": "Thinking about it"}],
    [{"reasoning_content": "Thinking about it"}, {"content": "  \n"}],
    [{"content": "<think>Thinking about it"}],
])
async def test_limit_while_thinking_is_a_saved_error_over_the_empty_answer_error(harness, deltas):
    _switch_on()
    harness.provider_rounds = [_stream(deltas, finish_reason="length")]

    _, body = await harness.send(_Request("agent"))

    assert _error(body) == {"error": THINKING_LIMIT, "status": 502}
    assert not [e for e in _data_events(body) if e.get("type") == "final_response"]
    [row] = harness.saved
    assert row.metadata["error"] == THINKING_LIMIT
    assert row.metadata["stop_reason"] == "error"
    assert row.metadata["thinking"] == "Thinking about it"
    assert row.content == ""
    assert [b["type"] for b in _blocks(row)] == ["reasoning"]


@pytest.mark.asyncio
async def test_limit_mid_answer_keeps_the_text_with_a_saved_notice(harness):
    _switch_on()
    harness.provider_rounds = [_stream(
        [{"reasoning_content": "Plan."}, {"content": "Cats are small"}], finish_reason="length",
    )]

    _, body = await harness.send(_Request("agent"))

    assert "event: error" not in body
    [final] = [e for e in _data_events(body) if e.get("type") == "final_response"]
    assert final["content"] == "Cats are small"
    [row] = harness.saved
    assert row.content == "Cats are small"
    assert row.metadata["notice"] == OUTPUT_LIMIT_NOTICE
    assert "error" not in row.metadata


@pytest.mark.asyncio
async def test_complete_answer_has_no_notice(harness):
    _switch_on()
    harness.provider_rounds = [_stream([{"content": "Cats are small."}], finish_reason="stop")]

    await harness.send(_Request("agent"))

    [row] = harness.saved
    assert "notice" not in row.metadata


@pytest.mark.asyncio
async def test_limit_mid_tool_call_is_an_error_and_the_call_is_not_run(harness):
    _switch_on()
    harness.provider_rounds = [
        _sse_tool_calls(("no_such_tool", {}), text="Let me check."),
        _cut_tool_call(),
    ]

    _, body = await harness.send(_Request("agent"))

    assert _error(body) == {"error": TOOL_CALL_LIMIT, "status": 502}
    assert len(harness.captured["provider_requests"]) == 2
    # Only the first round's call opened a card.
    assert [e["call_id"] for e in _data_events(body) if e.get("type") == "tool_start"] == ["call_0"]
    [row] = harness.saved
    assert row.metadata["error"] == TOOL_CALL_LIMIT
    first, cut = row.metadata["agent_v2"]["rounds"]
    assert [b["type"] for b in first["blocks"]] == ["text", "tool_call"]
    assert len(first["results"]) == 1
    # The cut call is not kept: a call without a result would not replay.
    assert cut["blocks"] == []


# --- generation-settings route ----------------------------------------------------


class _Row:
    thinking_mode = "off"
    temperature_override = 0.6
    max_tokens_override = None


class _Db:
    def __init__(self, row):
        self.row = row

    def query(self, *args):
        return self

    def filter(self, *args):
        return self

    def first(self):
        return self.row

    def commit(self):
        return None

    def close(self):
        return None


@pytest.mark.asyncio
async def test_generation_settings_route_accepts_a_null_temperature(monkeypatch):
    row = _Row()
    session = SimpleNamespace(model="m", thinking_mode="off", temperature_override=0.6, max_tokens_override=None)
    monkeypatch.setattr(history_routes, "_verify_session_owner", lambda *a, **k: None)
    monkeypatch.setattr(history_routes, "SessionLocal", lambda: _Db(row))
    router = history_routes.setup_history_routes(SimpleNamespace(get_session=lambda session_id: session))
    [endpoint] = [r.endpoint for r in router.routes if r.path == "/api/session/{session_id}/generation-settings"]

    async def body():
        return {"temperature_override": None}

    result = await endpoint(SimpleNamespace(json=body), "session-1")

    assert result["temperature_override"] is None
    assert row.temperature_override is None and session.temperature_override is None
