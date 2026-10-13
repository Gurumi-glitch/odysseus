"""Degenerate output and empty answers on the new runtime (ticket #92, spec #89).

Route tests drive POST /api/chat_stream with scripted provider streams and
assert the SSE error, how much of the stream was read, and the saved row.
"""

import asyncio
import json

import httpx
import pytest

from src.chat_agent.providers import chat_completions
from src.llm_core import _DegenerateStreamGuard
from tests.chat_agent.test_chat_agent_route import _data_events, _Request, _sse_tool_calls, _switch_on
from tests.chat_agent.test_chat_agent_route import harness  # noqa: F401

MODEL = "x-ai/grok-4.1-fast"
DEGENERATE = (
    f"Stopped: {MODEL} produced degenerate output (repeating tokens). "
    "Try lower temperature or reload the model."
)
NO_ANSWER = f"{MODEL} returned no answer"


def _chunk(delta):
    return ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n").encode()


def _stream(deltas):
    return b"".join(_chunk(delta) for delta in deltas) + b"data: [DONE]\n\n"


class _EndlessProvider:
    """Serves ``rounds`` (whole bodies) first; then streams ``deltas`` and
    repeats ``loop`` forever, so only the guard ends that request."""

    def __init__(self, monkeypatch, loop, deltas=(), rounds=()):
        self.chunks_read = 0
        self.requests = 0
        rounds = list(rounds)

        async def body():
            for delta in deltas:
                yield _chunk(delta)
            while True:
                self.chunks_read += 1
                yield _chunk(loop)
                await asyncio.sleep(0)

        def handle(request):
            if request.method == "GET":  # the OpenRouter capability probe (#90)
                return httpx.Response(404)
            if rounds:
                return httpx.Response(200, content=rounds.pop(0), headers={"content-type": "text/event-stream"})
            self.requests += 1
            return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)


def _error(body):
    lines = body.splitlines()
    index = lines.index("event: error")
    return json.loads(lines[index + 1][len("data: "):])


def _blocks(row):
    return [block for turn_round in row.metadata["agent_v2"]["rounds"] for block in turn_round["blocks"]]


def test_legacy_guard_check_is_none_unless_a_rule_fired():
    # The loop treats a truthy check() as tripped and ignores its text.
    guard = _DegenerateStreamGuard("m")
    assert all(guard.check(word + " ") is None for word in "the cat sat on a mat by the door".split())
    guard = _DegenerateStreamGuard("m")
    assert guard.check("//// " * 400) is None
    results = [guard.check("Var ") for _ in range(40)]
    assert any(results)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel, loop", [
    ("answer", {"content": "//////////"}),
    ("answer", {"content": "/ / / / / "}),
    ("thinking", {"reasoning_content": "/ / / / / "}),
    ("answer", {"content": "Var Var "}),
    ("thinking", {"reasoning_content": "Summer Summer "}),
])
async def test_degenerate_stream_is_cut_off_with_a_saved_error(harness, monkeypatch, channel, loop):
    _switch_on()
    provider = _EndlessProvider(monkeypatch, loop)

    _, body = await asyncio.wait_for(harness.send(_Request("agent")), 10)

    assert _error(body) == {"error": DEGENERATE, "status": 502}
    assert body.rstrip().endswith("data: [DONE]")
    assert provider.requests == 1
    # Well under the 16k-token runs seen without the guard.
    assert provider.chunks_read < 400
    [row] = harness.saved
    assert row.metadata["error"] == DEGENERATE
    assert row.metadata["stop_reason"] == "error"
    assert row.content == ""
    assert "thinking" not in row.metadata
    assert _blocks(row) == []


@pytest.mark.asyncio
async def test_degenerate_round_is_dropped_and_earlier_rounds_kept(harness, monkeypatch):
    _switch_on()
    endless = _EndlessProvider(
        monkeypatch, {"content": "//////////"}, deltas=[{"reasoning_content": "Hmm."}],
        rounds=[_sse_tool_calls(("no_such_tool", {}), text="Let me check.")],
    )

    _, body = await asyncio.wait_for(harness.send(_Request("agent")), 10)

    assert _error(body)["error"] == DEGENERATE
    assert endless.requests == 1
    [row] = harness.saved
    [kept] = row.metadata["agent_v2"]["rounds"]
    assert [b["type"] for b in kept["blocks"]] == ["text", "tool_call"]
    assert [r["call_id"] for r in kept["results"]] == ["call_0"]
    assert row.content == "Let me check."
    assert "Hmm." not in json.dumps(row.metadata)
    assert row.metadata["error"] == DEGENERATE


@pytest.mark.asyncio
async def test_punctuation_short_of_the_run_limit_is_an_answer(harness):
    _switch_on()
    # 999 non-alphanumeric characters, then a letter resets the run.
    harness.provider_rounds = [_stream([{"content": "-" * 500}, {"content": " | " * 499}, {"content": "ok"}])]

    _, body = await harness.send(_Request("agent"))

    assert "event: error" not in body
    [row] = harness.saved
    assert row.content.endswith("ok")
    assert "error" not in row.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("deltas", [
    [{"reasoning_content": "Thinking about it."}],
    [{"reasoning_content": "Thinking about it."}, {"content": " \n\n "}],
    [{"content": "<think>Thinking about it.</think>"}],
])
async def test_round_with_no_answer_ends_with_a_saved_error(harness, deltas):
    _switch_on()
    harness.provider_rounds = [_stream(deltas)]

    _, body = await harness.send(_Request("agent"))

    assert _error(body) == {"error": NO_ANSWER, "status": 502}
    assert not [e for e in _data_events(body) if e.get("type") == "final_response"]
    [row] = harness.saved
    assert row.metadata["error"] == NO_ANSWER
    assert row.metadata["stop_reason"] == "error"
    assert row.metadata["thinking"] == "Thinking about it."
    assert row.content == ""
    assert [b["type"] for b in _blocks(row)] == ["reasoning"]


@pytest.mark.asyncio
async def test_empty_stream_ends_with_a_saved_error(harness):
    _switch_on()
    harness.provider_rounds = [_stream([])]

    _, body = await harness.send(_Request("agent"))

    assert _error(body)["error"] == NO_ANSWER
    [row] = harness.saved
    assert row.metadata["error"] == NO_ANSWER
