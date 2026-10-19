"""Live thinking on the new runtime (ticket #91, spec #89 "Live thinking").

Adapter tests stream recorded Chat Completions deltas through the
adapter; route tests drive POST /api/chat_stream and assert the SSE events
and the saved turn record.
"""

import asyncio
import json

import httpx
import pytest

from src import agent_runs
from src.chat_agent import transcript
from src.chat_agent.providers import Endpoint, ReasoningDelta, RoundEnd, TextDelta, chat_completions
from tests.chat_agent.test_chat_agent_fallback import BACKUP_MODEL, PRIMARY_MODEL, _script
from tests.chat_agent.test_chat_agent_fallback import harness as fallback_harness  # noqa: F401
from tests.chat_agent.test_chat_agent_route import _data_events, _Request, _switch_on
from tests.chat_agent.test_chat_agent_route import harness  # noqa: F401

LLAMA = Endpoint("http://127.0.0.1:1235/v1", "Ternary-Bonsai-2-27B", None)


def _stream(deltas):
    lines = ["data: " + json.dumps({"choices": [{"delta": delta}]}) for delta in deltas]
    return ("\n\n".join(lines + ["data: [DONE]"]) + "\n\n").encode()


class _Provider:
    def __init__(self, monkeypatch, body):
        self.payloads = []
        self._body = body
        client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)

    def _handle(self, request):
        self.payloads.append(json.loads(request.content))
        return httpx.Response(200, content=self._body, headers={"content-type": "text/event-stream"})


async def _run(conversation=None, endpoint=LLAMA):
    conversation = conversation or [{"role": "user", "content": "hello"}]
    return [
        event
        async for event in chat_completions.stream_round(endpoint, conversation, temperature=0.2, max_tokens=0)
    ]


def _split(events):
    thinking = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
    answer = "".join(e.text for e in events if isinstance(e, TextDelta))
    [end] = [e for e in events if isinstance(e, RoundEnd)]
    return thinking, answer, end.blocks


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
async def test_reasoning_fields_stream_as_reasoning_deltas(monkeypatch, field):
    _Provider(monkeypatch, _stream([{field: "Let me "}, {field: "think."}, {"content": "Hi!"}]))

    events = await _run()

    assert [e.text for e in events if isinstance(e, ReasoningDelta)] == ["Let me ", "think."]
    thinking, answer, blocks = _split(events)
    assert (thinking, answer) == ("Let me think.", "Hi!")
    [reasoning] = [b for b in blocks if b["type"] == "reasoning"]
    assert reasoning["text"] == "Let me think."


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks, thinking, answer", [
    # Tags split across chunks, leading whitespace before the open tag.
    (["\n <th", "ink>Plan the ", "greeting.</th", "ink>Hello!"], "Plan the greeting.", "Hello!"),
    (["<thinking>", "Plan.", "</thinking>", "Hi"], "Plan.", "Hi"),
    # Cut before the close tag: all of it is thinking, none of it answer.
    (["<think>Still plan", "ning"], "Still planning", ""),
    # Only a round's opening content is inline thinking.
    (["Use <think> tags", " like this"], "", "Use <think> tags like this"),
    (["<thinker>"], "", "<thinker>"),
])
async def test_inline_thinking_opening_a_round_is_thinking_not_answer(monkeypatch, chunks, thinking, answer):
    _Provider(monkeypatch, _stream([{"content": chunk} for chunk in chunks]))

    streamed_thinking, streamed_answer, blocks = _split(await _run())

    assert (streamed_thinking, streamed_answer) == (thinking, answer)
    assert "".join(b["text"] for b in blocks if b["type"] == "text") == answer
    assert [b.get("text") for b in blocks if b["type"] == "reasoning"] == ([thinking] if thinking else [])


@pytest.mark.asyncio
async def test_inline_thinking_is_never_replayed(monkeypatch):
    provider = _Provider(monkeypatch, _stream([{"content": "<think>secret plan</think>Done."}]))
    question = [{"role": "user", "content": "hello"}]

    _, _, blocks = _split(await _run(question))
    await _run([*question, {"role": "assistant", "blocks": blocks}, {"role": "user", "content": "again"}])

    assert provider.payloads[1]["messages"][1] == {"role": "assistant", "content": "Done."}


def _live(body):
    """(thinking, answer) deltas as the browser receives them."""
    events = [e for e in _data_events(body) if "delta" in e]
    return [e["delta"] for e in events if e.get("thinking")], [e["delta"] for e in events if not e.get("thinking")]


@pytest.mark.asyncio
@pytest.mark.parametrize("deltas", [
    [{"reasoning_content": "Plan the "}, {"reasoning_content": "greeting."}, {"content": "Hello!"}],
    [{"reasoning": "Plan the "}, {"reasoning": "greeting."}, {"content": "Hello!"}],
    [{"content": "<think>Plan the "}, {"content": "greeting.</think>"}, {"content": "Hello!"}],
])
async def test_thinking_streams_live_and_is_kept_for_reload(harness, deltas):
    _switch_on()
    # Legacy hides reasoning while thinking is off; the new runtime shows what the server sends.
    harness.session.thinking_mode = "off"
    harness.provider_rounds = [_stream(deltas)]

    _, body = await harness.send(_Request("agent"))

    assert _live(body) == (["Plan the ", "greeting."], ["Hello!"])
    [row] = harness.saved
    assert row.content == "Hello!"
    assert row.metadata["thinking"] == "Plan the greeting."


def test_rows_saved_before_display_text_read_their_reasoning_content():
    record = transcript.TurnRecord(provider="openai", model="m")
    record.rounds = [
        {"blocks": [{"type": "reasoning", "provider": "openai", "model": "m", "reasoning_content": "old"}]},
        {"blocks": [{"type": "reasoning", "provider": "openai", "model": "m",
                     "reasoning_content": "new", "text": "new"}]},
    ]

    assert record.metadata()["thinking"] == "old\n\nnew"


@pytest.mark.asyncio
async def test_failure_after_thinking_streamed_is_not_failed_over(fallback_harness, monkeypatch):
    _switch_on()
    thought_then_failed = (
        "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "Thinking..."}}]}) + "\n\n"
        + "data: " + json.dumps({"error": {"message": "upstream overloaded", "code": 503}}) + "\n\n"
    ).encode()
    _script(monkeypatch, fallback_harness, {PRIMARY_MODEL: [thought_then_failed], BACKUP_MODEL: [["unused"]]})

    _, body = await fallback_harness.send(_Request("agent"))

    assert fallback_harness.captured["provider_calls"] == [PRIMARY_MODEL]
    assert "event: error" in body and "upstream overloaded" in body
    assert not [e for e in _data_events(body) if e.get("type") == "fallback"]


@pytest.mark.asyncio
async def test_stop_keeps_the_thinking_streamed_so_far(harness, monkeypatch):
    _switch_on()
    thought = asyncio.Event()

    async def thinking_then_silence():
        yield ("data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "Half a thought"}}]}) + "\n\n").encode()
        thought.set()
        await asyncio.Event().wait()

    def provider(request):
        if request.method == "GET":  # the OpenRouter capability probe (#90)
            return httpx.Response(404)
        return httpx.Response(200, content=thinking_then_silence(), headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)
    send = asyncio.create_task(harness.send(_Request("agent")))
    await asyncio.wait_for(thought.wait(), 5)
    # The loop forwards the delta before the stop lands.
    await asyncio.sleep(0.05)
    assert agent_runs.stop("session-1", agent_runs.get_run_id("session-1"))
    await asyncio.wait_for(send, 5)

    [row] = harness.saved
    assert row.metadata["stop_reason"] == "interrupted"
    assert row.metadata["thinking"] == "Half a thought"
