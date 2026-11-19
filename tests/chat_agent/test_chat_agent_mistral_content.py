"""Mistral reasoning models stream ``delta.content`` as chunk lists.

Frames are handwritten from docs.mistral.ai/inference/conversations/reasoning
(streaming: a list holding a ThinkChunk, then one list holding the closing
ThinkChunk and the first TextChunk, then plain strings) and the SDK's
ThinkChunk/TextChunk models; none is a recorded stream.
"""

import pytest

from tests.chat_agent.test_chat_agent_live_thinking import _live
from tests.chat_agent.test_chat_agent_reasoning_replay import (  # noqa: F401
    SEARCH,
    _assistant_with_tool_calls,
    _stream,
    _tool_turn,
    db,
    harness,
)
from tests.chat_agent.test_chat_agent_route import _as_history_row, _Request, _sse_events

MISTRAL = "https://api.mistral.ai/v1"


def _think(text, **extra):
    return {"type": "thinking", "thinking": [{"type": "text", "text": text}], **extra}


def _text(text):
    return {"type": "text", "text": text}


THINKING_THEN_ANSWER = [
    {"content": [_think("Plan ")]},
    {"content": [_think("it")]},
    {"content": [_think(".", closed=True), _text("Hel")]},
    {"content": "lo!"},
]


@pytest.mark.asyncio
async def test_chunk_list_content_streams_thinking_then_answer(harness):
    harness.session.endpoint_url, harness.session.model = MISTRAL, "magistral-medium-latest"
    harness.provider_rounds = [_stream(*THINKING_THEN_ANSWER)]

    _, body = await harness.send(_Request("agent"))

    assert _live(body) == (["Plan ", "it", "."], ["Hel", "lo!"])
    [row] = harness.saved
    assert row.content == "Hello!"
    assert row.metadata["thinking"] == "Plan it."


@pytest.mark.asyncio
async def test_assistant_message_goes_back_with_its_thinking_chunk(harness):
    await _tool_turn(harness, MISTRAL, "magistral-medium-latest", *THINKING_THEN_ANSWER)

    assistant = _assistant_with_tool_calls(harness.captured["provider_requests"][1])
    assert assistant["content"] == [_think("Plan it."), _text("Hello!")]


@pytest.mark.asyncio
async def test_saved_turn_replays_its_thinking_chunk_on_the_next_turn(harness):
    await _tool_turn(harness, MISTRAL, "magistral-medium-latest", *THINKING_THEN_ANSWER)
    harness.context.history = [harness.context.current, _as_history_row(harness.saved[0])]
    harness.provider_rounds = [_sse_events(["Again."])]

    await harness.send(_Request("agent"))

    assistant = _assistant_with_tool_calls(harness.captured["provider_requests"][2])
    assert assistant["content"] == [_think("Plan it."), _text("Hello!")]


@pytest.mark.asyncio
async def test_plain_string_content_stays_a_string(harness):
    await _tool_turn(harness, MISTRAL, "mistral-small-latest", {"content": "Look"}, {"content": "ing."})

    assistant = _assistant_with_tool_calls(harness.captured["provider_requests"][1])
    assert assistant["content"] == "Looking."
