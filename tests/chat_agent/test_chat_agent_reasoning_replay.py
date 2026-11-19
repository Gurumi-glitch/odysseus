"""Reasoning goes back in the field it arrived in (ticket #142, spec #146).

Driven through the chat route with scripted provider streams; the second
request the transport captured is what the provider would see.
"""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.tool_execution as tool_execution
import src.tool_security as tool_security
from core import database
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _as_history_row,
    _sse_events,
    _switch_on,
)

VLLM = "http://localhost:8000/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
DEEPSEEK = "https://api.deepseek.com/v1"
DETAILS = [{"type": "reasoning.text", "text": "plan", "format": "anthropic-claude-v1", "index": 0}]
SEARCH = {"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                          "function": {"name": "web_search", "arguments": '{"query": "cats"}'}}]}


def _stream(*deltas):
    chunks = [{"choices": [{"delta": delta}]} for delta in deltas]
    return b"".join(("data: " + json.dumps(chunk) + "\n\n").encode() for chunk in chunks) + b"data: [DONE]\n\n"


def _assistant_with_tool_calls(request):
    return next(m for m in request["messages"] if m["role"] == "assistant" and m.get("tool_calls"))


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))


@pytest.fixture
def harness(monkeypatch, tmp_path, db):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)

    async def worker(block, *args, **kwargs):
        return "web_search: ran", {"output": "ran", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", worker)
    harness = _Harness(monkeypatch, tmp_path)
    _switch_on()
    return harness


async def _tool_turn(harness, url, model, *first_round_deltas):
    harness.session.endpoint_url, harness.session.model = url, model
    harness.provider_rounds = [_stream(*first_round_deltas, SEARCH), _sse_events(["Done."])]
    _, body = await harness.send(_Request("agent", allow_web_search="true"))
    return body


@pytest.mark.asyncio
async def test_reasoning_field_goes_back_on_the_next_round_unchanged(harness):
    await _tool_turn(
        harness, VLLM, "glm-5.2", {"reasoning": "Need to "}, {"reasoning": "search."}, {"content": "Looking."},
    )

    first, second = harness.captured["provider_requests"]
    assert "reasoning" not in first["messages"][-1]
    assistant = _assistant_with_tool_calls(second)
    assert assistant["reasoning"] == "Need to search."
    assert "reasoning_content" not in assistant and "reasoning_details" not in assistant
    [row] = harness.saved
    assert row.metadata["thinking"] == "Need to search."


@pytest.mark.asyncio
async def test_reasoning_content_replays_as_today_without_a_reasoning_field(harness):
    await _tool_turn(harness, DEEPSEEK, "deepseek-v4", {"reasoning_content": "Think."}, {"content": "Looking."})

    assistant = _assistant_with_tool_calls(harness.captured["provider_requests"][1])
    assert assistant["reasoning_content"] == "Think."
    assert "reasoning" not in assistant


@pytest.mark.asyncio
async def test_openrouter_reasoning_next_to_details_is_not_doubled(harness):
    await _tool_turn(
        harness, OPENROUTER, "x-ai/grok-4.1-fast",
        {"reasoning": "plan", "reasoning_details": DETAILS}, {"content": "Looking."},
    )

    assistant = _assistant_with_tool_calls(harness.captured["provider_requests"][1])
    assert assistant["reasoning_details"] == DETAILS
    assert "reasoning" not in assistant


@pytest.mark.asyncio
async def test_another_models_reasoning_field_is_not_replayed(harness):
    await _tool_turn(harness, VLLM, "glm-5.2", {"reasoning": "Need to search."}, {"content": "Looking."})
    harness.context.history = [harness.context.current, _as_history_row(harness.saved[0])]
    harness.session.model = "qwen3-80b"
    harness.provider_rounds = [_sse_events(["Again."])]

    await harness.send(_Request("agent"))

    assistant = _assistant_with_tool_calls(harness.captured["provider_requests"][2])
    assert "reasoning" not in assistant


@pytest.mark.asyncio
async def test_saved_turn_replays_its_reasoning_field_on_the_next_turn(harness):
    await _tool_turn(harness, VLLM, "glm-5.2", {"reasoning": "Need to search."}, {"content": "Looking."})
    harness.context.history = [harness.context.current, _as_history_row(harness.saved[0])]
    harness.provider_rounds = [_sse_events(["Again."])]

    await harness.send(_Request("agent"))

    assistant = _assistant_with_tool_calls(harness.captured["provider_requests"][2])
    assert assistant["reasoning"] == "Need to search."
