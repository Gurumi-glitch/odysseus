"""A provider stop that is not a normal finish ends the turn.

Driven through the chat route with scripted provider streams; the executor's
inner worker counts the calls that really run.
"""

import json
import logging

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.tool_execution as tool_execution
import src.tool_security as tool_security
from core import database
from src.chat_agent.providers import chat_completions
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_tool_calls,
    _switch_on,
)

ZAI_PAAS = "https://api.z.ai/api/paas/v4"
OPENAI = "https://api.openai.com/v1"
ZAI_MODEL = "glm-5.2"
OPENAI_MODEL = "gpt-5.5"


def _stream(deltas, finish_reason=None):
    chunks = [{"choices": [{"delta": delta}]} for delta in deltas]
    if finish_reason:
        chunks.append({"choices": [{"delta": {}, "finish_reason": finish_reason}]})
    return b"".join(("data: " + json.dumps(chunk) + "\n\n").encode() for chunk in chunks) + b"data: [DONE]\n\n"


def _stopped_tool_call(finish_reason):
    """A complete web_search call whose round the provider then stops."""
    chunk = {"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                             "function": {"name": "web_search", "arguments": '{"query": "cats"}'}}]}
    return _stream([{"content": "Searching."}, chunk], finish_reason=finish_reason)


def _error(body):
    lines = body.splitlines()
    return json.loads(lines[lines.index("event: error") + 1][len("data: "):])


def _blocks(row):
    return [block for turn_round in row.metadata["agent_v2"]["rounds"] for block in turn_round["blocks"]]


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))


@pytest.fixture
def harness(monkeypatch, tmp_path, db):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)
    harness = _Harness(monkeypatch, tmp_path)
    _switch_on()
    return harness


@pytest.fixture
def worker_calls(monkeypatch):
    calls = []

    async def worker(block, *args, **kwargs):
        calls.append(block.tool_type)
        return "web_search: ran", {"output": "ran", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", worker)
    return calls


def _use(harness, url, model):
    harness.session.endpoint_url, harness.session.model = url, model


@pytest.mark.asyncio
async def test_sensitive_on_zai_keeps_the_partial_text_and_ends_in_a_saved_error(harness):
    _use(harness, ZAI_PAAS, ZAI_MODEL)
    harness.provider_rounds = [_stream([{"content": "Cats are"}], finish_reason="sensitive")]
    message = f"Stopped: {ZAI_MODEL} stopped the answer: the provider refused it (finish_reason: sensitive)."

    _, body = await harness.send(_Request("agent"))

    assert _error(body) == {"error": message, "status": 502}
    assert not [e for e in _data_events(body) if e.get("type") == "final_response"]
    assert [e["delta"] for e in _data_events(body) if "delta" in e] == ["Cats are"]
    [row] = harness.saved
    assert row.metadata["error"] == message
    assert row.metadata["stop_reason"] == "error"
    assert [b["text"] for b in _blocks(row) if b["type"] == "text"] == ["Cats are"]
    assert row.content == "Cats are"


@pytest.mark.asyncio
async def test_content_filter_on_openai_is_a_refusal(harness):
    _use(harness, OPENAI, OPENAI_MODEL)
    harness.provider_rounds = [_stream([{"content": "I can"}], finish_reason="content_filter")]
    message = f"Stopped: {OPENAI_MODEL} stopped the answer: the provider refused it (finish_reason: content_filter)."

    _, body = await harness.send(_Request("agent"))

    assert _error(body) == {"error": message, "status": 502}
    [row] = harness.saved
    assert row.metadata["error"] == message


@pytest.mark.asyncio
async def test_network_error_with_a_tool_call_does_not_run_the_call(harness, worker_calls):
    """The saved round drops the call: with no result it would not replay."""
    _use(harness, ZAI_PAAS, ZAI_MODEL)
    harness.provider_rounds = [_stopped_tool_call("network_error")]
    message = f"Stopped: {ZAI_MODEL} stopped the answer: the provider failed (finish_reason: network_error)."

    _, body = await harness.send(_Request("agent", allow_web_search="true"))

    assert _error(body) == {"error": message, "status": 502}
    assert worker_calls == []
    assert len(harness.captured["provider_requests"]) == 1
    assert not [e for e in _data_events(body) if e.get("type") == "tool_start"]
    [row] = harness.saved
    assert row.metadata["error"] == message
    assert [b["type"] for b in _blocks(row)] == ["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [
    "network_error", "error", "aborted", "insufficient_system_resource", "model_context_window_exceeded",
])
async def test_every_failed_value_ends_the_turn_naming_the_raw_reason(harness, reason):
    _use(harness, ZAI_PAAS, ZAI_MODEL)
    harness.provider_rounds = [_stream([{"content": "Partial"}], finish_reason=reason)]

    _, body = await harness.send(_Request("agent"))

    assert _error(body) == {
        "error": f"Stopped: {ZAI_MODEL} stopped the answer: the provider failed (finish_reason: {reason}).",
        "status": 502,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stop", "tool_calls", "function_call"])
async def test_normal_finish_reasons_complete_the_turn(harness, reason):
    harness.provider_rounds = [_stream([{"content": "Cats are small."}], finish_reason=reason)]

    _, body = await harness.send(_Request("agent"))

    assert "event: error" not in body
    [row] = harness.saved
    assert row.content == "Cats are small."
    assert "error" not in row.metadata


@pytest.mark.asyncio
async def test_unknown_finish_reason_is_logged_once_and_treated_as_normal(harness, caplog):
    harness.provider_rounds = [_stream([{"content": "Cats are small."}], finish_reason="brand_new_reason")]

    with caplog.at_level(logging.WARNING, logger=chat_completions.__name__):
        _, body = await harness.send(_Request("agent"))

    assert "event: error" not in body
    [row] = harness.saved
    assert row.content == "Cats are small."
    [record] = [r for r in caplog.records if "brand_new_reason" in r.getMessage()]
    assert record.levelno == logging.WARNING


@pytest.mark.asyncio
async def test_each_unknown_finish_reason_is_logged_once_per_process(harness, caplog):
    harness.provider_rounds = [
        _stream([{"content": "One."}], finish_reason="repeat_unknown_reason"),
        _stream([{"content": "Two."}], finish_reason="repeat_unknown_reason"),
        _stream([{"content": "Three."}], finish_reason="other_unknown_reason"),
    ]

    with caplog.at_level(logging.WARNING, logger=chat_completions.__name__):
        for _ in range(3):
            await harness.send(_Request("agent"))

    assert len(harness.saved) == 3
    logged = [r.getMessage() for r in caplog.records if "unknown_reason" in r.getMessage()]
    assert len(logged) == 2
    assert "repeat_unknown_reason" in logged[0] and "other_unknown_reason" in logged[1]
