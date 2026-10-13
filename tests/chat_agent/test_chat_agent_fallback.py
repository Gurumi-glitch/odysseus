"""Model fallback that sticks (ticket #55), on both runtimes.

Drives POST /api/chat_stream with a scripted provider transport and asserts
what the browser sees (the legacy ``fallback`` event), what is saved, and the
session route a reload reads back.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

import routes.chat_routes as chat_routes
import src.foreground_model_routing as foreground_model_routing
from src.chat_agent import loop, transcript
from src.chat_agent.providers import chat_completions
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _switch_on,
)

PRIMARY_MODEL = "x-ai/grok-4.1-fast"
BACKUP_URL = "https://backup.example/v1"
BACKUP_MODEL = "backup-model"
BACKUP_HEADERS = {"Authorization": "Bearer two"}


class _SessionRow:
    def __init__(self):
        self.model = PRIMARY_MODEL
        self.endpoint_url = "https://openrouter.ai/api/v1"
        self.endpoint_id = "primary-ep"
        self.headers = {}
        self.updated_at = None


class _RowQuery:
    def __init__(self, row):
        self.row = row

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.row


class _RecordingDb:
    """The sessions table as a reload reads it: one row, commits counted."""

    def __init__(self, row):
        self.row = row
        self.commits = 0

    def query(self, *args, **kwargs):
        return _RowQuery(self.row)

    def commit(self):
        self.commits += 1

    def close(self):
        return None


@pytest.fixture
def harness(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path)
    harness.session.endpoint_id = "primary-ep"
    monkeypatch.setattr(foreground_model_routing, "_load_policy_preferences", lambda owner=None: {
        "foreground_fallback_enabled": True,
        "foreground_model_fallbacks": [{"endpoint_id": "backup-ep", "model": BACKUP_MODEL}],
    })
    monkeypatch.setattr(
        foreground_model_routing, "resolve_fallback_entries",
        lambda *args, **kwargs: [(BACKUP_URL, BACKUP_MODEL, dict(BACKUP_HEADERS))],
    )
    monkeypatch.setattr(
        foreground_model_routing, "resolve_route_descriptor_by_id",
        lambda *args, **kwargs: {"endpoint_id": "primary-ep", "endpoint_label": "Primary"},
    )
    monkeypatch.setattr(foreground_model_routing, "endpoint_cost_tracked", lambda url: False)
    # Large window: the fallback fits without compaction unless a test says so.
    monkeypatch.setattr(transcript, "get_context_length", lambda url, model: 128_000)
    harness.row = _SessionRow()
    harness.db = _RecordingDb(harness.row)
    monkeypatch.setattr(chat_routes, "SessionLocal", lambda: harness.db)
    return harness


def _script(monkeypatch, harness, responses):
    """Answer each provider request from that model's queue: an HTTP status, or text chunks."""
    calls = harness.captured["provider_calls"] = []

    def provider(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        harness.captured["provider_requests"].append(payload)
        calls.append(payload["model"])
        answer = responses[payload["model"]].pop(0)
        if isinstance(answer, int):
            return httpx.Response(answer, json={"error": {"message": f"upstream {answer}"}})
        content = answer if isinstance(answer, bytes) else _sse_events(answer)
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)


@pytest.mark.asyncio
async def test_new_runtime_503_falls_back_and_sticks(harness, monkeypatch):
    _switch_on()
    _script(monkeypatch, harness, {PRIMARY_MODEL: [503], BACKUP_MODEL: [["From ", "backup"]]})

    response, body = await harness.send(_Request("agent"))

    assert response.headers["X-Odysseus-Runtime"] == "chat_agent"
    assert harness.captured["provider_calls"] == [PRIMARY_MODEL, BACKUP_MODEL]
    events = _data_events(body)
    [fallback] = [e for e in events if e.get("type") == "fallback"]
    assert fallback["selected_model"] == PRIMARY_MODEL
    assert fallback["answered_by"] == BACKUP_MODEL
    assert fallback["selected_endpoint_id"] == "primary-ep"
    assert fallback["answered_by_endpoint_id"] == "backup-ep"
    assert fallback["candidate_index"] == 1
    assert fallback["failures"] == [{"candidate_index": 0, "model": PRIMARY_MODEL, "status": 503}]
    assert "upstream 503" in fallback["reason"]
    # The notice arrives before the fallback's first output.
    assert events.index(fallback) < next(i for i, e in enumerate(events) if "delta" in e)
    assert [e["delta"] for e in events if "delta" in e] == ["From ", "backup"]
    assert "event: error" not in body

    [row] = harness.saved
    assert row.content == "From backup"
    assert row.metadata["model"] == BACKUP_MODEL
    assert row.metadata["agent_v2"]["model"] == BACKUP_MODEL

    # Sticky: the in-memory session and the row a reload reads both moved.
    assert (harness.session.model, harness.session.endpoint_url) == (BACKUP_MODEL, BACKUP_URL)
    assert harness.session.endpoint_id == "backup-ep"
    assert harness.session.headers == BACKUP_HEADERS
    assert (harness.row.model, harness.row.endpoint_url, harness.row.endpoint_id) == (
        BACKUP_MODEL, BACKUP_URL, "backup-ep",
    )
    assert harness.row.headers == BACKUP_HEADERS
    assert harness.db.commits == 1


@pytest.mark.asyncio
async def test_new_runtime_failure_after_output_is_not_failed_over(harness, monkeypatch):
    _switch_on()
    streamed_then_failed = (
        "data: " + json.dumps({"choices": [{"delta": {"content": "Half an answ"}}]}) + "\n\n"
        + "data: " + json.dumps({"error": {"message": "upstream overloaded", "code": 503}}) + "\n\n"
    ).encode()
    _script(monkeypatch, harness, {PRIMARY_MODEL: [streamed_then_failed], BACKUP_MODEL: [["unused"]]})

    _, body = await harness.send(_Request("agent"))

    assert harness.captured["provider_calls"] == [PRIMARY_MODEL]
    assert "event: error" in body and "upstream overloaded" in body
    assert not [e for e in _data_events(body) if e.get("type") == "fallback"]
    assert harness.session.model == PRIMARY_MODEL
    assert harness.db.commits == 0


@pytest.mark.asyncio
async def test_new_runtime_ineligible_status_is_not_failed_over(harness, monkeypatch):
    _switch_on()
    _script(monkeypatch, harness, {PRIMARY_MODEL: [400], BACKUP_MODEL: [["unused"]]})

    _, body = await harness.send(_Request("agent"))

    assert harness.captured["provider_calls"] == [PRIMARY_MODEL]
    assert "event: error" in body
    assert harness.session.model == PRIMARY_MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answers", "calls", "answered"),
    [([503, ["ok"]], 2, True), ([529, 503], 2, False), ([500, ["ok"]], 1, False)],
    ids=["retry-answers", "one-retry-only", "500-not-retried"],
)
async def test_new_runtime_single_candidate_retries_once(harness, monkeypatch, answers, calls, answered):
    _switch_on()
    monkeypatch.setattr(foreground_model_routing, "_load_policy_preferences", lambda owner=None: {})
    monkeypatch.setattr(loop.LLMConfig, "RETRY_DELAY", 0)
    _script(monkeypatch, harness, {PRIMARY_MODEL: answers})

    _, body = await harness.send(_Request("agent"))

    assert harness.captured["provider_calls"] == [PRIMARY_MODEL] * calls
    assert ("event: error" not in body) is answered
    assert not [e for e in _data_events(body) if e.get("type") == "fallback"]
    assert harness.db.commits == 0


@pytest.mark.asyncio
async def test_new_runtime_fallback_drops_reasoning_of_other_models(harness, monkeypatch):
    _switch_on()
    reasoning = [{"type": "reasoning.encrypted", "data": "opaque"}]
    harness.context.history = [
        {"role": "user", "content": "earlier question"},
        # A row the transcript writer signed (unsigned agent_v2 replays as text).
        {"role": "assistant", "content": "earlier answer", "metadata": transcript.signed_metadata(
            "earlier answer", {"agent_v2": {"rounds": [{"blocks": [
                {"type": "reasoning", "provider": "openrouter", "model": PRIMARY_MODEL,
                 "reasoning_details": reasoning},
                {"type": "text", "text": "earlier answer"},
            ]}]}},
        )},
    ]
    _script(monkeypatch, harness, {PRIMARY_MODEL: [503], BACKUP_MODEL: [["ok"]]})

    await harness.send(_Request("agent"))

    primary, backup = harness.captured["provider_requests"]
    assert primary["messages"][2]["reasoning_details"] == reasoning
    assert backup["messages"][2] == {"role": "assistant", "content": "earlier answer"}


@pytest.mark.asyncio
async def test_new_runtime_compacts_once_for_a_smaller_fallback_window(harness, monkeypatch):
    _switch_on()
    harness.context.history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "words " * 400}
        for i in range(8)
    ]
    windows = {PRIMARY_MODEL: 128_000, BACKUP_MODEL: 2_000}
    monkeypatch.setattr(transcript, "get_context_length", lambda url, model: windows[model])
    monkeypatch.setattr(transcript, "resolve_endpoint", lambda *a, **k: (None, None, None))
    summaries = []

    async def summarize(url, model, messages, **kwargs):
        summaries.append(model)
        return "the earlier turns, summarized"

    monkeypatch.setattr(transcript, "llm_call_async", summarize)
    _script(monkeypatch, harness, {PRIMARY_MODEL: [503], BACKUP_MODEL: [["ok"]]})

    _, body = await harness.send(_Request("agent"))

    assert "event: error" not in body
    assert summaries == [BACKUP_MODEL]
    primary, backup = harness.captured["provider_requests"]
    assert len(backup["messages"]) < len(primary["messages"])
    assert backup["messages"][0] == primary["messages"][0]
    assert "the earlier turns, summarized" in backup["messages"][1]["content"]
    assert backup["messages"][-1] == primary["messages"][-1]
    # The kept part starts at a turn boundary.
    assert backup["messages"][2]["role"] == "user"
    # Request-local: the next turn compacts the stored chat at the sticky window.
    assert harness.session.history == []


def _legacy_fallback_stream(answered_by_endpoint_id):
    async def stream(*args, **kwargs):
        yield "data: " + json.dumps({
            "type": "fallback", "selected_model": PRIMARY_MODEL, "answered_by": BACKUP_MODEL,
            "answered_by_endpoint_id": answered_by_endpoint_id, "candidate_index": 1,
        }) + "\n\n"
        yield "data: " + json.dumps({"delta": "From backup"}) + "\n\n"
        yield "data: [DONE]\n\n"
    return stream


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "legacy_stream"),
    [("chat", "stream_llm_with_fallback"), ("agent", "stream_agent_loop")],
)
async def test_legacy_fallback_sticks(harness, monkeypatch, mode, legacy_stream):
    monkeypatch.setattr(chat_routes, legacy_stream, _legacy_fallback_stream("backup-ep"))

    response, body = await harness.send(_Request(mode, use_rag="false"))

    assert response.headers["X-Odysseus-Runtime"] == "legacy"
    assert [e for e in _data_events(body) if e.get("type") == "fallback"]
    assert (harness.session.model, harness.session.endpoint_url) == (BACKUP_MODEL, BACKUP_URL)
    assert harness.session.endpoint_id == "backup-ep"
    assert harness.session.headers == BACKUP_HEADERS
    assert (harness.row.model, harness.row.endpoint_url, harness.row.endpoint_id) == (
        BACKUP_MODEL, BACKUP_URL, "backup-ep",
    )


@pytest.mark.asyncio
async def test_legacy_fallback_on_an_unknown_route_keeps_the_session(harness, monkeypatch):
    # Same model on another endpoint than the configured fallback: not ours to pin.
    monkeypatch.setattr(chat_routes, "stream_llm_with_fallback", _legacy_fallback_stream("other-ep"))

    await harness.send(_Request("chat", use_rag="false"))

    assert harness.session.model == PRIMARY_MODEL
    assert harness.row.model == PRIMARY_MODEL


@pytest.mark.asyncio
async def test_later_rounds_are_gated_against_the_fallbacks_window(harness, monkeypatch):
    # Context overflow gate (#95): after a fallback, the window that decides is
    # the answering model's, not the selected one's.
    from src.chat_agent import store
    from tests.chat_agent.test_chat_agent_route import _sse_tool_calls

    _switch_on()
    store.set_chat_settings("alice", "session-1", {store.CONTEXT_OVERFLOW: store.OVERFLOW_STOP})
    windows = {PRIMARY_MODEL: 128_000, BACKUP_MODEL: 16_384}
    monkeypatch.setattr(loop, "get_context_length_known", lambda url, model: (windows[model], True))
    # A ~12k-token call: the next request is over 85% of 16k, far under 128k.
    padded = _sse_tool_calls(("no_such_tool", {"pad": "a " * 20_000}))
    _script(monkeypatch, harness, {PRIMARY_MODEL: [503], BACKUP_MODEL: [padded, ["Done."]]})

    _, body = await harness.send(_Request("agent"))

    assert harness.captured["provider_calls"] == [PRIMARY_MODEL, BACKUP_MODEL]
    assert "tokens exceeds the 16384-token window" in body
