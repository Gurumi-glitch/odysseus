"""The live stream carries the context's rag_sources and memories_used.

The browser draws the sources box and the "N pinned, M recalled" pill from
these events; before this fix they only showed after a history reload.
"""

import pytest

import routes.chat_routes as chat_routes
from tests.chat_agent.test_chat_agent_route import _Harness, _Request, _data_events, _switch_on

RAG = [{"filename": "a.md", "similarity": 0.9, "snippet": "alpha"}]
MEMORIES = [{"type": "pinned", "text": "likes tea"}, {"type": "recalled", "text": "owns a cat"}]


def _with_context(harness, monkeypatch, **fields):
    build = chat_routes.build_chat_context

    async def build_with(*args, **kwargs):
        ctx = await build(*args, **kwargs)
        for name, value in fields.items():
            setattr(ctx, name, value)
        return ctx

    monkeypatch.setattr(chat_routes, "build_chat_context", build_with)


@pytest.mark.asyncio
async def test_rag_sources_then_memories_used_stream_before_first_delta(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path)
    _with_context(harness, monkeypatch, rag_sources=RAG, used_memories=MEMORIES)
    _switch_on()

    _, body = await harness.send(_Request("agent"))

    events = _data_events(body)
    kinds = [e.get("type") or ("delta" if "delta" in e else None) for e in events]
    assert kinds.index("rag_sources") < kinds.index("memories_used") < kinds.index("delta")
    assert {"type": "rag_sources", "data": RAG} in events
    assert {"type": "memories_used", "data": MEMORIES} in events
    [row] = harness.saved
    assert row.metadata["rag_sources"] == RAG
    assert row.metadata["memories_used"] == MEMORIES


@pytest.mark.asyncio
async def test_no_context_events_when_nothing_was_used(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path)
    _switch_on()

    _, body = await harness.send(_Request("agent"))

    types = {e.get("type") for e in _data_events(body)}
    assert "rag_sources" not in types
    assert "memories_used" not in types
