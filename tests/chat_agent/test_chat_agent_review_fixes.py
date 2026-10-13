"""Regressions for the spec #44 code-review findings."""
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import routes.history.history_routes as history_routes
from core import database
from core.models import ChatMessage
from src.chat_agent import store
from src.chat_agent import tools as chat_agent_tools
from src.chat_agent.loop import Candidate, _TurnUsage
from src.chat_agent.providers import Endpoint, RoundEnd
from src.model_context import estimate_text_tokens
from tests.chat_agent.test_chat_agent_asks_route import (  # noqa: F401 (fixtures)
    _bash,
    admin,
    answering,
    executor,
    send_and_answer,
    workspace,
)
from tests.chat_agent.test_chat_agent_route import _Request, _data_events, _sse_events, harness  # noqa: F401
from tests.chat_agent.test_chat_agent_security_pass import _JsonRequest
from tests.chat_agent.test_chat_agent_turn_record import _stop_while_an_ask_is_pending


def test_estimated_output_tokens_use_the_shared_unicode_aware_estimate():
    usage = _TurnUsage(0.0, Candidate(Endpoint("http://x", "m", None), {}))
    text = "你好世界" * 10
    usage.add_round(1, RoundEnd([{"type": "text", "text": text}], None), [])
    bucket = usage.buckets[0]
    assert bucket["usage_source"] == "estimated"
    assert bucket["output_tokens"] == estimate_text_tokens(text)


# --- P2: metrics while an ask waits, and on the row of a cancelled turn ------

TURN_END_METRIC_KEYS = {"model", "input_tokens", "output_tokens", "usage_source", "context_length"}


@pytest.mark.asyncio
async def test_metrics_are_sent_when_an_ask_starts_waiting(harness, admin, executor, workspace):
    store.set_chat_runtime("alice", store.RUNTIME_NEW)
    harness.provider_rounds = [_bash("ls"), _sse_events(["Done."])]

    body = await send_and_answer(
        harness, _Request("agent", allow_bash="true", workspace=workspace), answering(harness, ("once",)),
    )

    kinds = [e.get("type") for e in _data_events(body)]
    after_ask = kinds[kinds.index("ask_user") + 1]
    assert after_ask == "metrics"
    assert kinds.count("metrics") == 2


@pytest.mark.asyncio
async def test_a_turn_cancelled_during_an_ask_keeps_its_metrics_on_the_row(harness, monkeypatch, tmp_path):
    store.set_chat_runtime("alice", store.RUNTIME_NEW)
    await _stop_while_an_ask_is_pending(harness, monkeypatch, tmp_path)

    [row] = harness.saved
    assert TURN_END_METRIC_KEYS <= row.metadata.keys()
    assert row.metadata["input_tokens"] > 0


# --- P8: a searched schema keeps its full description on the compact surface --


def test_loaded_schemas_keep_full_descriptions_on_the_compact_surface():
    upfront = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    searched = {"type": "function", "function": {
        "name": "mcp__x__y", "description": "Does y.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "query"}}},
    }}

    offered = chat_agent_tools.offered_schemas(upfront, [searched], ["mcp__x__y"], "compact")

    assert offered[-1] == searched


# --- P3: a merged chat_agent row renders and replays from its content --------


async def _merge(monkeypatch, tmp_path, meta1, meta2):
    sess = SimpleNamespace(history=[
        ChatMessage("user", "list files"),
        ChatMessage("assistant", "Part one", metadata={**meta1, "stopped": True}),
        ChatMessage("user", "Continue from where you left off."),
        ChatMessage("assistant", "part two.", metadata=meta2),
    ])
    monkeypatch.setattr(history_routes, "_verify_session_owner", lambda request, session_id: None)
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    monkeypatch.setattr(history_routes, "SessionLocal", sessionmaker(bind=engine))
    router = history_routes.setup_history_routes(
        SimpleNamespace(get_session=lambda session_id: sess, save_sessions=lambda: None),
    )
    merge = next(r.endpoint for r in router.routes if r.path.endswith("/merge-last-assistant"))
    assert (await merge(_JsonRequest({}), "session-1"))["merged"] is True
    return sess.history[-1]


@pytest.mark.asyncio
async def test_merging_a_chat_agent_row_marks_it_edited(monkeypatch, tmp_path):
    chat_agent = {"runtime": "chat_agent", "agent_v2": {"rounds": []}, "round_texts": ["x"]}

    merged = await _merge(monkeypatch, tmp_path, chat_agent, dict(chat_agent))

    assert merged.content == "Part one\n\npart two."
    assert merged.metadata["edited"] is True


@pytest.mark.asyncio
async def test_merging_legacy_rows_does_not_mark_them_edited(monkeypatch, tmp_path):
    merged = await _merge(monkeypatch, tmp_path, {}, {})

    assert "edited" not in merged.metadata
