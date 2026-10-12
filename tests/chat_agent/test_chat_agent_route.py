"""Route-level behaviour of the runtime switch (ticket #48).

Drives POST /api/chat_stream with a scripted Chat Completions transport and
asserts what the browser sees (events, headers) and what is saved.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import routes.chat_routes as chat_routes
import src.constants as constants
import src.foreground_model_routing as foreground_model_routing
import src.settings as settings
from src.chat_agent import store
from src.chat_agent.providers import chat_completions, xai_responses
from src.tool_capabilities import capabilities_for_action

OPENROUTER = "https://openrouter.ai/api/v1"


class _EmptyQuery:
    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return None


class _EmptyDb:
    def query(self, *args, **kwargs):
        return _EmptyQuery()

    def close(self):
        return None


class _Request:
    def __init__(self, mode="agent", **form):
        self.headers = {}
        self.app = SimpleNamespace(state=SimpleNamespace(auth_manager=None))
        self.state = SimpleNamespace(current_user="alice")
        self._form = {"message": "hello", "session": "session-1", "mode": mode, **form}

    async def form(self):
        return self._form


def _sse_events(chunks):
    """Scripted provider stream: Chat Completions SSE lines for these text chunks."""
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]})
        for chunk in chunks
    ]
    return ("\n\n".join(lines + ["data: [DONE]"]) + "\n\n").encode()


class _Harness:
    def __init__(self, monkeypatch, tmp_path, *, chat_handler=None, chat_processor=None):
        self.captured = {"legacy_agent": 0, "legacy_chat": 0, "provider_requests": []}
        self.saved = []
        self.provider_chunks = ["Hel", "lo!"]
        self.monkeypatch = monkeypatch
        monkeypatch.setattr(constants, "CHAT_AGENT_DIR", str(tmp_path / "chat_agent"))
        monkeypatch.setattr(constants, "INSTRUCTIONS_DIR", str(tmp_path / "instructions"))
        monkeypatch.setenv("AUTH_ENABLED", "true")

        harness = self

        def add_message(message):
            message.metadata = dict(message.metadata or {}, _db_id=f"row-{len(harness.saved) + 1}")
            harness.saved.append(message)

        self.session = SimpleNamespace(
            id="session-1",
            endpoint_url=OPENROUTER,
            model="x-ai/grok-4.1-fast",
            headers={"Authorization": "Bearer key"},
            name="test",
            history=[],
            add_message=add_message,
        )
        session_manager = SimpleNamespace(
            get_session=lambda session_id: self.session,
            save_sessions=lambda: None,
        )
        self.context = SimpleNamespace(
            preface=[
                {"role": "system", "content": "You are Odysseus."},
                {"role": "user", "content": "[memory] likes tea"},
            ],
            history=[
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
            ],
            current={"role": "user", "content": "hello"},
            datetime_message={"role": "user", "content": "[Context — current date/time] now"},
        )

        async def fake_build_context(*args, **kwargs):
            self.captured["build_context"] = kwargs
            c = self.context
            return SimpleNamespace(
                user="alice",
                preface=c.preface,
                messages=[*c.preface, *c.history, c.datetime_message, c.current],
                route_messages=[*c.preface, *c.history, c.datetime_message, c.current],
                preprocessed=SimpleNamespace(attachment_meta=[]),
                auto_opened_docs=[],
                rag_sources=[],
                web_sources=[],
                used_memories=[],
                uploaded_files=[],
                uprefs={},
                was_compacted=False,
                context_trimmed=False,
                context_length=4096,
                context_messages_before_trim=1,
                context_messages_after_trim=1,
                context_tokens_before_trim=10,
                context_tokens_after_trim=10,
                preset=SimpleNamespace(temperature=0.2, max_tokens=128, character_name=None),
            )

        async def fake_legacy_agent(*args, **kwargs):
            self.captured["legacy_agent"] += 1
            yield f'data: {json.dumps({"delta": "legacy"})}\n\n'
            yield "data: [DONE]\n\n"

        async def fake_legacy_chat(*args, **kwargs):
            self.captured["legacy_chat"] += 1
            yield f'data: {json.dumps({"delta": "legacy"})}\n\n'
            yield "data: [DONE]\n\n"

        def provider(request: httpx.Request) -> httpx.Response:
            self.captured["provider_requests"].append(json.loads(request.content))
            return httpx.Response(
                200,
                content=_sse_events(self.provider_chunks),
                headers={"content-type": "text/event-stream"},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)

        for name, value in {
            "coerce_message_and_session": lambda *a, **k: ("hello", "session-1"),
            "_verify_session_owner": lambda *a, **k: None,
            "effective_user": lambda request: "alice",
            "_clear_orphaned_session_endpoint": lambda *a, **k: False,
            "_recover_empty_session_model": lambda *a, **k: False,
            "_enforce_chat_privileges": lambda *a, **k: None,
            "resolve_session_auth": lambda *a, **k: None,
            "get_session_mode": lambda session_id: "chat",
            "set_session_mode": lambda *a, **k: None,
            "build_chat_context": fake_build_context,
            "SessionLocal": _EmptyDb,
            "_is_image_generation_session": lambda *a, **k: False,
            "stream_llm_with_fallback": fake_legacy_chat,
            "stream_agent_loop": fake_legacy_agent,
            "save_assistant_response": lambda *a, **k: None,
            "run_post_response_tasks": lambda *a, **k: None,
            "estimate_tokens": lambda messages: 10,
            "accumulate_token_usage": lambda *a, **k: None,
        }.items():
            monkeypatch.setattr(chat_routes, name, value)
        monkeypatch.setattr(
            foreground_model_routing, "_load_policy_preferences", lambda owner=None: {},
        )
        monkeypatch.setattr(settings, "get_setting", lambda key, default=None: default)
        monkeypatch.setattr(
            settings, "get_user_setting", lambda key, owner="", default=None: default,
        )
        router = chat_routes.setup_chat_routes(
            session_manager, chat_handler or SimpleNamespace(),
            chat_processor or SimpleNamespace(), SimpleNamespace(),
            SimpleNamespace(), SimpleNamespace(),
        )
        self.routes = {route.path: route.endpoint for route in router.routes}
        self.endpoint = self.routes["/api/chat_stream"]

    async def send(self, request):
        response = await self.endpoint(request)
        body = "".join([chunk async for chunk in response.body_iterator])
        return response, body


def _data_events(body):
    events = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


@pytest.fixture
def harness(monkeypatch, tmp_path):
    return _Harness(monkeypatch, tmp_path)


def _switch_on():
    store.set_chat_runtime("alice", store.RUNTIME_NEW)


@pytest.mark.asyncio
async def test_switch_on_agent_turn_streams_and_saves_on_new_runtime(harness):
    _switch_on()

    response, body = await harness.send(_Request("agent"))

    assert response.headers["X-Odysseus-Runtime"] == "chat_agent"
    assert response.headers["X-Odysseus-Run-Id"]
    events = _data_events(body)
    assert [e["delta"] for e in events if "delta" in e] == ["Hel", "lo!"]
    assert {"type": "message_saved", "id": "row-1"} in events
    assert body.rstrip().endswith("data: [DONE]")
    assert harness.captured["legacy_agent"] == 0

    [row] = harness.saved
    assert row.role == "assistant"
    assert row.content == "Hello!"
    assert row.metadata["runtime"] == "chat_agent"
    assert row.metadata["round_texts"] == ["Hello!"]
    assert row.metadata["agent_v2"]["rounds"] == [
        {"blocks": [{"type": "text", "text": "Hello!"}]}
    ]
    assert row.metadata["agent_v2"]["provider"] == "openrouter"


@pytest.mark.asyncio
async def test_new_runtime_prompt_order_and_context_flags(harness):
    _switch_on()

    await harness.send(_Request("agent"))

    assert harness.captured["build_context"]["prefetch"] is False
    assert harness.captured["build_context"]["agent_mode"] is False
    [payload] = harness.captured["provider_requests"]
    assert payload["model"] == "x-ai/grok-4.1-fast"
    assert payload["stream"] is True
    assert "tools" not in payload
    messages = payload["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"].startswith("You are Odysseus.\n\n")
    assert "[memory]" not in messages[0]["content"]
    assert messages[1:3] == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    current = messages[3]["content"]
    assert current.index("[memory] likes tea") < current.index("current date/time") < current.index("hello")
    assert current.endswith("hello")


@pytest.mark.asyncio
async def test_provider_error_ends_turn_visibly_without_saving(harness, monkeypatch):
    _switch_on()

    def failing(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(failing))
    monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)

    _, body = await harness.send(_Request("agent"))

    assert "event: error" in body
    assert "slow down" in body
    assert harness.saved == []
    assert harness.captured["legacy_agent"] == 0


@pytest.mark.asyncio
async def test_switch_off_agent_turn_takes_legacy_loop(harness):
    response, _ = await harness.send(_Request("agent", compare_mode="false"))

    assert harness.captured["legacy_agent"] == 1
    assert harness.captured["provider_requests"] == []
    assert response.headers["X-Odysseus-Runtime"] == "legacy"


@pytest.mark.asyncio
async def test_chat_mode_turn_stays_legacy(harness):
    _switch_on()

    response, _ = await harness.send(_Request("chat", use_rag="false"))

    assert harness.captured["legacy_chat"] == 1
    assert harness.captured["provider_requests"] == []
    assert response.headers["X-Odysseus-Runtime"] == "legacy"


@pytest.mark.asyncio
async def test_compare_mode_stays_legacy(harness):
    _switch_on()

    await harness.send(_Request("agent", compare_mode="true"))

    assert harness.captured["legacy_agent"] == 1
    assert harness.captured["provider_requests"] == []


@pytest.mark.asyncio
async def test_compare_session_stays_legacy(harness):
    _switch_on()
    harness.session.name = "[CMP] pane"

    await harness.send(_Request("agent"))

    assert harness.captured["legacy_agent"] == 1
    assert harness.captured["provider_requests"] == []


@pytest.mark.asyncio
async def test_image_generation_session_stays_legacy(harness, monkeypatch):
    _switch_on()
    monkeypatch.setattr(chat_routes, "_is_image_generation_session", lambda *a, **k: True)
    monkeypatch.setattr(
        settings, "get_setting",
        lambda key, default=None: False if key == "image_gen_enabled" else default,
    )

    _, body = await harness.send(_Request("agent"))

    assert "Image generation is disabled" in body
    assert harness.captured["provider_requests"] == []


@pytest.mark.asyncio
async def test_research_turn_stays_legacy(harness, monkeypatch):
    _switch_on()
    request = _Request("agent", use_research="true")
    # Research privilege off: the legacy route falls through to its agent loop.
    request.app.state.auth_manager = SimpleNamespace(
        get_privileges=lambda user: {"can_use_research": False},
    )

    await harness.send(request)

    assert harness.captured["legacy_agent"] == 1
    assert harness.captured["provider_requests"] == []


@pytest.mark.asyncio
async def test_tui_surface_stays_legacy(harness):
    _switch_on()
    request = _Request(
        "agent",
        client_runtime_context=json.dumps({"surface": "odysseus-tui"}),
    )

    await harness.send(request)

    assert harness.captured["legacy_agent"] == 1
    assert harness.captured["provider_requests"] == []


@pytest.mark.asyncio
async def test_legacy_approval_continuation_stays_legacy(harness):
    _switch_on()
    pending = chat_routes.tool_approval_store.create(
        owner="alice",
        session_id="session-1",
        origin_run_id="run-1",
        tool_name="bash",
        content="printf exact",
        workspace=None,
        external_untrusted_context_seen=True,
        capabilities=capabilities_for_action("bash", "printf exact"),
    )

    await harness.send(_Request(
        "agent", tool_approval_id=pending.approval_id, tool_approval_decision="approve",
    ))

    assert harness.captured["legacy_agent"] == 1
    assert harness.captured["provider_requests"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://api.anthropic.com/v1",
        "http://localhost:11434",
        "https://chatgpt.com/backend-api/codex",
    ],
    ids=["anthropic", "ollama-native", "chatgpt-subscription"],
)
async def test_providers_without_adapter_stay_legacy(harness, endpoint_url):
    _switch_on()
    harness.session.endpoint_url = endpoint_url

    await harness.send(_Request("agent"))

    assert harness.captured["legacy_agent"] == 1
    assert harness.captured["provider_requests"] == []


@pytest.mark.asyncio
async def test_auth_disabled_stays_legacy(harness, monkeypatch):
    _switch_on()
    monkeypatch.setenv("AUTH_ENABLED", "false")

    await harness.send(_Request("agent"))

    assert harness.captured["legacy_agent"] == 1
    assert harness.captured["provider_requests"] == []


@pytest.mark.asyncio
async def test_text_tool_model_stays_legacy(harness):
    _switch_on()
    harness.session.endpoint_url = "https://llm.example/v1"
    harness.session.model = "gpt-oss-120b"

    await harness.send(_Request("agent"))

    assert harness.captured["legacy_agent"] == 1
    assert harness.captured["provider_requests"] == []


class _TextUploads:
    def __init__(self, path):
        self.path = str(path)

    def resolve_upload(self, upload_id, owner=None, **kwargs):
        return {"id": upload_id, "name": "notes.txt", "mime": "text/plain", "path": self.path}

    def _inside_upload_dir(self, path):
        return True

    def is_image_file(self, *args):
        return False

    def is_audio_file(self, *args):
        return False

    def is_document_file(self, *args):
        return True


@pytest.mark.asyncio
async def test_new_runtime_never_prefetches_a_pasted_url_but_keeps_attachments(
    monkeypatch, tmp_path,
):
    """Real build_chat_context, real ChatHandler and ChatProcessor."""
    import routes.chat_helpers as chat_helpers
    import src.chat_handler as chat_handler_module
    import src.chat_processor as chat_processor_module
    from src.chat_handler import ChatHandler
    from src.chat_processor import ChatProcessor

    fetched = []
    monkeypatch.setattr(
        chat_processor_module, "fetch_webpage_content",
        lambda url, *a, **k: fetched.append(url) or {"success": True, "content": "page"},
    )

    monkeypatch.setattr(
        chat_processor_module, "comprehensive_web_search",
        lambda *a, **k: fetched.append("web search") or ("", []),
    )

    async def no_transcript(*args, **kwargs):
        fetched.append("youtube")
        return {}

    monkeypatch.setattr(chat_handler_module, "extract_transcript_async", no_transcript)
    monkeypatch.setattr(chat_handler_module, "model_supports_vision", lambda *a, **k: False)
    attachment = tmp_path / "notes.txt"
    attachment.write_text("ATTACHED-NOTES-BODY", encoding="utf-8")
    handler = ChatHandler(
        session_manager=None, memory_manager=None, chat_processor=None,
        research_handler=None, preset_manager=None, upload_handler=_TextUploads(attachment),
    )
    processor = ChatProcessor(memory_manager=None, personal_docs_manager=None)

    harness = _Harness(monkeypatch, tmp_path, chat_handler=handler, chat_processor=processor)
    harness.session.history = []
    harness.session.get_context_messages = lambda: [m.to_dict() for m in harness.session.history]
    original_add = harness.session.add_message

    def add_message(message):
        harness.session.history.append(message)
        original_add(message)

    harness.session.add_message = add_message
    monkeypatch.setattr(chat_routes, "build_chat_context", chat_helpers.build_chat_context)
    monkeypatch.setattr(chat_helpers, "extract_preset", lambda *a: chat_helpers.PresetInfo(
        temperature=None, max_tokens=None, system_prompt="You are Odysseus.", character_name=None,
    ))
    monkeypatch.setattr(chat_helpers, "fire_message_event", lambda *a, **k: None)
    monkeypatch.setattr(chat_helpers, "load_prefs_for_user", lambda user: {})
    monkeypatch.setattr(chat_helpers, "effective_user", lambda request: "alice")
    monkeypatch.setattr(chat_helpers, "_normalize_model_id_from_cache", lambda sess: None)
    monkeypatch.setattr(chat_helpers, "normalize_model_id", lambda *a, **k: None)

    async def no_compaction(sess, url, model, messages, headers, owner=None, **kwargs):
        return messages, 128000, False

    monkeypatch.setattr(chat_helpers, "maybe_compact", no_compaction)
    _switch_on()
    url_message = "Summarize https://example.com/report and https://youtu.be/dQw4w9WgXcQ"
    monkeypatch.setattr(
        chat_routes, "coerce_message_and_session", lambda *a, **k: (url_message, "session-1"),
    )

    _, body = await harness.send(_Request(
        "agent", message=url_message, attachments=json.dumps(["upload-1"]),
        no_memory="true", use_rag="false", use_web="true",
    ))

    assert fetched == []
    assert "event: error" not in body
    [payload] = harness.captured["provider_requests"]
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    current = payload["messages"][-1]["content"]
    assert "current date/time" in current
    assert current.rstrip().endswith("ATTACHED-NOTES-BODY")
    assert "https://example.com/report" in current
    assert "Content from https://example.com/report" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_resume_of_a_live_new_runtime_run_carries_the_runtime_header(harness, monkeypatch):
    _switch_on()
    release = asyncio.Event()

    async def held_turn(turn):
        yield f'data: {json.dumps({"delta": "partial"})}\n\n'
        await release.wait()
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat_routes, "run_chat_agent_turn", held_turn)
    response = await harness.endpoint(_Request("agent"))
    resume = await harness.routes["/api/chat/resume/{session_id}"](_Request(), "session-1")

    assert response.headers["X-Odysseus-Runtime"] == "chat_agent"
    assert resume.headers["X-Odysseus-Runtime"] == "chat_agent"
    assert resume.headers["X-Odysseus-Run-Id"] == response.headers["X-Odysseus-Run-Id"]
    release.set()
    resumed_body = "".join([chunk async for chunk in resume.body_iterator])
    assert "partial" in resumed_body


# --- xAI Responses (ticket #51) -------------------------------------------

XAI_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "chat_agent" / "xai_responses_round2_text.sse"
).read_bytes()


def _xai_session(harness, monkeypatch, model="grok-4.7"):
    harness.session.endpoint_url = "https://api.x.ai/v1"
    harness.session.model = model
    requests = []

    def provider(request):
        requests.append(request)
        return httpx.Response(200, content=XAI_FIXTURE, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    monkeypatch.setattr(xai_responses, "_get_http_client", lambda: client)
    return requests


def _fixture_output_items():
    events = [json.loads(line[6:]) for line in XAI_FIXTURE.decode().splitlines() if line.startswith("data: {")]
    return [e["item"] for e in events if e["type"] == "response.output_item.done"]


def _as_history_row(row):
    """The saved row as build_chat_context returns it after a DB round trip."""
    return {"role": row.role, "content": row.content, "metadata": json.loads(json.dumps(row.metadata))}


@pytest.mark.asyncio
async def test_xai_turn_runs_on_responses_and_saves_tagged_reasoning(harness, monkeypatch):
    _switch_on()
    requests = _xai_session(harness, monkeypatch)

    response, body = await harness.send(_Request("agent"))

    assert response.headers["X-Odysseus-Runtime"] == "chat_agent"
    assert harness.captured["provider_requests"] == []
    [request] = requests
    assert str(request.url) == "https://api.x.ai/v1/responses"
    payload = json.loads(request.content)
    assert payload["store"] is False and payload["prompt_cache_key"] == "session-1"
    assert [e["delta"] for e in _data_events(body) if "delta" in e] == ["Added the note ", '"Dinner".']
    [row] = harness.saved
    assert row.content == 'Added the note "Dinner".'
    assert row.metadata["round_texts"] == ['Added the note "Dinner".']
    agent_v2 = row.metadata["agent_v2"]
    assert (agent_v2["provider"], agent_v2["model"]) == ("xai", "grok-4.7")
    [turn_round] = agent_v2["rounds"]
    reasoning = turn_round["blocks"][0]
    assert (reasoning["type"], reasoning["provider"], reasoning["model"]) == ("reasoning", "xai", "grok-4.7")
    assert turn_round["usage"]["cached_input_tokens"] == 1792


@pytest.mark.asyncio
async def test_next_turn_replays_the_saved_turn_byte_identical(harness, monkeypatch):
    _switch_on()
    requests = _xai_session(harness, monkeypatch)
    await harness.send(_Request("agent"))
    first_input = json.loads(requests[0].content)["input"]
    harness.context.history = [*harness.context.history, harness.context.current, _as_history_row(harness.saved[0])]

    await harness.send(_Request("agent"))

    replayed = ",".join(
        json.dumps(item, separators=(",", ":"), ensure_ascii=False) for item in _fixture_output_items()
    ).encode()
    assert replayed in requests[1].content
    second_input = json.loads(requests[1].content)["input"]
    # Append-only: everything before the first turn's current message is unchanged.
    assert second_input[: len(first_input) - 1] == first_input[:-1]


@pytest.mark.asyncio
async def test_model_switch_mid_chat_drops_the_other_models_reasoning(harness, monkeypatch):
    _switch_on()
    requests = _xai_session(harness, monkeypatch, model="grok-4.5")
    await harness.send(_Request("agent"))
    harness.context.history = [harness.context.current, _as_history_row(harness.saved[0])]
    harness.session.model = "grok-4.7"

    await harness.send(_Request("agent"))

    second_input = json.loads(requests[1].content)["input"]
    assert "reasoning" not in [item.get("type") for item in second_input]
    assert _fixture_output_items()[1] in second_input
