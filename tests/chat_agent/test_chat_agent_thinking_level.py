"""Thinking level: detection, per-dialect payload, storage and routes (ticket #96, spec #89).

Detection runs through the capabilities module's interface with recorded
OpenRouter and LM Studio model lists; payloads through a full new-runtime turn
on the chat route; the settings and capabilities routes over HTTP.
"""

import json
from pathlib import Path

import pytest

from src.chat_agent import capabilities
from tests.chat_agent.test_chat_agent_capabilities import (
    BONSAI,
    BONSAI_PROPS,
    LLAMA,
    LMSTUDIO_MODELS,
    _fresh_caches,  # noqa: F401  (autouse: clean capability caches)
    serve_probes,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "chat_agent"
# GET https://openrouter.ai/api/v1/models on 2026-10-13, four models kept,
# fields other than id/name/context/architecture/parameters/reasoning dropped.
OPENROUTER_MODELS = json.loads((FIXTURES / "openrouter_api_v1_models.json").read_text())

OPENROUTER = "https://openrouter.ai/api/v1"
LMSTUDIO = "http://localhost:1234/v1"
GRADED = ("low", "medium", "high", "xhigh", "max")


def _support(off, *graded):
    return {"off": off, **dict(zip(GRADED, graded))}


S, U, N = "supported", "unsupported", "unknown"


# --- Detection --------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "support"),
    [
        # mandatory: Off refused; only the listed efforts.
        ("stepfun/step-5-preview", _support(U, S, S, S, U, U)),
        # supported_efforts absent (null) = every level.
        ("cohere/command-a-plus", _support(S, S, S, S, S, S)),
        ("openai/gpt-6-luna-pro", _support(S, S, S, S, S, S)),
        # No reasoning object (a router that still takes reasoning params): reports nothing.
        ("typesafe/jev-router", _support(N, N, N, N, N, N)),
    ],
)
async def test_openrouter_reports_levels_per_model(monkeypatch, model, support):
    seen = serve_probes(monkeypatch, {"GET /api/v1/models": OPENROUTER_MODELS})

    levels = await capabilities.thinking_levels(OPENROUTER, model)

    assert levels.support == support
    assert levels.binary is False
    # Tools and levels come from the same catalog fetch.
    assert await capabilities.native_tools(OPENROUTER, model) == capabilities.YES
    assert seen == ["GET /api/v1/models"]


@pytest.mark.asyncio
async def test_lmstudio_off_on_model_is_binary_with_every_level_selectable(monkeypatch):
    serve_probes(monkeypatch, {"GET /api/v1/models": LMSTUDIO_MODELS})

    levels = await capabilities.thinking_levels(LMSTUDIO, "ternary-bonsai-2-27b-ptq1_0-mtp")

    assert levels.support == _support(S, S, S, S, S, S)
    assert levels.binary is True


@pytest.mark.asyncio
async def test_lmstudio_model_without_reasoning_config_reports_nothing(monkeypatch):
    serve_probes(monkeypatch, {"GET /api/v1/models": LMSTUDIO_MODELS})

    levels = await capabilities.thinking_levels(LMSTUDIO, "jev-style-qwen3.5-2b-decision")

    assert levels.support == _support(N, N, N, N, N, N)
    assert levels.binary is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "routes"),
    [
        ("https://api.openai.com/v1", {}),
        # llama.cpp levels come from the /apply-template probe (#97), not /props.
        (LLAMA, {"GET /props": BONSAI_PROPS}),
    ],
    ids=["openai", "llama-props"],
)
async def test_servers_reporting_no_levels_leave_every_level_selectable(monkeypatch, url, routes):
    serve_probes(monkeypatch, routes)

    levels = await capabilities.thinking_levels(url, BONSAI)

    assert levels.support == _support(N, N, N, N, N, N)
    assert levels.binary is False


# --- Payload per dialect, through a new-runtime turn on the chat route -------

THINKING_KEYS = ("reasoning", "reasoning_effort", "thinking", "chat_template_kwargs")
DEEPSEEK = "https://api.deepseek.com/v1"
XAI_URL = "https://api.x.ai/v1"
XAI_ROUND = (FIXTURES / "xai_responses_round2_text.sse").read_bytes()


@pytest.fixture
def harness(monkeypatch, tmp_path):
    from tests.chat_agent.test_chat_agent_route import _Harness

    harness = _Harness(monkeypatch, tmp_path)
    harness.session.headers = {}
    # Legacy's session switch; the new runtime must not read it.
    harness.session.thinking_mode = "off"
    return harness


async def _turn_payload(harness, monkeypatch, url, model, level, routes=None):
    from src.chat_agent import store
    from src.chat_agent.providers import xai_responses
    from tests.chat_agent.test_chat_agent_route import _Request, _sse_events

    import httpx

    store.set_chat_runtime("alice", store.RUNTIME_NEW)
    if level is not None:
        store.set_chat_settings("alice", "session-1", {store.THINKING_LEVEL: level})
    harness.session.endpoint_url, harness.session.model = url, model

    def chat(request):
        harness.captured["provider_requests"].append(json.loads(request.content))
        return httpx.Response(200, content=_sse_events(["Hi"]), headers={"content-type": "text/event-stream"})

    serve_probes(monkeypatch, routes or {}, chat=chat)

    def responses(request):
        harness.captured["provider_requests"].append(json.loads(request.content))
        return httpx.Response(200, content=XAI_ROUND, headers={"content-type": "text/event-stream"})

    xai_client = httpx.AsyncClient(transport=httpx.MockTransport(responses))
    monkeypatch.setattr(xai_responses, "_get_http_client", lambda: xai_client)

    response, _ = await harness.send(_Request("agent"))
    assert response.headers["X-Odysseus-Runtime"] == "chat_agent"
    [payload] = harness.captured["provider_requests"]
    return {key: payload[key] for key in THINKING_KEYS if key in payload}


OPENROUTER_ROUTES = {"GET /api/v1/models": OPENROUTER_MODELS}
LMSTUDIO_ROUTES = {"GET /api/v1/models": LMSTUDIO_MODELS}
LLAMA_ROUTES = {"GET /props": BONSAI_PROPS}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "model", "routes", "level", "sent"),
    [
        # Auto sends nothing: no forced OpenRouter effort none, no empty -> off.
        (OPENROUTER, "cohere/command-a-plus", OPENROUTER_ROUTES, None, {}),
        (OPENROUTER, "cohere/command-a-plus", OPENROUTER_ROUTES, "auto", {}),
        (LLAMA, BONSAI, LLAMA_ROUTES, "auto", {}),
        # OpenRouter: reasoning.effort.
        (OPENROUTER, "cohere/command-a-plus", OPENROUTER_ROUTES, "high", {"reasoning": {"effort": "high"}}),
        (OPENROUTER, "cohere/command-a-plus", OPENROUTER_ROUTES, "off", {"reasoning": {"effort": "none"}}),
        # A level the server reports unsupported is never sent.
        (OPENROUTER, "stepfun/step-5-preview", OPENROUTER_ROUTES, "off", {}),
        (OPENROUTER, "stepfun/step-5-preview", OPENROUTER_ROUTES, "max", {}),
        # Chat Completions: reasoning_effort, Off "none"; no enable_thinking kwarg.
        (LLAMA, BONSAI, LLAMA_ROUTES, "xhigh", {"reasoning_effort": "xhigh"}),
        (LLAMA, BONSAI, LLAMA_ROUTES, "off", {"reasoning_effort": "none"}),
        ("https://llm.example/v1", "qwen3-8b", {}, "max", {"reasoning_effort": "max"}),
        # Binary off/on (LM Studio): every graded level is "on" = medium.
        (LMSTUDIO, "ternary-bonsai-2-27b-ptq1_0-mtp", LMSTUDIO_ROUTES, "max", {"reasoning_effort": "medium"}),
        (LMSTUDIO, "ternary-bonsai-2-27b-ptq1_0-mtp", LMSTUDIO_ROUTES, "off", {"reasoning_effort": "none"}),
        # DeepSeek: Off is thinking.type disabled.
        (DEEPSEEK, "deepseek-flash", {}, "high", {"reasoning_effort": "high"}),
        (DEEPSEEK, "deepseek-flash", {}, "off", {"thinking": {"type": "disabled"}}),
        # xAI Responses: reasoning.effort.
        (XAI_URL, "grok-4.7", {}, "xhigh", {"reasoning": {"effort": "xhigh"}}),
        (XAI_URL, "grok-4.7", {}, "off", {"reasoning": {"effort": "none"}}),
    ],
)
async def test_thinking_level_payload_per_dialect(harness, monkeypatch, url, model, routes, level, sent):
    assert await _turn_payload(harness, monkeypatch, url, model, level, routes) == sent


# --- Settings and capabilities routes, over HTTP ----------------------------

ANTHROPIC = "https://api.anthropic.com/v1"


@pytest.fixture
def api(monkeypatch, tmp_path):
    """The chat_agent router behind a stand-in auth middleware (X-User header);
    the real owner check runs against a session row owned by alice."""
    from types import SimpleNamespace

    import httpx
    from fastapi import FastAPI

    import routes.session_routes as session_routes
    import src.constants as constants
    from routes.chat_agent_routes import setup_chat_agent_routes
    from tests.chat_agent.test_chat_agent_ask_steer_authority import _Db

    monkeypatch.setattr(constants, "CHAT_AGENT_DIR", str(tmp_path / "chat_agent"))
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setattr(session_routes, "SessionLocal", _Db)
    session = SimpleNamespace(endpoint_url=LLAMA, model=BONSAI, headers={})
    app = FastAPI()

    @app.middleware("http")
    async def user(request, call_next):
        request.state.current_user = request.headers.get("X-User", "alice")
        return await call_next(request)

    app.include_router(setup_chat_agent_routes(SimpleNamespace(get_session=lambda session_id: session)))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://odysseus")
    client.session = session
    return client


async def _level(api, chat, user="alice"):
    response = await api.get("/api/chat_agent/settings", params={"session_id": chat}, headers={"X-User": user})
    assert response.status_code == 200
    return response.json()["thinking_level"]


async def _set_level(api, chat, level):
    return await api.put("/api/chat_agent/settings", json={"session_id": chat, "thinking_level": level})


@pytest.mark.asyncio
async def test_level_is_stored_per_chat_and_new_chats_start_with_the_last_used(api):
    assert await _level(api, "chat-1") == "auto"

    response = await _set_level(api, "chat-1", "high")
    assert response.status_code == 200
    assert response.json()["thinking_level"] == "high"

    assert await _level(api, "chat-2") == "high"
    assert (await _set_level(api, "chat-2", "low")).status_code == 200
    assert await _level(api, "chat-1") == "high"
    assert await _level(api, "chat-3") == "low"
    assert await _level(api, "chat-1", user="bob") == "auto"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"session_id": "chat-1", "thinking_level": "on"},
        {"session_id": "chat-1", "thinking_level": "minimal"},
        {},
    ],
    ids=["legacy-on", "minimal", "empty"],
)
async def test_settings_write_validates_the_level(api, body):
    response = await api.put("/api/chat_agent/settings", json=body)

    assert response.status_code == 400
    assert await _level(api, "chat-1") == "auto"


@pytest.mark.asyncio
async def test_runtime_switch_write_still_works_alone(api):
    response = await api.put("/api/chat_agent/settings", json={"chat_runtime": "new"})

    assert response.status_code == 200
    assert response.json() == {"chat_runtime": "new", "available": True,
                               "thinking_level": "auto", "context_overflow": "auto"}


async def _capabilities(api, chat="chat-1", user="alice"):
    return await api.get(f"/api/chat_agent/capabilities/{chat}", headers={"X-User": user})


def _switch_on():
    from src.chat_agent import store

    store.set_chat_runtime("alice", store.RUNTIME_NEW)


@pytest.mark.asyncio
async def test_capabilities_report_new_runtime_and_levels_for_the_chats_model(api, monkeypatch):
    _switch_on()
    seen = serve_probes(monkeypatch, LLAMA_ROUTES)

    response = await _capabilities(api)

    assert response.status_code == 200
    assert response.json() == {
        "route": "chat_agent", "reason": "",
        "thinking": {"available": True, "binary": False, "levels": _support(N, N, N, N, N, N)},
    }
    # Route and levels share one probe.
    assert seen.count(f"GET /props?model={BONSAI}") == 1


@pytest.mark.asyncio
async def test_capabilities_report_binary_and_unsupported_levels(api, monkeypatch):
    _switch_on()
    serve_probes(monkeypatch, {"GET /api/v1/models": OPENROUTER_MODELS})
    api.session.endpoint_url, api.session.model = OPENROUTER, "stepfun/step-5-preview"

    thinking = (await _capabilities(api)).json()["thinking"]

    assert thinking == {"available": True, "binary": False, "levels": _support(U, S, S, S, U, U)}

    serve_probes(monkeypatch, LMSTUDIO_ROUTES)
    api.session.endpoint_url, api.session.model = LMSTUDIO, "ternary-bonsai-2-27b-ptq1_0-mtp"

    assert (await _capabilities(api)).json()["thinking"]["binary"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("switch_on", "url", "routes", "reason", "available"),
    [
        (False, LLAMA, LLAMA_ROUTES, "runtime_off", True),
        (True, LLAMA, {"GET /props": {**BONSAI_PROPS, "chat_template_caps": {"supports_tools": False}}},
         "no_native_tools", True),
        # No adapter for the dialect: no level can be carried, nothing is probed.
        (True, ANTHROPIC, {}, "no_adapter", False),
    ],
    ids=["switch-off", "no-native-tools", "anthropic"],
)
async def test_capabilities_report_legacy_with_the_reason(api, monkeypatch, switch_on, url, routes, reason, available):
    if switch_on:
        _switch_on()
    seen = serve_probes(monkeypatch, routes)
    api.session.endpoint_url = url

    body = (await _capabilities(api)).json()

    assert (body["route"], body["reason"], body["thinking"]["available"]) == ("legacy", reason, available)
    if not available:
        assert seen == []


@pytest.mark.asyncio
async def test_capabilities_of_another_users_chat_are_not_found(api, monkeypatch):
    seen = serve_probes(monkeypatch, LLAMA_ROUTES)

    response = await _capabilities(api, user="bob")

    assert response.status_code == 404
    assert seen == []
