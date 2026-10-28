"""A new chat before its first message (no session yet, issue #103).

Over HTTP: the capabilities route keyed by the picker's (endpoint, model),
with the chat route's endpoint visibility, and the settings route without a
session_id reading and writing the owner's last-used values.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core import database
from tests.chat_agent.test_chat_agent_capabilities import (
    BONSAI,
    BONSAI_PROPS,
    CODEX,
    CODEX_CATALOG,
    CODEX_LEVELS,
    CODEX_MODELS,
    LLAMA,
    _fresh_caches,  # noqa: F401  (autouse: clean capability caches)
    serve_probes,
)

LLAMA_ROUTES = {"GET /props": BONSAI_PROPS}
UNKNOWN = {level: "unknown" for level in ("off", "low", "medium", "high", "xhigh", "max")}


@pytest.fixture
def api(monkeypatch, tmp_path):
    """The chat_agent router behind a stand-in auth middleware (X-User header),
    over a real endpoint table: alice's, bob's, a shared and a disabled row."""
    from types import SimpleNamespace

    import httpx
    from fastapi import FastAPI

    import routes.chat_routes as chat_routes
    import src.constants as constants
    import src.endpoint_resolver as endpoint_resolver
    from routes.chat_agent_routes import setup_chat_agent_routes

    monkeypatch.setattr(constants, "CHAT_AGENT_DIR", str(tmp_path / "chat_agent"))
    monkeypatch.setenv("AUTH_ENABLED", "true")
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(chat_routes, "SessionLocal", factory)
    monkeypatch.setattr(endpoint_resolver, "SessionLocal", factory)
    with factory() as db:
        for endpoint_id, owner, base, enabled in [
            ("alice-llama", "alice", LLAMA, True),
            ("bob-llama", "bob", "http://localhost:1236/v1", True),
            ("shared-llama", None, "http://localhost:1237/v1", True),
            ("alice-off", "alice", "http://localhost:1238/v1", False),
        ]:
            db.add(database.ModelEndpoint(id=endpoint_id, name=endpoint_id, owner=owner,
                                          base_url=base, is_enabled=enabled))
        # Its bearer lives in a ProviderAuthSession, never in api_key.
        db.add(database.ModelEndpoint(id="alice-chatgpt", name="ChatGPT", owner="alice", base_url=CODEX,
                                      provider_auth_id="auth-1", is_enabled=True))
        db.commit()

    app = FastAPI()

    @app.middleware("http")
    async def user(request, call_next):
        request.state.current_user = request.headers.get("X-User", "alice")
        return await call_next(request)

    app.include_router(setup_chat_agent_routes(SimpleNamespace(get_session=lambda session_id: None)))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://odysseus")


def _switch_on(owner="alice"):
    from src.chat_agent import store

    store.set_chat_runtime(owner, store.RUNTIME_NEW)


async def _capabilities(api, user="alice", **params):
    return await api.get("/api/chat_agent/capabilities", params={"model": BONSAI, **params},
                         headers={"X-User": user})


# --- Capabilities by (endpoint, model) -------------------------------------------


@pytest.mark.asyncio
async def test_pending_chat_capabilities_answer_for_the_pickers_endpoint_and_model(api, monkeypatch):
    _switch_on()
    seen = serve_probes(monkeypatch, LLAMA_ROUTES)

    response = await _capabilities(api, endpoint_id="alice-llama", endpoint_url=LLAMA)

    assert response.status_code == 200
    assert response.json() == {
        "route": "chat_agent", "reason": "",
        "thinking": {"available": True, "binary": False, "levels": UNKNOWN},
    }
    # The picked endpoint's server was asked, once for route and levels.
    assert seen.count(f"GET /props?model={BONSAI}") == 1


@pytest.mark.asyncio
async def test_pending_chat_capabilities_resolve_a_url_only_pick(api, monkeypatch):
    _switch_on()
    serve_probes(monkeypatch, LLAMA_ROUTES)

    response = await _capabilities(api, endpoint_url=LLAMA + "/chat/completions")

    assert response.status_code == 200
    assert response.json()["route"] == "chat_agent"


@pytest.mark.asyncio
async def test_pending_chat_capabilities_follow_the_runtime_switch(api, monkeypatch):
    serve_probes(monkeypatch, LLAMA_ROUTES)

    body = (await _capabilities(api, endpoint_id="alice-llama")).json()

    assert (body["route"], body["reason"]) == ("legacy", "runtime_off")


@pytest.mark.asyncio
async def test_a_shared_endpoint_is_visible(api, monkeypatch):
    _switch_on()
    serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS})

    response = await _capabilities(api, endpoint_id="shared-llama")

    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"endpoint_id": "bob-llama"},
        {"endpoint_url": "http://localhost:1236/v1"},
        {"endpoint_id": "alice-off"},
        {"endpoint_id": "no-such-endpoint"},
        # A URL that is not a registered endpoint is never probed.
        {"endpoint_url": "http://169.254.169.254/v1"},
        {},
    ],
    ids=["other-users-id", "other-users-url", "disabled", "unknown-id", "unregistered-url", "no-endpoint"],
)
async def test_endpoints_the_owner_cannot_chat_with_are_not_found(api, monkeypatch, params):
    _switch_on()
    seen = serve_probes(monkeypatch, LLAMA_ROUTES)

    response = await _capabilities(api, **params)

    assert response.status_code == 404
    assert seen == []


def _codex_catalog(monkeypatch):
    """Serve the catalog only to a bearer, as the Codex backend does; returns the requests."""
    import httpx

    requests = []

    def catalog(request):
        requests.append(request)
        if "Authorization" not in request.headers:
            return httpx.Response(401, json={"detail": "Unauthorized"})
        return httpx.Response(200, json=CODEX_MODELS)

    serve_probes(monkeypatch, {CODEX_CATALOG: catalog})
    return requests


@pytest.mark.asyncio
async def test_pending_chatgpt_pick_probes_the_catalog_with_a_fresh_bearer(api, monkeypatch):
    import src.endpoint_resolver as endpoint_resolver

    _switch_on()
    requests = _codex_catalog(monkeypatch)
    monkeypatch.setattr(endpoint_resolver, "resolve_endpoint_runtime",
                        lambda ep, owner=None: (CODEX, "live-access-token"))

    response = await _capabilities(api, model="gpt-6.1-sol", endpoint_id="alice-chatgpt")

    assert response.status_code == 200
    assert response.json()["thinking"]["levels"] == CODEX_LEVELS
    [request] = requests
    assert request.headers["Authorization"] == "Bearer live-access-token"


@pytest.mark.asyncio
async def test_pending_chatgpt_pick_with_a_failed_refresh_leaves_levels_unknown(api, monkeypatch):
    import src.endpoint_resolver as endpoint_resolver
    from src.chatgpt_subscription import ChatGPTSubscriptionReauthRequired

    _switch_on()
    requests = _codex_catalog(monkeypatch)

    def refresh_fails(ep, owner=None):
        raise ChatGPTSubscriptionReauthRequired("ChatGPT Subscription token refresh failed with HTTP 401.")

    monkeypatch.setattr(endpoint_resolver, "resolve_endpoint_runtime", refresh_fails)

    response = await _capabilities(api, model="gpt-6.1-sol", endpoint_id="alice-chatgpt")

    assert response.status_code == 200
    assert response.json()["thinking"]["levels"] == UNKNOWN
    # Asked without a bearer, refused, and never a sign-in prompt.
    assert requests and all("Authorization" not in request.headers for request in requests)


# --- Settings without a session ----------------------------------------------------


async def _settings(api, **params):
    response = await api.get("/api/chat_agent/settings", params=params)
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
async def test_pending_chat_reads_and_writes_the_owners_last_used_values(api):
    before = await _settings(api)
    assert (before["thinking_level"], before["context_overflow"]) == ("auto", "auto")

    response = await api.put("/api/chat_agent/settings",
                             json={"thinking_level": "off", "context_overflow": "rolling_window"})

    assert response.status_code == 200
    assert (response.json()["thinking_level"], response.json()["context_overflow"]) == ("off", "rolling_window")
    # The chat the first message creates inherits them.
    created = await _settings(api, session_id="chat-new")
    assert (created["thinking_level"], created["context_overflow"]) == ("off", "rolling_window")


@pytest.mark.asyncio
async def test_pending_chat_write_leaves_chats_with_their_own_value_alone(api):
    await api.put("/api/chat_agent/settings", json={"session_id": "chat-1", "thinking_level": "high"})

    await api.put("/api/chat_agent/settings", json={"thinking_level": "low"})

    assert (await _settings(api, session_id="chat-1"))["thinking_level"] == "high"
    assert (await _settings(api))["thinking_level"] == "low"


@pytest.mark.asyncio
async def test_pending_chat_write_still_validates(api):
    response = await api.put("/api/chat_agent/settings", json={"thinking_level": "on"})

    assert response.status_code == 400
    assert (await _settings(api))["thinking_level"] == "auto"
