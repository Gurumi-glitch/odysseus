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
    from routes.chat_agent_routes import setup_chat_agent_routes

    monkeypatch.setattr(constants, "CHAT_AGENT_DIR", str(tmp_path / "chat_agent"))
    monkeypatch.setenv("AUTH_ENABLED", "true")
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(chat_routes, "SessionLocal", factory)
    with factory() as db:
        for endpoint_id, owner, base, enabled in [
            ("alice-llama", "alice", LLAMA, True),
            ("bob-llama", "bob", "http://localhost:1236/v1", True),
            ("shared-llama", None, "http://localhost:1237/v1", True),
            ("alice-off", "alice", "http://localhost:1238/v1", False),
        ]:
            db.add(database.ModelEndpoint(id=endpoint_id, name=endpoint_id, owner=owner,
                                          base_url=base, is_enabled=enabled))
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
