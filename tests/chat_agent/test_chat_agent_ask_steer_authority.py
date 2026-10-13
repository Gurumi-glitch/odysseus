"""Who may answer an ask or steer a turn: every caller class, exact status and detail.

Pins the ask/steer column of the outcome table in
docs/spec/chat-agent-arch/issues/05-ask-steer-endpoints-move.md. The checks run
for real (auth helpers, owner check); only the session lookup is stubbed.
"""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.session_routes as session_routes
from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
from routes.chat_agent_routes import setup_chat_agent_routes
from src.chat_agent import asks

ASK = "/api/chat/ask/{session_id}"
STEER = "/api/chat/steer/{session_id}"
SESSION = "session-1"

DENIED = {
    ASK: "Only you, in an interactive session, can answer an ask.",
    STEER: "Only you, in an interactive session, can steer a turn.",
}
NO_MATCH = {
    ASK: "No open ask matches this answer.",
    STEER: "No running turn in this chat takes steering messages.",
}
BODY = {
    ASK: {"ask_id": "ask-1", "decision": "once"},
    STEER: {"text": "use -la"},
}


class _Query:
    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return SimpleNamespace(owner="alice")


class _Db:
    def query(self, *args, **kwargs):
        return _Query()

    def close(self):
        return None


def _routes():
    session_manager = SimpleNamespace(get_session=lambda session_id: SimpleNamespace(model="m"))
    router = setup_chat_agent_routes(session_manager)
    return {route.path: route.endpoint for route in router.routes}


def _request(user="alice", *, internal=False, bearer=False):
    state = SimpleNamespace(current_user=user)
    if bearer:
        state.api_token = True
        state.api_token_owner = "alice"
    return SimpleNamespace(
        headers={INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN} if internal else {},
        state=state,
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
        client=SimpleNamespace(host="203.0.113.9"),
    )


@pytest.fixture(autouse=True)
def _session_owned_by_alice(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setattr(session_routes, "SessionLocal", _Db)


async def _refusal(path, request):
    with pytest.raises(HTTPException) as refused:
        await _routes()[path](request, SESSION, dict(BODY[path]))
    return refused.value.status_code, refused.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [ASK, STEER])
async def test_internal_loopback_is_refused(path):
    assert await _refusal(path, _request(internal=True)) == (403, DENIED[path])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [ASK, STEER])
async def test_a_bearer_token_is_refused_even_for_the_owner(path):
    assert await _refusal(path, _request(bearer=True)) == (403, DENIED[path])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [ASK, STEER])
async def test_internal_and_bearer_are_still_refused_with_auth_disabled(path, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")

    assert await _refusal(path, _request(None, internal=True)) == (403, DENIED[path])
    assert await _refusal(path, _request(None, bearer=True)) == (403, DENIED[path])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [ASK, STEER])
async def test_other_callers_pass_the_check_with_auth_disabled(path, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")

    # Past the interactive check and the owner check: the registry answers.
    assert await _refusal(path, _request(None)) == (409, NO_MATCH[path])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [ASK, STEER])
async def test_no_user_with_auth_on_passes_the_check_then_fails_the_owner_check(path):
    assert await _refusal(path, _request(None)) == (401, "Authentication required")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [ASK, STEER])
async def test_another_owner_does_not_see_the_session(path):
    assert await _refusal(path, _request("bob")) == (404, f"Session {SESSION} not found")


@pytest.mark.asyncio
async def test_a_signed_in_owner_answers_an_open_ask():
    card = {"ask_id": "ask-1", "ask_kind": asks.PERMISSION}
    waiting = asyncio.ensure_future(asks.wait(SESSION, "alice", card))
    await asyncio.sleep(0)

    result = await _routes()[ASK](_request(), SESSION, dict(BODY[ASK]))

    assert result == {"accepted": True}
    assert await waiting == asks.Answer(asks.ONCE, "")


@pytest.mark.asyncio
async def test_a_signed_in_owner_steers_a_running_turn():
    queue = asks.SteerQueue(SESSION, "alice")
    try:
        result = await _routes()[STEER](_request(), SESSION, dict(BODY[STEER]))
    finally:
        taken = queue.close()

    assert result == {"accepted": True}
    assert taken == ["use -la"]
