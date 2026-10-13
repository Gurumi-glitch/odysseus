"""The runtime switch lives in the chat_agent store; only an interactive user may flip it."""

import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import src.constants as constants
from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
from routes.chat_agent_routes import setup_chat_agent_routes
from src.chat_agent import store


@pytest.fixture(autouse=True)
def _store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "CHAT_AGENT_DIR", str(tmp_path / "chat_agent"))
    monkeypatch.setenv("AUTH_ENABLED", "true")
    return tmp_path / "chat_agent"


def _request(user="alice", *, headers=None, api_token=False):
    state = SimpleNamespace(current_user=user)
    if api_token:
        state.api_token = True
        state.api_token_owner = user
    return SimpleNamespace(
        headers=headers or {},
        state=state,
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
        client=SimpleNamespace(host="203.0.113.9"),
    )


def _endpoint(method):
    router = setup_chat_agent_routes(SimpleNamespace())
    return next(
        route.endpoint for route in router.routes
        if route.path == "/api/chat_agent/settings" and method in route.methods
    )


@pytest.mark.asyncio
async def test_switch_is_off_by_default_and_persists_per_user(_store_dir):
    get_settings, put_settings = _endpoint("GET"), _endpoint("PUT")

    assert await get_settings(_request()) == {"chat_runtime": "legacy", "available": True}

    assert await put_settings(_request(), {"chat_runtime": "new"}) == {
        "chat_runtime": "new", "available": True,
    }
    assert await get_settings(_request()) == {"chat_runtime": "new", "available": True}
    assert await get_settings(_request("bob")) == {"chat_runtime": "legacy", "available": True}
    assert os.listdir(_store_dir) == ["alice.json"]


@pytest.mark.asyncio
async def test_store_file_name_is_percent_encoded_and_confined(_store_dir):
    put_settings = _endpoint("PUT")
    await put_settings(_request("../Evil/User"), {"chat_runtime": "new"})

    assert os.listdir(_store_dir) == ["..%2Fevil%2Fuser.json"]
    assert store.chat_runtime("../evil/USER") == "new"


@pytest.mark.asyncio
async def test_switch_rejects_unknown_value():
    with pytest.raises(HTTPException) as exc:
        await _endpoint("PUT")(_request(), {"chat_runtime": "agent_runtime"})
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_factory",
    [
        lambda: _request(headers={INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN}),
        lambda: _request("internal-tool"),
        lambda: _request(api_token=True),
    ],
    ids=["internal-tool-header", "internal-tool-user", "delegated-token"],
)
async def test_switch_write_refuses_non_interactive_callers(request_factory, _store_dir):
    with pytest.raises(HTTPException) as exc:
        await _endpoint("PUT")(request_factory(), {"chat_runtime": "new"})
    assert exc.value.status_code == 403
    assert not _store_dir.exists()


@pytest.mark.asyncio
async def test_switch_is_unavailable_when_auth_is_disabled(monkeypatch, _store_dir):
    monkeypatch.setenv("AUTH_ENABLED", "false")

    assert await _endpoint("GET")(_request(None)) == {"chat_runtime": "legacy", "available": False}
    with pytest.raises(HTTPException) as exc:
        await _endpoint("PUT")(_request(None), {"chat_runtime": "new"})
    assert exc.value.status_code == 403
    assert not _store_dir.exists()


def _rules_endpoint(method):
    router = setup_chat_agent_routes(SimpleNamespace())
    return next(
        route.endpoint for route in router.routes
        if route.path == "/api/chat_agent/rules" and method in route.methods
    )


@pytest.mark.asyncio
async def test_rules_are_saved_normalized_per_user():
    saved = await _rules_endpoint("PUT")(_request(), {"rules": {
        "allow": ["bash", "mcp__email__send_email", "bash"],
        "deny": ["web_fetch(domain:Evil.com)"],
    }})

    assert saved == {"available": True, "rules": {
        "allow": ["bash", "send_email"], "ask": [], "deny": ["web_fetch(domain:evil.com)"],
    }}
    assert await _rules_endpoint("GET")(_request()) == saved
    assert (await _rules_endpoint("GET")(_request("bob")))["rules"] == {"allow": [], "ask": [], "deny": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {"rules": {"allow": ["web_search(domain:x.com)"]}},
    {"rules": {"allow": "bash"}},
    {"rules": {"always": ["bash"]}},
    {"rules": ["bash"]},
])
async def test_rules_outside_the_table_are_refused_and_nothing_is_written(body, _store_dir):
    with pytest.raises(HTTPException) as exc:
        await _rules_endpoint("PUT")(_request(), body)
    assert exc.value.status_code == 400
    assert not _store_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("rule", ["bash(git:*)", "python(print(1))", "host_shell(docker ps)"])
async def test_shell_command_specifiers_are_refused_with_the_bare_name_to_use(rule, _store_dir):
    with pytest.raises(HTTPException) as exc:
        await _rules_endpoint("PUT")(_request(), {"rules": {"deny": [rule]}})
    assert exc.value.status_code == 400
    assert "not supported yet" in exc.value.detail
    assert f"bare name {rule.split('(')[0]}" in exc.value.detail
    assert not _store_dir.exists()


@pytest.mark.asyncio
async def test_rules_write_refuses_tool_loopback(_store_dir):
    with pytest.raises(HTTPException) as exc:
        await _rules_endpoint("PUT")(
            _request(headers={INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN}), {"rules": {"allow": ["bash"]}},
        )
    assert exc.value.status_code == 403
    assert not _store_dir.exists()


def test_flipping_the_runtime_drops_session_rules_and_keeps_user_rules():
    store.set_chat_runtime("alice", store.RUNTIME_NEW)
    store.set_user_rules("alice", {"allow": ["bash"]})
    store.add_session_rule("alice", "s1", ["bash", "any", ""])

    store.set_chat_runtime("alice", store.RUNTIME_NEW)
    assert store.session_rules("alice", "s1") == [["bash", "any", ""]]
    store.set_chat_runtime("alice", store.RUNTIME_LEGACY)

    assert store.session_rules("alice", "s1") == []
    assert store.user_rules("alice")["allow"] == ["bash"]


def test_deleting_a_user_removes_their_rules():
    store.set_user_rules("alice", {"allow": ["bash"]})
    store.delete_owner("alice")
    assert store.user_rules("alice") == {"allow": [], "ask": [], "deny": []}
