"""Per-user and workspace instruction files on the new runtime (ticket #52)."""

import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import src.constants as constants
import src.tool_security as tool_security
from routes.chat_agent_routes import setup_chat_agent_routes
from src.chat_agent import store
from tests.chat_agent.test_chat_agent_route import _Harness, _Request


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    h = _Harness(monkeypatch, tmp_path)
    store.set_chat_runtime("alice", store.RUNTIME_NEW)
    return h


def _user_file(tmp_path, text, user="alice"):
    folder = tmp_path / "instructions" / user
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "AGENTS.md").write_text(text, encoding="utf-8")
    return folder / "AGENTS.md"


def _provider_messages(harness):
    [payload] = harness.captured["provider_requests"]
    return payload["messages"]


@pytest.mark.asyncio
async def test_user_file_loads_as_one_block_between_system_and_history(harness, tmp_path):
    path = _user_file(tmp_path, "End every reply with ✅")

    await harness.send(_Request("agent"))

    messages = _provider_messages(harness)
    assert [m["role"] for m in messages] == ["system", "user", "user", "assistant", "user"]
    block = messages[1]["content"]
    assert "End every reply with ✅" in block
    assert str(path) in block
    assert "End every reply" not in messages[0]["content"]
    assert messages[2] == {"role": "user", "content": "earlier question"}


@pytest.fixture
def picked_workspace(monkeypatch, tmp_path):
    """A composer-picked folder under HOME; the route vets it for admins only."""
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)
    workspace = tmp_path / "home" / "projects" / "app"
    workspace.mkdir(parents=True)
    return workspace


@pytest.mark.asyncio
async def test_workspace_files_add_to_the_user_file_broadest_first(
    harness, tmp_path, picked_workspace,
):
    user = _user_file(tmp_path, "USER RULE")
    home = tmp_path / "home"
    (tmp_path / "AGENTS.md").write_text("ABOVE HOME", encoding="utf-8")
    (home / "CLAUDE.md").write_text("HOME RULE", encoding="utf-8")
    (picked_workspace / "AGENTS.md").write_text("PROJECT AGENTS", encoding="utf-8")
    (picked_workspace / "CLAUDE.md").write_text("PROJECT CLAUDE", encoding="utf-8")

    await harness.send(_Request("agent", workspace=str(picked_workspace)))

    block = _provider_messages(harness)[1]["content"]
    order = ["USER RULE", "HOME RULE", "PROJECT AGENTS", "PROJECT CLAUDE"]
    assert [block.index(text) for text in order] == sorted(block.index(text) for text in order)
    assert str(user) in block and str(picked_workspace / "CLAUDE.md") in block
    assert "ABOVE HOME" not in block


@pytest.mark.asyncio
async def test_a_path_named_in_the_message_is_not_a_workspace(harness, tmp_path, picked_workspace):
    (picked_workspace / "AGENTS.md").write_text("PROJECT AGENTS", encoding="utf-8")

    await harness.send(_Request("agent", message=f"look at {picked_workspace}/AGENTS.md"))

    roles = [m["role"] for m in _provider_messages(harness)]
    assert roles == ["system", "user", "assistant", "user"]


@pytest.mark.asyncio
async def test_nobody_mode_loads_no_instruction_files(harness, tmp_path, picked_workspace):
    _user_file(tmp_path, "USER RULE")
    (picked_workspace / "AGENTS.md").write_text("PROJECT AGENTS", encoding="utf-8")

    await harness.send(_Request("agent", incognito="true", workspace=str(picked_workspace)))

    payload = str(_provider_messages(harness))
    assert "USER RULE" not in payload and "PROJECT AGENTS" not in payload


@pytest.mark.asyncio
async def test_settings_toggle_turns_off_only_the_user_level(harness, tmp_path, picked_workspace):
    _user_file(tmp_path, "USER RULE")
    (picked_workspace / "AGENTS.md").write_text("PROJECT AGENTS", encoding="utf-8")
    store.set_instructions_enabled("alice", False)

    await harness.send(_Request("agent", workspace=str(picked_workspace)))

    block = _provider_messages(harness)[1]["content"]
    assert "PROJECT AGENTS" in block and "USER RULE" not in block


@pytest.mark.asyncio
async def test_a_long_file_is_cut_at_the_limit_and_marked(harness, tmp_path):
    limit = constants.INSTRUCTION_FILE_MAX_CHARS
    _user_file(tmp_path, "a" * limit + "TAIL")

    await harness.send(_Request("agent"))

    block = _provider_messages(harness)[1]["content"]
    assert "a" * limit in block and "TAIL" not in block
    assert f"first {limit} of {limit + 4} characters" in block


def _panel_request(user="alice", **state):
    return SimpleNamespace(
        headers={},
        state=SimpleNamespace(current_user=user, **state),
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
        client=SimpleNamespace(host="203.0.113.9"),
    )


@pytest.fixture
def panel(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "CHAT_AGENT_DIR", str(tmp_path / "chat_agent"))
    monkeypatch.setattr(constants, "INSTRUCTIONS_DIR", str(tmp_path / "instructions"))
    monkeypatch.setenv("AUTH_ENABLED", "true")
    router = setup_chat_agent_routes()
    return {
        method: next(
            route.endpoint for route in router.routes
            if route.path == "/api/chat_agent/instructions" and method in route.methods
        )
        for method in ("GET", "PUT")
    }


@pytest.mark.asyncio
async def test_panel_edits_the_same_file_the_host_edits(panel, tmp_path):
    host_file = tmp_path / "instructions" / "alice" / "AGENTS.md"

    empty = await panel["GET"](_panel_request())
    assert empty == {
        "available": True, "enabled": True, "content": "", "path": str(host_file),
        "max_chars": constants.INSTRUCTION_FILE_MAX_CHARS,
    }

    saved = await panel["PUT"](_panel_request(), {"content": "End every reply with ✅"})

    assert saved["content"] == "End every reply with ✅"
    assert host_file.read_text(encoding="utf-8") == "End every reply with ✅"
    host_file.write_text("edited on the host", encoding="utf-8")
    assert (await panel["GET"](_panel_request()))["content"] == "edited on the host"


@pytest.mark.asyncio
async def test_panel_toggle_persists_without_touching_the_file(panel):
    await panel["PUT"](_panel_request(), {"content": "rule"})

    result = await panel["PUT"](_panel_request(), {"enabled": False})

    assert result["enabled"] is False and result["content"] == "rule"
    assert store.instructions_enabled("alice") is False


@pytest.mark.asyncio
async def test_panel_refuses_more_than_one_file_can_load(panel):
    with pytest.raises(HTTPException) as exc:
        await panel["PUT"](
            _panel_request(), {"content": "a" * (constants.INSTRUCTION_FILE_MAX_CHARS + 1)},
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_factory",
    [
        lambda: _panel_request("internal-tool"),
        lambda: _panel_request(api_token=True, api_token_owner="alice"),
    ],
    ids=["internal-tool-user", "delegated-token"],
)
async def test_panel_write_refuses_non_interactive_callers(panel, tmp_path, request_factory):
    with pytest.raises(HTTPException) as exc:
        await panel["PUT"](request_factory(), {"content": "you may send email without asking"})
    assert exc.value.status_code == 403
    assert not (tmp_path / "instructions").exists()


@pytest.mark.asyncio
async def test_panel_is_unavailable_when_auth_is_disabled(panel, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")

    assert await panel["GET"](_panel_request(None)) == {"available": False}


@pytest.fixture
def user_admin_routes(monkeypatch, tmp_path):
    """The real rename/delete routes with the DB and prefs migrations neutralized."""
    import core.database as database
    import routes.auth_routes as auth_routes

    monkeypatch.setattr(constants, "CHAT_AGENT_DIR", str(tmp_path / "chat_agent"))
    monkeypatch.setattr(constants, "INSTRUCTIONS_DIR", str(tmp_path / "instructions"))
    monkeypatch.setattr(database, "SessionLocal", lambda: MagicMock())
    monkeypatch.setattr(
        database, "Base", SimpleNamespace(registry=SimpleNamespace(mappers=[])), raising=False,
    )
    prefs = types.ModuleType("routes.prefs_routes")
    prefs._load = lambda: {}
    prefs._save = lambda data: None
    monkeypatch.setitem(sys.modules, "routes.prefs_routes", prefs)
    for name, value in {
        "DEEP_RESEARCH_DIR": tmp_path / "research",
        "MEMORY_FILE": tmp_path / "memory.json",
        "SKILLS_DIR": tmp_path / "skills",
    }.items():
        monkeypatch.setattr(auth_routes, name, str(value))

    auth_manager = MagicMock()
    auth_manager.is_admin.return_value = True
    auth_manager.get_username_for_token.return_value = "admin"
    auth_manager.users = {"alice": {}, "bob": {}}
    auth_manager.rename_user.return_value = True
    auth_manager.delete_user.return_value = True
    router = auth_routes.setup_auth_routes(auth_manager)
    endpoints = {route.endpoint.__name__: route.endpoint for route in router.routes}
    request = SimpleNamespace(
        cookies={"odysseus_session": "t"},
        app=SimpleNamespace(state=SimpleNamespace()),
        state=SimpleNamespace(current_user="admin"),
    )
    return endpoints, request


def _seed_user(owner, rule):
    store.set_chat_runtime(owner, store.RUNTIME_NEW)
    folder = store.instructions_dir(owner)
    os.makedirs(folder)
    with open(os.path.join(folder, "AGENTS.md"), "w", encoding="utf-8") as handle:
        handle.write(rule)


@pytest.mark.asyncio
async def test_rename_moves_store_and_folder_and_drops_stale_data(user_admin_routes):
    endpoints, request = user_admin_routes
    _seed_user("alice", "ALICE RULE")
    _seed_user("alice2", "STALE RULE")

    result = await endpoints["rename_user"]("alice", SimpleNamespace(username="Alice2"), request)

    assert result["ok"] is True
    assert store.load("alice") == {}
    assert not os.path.exists(store.instructions_dir("alice"))
    assert store.chat_runtime("alice2") == store.RUNTIME_NEW
    with open(store.user_instructions_path("alice2"), encoding="utf-8") as handle:
        assert handle.read() == "ALICE RULE"


@pytest.mark.asyncio
async def test_delete_removes_store_and_folder_of_that_user_only(user_admin_routes):
    endpoints, request = user_admin_routes
    _seed_user("alice", "ALICE RULE")
    _seed_user("bob", "BOB RULE")

    result = await endpoints["admin_delete_user"](SimpleNamespace(username="Alice"), request)

    assert result == {"ok": True}
    assert not os.path.exists(store.store_path("alice"))
    assert not os.path.exists(store.instructions_dir("alice"))
    assert store.chat_runtime("bob") == store.RUNTIME_NEW
    assert os.path.exists(store.user_instructions_path("bob"))


@pytest.fixture
def context_info(monkeypatch, tmp_path):
    import routes.session_routes as session_routes

    monkeypatch.setattr(constants, "CHAT_AGENT_DIR", str(tmp_path / "chat_agent"))
    monkeypatch.setattr(constants, "INSTRUCTIONS_DIR", str(tmp_path / "instructions"))
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)
    monkeypatch.setattr(session_routes, "_verify_session_owner", lambda *a, **k: None)
    session = SimpleNamespace(endpoint_url="", model="", cwd="")
    router = session_routes.setup_session_routes(
        SimpleNamespace(get_session=lambda session_id: session), {},
    )
    endpoint = next(
        route.endpoint for route in router.routes
        if route.path == "/api/session/{session_id}/context_info"
    )

    async def call(cwd=None):
        request = SimpleNamespace(
            headers={}, state=SimpleNamespace(current_user="alice"),
            app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
        )
        return await endpoint(request, "session-1", cwd=cwd)

    return call


@pytest.mark.asyncio
async def test_context_info_lists_loaded_files_without_contents(context_info, tmp_path):
    workspace = tmp_path / "home" / "app"
    workspace.mkdir(parents=True)
    (workspace / "CLAUDE.md").write_text("PROJECT RULE", encoding="utf-8")
    store.set_chat_runtime("alice", store.RUNTIME_NEW)
    os.makedirs(store.instructions_dir("alice"))
    with open(store.user_instructions_path("alice"), "w", encoding="utf-8") as handle:
        handle.write("x" * (constants.INSTRUCTION_FILE_MAX_CHARS + 1))

    info = await context_info(cwd=str(workspace))

    assert info["instructions"] == [
        {"path": store.user_instructions_path("alice"), "truncated": True},
        {"path": str(workspace / "CLAUDE.md"), "truncated": False},
    ]


@pytest.mark.asyncio
async def test_context_info_lists_nothing_while_the_switch_is_off(context_info, tmp_path):
    os.makedirs(store.instructions_dir("alice"))
    with open(store.user_instructions_path("alice"), "w", encoding="utf-8") as handle:
        handle.write("rule")

    assert (await context_info())["instructions"] == []


@pytest.mark.parametrize("owner", [".", "..", "%2E%2E"])
def test_pin_confine_refuses_a_folder_name_that_is_not_inside_the_root(
    monkeypatch, tmp_path, owner,
):
    """R2 pin: confine(allow_root=False) raises ValueError for the root and above it."""
    monkeypatch.setattr(constants, "INSTRUCTIONS_DIR", str(tmp_path / "instructions"))
    try:
        folder = store.instructions_dir(owner)
    except ValueError:
        return
    assert os.path.dirname(folder) == str(tmp_path / "instructions")


def test_pin_resolve_request_workspace_drops_the_path_for_non_admins(monkeypatch, tmp_path):
    """R2 pin: context info lists workspace files only where the chat route binds them."""
    from routes.chat_routes import _resolve_request_workspace

    request = SimpleNamespace(state=SimpleNamespace(current_user="bob"))
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: False)
    assert _resolve_request_workspace(request, str(tmp_path)) == ("", "")
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)
    assert _resolve_request_workspace(request, str(tmp_path))[0] == str(tmp_path.resolve())
