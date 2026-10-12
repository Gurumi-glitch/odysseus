"""Deferred tools and tool search on the new runtime (ticket #58), driven
through the chat route with a scripted Chat Completions transport."""

import json
from types import SimpleNamespace

import pytest

import routes.chat_routes as chat_routes
import src.tool_execution as tool_execution
import src.tool_utils as tool_utils
from core import database
from src.chat_agent import transcript
from src.chat_agent.tools import TOOL_SEARCH, TOOL_SEARCH_MAX_SCHEMAS
from src.mcp_manager import McpManager
from tests.chat_agent.test_chat_agent_asks_route import answering, send_and_answer
from tests.chat_agent.test_chat_agent_route import _Request, _data_events, _sse_events, _sse_tool_calls
from tests.chat_agent.test_chat_agent_tools_route import (  # noqa: F401 (fixtures)
    _tool_events,
    _tool_messages,
    _tool_names,
    admin,
    db,
    harness,
)

_EMPTY_ARGS = {"type": "object", "properties": {}}


def _mcp_tool(name, description):
    return {"name": name, "description": description, "input_schema": _EMPTY_ARGS}


class _GithubSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return SimpleNamespace(content=[SimpleNamespace(text="3 open issues")], isError=False)


@pytest.fixture
def mcp(monkeypatch):
    """A real McpManager holding one custom server plus the servers the
    catalog must never list (internal, raw browser, built-in email)."""
    manager = McpManager()
    manager._tools = {
        "github": [
            _mcp_tool("list_issues", "List open issues in a GitHub repository"),
            _mcp_tool("create_issue", "Create a GitHub issue"),
        ],
        "memory": [_mcp_tool("memory_search", "Search memories")],
        "rag": [_mcp_tool("rag_search", "Search documents")],
        "image_gen": [_mcp_tool("generate", "Generate an image")],
        "builtin_browser": [_mcp_tool("browser_navigate", "Open a page")],
        "email": [_mcp_tool("list_emails", "List emails")],
    }
    manager._connections = {"github": {"status": "connected", "name": "GitHub"}}
    session = _GithubSession()
    manager._sessions = {"github": session}
    # What connect_server records for a stdio server, so the executor can
    # bind the call to this exact connection.
    manager._resource_endpoints["github"] = ("stdio:github", "config-1")
    manager._register_resource_connection("github", session)
    monkeypatch.setattr(tool_utils, "_mcp_manager", manager)
    return session


@pytest.fixture
def executor(monkeypatch):
    ran = []

    async def run(block, *args, **kwargs):
        ran.append(block.tool_type)
        return block.tool_type, {"output": "served: qwen3 on :8001", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)
    return ran


def _system(payload):
    return payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_list_served_models_goes_through_tool_search_first(harness, admin, executor):
    harness.provider_rounds = [
        _sse_tool_calls((TOOL_SEARCH, {"query": "served models"})),
        _sse_tool_calls(("list_served_models", {})),
        _sse_events(["qwen3 is served on :8001."]),
    ]

    _, body = await harness.send(_Request("agent", message="list my served models"))

    first, second, third = harness.captured["provider_requests"]
    assert TOOL_SEARCH in _tool_names(first)
    assert "list_served_models" not in _tool_names(first)
    assert "list_served_models" in _system(first)
    assert "list_served_models" in _tool_names(second)
    assert "list_served_models" in _tool_names(third)
    events = _data_events(body)
    assert [e["tool"] for e in _tool_events(events, "tool_start")] == [TOOL_SEARCH, "list_served_models"]
    assert [e["exit_code"] for e in _tool_events(events, "tool_output")] == [0, 0]
    # tool_search never reaches the legacy executor.
    assert executor == ["list_served_models"]
    [row] = harness.saved
    [search_result] = row.metadata["agent_v2"]["rounds"][0]["results"]
    assert "list_served_models" in search_result["loaded"]


@pytest.mark.asyncio
async def test_catalog_lists_custom_mcp_by_server_and_hides_internal_ones(harness, admin, mcp, db):
    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    system = _system(payload)
    assert "mcp__github__list_issues" in system and "mcp__github__create_issue" in system
    for hidden in ("mcp__memory__", "mcp__rag__", "mcp__image_gen__", "mcp__builtin_browser__", "mcp__email__"):
        assert hidden not in system
    assert not [name for name in _tool_names(payload) if name.startswith("mcp__")]


def _search_then_call_issue_lister(harness):
    harness.provider_rounds = [
        _sse_tool_calls((TOOL_SEARCH, {"query": "select:mcp__github__list_issues"})),
        _sse_tool_calls(("mcp__github__list_issues", {})),
        _sse_events(["There are 3 open issues."]),
    ]


@pytest.mark.asyncio
async def test_a_custom_mcp_tool_is_callable_after_search(harness, admin, mcp, db, monkeypatch):
    # The executor's own copy of the admin check (custom MCP is admin-only).
    monkeypatch.setattr(tool_execution, "owner_is_admin_or_single_user", lambda owner: True)
    # The ask endpoint answers as whoever the request is signed in as.
    monkeypatch.setattr(chat_routes, "effective_user", lambda request: request.state.current_user)
    _search_then_call_issue_lister(harness)

    # Custom MCP tools ask by default (spec #44 default posture).
    body = await send_and_answer(harness, _Request("agent"), answering(harness, ("once",)))

    _, second, _ = harness.captured["provider_requests"]
    assert "mcp__github__list_issues" in _tool_names(second)
    [ask] = _tool_events(_data_events(body), "ask_user")
    assert ask["data"]["action"]["tool"] == "mcp__github__list_issues"
    assert mcp.calls == [("list_issues", {})]
    outputs = _tool_events(_data_events(body), "tool_output")
    assert [(o["tool"], o["exit_code"]) for o in outputs] == [
        (TOOL_SEARCH, 0), ("mcp__github__list_issues", 0),
    ]
    assert "3 open issues" in outputs[1]["output"]


def _history_turn_that_loaded(name):
    # A row the transcript writer signed (unsigned agent_v2 replays as text).
    return {
        "role": "assistant", "content": "",
        "metadata": transcript.signed_metadata("", {"agent_v2": {"rounds": [
            {"blocks": [{"type": "tool_call", "call_id": "old_0", "name": TOOL_SEARCH,
                         "arguments": json.dumps({"query": f"select:{name}"})}],
             "results": [{"call_id": "old_0", "ok": True, "content": f"Loaded: {name}",
                          "error": None, "loaded": [name]}]},
            {"blocks": [{"type": "text", "text": "Loaded."}]},
        ]}}),
    }


@pytest.mark.asyncio
async def test_loaded_tools_stay_loaded_on_the_next_turn(harness, admin, mcp, db):
    harness.context.history = [
        {"role": "user", "content": "load the issue lister"},
        _history_turn_that_loaded("mcp__github__list_issues"),
    ]

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert "mcp__github__list_issues" in _tool_names(payload)


def _disable_issue_creator(db):
    with db() as session:
        session.add(database.McpServer(
            id="github", name="GitHub", transport="stdio", disabled_tools=json.dumps(["create_issue"]),
        ))
        session.commit()


class _RoundsThatDisableBeforeTheCall(list):
    """Scripted rounds; the admin disables the tool once the model has loaded it."""

    def __init__(self, rounds, db):
        super().__init__(rounds)
        self.db = db

    def pop(self, index=-1):
        if len(self) == 2:
            _disable_issue_creator(self.db)
        return super().pop(index)


@pytest.mark.asyncio
async def test_a_per_server_disabled_mcp_tool_is_refused_at_call_time(harness, admin, mcp, db):
    harness.provider_rounds = _RoundsThatDisableBeforeTheCall([
        _sse_tool_calls((TOOL_SEARCH, {"query": "select:mcp__github__create_issue"})),
        _sse_tool_calls(("mcp__github__create_issue", {"title": "x"})),
        _sse_events(["That tool is disabled."]),
    ], db)

    _, body = await harness.send(_Request("agent"))

    _, second, third = harness.captured["provider_requests"]
    assert "mcp__github__create_issue" in _tool_names(second)
    assert mcp.calls == []
    _, output = _tool_events(_data_events(body), "tool_output")
    assert output["exit_code"] == 126
    assert "disabled on its MCP server" in output["output"]
    assert json.loads(_tool_messages(third)[-1]["content"])["status"] == "denied"


@pytest.mark.asyncio
async def test_a_per_server_disabled_mcp_tool_leaves_the_catalog_and_the_loaded_list(harness, admin, mcp, db):
    harness.context.history = [
        {"role": "user", "content": "load the issue creator"},
        _history_turn_that_loaded("mcp__github__create_issue"),
    ]
    _disable_issue_creator(db)

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert "mcp__github__create_issue" not in _system(payload)
    assert "mcp__github__create_issue" not in _tool_names(payload)


@pytest.mark.asyncio
async def test_non_admins_never_see_or_load_mcp_tools(harness, mcp, db):
    harness.provider_rounds = [
        _sse_tool_calls((TOOL_SEARCH, {"query": "select:mcp__github__list_issues"})),
        _sse_events(["No such tool."]),
    ]

    await harness.send(_Request("agent"))

    first, second = harness.captured["provider_requests"]
    assert "mcp__" not in _system(first)
    assert not [name for name in _tool_names(second) if name.startswith("mcp__")]


@pytest.mark.asyncio
async def test_one_search_loads_at_most_the_schema_limit(harness, admin, executor):
    harness.provider_rounds = [
        _sse_tool_calls((TOOL_SEARCH, {"query": "list models sessions downloads presets servers"})),
        _sse_events(["Loaded."]),
    ]

    await harness.send(_Request("agent"))

    first, second = harness.captured["provider_requests"]
    added = set(_tool_names(second)) - set(_tool_names(first))
    assert len(added) == TOOL_SEARCH_MAX_SCHEMAS


@pytest.mark.asyncio
async def test_the_bg_job_hint_tool_is_loadable(harness, admin):
    harness.provider_rounds = [
        _sse_tool_calls((TOOL_SEARCH, {"query": "background jobs"})),
        _sse_events(["Loaded."]),
    ]

    await harness.send(_Request("agent", allow_bash="true"))

    _, second = harness.captured["provider_requests"]
    assert "manage_bg_jobs" in _tool_names(second)
