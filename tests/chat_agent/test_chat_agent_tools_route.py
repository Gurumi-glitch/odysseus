"""Everyday tools on the new runtime (ticket #50), driven through the chat route.

A scripted Chat Completions transport plays the model; tools run through the
real legacy executor against a throwaway SQLite database.
"""

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.search.content as search_content
import src.tool_security as tool_security
import src.tool_utils as tool_utils
from core import database
from src.chat_agent.tools import DEFERRED_TOOLS
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _sse_tool_calls,
    _switch_on,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    return factory


@pytest.fixture
def harness(monkeypatch, tmp_path, db):
    _switch_on_after = _Harness(monkeypatch, tmp_path)
    _switch_on()
    return _switch_on_after


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)


def _tool_events(events, kind):
    return [e for e in events if e.get("type") == kind]


def _tool_names(payload):
    return [tool["function"]["name"] for tool in payload.get("tools", [])]


def _tool_messages(payload):
    return [m for m in payload["messages"] if m["role"] == "tool"]


@pytest.mark.asyncio
async def test_add_note_runs_with_no_ask_and_saves_a_done_tool_event(harness, db):
    harness.provider_rounds = [
        _sse_tool_calls(("manage_notes", {"action": "add", "title": "dinner time with mom"})),
        _sse_events(["Added your note."]),
    ]

    _, body = await harness.send(_Request(
        "agent", message="add note for me: dinner time with mom",
    ))

    events = _data_events(body)
    assert not [e for e in events if e.get("type") == "ask_user"]
    [start] = _tool_events(events, "tool_start")
    [output] = _tool_events(events, "tool_output")
    assert (start["tool"], start["call_id"], start["round"]) == ("manage_notes", "call_0", 1)
    assert (output["exit_code"], output["call_id"]) == (0, "call_0")
    with db() as session:
        [note] = session.query(database.Note).all()
    assert (note.title, note.owner) == ("dinner time with mom", "alice")

    first, second = harness.captured["provider_requests"]
    assert first["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in first
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "manage_notes"
    [tool_message] = _tool_messages(second)
    assert tool_message["tool_call_id"] == "call_0"

    [row] = harness.saved
    [event] = row.metadata["tool_events"]
    assert (event["status"], event["exit_code"], event["round"]) == ("done", 0, 1)
    assert row.metadata["round_texts"] == ["", "Added your note."]
    first_round = row.metadata["agent_v2"]["rounds"][0]
    assert [b["type"] for b in first_round["blocks"]] == ["tool_call"]
    [result] = first_round["results"]
    assert (result["call_id"], result["ok"], result["error"]) == ("call_0", True, None)


@pytest.mark.asyncio
async def test_calls_in_one_round_run_one_at_a_time_in_call_order(harness, db):
    harness.provider_rounds = [
        _sse_tool_calls(
            ("manage_notes", {"action": "add", "title": "first"}),
            ("manage_notes", {"action": "add", "title": "second"}),
        ),
        _sse_events(["Both added."]),
    ]

    _, body = await harness.send(_Request("agent"))

    kinds = [(e["type"], e["call_id"]) for e in _data_events(body) if e.get("type", "").startswith("tool_")]
    assert kinds == [
        ("tool_start", "call_0"), ("tool_output", "call_0"),
        ("tool_start", "call_1"), ("tool_output", "call_1"),
    ]
    with db() as session:
        titles = [n.title for n in session.query(database.Note).order_by(database.Note.created_at)]
    assert titles == ["first", "second"]


@pytest.mark.asyncio
async def test_upfront_catalog_is_the_everyday_built_ins_once(harness, admin):
    await harness.send(_Request("agent", allow_web_search="true", allow_bash="true"))

    [payload] = harness.captured["provider_requests"]
    names = _tool_names(payload)
    assert len(names) == 57
    assert len(set(names)) == len(names)
    assert not set(names) & DEFERRED_TOOLS
    assert {"manage_notes", "send_email", "list_emails", "web_search", "bash"} <= set(names)
    assert not [n for n in names if n.startswith("mcp__")]
    assert "Unavailable" not in payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_web_off_removes_web_tools_lists_them_and_denies_a_call(harness, monkeypatch):
    fetched = []
    monkeypatch.setattr(search_content, "fetch_webpage_content",
                        lambda url, **kw: fetched.append(url) or {"content": "page"})
    harness.provider_rounds = [
        _sse_tool_calls(("web_fetch", {"url": "https://example.com"})),
        _sse_events(["Web access is off."]),
    ]

    _, body = await harness.send(_Request("agent", allow_web_search="false"))

    first, second = harness.captured["provider_requests"]
    names = set(_tool_names(first))
    assert not names & {"web_search", "web_fetch", "get_weather", "youtube_tool", "private_browser"}
    system = first["messages"][0]["content"]
    assert "Unavailable (disabled by user/admin):" in system
    assert "web_search" in system.split("Unavailable", 1)[1]
    assert fetched == []
    [output] = _tool_events(_data_events(body), "tool_output")
    assert output["exit_code"] == 126
    error = json.loads(_tool_messages(second)[0]["content"])
    assert error["status"] == "denied"


@pytest.mark.asyncio
async def test_nobody_mode_denies_memory_tools(harness, admin):
    harness.provider_rounds = [
        _sse_tool_calls(("manage_memory", {"action": "add", "content": "likes tea"})),
        _sse_events(["Memory is off in Nobody mode."]),
    ]

    _, body = await harness.send(_Request("agent", incognito="true"))

    first, second = harness.captured["provider_requests"]
    assert "manage_memory" not in _tool_names(first)
    assert "manage_memory" in first["messages"][0]["content"]
    assert [e["tool"] for e in _tool_events(_data_events(body), "tool_start")] == []
    assert json.loads(_tool_messages(second)[0]["content"])["status"] == "denied"


class _FakeMcp:
    def plan_mode_blocked_mcp(self):
        return {"github": {"create_issue"}}, {"mcp__github__create_issue"}


@pytest.mark.asyncio
async def test_plan_mode_denies_write_tools_including_mcp_writes(harness, admin, monkeypatch, db):
    monkeypatch.setattr(tool_utils, "_mcp_manager", _FakeMcp())
    harness.provider_rounds = [
        _sse_tool_calls(
            ("manage_notes", {"action": "add", "title": "plan only"}),
            ("mcp__github__create_issue", {"title": "x"}),
        ),
        _sse_events(["Here is the plan."]),
    ]

    _, body = await harness.send(_Request("agent", plan_mode="true", allow_web_search="true"))

    first, second = harness.captured["provider_requests"]
    names = set(_tool_names(first))
    assert not names & {"manage_notes", "write_file", "bash", "send_email"}
    assert {"read_file", "web_search"} <= names
    assert [o["exit_code"] for o in _tool_events(_data_events(body), "tool_output")] == [126, 126]
    assert [json.loads(m["content"])["status"] for m in _tool_messages(second)] == ["denied", "denied"]
    with db() as session:
        assert session.query(database.Note).count() == 0


@pytest.mark.asyncio
async def test_a_404_web_fetch_returns_an_error_result_and_the_turn_goes_on(harness, monkeypatch):
    monkeypatch.setattr(search_content, "fetch_webpage_content",
                        lambda url, **kw: {"error": "HTTP 404 Not Found"})
    harness.provider_rounds = [
        _sse_tool_calls(("web_fetch", {"url": "https://example.com/missing"})),
        _sse_events(["That page returned 404."]),
    ]

    _, body = await harness.send(_Request("agent", allow_web_search="true"))

    [output] = _tool_events(_data_events(body), "tool_output")
    assert output["exit_code"] == 1
    _, second = harness.captured["provider_requests"]
    error = json.loads(_tool_messages(second)[0]["content"])
    assert error["status"] == "failed"
    assert "404" in error["error"]
    [row] = harness.saved
    assert row.content == "That page returned 404."
    [event] = row.metadata["tool_events"]
    assert (event["status"], event["exit_code"], event["error"]) == ("failed", 1, True)


@pytest.mark.asyncio
async def test_an_ask_posture_call_returns_a_denial_until_asks_arrive(harness, admin, monkeypatch):
    import src.tool_execution as tool_execution
    ran = []

    async def executor(*args, **kwargs):
        ran.append(args)
        return "bash", {"output": "", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", executor)
    harness.provider_rounds = [
        _sse_tool_calls(("bash", {"command": "ls"})),
        _sse_events(["I need your approval to run that."]),
    ]

    _, body = await harness.send(_Request("agent", allow_bash="true"))

    assert ran == []
    [output] = _tool_events(_data_events(body), "tool_output")
    assert output["exit_code"] == 126
    _, second = harness.captured["provider_requests"]
    assert "approval" in json.loads(_tool_messages(second)[0]["content"])["error"]


@pytest.mark.asyncio
async def test_unusable_arguments_fail_back_to_the_model(harness):
    harness.provider_rounds = [
        _sse_tool_calls(("web_search", {})),
        _sse_events(["Let me retry."]),
    ]

    _, body = await harness.send(_Request("agent", allow_web_search="true"))

    [output] = _tool_events(_data_events(body), "tool_output")
    assert output["exit_code"] == 1
    _, second = harness.captured["provider_requests"]
    assert json.loads(_tool_messages(second)[0]["content"])["status"] == "failed"


@pytest.mark.asyncio
async def test_per_model_none_surface_sends_no_tools(harness, db):
    with db() as session:
        session.add(database.ModelEndpoint(
            id="ep-1", name="OpenRouter", base_url="https://openrouter.ai/api/v1",
            owner="alice", model_tool_modes=json.dumps({"x-ai/grok-4.1-fast": "none"}),
        ))
        session.commit()

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert "tools" not in payload and "tool_choice" not in payload


@pytest.mark.asyncio
async def test_per_model_compact_surface_strips_descriptions(harness, db):
    with db() as session:
        session.add(database.ModelEndpoint(
            id="ep-1", name="OpenRouter", base_url="https://openrouter.ai/api/v1",
            owner="alice", model_tool_modes=json.dumps({"x-ai/grok-4.1-fast": "compact"}),
        ))
        session.commit()

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    notes = next(t for t in payload["tools"] if t["function"]["name"] == "manage_notes")
    from src.agent_tools import FUNCTION_TOOL_SCHEMAS
    full = next(t for t in FUNCTION_TOOL_SCHEMAS if t["function"]["name"] == "manage_notes")
    assert len(json.dumps(notes)) < len(json.dumps(full))
