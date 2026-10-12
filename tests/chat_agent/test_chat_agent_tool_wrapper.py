"""No-LLM regression of the per-call execution wrapper, one test per family (#34).

Each call runs through the real legacy executor with the wrapper's fresh
per-call authority. The assertion that matters is that resource binding
admits the call: no BLOCKED, no ``*_denied`` failure kind. A handler that
drifts after an upstream merge fails here.
"""

import json
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core import database
from src import bg_jobs, containment, tool_execution
from src.chat_agent import tools
from src.tool_schemas import function_call_to_tool_block


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    from src.agent_runtime import process_resources
    monkeypatch.setattr(process_resources, "_LAUNCH_DIR", tmp_path / "private" / "launches")
    monkeypatch.setattr(bg_jobs, "_JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(bg_jobs, "_STORE", tmp_path / "jobs.json")
    monkeypatch.setattr(containment, "_store_path", lambda: tmp_path / "grants.json")
    # The test image has no user namespaces for bubblewrap (#47); the wrapper's
    # binding is what is under test, not the sandbox.
    monkeypatch.setattr(containment, "CONTAINMENT_MODE", containment.MODE_REPORT_ONLY)
    monkeypatch.setattr(containment, "MECHANISMS",
                        tuple(m for m in containment.MECHANISMS if m.name == "process_group"))
    monkeypatch.setattr(tool_execution, "_owner_is_admin", lambda owner: True)
    path = tmp_path / "workspace"
    path.mkdir()
    return str(path)


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    return factory


def _scope(workspace=None, owner="Alice", approved_task=False):
    return tools.ExecutionScope(
        owner=owner, session_id="session-1", workspace=workspace,
        client_runtime_context=None, denied_names=frozenset(), approved_task=approved_task,
    )


async def _call(name, arguments, scope):
    block = function_call_to_tool_block(name, json.dumps(arguments))
    outcome = await tools.execute("call-1", block, scope)
    assert not outcome.result.get("blocked"), outcome.result
    assert not str(outcome.result.get("failure_kind") or "").endswith("_denied"), outcome.result
    return outcome


@pytest.mark.asyncio
async def test_files_family_writes_and_reads_inside_the_workspace(workspace):
    scope = _scope(workspace)

    written = await _call("write_file", {"path": "notes.txt", "content": "hello"}, scope)
    read = await _call("read_file", {"path": "notes.txt"}, scope)
    listed = await _call("ls", {"path": "."}, scope)

    assert written.ok and read.ok and listed.ok
    assert "hello" in read.model_text
    assert "notes.txt" in listed.model_text


@pytest.mark.asyncio
async def test_bash_family_runs_in_the_workspace(workspace):
    outcome = await _call("bash", {"command": "pwd"}, _scope(workspace))

    assert outcome.ok, outcome.result
    assert workspace in outcome.model_text


@pytest.mark.asyncio
async def test_background_job_started_in_one_turn_is_managed_in_the_next(workspace):
    started = await _call("bash", {"command": "#!bg\nsleep 30"}, _scope(workspace))
    assert started.ok, started.result
    [job_id] = list(bg_jobs._load())

    # A later turn: a fresh call with its own fresh authority.
    listed = await _call("manage_bg_jobs", {"action": "list"}, _scope(workspace))
    killed = await _call("manage_bg_jobs", {"action": "kill", "job_id": job_id}, _scope(workspace))

    assert job_id in listed.model_text
    assert killed.ok, killed.result
    deadline = time.monotonic() + 10
    while bg_jobs._load()[job_id]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert bg_jobs._load()[job_id]["status"] != "running"


@pytest.mark.asyncio
async def test_notes_family_is_scoped_to_the_owner(db):
    await _call("manage_notes", {"action": "add", "title": "bob's note"}, _scope(owner="bob"))
    await _call("manage_notes", {"action": "add", "title": "alice's note"}, _scope(owner="alice"))

    listed = await _call("manage_notes", {"action": "list"}, _scope(owner="alice"))

    assert "alice's note" in listed.model_text
    assert "bob's note" not in listed.model_text
    with db() as session:
        owners = {n.title: n.owner for n in session.query(database.Note)}
    assert owners == {"bob's note": "bob", "alice's note": "alice"}


@pytest.mark.asyncio
async def test_owners_that_casefold_alike_stay_separate_accounts(db):
    # AuthManager keys usernames by strip().lower(), so these are two accounts.
    await _call("manage_notes", {"action": "add", "title": "strasse's note"}, _scope(owner="strasse"))

    listed = await _call("manage_notes", {"action": "list"}, _scope(owner="straße"))

    assert "strasse's note" not in listed.model_text
    with db() as session:
        assert {n.owner for n in session.query(database.Note)} == {"strasse"}


def _sealed_tools(db, name):
    with db() as session:
        task = session.query(database.ScheduledTask).filter_by(name=name).one()
        return {grant["tool"] for grant in json.loads(task.request_authority_json)["authority"]["grants"]}


@pytest.mark.asyncio
async def test_an_approved_task_is_sealed_with_its_prompt_grants_like_a_ui_created_one(db, monkeypatch):
    monkeypatch.setattr(tool_execution, "_owner_is_admin", lambda owner: True)
    # Relies on: create_request_authority(prompt) is what a task created from
    # the UI is sealed with (seal_task_authority with no parent authority).
    from src.agent_runtime.authority import create_request_authority
    prompt = "Search the web for AI news"
    expected = {g.tool for g in create_request_authority(prompt, owner="alice").grants}
    assert "web_search" in expected

    task = {"action": "create", "prompt": prompt, "schedule": "daily", "scheduled_time": "08:00"}
    approved = await _call("manage_tasks", {**task, "name": "news"}, _scope(owner="alice", approved_task=True))
    # Admitted any other way (a session rule, later auto mode): no prompt grants.
    unapproved = await _call("manage_tasks", {**task, "name": "news-2"}, _scope(owner="alice"))

    assert approved.ok and unapproved.ok
    assert _sealed_tools(db, "news") == expected
    assert _sealed_tools(db, "news-2") == set()
