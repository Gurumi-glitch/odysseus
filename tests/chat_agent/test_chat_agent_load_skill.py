"""Skills load on demand on the new runtime (ticket #66), driven through the
chat route with a scripted Chat Completions transport and a real SkillsManager."""

import json
import os
from types import SimpleNamespace

import pytest

import routes.prefs_routes as prefs_routes
import src.chat_agent.tools as chat_agent_tools
import src.constants as constants
import src.tool_execution as tool_execution
from services.memory.skills import SkillsManager
from src.chat_agent import prompt
from src.chat_agent.tools import LOAD_SKILL, USER_ONLY_REASON
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _sse_tool_calls,
    _switch_on,
)
from tests.chat_agent.test_chat_agent_tools_route import (  # noqa: F401 (fixtures)
    _tool_events,
    _tool_messages,
    _tool_names,
    admin,
    db,
)

OWNER = "alice"
LOGS = "rotate-nginx-logs"
DEPLOY = "deploy-the-blog"
SERVED = "check-served-models"


@pytest.fixture
def prefs(monkeypatch):
    values = {"skills_enabled": True, "auto_approve_skills": True, "skill_min_confidence": 0.85}

    def load(user=None):
        return dict(values)

    # is_usable reads the approval policy through prefs_routes; the runtime
    # reads the Skill injection toggle through its own import.
    monkeypatch.setattr(prefs_routes, "_load_for_user", load)
    monkeypatch.setattr(chat_agent_tools, "_load_for_user", load)
    return values


@pytest.fixture
def skills(tmp_path, monkeypatch, prefs):
    data_dir = tmp_path / "skills-data"
    data_dir.mkdir()
    monkeypatch.setattr(constants, "BUILTIN_SKILLS_DIR", str(tmp_path / "no-builtins"))
    sm = SkillsManager(str(data_dir))
    _add(sm, LOGS, "Rotate the nginx access logs", ["run logrotate -f /etc/logrotate.d/nginx"])
    _add(sm, SERVED, "Check which local models are served", ["call list_served_models"],
         requires_toolsets=["list_served_models"])
    _add(sm, DEPLOY, "Deploy the blog to production", ["run ./deploy.sh"],
         requires_toolsets=["list_served_models"], user_only=True)
    _add(sm, "half-written-draft", "A draft nobody approved", ["do nothing"], approve=False)
    return sm


def _add(sm, name, description, procedure, *, requires_toolsets=None, user_only=False, approve=True):
    sm.add_skill(name=name, description=description, owner=OWNER, source="user",
                 status="published" if approve else "draft", procedure=procedure,
                 requires_toolsets=requires_toolsets)
    if user_only:
        sm.update_skill(name, {"disable_model_invocation": True}, owner=OWNER)
    if approve:
        # What a Skills page publish stamps (ADR 0008).
        sm.set_user_approval(next(s for s in sm.load(owner=OWNER) if s["name"] == name))


@pytest.fixture
def harness(monkeypatch, tmp_path, db, skills):
    harness = _Harness(monkeypatch, tmp_path, skills_manager=skills)
    _switch_on()
    return harness


@pytest.fixture
def executor(monkeypatch):
    ran = []

    async def run(block, *args, **kwargs):
        ran.append(block.tool_type)
        return block.tool_type, {"output": "served: qwen3 on :8001", "exit_code": 0}

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", run)
    return ran


def _instructions(payload):
    message = payload["messages"][1]
    return message["content"] if message["role"] == "user" else ""


def _uses(sm, name):
    return next(s for s in sm.load(owner=OWNER) if s["name"] == name)["uses"]


@pytest.mark.asyncio
async def test_a_hand_published_skill_is_indexed_loaded_and_counted(harness, skills):
    harness.provider_rounds = [
        _sse_tool_calls((LOAD_SKILL, {"name": LOGS})),
        _sse_events(["Running logrotate -f now."]),
    ]

    _, body = await harness.send(_Request("agent", message="rotate my nginx logs"))

    first, second = harness.captured["provider_requests"]
    index = _instructions(first)
    assert f"{LOGS}: Rotate the nginx access logs" in index
    assert LOAD_SKILL in _tool_names(first)
    [result] = _tool_messages(second)
    assert "logrotate -f /etc/logrotate.d/nginx" in result["content"]
    outputs = _tool_events(_data_events(body), "tool_output")
    assert [(o["tool"], o["exit_code"]) for o in outputs] == [(LOAD_SKILL, 0)]
    assert _uses(skills, LOGS) == 1


@pytest.mark.asyncio
async def test_user_only_and_unapproved_skills_never_reach_the_model(harness, skills):
    harness.provider_rounds = [
        _sse_tool_calls((LOAD_SKILL, {"name": DEPLOY}), (LOAD_SKILL, {"name": "half-written-draft"})),
        _sse_events(["Start it yourself with /deploy-the-blog."]),
    ]

    await harness.send(_Request("agent", message="deploy the blog"))

    first, second = harness.captured["provider_requests"]
    index = _instructions(first)
    assert DEPLOY not in index and "Deploy the blog" not in index
    assert "half-written-draft" not in index
    user_only, draft = _tool_messages(second)
    assert USER_ONLY_REASON.format(name=DEPLOY) in user_only["content"]
    assert "./deploy.sh" not in user_only["content"]
    assert "not available" in draft["content"]
    assert _uses(skills, DEPLOY) == 0


@pytest.mark.asyncio
async def test_loading_a_skill_makes_its_deferred_tool_callable_without_tool_search(
    harness, admin, executor,
):
    harness.provider_rounds = [
        _sse_tool_calls((LOAD_SKILL, {"name": SERVED})),
        _sse_tool_calls(("list_served_models", {})),
        _sse_events(["qwen3 is served on :8001."]),
    ]

    _, body = await harness.send(_Request("agent", message="which models are served?"))

    first, second, _ = harness.captured["provider_requests"]
    assert "list_served_models" not in _tool_names(first)
    assert "list_served_models" in _tool_names(second)
    assert executor == ["list_served_models"]
    outputs = _tool_events(_data_events(body), "tool_output")
    assert [(o["tool"], o["exit_code"]) for o in outputs] == [(LOAD_SKILL, 0), ("list_served_models", 0)]
    [row] = harness.saved
    [load_result] = row.metadata["agent_v2"]["rounds"][0]["results"]
    assert load_result["loaded"] == ["list_served_models"]


@pytest.mark.asyncio
async def test_slash_invoked_user_only_skill_loads_its_tools_that_turn(harness, admin, skills):
    await harness.send(_Request("agent", message="Apply the skill below...", invoked_skill=DEPLOY))

    [payload] = harness.captured["provider_requests"]
    assert "list_served_models" in _tool_names(payload)
    # The model still never sees the user-only skill in its index.
    assert DEPLOY not in _instructions(payload)


@pytest.mark.asyncio
async def test_invoking_an_unapproved_skill_loads_nothing(harness, admin, skills):
    skills.update_skill(SERVED, {"description": "Edited after approval"}, owner=OWNER)

    await harness.send(_Request("agent", invoked_skill=SERVED))

    [payload] = harness.captured["provider_requests"]
    assert "list_served_models" not in _tool_names(payload)


@pytest.mark.asyncio
async def test_skill_injection_off_means_no_index_and_no_load_skill(harness, prefs):
    prefs["skills_enabled"] = False

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert LOGS not in json.dumps(payload["messages"])
    assert LOAD_SKILL not in _tool_names(payload)


@pytest.mark.asyncio
async def test_session_skill_toggle_off_means_no_index_and_no_load_skill(harness):
    harness.session.skill_injection_enabled = False
    harness.provider_rounds = [
        _sse_tool_calls((LOAD_SKILL, {"name": LOGS})),
        _sse_events(["ok"]),
    ]

    await harness.send(_Request("agent"))

    first, second = harness.captured["provider_requests"]
    assert LOGS not in json.dumps(first["messages"])
    assert LOAD_SKILL not in _tool_names(first)
    [refused] = _tool_messages(second)
    assert "logrotate" not in refused["content"]


@pytest.mark.asyncio
async def test_nobody_mode_has_no_skills(harness):
    await harness.send(_Request("agent", incognito="true"))

    [payload] = harness.captured["provider_requests"]
    assert LOGS not in json.dumps(payload["messages"])
    assert LOAD_SKILL not in _tool_names(payload)


@pytest.mark.asyncio
async def test_load_skill_reads_a_supporting_file_and_refuses_traversal(harness, skills):
    [skill_md] = [p for p in skills._iter_skill_files() if f"/{LOGS}/" in p]
    references = skill_md.rsplit("/", 1)[0] + "/references"
    os.makedirs(references)
    with open(f"{references}/flags.md", "w", encoding="utf-8") as handle:
        handle.write("use -f to force")
    harness.provider_rounds = [
        _sse_tool_calls(
            (LOAD_SKILL, {"name": LOGS, "path": "references/flags.md"}),
            (LOAD_SKILL, {"name": LOGS, "path": "../../../../etc/passwd"}),
        ),
        _sse_events(["done"]),
    ]

    await harness.send(_Request("agent"))

    _, second = harness.captured["provider_requests"]
    found, refused = _tool_messages(second)
    assert found["content"] == "use -f to force"
    assert "not found" in refused["content"]


def test_index_lists_every_name_and_descriptions_within_the_budget():
    entries = [
        {"name": "alpha", "description": "a" * 30},
        {"name": "beta", "description": "b" * 30},
        {"name": "gamma", "description": "c" * 30},
    ]

    section = prompt.skill_index_section(entries, budget=65)

    assert f"- alpha: {'a' * 30}" in section
    assert f"- beta: {'b' * 30}" in section
    assert "- gamma\n" in section + "\n" and "c" * 30 not in section
    assert prompt.skill_index_section([], budget=65) == ""


def test_skills_are_on_for_a_user_with_no_saved_prefs(tmp_path, monkeypatch):
    # Pins the relied-on assumption: _load_for_user returns {} for a user
    # without prefs, so a missing toggle reads as on (legacy parity).
    monkeypatch.setattr(prefs_routes, "PREFS_FILE", str(tmp_path / "missing.json"))

    assert prefs_routes._load_for_user("bob") == {}
    assert chat_agent_tools.skills_enabled("bob", SimpleNamespace(), incognito=False)
