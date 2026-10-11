"""Ticket #49 / ADR 0008: a Skills page publish approves a skill, bound to its content."""

import asyncio
import json

import pytest
from fastapi import HTTPException, Request
from fastapi.datastructures import State

import routes.prefs_routes as prefs_routes
from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
from routes.skills_routes import SkillUpdateRequest, setup_skills_routes
from services.memory.skills import SkillsManager

OWNER = "alice"
TOPIC = "rotate the nginx access logs"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Real SkillsManager on a tmp data dir, shared by routes, tools and the legacy loop."""
    import src.constants as constants
    from src import agent_loop

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(constants, "DATA_DIR", str(data_dir))
    monkeypatch.setenv("ODYSSEUS_SKILL_SEMANTIC_RETRIEVAL", "0")
    prefs = {"skills_enabled": True, "auto_approve_skills": True, "skill_min_confidence": 0.85}
    monkeypatch.setattr(prefs_routes, "_load_for_user", lambda user=None: dict(prefs))
    agent_loop._cached_base_prompt = None
    agent_loop._cached_base_prompt_key = None
    sm = SkillsManager(str(data_dir))
    return sm, prefs


def _request(body=None, *, internal=False, method="PUT") -> Request:
    class DummyApp:
        state = State()

    payload = json.dumps(body if body is not None else {}).encode("utf-8")
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": payload, "more_body": False}

    headers = [(b"content-type", b"application/json")]
    if internal:
        headers.append((INTERNAL_TOOL_HEADER.lower().encode(), INTERNAL_TOOL_TOKEN.encode()))
    return Request(scope={
        "type": "http", "method": method, "headers": headers,
        "app": DummyApp(), "state": {"current_user": OWNER},
    }, receive=receive)


def _handler(router, path, method):
    return next(r.endpoint for r in router.routes if r.path == path and method in r.methods)


def _add(sm, name=TOPIC.replace(" ", "-"), *, source="user", status="draft", description=TOPIC,
         when_to_use=f"when the user asks to {TOPIC}"):
    return sm.add_skill(
        name=name, description=description, owner=OWNER, source=source, status=status,
        when_to_use=when_to_use, procedure=["run logrotate -f"],
    )["name"]


def _publish(sm, name, *, internal=False):
    router = setup_skills_routes(sm)
    put = _handler(router, "/api/skills/{skill_id}", "PUT")
    return asyncio.run(put(_request({"status": "published"}, internal=internal), name,
                           SkillUpdateRequest(status="published")))


def _legacy_prompt_text(owner=OWNER, query=f"please {TOPIC}"):
    from src.agent_loop import _build_system_prompt
    out, _ = _build_system_prompt(
        messages=[{"role": "user", "content": query}], model="test-model",
        active_document=None, mcp_mgr=None, owner=owner,
    )
    return "\n".join(str(m.get("content") or "") for m in out)


def _skill(sm, name):
    return next(s for s in sm.load(owner=OWNER) if s["name"] == name)


def test_hand_published_inconclusive_skill_is_injected_on_legacy_loop(env):
    sm, _prefs = env
    name = _add(sm)
    sm.set_audit(name, "inconclusive", owner=OWNER)

    _publish(sm, name)
    text = _legacy_prompt_text()

    assert f"### {name}" in text
    assert _skill(sm, name)["uses"] == 1


def _manage_skills(args):
    from src.tools.system import do_manage_skills
    return asyncio.run(do_manage_skills(json.dumps(args), owner=OWNER))


def _list_states(sm):
    router = setup_skills_routes(sm)
    listing = asyncio.run(_handler(router, "/api/skills", "GET")(_request(method="GET")))
    return {s["name"]: s["state"] for s in listing["skills"]}


def _index_names(sm):
    return {s["name"] for s in sm.index_for(owner=OWNER)}


def test_agent_edit_lapses_user_approval(env):
    sm, _prefs = env
    name = _add(sm)
    _publish(sm, name)
    assert name in _index_names(sm)

    _manage_skills({"action": "edit", "name": name, "procedure": ["rm -rf /var/log/nginx"]})

    assert name not in _index_names(sm)
    assert _list_states(sm)[name] == "awaiting_audit"
    assert f"### {name}" not in _legacy_prompt_text()


def test_agent_published_skill_waits_for_a_passing_audit(env):
    sm, _prefs = env
    name = _add(sm, source="learned")
    _manage_skills({"action": "publish", "name": name})
    assert name not in _index_names(sm)
    assert _list_states(sm)[name] == "awaiting_audit"

    sm.set_audit(name, "pass", owner=OWNER)
    sm.update_skill(name, {"confidence": 0.95}, owner=OWNER)

    assert name in _index_names(sm)
    assert _list_states(sm)[name] == "published"


def test_teacher_draft_awaits_the_audit_outside_the_index(env):
    sm, _prefs = env
    name = _add(sm, source="teacher-escalation")

    assert name not in _index_names(sm)
    assert _list_states(sm)[name] == "awaiting_audit"


def test_internal_tool_cannot_publish_from_the_skills_route(env):
    sm, _prefs = env
    name = _add(sm)

    with pytest.raises(HTTPException) as exc:
        _publish(sm, name, internal=True)

    assert exc.value.status_code == 403
    assert _skill(sm, name)["status"] == "draft"
    assert not _skill(sm, name)["user_approved"]


def test_user_only_skill_is_hidden_from_the_model_but_slash_starts_it(env):
    sm, _prefs = env
    name = _add(sm)
    router = setup_skills_routes(sm)
    put = _handler(router, "/api/skills/{skill_id}", "PUT")
    asyncio.run(put(_request(), name, SkillUpdateRequest(disable_model_invocation=True)))
    _publish(sm, name)
    _manage_skills({"action": "edit", "name": name, "tags": ["nginx"]})

    assert "disable-model-invocation: true" in sm.read_skill_md(name, owner=OWNER)
    assert name not in _index_names(sm)
    assert name not in _legacy_prompt_text()
    assert name not in _manage_skills({"action": "list"}).get("results", "")
    assert name not in _manage_skills({"action": "search", "query": TOPIC}).get("results", "")
    assert "error" in _manage_skills({"action": "view", "name": name})

    catalog = asyncio.run(_handler(router, "/api/skills/slash-catalog", "GET")(_request(method="GET")))
    assert f"/{name}" in {e["token"] for e in catalog["skills"]}
    invoke = _handler(router, "/api/skills/{skill_id}/invoke", "POST")
    started = asyncio.run(invoke(_request({"request": "now"}, method="POST"), name))
    assert "run logrotate -f" in started["message"]
    assert _skill(sm, name)["uses"] == 1


def _builtin(sm, name="shipped-procedure"):
    from services.memory.skill_format import Skill
    sm.sync_builtin_skill(Skill(name=name, description="shipped", status="published",
                                source="builtin", procedure=["do it"]))
    return name


def _audit_passed(sm, name, confidence=0.95, status="published"):
    name = _add(sm, name=name, description=f"{name} procedure", source="learned", status=status,
                when_to_use=f"only for {name}")
    sm.set_audit(name, "pass", owner=OWNER)
    sm.update_skill(name, {"confidence": confidence}, owner=OWNER)
    return name


def test_auto_approve_off_leaves_only_builtins_and_user_approved(env):
    sm, prefs = env
    builtin = _builtin(sm)
    approved = _add(sm)
    _publish(sm, approved)
    audited = _audit_passed(sm, "audited-skill")
    assert _index_names(sm) == {builtin, approved, audited}
    prefs["auto_approve_skills"] = False

    assert _index_names(sm) == {builtin, approved}


def test_threshold_zero_admits_low_confidence_passes_but_never_drafts(env):
    sm, prefs = env
    low = _audit_passed(sm, "low-confidence-pass", confidence=0.1)
    draft = _audit_passed(sm, "passing-draft", status="draft")
    prefs["skill_min_confidence"] = 0

    assert low in _index_names(sm)
    assert draft not in _index_names(sm)
    assert _list_states(sm)[draft] == "draft"


def test_renaming_the_user_keeps_the_approval(env, monkeypatch):
    import sys
    import types
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import core.database as cdb
    import routes.auth_routes as ar

    sm, _prefs = env
    name = _add(sm)
    sm.set_audit(name, "inconclusive", owner=OWNER)
    _publish(sm, name)

    monkeypatch.setattr(cdb, "SessionLocal", lambda: MagicMock())
    monkeypatch.setattr(cdb, "Base", SimpleNamespace(registry=SimpleNamespace(mappers=[])), raising=False)
    fake_prefs = types.ModuleType("routes.prefs_routes")
    fake_prefs._load = lambda: {}
    fake_prefs._save = lambda d: None
    fake_prefs._load_for_user = lambda user=None: {"skills_enabled": True}
    monkeypatch.setitem(sys.modules, "routes.prefs_routes", fake_prefs)
    monkeypatch.setattr(ar, "SKILLS_DIR", sm.skills_root)
    monkeypatch.setattr(ar, "DEEP_RESEARCH_DIR", str(sm.data_dir) + "/deep_research")
    monkeypatch.setattr(ar, "MEMORY_FILE", str(sm.data_dir) + "/memory.json")
    am = MagicMock()
    am.is_admin.return_value = True
    am.get_username_for_token.return_value = "admin"
    am.users = {OWNER: {}}
    am.rename_user.return_value = True
    rename = next(r.endpoint for r in ar.setup_auth_routes(am).routes
                  if getattr(r.endpoint, "__name__", "") == "rename_user")
    state = SimpleNamespace(invalidate_token_cache=lambda: None, session_manager=None,
                            research_handler=None, upload_handler=None, personal_docs_manager=None)
    asyncio.run(rename(OWNER, SimpleNamespace(username="alice2"), SimpleNamespace(
        cookies={"odysseus_session": "t"}, app=SimpleNamespace(state=state),
        state=SimpleNamespace(current_user="admin"))))

    assert name in {s["name"] for s in sm.index_for(owner="alice2")}


def test_skills_page_save_restamps_and_unpublish_withdraws(env):
    sm, _prefs = env
    name = _add(sm)
    _publish(sm, name)
    _manage_skills({"action": "edit", "name": name, "procedure": ["logrotate -d first"]})
    assert name not in _index_names(sm)

    router = setup_skills_routes(sm)
    save = _handler(router, "/api/skills/{skill_id}/markdown", "POST")
    md = sm.read_skill_md(name, owner=OWNER)
    asyncio.run(save(_request({"markdown": md}, method="POST"), name))
    assert name in _index_names(sm)

    put = _handler(router, "/api/skills/{skill_id}", "PUT")
    asyncio.run(put(_request(), name, SkillUpdateRequest(status="draft")))
    _manage_skills({"action": "publish", "name": name})
    assert name not in _index_names(sm)


def test_user_approved_skills_leave_the_automatic_audit(env):
    from services.memory.skill_lifecycle import automatic_audit_candidates
    sm, _prefs = env
    approved = _add(sm)
    _publish(sm, approved)
    pending = _add(sm, name="agent-skill", description="agent skill", source="learned",
                   when_to_use="only for the agent skill")

    names = {s["name"] for s in automatic_audit_candidates(sm.load(owner=OWNER))}

    assert names == {pending}


def test_manual_audit_of_a_user_approved_skill_only_records_the_badge(env, monkeypatch):
    import routes.skills_routes as skills_routes
    sm, _prefs = env
    name = _add(sm)
    _publish(sm, name)
    md_before = sm.read_skill_md(name, owner=OWNER)

    async def failing_arm(*_args, **_kwargs):
        return "transcript", {}, None

    async def failing_judge(*_args, **_kwargs):
        return {"verdict": "fail", "confidence": 1.0, "summary": "wrong command", "issues": ["fix it"]}

    async def rewrite(*_args, **_kwargs):
        return md_before.replace("logrotate -f", "logrotate -v")

    async def no_necessity(*_args, **_kwargs):
        return None

    monkeypatch.setattr(skills_routes, "_run_skill_audit_arm", failing_arm)
    monkeypatch.setattr(skills_routes, "_eval_skill_run", failing_judge)
    monkeypatch.setattr(skills_routes, "_improve_skill_md", rewrite)
    monkeypatch.setattr(skills_routes, "_eval_skill_necessity", no_necessity)
    monkeypatch.setattr(skills_routes, "_should_check_retrieval_precision", lambda skill: True)
    monkeypatch.setattr(skills_routes, "_eval_skill_retrieval_precision", failing_judge)

    results = [asyncio.run(skills_routes._audit_one_skill(
        sm, _skill(sm, name), "http://judge", "judge-model", {}, None, OWNER, lambda msg: None))]
    skills_routes._finalize_audit_batch(sm, results, OWNER, lambda msg: None)

    after = _skill(sm, name)
    assert sm.read_skill_md(name, owner=OWNER) == md_before
    assert after["audit_verdict"] == "fail"
    assert after["status"] == "published"
    assert name in _index_names(sm)
