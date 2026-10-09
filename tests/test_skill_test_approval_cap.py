"""A manual skill test stops after a bounded number of exact approvals (fork #15)."""
import asyncio
import json

import pytest

import routes.skills_routes as skills_routes
from src.agent_runtime.authority import with_request_authority


def _approval(n):
    return {"kind": "tool_approval", "approval_id": f"approval-{n}",
            "question": "Allow this exact action once?"}


@pytest.fixture
def approval_stream(monkeypatch):
    issued = []

    @with_request_authority
    async def _stream(url, model, messages, *, request_authority=None, **kwargs):
        issued.append(_approval(len(issued) + 1))
        yield "data: " + json.dumps({
            "type": "tool_output", "tool": "manage_mcp",
            "output": "Waiting for an exact user approval.", "ask_user": issued[-1],
        })

    monkeypatch.setattr("src.agent_loop.stream_agent_loop", _stream)
    return issued


@pytest.fixture
def retired(monkeypatch):
    from src.tool_approvals import tool_approval_store
    calls = []
    monkeypatch.setattr(tool_approval_store, "consume",
                        lambda approval_id, **kwargs: calls.append((approval_id, kwargs["decision"])))
    return calls


def _run(key):
    asyncio.run(skills_routes._run_skill_test_job(
        key, "looping-skill", "skill markdown", "task",
        "http://example.test", "model", None, "owner",
    ))
    return skills_routes._skill_test_jobs[key]


def test_skill_test_stops_once_approvals_exceed_the_cap(approval_stream, retired):
    key = ("owner", "looping-skill")
    skills_routes._skill_test_jobs[key] = {"status": "running", "log": [], "verdict": None}
    try:
        for n in range(1, skills_routes.SKILL_TEST_MAX_APPROVALS + 1):
            job = _run(key)
            assert job["status"] == "awaiting_approval", n
            assert job["approval"] == approval_stream[-1]

        job = _run(key)
    finally:
        skills_routes._skill_test_jobs.pop(key, None)

    assert skills_routes.SKILL_TEST_MAX_APPROVALS == 5
    assert job["status"] == "done"
    assert "approval" not in job and "_transcript" not in job
    assert job["verdict"]["verdict"] == "inconclusive"
    assert "5" in job["verdict"]["summary"]
    assert retired == [(approval_stream[-1]["approval_id"], "deny")]


def _run_past_the_cap(key):
    skills_routes._skill_test_jobs[key] = {
        "status": "running", "log": [], "verdict": None,
        "approvals_requested": skills_routes.SKILL_TEST_MAX_APPROVALS,
    }
    try:
        return _run(key)
    finally:
        skills_routes._skill_test_jobs.pop(key, None)


def test_cap_stop_is_logged_as_its_own_event(approval_stream, retired):
    job = _run_past_the_cap(("owner", "cap-log-skill"))

    assert [entry["type"] for entry in job["log"]][-1] == "approval_limit"
    assert not any(entry["type"] == "approval_denied" for entry in job["log"])


def test_cap_stops_the_test_even_if_retiring_the_approval_fails(approval_stream, monkeypatch):
    from src.tool_approvals import tool_approval_store

    def _broken_consume(*args, **kwargs):
        raise RuntimeError("approval store unavailable")

    async def _no_grading(*args, **kwargs):
        raise AssertionError("a capped test must not be graded")

    monkeypatch.setattr(tool_approval_store, "consume", _broken_consume)
    monkeypatch.setattr(skills_routes, "_eval_skill_run", _no_grading)

    job = _run_past_the_cap(("owner", "cap-retire-fails-skill"))

    assert job["status"] == "done"
    assert job["verdict"]["verdict"] == "inconclusive"
