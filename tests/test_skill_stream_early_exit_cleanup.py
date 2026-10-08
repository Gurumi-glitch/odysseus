"""Skill runners that stop at a tool approval must close the agent stream.

Abandoning the stream left its ContextVar cleanup to asyncio's finalizer in
another Context, which raised "created in a different Context" and kept the
run's request authority bound in the consuming task (fork #7).
"""
import asyncio
import gc
import json

import pytest

import routes.skills_routes as skills_routes
from src.agent_runtime.authority import (
    active_request_authority,
    with_request_authority,
)
from src.turn_contract import active_turn_contract, with_turn_contract

APPROVAL = {
    "kind": "tool_approval",
    "approval_id": "opaque",
    "question": "Allow this exact action once?",
}


@with_request_authority
@with_turn_contract
async def _stream_stopping_at_approval(url, model, messages, *, request_authority=None,
                                       turn_contract=None, **kwargs):
    yield "data: " + json.dumps({
        "type": "tool_output",
        "tool": "bash",
        "output": "Waiting for an exact user approval.",
        "ask_user": APPROVAL,
    })
    yield "data: " + json.dumps({"delta": "never consumed"})


def _run_and_collect_loop_errors(runner):
    """Run *runner* in one task; return its result, post-run context, loop errors."""
    errors = []

    async def main():
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: errors.append(context))
        result = await runner()
        bound_after = (active_request_authority(), active_turn_contract())
        gc.collect()
        for _ in range(3):
            await asyncio.sleep(0)
        gc.collect()
        return result, bound_after

    result, bound_after = asyncio.run(main())
    return result, bound_after, errors


@pytest.fixture
def stub_stream(monkeypatch):
    monkeypatch.setattr("src.agent_loop.stream_agent_loop", _stream_stopping_at_approval)


def test_skill_test_job_closes_stream_when_pausing_for_approval(stub_stream):
    key = ("owner", "early-exit-skill")
    skills_routes._skill_test_jobs[key] = {"status": "running", "log": [], "verdict": None}
    try:
        _, bound_after, errors = _run_and_collect_loop_errors(lambda: skills_routes._run_skill_test_job(
            key, "early-exit-skill", "skill markdown", "task",
            "http://example.test", "model", None, "owner",
        ))
        job = skills_routes._skill_test_jobs[key]
        assert job["status"] == "awaiting_approval"
        assert job["approval"] == APPROVAL
    finally:
        skills_routes._skill_test_jobs.pop(key, None)

    assert bound_after == (None, None)
    assert errors == []


def test_skill_audit_arm_closes_stream_when_stopping_at_approval(stub_stream):
    async def run_arm():
        return await skills_routes._run_skill_audit_arm(
            [], "http://example.test", "model", {}, "owner")

    (transcript, _stats, approval), bound_after, errors = _run_and_collect_loop_errors(run_arm)

    assert approval == APPROVAL
    assert "never consumed" not in transcript
    assert bound_after == (None, None)
    assert errors == []
