"""Agent stream consumers that stop early must close the stream in their own task
(fork #7)."""
import asyncio
import gc
import json

import pytest

import routes.chat_routes as chat_routes
import routes.skills_routes as skills_routes
from src.agent_runtime.authority import (
    RequestAuthority,
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


def test_chat_bridge_closes_stream_when_client_disconnects(monkeypatch):
    monkeypatch.setattr(chat_routes, "stream_agent_loop", _stream_stopping_at_approval)

    async def disconnect_after_first_chunk():
        stream = chat_routes._stream_agent_with_execution_bridge(
            None, "http://example.test", "model", [], owner="owner",
            request_authority=RequestAuthority.empty(owner="owner"))
        first = await anext(stream)
        await stream.aclose()
        return first

    first, bound_after, errors = _run_and_collect_loop_errors(disconnect_after_first_chunk)

    assert "Waiting for an exact user approval." in first
    assert bound_after == (None, None)
    assert errors == []
