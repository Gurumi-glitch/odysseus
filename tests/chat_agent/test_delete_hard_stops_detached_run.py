"""DELETE /api/session relies on agent_runs.stop_and_wait reaching a run whose
provider read never returns (a stuck cloud stream). Pins that assumption for
both runtimes: the cancel lands inside the parked generator and the run ends.
"""

import asyncio

import pytest

from src import agent_runs


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ["legacy", "chat_agent"])
async def test_stop_and_wait_cancels_a_run_parked_on_a_silent_provider(runtime):
    cancelled_inside_stream = asyncio.Event()

    async def stuck_stream():
        yield 'data: {"delta": "partial"}\n\n'
        try:
            await asyncio.Event().wait()  # the provider never sends another byte
        except asyncio.CancelledError:
            cancelled_inside_stream.set()
            raise

    session_id = f"delete-hard-stop-{runtime}"
    run = agent_runs.start(session_id, stuck_stream(), runtime=runtime)
    try:
        while not run.buffer:
            await asyncio.sleep(0)
        assert agent_runs.is_active(session_id)

        await agent_runs.stop_and_wait(session_id)

        assert cancelled_inside_stream.is_set()
        assert run.status == "stopped"
        assert not agent_runs.is_active(session_id)
        assert agent_runs.get_active_run(session_id) is None
    finally:
        if run.evict_task is not None:
            run.evict_task.cancel()
        agent_runs._RUNS.pop(session_id, None)


@pytest.mark.asyncio
async def test_stop_and_wait_without_a_run_is_a_no_op():
    await agent_runs.stop_and_wait("never-started")
    assert agent_runs.get_status("never-started") is None
