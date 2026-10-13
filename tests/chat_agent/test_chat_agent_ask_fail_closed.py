"""A permission ask runs the tool only on an explicit allow (spec item E, review S1).

The reply is injected where the loop receives it (``asks.wait``), so the loop's
own mapping from reply to decision is what is under test, not the HTTP
endpoint's validation of decision values.
"""

import pytest

from src.chat_agent import asks
from tests.chat_agent.test_chat_agent_asks_route import (  # noqa: F401 (fixtures)
    _bash,
    _events,
    _tool_messages,
    admin,
    executor,
    harness,
    send_and_answer,
    workspace,
)
from tests.chat_agent.test_chat_agent_route import _Request, _sse_events


async def _ignore(card):
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "ONCE", "", "answer"])
async def test_an_unknown_reply_decision_denies_the_tool(
    monkeypatch, harness, admin, executor, workspace, decision,
):
    async def reply(session_id, owner, card):
        return asks.Answer(decision)

    monkeypatch.setattr(asks, "wait", reply)
    harness.provider_rounds = [_bash("ls"), _sse_events(["I did not run it."])]

    body = await send_and_answer(harness, _Request("agent", allow_bash="true", workspace=workspace), _ignore)

    assert executor == []
    [output] = _events(body, "tool_output")
    assert output["exit_code"] == 126
    assert output["execution_attempted"] is False
    _, second = harness.captured["provider_requests"]
    assert "denied" in _tool_messages(second)[0].lower()
