"""Backend gaps found by the #65 UI pass (spec #44), driven through the chat route."""

import pytest

from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _sse_tool_calls,
    _switch_on,
)
from tests.chat_agent.test_chat_agent_tools_route import db  # noqa: F401  (fixture)


@pytest.fixture
def harness(monkeypatch, tmp_path, db):  # noqa: F811
    built = _Harness(monkeypatch, tmp_path)
    _switch_on()
    return built


def _tool_pairs(events):
    return [(e["type"], e["call_id"]) for e in events if e.get("type") in ("tool_start", "tool_output")]


@pytest.mark.asyncio
async def test_refused_and_unoffered_calls_get_a_tool_start_before_their_output(harness):
    # A denied call (web is off) and an unoffered tool: the card for each call
    # opens on tool_start, so a refusal must not arrive as an orphan output.
    harness.provider_rounds = [
        _sse_tool_calls(("web_fetch", {"url": "https://example.com"}), ("no_such_tool", {})),
        _sse_events(["Could not do that."]),
    ]

    _, body = await harness.send(_Request("agent", allow_web_search="false"))

    events = _data_events(body)
    assert _tool_pairs(events) == [
        ("tool_start", "call_0"), ("tool_output", "call_0"),
        ("tool_start", "call_1"), ("tool_output", "call_1"),
    ]
    first, second = [e for e in events if e.get("type") == "tool_start"]
    assert (first["tool"], first["round"]) == ("web_fetch", 1)
    assert "example.com" in first["full_command"]
    assert second["tool"] == "no_such_tool"
