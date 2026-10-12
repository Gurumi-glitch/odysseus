"""Backend gaps found by the #65 UI pass (spec #44), driven through the chat route."""

from types import SimpleNamespace

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


@pytest.mark.parametrize("form_source, saved", [("continue", "continue"), ("merge", None), ("", None)])
@pytest.mark.asyncio
async def test_continue_marks_the_new_turn_user_row_as_continue(harness, form_source, saved):
    # Spec #44 Transcript: Continue after a capped turn sends a new turn whose
    # user row records metadata.source = "continue"; no other value is stored.
    await harness.send(_Request("agent", source=form_source))

    assert harness.captured["build_context"]["source"] == saved


@pytest.mark.asyncio
async def test_turn_end_metrics_are_saved_flat_on_the_assistant_row(harness):
    # The history renderer calls displayMetrics(wrap, metadata): legacy saves
    # the final metrics flat into the row's metadata (save_assistant_response).
    _, body = await harness.send(_Request("agent"))

    [metrics] = [e["data"] for e in _data_events(body) if e.get("type") == "metrics"]
    [row] = harness.saved
    assert metrics.keys() <= row.metadata.keys()
    for key in ("response_time", "input_tokens", "output_tokens", "tokens_per_second", "requested_model"):
        assert row.metadata[key] == metrics[key]
    # Projections still own their keys, and the record still verifies.
    assert row.metadata["runtime"] == "chat_agent"
    from src.chat_agent.transcript import trusted_rounds
    assert trusted_rounds({"content": row.content, "metadata": row.metadata}) is not None


def test_add_user_message_stores_the_source_on_the_user_row():
    from routes.chat_helpers import PreprocessedMessage, add_user_message

    rows = []
    sess = SimpleNamespace(add_message=rows.append)
    handler = SimpleNamespace(update_session_name_if_needed=lambda *a, **k: None)
    preprocessed = PreprocessedMessage(
        enhanced_message="Continue", user_content="Continue", text_for_context="Continue",
        youtube_transcripts=[], attachment_meta=[],
    )

    add_user_message(sess, handler, preprocessed, interaction_mode="agent", source="continue")
    add_user_message(sess, handler, preprocessed, interaction_mode="agent")

    assert [row.metadata for row in rows] == [
        {"interaction_mode": "agent", "source": "continue"}, {"interaction_mode": "agent"},
    ]
