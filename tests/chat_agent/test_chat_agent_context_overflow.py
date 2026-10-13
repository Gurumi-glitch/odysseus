"""Context overflow policy per chat on the new runtime (ticket #95, spec #89).

Route tests drive a multi-round tool turn through POST /api/chat_stream whose
later request crosses the gate, and assert what each policy sends, drops and
reports; the settings route is driven over HTTP.
"""

import json

import pytest

from src.chat_agent import loop, store
from tests.chat_agent.test_chat_agent_route import _data_events, _Request, _sse_events, _sse_tool_calls, _switch_on
from tests.chat_agent.test_chat_agent_route import harness  # noqa: F401
from tests.chat_agent.test_chat_agent_fallback import harness as fallback_harness  # noqa: F401
from tests.chat_agent.test_chat_agent_thinking_level import api  # noqa: F401

# Bonsai's llama-server window (spec #89 acceptance setup); the harness's
# compact tool set takes ~2.6k of it, the gate sits at 85% (~13.9k).
WINDOW = 16_384
# "a " estimates at ~0.6 tokens: one padded call ~6k, two ~12k.
PAD = 10_000


def _window(monkeypatch, window=WINDOW, measured=True):
    monkeypatch.setattr(loop, "get_context_length_known", lambda url, model: (window, measured))


def _policy(value):
    store.set_chat_settings("alice", "session-1", {store.CONTEXT_OVERFLOW: value})


def _padded_call(letter, words=PAD):
    # A refused call: the large part of the request is the call itself, so no
    # tool has to run for the turn to cross the gate on its third round.
    return _sse_tool_calls(("no_such_tool", {"pad": f"{letter} " * words}))


def _three_rounds(harness, second_words=PAD):
    harness.provider_rounds = [_padded_call("a"), _padded_call("b", second_words), _sse_events(["Done."])]


def _texts(payload):
    texts = []
    for message in payload["messages"]:
        content = message.get("content")
        texts.append(content if isinstance(content, str) else json.dumps(content))
        for call in message.get("tool_calls") or ():
            texts.append(call["function"]["arguments"])
    return texts


def _has(payload, needle):
    return any(needle in text for text in _texts(payload))


def _error(body):
    lines = body.splitlines()
    return json.loads(lines[lines.index("event: error") + 1][len("data: "):])["error"]


def _trimmed(body):
    return [event for event in _data_events(body) if event.get("type") == "trimmed"]


@pytest.mark.asyncio
async def test_auto_sends_later_rounds_over_the_gate_as_built(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch)
    _three_rounds(harness)

    response, body = await harness.send(_Request("agent"))

    requests = harness.captured["provider_requests"]
    assert len(requests) == 3
    assert _has(requests[2], "a a a") and _has(requests[2], "earlier question")
    assert not _trimmed(body)
    assert harness.saved[-1].metadata.get("error") is None


@pytest.mark.asyncio
async def test_stop_at_limit_ends_the_turn_when_a_later_round_crosses_the_gate(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch)
    _policy("stop_at_limit")
    _three_rounds(harness)

    response, body = await harness.send(_Request("agent"))

    assert len(harness.captured["provider_requests"]) == 2
    error = _error(body)
    assert error.endswith(f" tokens exceeds the {WINDOW}-token window")
    assert harness.saved[-1].metadata["error"] == error


@pytest.mark.asyncio
async def test_rolling_window_drops_earlier_turns_then_earlier_tool_rounds(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch)
    _policy("rolling_window")
    _three_rounds(harness)

    response, body = await harness.send(_Request("agent"))

    requests = harness.captured["provider_requests"]
    assert len(requests) == 3
    third = requests[2]
    # Earlier turn (2 messages) and the first tool round (call + result) gone.
    assert not _has(third, "earlier question") and not _has(third, "earlier answer")
    assert not _has(third, "a a a")
    roles = [m["role"] for m in third["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert third["messages"][0] == requests[0]["messages"][0]
    assert third["messages"][1] == requests[0]["messages"][-1]
    assert _has(third, "b b b") and third["tools"] == requests[0]["tools"]
    assert _trimmed(body) == [{"type": "trimmed", "dropped": 4, "message": "dropped 4 messages to fit the window"}]
    # Request-local: the stored turn keeps both rounds, and no notice is saved.
    row = harness.saved[-1]
    assert row.metadata.get("error") is None
    assert len(row.metadata["agent_v2"]["rounds"]) == 3
    assert "trimmed" not in json.dumps(row.metadata)


@pytest.mark.asyncio
async def test_truncate_middle_also_keeps_the_oldest_user_message(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch)
    _policy("truncate_middle")
    _three_rounds(harness)

    response, body = await harness.send(_Request("agent"))

    third = harness.captured["provider_requests"][2]
    assert _has(third, "earlier question")
    assert not _has(third, "earlier answer") and not _has(third, "a a a")
    assert [m["role"] for m in third["messages"]] == ["system", "user", "user", "assistant", "tool"]
    assert [event["dropped"] for event in _trimmed(body)] == [3]


@pytest.mark.asyncio
async def test_a_compact_summary_is_never_dropped(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch)
    _policy("rolling_window")
    harness.context.history = [
        {"role": "system", "content": "[Conversation summary] the user likes tea"},
        *harness.context.history,
    ]
    _three_rounds(harness)

    await harness.send(_Request("agent"))

    third = harness.captured["provider_requests"][2]
    assert _has(third, "the user likes tea") and not _has(third, "earlier question")


@pytest.mark.asyncio
async def test_a_fallback_after_trimming_regates_later_rounds_from_the_untrimmed_chat(
    fallback_harness, monkeypatch,
):
    from tests.chat_agent.test_chat_agent_fallback import BACKUP_MODEL, PRIMARY_MODEL, _script

    _switch_on()
    # Only the primary's window is small: on the backup nothing needs dropping.
    windows = {PRIMARY_MODEL: WINDOW, BACKUP_MODEL: 128_000}
    monkeypatch.setattr(loop, "get_context_length_known", lambda url, model: (windows[model], True))
    _policy("rolling_window")
    _script(monkeypatch, fallback_harness, {
        PRIMARY_MODEL: [_padded_call("a"), _padded_call("b"), 503],
        BACKUP_MODEL: [_sse_tool_calls(("no_such_tool", {})), _sse_events(["Done."])],
    })

    response, body = await fallback_harness.send(_Request("agent"))

    requests = fallback_harness.captured["provider_requests"]
    assert [request["model"] for request in requests] == [PRIMARY_MODEL] * 3 + [BACKUP_MODEL] * 2
    assert not _has(requests[2], "a a a")
    # The trim was the primary's request alone; the backup's next round has it all.
    assert _has(requests[4], "earlier question") and _has(requests[4], "a a a")


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["rolling_window", "truncate_middle"])
async def test_still_over_after_trimming_is_the_stop_at_limit_error(harness, monkeypatch, policy):
    _switch_on()
    _window(monkeypatch)
    _policy(policy)
    # The latest tool round alone is over the gate: nothing droppable fits it.
    _three_rounds(harness, second_words=3 * PAD)

    response, body = await harness.send(_Request("agent"))

    assert len(harness.captured["provider_requests"]) == 2
    assert _error(body).endswith(f" tokens exceeds the {WINDOW}-token window")
    assert not _trimmed(body)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["auto", "stop_at_limit", "rolling_window", "truncate_middle"])
async def test_an_unmeasured_window_never_gates(harness, monkeypatch, policy):
    _switch_on()
    _window(monkeypatch, measured=False)
    _policy(policy)
    _three_rounds(harness)

    response, body = await harness.send(_Request("agent"))

    assert len(harness.captured["provider_requests"]) == 3
    assert not _trimmed(body)


# --- Settings route -------------------------------------------------------------


async def _overflow(api, chat, user="alice"):
    response = await api.get("/api/chat_agent/settings", params={"session_id": chat}, headers={"X-User": user})
    assert response.status_code == 200
    return response.json()["context_overflow"]


async def _set(api, body):
    return await api.put("/api/chat_agent/settings", json=body)


@pytest.mark.asyncio
async def test_overflow_is_stored_per_chat_and_new_chats_start_with_the_last_used(api):
    assert await _overflow(api, "chat-1") == "auto"

    response = await _set(api, {"session_id": "chat-1", "context_overflow": "rolling_window"})
    assert response.status_code == 200
    assert response.json()["context_overflow"] == "rolling_window"

    assert await _overflow(api, "chat-2") == "rolling_window"
    assert (await _set(api, {"session_id": "chat-2", "context_overflow": "stop_at_limit"})).status_code == 200
    assert await _overflow(api, "chat-1") == "rolling_window"
    assert await _overflow(api, "chat-3") == "stop_at_limit"
    assert await _overflow(api, "chat-1", user="bob") == "auto"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"session_id": "chat-1", "context_overflow": "truncate"},
        {"context_overflow": "rolling_window"},
        # Nothing is written when one of two values is bad.
        {"session_id": "chat-1", "thinking_level": "high", "context_overflow": "drop"},
    ],
    ids=["unknown", "no-chat", "one-bad-of-two"],
)
async def test_settings_write_validates_the_overflow_policy(api, body):
    response = await _set(api, body)

    assert response.status_code == 400
    settings = (await api.get("/api/chat_agent/settings", params={"session_id": "chat-1"})).json()
    assert settings["context_overflow"] == "auto" and settings["thinking_level"] == "auto"


@pytest.mark.asyncio
async def test_both_chat_settings_write_in_one_request(api):
    response = await _set(api, {"session_id": "chat-1", "thinking_level": "low", "context_overflow": "truncate_middle"})

    assert response.status_code == 200
    assert response.json()["thinking_level"] == "low"
    assert response.json()["context_overflow"] == "truncate_middle"
