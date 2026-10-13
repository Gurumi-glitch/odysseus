"""Permission gate of the new runtime as table tests: (call, turn) -> decision."""

import json

import pytest

import src.tool_security as tool_security
from src.agent_tools import FUNCTION_TOOL_SCHEMAS
from src.chat_agent import permissions
from src.chat_agent.permissions import ALLOW, ASK, DENY
from src.chat_agent.routing import ComposerToggles
from src.tool_security import BUILTIN_EMAIL_TOOLS


def _toggles(**overrides):
    values = dict(mode="agent", allow_bash=None, allow_web_search="true", use_web=None,
                  plan_mode=False, incognito=False, workspace=None)
    values.update(overrides)
    return ComposerToggles(**values)


def _layers(admin=True, delegated=False, privileges=None, email_open=False,
            global_disabled=None, **toggles):
    return permissions.build_deny_layers(
        owner="alice", toggles=_toggles(**toggles), privileges=privileges or {},
        delegated_credential=delegated, active_email_open=email_open,
        global_disabled_tools=global_disabled or [], rules=permissions.Rules(),
    )


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)


@pytest.fixture
def non_admin(monkeypatch):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: False)


def test_every_native_and_email_tool_has_a_written_posture():
    names = {s["function"]["name"] for s in FUNCTION_TOOL_SCHEMAS} | BUILTIN_EMAIL_TOOLS
    assert sorted(names - set(permissions.POSTURE)) == []


# Written out on purpose: the sensitive and protected sets are derived from
# POSTURE, so this is the independent record of what they were before (#03).
_SENSITIVE_NAMES = {
    "send_email", "reply_to_email", "bulk_email", "delete_email", "unsubscribe_email",
    "block_sender", "manage_settings", "manage_endpoints", "manage_mcp", "manage_webhooks",
    "manage_tokens", "vault_unlock", "vault_get",
}
_PROTECTED_NAMES = {
    "manage_settings", "manage_endpoints", "manage_mcp", "manage_webhooks", "manage_tokens",
    "vault_unlock", "vault_get", "bulk_email",
}
_NON_SENSITIVE_ASK_NAMES = {
    "write_file", "edit_file", "apply_patch", "download_attachment",
    "bash", "python", "host_shell", "manage_bg_jobs", "api_call", "app_api",
    "download_model", "serve_model", "serve_preset", "stop_served_model",
    "adopt_served_model", "cancel_download",
}


def test_sensitive_and_protected_sets_are_pinned():
    assert set(permissions._SENSITIVE_TOOLS) == _SENSITIVE_NAMES
    assert set(permissions._PROTECTED_TOOLS) == _PROTECTED_NAMES


def test_protected_tools_are_a_subset_of_sensitive_tools():
    assert permissions._PROTECTED_TOOLS <= permissions._SENSITIVE_TOOLS


def test_posture_stays_allow_or_ask_and_asks_exactly_where_it_did():
    # posture() hands back the table's value for a tool; the finer table levels
    # must never leak out of it.
    asking = _SENSITIVE_NAMES | _NON_SENSITIVE_ASK_NAMES
    for tool in permissions.POSTURE:
        expected = ASK if tool in asking else ALLOW
        assert permissions.posture(tool, "") == expected, tool


@pytest.mark.parametrize(("name", "content", "expected"), [
    ("manage_notes", json.dumps({"action": "add", "title": "dinner time with mom"}), ALLOW),
    ("manage_notes", json.dumps({"action": "delete", "id": "n1"}), ASK),
    ("manage_calendar", json.dumps({"action": "delete_event", "uid": "e1"}), ASK),
    ("manage_tasks", json.dumps({"action": "list"}), ALLOW),
    ("manage_tasks", json.dumps({"action": "create", "task": "daily news"}), ASK),
    ("manage_tasks", json.dumps({"task": "implicit create"}), ASK),
    ("manage_tasks", json.dumps({"action": "run", "task_id": "t1"}), ASK),
    ("ui_control", "toggle web on", ASK),
    ("ui_control", "toggle shell on", ASK),
    ("ui_control", "toggle incognito off", ASK),
    ("ui_control", "toggle web off", ALLOW),
    ("ui_control", "toggle incognito on", ALLOW),
    ("ui_control", "switch_model grok-4", ASK),
    ("ui_control", "open_panel notes", ALLOW),
    ("web_fetch", json.dumps({"url": "https://example.com"}), ALLOW),
    ("read_file", "notes.txt", ALLOW),
    ("write_file", "notes.txt\nhello", ASK),
    ("bash", "ls", ASK),
    ("mcp__email__list_emails", "{}", ALLOW),
    ("mcp__email__send_email", "{}", ASK),
    ("mcp__email__delete_email", "{}", ASK),
    ("mcp__email__archive_email", "{}", ALLOW),
    ("mcp__github__create_issue", "{}", ASK),
    ("mcp__builtin_browser__browser_snapshot", "{}", ALLOW),
    ("manage_settings", json.dumps({"action": "get", "key": "theme"}), ASK),
])
def test_default_posture(admin, name, content, expected):
    assert permissions.decide(_layers(), name, content).outcome == expected


@pytest.mark.parametrize(("layers", "name", "expected"), [
    ({}, "bash", ASK),
    ({"allow_bash": "false"}, "bash", DENY),
    ({"allow_web_search": "false"}, "web_search", DENY),
    ({"allow_web_search": "false"}, "private_browser", DENY),
    ({"allow_web_search": None, "use_web": "true"}, "web_search", ALLOW),
    ({"incognito": True}, "manage_memory", DENY),
    ({"incognito": True}, "search_chats", DENY),
    ({"email_open": True}, "mcp__email__send_email", DENY),
    ({"plan_mode": True}, "manage_notes", DENY),
    ({"plan_mode": True}, "read_file", ALLOW),
    ({"global_disabled": ["send_email"]}, "mcp__email__send_email", DENY),
    ({"privileges": {"can_use_bash": False}}, "read_file", DENY),
    ({"privileges": {"can_manage_memory": False}}, "manage_memory", DENY),
])
def test_deny_layers(admin, layers, name, expected):
    assert permissions.decide(_layers(**layers), name, "{}").outcome == expected


def test_non_admin_never_reaches_admin_tools_or_custom_mcp(non_admin):
    layers = _layers()
    assert "bash" in layers.hidden
    assert permissions.decide(layers, "mcp__github__create_issue", "{}").outcome == DENY
    assert permissions.decide(layers, "mcp__email__list_emails", "{}").outcome == DENY
    assert permissions.decide(layers, "manage_notes", json.dumps({"action": "list"})).outcome == ALLOW


def test_delegated_token_cannot_read_the_workspace_through_any_tool(admin):
    layers = _layers(delegated=True)
    assert permissions.decide(layers, "web_fetch", json.dumps({"url": "/workspace/secret.txt"})).outcome == DENY
    assert permissions.decide(layers, "web_fetch", json.dumps({"url": "https://example.com"})).outcome == ALLOW


def test_hidden_tools_are_not_listed_as_unavailable(non_admin):
    layers = _layers(allow_web_search="false")
    assert "web_search" in layers.disabled
    assert "bash" not in layers.disabled


class _FakeMcp:
    def plan_mode_blocked_mcp(self):
        return {"github": {"create_issue"}}, {"mcp__github__create_issue"}


def test_plan_mode_denies_mcp_write_tools(admin, monkeypatch):
    import src.tool_utils as tool_utils
    monkeypatch.setattr(tool_utils, "_mcp_manager", _FakeMcp())
    layers = _layers(plan_mode=True)
    assert permissions.decide(layers, "mcp__github__create_issue", "{}").outcome == DENY
    assert permissions.decide(layers, "mcp__github__list_issues", "{}").outcome == ASK


@pytest.mark.parametrize("name", ["mcp__Email__send_email", "MCP__email__send_email", "mcp__email__SEND_EMAIL"])
def test_spelling_variants_of_a_sensitive_tool_never_run_unasked(admin, name):
    assert permissions.decide(_layers(), name, "{}").outcome != ALLOW
