"""The permission module's tool tables fail open when a tool is missing from
them (a new shell tool skips the shell rules, a new writer skips the
instruction-file guard). These pins fail when a known tool is neither listed
nor explicitly exempted, and a known workspace writer the module cannot read
a path from is protected."""

import json

import pytest

import src.tool_capabilities as tool_capabilities
import src.tool_security as tool_security
from src.chat_agent import permissions
from src.chat_agent.permissions import ALLOW, ASK, Rule, Rules
from src.tool_capabilities import KNOWN_CAPABILITY_TOOLS, TOOL_CAPABILITIES, ToolCapabilities, ToolEffect
from tests.chat_agent.test_chat_agent_rules import _layers


@pytest.fixture(autouse=True)
def admin(monkeypatch):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)


def _known_with(*effects):
    return sorted(name for name, caps in TOOL_CAPABILITIES.items()
                  if caps.effects & set(effects))


# --- Exemptions: one line of reason per tool, kept next to the pins ----------
# An exemption records today's behaviour; it is not a claim the tool is right
# to be left out. Entries marked FINDING are gaps reported to the user.

_SHELL_EXEMPT = {
    "manage_bg_jobs": "lists, reads and kills existing jobs; runs no command text",
}
_PATH_EXEMPT = {
    "get_workspace": "takes no path",
    "glob": "searches a root with a pattern; a path rule scopes one file",
    "grep": "searches a root with a pattern; a path rule scopes one file",
    "ls": "lists a directory; a path rule scopes one file",
    "inspect_media": "media reader; read-only by default, no path rule syntax yet",
    "extract_text": "document reader; read-only, no path rule syntax yet",
    "transcribe_media": "media reader; read-only, no path rule syntax yet",
    "apply_patch": "paths live inside the patch text; read by _patch_paths, not a path rule",
    "download_attachment": "saves under MAIL_ATTACHMENTS_DIR; takes no path argument",
}
_WRITE_PATH_EXEMPT = {
    "apply_patch": "paths live inside the patch text; _writes_instruction_file reads it with _patch_paths",
    "download_attachment": "saves under MAIL_ATTACHMENTS_DIR in a hashed folder; takes no path argument",
}
_EXTERNAL_EXEMPT = {
    "list_email_accounts": "account labels the user configured; no message content",
    "search_hf_models": "FINDING: registered untrusted network read (model-card text), yet not an external source",
    "download_attachment": "FINDING: returns attachment text inline but does not arm the external leg",
}


@pytest.mark.parametrize("tool", _known_with(ToolEffect.EXECUTE_CODE))
def test_every_known_code_executor_is_a_shell_tool_or_exempt(tool):
    assert tool in permissions._SHELL_TOOLS or tool in _SHELL_EXEMPT, (
        f"{tool} executes code: add it to _SHELL_TOOLS or exempt it with a reason")


@pytest.mark.parametrize("tool", _known_with(ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE))
def test_every_known_workspace_file_tool_is_a_path_tool_or_exempt(tool):
    assert tool in permissions._PATH_TOOLS or tool in _PATH_EXEMPT, (
        f"{tool} touches the workspace: add it to _PATH_TOOLS or exempt it with a reason")


@pytest.mark.parametrize("tool", _known_with(ToolEffect.WRITE_WORKSPACE))
def test_every_known_workspace_writer_is_a_write_path_tool_or_exempt(tool):
    assert tool in permissions._WRITE_PATH_TOOLS or tool in _WRITE_PATH_EXEMPT, (
        f"{tool} writes the workspace: add it to _WRITE_PATH_TOOLS or exempt it with a reason")


def _external_candidates():
    # Web readers (the registry's brokered network reads) and the built-in
    # email tools that read mailbox content.
    email = {name for name in tool_security.BUILTIN_EMAIL_TOOLS
             if ToolEffect.READ_PRIVATE in TOOL_CAPABILITIES[name].effects}
    return sorted(set(_known_with(ToolEffect.BROKERED_NETWORK_READ)) | email)


@pytest.mark.parametrize("tool", _external_candidates())
def test_every_known_outside_content_reader_is_an_external_source_or_exempt(tool):
    assert tool in permissions._EXTERNAL_SOURCES or tool in _EXTERNAL_EXEMPT, (
        f"{tool} brings outside content in: add it to _EXTERNAL_SOURCES or exempt it with a reason")


@pytest.mark.parametrize(("table", "exempt"), [
    (permissions._SHELL_TOOLS, _SHELL_EXEMPT),
    (permissions._PATH_TOOLS, _PATH_EXEMPT),
    (permissions._WRITE_PATH_TOOLS, _WRITE_PATH_EXEMPT),
    (permissions._EXTERNAL_SOURCES, _EXTERNAL_EXEMPT),
], ids=["shell", "path", "write-path", "external"])
def test_an_exemption_is_not_also_listed_and_names_a_known_tool(table, exempt):
    # A stale exemption would hide a later drop of the same tool.
    assert not set(exempt) & set(table)
    assert set(exempt) <= KNOWN_CAPABILITY_TOOLS


# --- A known workspace writer the module cannot read a path from ------------

# Relies on: capabilities_for_tool(name) reads TOOL_CAPABILITIES at call time,
# so a registered synthetic tool is "known" with exactly these effects.
_WRITER = "synthetic_workspace_writer"


@pytest.fixture
def synthetic_writer(monkeypatch):
    registry = {**TOOL_CAPABILITIES, _WRITER: ToolCapabilities(frozenset({ToolEffect.WRITE_WORKSPACE}))}
    monkeypatch.setattr(tool_capabilities, "TOOL_CAPABILITIES", registry)


def test_a_known_workspace_writer_without_a_readable_path_asks_every_time(synthetic_writer):
    content = json.dumps({"target": "AGENTS.md", "body": "x"})
    session = [Rule(_WRITER), permissions.session_rule(_WRITER, content)]
    decision = permissions.decide(_layers(), _WRITER, content, session=session, owner="alice")
    assert decision.outcome == ASK
    # No "Allow for this chat", and auto mode's classifier does not answer.
    assert (decision.chat_scope, decision.auto) == (False, False)


def test_an_allow_rule_does_not_lift_an_unreadable_writer(synthetic_writer):
    decision = permissions.decide(_layers(), _WRITER, "{}", rules=Rules(allow=(Rule(_WRITER),)),
                                  owner="alice")
    assert decision.outcome == ASK


_PATCH = "*** Begin Patch\n*** Update File: notes.txt\n@@\n-a\n+b\n*** End Patch"
_BENIGN_WRITES = {
    "write_file": json.dumps({"path": "notes.txt", "content": "x"}),
    "edit_file": json.dumps({"path": "notes.txt", "old_string": "a", "new_string": "b"}),
    "apply_patch": _PATCH,
    "download_attachment": json.dumps({"uid": "1", "index": 0}),
}


def test_today_s_workspace_writers_are_all_covered_by_the_benign_calls_below():
    assert set(_BENIGN_WRITES) == set(_known_with(ToolEffect.WRITE_WORKSPACE))


@pytest.mark.parametrize("name", sorted(_BENIGN_WRITES))
def test_a_registered_writer_that_reads_its_path_or_has_none_is_not_newly_protected(name):
    # download_attachment is exempt because it takes no path; the others are
    # judged by the path they name, so an ordinary file stays an ordinary ask.
    assert permissions.is_protected(name, _BENIGN_WRITES[name], owner="alice") is False


@pytest.mark.parametrize("name", ["mcp__github__create_file", "custom_writer"])
def test_a_tool_with_unknown_capabilities_keeps_its_handling(name):
    # Custom and MCP tools fail high on effects but stay rule-governed: the
    # unreadable-path rule needs the tool's capabilities to be known.
    assert not tool_capabilities.capabilities_for_tool(name).known
    assert permissions.is_protected(name, "{}", owner="alice") is False
    decision = permissions.decide(_layers(), name, "{}", session=[Rule(name)], owner="alice")
    assert (decision.outcome, decision.chat_scope) == (ALLOW, True)
