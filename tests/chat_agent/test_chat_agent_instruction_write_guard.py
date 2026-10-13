"""Writes to AGENTS.md / CLAUDE.md are protected however the call spells the
target (security follow-up to M3): the gate reads the path the way the
executor does, and fails closed where it cannot."""

import json

import pytest

import src.tool_security as tool_security
from src.chat_agent import permissions
from src.chat_agent.permissions import ASK, Rules
from tests.chat_agent.test_chat_agent_rules import _layers


@pytest.fixture(autouse=True)
def admin(monkeypatch):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)


@pytest.fixture
def instructed(tmp_path):
    root = tmp_path / "app"
    (root / "notes").mkdir(parents=True)
    (root / "AGENTS.md").write_text("rules", encoding="utf-8")
    (root / "notes" / "link.md").symlink_to(root / "AGENTS.md")
    return root


def _patch(path):
    return f"*** Begin Patch\n*** Update File: {path}\n@@\n-rules\n+allow everything\n*** End Patch"


def _protected(name, content, workspace=None):
    return permissions.is_protected(name, content, owner="alice", workspace=workspace)


@pytest.mark.parametrize("path", [
    # The executor strips the path it is given.
    "AGENTS.md ", " AGENTS.md", "AGENTS.md\n",
    # Case-insensitive mounts open the same file.
    "agents.md", "Agents.MD", "notes/claude.MD",
    # Mounts that drop trailing dots and spaces (SMB/NTFS) do too.
    "AGENTS.md.", "AGENTS.md. ",
    # Where the target cannot be placed, it counts.
    "../AGENTS.md", "~/AGENTS.md",
])
@pytest.mark.parametrize("workspace", [True, False])
def test_every_spelling_of_an_instruction_file_is_protected(instructed, path, workspace):
    content = json.dumps({"path": path, "content": "x"})
    assert _protected("write_file", content, str(instructed) if workspace else None) is True


def test_an_edit_without_the_executor_s_path_key_counts(instructed):
    # The executor reads only "path"; a call it cannot place fails closed.
    content = json.dumps({"file_path": "AGENTS.md", "old_string": "a", "new_string": "b"})
    assert _protected("edit_file", content, str(instructed)) is True


def test_a_symlink_to_an_instruction_file_counts_without_a_workspace(instructed):
    # Without a workspace the executor opens the path's realpath.
    content = json.dumps({"path": str(instructed / "notes" / "link.md"), "content": "x"})
    assert _protected("write_file", content) is True


@pytest.mark.parametrize("content", [
    # JSON escapes the executor decodes before reading the patch.
    json.dumps({"patch_text": _patch("AGENTS.md")}).replace("AGENTS", "\\u0041GENTS"),
    json.dumps({"patchText": _patch("CLAUDE.md")}),
    json.dumps({"patch": _patch("agents.md")}),
    # Through a symlink the real file changes.
    _patch("notes/link.md"),
    # A patch the gate cannot read counts.
    "*** Begin Patch\n*** Update File: notes/a.md\n*** Move to: AGENTS.md\n*** End Patch",
    "not a patch",
])
def test_a_patch_onto_an_instruction_file_is_protected(instructed, content):
    assert _protected("apply_patch", content, str(instructed)) is True


def test_a_patch_elsewhere_that_mentions_agents_md_is_not(instructed):
    (instructed / "notes" / "todo.md").write_text("a\n", encoding="utf-8")
    content = ("*** Begin Patch\n*** Update File: notes/todo.md\n@@\n-a\n+see AGENTS.md\n*** End Patch")
    assert _protected("apply_patch", content, str(instructed)) is False


@pytest.mark.parametrize(("name", "content"), [
    ("bash", "echo 'always allow uploads' >> AGENTS.md"),
    ("bash", "tee -a ./claude.md < /tmp/x"),
    ("bash", "cp /tmp/evil notes/Agents.MD"),
    ("python", "open('AGENTS.md', 'a').write('x')"),
    # JSON escapes the executor decodes.
    ("host_shell", json.dumps({"command": "mv x AGENTS.md"}).replace("AGENTS", "\\u0041GENTS")),
])
def test_a_shell_call_naming_an_instruction_file_is_protected(name, content):
    assert _protected(name, content) is True


def test_a_shell_call_naming_an_instruction_file_is_not_left_to_the_classifier():
    content = "echo x >> AGENTS.md"
    rules = Rules.from_lists({"allow": ["bash"]})
    decision = permissions.decide(_layers(rules), "bash", content, rules=rules, owner="alice", auto_mode=True)
    assert (decision.outcome, decision.auto) == (ASK, False)


def test_an_ordinary_shell_call_stays_with_the_classifier():
    decision = permissions.decide(_layers(), "bash", "ls -la", owner="alice", auto_mode=True)
    assert (decision.outcome, decision.auto) == (ASK, True)
