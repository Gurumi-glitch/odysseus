"""What "Allow for this chat" covers (ticket #56 interim session rules)."""

import json

import pytest

from src.chat_agent import session_rules


@pytest.fixture(autouse=True)
def fresh_rules(monkeypatch):
    monkeypatch.setattr(session_rules, "_RULES", {})


def _allow(tool, content, owner="alice", session_id="s1"):
    session_rules.grant(owner, session_id, session_rules.rule_for(tool, content))


@pytest.mark.parametrize("approved, later, covered", [
    ("git status", "git log -5", True),
    ("git status", "git status && git diff", True),
    ("git status", "git status; rm -rf ~", False),
    ("git status", "git log | sh", False),
    ("git status", "git log $(rm -rf ~)", False),
    ("git status", "git log `id`", False),
    ("git status", "git log > ~/.bashrc", False),
    ("git status", "FOO=1 rm -rf ~", False),
    ("git status", "gitk", False),
    # A compound of different commands can't be scoped: only the exact call.
    ("ls | grep x", "ls | grep x", True),
    ("ls | grep x", "ls | grep y", False),
])
def test_bash_rule_is_a_command_prefix_every_part_must_match(approved, later, covered):
    _allow("bash", approved)
    assert session_rules.allows("alice", "s1", "bash", later) is covered


def test_rule_labels_use_the_claude_code_syntax():
    assert session_rules.rule_for("bash", "git status").label == "bash(git:*)"
    assert session_rules.rule_for("manage_tasks", json.dumps({"action": "create"})).label == (
        "manage_tasks(action:create)"
    )
    assert session_rules.rule_for("write_file", "notes/a.md\nhello").label == "write_file(notes/a.md)"
    assert session_rules.rule_for("mcp__github__create_issue", "{}").label == "mcp__github__create_issue"


def test_host_shell_reads_the_command_from_its_arguments():
    _allow("host_shell", json.dumps({"command": "docker ps"}))
    assert session_rules.allows("alice", "s1", "host_shell", json.dumps({"command": "docker ps -a"}))
    assert not session_rules.allows("alice", "s1", "host_shell", json.dumps({"command": "rm x"}))


def test_file_rule_covers_only_the_approved_path():
    _allow("write_file", "notes/a.md\nhello")
    assert session_rules.allows("alice", "s1", "write_file", "notes/a.md\nbye")
    assert session_rules.allows("alice", "s1", "edit_file", json.dumps({"path": "notes/a.md"})) is False
    assert not session_rules.allows("alice", "s1", "write_file", "notes/b.md\nbye")


def test_manage_rule_covers_only_the_approved_action():
    _allow("manage_tasks", json.dumps({"action": "create", "task": "a"}))
    assert session_rules.allows("alice", "s1", "manage_tasks", json.dumps({"action": "create", "task": "b"}))
    assert not session_rules.allows("alice", "s1", "manage_tasks", json.dumps({"action": "run", "id": "1"}))


def test_email_rule_covers_only_the_approved_recipient():
    _allow("mcp__email__send_email", json.dumps({"to": "Mom@example.com", "subject": "a", "body": "b"}))
    assert session_rules.allows("alice", "s1", "send_email", json.dumps({"to": "mom@example.com", "subject": "c"}))
    assert not session_rules.allows("alice", "s1", "send_email", json.dumps({"to": "boss@example.com"}))
    assert not session_rules.allows(
        "alice", "s1", "send_email", json.dumps({"to": "mom@example.com", "cc": "boss@example.com"}),
    )


def test_unscoped_tools_cover_only_the_exact_call():
    _allow("python", "print(1)")
    assert session_rules.allows("alice", "s1", "python", "print(1)")
    assert not session_rules.allows("alice", "s1", "python", "print(2)")


def test_a_rule_is_bound_to_its_owner_and_chat():
    _allow("bash", "ls", owner="Alice")
    assert session_rules.allows("alice", "s1", "bash", "ls")
    assert not session_rules.allows("bob", "s1", "bash", "ls")
    assert not session_rules.allows("alice", "s2", "bash", "ls")
    assert not session_rules.allows("alice", "s1", "python", "ls")
