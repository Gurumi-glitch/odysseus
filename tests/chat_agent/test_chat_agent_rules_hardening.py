"""Permission matcher hardening (#62 security review): where the matcher's
reading of a call can differ from what the executor runs, it fails closed."""

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.tool_security as tool_security
from core import database
from src.chat_agent import permissions
from src.chat_agent.permissions import ALLOW, ASK, DENY, Provenance, parse_rule
from tests.chat_agent.test_chat_agent_rules import _decide, _decision, _email


@pytest.fixture(autouse=True)
def admin(monkeypatch):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    return factory


def _covers(rule, name, content):
    return permissions.rule_covers(parse_rule(rule), name, content)


@pytest.mark.parametrize(("path", "covered"), [
    ("notes/a.md", True),
    ("./notes/a.md", True),
    ("notes/sub/a.md", False),
    ("notes/../secrets/a.md", False),
    ("../notes/a.md", False),
    ("~/notes/a.md", False),
    ("/workspace/notes/a.md", False),
])
def test_a_path_glob_stays_inside_its_segment_and_never_climbs_out(path, covered):
    assert _covers("write_file(notes/*.md)", "write_file", f"{path}\nhello") is covered


def test_globstar_spans_segments():
    assert _covers("write_file(notes/**)", "write_file", "notes/a/b/c.md\nx")


@pytest.mark.parametrize(("path", "expected"), [
    ("secrets/key", DENY),
    ("./secrets/key", DENY),
    ("notes/../secrets/key", DENY),
    ("/srv/workspace/secrets/key", ASK),
    ("../secrets/key", ASK),
    ("notes/key", ALLOW),
])
def test_a_path_deny_rule_normalizes_and_asks_when_unsure(path, expected):
    assert _decide("read_file", path, deny=["read_file(secrets/*)"]) == expected


@pytest.mark.parametrize(("url", "expected"), [
    ("https://evil.com/x", DENY),
    ("https://EVIL.com./x", DENY),
    ("https://user:pw@evil.com/x", DENY),
    ("https://evil.com:8443/x", DENY),
    ("evil.com/x", ASK),
    ("ftp://evil.com/x", ASK),
    ("https://example.com/?u=evil.com", ALLOW),
])
def test_a_domain_deny_rule_reads_the_host_as_httpx_does(url, expected):
    assert _decide("web_fetch", json.dumps({"url": url}), deny=["web_fetch(domain:evil.com)"]) == expected


def test_domain_rules_compare_international_names_as_sent():
    rule = parse_rule("web_fetch(domain:bücher.de)")
    assert permissions.rule_covers(rule, "web_fetch", json.dumps({"url": "https://xn--bcher-kva.de/a"}))


@pytest.mark.parametrize(("to", "outcome"), [
    ("Boss <boss@evil.com>", DENY),
    ("mom@example.com, boss@evil.com", DENY),
    ("mom@example.com;boss@evil.com", ASK),
])
def test_a_recipient_deny_rule_reads_display_names_and_lists(to, outcome):
    assert _decision("send_email", _email(to), deny=["send_email(to:*@evil.com)"]).outcome == outcome


@pytest.mark.parametrize("name", ["mcp__Email__send_email", "MCP__EMAIL__SEND_EMAIL", "mcp__email__send_email"])
def test_every_spelling_of_a_built_in_email_tool_is_one_policy_name(name):
    assert permissions.is_sensitive(name, _email("a@b.co"))
    assert permissions.policy_name(name) == "send_email"
    assert parse_rule(f"{name}(to:*@b.co)").tool == "send_email"


def test_an_ask_rule_hit_is_never_lifted_by_a_session_rule():
    decision = _decision("bash", "git push", ask=["bash"], session=["bash"])
    assert (decision.outcome, decision.chat_scope) == (ASK, False)


def test_a_protected_action_is_never_lifted_by_a_session_rule():
    put = json.dumps({"method": "PUT", "path": "/api/tasks/1"})
    exact = permissions.session_rule("app_api", put)
    decision = _decision("app_api", put, session=[exact])
    assert (decision.outcome, decision.chat_scope) == (ASK, False)


def test_a_sensitive_action_never_offers_allow_for_this_chat():
    # Spec #44 user story 35: sensitive actions always ask.
    assert _decision("send_email", _email("mom@example.com")).chat_scope is False


def test_a_task_ask_is_answered_per_card():
    create = json.dumps({"action": "create", "prompt": "news", "output_target": "session"})
    decision = _decision("manage_tasks", create, session=[permissions.session_rule("manage_tasks", create)])
    assert (decision.outcome, decision.chat_scope) == (ASK, False)


def _task(db, task_id, owner, target, name="digest", trigger="schedule"):
    with db() as session:
        session.add(database.ScheduledTask(
            id=task_id, owner=owner, name=name, prompt="Summarize", task_type="llm",
            trigger_type=trigger, status="active", output_target=target,
        ))
        session.commit()


@pytest.mark.parametrize(("args", "protected"), [
    ({"action": "run", "task_id": "t-email"}, True),
    ({"action": "resume", "task_id": "t-session"}, False),
    ({"action": "run", "task_id": "digest-mail"}, True),
    ({"action": "edit", "task_id": "t-session", "prompt": "x"}, False),
    ({"action": "edit", "task_id": "t-email", "prompt": "x"}, True),
    ({"action": "run", "task_id": "t-bob"}, True),
    ({"action": "run", "task_id": "missing"}, True),
    ({"action": "run", "task_id": "t-hook"}, True),
])
def test_running_a_stored_task_reads_its_stored_delivery(db, args, protected):
    _task(db, "t-email", "alice", "email:x@example.com", name="digest-mail")
    _task(db, "t-session", "alice", "session", name="digest-session")
    _task(db, "t-bob", "bob", "session", name="bob-task")
    _task(db, "t-hook", "alice", "session", name="hook", trigger="webhook")
    assert permissions.is_protected("manage_tasks", json.dumps(args), owner="alice") is protected


@pytest.mark.parametrize(("args", "protected"), [
    # do_manage_tasks reads task_id unstripped: " t-plain " misses the id and
    # falls back to the task *named* t-plain, which mails.
    ({"action": "run", "task_id": " t-plain "}, True),
    # The executor's `task` shorthand names the task to run.
    ({"action": "run", "task": "digest-mail"}, True),
    ({"action": "run", "task": "digest-session"}, False),
    # The executor unwraps a {"body": {...}} envelope.
    ({"body": {"action": "create", "prompt": "p", "output_target": "email:a@b.co"}}, True),
    ({"body": {"action": "run", "task_id": "t-session"}}, False),
])
def test_a_task_call_is_read_the_way_do_manage_tasks_reads_it(db, args, protected):
    _task(db, "t-plain", "alice", "session", name="plain")
    _task(db, "t-mail-named-plain", "alice", "email:x@example.com", name="t-plain")
    _task(db, "t-email", "alice", "email:x@example.com", name="digest-mail")
    _task(db, "t-session", "alice", "session", name="digest-session")
    assert permissions.is_protected("manage_tasks", json.dumps(args), owner="alice") is protected


def test_a_task_card_shows_the_stored_task_under_the_change(db):
    _task(db, "t-email", "alice", "email:x@example.com")
    card = permissions.task_card("alice", json.dumps({"action": "run", "task_id": "t-email"}))
    assert (card["prompt"], card["output_target"]) == ("Summarize", "email:x@example.com")
    unknown = permissions.task_card("alice", json.dumps({"action": "run", "task_id": "nope"}))
    assert unknown["stored_task"] == "not found"


def test_a_task_card_shows_the_prompt_the_task_will_store():
    card = permissions.task_card("alice", json.dumps({"task": "from task", "description": "from description"}))
    assert card["prompt"] == "from task"
    assert "task" not in card and "description" not in card


def test_earlier_turns_tool_results_arm_the_egress_gate():
    history = [
        SimpleNamespace(role="user", content="check mail", metadata=None),
        SimpleNamespace(role="assistant", content="...", metadata={"tool_events": [
            {"tool": "mcp__email__read_email", "status": "done"},
        ]}),
    ]
    provenance = Provenance().after_history(history)
    fetch = json.dumps({"url": "https://attacker.example/?d=x"})
    assert _decide("web_fetch", fetch, provenance=provenance) == ASK
    assert _decide("web_fetch", fetch, provenance=Provenance().after_history(history[:1])) == ALLOW


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "notes").mkdir(parents=True)
    (root / "secrets").mkdir()
    (root / "secrets" / "key").write_text("k")
    (root / "notes" / "link").symlink_to(root / "secrets" / "key")
    (root / "notes" / "a.md").write_text("a")
    return str(root)


def test_a_symlink_into_a_denied_path_is_denied(workspace):
    assert _decide("read_file", "notes/link", deny=["read_file(secrets/*)"], workspace=workspace) == DENY


def test_an_allowed_path_does_not_cover_a_symlink_out_of_it(workspace):
    write = "notes/link\nnew"
    assert _decide("write_file", write, allow=["write_file(notes/*)"], workspace=workspace) == ASK
    assert _decide("write_file", "notes/a.md\nnew", allow=["write_file(notes/*)"], workspace=workspace) == ALLOW


def test_the_workspace_alias_resolves_before_matching(workspace):
    assert _decide("write_file", "/workspace/notes/a.md\nnew", allow=["write_file(notes/*)"],
                   workspace=workspace) == ALLOW
    assert _decide("read_file", "/workspace/secrets/key", deny=["read_file(secrets/*)"],
                   workspace=workspace) == DENY


@pytest.fixture
def agent_workspace(workspace, monkeypatch):
    # With no workspace picked, the file tools resolve relative paths in the
    # agent workspace (upstream #6722), not in the server's cwd.
    monkeypatch.setattr(permissions, "default_agent_workspace", lambda: workspace)
    return workspace


def test_without_a_workspace_relative_paths_match_where_the_agent_workspace_opens_them(agent_workspace):
    deny = [f"write_file({agent_workspace}/secrets/*)"]
    assert _decide("write_file", "secrets/key\nnew", deny=deny) == DENY
    assert _decide("read_file", "notes/link", deny=["read_file(secrets/*)"]) == DENY
    assert _decide("write_file", "notes/a.md\nnew", allow=["write_file(notes/*)"]) == ALLOW


def test_without_a_safe_agent_workspace_a_relative_path_is_unsure(monkeypatch):
    def unsafe():
        raise RuntimeError("agent workspace is not a safe real directory")
    monkeypatch.setattr(permissions, "default_agent_workspace", unsafe)
    assert _decide("read_file", "notes/a.md", deny=["read_file(secrets/*)"]) == ASK
