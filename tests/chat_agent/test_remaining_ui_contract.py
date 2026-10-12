"""The form fields the browser sends the chat_agent runtime (spec #44).

Source pins in the style of tests/test_tool_approval_task_scope.py: each
field is read by routes/chat_routes.py, so the sender and the reader must
agree on its name and value.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_unwatched_senders_say_so():
    # Notes agent-solve and group panes have no ask card (#61); compare
    # synthesis stays on the legacy runtime and sends nothing.
    assert "fd.append('interactive', 'false');" in _source("static/js/notes.js")
    assert "fd.append('interactive', 'false');" in _source("static/js/group.js")
    assert "interactive" not in _source("static/js/compare/stream.js")
    assert 'str(form_data.get("interactive", "")).lower() != "false"' in _source("routes/chat_routes.py")


def test_slash_invoked_skill_rides_the_next_send_once():
    slash = _source("static/js/slashCommands.js")
    chat = _source("static/js/chat.js")
    assert "window.__odysseusPendingInvokedSkill = data.name || '';" in slash
    assert "fd.append('invoked_skill', window.__odysseusPendingInvokedSkill);" in chat
    assert chat.count("window.__odysseusPendingInvokedSkill = '';") == 1
    assert 'invoked_skill=str(form_data.get("invoked_skill") or "").strip()' in _source("routes/chat_routes.py")
