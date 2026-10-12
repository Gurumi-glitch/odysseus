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
