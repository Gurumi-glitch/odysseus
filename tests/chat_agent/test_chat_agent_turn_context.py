"""The legacy prompt's user-facing context on the new runtime (ticket #59).

Driven through the chat route with a scripted Chat Completions transport;
document tools run through the real executor against a throwaway SQLite
database.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core import database
import src.database as src_database
import src.settings as settings
from src.agent_loop import PLAN_MODE_DIRECTIVE
from src.chat_agent import turn_context
from src.tool_policy import GUIDE_ONLY_DIRECTIVE
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _sse_tool_calls,
    _switch_on,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", factory)
    # Document tools import SessionLocal through the src.database re-export.
    monkeypatch.setattr(src_database, "SessionLocal", factory)
    return factory


@pytest.fixture
def harness(monkeypatch, tmp_path, db):
    built = _Harness(monkeypatch, tmp_path)
    _switch_on()
    return built


def _add_document(db, doc_id, content, *, session_id, updated_at, language="markdown"):
    with db() as session:
        session.add(database.Document(
            id=doc_id, title=doc_id, language=language, current_content=content,
            owner="alice", session_id=None, updated_at=updated_at,
        ))
        session.commit()


def _document_text(db, doc_id):
    with db() as session:
        return session.get(database.Document, doc_id).current_content


def _current_message(payload):
    return payload["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_open_document_edit_lands_in_that_document_not_another_chats(harness, db):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    _add_document(db, "doc-open", "Intro: a very long and winding opening.\nBody.",
                  session_id="session-1", updated_at=now - timedelta(hours=1))
    # Another chat's document, edited more recently: the "latest" fallback.
    _add_document(db, "doc-other", "Intro: a very long and winding opening.\nOther.",
                  session_id="session-2", updated_at=now)
    harness.provider_rounds = [
        _sse_tool_calls(("edit_document", {"edits": [{
            "find": "a very long and winding opening.", "replace": "short.",
        }]})),
        _sse_events(["Shortened the intro."]),
    ]

    _, body = await harness.send(_Request(
        "agent", message="make the intro shorter",
        active_doc_id="doc-open", active_doc_state="visible",
    ))

    [output] = [e for e in _data_events(body) if e.get("type") == "tool_output"]
    assert output["exit_code"] == 0, output["output"]
    assert _document_text(db, "doc-open") == "Intro: short.\nBody."
    assert _document_text(db, "doc-other") == "Intro: a very long and winding opening.\nOther."
    first = harness.captured["provider_requests"][0]
    assert "a very long and winding opening." in _current_message(first)
    assert "a very long and winding opening." not in first["messages"][0]["content"]


@pytest.fixture
def saved_settings(monkeypatch):
    values = {}
    monkeypatch.setattr(settings, "load_settings", lambda: values)
    return values


@pytest.mark.asyncio
async def test_open_email_reply_turn_carries_the_email_and_my_email_style(harness, saved_settings):
    saved_settings.update({
        "email_writing_style": "Global style",
        "email_writing_styles_by_account": {"work": "Hiya NAME,\n...\nBest,\nAlice"},
    })
    harness.provider_rounds = [_sse_events(["Drafted."])]
    harness.context.current = {"role": "user", "content": "draft a reply"}

    await harness.send(_Request(
        "agent", message="draft a reply",
        active_email_uid="4242", active_email_folder="INBOX", active_email_account="work",
    ))

    [payload] = harness.captured["provider_requests"]
    current = _current_message(payload)
    assert "Message UID: 4242" in current
    assert "draft_email_reply" in current
    assert "Hiya NAME" in current
    assert "Global style" not in current
    assert current.index("Hiya NAME") < current.index("draft a reply")
    assert turn_context.WRITING_RULES in payload["messages"][0]["content"]
    assert "Hiya NAME" not in payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_plan_mode_turn_carries_the_plan_directive(harness):
    harness.provider_rounds = [_sse_events(["- [ ] step"])]

    await harness.send(_Request("agent", message="move the blog to hugo", plan_mode="true"))

    [payload] = harness.captured["provider_requests"]
    assert _current_message(payload).startswith(PLAN_MODE_DIRECTIVE)
    assert "PLAN MODE" not in payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_approved_plan_is_pinned_while_it_runs(harness):
    harness.provider_rounds = [_sse_events(["Working."])]

    await harness.send(_Request("agent", message="go", approved_plan="- [ ] write the post"))

    [payload] = harness.captured["provider_requests"]
    assert _current_message(payload).startswith("## ACTIVE PLAN")
    assert "- [ ] write the post" in _current_message(payload)


@pytest.mark.asyncio
async def test_guide_only_request_gets_the_guide_only_directive(harness):
    harness.provider_rounds = [_sse_events(["Run ls yourself."])]
    harness.context.current = {"role": "user", "content": "don't use tools, how do I list files?"}

    await harness.send(_Request("agent", message="don't use tools, how do I list files?"))

    [payload] = harness.captured["provider_requests"]
    assert _current_message(payload).startswith(GUIDE_ONLY_DIRECTIVE)


@pytest.mark.asyncio
async def test_system_prompt_is_byte_identical_across_turns_of_a_chat(harness, db, saved_settings):
    saved_settings["email_writing_style"] = "Best, Alice"
    _add_document(db, "doc-open", "Some text.", session_id="session-1",
                  updated_at=datetime.now(timezone.utc).replace(tzinfo=None))
    harness.provider_rounds = [_sse_events(["One."]), _sse_events(["Two."])]

    await harness.send(_Request(
        "agent", message="tighten this", active_doc_id="doc-open", active_doc_state="visible",
    ))
    await harness.send(_Request(
        "agent", message="now reply", active_email_uid="7", approved_plan="- [ ] reply",
    ))

    first, second = harness.captured["provider_requests"]
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][0]["content"] == second["messages"][0]["content"]
    assert _current_message(first) != _current_message(second)


@pytest.mark.asyncio
async def test_note_title_in_the_answer_is_a_link(harness, db):
    harness.provider_rounds = [
        _sse_tool_calls(("manage_notes", {"action": "add", "title": "dinner time with mom"})),
        _sse_events(["Added **dinner time with mom**."]),
    ]

    _, body = await harness.send(_Request("agent", message="add note: dinner time with mom"))

    with db() as session:
        [note] = session.query(database.Note).all()
    link = f"**[dinner time with mom](#note-{note.id})**"
    [final] = [e for e in _data_events(body) if e.get("type") == "final_response"]
    assert final["content"] == f"Added {link}."
    [row] = harness.saved
    assert row.content == f"Added {link}."
    assert row.metadata["round_texts"][-1] == f"Added {link}."
    # The model's own words stay unchanged for replay.
    assert row.metadata["agent_v2"]["rounds"][-1]["blocks"] == [
        {"type": "text", "text": "Added **dinner time with mom**."},
    ]
