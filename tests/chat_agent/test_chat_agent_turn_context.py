"""The legacy prompt's user-facing context on the new runtime (ticket #59).

Driven through the chat route with a scripted Chat Completions transport;
document tools run through the real executor against a throwaway SQLite
database.
"""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core import database
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
    now = datetime.utcnow()
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
