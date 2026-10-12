"""What the user has open and asked for this turn, as context for the model.

The parts of the legacy prompt a user notices losing (spec #44, Context and
prompt; ticket #59). Anything that changes from turn to turn goes on the
current user message, so the system prompt stays byte-identical across a
chat's turns and providers can reuse their cached prefix. Text a user or a
third party can edit (documents, email, styles, integration descriptions) is
wrapped as untrusted data; the runtime's own guidance is plain text.
No wording heuristic picks any of it: each part follows from what is open,
toggled or configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import database
# Relies on: PLAN_MODE_DIRECTIVE is the legacy plan-mode prompt; it promises
# only what the deny layers enforce (plan_mode_disabled_tools).
# build_active_plan_note(plan) returns "" for a blank plan.
from src.agent_loop import PLAN_MODE_DIRECTIVE, build_active_plan_note
# Relies on: returns the docker network/commands block as an untrusted
# message, or None outside a container.
from src.agent_loop import _backend_runtime_context_message
# Relies on: shortens an email compose document (headers, "---", body) to its
# headers, the user's own text and an excerpt of the quoted history.
from src.agent_loop import _compact_email_draft_context
# Relies on: _has_recent_email_tool_context(messages) looks back from the last
# user message over earlier rows' metadata.tool_events; the second builds the
# untrusted "recent tool context" message from the same rows, or None.
from src.agent_loop import _has_recent_email_tool_context, _minimal_recent_notes_tool_context_message
# Relies on: one untrusted message listing each upload's id, name, mime, size
# and path, telling the model to read_file non-image uploads; None for [].
from src.agent_loop import _uploaded_files_context_message
# Relies on: the "## Workspace coding mode" rules for this folder, "" for none;
# host_bridge=False is the backend-tools variant.
from src.agent_loop import _workspace_coding_rules
from src.agent_tools.document_tools import set_active_document, set_active_model
from src.chat_agent.permissions import EMAIL_READER_DENIED
# Relies on: an untrusted message with the reader's uid, folder, account,
# subject, sender and body preview; None without a uid.
from src.clean_agent_preview import active_email_context_message
# Relies on: the account's saved email style, else the global one, as an
# untrusted message; None when neither is set.
from src.email_task_intent import email_style_context
# Relies on: returns "" when no integration is enabled.
from src.integrations import get_integrations_prompt
# Relies on: a non-empty upload id only for a document rendered from a PDF
# (pdf_form_source or pdf_source front matter), as legacy's form mode.
from src.pdf_form_doc import find_source_upload_id
from src.prompt_security import untrusted_context_message
from src import settings
# Relies on: detect_guide_only_turn returns a reason when the user's own
# message forbids tool use, else None.
from src.tool_policy import GUIDE_ONLY_DIRECTIVE, detect_guide_only_turn

EMAIL_LANGUAGE = "email"

DOCUMENT_GUIDANCE = (
    "The user has the document below open in the editor; requests are about it "
    "unless they name another target. It lives in the editor, not on disk: use it "
    "as given here, not file tools. Edit it with edit_document (the FIND text must "
    "match exactly and never includes the line numbers), rewrite it whole with "
    "update_document, and give suggestions or a review with suggest_document."
)
NO_DOCUMENT_STYLE = (
    "No document writing style is saved. If asked to write in the user's style, "
    "ask for a sample instead of inferring one from memories or public facts."
)
PDF_FORM_GUIDANCE = (
    "The open document is a fillable PDF form rendered as markdown; every field on "
    "every page is a bullet line below, so don't read the PDF or ask for it. Fill it "
    "with edit_document, replacing whole bullet lines and keeping each trailing "
    "<!-- field=... --> comment unchanged: text fields take the value after the label, "
    "choice fields one of the listed options verbatim, checkboxes toggle [ ] and [x]. "
    "Never invent values (ask), never edit the pdf_form_source front matter or the "
    "## Page headers, never touch signature fields (the user signs on the rendered "
    "PDF), and don't export: the user has an Export button."
)
EMAIL_DRAFT_GUIDANCE = (
    "The open document is an email draft in the compose window. A request to write, "
    "reply or change the wording without another target edits this draft with "
    "update_document or edit_document; don't create another document and don't send "
    "it. Keep every header line and the --- separator exactly as they are, and change "
    "only the text above \"---------- Previous message ----------\": Odysseus keeps "
    "the quoted history. Write as the mailbox owner only. After a successful edit, "
    "confirm briefly instead of pasting the email back."
)
OPEN_EMAIL_GUIDANCE = (
    "The user has the email below open in the reader. Requests about email this turn "
    "refer to it unless the user names another message or recipient. Reply by "
    "drafting with draft_email_reply using its UID, folder and account (the user "
    "reviews the draft before sending); read_email gives the full body. While it is "
    "open, {denied} are unavailable."
)
# Hardcoded identity and mechanical rules legacy keeps beside a saved style.
WRITING_RULES = (
    "When writing for the user, write as the user or mailbox owner only: never sign "
    "as, speak as or imply you are the recipient, the original or quoted sender, or "
    "any third party, and never copy a name from a quoted thread into the sign-off. "
    "Never use em or en dashes (use --) or curly apostrophes. In English emails "
    "greet with the saved style's greeting, not \"Hey\"."
)
IMAGE_EDIT_HINT = (
    "To edit an attached image, call edit_image with action=prompt, image_id set to "
    "its reference ({refs}) and the requested changes as prompt; the backend sends "
    "the real pixels. A description or a stock image is not an edited image."
)


@dataclass(frozen=True)
class OpenItems:
    """What the browser says is open this turn, already owner-checked."""

    # A detached core.database.Document row, or None.
    document: Any = None
    # The email reader's uid/folder/account plus cached subject, from, preview.
    email: Optional[Dict[str, str]] = None
    # The checklist the user approved, sent back while it runs.
    approved_plan: str = ""
    # This turn's denied tool names (permissions.DenyLayers.denied_names).
    denied: frozenset = field(default_factory=frozenset)


def load_document(owner: str, document_id: str) -> Any:
    """The visible editor tab's document when ``owner`` owns it.

    Only the id the browser sends counts: the web UI always says which tab is
    visible, so the legacy session and in-memory fallbacks don't apply.
    """
    if not document_id:
        return None
    db = database.SessionLocal()
    try:
        document = db.query(database.Document).filter(
            database.Document.id == document_id, database.Document.owner == owner,
        ).first()
        if document is not None:
            db.expunge(document)
        return document
    finally:
        db.close()


def bind_process_state(document: Any, model: str) -> None:
    # Document tools fall back to this process-wide pointer, and version
    # summaries name this model; set both every turn so a previous turn's
    # document never leaks in.
    set_active_document(document.id if document is not None else None)
    set_active_model(model)


def system_sections() -> List[str]:
    """Context that holds for every turn of a chat (system prompt)."""
    saved = settings.load_settings()
    return [WRITING_RULES] if _saved_email_style(saved) or _document_style(saved) else []


def messages(
    *,
    open_items: OpenItems,
    plan_mode: bool,
    workspace: Optional[str],
    current_text: str,
    history: List[Dict[str, Any]],
    uploaded_files: List[Dict[str, Any]],
    attachment_meta: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """This turn's context, in order, for the current user message."""
    saved = settings.load_settings()
    parts: List[Optional[Dict[str, Any]]] = []
    parts += _directives(plan_mode, open_items.approved_plan, current_text)
    # Untrusted-wrapped, so it stays out of the system prompt as on legacy.
    parts.append(_backend_runtime_context_message())
    if workspace:
        parts.append(_text(_workspace_coding_rules(workspace).strip()))
    parts += _document_messages(open_items.document, saved)
    parts += _email_messages(open_items, saved)
    parts.append(_recent_email_context(history, current_text))
    if "api_call" not in open_items.denied:
        integrations = get_integrations_prompt()
        if integrations:
            parts.append(untrusted_context_message("integrations", integrations))
    parts.append(_uploaded_files_context_message(uploaded_files))
    if "edit_image" not in open_items.denied:
        parts.append(_image_edit_hint(attachment_meta))
    return [part for part in parts if part]


def _text(content: str) -> Optional[Dict[str, Any]]:
    return {"role": "user", "content": content} if content else None


def _directives(plan_mode: bool, approved_plan: str, current_text: str) -> List[Optional[Dict[str, Any]]]:
    if detect_guide_only_turn(current_text):
        # The user forbade tools; plan directives would ask for them.
        return [_text(GUIDE_ONLY_DIRECTIVE)]
    if plan_mode:
        return [_text(PLAN_MODE_DIRECTIVE)]
    return [_text(build_active_plan_note(approved_plan))]


def _document_messages(document: Any, saved: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    if document is None:
        return []
    content = document.current_content or ""
    title = f"Title: \"{document.title}\""
    if document.language == EMAIL_LANGUAGE:
        return [_text(EMAIL_DRAFT_GUIDANCE), untrusted_context_message(
            "active email draft", f"{title}\n```\n{_compact_email_draft_context(content)}\n```",
        )]
    if find_source_upload_id(content):
        return [_text(PDF_FORM_GUIDANCE), untrusted_context_message(
            "active PDF form", f"{title}\n```\n{content}\n```",
        )]
    numbered = "\n".join(f"{number}\t{line}" for number, line in enumerate(content.split("\n"), 1))
    style = _document_style(saved)
    return [
        _text(DOCUMENT_GUIDANCE if style else f"{DOCUMENT_GUIDANCE}\n{NO_DOCUMENT_STYLE}"),
        untrusted_context_message("active editor document", (
            f"{title} | Language: {document.language or 'text'}\n"
            "Each line starts with its number and a tab so the user can point at it "
            "(\"[Doc edit: L25]\"); they are not part of the document.\n"
            f"```\n{numbered}\n```"
        )),
        untrusted_context_message("document writing style", style) if style else None,
    ]


def _email_messages(open_items: OpenItems, saved: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    email = open_items.email or {}
    document = open_items.document
    draft_open = document is not None and document.language == EMAIL_LANGUAGE
    parts: List[Optional[Dict[str, Any]]] = []
    if email.get("uid") and not draft_open:
        # A draft being written is the target, as on legacy.
        denied = ", ".join(sorted(EMAIL_READER_DENIED))
        parts += [_text(OPEN_EMAIL_GUIDANCE.format(denied=denied)), active_email_context_message(email)]
    account = (getattr(document, "source_email_account_id", None) if draft_open else None) or email.get("account") or ""
    parts.append(email_style_context(saved, account=str(account).strip()))
    return parts


def _recent_email_context(history: List[Dict[str, Any]], current_text: str) -> Optional[Dict[str, Any]]:
    # Runtime turns replay their tool results (ADR 0007). Rows from legacy turns
    # replay as text only, so their email/note/calendar ids come from here.
    legacy_rows = [m for m in history if not (m.get("metadata") or {}).get("agent_v2")]
    rows = [*legacy_rows, {"role": "user", "content": current_text}]
    if not _has_recent_email_tool_context(rows):
        return None
    return _minimal_recent_notes_tool_context_message(rows)


def _image_edit_hint(attachment_meta: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    refs = [
        f"odysseus://attachment/{meta['id']}" for meta in attachment_meta or ()
        if meta.get("id") and str(meta.get("mime") or "").startswith("image/")
    ]
    return _text(IMAGE_EDIT_HINT.format(refs=", ".join(refs))) if refs else None


def _saved_email_style(saved: Dict[str, Any]) -> bool:
    by_account = saved.get("email_writing_styles_by_account")
    return bool(str(saved.get("email_writing_style") or "").strip()) or (
        isinstance(by_account, dict) and any(str(v or "").strip() for v in by_account.values())
    )


def _document_style(saved: Dict[str, Any]) -> str:
    return str(saved.get("document_writing_style") or "").strip()
