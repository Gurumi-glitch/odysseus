"""Turn one executed tool call into chat UI events, model text and its tool event.

A port of the legacy result-to-event adapter that runs inline in
``stream_agent_loop`` after each tool call (ADR 0005, spec #44 Stream events).
A legacy branch is kept only when a frontend reader consumes its output;
``tests/chat_agent/test_chat_agent_result_events_parity.py`` runs the same
fixtures through both. Dropped branches (no reader, or owned by another
ticket): ``action_receipt``, ``tool_call_id``, ``note_title``, the ask_user
card (#56), the tool event's endpoint fields, the research/note/calendar link deltas
(final-text linkify, #59), and the legacy heuristics that queue follow-up
calls or rewrite calls and results from the user's wording.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

# Relies on: _document_stream_events returns the editor's doc_stream_open /
# doc_stream_delta events for create_document and update_document blocks, and
# [] for every other tool.
# Relies on: _inherit_calendar_open_range_from_tool_events sets target_date,
# view and results on an open_panel calendar ui_control result from the last
# manage_calendar list event, matched by that event's tool and JSON command.
# Relies on: _compact_web_search_tool_text_for_model returns text of at most
# max_chars (3200) that keeps the sources block, query and results summary.
from src.agent_loop import (
    _compact_web_search_tool_text_for_model,
    _document_stream_events,
    _inherit_calendar_open_range_from_tool_events,
)
from src.chat_agent.tools import STATUS_DENIED, ToolOutcome
from src.tool_capabilities import tool_result_is_successful
from src.tool_execution import format_tool_result
# Relies on: _truncate always returns a string, cut to the executor's output cap.
from src.tool_utils import _truncate

SOURCES_MARKER = "<!-- SOURCES:"
SOURCES_MARKER_END = " -->"
# The document panel shows the content; the card shows this many chars of the
# first line (legacy cmd_display).
DOC_COMMAND_PREVIEW_CHARS = 80
DOC_TOOLS = frozenset({"create_document", "update_document", "edit_document", "suggest_document"})
REOPEN_DOC_TOOLS = frozenset({"create_document", "update_document", "edit_document"})
DRAFT_EMAIL_TOOLS = frozenset({
    "draft_email", "mcp__email__draft_email",
    "draft_email_reply", "mcp__email__draft_email_reply",
    "ai_draft_email_reply", "mcp__email__ai_draft_email_reply",
})
# The email library reconciles an agent cleanup from these exact arguments.
TOOL_ARGS_TOOLS = frozenset({
    "bulk_email", "mcp__email__bulk_email",
    "delete_email", "mcp__email__delete_email",
    "unsubscribe_email", "mcp__email__unsubscribe_email",
    "private_browser", "mcp__private_browser",
})
UI_CONTROL_FIELDS = (
    "toggle_name", "state", "mode", "model", "endpoint_url", "theme_name", "colors",
    "uid", "folder", "account_id", "body", "panel", "view", "target_date",
)
IMAGE_FIELDS = ("image_url", "image_id", "image_prompt", "image_model", "image_size", "image_quality")
IMAGE_EVENT_FIELDS = ("image_url", "image_prompt", "image_model", "image_size", "image_quality")
ENTITY_FIELDS = (
    "memory_id", "task_id", "research_session_id", "skill_name", "session_id",
    "doc_id", "document_id", "uid",
)
_DRAFT_DOC_ID = (
    re.compile(r"#document-([0-9a-fA-F-]{8,64})"),
    re.compile(r"document ID:\s*([0-9a-fA-F-]{8,64})", re.IGNORECASE),
)


@dataclass
class Adapted:
    # In emission order; tool_output is among them.
    events: List[Dict[str, Any]]
    model_text: str
    tool_event: Dict[str, Any]


def command_display(block: Any) -> str:
    if block.tool_type in DOC_TOOLS:
        return block.content.split("\n")[0].strip()[:DOC_COMMAND_PREVIEW_CHARS]
    return block.content.strip()


def adapt(
    outcome: ToolOutcome, block: Any, *, round_number: int, model: str,
    prior_tool_events: Sequence[Dict[str, Any]],
) -> Adapted:
    """Events, model text and tool event for a call the executor ran.

    Mutates ``outcome.result`` as legacy does (marker stripped, draft id and
    calendar range added), so the model text reflects those changes.
    """
    result = outcome.result
    events: List[Dict[str, Any]] = []
    sources = _strip_sources(block, result)
    if sources is not None:
        events.append({"type": "web_sources", "data": sources})
    if tool_result_is_successful(result):
        # The editor stream must open before any doc_update finalizes it.
        events.extend(_document_stream_events(block))
    is_doc_tool = block.tool_type in DOC_TOOLS and "action" in result
    if is_doc_tool:
        events.append(_document_event(result))
    if "ui_event" in result:
        events.append({"type": "ui_control", "data": dict(result)})
    if "plan_update" in result:
        events.append({"type": "plan_update", "data": result["plan_update"]})

    output = _display_output(block, result, is_doc_tool)
    if block.tool_type in DRAFT_EMAIL_TOOLS and not result.get("doc_id"):
        _add_draft_document(result, output)
    if block.tool_type == "ui_control":
        _inherit_calendar_open_range_from_tool_events(result, list(prior_tool_events))

    command = command_display(block)
    events.append(_tool_output(outcome, block, command, output, is_doc_tool))
    if result.get("image_url"):
        image = {"type": "generated_image", "url": result.get("image_url")}
        image.update({key: result[key] for key in IMAGE_FIELDS if key in result})
        events.append(image)
    if block.tool_type in REOPEN_DOC_TOOLS and result.get("doc_id"):
        # Activates the saved document, so the next turn sends it back as the
        # active document.
        events.append({
            "type": "doc_update", "doc_id": result["doc_id"], "title": result.get("title", ""),
            "language": result.get("language", ""), "content": result.get("content", ""),
            "version": result.get("version", 1),
        })

    model_text = format_tool_result(outcome.desc, result)
    if block.tool_type == "web_search":
        model_text = _compact_web_search_tool_text_for_model(model_text)
    return Adapted(events, model_text, _tool_event(outcome, round_number, model, command, output))


def _strip_sources(block: Any, result: Dict[str, Any]):
    """The SOURCES marker's list, removed from the result; None when absent."""
    if block.tool_type != "web_search":
        return None
    for key in ("output", "results", "stdout"):
        if result.get(key):
            break
    else:
        return None
    text = result[key]
    start = text.find(SOURCES_MARKER)
    end = text.find(SOURCES_MARKER_END, start) if start >= 0 else -1
    if end < 0:
        return None
    try:
        sources = json.loads(text[start + len(SOURCES_MARKER):end])
    except ValueError:
        return None
    result[key] = text[:start].rstrip()
    return sources


def _document_event(result: Dict[str, Any]) -> Dict[str, Any]:
    if result["action"] == "suggest":
        return {"type": "doc_suggestions", "doc_id": result["doc_id"], "suggestions": result["suggestions"]}
    return {
        "type": "doc_update", "doc_id": result["doc_id"], "content": result["content"],
        "version": result["version"], "title": result.get("title", ""), "language": result.get("language"),
    }


def _display_output(block: Any, result: Dict[str, Any], is_doc_tool: bool) -> str:
    """The tool card's text; document content goes to the editor instead."""
    output = ""
    if is_doc_tool:
        title, version = result.get("title", ""), result.get("version", "?")
        output = {
            "create": f'Document created: "{title}" (v{version})',
            "edit": f'Document edited: "{title}" (v{version}, {result.get("applied", 0)} edit(s))',
            "update": f'Document updated: "{title}" (v{version})',
        }.get(result["action"], "")
    elif "stdout" in result:
        # A bash/python timeout carries an empty stdout and the reason in error.
        output = _truncate(result["stdout"] or result["stderr"] or result.get("error", ""))
    elif "output" in result:
        output = _truncate(result["output"] or "")
    elif "response" in result:
        label = result.get("model", result.get("session_name", "AI"))
        output = _truncate(f"{label}: {result['response']}")
    elif "content" in result:
        output = _truncate(result["content"])
    elif "results" in result:
        output = _truncate(result["results"])
    if block.tool_type == "manage_memory" and result.get("memory_id"):
        memory_id = str(result["memory_id"]).strip()
        if memory_id and memory_id not in output:
            output = (output.rstrip() + f"\nMemory id: {memory_id}").strip()
    elif "session_id" in result and "name" in result:
        output = f"Session created: {result['name']} (id: {result['session_id']})"
    elif "success" in result:
        output = f"Written: {result.get('path', '')}" if result["success"] else f"Error: {result.get('error', '')}"
    elif "error" in result:
        output = _truncate(result["error"])
    return output


def _add_draft_document(result: Dict[str, Any], output: str) -> None:
    """Email draft tools name the draft's document only in their text."""
    for pattern in _DRAFT_DOC_ID:
        match = pattern.search(output)
        if match:
            result["doc_id"] = match.group(1)
            result.setdefault("title", "Email draft")
            result.setdefault("language", "email")
            return


def _tool_output(
    outcome: ToolOutcome, block: Any, command: str, output: str, is_doc_tool: bool,
) -> Dict[str, Any]:
    result = outcome.result
    data = {
        "type": "tool_output", "tool": block.tool_type, "command": command, "output": output,
        "exit_code": outcome.exit_code, "execution_attempted": True,
        "blocked": outcome.status == STATUS_DENIED, "call_id": outcome.call_id,
    }
    if block.tool_type in TOOL_ARGS_TOOLS:
        try:
            args = json.loads(block.content.strip() or "{}")
        except ValueError:
            args = None
        if isinstance(args, dict):
            data["tool_args"] = args
    if is_doc_tool:
        data.update({
            "doc_id": result.get("doc_id"), "document_action": result.get("action"),
            "document_title": result.get("title", ""), "document_language": result.get("language", ""),
            "document_version": result.get("version"), "document_content": result.get("content", ""),
        })
    elif block.tool_type in DRAFT_EMAIL_TOOLS and result.get("doc_id"):
        data.update({
            "doc_id": result.get("doc_id"), "document_action": "create",
            "document_title": result.get("title", ""), "document_language": result.get("language", "email"),
            "document_version": result.get("version", 1), "document_content": result.get("content", ""),
        })
    if "ui_event" in result:
        data["ui_event"] = result["ui_event"]
        data.update({key: result[key] for key in UI_CONTROL_FIELDS if key in result})
    data.update({key: result[key] for key in IMAGE_FIELDS if key in result})
    screenshot = _screenshot(result)
    if screenshot:
        data["screenshot"] = screenshot
    if block.tool_type == "manage_calendar":
        data.update(_calendar_fields(result))
    data.update({key: result[key] for key in ENTITY_FIELDS if result.get(key) is not None})
    if "diff" in result:
        data["diff"] = result["diff"]
    return data


def _screenshot(result: Dict[str, Any]) -> str:
    images = result.get("images")
    if not images:
        return ""
    image = images[0]
    if not isinstance(image, dict) or not image.get("data") or not image.get("mimeType"):
        return ""
    return f"data:{image['mimeType']};base64,{image['data']}"


def _calendar_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    fields = {key: result[key] for key in ("uid", "anchor") if result.get(key)}
    if isinstance(result.get("events"), list):
        fields["events"] = result["events"]
    return fields


def _tool_event(
    outcome: ToolOutcome, round_number: int, model: str, command: str, output: str,
) -> Dict[str, Any]:
    """The persisted tool event: what history reload re-renders."""
    result = outcome.result
    event = outcome.tool_event(round_number, model)
    event.update({"command": command, "output": output})
    if result.get("image_url"):
        event.update({key: result[key] for key in IMAGE_EVENT_FIELDS if result.get(key)})
    screenshot = _screenshot(result)
    if screenshot:
        event["screenshot"] = screenshot
    if result.get("doc_id"):
        event["doc_id"] = result["doc_id"]
        event["doc_title"] = result.get("title", "")
    if outcome.tool == "manage_calendar":
        event.update(_calendar_fields(result))
    if result.get("diff"):
        event["diff"] = result["diff"]
    return event
