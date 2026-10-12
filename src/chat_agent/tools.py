"""Tool catalog and per-call execution wrapper of the chat_agent runtime.

The catalog is the native schemas minus the deferred table (ADR 0003); the
tool list never changes with the user's wording. The wrapper runs one admitted
call through the legacy executor whole, with a fresh request authority that
binds resources and decides nothing (#34): the permission gate already did.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

# Relies on: _apply_tool_surface_to_schemas returns [] for "none" and
# description-stripped schemas for "compact"; _load_mcp_disabled_map reads
# every server's disabled_tools from the database on each call, as
# {server_id: set of server-local tool names}.
from src.agent_loop import _apply_tool_surface_to_schemas, _load_mcp_disabled_map
from src.agent_runtime.authority import OperationGrant, RequestAuthority, _owner as authority_owner
# Relies on: seal_backends seals the backends the listed tools may reach from
# the client runtime context, as create_request_authority does.
from src.agent_runtime.remote_resources import seal_backends
from src.agent_tools import FUNCTION_TOOL_SCHEMAS
# Relies on: execute_tool_block returns (description, result dict) and, given
# NO_TOOL_SECURITY_CONTEXT and an explicit authority, applies no taint gate and
# intersects with no parent authority when none is bound.
from src.tool_execution import NO_TOOL_SECURITY_CONTEXT, execute_tool_block, format_tool_result
from src.tool_capabilities import tool_result_is_successful, tool_result_was_denied
from src.turn_contract import canonical_tool
# get_mcp_manager: the process-wide MCP manager, None before startup.
from src.tool_utils import get_mcp_manager

# Rare built-ins the model finds with tool search instead of seeing upfront
# (#27, ADR 0003). The split is this one table.
DEFERRED_TOOLS = frozenset({
    # model ops
    "adopt_served_model", "cancel_download", "download_model", "list_cached_models",
    "list_cookbook_servers", "list_downloads", "list_models", "list_serve_presets",
    "list_served_models", "search_hf_models", "serve_model", "serve_preset",
    "stop_served_model", "tail_serve_output",
    # admin and settings
    "manage_endpoints", "manage_tokens", "manage_webhooks", "manage_settings",
    "manage_mcp", "manage_skills", "manage_bg_jobs", "host_shell",
    # sessions and other models
    "create_session", "list_sessions", "manage_session", "send_to_session",
    "chat_with_model", "ask_teacher",
    # raw access
    "api_call", "app_api", "pipeline",
})

# The runtime's own tool: it loads deferred schemas and never reaches the
# legacy executor (spec #44, tools).
TOOL_SEARCH = "tool_search"
TOOL_SEARCH_MAX_SCHEMAS = 5
TOOL_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_SEARCH,
        "description": (
            "Load deferred tools listed in the system prompt so you can call them. "
            "Pass keywords to match tool names and descriptions, or "
            "'select:name_a,name_b' for exact names. Loads at most "
            f"{TOOL_SEARCH_MAX_SCHEMAS} tools per call; they become callable from "
            "your next step."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
_SELECT_PREFIX = "select:"
# MCP servers whose tools never enter the catalog: built-in email is upfront
# under its bare names; raw Playwright always loses to private_browser.
# The internal Python servers (memory, RAG, image generation) are already
# left out by get_all_openai_schemas.
_CATALOG_EXCLUDED_MCP_SERVERS = frozenset({"email", "builtin_browser"})

STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_DENIED = "denied"
# Shell conventions, so tool cards and transcripts read like a terminal.
EXIT_DONE = 0
EXIT_FAILED = 1
EXIT_DENIED = 126


def schema_name(schema: Dict[str, Any]) -> str:
    return str((schema.get("function") or {}).get("name") or "")


def upfront_tool_names() -> List[str]:
    return [schema_name(s) for s in FUNCTION_TOOL_SCHEMAS if schema_name(s) not in DEFERRED_TOOLS]


def upfront_schemas(denied: Iterable[str], surface: str) -> List[Dict[str, Any]]:
    denied = set(denied)
    schemas = [
        schema for schema in FUNCTION_TOOL_SCHEMAS
        if schema_name(schema) not in DEFERRED_TOOLS and schema_name(schema) not in denied
    ]
    return _apply_tool_surface_to_schemas(schemas, surface)


def unavailable_line(names: Iterable[str]) -> str:
    names = sorted(set(names))
    if not names:
        return ""
    return (
        f"Unavailable (disabled by user/admin): {', '.join(names)}. "
        "Don't call them; tell the user or use an alternative."
    )


def deferred_catalog(denied: Iterable[str], *, include_mcp: bool) -> List[Dict[str, Any]]:
    """Deferred built-ins, then MCP tools, minus what this turn denies."""
    denied = set(denied)
    catalog = [
        schema for schema in FUNCTION_TOOL_SCHEMAS
        if schema_name(schema) in DEFERRED_TOOLS and schema_name(schema) not in denied
    ]
    manager = get_mcp_manager()
    if include_mcp and manager is not None:
        # Relies on: get_all_openai_schemas names each tool
        # mcp__<server_id>__<tool> and skips disabled_map's tools and the
        # internal Python servers.
        catalog += [
            schema for schema in manager.get_all_openai_schemas(_load_mcp_disabled_map())
            if mcp_server(schema_name(schema)) not in _CATALOG_EXCLUDED_MCP_SERVERS
            and schema_name(schema) not in denied
        ]
    return catalog


def mcp_server(name: str) -> str:
    """The server id of an mcp__<server>__<tool> name, else ""."""
    parts = name.split("__", 2)
    return parts[1] if len(parts) == 3 and parts[0] == "mcp" else ""


def catalog_section(catalog: Sequence[Dict[str, Any]]) -> str:
    if not catalog:
        return ""
    built_in = [schema_name(s) for s in catalog if not mcp_server(schema_name(s))]
    by_server: Dict[str, List[str]] = {}
    for schema in catalog:
        server = mcp_server(schema_name(schema))
        if server:
            by_server.setdefault(server, []).append(schema_name(schema))
    lines = [f"Deferred tools exist but are not loaded; load them with {TOOL_SEARCH} before calling them."]
    if built_in:
        lines.append(f"Built-in: {', '.join(built_in)}")
    lines += [f"MCP server {server}: {', '.join(names)}" for server, names in by_server.items()]
    return "\n".join(lines)


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search(catalog: Sequence[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Exact names after ``select:``, else catalog entries ranked by how many
    query words their name and description share (catalog order breaks ties)."""
    query = query.strip()
    if query.lower().startswith(_SELECT_PREFIX):
        by_name = {schema_name(schema): schema for schema in catalog}
        wanted = dict.fromkeys(name.strip() for name in query[len(_SELECT_PREFIX):].split(","))
        return [by_name[name] for name in wanted if name in by_name][:TOOL_SEARCH_MAX_SCHEMAS]
    terms = _words(query)
    scored = []
    for index, schema in enumerate(catalog):
        function = schema.get("function") or {}
        score = len(terms & _words(f"{function.get('name', '')} {function.get('description', '')}"))
        if score:
            scored.append((-score, index))
    return [catalog[index] for _, index in sorted(scored)[:TOOL_SEARCH_MAX_SCHEMAS]]


def loaded_names(conversation: Iterable[Dict[str, Any]]) -> List[str]:
    """Tools that tool_search loaded in the replayed turns, in load order.
    Compaction drops those rounds, so their tools unload with them."""
    names: Dict[str, None] = {}
    for message in conversation:
        if message.get("role") == "tool":
            names.update(dict.fromkeys(message.get("loaded") or ()))
    return list(names)


def offered_schemas(
    upfront: List[Dict[str, Any]], catalog: Sequence[Dict[str, Any]], loaded: Iterable[str], surface: str,
) -> List[Dict[str, Any]]:
    """The round's tool list: upfront, tool_search, then loaded tools in load
    order, so within a chat the list only grows and the prompt cache holds."""
    if not upfront:
        return []
    by_name = {schema_name(schema): schema for schema in catalog}
    extra = [TOOL_SEARCH_SCHEMA] if catalog else []
    extra += [by_name[name] for name in loaded if name in by_name]
    return upfront + _apply_tool_surface_to_schemas(extra, surface)


def run_search(call: Dict[str, Any], catalog: Sequence[Dict[str, Any]]) -> "ToolOutcome":
    """Answer one tool_search call; the loop adds the loaded tools next round."""
    try:
        arguments = json.loads(call["arguments"] or "{}")
    except (TypeError, ValueError):
        arguments = None
    if not isinstance(arguments, dict) or not isinstance(arguments.get("query"), str):
        return ToolOutcome(call["call_id"], TOOL_SEARCH, str(call["arguments"]), STATUS_FAILED,
                           EXIT_FAILED, f"{TOOL_SEARCH}: pass a 'query' string.")
    query = arguments["query"]
    found = [schema_name(schema) for schema in search(catalog, query)]
    if found:
        text = f"Loaded {', '.join(found)}; callable from your next step."
    else:
        text = f"No deferred tool matched '{query}'. Use names from the deferred tools list."
    return ToolOutcome(call["call_id"], TOOL_SEARCH, query, STATUS_DONE, EXIT_DONE, text,
                       loaded=tuple(found))


@dataclass(frozen=True)
class ExecutionScope:
    """What a tool call is bound to: the real user, chat and workspace."""

    owner: str
    session_id: str
    workspace: Optional[str]
    client_runtime_context: Optional[Dict[str, Any]]
    denied_names: frozenset


@dataclass
class ToolOutcome:
    """One call's result: the model-facing envelope plus the tool-card fields."""

    call_id: str
    tool: str
    command: str
    status: str
    exit_code: int
    model_text: str
    desc: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    # Tools a tool_search call loaded; replayed turns carry them (loaded_names).
    loaded: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == STATUS_DONE

    def envelope(self) -> Dict[str, Any]:
        """The shared tool-result shape (src/chat_agent/providers/__init__.py)."""
        envelope = {"call_id": self.call_id, "ok": self.ok, "content": self.model_text,
                    "error": None if self.ok else self.status}
        if self.loaded:
            envelope["loaded"] = list(self.loaded)
        return envelope

    def tool_event(self, round_number: int, model: str) -> Dict[str, Any]:
        """The legacy ``tool_events`` projection, plus ``status`` and ``call_id``."""
        event = {
            "round": round_number, "model": model, "tool": self.tool, "desc": self.desc,
            "command": self.command, "output": self.model_text, "exit_code": self.exit_code,
            "status": self.status, "call_id": self.call_id,
        }
        if not self.ok:
            event["error"] = True
        return event


def refused_outcome(call: Dict[str, Any], block: Any, reason: str, *, denied: bool) -> ToolOutcome:
    """A call the gate did not run: denied, or failed for unusable arguments."""
    tool, command = (block.tool_type, block.content) if block is not None else (call["name"], call["arguments"])
    if denied:
        return ToolOutcome(call["call_id"], tool, command, STATUS_DENIED, EXIT_DENIED, reason)
    return ToolOutcome(call["call_id"], tool, command, STATUS_FAILED, EXIT_FAILED, reason)


async def execute(call_id: str, block: Any, scope: ExecutionScope) -> ToolOutcome:
    """Run one admitted call through the legacy executor."""
    # Same owner strings as the legacy path: tools get the AuthManager key as
    # is (stores key rows by it), and only the authority is casefolded, as
    # create_request_authority does. Casefolding the tool owner would merge
    # accounts such as "straße" and "strasse".
    authority = RequestAuthority(
        uuid4().hex, authority_owner(scope.owner), scope.session_id, scope.workspace or "",
        (OperationGrant(canonical_tool(block.tool_type)),),
    )
    if scope.client_runtime_context is not None:
        authority = replace(authority, backend_resources=seal_backends(
            (g.tool for g in authority.grants), context=scope.client_runtime_context, owner=authority.owner))
    desc, result = await execute_tool_block(
        block,
        session_id=scope.session_id,
        disabled_tools=set(scope.denied_names),
        owner=scope.owner,
        workspace=scope.workspace,
        tool_policy=None,
        security_context=NO_TOOL_SECURITY_CONTEXT,
        exact_approval=None,
        client_runtime_context=scope.client_runtime_context,
        request_authority=authority,
    )
    if tool_result_is_successful(result):
        status, exit_code = STATUS_DONE, EXIT_DONE
    elif tool_result_was_denied(result):
        status, exit_code = STATUS_DENIED, EXIT_DENIED
    else:
        code = result.get("exit_code")
        status = STATUS_FAILED
        exit_code = code if isinstance(code, int) and code != EXIT_DONE else EXIT_FAILED
    return ToolOutcome(
        call_id, block.tool_type, block.content, status, exit_code,
        format_tool_result(desc, result), desc=desc, result=result,
    )
