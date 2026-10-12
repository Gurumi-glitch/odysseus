"""Tool catalog and per-call execution wrapper of the chat_agent runtime.

The catalog is the native schemas minus the deferred table (ADR 0003); the
tool list never changes with the user's wording. The wrapper runs one admitted
call through the legacy executor whole, with a fresh request authority that
binds resources and decides nothing (#34): the permission gate already did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

# Relies on: _apply_tool_surface_to_schemas returns [] for "none" and
# description-stripped schemas for "compact".
from src.agent_loop import _apply_tool_surface_to_schemas
from src.agent_runtime.authority import OperationGrant, RequestAuthority
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

    @property
    def ok(self) -> bool:
        return self.status == STATUS_DONE

    def model_content(self) -> str:
        if self.ok:
            return self.model_text
        return json.dumps({"error": self.model_text, "status": self.status})


def refused_outcome(call: Any, block: Any, decision: Any) -> ToolOutcome:
    """A call the gate did not run: unusable arguments fail, the rest are denied."""
    if block is None:
        return ToolOutcome(call.call_id, call.name, call.arguments, STATUS_FAILED,
                           EXIT_FAILED, decision.reason)
    return ToolOutcome(call.call_id, block.tool_type, block.content, STATUS_DENIED,
                       EXIT_DENIED, decision.reason)


async def execute(call_id: str, block: Any, scope: ExecutionScope) -> ToolOutcome:
    """Run one admitted call through the legacy executor."""
    owner = scope.owner.strip().casefold()
    authority = RequestAuthority(
        uuid4().hex, owner, scope.session_id, scope.workspace or "",
        (OperationGrant(canonical_tool(block.tool_type)),),
    )
    if scope.client_runtime_context is not None:
        authority = replace(authority, backend_resources=seal_backends(
            (g.tool for g in authority.grants), context=scope.client_runtime_context, owner=owner))
    desc, result = await execute_tool_block(
        block,
        session_id=scope.session_id,
        disabled_tools=set(scope.denied_names),
        owner=owner,
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
