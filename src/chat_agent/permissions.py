"""Permission gate of the chat_agent runtime: deny layers, then default posture.

Deny layers (spec #44, Permissions) are hard limits user rules can never
override. Every call they do not deny gets the default posture: ``allow``
runs, ``ask`` needs the user. Rules, session rules, the protected set, the
egress gate and auto mode join in later tickets (#62, #68).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.chat_agent.routing import ComposerToggles
# get_mcp_manager: the process-wide MCP manager, None before startup.
from src.tool_utils import get_mcp_manager
# Each helper below is relied on as:
#   blocked_tools_for_owner: the non-admin hiding set, empty for admins;
#   delegated_credential_blocked_tools: names an API-token run may never reach;
#   plan_mode_disabled_tools: every built-in name plan mode blocks (bare spellings);
#   email_tool_policy_names: a tool's bare and mcp__email__ spellings.
from src.tool_security import (
    blocked_tools_for_owner,
    delegated_credential_blocked_tools,
    email_tool_policy_names,
    plan_mode_disabled_tools,
)
# capabilities_for_action: the effects of one concrete call, unknown tools
# failing high; delegated_tool_is_blocked: the content-level token block;
# _action_from_content: a multiplexed tool's action as the executor reads it,
# including manage_tasks' implicit "create".
from src.tool_capabilities import (
    ToolEffect,
    _action_from_content,
    capabilities_for_action,
    delegated_tool_is_blocked,
)
# WEB_ACCESS_TOOL_NAMES is exactly the spec's Web-off family; FAMILY_TOOLS
# "shell_files" is what legacy denies when can_use_bash is off.
from src.tool_policy import WEB_ACCESS_TOOL_NAMES, web_search_enabled_for_turn
from src.turn_contract import FAMILY_TOOLS, canonical_tool

ALLOW = "allow"
ASK = "ask"
DENY = "deny"
# A call whose arguments the executor cannot take; it fails without a gate.
INVALID = "invalid"

# Nobody mode: memory, chat search, skills and session tools (legacy parity).
_NOBODY_DENIED = frozenset({
    "manage_memory", "search_chats", "manage_skills", "create_session",
    "list_sessions", "manage_session", "send_to_session", "chat_with_model",
})
# An open email reader: the agent drafts through the reader, never sends.
_EMAIL_READER_DENIED = frozenset({"create_document", "send_email", "reply_to_email"})
# Per-user privileges -> tools they deny (legacy chat route parity).
_PRIVILEGE_DENIED: Mapping[str, frozenset] = {
    "can_use_bash": FAMILY_TOOLS["shell_files"],
    "can_use_browser": frozenset({"private_browser"}),
    "can_use_documents": frozenset({
        "manage_documents", "create_document", "edit_document", "update_document",
        "suggest_document",
    }),
    "can_generate_images": frozenset({"generate_image", "edit_image"}),
    "can_manage_memory": frozenset({"manage_memory", "manage_skills"}),
    "can_use_research": frozenset({"trigger_research", "manage_research"}),
}

# Default posture of every native tool and built-in email tool (spec #44,
# Permissions; #25). Tools the decided table did not name got the posture
# derived from their capability classification (_derived_posture) and are
# written here so the whole table is reviewable in one place.
POSTURE: Mapping[str, str] = {
    # user interaction
    "ask_user": ALLOW, "update_plan": ALLOW,
    # workspace reads
    "get_workspace": ALLOW, "glob": ALLOW, "grep": ALLOW, "ls": ALLOW, "read_file": ALLOW,
    "inspect_media": ALLOW, "extract_text": ALLOW, "transcribe_media": ALLOW,
    # private reads
    "search_chats": ALLOW, "list_sessions": ALLOW, "resolve_contact": ALLOW,
    "list_email_accounts": ALLOW, "list_emails": ALLOW, "read_email": ALLOW,
    "search_emails": ALLOW, "scan_spam": ALLOW, "scan_email_unsubscribes": ALLOW,
    "list_models": ALLOW, "list_served_models": ALLOW, "list_downloads": ALLOW,
    "list_cached_models": ALLOW, "list_serve_presets": ALLOW, "list_cookbook_servers": ALLOW,
    "tail_serve_output": ALLOW, "search_hf_models": ALLOW,
    # web
    "web_search": ALLOW, "web_fetch": ALLOW, "get_weather": ALLOW, "youtube_tool": ALLOW,
    "private_browser": ALLOW, "pdf_extract": ALLOW,
    # Odysseus-data writes (destructive actions ask, see posture())
    "todowrite": ALLOW, "create_document": ALLOW, "edit_document": ALLOW,
    "suggest_document": ALLOW, "update_document": ALLOW, "manage_documents": ALLOW,
    "manage_memory": ALLOW, "manage_notes": ALLOW, "manage_calendar": ALLOW,
    "manage_contact": ALLOW, "manage_research": ALLOW, "manage_session": ALLOW,
    "manage_tasks": ALLOW, "manage_skills": ALLOW, "create_session": ALLOW,
    "draft_email": ALLOW, "draft_email_reply": ALLOW, "ai_draft_email_reply": ALLOW,
    # reversible email state
    "archive_email": ALLOW, "mark_email_read": ALLOW, "manage_email_state": ALLOW,
    # UI (loosening actions ask, see posture())
    "ui_control": ALLOW,
    # internal model calls
    "chat_with_model": ALLOW, "ask_teacher": ALLOW, "generate_image": ALLOW,
    "edit_image": ALLOW, "trigger_research": ALLOW, "send_to_session": ALLOW,
    "pipeline": ALLOW,
    # workspace writes
    "write_file": ASK, "edit_file": ASK, "apply_patch": ASK, "download_attachment": ASK,
    # execution
    "bash": ASK, "python": ASK, "host_shell": ASK, "manage_bg_jobs": ASK,
    # raw access
    "api_call": ASK, "app_api": ASK,
    # model download / serve / stop
    "download_model": ASK, "serve_model": ASK, "serve_preset": ASK,
    "stop_served_model": ASK, "adopt_served_model": ASK, "cancel_download": ASK,
    # sensitive: always ask
    "send_email": ASK, "reply_to_email": ASK, "bulk_email": ASK, "delete_email": ASK,
    "unsubscribe_email": ASK, "block_sender": ASK, "manage_settings": ASK,
    "manage_endpoints": ASK, "manage_mcp": ASK, "manage_webhooks": ASK,
    "manage_tokens": ASK, "vault_unlock": ASK, "vault_get": ASK,
}

# Effects of a call that may run without asking (spec #44 default posture:
# reads, web, Odysseus-data writes, UI, user interaction).
_ALLOWED_EFFECTS = frozenset({
    ToolEffect.READ_PUBLIC, ToolEffect.READ_WORKSPACE, ToolEffect.READ_PRIVATE,
    ToolEffect.BROKERED_NETWORK_READ, ToolEffect.WRITE_PRIVATE,
    ToolEffect.UI_SIDE_EFFECT, ToolEffect.USER_INTERACTION,
})
# manage_tasks actions that can set up an unattended delivery channel (#40).
_TASK_ACTIONS_THAT_ASK = frozenset({"create", "edit", "resume", "run"})
# ui_control toggle names (with the executor's aliases) whose "on" loosens a
# composer limit, and the Nobody-mode names whose "off" does.
_LOOSENING_ON_TOGGLES = frozenset({"web", "search", "websearch", "web_search",
                                   "bash", "shell", "terminal"})
_NOBODY_TOGGLES = frozenset({"incognito", "private"})
_TOGGLE_ON_VALUES = frozenset({"on", "true", "1", "yes", "enable", "enabled"})

APPROVAL_PENDING_REASON = (
    "This action needs your approval; approvals arrive soon on this runtime, "
    "so it was not run."
)


@dataclass(frozen=True)
class Decision:
    outcome: str
    reason: str = ""


@dataclass(frozen=True)
class DenyLayers:
    """The turn's hard limits. ``hidden`` never reaches the model;
    ``disabled`` is listed to it as unavailable."""

    hidden: frozenset
    disabled: frozenset
    hide_all_mcp: bool
    delegated_credential: bool

    @property
    def denied_names(self) -> frozenset:
        return self.hidden | self.disabled

    def deny_reason(self, name: str, content: Any) -> Optional[str]:
        spellings = email_tool_policy_names(name)
        if spellings & self.hidden or (self.hide_all_mcp and _is_custom_mcp(name)):
            return f"Tool '{name}' is not available to this account."
        if spellings & self.disabled:
            return f"Tool '{name}' is disabled for this turn by the user or admin."
        if self.delegated_credential and delegated_tool_is_blocked(name, content):
            return f"Tool '{name}' cannot read the workspace for an API-token request."
        return None


def build_deny_layers(
    *,
    owner: str,
    toggles: ComposerToggles,
    privileges: Mapping[str, Any],
    delegated_credential: bool,
    active_email_open: bool,
    global_disabled_tools: Any,
) -> DenyLayers:
    hidden = set(blocked_tools_for_owner(owner))
    # Non-admins never see MCP tools (managed deny, #38); built-in email is
    # covered by its bare names in the non-admin set.
    hide_all_mcp = bool(hidden) or delegated_credential
    if delegated_credential:
        hidden |= delegated_credential_blocked_tools()

    disabled = set()
    if isinstance(global_disabled_tools, list):
        disabled |= {str(name) for name in global_disabled_tools}
    for privilege, names in _PRIVILEGE_DENIED.items():
        if not privileges.get(privilege, True):
            disabled |= names
    # A missing Shell field means allowed, as on legacy.
    if toggles.allow_bash is not None and str(toggles.allow_bash).lower() != "true":
        disabled.add("bash")
    if not web_search_enabled_for_turn(toggles.allow_web_search, toggles.use_web):
        disabled |= WEB_ACCESS_TOOL_NAMES
    if toggles.plan_mode:
        disabled |= plan_mode_disabled_tools() | _plan_mode_mcp_writes()
    if toggles.incognito:
        disabled |= _NOBODY_DENIED
    if active_email_open:
        disabled |= _EMAIL_READER_DENIED

    return DenyLayers(
        hidden=frozenset(hidden),
        disabled=frozenset(disabled - hidden),
        hide_all_mcp=hide_all_mcp,
        delegated_credential=delegated_credential,
    )


def _plan_mode_mcp_writes() -> set:
    # Relies on: plan_mode_blocked_mcp() returns, second, the qualified
    # mcp__<server>__<tool> names of every MCP tool that is not read-only.
    manager = get_mcp_manager()
    if manager is None:
        return set()
    _, qualified = manager.plan_mode_blocked_mcp()
    return set(qualified)


def _is_custom_mcp(name: str) -> bool:
    return name.startswith("mcp__") and not name.startswith("mcp__email__")


def decide(layers: DenyLayers, name: str, content: Any) -> Decision:
    reason = layers.deny_reason(name, content)
    if reason:
        return Decision(DENY, reason)
    if posture(name, content) == ASK:
        return Decision(ASK, APPROVAL_PENDING_REASON)
    return Decision(ALLOW)


def posture(name: str, content: Any) -> str:
    tool = canonical_tool(name)
    capabilities = capabilities_for_action(tool, content)
    if ToolEffect.DESTRUCTIVE in capabilities.effects:
        return ASK
    if tool == "manage_tasks" and _action_from_content(tool, content) in _TASK_ACTIONS_THAT_ASK:
        return ASK
    if tool == "ui_control" and _ui_control_loosens(content):
        return ASK
    return POSTURE.get(tool) or _derived_posture(capabilities.effects)


def _derived_posture(effects) -> str:
    return ALLOW if effects <= _ALLOWED_EFFECTS else ASK


def _ui_control_loosens(content: Any) -> bool:
    # ui_control content is "<action> <name> <value>" on its first line.
    parts = str(content or "").strip().split("\n", 1)[0].split()
    if not parts:
        return False
    action = parts[0].lower()
    if action == "switch_model":
        return True
    if action != "toggle" or len(parts) < 3:
        return False
    name, turned_on = parts[1].lower(), parts[2].lower() in _TOGGLE_ON_VALUES
    return (name in _LOOSENING_ON_TOGGLES and turned_on) or (name in _NOBODY_TOGGLES and not turned_on)
