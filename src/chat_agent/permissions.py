"""Permission gate of the chat_agent runtime (spec #44, Permissions; ADR 0001, 0002).

Order: deny layers (hard limits user rules never override), the user's deny
and ask rules, the sensitive set and the egress gate (asks no allow rule
lifts), the user's allow rules, then the default posture. A session rule
("Allow for this chat") lifts any ask and never a deny. Auto mode (#68)
replaces asks that are not protected.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import logging
import os
import posixpath
import re
from dataclasses import dataclass, replace
from email.utils import getaddresses
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import httpx
from sqlalchemy.exc import SQLAlchemyError

from core import database
from src import constants

# Relies on: _load_mcp_disabled_map reads every server's disabled_tools from
# the database on each call, as {server_id: set of server-local tool names}.
from src.agent_loop import _load_mcp_disabled_map
# Relies on: _parse_agent_patch is ApplyPatchTool's own patch reader; it
# returns one {"path": ...} per file the patch adds, updates or deletes, and
# raises ValueError on a patch the tool refuses.
from src.agent_tools.filesystem_tools import _parse_agent_patch
from src.chat_agent.routing import ComposerToggles
from src.chat_agent.store import RULE_LISTS
from src.path_confinement import canonical_root
# Relies on: _is_email_output_target is the predicate the scheduler routes a
# task's output_target to email by (email, email:..., or a bare address).
from src.task_scheduler import TaskScheduler
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
# Relies on: _resolve_tool_path_in_workspace returns the realpath the file
# tools open inside the workspace, or raises ValueError where they refuse.
from src.tool_execution import _resolve_tool_path_in_workspace
# manage_tasks_args / find_task are do_manage_tasks' own parsing and lookup,
# so the gate judges the task the executor will act on (no second reading).
from src.tools.system import find_task, manage_tasks_args
from src.turn_contract import FAMILY_TOOLS, canonical_tool

logger = logging.getLogger(__name__)

ALLOW = "allow"
ASK = "ask"
# Table-only levels, both ask by default: sensitive always asks whatever the
# allow rules say; protected is sensitive and asks even in auto mode.
SENSITIVE = "sensitive"
PROTECTED = "protected"
DENY = "deny"
# A call whose arguments the executor cannot take; it fails without a gate.
INVALID = "invalid"

# Nobody mode: memory, chat search, skills and session tools (legacy parity).
_NOBODY_DENIED = frozenset({
    "manage_memory", "search_chats", "manage_skills", "create_session",
    "list_sessions", "manage_session", "send_to_session", "chat_with_model",
})
# An open email reader: the agent drafts through the reader, never sends.
EMAIL_READER_DENIED = frozenset({"create_document", "send_email", "reply_to_email"})
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
# written here so the whole table is reviewable in one place. The sensitive and
# protected sets are derived from it, so a tool cannot ask here and be missing
# there. A SENSITIVE or PROTECTED row never reaches posture()'s lookup:
# is_sensitive() answers first, from the same rows.
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
    "send_email": SENSITIVE, "reply_to_email": SENSITIVE, "delete_email": SENSITIVE,
    "unsubscribe_email": SENSITIVE, "block_sender": SENSITIVE,
    # protected: auto mode never answers
    "bulk_email": PROTECTED, "manage_settings": PROTECTED, "manage_endpoints": PROTECTED,
    "manage_mcp": PROTECTED, "manage_webhooks": PROTECTED, "manage_tokens": PROTECTED,
    "vault_unlock": PROTECTED, "vault_get": PROTECTED,
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

@dataclass(frozen=True)
class Decision:
    outcome: str
    reason: str = ""
    # An ask "Allow for this chat" may answer; False for asks only a fresh
    # answer each time may lift (see decide()).
    chat_scope: bool = True
    # An ask auto mode's classifier may answer: not an explicit ask rule, a
    # deny rule the matcher cannot rule out, or a protected action (ADR 0006).
    auto: bool = False
    # Who decided, when not the rules: "auto_mode" for a classifier denial.
    by: str = ""


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
    rules: "Rules",
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
        disabled |= EMAIL_READER_DENIED
    # A bare deny rule removes the tool from context; a scoped one is decided per call.
    disabled |= {rule.tool for rule in rules.deny if rule.kind == ANY}

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


def mcp_server(name: str) -> str:
    """The server id of an mcp__<server>__<tool> name, else ""."""
    parts = name.split("__", 2)
    return parts[1] if len(parts) == 3 and parts[0] == "mcp" else ""


def _is_custom_mcp(name: str) -> bool:
    return name.startswith("mcp__") and not name.startswith(_EMAIL_MCP_PREFIX)


def _mcp_server_disabled_reason(name: str) -> Optional[str]:
    # Checked per call, not per turn: a tool loaded this turn stays offered
    # until the turn ends, even after the admin disables it on its server.
    server = mcp_server(name)
    if not server or not _is_custom_mcp(name):
        return None
    if name[len(f"mcp__{server}__"):] in _load_mcp_disabled_map().get(server, ()):
        return f"Tool '{name}' is disabled on its MCP server."
    return None


# --- Rules: `tool` or `tool(specifier)` (spec #44, Permissions; #25) ---------
#
# The matcher reads a call's path, host or recipients itself, so it can
# disagree with what the executor will do. Where its reading is not certain
# it fails closed: an allow rule does not cover the call, and a deny or ask
# rule turns it into an ask (UNSURE).

ANY = "any"
# The whole call, which only "Allow for this chat" saves (the rule syntax
# cannot spell it).
EXACT = "exact"
PATH = "path"
DOMAIN = "domain"
ACTION = "action"
RECIPIENT = "recipient"

# Shell specifiers (`bash(git:*)`) are not supported yet (2026-10-13, see
# README "Not decided yet"): these tools take the bare name only.
_SHELL_TOOLS = frozenset({"bash", "host_shell", "python"})
_PATH_TOOLS = frozenset({"read_file", "write_file", "edit_file"})
_WRITE_PATH_TOOLS = _PATH_TOOLS - {"read_file"}
_RECIPIENT_TOOLS = frozenset({"send_email"})
# The specifier word each scoped kind is written with.
_KIND_WORDS = {DOMAIN: "domain:", ACTION: "action:", RECIPIENT: "to:"}
_RULE_TEXT = re.compile(r"^([A-Za-z0-9_\-]+)(?:\((.*)\))?$", re.S)
_HTTP_SCHEMES = frozenset({"http", "https"})
_EMAIL_MCP_PREFIX = "mcp__email__"
_PATH_GLOBSTAR = "**"

# The outcome of matching one rule against one call.
HIT = "hit"
MISS = "miss"
UNSURE = "unsure"


def policy_name(name: str) -> str:
    """One spelling per tool for rules and the sensitive/protected sets: the
    bare email name for any casing of ``mcp__email__``."""
    name = str(name or "")
    if name.casefold().startswith(_EMAIL_MCP_PREFIX):
        return name[len(_EMAIL_MCP_PREFIX):].casefold()
    return canonical_tool(name)


@dataclass(frozen=True)
class Rule:
    tool: str
    kind: str = ANY
    value: str = ""

    @property
    def text(self) -> str:
        """The rule as written, also the "Allow for this chat" card label."""
        if self.kind == ANY:
            return self.tool
        if self.kind in _KIND_WORDS:
            return f"{self.tool}({_KIND_WORDS[self.kind]}{self.value})"
        if self.kind == EXACT:
            return f"this exact {self.tool} call"
        return f"{self.tool}({self.value})"

    def record(self) -> List[str]:
        """The store's form: EXACT rules have no text form."""
        return [self.tool, self.kind, self.value]


def rule_from_record(record: Any) -> Rule:
    tool, kind, value = record
    return Rule(str(tool), str(kind), str(value))


def parse_rule(text: str) -> Rule:
    """A rule as written in the Settings editor; ValueError outside the table."""
    match = _RULE_TEXT.match(str(text or "").strip())
    if not match:
        raise ValueError(f"Not a rule: {text!r}. Write tool or tool(specifier).")
    tool, spec = policy_name(match.group(1)), match.group(2)
    if spec is None:
        return Rule(tool)
    spec = spec.strip()
    if not spec:
        raise ValueError(f"Empty specifier in {text!r}.")
    if tool in _SHELL_TOOLS:
        raise ValueError(
            f"Shell command specifiers like {tool}(git:*) are not supported yet; "
            f"write the bare name {tool} to allow, ask or deny the whole tool.")
    if tool in _PATH_TOOLS:
        return Rule(tool, PATH, posixpath.normpath(spec))
    kind = (DOMAIN if tool == "web_fetch" else ACTION if tool.startswith("manage_")
            else RECIPIENT if tool in _RECIPIENT_TOOLS else None)
    if kind is None:
        raise ValueError(f"{tool} takes no specifier; write the bare name.")
    word = _KIND_WORDS[kind]
    value = spec[len(word):].strip() if spec.startswith(word) else ""
    if not value:
        raise ValueError(f"{tool} takes {word}<pattern>, not {spec!r}.")
    if kind == ACTION:
        # Actions compare as the executor normalizes them.
        return Rule(tool, kind, value.replace("-", "_").casefold())
    if kind == DOMAIN:
        return Rule(tool, kind, _domain_pattern(value))
    return Rule(tool, kind, value.lower())


def _domain_pattern(value: str) -> str:
    """Hosts compare as httpx sends them: lowercase IDNA, no trailing dot."""
    labels = value.lower().rstrip(".").split(".")
    try:
        return ".".join(label if "*" in label else label.encode("idna").decode("ascii") for label in labels)
    except UnicodeError as exc:
        raise ValueError(f"Not a domain: {value!r}.") from exc


@dataclass(frozen=True)
class Rules:
    """The user's persistent rules (Settings editor, #64)."""

    allow: Tuple[Rule, ...] = ()
    ask: Tuple[Rule, ...] = ()
    deny: Tuple[Rule, ...] = ()

    @classmethod
    def from_lists(cls, lists: Mapping[str, Iterable[str]]) -> "Rules":
        """Rules as the store keeps them; one unreadable entry never voids the rest."""
        parsed = {}
        for name in RULE_LISTS:
            parsed[name] = []
            for text in lists.get(name) or ():
                try:
                    parsed[name].append(parse_rule(text))
                except ValueError:
                    logger.warning("chat_agent: ignoring unreadable %s rule %r", name, text)
        return cls(**{name: tuple(rules) for name, rules in parsed.items()})


def _args(content: Any) -> dict:
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _path(content: Any) -> str:
    args = _args(content)
    if args:
        # The file tools strip the path they are given.
        return str(args.get("path") or "").strip()
    # read_file: a bare path; write_file with a body: "<path>\n<content>".
    return str(content or "").split("\n", 1)[0].strip()


def _clean_path(path: str) -> Optional[str]:
    """The path as a glob compares it; None when the matcher cannot tell
    where it points (home, backslashes, climbing out with ..)."""
    if not path or path.startswith("~") or "\\" in path:
        return None
    path = posixpath.normpath(path)
    if path == ".." or path.startswith("../") or "/../" in path:
        return None
    return path


def _glob_matches(pattern: str, path: str) -> bool:
    """Gitignore-style: ``*`` stays inside one path segment, ``**`` spans any."""
    def match(pattern_parts, path_parts):
        if not pattern_parts:
            return not path_parts
        if pattern_parts[0] == _PATH_GLOBSTAR:
            return any(match(pattern_parts[1:], path_parts[i:]) for i in range(len(path_parts) + 1))
        return bool(path_parts) and fnmatch.fnmatchcase(path_parts[0], pattern_parts[0]) and match(
            pattern_parts[1:], path_parts[1:])
    return match(pattern.split("/"), path.split("/"))


def _hosts(content: Any) -> Optional[List[str]]:
    """Every host the call fetches, as httpx parses it; None when unreadable."""
    args = _args(content)
    urls = args.get("urls") if isinstance(args.get("urls"), list) else [args.get("url")]
    hosts = []
    for url in urls:
        url = url.get("url") if isinstance(url, dict) else url
        try:
            parsed = httpx.URL(str(url or "").strip())
        except (httpx.InvalidURL, TypeError, ValueError):
            return None
        if parsed.scheme not in _HTTP_SCHEMES or not parsed.host:
            return None
        # raw_host is what goes on the wire: IDNA-encoded, unlike .host.
        hosts.append(parsed.raw_host.decode("ascii").lower().rstrip("."))
    return hosts


def _recipients(content: Any) -> Optional[List[str]]:
    """Every address the mail goes to; None when one does not parse cleanly."""
    args = _args(content)
    fields = []
    for key in ("to", "cc", "bcc"):
        value = args.get(key) or ""
        fields += [str(item) for item in value] if isinstance(value, list) else [str(value)]
    addresses = [address.strip().lower() for _, address in getaddresses([f for f in fields if f.strip()])]
    if any(address.count("@") != 1 or address.startswith("@") or address.endswith("@")
           for address in addresses):
        return None
    return addresses


def call_targets(content: Any) -> List[str]:
    """The addresses and hosts a call's arguments send to, as the matcher reads them."""
    return [*(_recipients(content) or ()), *(_hosts(content) or ())]


def _combine(outcomes: List[str], *, allow: bool) -> str:
    """An allow covers a call when every part hits; a deny or ask hits on any
    part, and an unreadable part makes it unsure."""
    if allow:
        return HIT if outcomes and all(o == HIT for o in outcomes) else MISS
    if HIT in outcomes:
        return HIT
    return UNSURE if UNSURE in outcomes or not outcomes else MISS


def _path_views(path: str, workspace: Optional[str]) -> Optional[List[str]]:
    """The path as written and, inside a workspace, where the executor will
    really open it (symlinks followed, /workspace alias resolved)."""
    clean = _clean_path(path)
    if clean is None:
        return None
    if not workspace:
        # Without a workspace the file tools open the path's realpath.
        return [clean, os.path.realpath(os.path.expanduser(path))]
    try:
        resolved = _resolve_tool_path_in_workspace(workspace, path)
    except ValueError:
        # The executor refuses it too; the matcher cannot say where it points.
        return None
    return [clean, os.path.relpath(resolved, canonical_root(workspace)), resolved]


def _path_match(rule: Rule, path: str, workspace: Optional[str], *, allow: bool) -> str:
    views = _path_views(path, workspace)
    comparable = [view for view in views or () if posixpath.isabs(view) == posixpath.isabs(rule.value)]
    if not comparable:
        return MISS if allow else UNSURE
    return _combine([HIT if _glob_matches(rule.value, view) else MISS for view in comparable], allow=allow)


def _match(rule: Rule, name: str, content: Any, *, allow: bool, workspace: Optional[str] = None) -> str:
    tool = policy_name(name)
    if rule.tool != tool:
        return MISS
    if rule.kind == ANY:
        return HIT
    if rule.kind == EXACT:
        return HIT if str(content or "") == rule.value else MISS
    if rule.kind == PATH:
        return _path_match(rule, _path(content), workspace, allow=allow)
    if rule.kind == ACTION:
        action = _action_from_content(tool, content)
        if not action:
            return MISS if allow else UNSURE
        return HIT if fnmatch.fnmatchcase(action, rule.value) else MISS
    values = _hosts(content) if rule.kind == DOMAIN else _recipients(content)
    if not values:
        return MISS if allow else UNSURE
    return _combine([HIT if fnmatch.fnmatchcase(v, rule.value) else MISS for v in values], allow=allow)


def rule_covers(rule: Rule, name: str, content: Any, workspace: Optional[str] = None) -> bool:
    """Whether an allow rule (or a session rule) admits this call."""
    return _match(rule, name, content, allow=True, workspace=workspace) == HIT


def session_rule(name: str, content: Any) -> Rule:
    """The rule "Allow for this chat" saves for this call: its narrowest scope
    that still covers it, else the exact call. Shell tools: the whole tool."""
    tool = policy_name(name)
    candidate = None
    if tool in _SHELL_TOOLS or tool.startswith(_MCP_PREFIX):
        candidate = Rule(tool)
    elif tool in _PATH_TOOLS and _clean_path(_path(content)):
        candidate = Rule(tool, PATH, glob.escape(_clean_path(_path(content))))
    elif tool == "web_fetch" and _hosts(content) and len(set(_hosts(content))) == 1:
        candidate = Rule(tool, DOMAIN, glob.escape(_hosts(content)[0]))
    elif tool == "manage_tasks":
        # One approved task is not every task: a later create could deliver
        # elsewhere, so the chat-wide scope is the exact call.
        candidate = None
    elif tool.startswith("manage_") and _action_from_content(tool, content):
        candidate = Rule(tool, ACTION, glob.escape(_action_from_content(tool, content)))
    elif tool in _RECIPIENT_TOOLS and _recipients(content) and len(_recipients(content)) == 1:
        candidate = Rule(tool, RECIPIENT, glob.escape(_recipients(content)[0]))
    if candidate is not None and rule_covers(candidate, tool, content):
        return candidate
    return Rule(tool, EXACT, str(content or ""))


# --- Sensitive and protected sets --------------------------------------------

# Always ask, whatever the allow rules say (spec #44, Sensitive); protected
# tools are sensitive too, so they ask even in auto mode (spec #44, Protected).
_SENSITIVE_TOOLS = frozenset(t for t, level in POSTURE.items() if level in (SENSITIVE, PROTECTED))
_PROTECTED_TOOLS = frozenset(t for t, level in POSTURE.items() if level == PROTECTED)
_RAW_API_TOOLS = frozenset({"app_api", "api_call"})
_READ_METHOD = "GET"
# Task triggers that fire from outside the user's session (#40).
_WEBHOOK_TRIGGER = "webhook"
_EMAIL_TRIGGER_EVENT = "email_received"
_MCP_PREFIX = "mcp__"
# Actions that run or reconfigure an existing task: its stored delivery counts.
_STORED_TASK_ACTIONS = frozenset({"edit", "resume", "run"})


_INSTRUCTION_NAMES = frozenset(name.casefold() for name in constants.INSTRUCTION_FILENAMES)


# The keys ApplyPatchTool reads a JSON-wrapped patch from, in its order.
_PATCH_KEYS = ("patch_text", "patchText", "patch")
# Trailing characters SMB/NTFS-style mounts drop from a file name.
_DROPPED_NAME_TAIL = ". "


def _is_instruction_name(view: str) -> bool:
    # Casefolded: a case-insensitive mount opens AGENTS.md for agents.md.
    return posixpath.basename(view).rstrip(_DROPPED_NAME_TAIL).casefold() in _INSTRUCTION_NAMES


def _targets_instruction_file(path: str, workspace: Optional[str]) -> bool:
    views = _path_views(path, workspace)
    return views is None or any(_is_instruction_name(view) for view in views)


def _patch_paths(content: Any) -> Optional[List[str]]:
    """The files an apply_patch call touches, as the tool reads them; None
    when it would refuse the patch."""
    text = str(content or "")
    args = _args(text.strip()) if text.strip().startswith("{") else {}
    if args:
        text = str(next((args[key] for key in _PATCH_KEYS if args.get(key)), ""))
    try:
        return [op["path"] for op in _parse_agent_patch(text)]
    except ValueError:
        return None


def _mentions_instruction_file(content: Any) -> bool:
    """A shell call that names an instruction file, raw or JSON-decoded. A
    tripwire only: a command can build the name at run time."""
    raw = str(content or "")
    texts = [raw]
    try:
        texts.append(json.dumps(json.loads(raw), ensure_ascii=False))
    except ValueError:
        pass
    return any(name in text.casefold() for text in texts for name in _INSTRUCTION_NAMES)


def _writes_instruction_file(tool: str, content: Any, workspace: Optional[str]) -> bool:
    """A write the next prompt and classifier would trust as the user's own
    instructions; one whose target the gate cannot place counts."""
    if tool == "apply_patch":
        # The TUI host bridge applies the patch with its own reader: a patch
        # this one refuses still counts.
        paths = _patch_paths(content)
        return paths is None or any(_targets_instruction_file(path, workspace) for path in paths)
    if tool in _SHELL_TOOLS:
        return _mentions_instruction_file(content)
    if tool not in _WRITE_PATH_TOOLS:
        return False
    return _targets_instruction_file(_path(content), workspace)


def _raw_api_writes(tool: str, content: Any) -> bool:
    """A non-GET app_api / api_call; arguments it cannot read count as a write."""
    raw = str(content or "").strip()
    args = _args(raw)
    if not args:
        if tool == "api_call" and not raw.startswith("{"):
            # The executor's line form: "integration\nMETHOD path\nbody".
            words = (raw.split("\n") + [""])[1].split()
            return bool(words) and words[0].upper() != _READ_METHOD
        return bool(raw) and raw != "{}"
    if tool == "app_api" and str(args.get("action") or "call").lower() == "endpoints":
        return False
    return str(args.get("method") or _READ_METHOD).strip().upper() != _READ_METHOD


def is_sensitive(name: str, content: Any) -> bool:
    # Casefolded: a spelling the sets miss must not skip the ask (fail closed).
    tool = policy_name(name).casefold()
    if tool in _SENSITIVE_TOOLS:
        return True
    capabilities = capabilities_for_action(tool, content)
    # Unknown tools (custom MCP) fail high on effects but stay rule-governed.
    if capabilities.known and ToolEffect.DESTRUCTIVE in capabilities.effects:
        return True
    if tool == "manage_tasks" and _action_from_content(tool, content) in _TASK_ACTIONS_THAT_ASK:
        return True
    if tool == "ui_control" and _ui_control_loosens(content):
        return True
    return tool in _RAW_API_TOOLS and _raw_api_writes(tool, content)


def task_args(content: Any) -> Optional[Dict[str, Any]]:
    """A manage_tasks call as do_manage_tasks reads it; None when it would reject it."""
    try:
        return manage_tasks_args(content)
    except ValueError:
        return None


def stored_task(owner: Optional[str], args: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The owner's task an edit/resume/run acts on, by do_manage_tasks' own
    lookup; None when that lookup fails or lands on another owner's task."""
    if not owner:
        return None
    try:
        db = database.SessionLocal()
    except SQLAlchemyError:
        logger.warning("chat_agent: task store unreadable; treating the task as unknown", exc_info=True)
        return None
    try:
        task, error = find_task(db, dict(args), owner, "lookup")
        if error or task.owner != owner:
            return None
        return _task_fields(task)
    except SQLAlchemyError:
        logger.warning("chat_agent: task lookup failed; treating the task as unknown", exc_info=True)
        return None
    finally:
        db.close()


def _task_fields(task: Any) -> Dict[str, Any]:
    return {
        "task_id": task.id, "name": task.name, "prompt": task.prompt, "task_type": task.task_type,
        "action_name": task.action, "schedule": task.schedule, "scheduled_time": task.scheduled_time,
        "cron_expression": task.cron_expression, "trigger_type": task.trigger_type,
        "trigger_event": task.trigger_event, "output_target": task.output_target,
    }


def task_card(owner: Optional[str], content: Any) -> Dict[str, Any]:
    """What a manage_tasks ask shows: the stored task under the call's changes."""
    args = task_args(content) or {}
    action = _action_from_content("manage_tasks", content)
    stored = stored_task(owner, args) if action in _STORED_TASK_ACTIONS else None
    fields = {**(stored or {}), **{k: v for k, v in args.items() if v not in (None, "")}}
    # Shorthands manage_tasks_args already folded into prompt and name.
    for alias in ("task", "description"):
        fields.pop(alias, None)
    if action == "create":
        fields.setdefault("output_target", _TASK_DEFAULT_TARGET)
    if action in _STORED_TASK_ACTIONS and stored is None:
        fields["stored_task"] = "not found"
    return fields


_TASK_DEFAULT_TARGET = "session"


def _task_delivery_protected(fields: Mapping[str, Any]) -> bool:
    target = str(fields.get("output_target") or "").strip()
    return bool(
        (target and TaskScheduler._is_email_output_target(target))
        or target.startswith(_MCP_PREFIX)
        or str(fields.get("trigger_type") or "").strip() == _WEBHOOK_TRIGGER
        or str(fields.get("trigger_event") or "").strip() == _EMAIL_TRIGGER_EVENT
    )


def is_protected(name: str, content: Any, *, owner: Optional[str] = None,
                 workspace: Optional[str] = None) -> bool:
    """Actions auto mode never answers for the user (#68). A task edit, resume
    or run whose stored task cannot be read counts as protected."""
    tool = policy_name(name).casefold()
    if tool in _PROTECTED_TOOLS or _writes_instruction_file(tool, content, workspace):
        return True
    if tool in _RAW_API_TOOLS:
        return _raw_api_writes(tool, content)
    action = _action_from_content(tool, content) if tool == "manage_tasks" else None
    if action not in _TASK_ACTIONS_THAT_ASK:
        return False
    args = task_args(content)
    if args is None or _task_delivery_protected(args):
        return True
    if action in _STORED_TASK_ACTIONS:
        stored = stored_task(owner, args)
        return stored is None or _task_delivery_protected(stored)
    return False


# --- Egress gate (ADR 0002) ---------------------------------------------------

# The runtime's own source table (#38). Email reads are both legs: a message
# body is attacker-writable content and the user's private data. The scans
# return senders and subjects, which are both too.
_EMAIL_READS = frozenset({
    "list_emails", "read_email", "search_emails", "scan_spam", "scan_email_unsubscribes",
})
_EXTERNAL_SOURCES = (
    WEB_ACCESS_TOOL_NAMES | _EMAIL_READS
    | {"pdf_extract", "trigger_research", "manage_research", "manage_bg_jobs"}
)


def _reads_private(tool: str, content: Any) -> bool:
    """A call that brings the user's private data into the context (2026-10-13
    decision, ADR 0002 amendment): every READ_PRIVATE call, which covers
    email, chat search, vault, the session/contact/account listers and the
    read actions of the manage_* tools, plus a GET raw API call. Derived for
    every tool, so a new one is classified with no list to update."""
    if tool in _RAW_API_TOOLS:
        return not _raw_api_writes(tool, content)
    # Relies on: capabilities_for_action gives a manage_* read action only
    # READ_PRIVATE, a write action only its writes, an unreadable action both,
    # and an unknown tool (custom MCP) every effect, so whatever the gate
    # cannot classify counts as a private read.
    capabilities = capabilities_for_action(tool, content)
    return ToolEffect.READ_PRIVATE in capabilities.effects


# Row metadata where a compaction summary keeps the provenance of the rows it
# summarized: their tool events are gone, the content they brought is not (#67).
PROVENANCE_KEY = "egress_provenance"


@dataclass(frozen=True)
class Provenance:
    """What the chat's context holds: external content, a private-data read."""

    external: bool = False
    private: bool = False
    # Where the external content came from, for auto mode's classifier.
    origins: Tuple[str, ...] = ()

    def after(self, name: str, content: Any = "") -> "Provenance":
        """The provenance once a call to ``name`` with ``content`` has run."""
        tool = policy_name(name)
        external = tool in _EXTERNAL_SOURCES or tool.startswith(_MCP_PREFIX)
        return Provenance(
            external=self.external or external,
            private=self.private or _reads_private(tool, content),
            origins=self.origins + ((tool,) if external and tool not in self.origins else ()),
        )

    def after_history(self, rows: Iterable[Any]) -> "Provenance":
        """Earlier turns' tool results stay in the replayed context, so their
        calls (tool events on either runtime's rows) count too, and so does
        what a compaction summary row recorded of the rows it replaced."""
        provenance = self
        for row in rows:
            metadata = (getattr(row, "metadata", None) if not isinstance(row, dict) else row.get("metadata")) or {}
            for event in metadata.get("tool_events") or ():
                if isinstance(event, dict) and event.get("tool"):
                    provenance = provenance.after(str(event["tool"]), event.get("command") or "")
            recorded = metadata.get(PROVENANCE_KEY)
            if isinstance(recorded, dict):
                # A recorded leg only ever arms the gate, so a forged one costs an ask.
                origins = recorded.get("origins") if isinstance(recorded.get("origins"), list) else []
                provenance = Provenance(
                    external=provenance.external or bool(recorded.get("external")),
                    private=provenance.private or bool(recorded.get("private")),
                    origins=provenance.origins + tuple(
                        str(o) for o in origins if str(o) not in provenance.origins),
                )
        return provenance


def _egress(name: str, content: Any) -> bool:
    tool = canonical_tool(name)
    effects = capabilities_for_action(tool, content).effects
    if ToolEffect.NETWORK_EGRESS in effects:
        return True
    # The shared registry leaves these readers off NETWORK_EGRESS, which the
    # legacy preflight blocks; here each one is read from the call it runs.
    if tool == "pdf_extract":
        # Relies on: capabilities_for_action picks pdf_extract's source the way
        # the tool does and drops the network read only when it is local.
        return ToolEffect.BROKERED_NETWORK_READ in effects
    if tool == "youtube_tool":
        return _youtube_fetches_elsewhere(content)
    return False


_YOUTUBE_HOSTS = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be",
})
# Every argument YouTubeTool reads a target from.
_YOUTUBE_TARGET_KEYS = ("url", "video_url", "video_id", "channel_url", "handle")


def _youtube_fetches_elsewhere(content: Any) -> bool:
    """A target URL off YouTube: the channel actions hand it to yt-dlp as is."""
    raw = str(content or "").strip()
    try:
        args = json.loads(raw)
    except ValueError:
        # YouTubeTool._parse_args reads non-JSON text as the url.
        args = {"url": raw}
    if not isinstance(args, dict):
        return False
    for key in _YOUTUBE_TARGET_KEYS:
        target = str(args.get(key) or "").strip()
        if not target.casefold().startswith(("http://", "https://")):
            continue
        try:
            host = (httpx.URL(target).host or "").lower().rstrip(".")
        except (httpx.InvalidURL, ValueError):
            return True
        if host not in _YOUTUBE_HOSTS:
            return True
    return False


# --- The gate ------------------------------------------------------------------


def decide(
    layers: DenyLayers,
    name: str,
    content: Any,
    *,
    rules: Rules = Rules(),
    session: Iterable[Rule] = (),
    provenance: Provenance = Provenance(),
    owner: Optional[str] = None,
    workspace: Optional[str] = None,
    auto_mode: bool = False,
) -> Decision:
    reason = layers.deny_reason(name, content) or _mcp_server_disabled_reason(name)
    if reason:
        return Decision(DENY, reason)
    deny = [_match(rule, name, content, allow=False, workspace=workspace) for rule in rules.deny]
    for rule, outcome in zip(rules.deny, deny):
        if outcome == HIT:
            return Decision(DENY, f"Denied by the user's permission rule {rule.text}.")
    if auto_mode:
        # Auto mode only adds checks: whole-tool shell allows would let every
        # command past the classifier (ADR 0006).
        rules = replace(rules, allow=tuple(rule for rule in rules.allow if rule.tool not in _SHELL_TOOLS))
        session = [rule for rule in session if rule.tool not in _SHELL_TOOLS]
    # Asks only the user may answer, every time.
    user_only = (
        UNSURE in deny
        or any(_match(rule, name, content, allow=False, workspace=workspace) != MISS for rule in rules.ask)
        or is_protected(name, content, owner=owner, workspace=workspace)
        # A task's grants come from its prompt and it runs unattended later,
        # so each one needs the user's own answer.
        or (policy_name(name) == "manage_tasks"
            and _action_from_content("manage_tasks", content) in _TASK_ACTIONS_THAT_ASK)
    )
    # Asks "Allow for this chat" never lifts: the above, sensitive actions
    # ("always ask", spec #44) and the egress gate.
    pinned = (
        user_only
        or is_sensitive(name, content)
        or (provenance.external and provenance.private and _egress(name, content))
    )
    asks = pinned or (
        posture(name, content) == ASK
        and not any(rule_covers(rule, name, content, workspace) for rule in rules.allow)
    )
    if not asks:
        return Decision(ALLOW)
    if not pinned and any(rule_covers(rule, name, content, workspace) for rule in session):
        return Decision(ALLOW)
    return Decision(ASK, chat_scope=not pinned, auto=not user_only)


def posture(name: str, content: Any) -> str:
    """The default for a call no rule decides."""
    tool = policy_name(name)
    if is_sensitive(tool, content):
        return ASK
    level = POSTURE.get(tool)
    # Callers compare against ASK; a table-only level must never read as allow.
    if level in (SENSITIVE, PROTECTED):
        return ASK
    return level or _derived_posture(capabilities_for_action(canonical_tool(name), content).effects)


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
