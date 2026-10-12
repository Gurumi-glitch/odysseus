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
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

# Relies on: _load_mcp_disabled_map reads every server's disabled_tools from
# the database on each call, as {server_id: set of server-local tool names}.
from src.agent_loop import _load_mcp_disabled_map
from src.chat_agent.routing import ComposerToggles
from src.chat_agent.store import RULE_LISTS
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
from src.turn_contract import FAMILY_TOOLS, canonical_tool

logger = logging.getLogger(__name__)

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


def _is_custom_mcp(name: str) -> bool:
    return name.startswith("mcp__") and not name.startswith("mcp__email__")


def _mcp_server_disabled_reason(name: str) -> Optional[str]:
    # Checked per call, not per turn: a tool loaded this turn stays offered
    # until the turn ends, even after the admin disables it on its server.
    parts = name.split("__", 2)
    if not _is_custom_mcp(name) or len(parts) != 3:
        return None
    if parts[2] in _load_mcp_disabled_map().get(parts[1], ()):
        return f"Tool '{name}' is disabled on its MCP server."
    return None


# --- Rules: `tool` or `tool(specifier)` (spec #44, Permissions; #25) ---------

ANY = "any"
PREFIX = "prefix"
# An exact command for shell tools; for any other tool the whole call, which
# only "Allow for this chat" saves (the rule syntax cannot spell it).
EXACT = "exact"
PATH = "path"
DOMAIN = "domain"
ACTION = "action"
RECIPIENT = "recipient"

_SHELL_TOOLS = frozenset({"bash", "host_shell"})
_PREFIX_TOOLS = _SHELL_TOOLS | {"python"}
_PATH_TOOLS = frozenset({"read_file", "write_file", "edit_file"})
_RECIPIENT_TOOLS = frozenset({"send_email"})
# The specifier word each scoped kind is written with.
_KIND_WORDS = {DOMAIN: "domain:", ACTION: "action:", RECIPIENT: "to:"}
_PREFIX_SUFFIX = ":*"
# Command substitution and redirection can run or write past an allowed
# prefix, so a command using them is never covered by a prefix allow.
_UNSCOPABLE_SHELL = re.compile(r"`|\$\(|<\(|>")
_SHELL_SEPARATORS = re.compile(r"&&|\|\||[;|&\n]")
# A deny or ask rule also looks inside substitutions and subshells.
_SHELL_SEPARATORS_DEEP = re.compile(r"&&|\|\||\$\(|[;|&\n`()<>]")
_RULE_TEXT = re.compile(r"^([A-Za-z0-9_\-]+)(?:\((.*)\))?$", re.S)


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
        if self.kind == PREFIX:
            return f"{self.tool}({self.value}{_PREFIX_SUFFIX})"
        if self.kind in _KIND_WORDS:
            return f"{self.tool}({_KIND_WORDS[self.kind]}{self.value})"
        if self.kind == EXACT and self.tool not in _PREFIX_TOOLS:
            return f"this exact {self.tool} call"
        return f"{self.tool}({self.value})"

    def record(self) -> List[str]:
        """The store's form: EXACT rules of non-shell tools have no text form."""
        return [self.tool, self.kind, self.value]


def rule_from_record(record: Any) -> Rule:
    tool, kind, value = record
    return Rule(str(tool), str(kind), str(value))


def parse_rule(text: str) -> Rule:
    """A rule as written in the Settings editor; ValueError outside the table."""
    match = _RULE_TEXT.match(str(text or "").strip())
    if not match:
        raise ValueError(f"Not a rule: {text!r}. Write tool or tool(specifier).")
    tool, spec = canonical_tool(match.group(1)), match.group(2)
    if spec is None:
        return Rule(tool)
    spec = spec.strip()
    if not spec:
        raise ValueError(f"Empty specifier in {text!r}.")
    if tool in _PREFIX_TOOLS:
        prefix = spec[:-len(_PREFIX_SUFFIX)].strip() if spec.endswith(_PREFIX_SUFFIX) else ""
        return Rule(tool, PREFIX, prefix) if prefix else Rule(tool, EXACT, spec)
    if tool in _PATH_TOOLS:
        return Rule(tool, PATH, spec)
    kind = (DOMAIN if tool == "web_fetch" else ACTION if tool.startswith("manage_")
            else RECIPIENT if tool in _RECIPIENT_TOOLS else None)
    if kind is None:
        raise ValueError(f"{tool} takes no specifier; write the bare name.")
    word = _KIND_WORDS[kind]
    value = spec[len(word):].strip() if spec.startswith(word) else ""
    if not value:
        raise ValueError(f"{tool} takes {word}<pattern>, not {spec!r}.")
    # Actions compare as the executor normalizes them; hosts and addresses ignore case.
    return Rule(tool, kind, value.replace("-", "_").casefold() if kind == ACTION else value.lower())


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


def _command(tool: str, content: Any) -> str:
    # bash and python blocks carry the raw text; host_shell blocks carry JSON.
    return str(_args(content).get("command") or "") if tool == "host_shell" else str(content or "")


def _shell_parts(tool: str, command: str, *, deep: bool) -> List[str]:
    if tool == "python":
        return [command.strip()] if command.strip() else []
    separators = _SHELL_SEPARATORS_DEEP if deep else _SHELL_SEPARATORS
    return [part.strip() for part in separators.split(command) if part.strip()]


def _path(content: Any) -> str:
    args = _args(content)
    if args:
        return str(args.get("path") or "")
    # read_file: a bare path; write_file with a body: "<path>\n<content>".
    return str(content or "").split("\n", 1)[0].strip()


def _hosts(content: Any) -> List[str]:
    args = _args(content)
    urls = args.get("urls") if isinstance(args.get("urls"), list) else [args.get("url")]
    urls = [url.get("url") if isinstance(url, dict) else url for url in urls]
    return [str(urlsplit(str(url or "")).hostname or "") for url in urls]


def _recipients(content: Any) -> List[str]:
    args = _args(content)
    found = []
    for key in ("to", "cc", "bcc"):
        value = args.get(key) or ""
        items = value if isinstance(value, list) else str(value).split(",")
        found += [str(item).strip().lower() for item in items if str(item).strip()]
    return found


def _values(rule: Rule, tool: str, content: Any, *, deep: bool) -> List[str]:
    """What a scoped rule is matched against in this call."""
    if rule.kind == PREFIX:
        command = _command(tool, content)
        if not deep and tool in _SHELL_TOOLS and _UNSCOPABLE_SHELL.search(command):
            return []
        return _shell_parts(tool, command, deep=deep)
    if rule.kind == EXACT:
        return [_command(tool, content) if tool in _PREFIX_TOOLS else str(content or "")]
    if rule.kind == PATH:
        return [_path(content)]
    if rule.kind == DOMAIN:
        return _hosts(content)
    if rule.kind == ACTION:
        return [_action_from_content(tool, content) or ""]
    if rule.kind == RECIPIENT:
        return _recipients(content)
    return []


def _value_matches(rule: Rule, value: str) -> bool:
    if rule.kind == PREFIX:
        # "git" covers "git log" and "git", never "gitk".
        return re.match(re.escape(rule.value) + r"(?![\w.\-])", value) is not None
    if rule.kind == EXACT:
        return value.strip() == rule.value.strip()
    return bool(value) and fnmatch.fnmatchcase(value, rule.value)


def _matches(rule: Rule, name: str, content: Any, *, every: bool) -> bool:
    """``every``: an allow covers a call only when every part matches; a deny
    or ask hits when any part does."""
    tool = canonical_tool(name)
    if rule.tool != tool:
        return False
    if rule.kind == ANY:
        return True
    values = _values(rule, tool, content, deep=not every)
    if not values:
        return False
    hits = [_value_matches(rule, value) for value in values]
    return all(hits) if every else any(hits)


def rule_covers(rule: Rule, name: str, content: Any) -> bool:
    """Whether an allow rule (or a session rule) admits this call."""
    return _matches(rule, name, content, every=True)


def session_rule(name: str, content: Any) -> Rule:
    """The rule "Allow for this chat" saves for this call: its narrowest scope."""
    tool = canonical_tool(name)
    candidate = None
    if tool in _SHELL_TOOLS:
        firsts = {part.split()[0] for part in _shell_parts(tool, _command(tool, content), deep=False)}
        if len(firsts) == 1:
            candidate = Rule(tool, PREFIX, firsts.pop())
    elif tool in _PATH_TOOLS and _path(content):
        candidate = Rule(tool, PATH, glob.escape(_path(content)))
    elif tool == "web_fetch" and len(set(_hosts(content))) == 1:
        candidate = Rule(tool, DOMAIN, glob.escape(_hosts(content)[0]))
    elif tool.startswith("manage_") and _action_from_content(tool, content):
        candidate = Rule(tool, ACTION, _action_from_content(tool, content))
    elif tool in _RECIPIENT_TOOLS and len(_recipients(content)) == 1:
        candidate = Rule(tool, RECIPIENT, glob.escape(_recipients(content)[0]))
    elif tool.startswith(_MCP_PREFIX):
        candidate = Rule(tool)
    if candidate is not None and rule_covers(candidate, tool, content):
        return candidate
    return Rule(tool, EXACT, _command(tool, content) if tool in _PREFIX_TOOLS else str(content or ""))


# --- Sensitive and protected sets --------------------------------------------

# Always ask, whatever the allow rules say (spec #44, Sensitive).
_SENSITIVE_TOOLS = frozenset({
    "send_email", "reply_to_email", "bulk_email", "delete_email", "unsubscribe_email",
    "block_sender", "manage_settings", "manage_endpoints", "manage_mcp", "manage_webhooks",
    "manage_tokens", "vault_unlock", "vault_get",
})
# Ask even in auto mode (spec #44, Protected).
_PROTECTED_TOOLS = frozenset({
    "manage_settings", "manage_endpoints", "manage_mcp", "manage_webhooks", "manage_tokens",
    "vault_unlock", "vault_get", "bulk_email",
})
_RAW_API_TOOLS = frozenset({"app_api", "api_call"})
_READ_METHOD = "GET"
# Task triggers that fire from outside the user's session (#40).
_WEBHOOK_TRIGGER = "webhook"
_EMAIL_TRIGGER_EVENT = "email_received"
_MCP_PREFIX = "mcp__"


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
    tool = canonical_tool(name)
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


def is_protected(name: str, content: Any) -> bool:
    """Actions auto mode never answers for the user (#68)."""
    tool = canonical_tool(name)
    if tool in _PROTECTED_TOOLS:
        return True
    if tool in _RAW_API_TOOLS:
        return _raw_api_writes(tool, content)
    if tool != "manage_tasks" or _action_from_content(tool, content) not in _TASK_ACTIONS_THAT_ASK:
        return False
    args = _args(content)
    target = str(args.get("output_target") or "").strip()
    return bool(
        (target and TaskScheduler._is_email_output_target(target))
        or target.startswith(_MCP_PREFIX)
        or str(args.get("trigger_type") or "").strip() == _WEBHOOK_TRIGGER
        or str(args.get("trigger_event") or "").strip() == _EMAIL_TRIGGER_EVENT
    )


# --- Egress gate (ADR 0002) ---------------------------------------------------

# The runtime's own source table (#38). Email reads are both legs: a message
# body is attacker-writable content and the user's private data.
_EMAIL_READS = frozenset({"list_emails", "read_email", "search_emails"})
_EXTERNAL_SOURCES = (
    WEB_ACCESS_TOOL_NAMES | _EMAIL_READS
    | {"pdf_extract", "trigger_research", "manage_research", "manage_bg_jobs"}
)
_PRIVATE_SOURCES = _EMAIL_READS | {"search_chats", "vault_get", "vault_search"}


@dataclass(frozen=True)
class Provenance:
    """What this turn has taken in: external content, a private-data read."""

    external: bool = False
    private: bool = False

    def after(self, name: str) -> "Provenance":
        """The provenance once a call to ``name`` has run."""
        tool = canonical_tool(name)
        return Provenance(
            external=self.external or tool in _EXTERNAL_SOURCES or tool.startswith(_MCP_PREFIX),
            private=self.private or tool in _PRIVATE_SOURCES,
        )


def _egress(name: str, content: Any) -> bool:
    return ToolEffect.NETWORK_EGRESS in capabilities_for_action(canonical_tool(name), content).effects


# --- The gate ------------------------------------------------------------------


def decide(
    layers: DenyLayers,
    name: str,
    content: Any,
    *,
    rules: Rules = Rules(),
    session: Iterable[Rule] = (),
    provenance: Provenance = Provenance(),
) -> Decision:
    reason = layers.deny_reason(name, content) or _mcp_server_disabled_reason(name)
    if reason:
        return Decision(DENY, reason)
    for rule in rules.deny:
        if _matches(rule, name, content, every=False):
            return Decision(DENY, f"Denied by the user's permission rule {rule.text}.")
    asks = (
        any(_matches(rule, name, content, every=False) for rule in rules.ask)
        or is_sensitive(name, content)
        or (provenance.external and provenance.private and _egress(name, content))
        or (posture(name, content) == ASK
            and not any(rule_covers(rule, name, content) for rule in rules.allow))
    )
    if asks and not any(rule_covers(rule, name, content) for rule in session):
        return Decision(ASK)
    return Decision(ALLOW)


def posture(name: str, content: Any) -> str:
    """The default for a call no rule decides."""
    tool = canonical_tool(name)
    if is_sensitive(tool, content):
        return ASK
    return POSTURE.get(tool) or _derived_posture(capabilities_for_action(tool, content).effects)

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
