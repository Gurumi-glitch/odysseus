"""Interim session rules: what "Allow for this chat" on an ask card allows.

A rule is ``tool(specifier)`` derived from the approved call (spec #44, Rules
and the ask card), bound to the owner and the chat, held in memory only and
read only by this runtime. #62 moves session rules into the runtime store and
brings the full matcher; until then a call this module can't scope is allowed
again only when it is exactly the approved call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from src.chat_agent.asks import owner_key
# _action_from_content: a multiplexed tool's action as the executor reads it.
from src.tool_capabilities import _action_from_content
from src.turn_contract import canonical_tool

PREFIX = "prefix"
PATH = "path"
ACTION = "action"
RECIPIENT = "recipient"
ANY = "any"
EXACT = "exact"

_SHELL_TOOLS = frozenset({"bash", "host_shell"})
_PATH_TOOLS = frozenset({"write_file", "edit_file"})
# Command substitution and redirection can run or write past the approved
# prefix, so a command that uses them is never covered by a prefix rule.
_UNSCOPABLE_SHELL = re.compile(r"`|\$\(|<\(|>")
_SHELL_SEPARATORS = re.compile(r"&&|\|\||[;|&\n]")


@dataclass(frozen=True)
class SessionRule:
    tool: str
    kind: str
    value: str

    @property
    def label(self) -> str:
        if self.kind == PREFIX:
            return f"{self.tool}({self.value}:*)"
        if self.kind == ACTION:
            return f"{self.tool}(action:{self.value})"
        if self.kind in (PATH, RECIPIENT):
            return f"{self.tool}({self.value})"
        if self.kind == ANY:
            return self.tool
        return f"this exact {self.tool} call"


_RULES: Dict[Tuple[str, str], Set[SessionRule]] = {}


def _args(content: Any) -> dict:
    try:
        parsed = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _shell_command(tool: str, content: Any) -> str:
    # bash blocks carry the raw command; host_shell blocks carry JSON args.
    return str(_args(content).get("command") or "") if tool == "host_shell" else str(content or "")


def _shell_parts(command: str) -> Optional[list]:
    if _UNSCOPABLE_SHELL.search(command):
        return None
    parts = [part.split() for part in _SHELL_SEPARATORS.split(command)]
    return [words[0] for words in parts if words] or None


def _path(tool: str, content: Any) -> str:
    args = _args(content)
    if args:
        return str(args.get("path") or "")
    # write_file with a body: "<path>\n<content>" (function_call_to_tool_block).
    return str(content or "").split("\n", 1)[0].strip() if tool == "write_file" else ""


def _recipient(content: Any) -> str:
    args = _args(content)
    if args.get("cc") or args.get("bcc"):
        return ""
    return str(args.get("to") or "").strip().lower()


def _specifier(tool: str, content: Any) -> Tuple[str, str]:
    if tool in _SHELL_TOOLS:
        commands = _shell_parts(_shell_command(tool, content))
        if commands and len(set(commands)) == 1:
            return PREFIX, commands[0]
    elif tool in _PATH_TOOLS:
        path = _path(tool, content)
        if path:
            return PATH, path
    elif tool.startswith("manage_"):
        action = _action_from_content(tool, content)
        if action:
            return ACTION, action
    elif tool == "send_email":
        recipient = _recipient(content)
        if recipient:
            return RECIPIENT, recipient
    elif tool.startswith("mcp__"):
        return ANY, ""
    return EXACT, str(content or "")


def rule_for(tool_name: str, content: Any) -> SessionRule:
    """The rule "Allow for this chat" saves for this exact call."""
    tool = canonical_tool(tool_name)
    kind, value = _specifier(tool, content)
    return SessionRule(tool, kind, value)


def _covers(rule: SessionRule, tool: str, content: Any) -> bool:
    if rule.tool != tool:
        return False
    if rule.kind == PREFIX:
        commands = _shell_parts(_shell_command(tool, content))
        return bool(commands) and all(command == rule.value for command in commands)
    return rule_for(tool, content) == rule or rule.kind == ANY


def grant(owner: str, session_id: str, rule: SessionRule) -> None:
    _RULES.setdefault((owner_key(owner), session_id), set()).add(rule)


def allows(owner: str, session_id: str, tool_name: str, content: Any) -> bool:
    tool = canonical_tool(tool_name)
    rules = _RULES.get((owner_key(owner), session_id), ())
    return any(_covers(rule, tool, content) for rule in rules)
