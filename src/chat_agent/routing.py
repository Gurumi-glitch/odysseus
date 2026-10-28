"""Pick the runtime for one chat turn, once, right after request parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from src.chat_agent import capabilities, store
from src.llm_core import _detect_provider
from src.owner_identity import auth_disabled

LEGACY = "legacy"
CHAT_AGENT = "chat_agent"

# Dialects the runtime has no adapter for (spec: Routing); everything else
# _detect_provider returns speaks OpenAI Chat Completions.
_PROVIDERS_WITHOUT_ADAPTER = frozenset({"anthropic", "ollama"})


@dataclass(frozen=True)
class ComposerToggles:
    """The composer toggles as the request sent them, before the route's
    wording-based promotion rewrites ``allow_bash``/``allow_web_search``."""

    mode: str
    allow_bash: Optional[str]
    allow_web_search: Optional[str]
    use_web: Optional[str]
    plan_mode: bool
    incognito: bool
    workspace: Optional[str]


@dataclass(frozen=True)
class TurnFacts:
    owner: Optional[str]
    agent_turn: bool
    compare: bool
    client_surface: bool
    image_generation: bool
    research: bool
    approval_continuation: bool
    internal_tool_request: bool
    endpoint_url: str
    model: str
    headers: Any


# Why the next agent turn of a chat goes to legacy (capabilities route).
REASON_AUTH_DISABLED = "auth_disabled"
REASON_RUNTIME_OFF = "runtime_off"
REASON_NO_ADAPTER = "no_adapter"
REASON_NO_NATIVE_TOOLS = "no_native_tools"


async def choose_runtime(facts: TurnFacts) -> str:
    if not facts.agent_turn:
        return LEGACY
    if (
        facts.compare
        or facts.client_surface
        or facts.image_generation
        or facts.research
        or facts.approval_continuation
        or facts.internal_tool_request
    ):
        return LEGACY
    runtime, _ = await next_turn_route(facts.owner, facts.endpoint_url, facts.model, facts.headers)
    return runtime


async def next_turn_route(owner: Optional[str], endpoint_url: str, model: str, headers: Any) -> Tuple[str, str]:
    """The runtime an ordinary agent turn of this owner on this (endpoint,
    model) takes, and why it is legacy ("" when it isn't)."""
    if auth_disabled() or not owner:
        return LEGACY, REASON_AUTH_DISABLED
    if store.chat_runtime(owner) != store.RUNTIME_NEW:
        return LEGACY, REASON_RUNTIME_OFF
    if not has_adapter(endpoint_url):
        return LEGACY, REASON_NO_ADAPTER
    # Only the server's report counts: supports_tools, the Cookbook regex and
    # model-name lists stay legacy-only (#79). Unknown -> try native tools.
    if await capabilities.native_tools(endpoint_url, model, headers) == capabilities.NO:
        return LEGACY, REASON_NO_NATIVE_TOOLS
    return CHAT_AGENT, ""


async def chat_capabilities(owner: Optional[str], endpoint_url: str, model: str, headers: Any) -> dict:
    """The capabilities route's answer: the route the chat's next agent turn
    takes and the thinking levels its (endpoint, model) takes."""
    runtime, reason = await next_turn_route(owner, endpoint_url, model, headers)
    available = has_adapter(endpoint_url)
    levels = (
        await capabilities.thinking_levels(endpoint_url, model, headers) if available
        else capabilities.ThinkingLevels()
    )
    return {
        "route": runtime,
        "reason": reason,
        "thinking": {"available": available, "binary": levels.binary, "levels": levels.support},
    }


def has_adapter(endpoint_url: str) -> bool:
    # Relies on: _detect_provider maps a configured endpoint URL to its wire
    # dialect, with "openai" as the Chat Completions default.
    return _detect_provider(endpoint_url) not in _PROVIDERS_WITHOUT_ADAPTER
