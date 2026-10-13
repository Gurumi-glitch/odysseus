"""Pick the runtime for one chat turn, once, right after request parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.chat_agent import capabilities, store
from src.llm_core import _detect_provider
from src.owner_identity import auth_disabled

LEGACY = "legacy"
CHAT_AGENT = "chat_agent"

# Dialects the runtime has no adapter for (spec: Routing); everything else
# _detect_provider returns speaks OpenAI Chat Completions.
_PROVIDERS_WITHOUT_ADAPTER = frozenset({"anthropic", "ollama", "chatgpt-subscription"})


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


async def choose_runtime(facts: TurnFacts) -> str:
    if auth_disabled() or not facts.owner:
        return LEGACY
    if store.chat_runtime(facts.owner) != store.RUNTIME_NEW:
        return LEGACY
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
    if not await endpoint_has_adapter(facts.endpoint_url, facts.model, facts.headers):
        return LEGACY
    return CHAT_AGENT


async def endpoint_has_adapter(endpoint_url: str, model: str, headers: Any) -> bool:
    # Relies on: _detect_provider maps a configured endpoint URL to its wire
    # dialect, with "openai" as the Chat Completions default.
    if _detect_provider(endpoint_url) in _PROVIDERS_WITHOUT_ADAPTER:
        return False
    # Only the server's report counts: supports_tools, the Cookbook regex and
    # model-name lists stay legacy-only (#79). Unknown -> try native tools.
    return await capabilities.native_tools(endpoint_url, model, headers) != capabilities.NO
