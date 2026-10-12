"""Provider dialect adapters of the chat_agent runtime and the shapes they share.

Canonical conversation, built by ``prompt`` and serialized by each adapter:

- ``{"role": "system" | "user", "content": str | list}``
- ``{"role": "assistant", "content": str}``: a row replayed as plain text;
- ``{"role": "assistant", "blocks": [...]}``: one round of a runtime turn;
- ``{"role": "tool", "call_id", "ok", "content", "error"}``: a tool result,
  ``error`` is ``failed | denied | cancelled`` when ``ok`` is false.

Blocks: ``text {text}``, ``tool_call {call_id, name, arguments}`` and
``reasoning {provider, model, ...opaque}``. A block that carries ``item`` (the
provider's own output item) also carries ``provider``; only that provider
replays the item, unchanged (ADR 0004).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Relied on: _host_match compares the URL's hostname (or a subdomain of it),
# never a substring of the path; _detect_provider has no xAI branch and
# returns "openai" for api.x.ai.
from src.llm_core import _detect_provider, _host_match

XAI = "xai"


def provider_key(url: str) -> str:
    return XAI if _host_match(url, "x.ai") else _detect_provider(url)


@dataclass(frozen=True)
class Endpoint:
    url: str
    model: str
    headers: Optional[Dict[str, str]]


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ProviderError:
    status: Optional[int]
    message: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class RoundEnd:
    blocks: List[Dict[str, Any]]
    usage: Optional[Usage]


RoundEvent = Union[TextDelta, ProviderError, RoundEnd]


def replays_reasoning(block: Dict[str, Any], provider: str, model: str) -> bool:
    """Reasoning replays only to the provider and model that produced it."""
    return block.get("provider") == provider and block.get("model") == model


def tool_output_text(result: Dict[str, Any]) -> str:
    """OpenAI dialects have no error flag; a failure goes out as JSON (#31)."""
    if result.get("ok"):
        return str(result.get("content") or "")
    return json.dumps({"error": str(result.get("content") or ""), "status": result.get("error") or "failed"})
