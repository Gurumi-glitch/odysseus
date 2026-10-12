"""OpenAI Chat Completions dialect (OpenRouter, NVIDIA, LM Studio, other compatible).

Transport, headers and endpoint resolution come from the legacy provider layer.
The legacy message sanitizer is deliberately not used: it drops
``reasoning_details`` and rewrites history (spec #44, Providers).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

# Each helper below is module-level in llm_core and relied on as:
#   _detect_provider: endpoint URL -> provider key ("openrouter", "openai", ...);
#   _normalize_openai_chat_url: base URL -> its /chat/completions URL;
#   _provider_headers: session headers + provider-required headers;
#   _omit_temperature / _uses_max_completion_tokens: per-model payload rules;
#   _get_http_client: the shared AsyncClient (connection reuse, TLS overrides);
#   _stream_timeout: streaming httpx.Timeout from a read timeout in seconds;
#   _format_upstream_error: HTTP status + body -> user-readable sentence.
from src.llm_core import (
    LLMConfig,
    _detect_provider,
    _format_upstream_error,
    _get_http_client,
    _normalize_openai_chat_url,
    _omit_temperature,
    _provider_headers,
    _stream_timeout,
    _uses_max_completion_tokens,
)


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
class ToolCall:
    call_id: str
    name: str
    arguments: str

    def to_message(self) -> Dict[str, Any]:
        return {"id": self.call_id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments}}


RoundEvent = Union[TextDelta, ToolCall, ProviderError]


def assistant_message(text: str, calls: List[ToolCall]) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = [call.to_message() for call in calls]
    return message


def tool_message(call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def build_payload(
    endpoint: Endpoint,
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    provider = _detect_provider(endpoint.url)
    payload: Dict[str, Any] = {"model": endpoint.model, "messages": messages, "stream": True}
    if tools:
        # parallel_tool_calls is never sent: the model may still issue several
        # calls per round, and the loop runs them one at a time (spec #44).
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if not _omit_temperature(provider, endpoint.model):
        payload["temperature"] = temperature
    if max_tokens > 0:
        key = "max_completion_tokens" if _uses_max_completion_tokens(endpoint.model) else "max_tokens"
        payload[key] = max_tokens
    return payload


def _request_headers(endpoint: Endpoint, messages: List[Dict[str, Any]]) -> Dict[str, str]:
    provider = _detect_provider(endpoint.url)
    headers = _provider_headers(provider, endpoint.headers)
    if provider == "copilot":
        # Copilot rejects requests without its per-request initiator header.
        from src.copilot import apply_request_headers
        apply_request_headers(headers, messages)
    return headers


async def stream_round(
    endpoint: Endpoint,
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[RoundEvent]:
    """Stream one round: text deltas as they arrive, then its tool calls."""
    payload = build_payload(endpoint, messages, temperature=temperature,
                            max_tokens=max_tokens, tools=tools)
    # Tool calls arrive as fragments keyed by index; they are complete only
    # when the stream ends.
    partial_calls: Dict[int, Dict[str, str]] = {}
    target_url = _normalize_openai_chat_url(endpoint.url)
    client = _get_http_client()
    async with client.stream(
        "POST",
        target_url,
        json=payload,
        headers=_request_headers(endpoint, messages),
        timeout=_stream_timeout(LLMConfig.STREAM_TIMEOUT),
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            yield ProviderError(
                response.status_code,
                _format_upstream_error(response.status_code, body, endpoint.url),
            )
            return
        async for line in response.aiter_lines():
            # SSE allows "data:value" without the space.
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("error"), dict):
                yield ProviderError(None, str(chunk["error"].get("message") or "Provider stream error"))
                return
            for choice in chunk.get("choices") or ():
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    yield TextDelta(delta["content"])
                for fragment in delta.get("tool_calls") or ():
                    _merge_tool_call_fragment(partial_calls, fragment)
    for index in sorted(partial_calls):
        call = partial_calls[index]
        yield ToolCall(call["id"] or f"call_{uuid.uuid4().hex}", call["name"], call["arguments"])


def _merge_tool_call_fragment(partial_calls: Dict[int, Dict[str, str]], fragment: Dict[str, Any]) -> None:
    call = partial_calls.setdefault(int(fragment.get("index") or 0),
                                    {"id": "", "name": "", "arguments": ""})
    if fragment.get("id"):
        call["id"] = fragment["id"]
    function = fragment.get("function") or {}
    if function.get("name"):
        call["name"] += function["name"]
    if function.get("arguments"):
        call["arguments"] += function["arguments"]
