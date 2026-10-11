"""OpenAI Chat Completions dialect (OpenRouter, NVIDIA, LM Studio, other compatible).

Transport, headers and endpoint resolution come from the legacy provider layer.
The legacy message sanitizer is deliberately not used: it drops
``reasoning_details`` and rewrites history (spec #44, Providers).
"""

from __future__ import annotations

import json
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


RoundEvent = Union[TextDelta, ProviderError]


def build_payload(
    endpoint: Endpoint,
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    provider = _detect_provider(endpoint.url)
    payload: Dict[str, Any] = {"model": endpoint.model, "messages": messages, "stream": True}
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
) -> AsyncIterator[RoundEvent]:
    payload = build_payload(endpoint, messages, temperature=temperature, max_tokens=max_tokens)
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
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("error"), dict):
                yield ProviderError(None, str(chunk["error"].get("message") or "Provider stream error"))
                return
            for choice in chunk.get("choices") or ():
                text = (choice.get("delta") or {}).get("content")
                if text:
                    yield TextDelta(text)
