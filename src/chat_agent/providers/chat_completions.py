"""OpenAI Chat Completions dialect (OpenRouter, NVIDIA, LM Studio, other compatible).

Transport, headers and endpoint resolution come from the legacy provider layer.
The legacy message sanitizer is deliberately not used: it drops
``reasoning_details`` and rewrites history (spec #44, Providers).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Sequence

from src.chat_agent.providers import (
    Endpoint,
    ProviderError,
    RoundEnd,
    RoundEvent,
    TextDelta,
    provider_key,
    replays_reasoning,
    tool_output_text,
)

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

# The fields reasoning comes back on and must go back on, unchanged: OpenRouter
# reasoning_details (in order) and the reasoning_content of DeepSeek-style servers.
_REASONING_FIELDS = ("reasoning_details", "reasoning_content")


def build_payload(
    endpoint: Endpoint,
    conversation: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    tools: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    provider = _detect_provider(endpoint.url)
    messages = _messages(conversation, provider_key(endpoint.url), endpoint.model)
    payload: Dict[str, Any] = {"model": endpoint.model, "messages": messages, "stream": True}
    if tools:
        # parallel_tool_calls is never sent: the model may still issue several
        # calls per round, and the loop runs them one at a time (spec #44).
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    if not _omit_temperature(provider, endpoint.model):
        payload["temperature"] = temperature
    if max_tokens > 0:
        key = "max_completion_tokens" if _uses_max_completion_tokens(endpoint.model) else "max_tokens"
        payload[key] = max_tokens
    return payload


def _messages(conversation: List[Dict[str, Any]], provider: str, model: str) -> List[Dict[str, Any]]:
    messages = []
    for message in conversation:
        if message["role"] == "tool":
            messages.append({
                "role": "tool", "tool_call_id": message["call_id"], "content": tool_output_text(message),
            })
        elif "blocks" in message:
            messages.append(_assistant_message(message["blocks"], provider, model))
        else:
            messages.append({"role": message["role"], "content": message.get("content") or ""})
    return messages


def _assistant_message(blocks: List[Dict[str, Any]], provider: str, model: str) -> Dict[str, Any]:
    text = "".join(block["text"] for block in blocks if block["type"] == "text")
    tool_calls = [
        {"id": block["call_id"], "type": "function",
         "function": {"name": block["name"], "arguments": block["arguments"]}}
        for block in blocks if block["type"] == "tool_call"
    ]
    message: Dict[str, Any] = {"role": "assistant", "content": text or (None if tool_calls else "")}
    if tool_calls:
        message["tool_calls"] = tool_calls
    for block in blocks:
        if block["type"] == "reasoning" and replays_reasoning(block, provider, model):
            message.update({field: block[field] for field in _REASONING_FIELDS if field in block})
    return message


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
    conversation: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    tools: Sequence[Dict[str, Any]] = (),
) -> AsyncIterator[RoundEvent]:
    payload = build_payload(endpoint, conversation, temperature=temperature,
                            max_tokens=max_tokens, tools=tools)
    target_url = _normalize_openai_chat_url(endpoint.url)
    client = _get_http_client()
    async with client.stream(
        "POST",
        target_url,
        json=payload,
        headers=_request_headers(endpoint, payload["messages"]),
        timeout=_stream_timeout(LLMConfig.STREAM_TIMEOUT),
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            yield ProviderError(
                response.status_code,
                _format_upstream_error(response.status_code, body, endpoint.url),
            )
            return
        text = ""
        details: List[Any] = []
        reasoning_content = ""
        # Tool calls stream as fragments keyed by index; complete at stream end.
        calls: Dict[int, Dict[str, str]] = {}
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
                # OpenRouter: "the complete reasoning sequence is built by
                # concatenating all chunks in order".
                details.extend(delta.get("reasoning_details") or ())
                reasoning_content += delta.get("reasoning_content") or ""
                if delta.get("content"):
                    text += delta["content"]
                    yield TextDelta(delta["content"])
                for fragment in delta.get("tool_calls") or ():
                    _merge_tool_call(calls, fragment)
        blocks = _blocks(endpoint, text, details, reasoning_content)
        blocks += [
            {"type": "tool_call", "call_id": call["call_id"] or f"call_{uuid.uuid4().hex}",
             "name": call["name"], "arguments": call["arguments"]}
            for _, call in sorted(calls.items())
        ]
        yield RoundEnd(blocks, None)


def _merge_tool_call(calls: Dict[int, Dict[str, str]], fragment: Dict[str, Any]) -> None:
    call = calls.setdefault(int(fragment.get("index") or 0), {"call_id": "", "name": "", "arguments": ""})
    if fragment.get("id"):
        call["call_id"] = fragment["id"]
    function = fragment.get("function") or {}
    call["name"] += function.get("name") or ""
    call["arguments"] += function.get("arguments") or ""


def _blocks(endpoint: Endpoint, text: str, details: List[Any], reasoning_content: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    reasoning = {
        field: value
        for field, value in (("reasoning_details", details), ("reasoning_content", reasoning_content))
        if value
    }
    if reasoning:
        blocks.append({
            "type": "reasoning", "provider": provider_key(endpoint.url), "model": endpoint.model, **reasoning,
        })
    if text:
        blocks.append({"type": "text", "text": text})
    return blocks
