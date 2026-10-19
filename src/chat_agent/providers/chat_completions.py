"""OpenAI Chat Completions dialect (OpenRouter, NVIDIA, LM Studio, other compatible).

Transport, headers and endpoint resolution come from the legacy provider layer.
The legacy message sanitizer is deliberately not used: it drops
``reasoning_details`` and rewrites history (spec #44, Providers).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from src.chat_agent.providers import (
    OFF_EFFORT,
    Endpoint,
    ProviderError,
    RoundEnd,
    ReasoningDelta,
    RoundEvent,
    TextDelta,
    Usage,
    provider_key,
    replays_reasoning,
    tool_output_text,
    wire_effort,
)

# Each helper below is module-level in llm_core and relied on as:
#   _detect_provider: endpoint URL -> provider key ("openrouter", "openai", ...);
#   _normalize_openai_chat_url: base URL -> its /chat/completions URL;
#   _provider_headers: session headers + provider-required headers;
#   _omit_temperature / _uses_max_completion_tokens: per-model payload rules;
#   _get_http_client: the shared AsyncClient (connection reuse, TLS overrides);
#   _stream_timeout: streaming httpx.Timeout from a read timeout in seconds;
#   _format_upstream_error: HTTP status + body -> user-readable sentence;
#   _host_match: the URL's hostname is the domain or a subdomain of it.
from src.llm_core import (
    LLMConfig,
    _detect_provider,
    _format_upstream_error,
    _get_http_client,
    _host_match,
    _normalize_openai_chat_url,
    _omit_temperature,
    _provider_headers,
    _stream_timeout,
    _uses_max_completion_tokens,
)

# The fields reasoning comes back on and must go back on, unchanged: OpenRouter
# reasoning_details (in order) and the reasoning_content of DeepSeek-style servers.
_REASONING_FIELDS = ("reasoning_details", "reasoning_content")
# Legacy stream_llm sends stream_options to every Chat Completions provider
# except these two (llm_core, "Improve OpenRouter and Groq provider requests");
# OpenRouter reports usage in its last chunk without it.
_NO_STREAM_OPTIONS = frozenset({"openrouter", "groq"})


def build_payload(
    endpoint: Endpoint,
    conversation: List[Dict[str, Any]],
    *,
    temperature: Optional[float],
    max_tokens: int,
    tools: Sequence[Dict[str, Any]] = (),
    thinking: Optional[str] = None,
) -> Dict[str, Any]:
    """``thinking``: a level to send (off, low ... max); None sends none."""
    provider = _detect_provider(endpoint.url)
    messages = _messages(conversation, provider_key(endpoint.url), endpoint.model)
    payload: Dict[str, Any] = {"model": endpoint.model, "messages": messages, "stream": True}
    if provider not in _NO_STREAM_OPTIONS:
        # Compaction triggers on the usage the provider reports (#67).
        payload["stream_options"] = {"include_usage": True}
    if tools:
        # parallel_tool_calls is never sent: the model may still issue several
        # calls per round, and the loop runs them one at a time (spec #44).
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    # Unset sends nothing, so the server's own sampling default applies (spec #89).
    if temperature is not None and not _omit_temperature(provider, endpoint.model):
        payload["temperature"] = temperature
    if max_tokens > 0:
        key = "max_completion_tokens" if _uses_max_completion_tokens(endpoint.model) else "max_tokens"
        payload[key] = max_tokens
    if thinking is not None:
        effort = wire_effort(thinking)
        if provider == "openrouter":
            payload["reasoning"] = {"effort": effort}
        elif effort == OFF_EFFORT and _host_match(endpoint.url, "deepseek.com"):
            # DeepSeek turns thinking off on its own axis, not with an effort.
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["reasoning_effort"] = effort
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
    temperature: Optional[float],
    max_tokens: int,
    tools: Sequence[Dict[str, Any]] = (),
    thinking: Optional[str] = None,
) -> AsyncIterator[RoundEvent]:
    payload = build_payload(endpoint, conversation, temperature=temperature,
                            max_tokens=max_tokens, tools=tools, thinking=thinking)
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
        details: List[Any] = []
        reasoning_content = ""
        streamed = _RoundText()
        # Tool calls stream as fragments keyed by index; complete at stream end.
        calls: Dict[int, Dict[str, str]] = {}
        usage = None
        output_limit = False
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
            if isinstance(chunk.get("usage"), dict):
                usage = _usage(chunk["usage"])
            for choice in chunk.get("choices") or ():
                delta = choice.get("delta") or {}
                output_limit = output_limit or choice.get("finish_reason") == "length"
                # OpenRouter: "the complete reasoning sequence is built by
                # concatenating all chunks in order".
                details.extend(delta.get("reasoning_details") or ())
                reasoning_content += delta.get("reasoning_content") or ""
                # vLLM >= 0.20 and OpenRouter send readable reasoning as
                # "reasoning"; read one field so a server sending both isn't doubled.
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                for event in streamed.feed_thinking(reasoning or ""):
                    yield event
                if delta.get("content"):
                    for event in streamed.feed(delta["content"]):
                        yield event
                for fragment in delta.get("tool_calls") or ():
                    _merge_tool_call(calls, fragment)
        for event in streamed.flush():
            yield event
        blocks = _blocks(endpoint, streamed.text, details, reasoning_content, streamed.thinking)
        blocks += [
            {"type": "tool_call", "call_id": call["call_id"] or f"call_{uuid.uuid4().hex}",
             "name": call["name"], "arguments": call["arguments"]}
            for _, call in sorted(calls.items())
        ]
        yield RoundEnd(blocks, usage, output_limit)


# Inline thinking (spec #89): only these tags, only when they open a round's
# content; <mm:think>, <thought> and a stray </think> stay answer text (#84 cut list).
_INLINE_TAGS = {"<think>": "</think>", "<thinking>": "</thinking>"}


class _RoundText:
    """A round's answer and readable thinking. Content is split into the two,
    holding back only what may still be part of a tag split across chunks."""

    def __init__(self) -> None:
        self.text = ""
        self.thinking = ""
        self._held = ""
        self._opening = True
        self._close: Optional[str] = None

    def feed(self, chunk: str) -> List[RoundEvent]:
        if not self._opening and self._close is None:
            return self._answer(chunk)
        self._held += chunk
        if self._opening:
            head = self._held.lstrip()
            tag = next((tag for tag in _INLINE_TAGS if head.startswith(tag)), None)
            if tag is None:
                if any(tag.startswith(head) for tag in _INLINE_TAGS):
                    return []
                self._opening = False
                return self._answer(self._take(len(self._held)))
            self._opening = False
            self._close = _INLINE_TAGS[tag]
            self._held = head[len(tag):]
        index = self._held.find(self._close)
        if index >= 0:
            thought = self._take(index)
            rest = self._held[len(self._close):]
            self._held, self._close = "", None
            return self.feed_thinking(thought) + self._answer(rest)
        return self.feed_thinking(self._take(len(self._held) - _close_prefix(self._held, self._close)))

    def flush(self) -> List[RoundEvent]:
        held = self._take(len(self._held))
        return self.feed_thinking(held) if self._close else self._answer(held)

    def _take(self, size: int) -> str:
        taken, self._held = self._held[:size], self._held[size:]
        return taken

    def _answer(self, text: str) -> List[RoundEvent]:
        self.text += text
        return [TextDelta(text)] if text else []

    def feed_thinking(self, text: str) -> List[RoundEvent]:
        self.thinking += text
        return [ReasoningDelta(text)] if text else []


def _close_prefix(held: str, close: str) -> int:
    """Length of the longest end of ``held`` that could begin ``close``."""
    return next((size for size in range(min(len(held), len(close) - 1), 0, -1) if close.startswith(held[-size:])), 0)


def _usage(raw: Dict[str, Any]) -> Usage:
    # prompt_tokens counts cached prompt tokens too; the details give the cached share.
    return Usage(
        input_tokens=int(raw.get("prompt_tokens") or 0),
        cached_input_tokens=int((raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        reasoning_tokens=int((raw.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
    )


def _merge_tool_call(calls: Dict[int, Dict[str, str]], fragment: Dict[str, Any]) -> None:
    call = calls.setdefault(int(fragment.get("index") or 0), {"call_id": "", "name": "", "arguments": ""})
    if fragment.get("id"):
        call["call_id"] = fragment["id"]
    function = fragment.get("function") or {}
    call["name"] += function.get("name") or ""
    call["arguments"] += function.get("arguments") or ""


def _blocks(
    endpoint: Endpoint, text: str, details: List[Any], reasoning_content: str, thinking: str,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    reasoning = {
        field: value
        for field, value in (
            ("reasoning_details", details), ("reasoning_content", reasoning_content), ("text", thinking),
        )
        if value
    }
    if reasoning:
        blocks.append({
            "type": "reasoning", "provider": provider_key(endpoint.url), "model": endpoint.model, **reasoning,
        })
    if text:
        blocks.append({"type": "text", "text": text})
    return blocks
