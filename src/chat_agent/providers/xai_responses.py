"""xAI Responses dialect: stateless ``/v1/responses`` with replayed reasoning (ADR 0004).

Every request is ``store: false`` and carries ``prompt_cache_key`` = chat id.
Each output item is kept on its block and replayed unchanged, so the input
only ever grows by appending and xAI's exact-prefix cache keeps hitting.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from src.chat_agent.providers import (
    XAI,
    Endpoint,
    ProviderError,
    RoundEnd,
    RoundEvent,
    TextDelta,
    Usage,
    replays_reasoning,
    tool_output_text,
    wire_effort,
)
# Relied on: normalize_base strips a trailing /chat/completions, /responses,
# /models (and "/"), leaving the configured API base such as .../v1.
from src.endpoint_resolver import normalize_base
# Same llm_core helpers and assumptions as providers/chat_completions.py.
from src.llm_core import (
    LLMConfig,
    _detect_provider,
    _format_upstream_error,
    _get_http_client,
    _omit_temperature,
    _provider_headers,
    _stream_timeout,
)

logger = logging.getLogger(__name__)

# Encrypted reasoning is only returned when asked for (grok-4.7 always sends it).
_INCLUDE = ["reasoning.encrypted_content"]
# The status xAI answers an input it cannot accept with, replayed reasoning included.
_REJECTED_INPUT = 400
_FAILURE_EVENTS = frozenset({"error", "response.failed", "response.incomplete"})


def build_payload(
    endpoint: Endpoint,
    conversation: List[Dict[str, Any]],
    *,
    cache_key: str,
    temperature: Optional[float],
    max_tokens: int,
    tools: Sequence[Dict[str, Any]] = (),
    thinking: Optional[str] = None,
) -> Dict[str, Any]:
    """``thinking``: a level to send (off, low ... max); None sends none."""
    payload: Dict[str, Any] = {
        "model": endpoint.model,
        "input": _input_items(conversation, endpoint.model),
        "stream": True,
        "store": False,
        "include": _INCLUDE,
        "prompt_cache_key": cache_key,
    }
    if temperature is not None and not _omit_temperature(_detect_provider(endpoint.url), endpoint.model):
        payload["temperature"] = temperature
    if max_tokens > 0:
        payload["max_output_tokens"] = max_tokens
    if tools:
        payload["tools"] = [_function_tool(tool) for tool in tools]
        payload["tool_choice"] = "auto"
    if thinking is not None:
        payload["reasoning"] = {"effort": wire_effort(thinking)}
    return payload


def _input_items(conversation: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for message in conversation:
        if message["role"] == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": message["call_id"],
                "output": tool_output_text(message),
            })
        elif "blocks" in message:
            items.extend(_replayed_blocks(message["blocks"], model))
        else:
            items.append({"role": message["role"], "content": _content(message.get("content"))})
    return items


def _replayed_blocks(blocks: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    items = []
    for block in blocks:
        if block["type"] == "reasoning":
            if replays_reasoning(block, XAI, model) and _valid_reasoning(block.get("item")):
                items.append(block["item"])
        elif block.get("provider") == XAI and isinstance(block.get("item"), dict):
            items.append(block["item"])
        elif block["type"] == "text":
            items.append({"role": "assistant", "content": block["text"]})
        elif block["type"] == "tool_call":
            items.append({
                "type": "function_call",
                "call_id": block["call_id"],
                "name": block["name"],
                "arguments": block["arguments"],
            })
    return items


def _valid_reasoning(item: Any) -> bool:
    """Shape check: with store:false a reasoning item is useless without its ciphertext."""
    return (
        isinstance(item, dict)
        and item.get("type") == "reasoning"
        and isinstance(item.get("encrypted_content"), str)
        and bool(item["encrypted_content"])
    )


def _content(content: Any) -> Any:
    if not isinstance(content, list):
        return content or ""
    return [_input_part(part) for part in content]


def _input_part(part: Dict[str, Any]) -> Dict[str, Any]:
    if part.get("type") == "text":
        return {"type": "input_text", "text": part.get("text") or ""}
    if part.get("type") == "image_url":
        return {"type": "input_image", "image_url": (part.get("image_url") or {}).get("url")}
    return part


def _function_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Responses function tools are flat; the executor's schemas are Chat Completions shaped."""
    return {"type": "function", **tool["function"]}


async def stream_round(
    endpoint: Endpoint,
    conversation: List[Dict[str, Any]],
    *,
    cache_key: str,
    temperature: Optional[float],
    max_tokens: int,
    tools: Sequence[Dict[str, Any]] = (),
    thinking: Optional[str] = None,
) -> AsyncIterator[RoundEvent]:
    payload = build_payload(
        endpoint, conversation,
        cache_key=cache_key, temperature=temperature, max_tokens=max_tokens, tools=tools,
        thinking=thinking,
    )
    events = _attempt(endpoint, payload)
    first = await anext(events, None)
    if _rejected(first) and _has_reasoning(payload):
        # Spec #44 Providers: drop the encrypted reasoning, keep the text, retry once.
        await events.aclose()
        logger.warning("xAI rejected replayed reasoning for %s; retrying without it: %s",
                       endpoint.model, first.message)
        payload = {**payload, "input": [i for i in payload["input"] if i.get("type") != "reasoning"]}
        events = _attempt(endpoint, payload)
        first = await anext(events, None)
    if first is not None:
        yield first
    async for event in events:
        yield event


def _rejected(event: Optional[RoundEvent]) -> bool:
    return isinstance(event, ProviderError) and event.status == _REJECTED_INPUT


def _has_reasoning(payload: Dict[str, Any]) -> bool:
    return any(item.get("type") == "reasoning" for item in payload["input"])


async def _attempt(endpoint: Endpoint, payload: Dict[str, Any]) -> AsyncIterator[RoundEvent]:
    client = _get_http_client()
    async with client.stream(
        "POST",
        normalize_base(endpoint.url) + "/responses",
        json=payload,
        headers=_provider_headers(_detect_provider(endpoint.url), endpoint.headers),
        timeout=_stream_timeout(LLMConfig.STREAM_TIMEOUT),
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            yield ProviderError(
                response.status_code,
                _format_upstream_error(response.status_code, body, endpoint.url),
            )
            return
        blocks: List[Dict[str, Any]] = []
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue  # "[DONE]" and keep-alives
            kind = event.get("type")
            if kind == "response.output_text.delta":
                yield TextDelta(event.get("delta") or "")
            elif kind == "response.output_item.done":
                block = _block(event.get("item") or {}, endpoint.model)
                if block:
                    blocks.append(block)
            elif kind == "response.completed":
                yield RoundEnd(blocks, _usage((event.get("response") or {}).get("usage")))
                return
            elif kind in _FAILURE_EVENTS:
                yield ProviderError(None, _failure_message(event))
                return
        yield ProviderError(None, "xAI ended the stream before the response completed.")


def _block(item: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    kind = item.get("type")
    if kind == "reasoning":
        return {"type": "reasoning", "provider": XAI, "model": model, "item": item}
    if kind == "message":
        text = "".join(
            part.get("text") or "" for part in item.get("content") or () if part.get("type") == "output_text"
        )
        return {"type": "text", "provider": XAI, "text": text, "item": item}
    if kind == "function_call":
        return {
            "type": "tool_call",
            "provider": XAI,
            "call_id": item.get("call_id"),
            "name": item.get("name"),
            "arguments": item.get("arguments") or "",
            "item": item,
        }
    # Server-side tools are never requested (spec #44 Out of scope).
    logger.warning("xAI returned an unexpected output item type %r; not kept", kind)
    return None


def _usage(raw: Any) -> Optional[Usage]:
    if not isinstance(raw, dict):
        return None
    return Usage(
        input_tokens=int(raw.get("input_tokens") or 0),
        cached_input_tokens=int((raw.get("input_tokens_details") or {}).get("cached_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        reasoning_tokens=int((raw.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
    )


def _failure_message(event: Dict[str, Any]) -> str:
    response = event.get("response") or {}
    error = response.get("error") or {}
    reason = (response.get("incomplete_details") or {}).get("reason")
    detail = event.get("message") or error.get("message") or reason or "unknown error"
    return f"xAI stopped the response: {detail}"
