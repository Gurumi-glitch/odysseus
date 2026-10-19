"""What an (endpoint, model) can do, as its server reports it (spec #89).

Native tool support and thinking levels, answered by one probe and cached in
process; sources, verdicts and cache lifetimes: src/chat_agent/README.md.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from urllib.parse import quote, urlparse

import httpx

from src import model_capabilities as mc
# Relies on: _probe_lmstudio_models(url) returns LM Studio's native model list
# (dicts with "key"), or None when the host isn't LM Studio or is unreachable;
# it caches per (host, port) in _lmstudio_models_cache with a 60s TTL.
# _is_local_host is True only for loopback/LAN/Tailscale hosts.
from src.chat_helpers import _is_local_host, _lmstudio_models_cache, _probe_lmstudio_models
# Relies on: _serving_base strips the "/v1..." API path off an endpoint URL,
# leaving the server root llama.cpp and Ollama serve native routes from;
# _origin is (scheme, host, port) with the scheme's default port filled in.
from src.agent_runtime.context_resolution import _origin, _serving_base
from src.chat_agent.providers import THINKING_OFF, chat_completions, wire_effort
from src.chat_agent.store import THINKING_AUTO, THINKING_GRADED
from src.endpoint_resolver import normalize_base
# Relies on: _native_tool_transport_rejected(text) matches the legacy markers
# of a provider refusing tool calls; plain (non-SSE) text is matched as is.
from src.llm_core import _detect_provider, _native_tool_transport_rejected, _provider_headers
from src.model_capability_readers import llamacpp, lmstudio, ollama, openrouter
from src.model_capability_readers.base import VENDOR_OLLAMA, VENDOR_SGLANG, VENDOR_VLLM, detect_vendor

logger = logging.getLogger(__name__)

YES = "yes"
NO = "no"
UNKNOWN = "unknown"
SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
# The levels a server can be asked about; Auto sends nothing and is always selectable.
LEVELS = (THINKING_OFF,) + THINKING_GRADED
# What a binary model's "on" is sent as: LM Studio /v1 answers "on" with a
# 400 and treats "medium" exactly like its default "on" (#80 LM Studio probe).
BINARY_ON = "medium"

NATIVE_TOOLS_REJECTED = "This model doesn't support native tools; the next message uses legacy."
LEVEL_REFUSED = "{model} doesn't support thinking level {level}; it answered at its default level."

PROBE_TIMEOUT_SECONDS = 1.0
SERVER_REPORTED_TTL_SECONDS = 60.0
# Ollama's wording; the legacy matcher predates Ollama's tool check.
_OLLAMA_TOOLS_REJECTED = "does not support tools"
# Local servers on these ports have no capability endpoint (vLLM, SGLang).
_SILENT_LOCAL_VENDORS = frozenset({VENDOR_VLLM, VENDOR_SGLANG})
# The effort parameter by name ("Unexpected reasoning effort" from llama.cpp
# templates, LM Studio's "reasoning_effort" 400) or OpenRouter's refusal to turn
# mandatory reasoning off (#80); bare "reasoning" also names models and fields.
_EFFORT_REJECTED = re.compile(r"effort|reasoning is mandatory", re.IGNORECASE)
# The key a learned "no native tools" is kept under, beside learned levels.
_TOOLS = "tools"


@dataclass(frozen=True)
class ThinkingLevels:
    """Per level in LEVELS: SUPPORTED / UNSUPPORTED / UNKNOWN."""

    support: Dict[str, str] = field(default_factory=lambda: dict.fromkeys(LEVELS, UNKNOWN))
    binary: bool = False

    def to_send(self, level: str) -> Optional[str]:
        """The level the request carries; None sends no thinking parameter."""
        if level == THINKING_AUTO or self.support.get(level, UNSUPPORTED) == UNSUPPORTED:
            return None
        if self.binary and level in THINKING_GRADED:
            return BINARY_ON
        return level


@dataclass(frozen=True)
class _Report:
    tools: str = UNKNOWN
    thinking: ThinkingLevels = field(default_factory=ThinkingLevels)


_Key = Tuple[Tuple[str, str, Optional[int]], str]
# (origin, model) -> (report, expiry): what the server said, for 60s.
_reported: Dict[_Key, Tuple[_Report, float]] = {}
# (origin, model) -> {"tools" or a level: NO / UNSUPPORTED}: learned from a
# rejection, outranks the server, kept until refresh.
_learned: Dict[_Key, Dict[str, str]] = {}
_now = time.monotonic


async def native_tools(endpoint_url: str, model: str, headers: Any = None) -> str:
    """YES / NO / UNKNOWN for this (endpoint, model); never raises."""
    key = (_origin(endpoint_url), model)
    if _TOOLS in _learned.get(key, {}):
        return _learned[key][_TOOLS]
    verdict = (await _report(key, endpoint_url, model, headers)).tools
    logger.info("chat_agent: %s reports native tools %s", model, verdict)
    return verdict


async def thinking_levels(endpoint_url: str, model: str, headers: Any = None) -> ThinkingLevels:
    """The thinking levels this (endpoint, model) takes; never raises."""
    key = (_origin(endpoint_url), model)
    reported = (await _report(key, endpoint_url, model, headers)).thinking
    learned = _learned.get(key, {})
    support = {level: learned.get(level, verdict) for level, verdict in reported.support.items()}
    return ThinkingLevels(support, reported.binary)


def learn_no_native_tools(endpoint_url: str, model: str) -> None:
    _learned.setdefault((_origin(endpoint_url), model), {})[_TOOLS] = NO


def learn_unsupported_level(endpoint_url: str, model: str, level: str) -> None:
    _learned.setdefault((_origin(endpoint_url), model), {})[level] = UNSUPPORTED


def forget_endpoint(endpoint_url: str) -> None:
    """Drop everything known about this endpoint's server (model-list refresh)."""
    origin = _origin(endpoint_url)
    # Runs on the model-refresh thread: copy the keys, pop tolerantly.
    for cache in (_reported, _learned):
        for key in list(cache):
            if key[0] == origin:
                cache.pop(key, None)
    parsed = urlparse(endpoint_url)
    # Same key _probe_lmstudio_models stores under: hostname and explicit port.
    _lmstudio_models_cache.pop((parsed.hostname or "", parsed.port), None)


async def _report(key: _Key, endpoint_url: str, model: str, headers: Any) -> _Report:
    cached = _reported.get(key)
    if cached is not None and cached[1] > _now():
        return cached[0]
    try:
        # A hard bound: per-request timeouts don't cover a body that trickles.
        reported = await asyncio.wait_for(_probe(endpoint_url, model, headers), PROBE_TIMEOUT_SECONDS)
    except (httpx.HTTPError, ValueError, asyncio.TimeoutError) as exc:
        logger.info("chat_agent: capability probe for %s failed (%s)", model, type(exc).__name__)
        return _Report()
    expiry = _now() + SERVER_REPORTED_TTL_SECONDS
    for listed_model, report in reported.items():
        _reported[(key[0], listed_model)] = (report, expiry)
    return reported.get(model, _Report())


def is_tools_rejection(status: Optional[int], message: str) -> bool:
    if status != 400:
        return False
    return _native_tool_transport_rejected(message) or _OLLAMA_TOOLS_REJECTED in (message or "").lower()


def is_effort_rejection(status: Optional[int], message: str) -> bool:
    return status is not None and 400 <= status < 600 and bool(_EFFORT_REJECTED.search(message or ""))


async def _probe(endpoint_url: str, model: str, headers: Any) -> Dict[str, _Report]:
    """What the server reports, by model; empty when it reports nothing."""
    provider = _detect_provider(endpoint_url)
    if provider in ("openrouter", "mistral", "copilot"):
        return await _probe_catalog(provider, endpoint_url, headers)
    if provider != "openai" or not _is_local_host(_origin(endpoint_url)[1]):
        return {}
    vendor = detect_vendor(endpoint_url)
    if vendor in _SILENT_LOCAL_VENDORS:
        return {}
    base = _serving_base(endpoint_url)
    if vendor == VENDOR_OLLAMA:
        response = await _client().post(
            f"{base}/api/show", json={"model": model}, timeout=PROBE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        record = ollama.record_from_show_payload(model, response.json())
        return {model: _Report(_verdict(record))}
    lmstudio_models = await asyncio.to_thread(_probe_lmstudio_models, endpoint_url)
    if lmstudio_models:
        # LM Studio reports trained_for_tool_use only inside "capabilities".
        listed = [item for item in lmstudio_models if isinstance(item.get("capabilities"), dict)]
        return _reports_by_model(
            listed, lmstudio.record_from_native_model,
            lambda item: _lmstudio_levels(item["capabilities"].get("reasoning")),
        )
    response = await _client().get(
        f"{base}/props?model={quote(model, safe='')}", timeout=PROBE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    props = response.json()
    if not isinstance(props, dict) or not isinstance(props.get("chat_template_caps"), dict):
        return {}
    # /props names the loaded file, not the id the chat uses: it answers for `model`.
    return {model: _Report(
        _verdict(llamacpp.record_from_props_payload(props)), await _llamacpp_levels(base, model),
    )}


async def _llamacpp_levels(base: str, model: str) -> ThinkingLevels:
    """Render the prompt once for Auto and once per level with POST
    /apply-template (no generation; ~1ms each on the PriSM fork, #80). A
    failed probe reports nothing, and the tools verdict from /props stands."""
    try:
        auto = await _render(base, model, None)
        if auto is None:
            return ThinkingLevels()
        renders = {level: await _render(base, model, level) for level in LEVELS}
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("chat_agent: thinking level probe for %s failed (%s)", model, type(exc).__name__)
        return ThinkingLevels()
    graded = [renders[level] for level in THINKING_GRADED if renders[level] is not None]
    # A level equal to Auto may just be the template default (Bonsai's xhigh):
    # only when every graded level renders like Auto does effort do nothing.
    graded_effect = any(prompt != auto for prompt in graded)
    support = {level: UNSUPPORTED if renders[level] is None or not graded_effect else SUPPORTED
               for level in THINKING_GRADED}
    support[THINKING_OFF] = SUPPORTED if renders[THINKING_OFF] not in (None, auto) else UNSUPPORTED
    return ThinkingLevels(support)


async def _render(base: str, model: str, level: Optional[str]) -> Optional[str]:
    """The rendered prompt, or None when the template refuses the level (HTTP 500)."""
    body: Dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    if level is not None:
        body["reasoning_effort"] = wire_effort(level)
    response = await _client().post(f"{base}/apply-template", json=body, timeout=PROBE_TIMEOUT_SECONDS)
    if response.status_code == 500:
        return None
    response.raise_for_status()
    payload = response.json()
    prompt = payload.get("prompt") if isinstance(payload, dict) else None
    if not isinstance(prompt, str):
        raise ValueError("apply-template answered without a prompt")
    return prompt


async def _probe_catalog(provider: str, endpoint_url: str, headers: Any) -> Dict[str, _Report]:
    response = await _client().get(
        f"{normalize_base(endpoint_url)}/models",
        headers=_provider_headers(provider, headers if isinstance(headers, dict) else None),
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return {}
    items = [item for item in payload["data"] if isinstance(item, dict) and item.get("id")]
    if provider == "openrouter":
        return _reports_by_model(
            [item for item in items if isinstance(item.get("supported_parameters"), list)],
            openrouter.record_from_model,
            lambda item: _openrouter_levels(item.get("reasoning")),
        )
    reports = {}
    for item in items:
        flags = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
        if provider == "copilot":
            # Same field copilot.fetch_models reads for the picker.
            flags = flags.get("supports") if isinstance(flags.get("supports"), dict) else {}
            flag = flags.get("tool_calls")
        else:
            flag = flags.get("function_calling")
        if isinstance(flag, bool):
            reports[item["id"]] = _Report(YES if flag else NO)
    return reports


def _reports_by_model(
    items: Iterable[Dict[str, Any]], read: Callable[[Dict[str, Any]], Any],
    levels: Callable[[Dict[str, Any]], ThinkingLevels],
) -> Dict[str, _Report]:
    reports = {}
    for item in items:
        record = read(item)
        if record is not None:
            reports[record.model_id] = _Report(_verdict(record), levels(item))
    return reports


def _openrouter_levels(reasoning: Any) -> ThinkingLevels:
    # No reasoning object reports nothing: routers that take reasoning
    # parameters omit it too (catalog of 2026-10-13).
    if not isinstance(reasoning, dict):
        return ThinkingLevels()
    efforts = reasoning.get("supported_efforts")
    support = {
        level: SUPPORTED if efforts is None or level in efforts else UNSUPPORTED
        for level in THINKING_GRADED
    }
    # "mandatory: true -> effort none is rejected"; Off is not in the effort list.
    support[THINKING_OFF] = UNSUPPORTED if reasoning.get("mandatory") else SUPPORTED
    return ThinkingLevels(support)


def _lmstudio_levels(reasoning: Any) -> ThinkingLevels:
    options = reasoning.get("allowed_options") if isinstance(reasoning, dict) else None
    if not isinstance(options, list):
        return ThinkingLevels()
    # An off/on model: every graded level means "on" (#80 amendment).
    binary = "on" in options
    support = {
        level: SUPPORTED if binary or level in options else UNSUPPORTED for level in THINKING_GRADED
    }
    support[THINKING_OFF] = SUPPORTED if THINKING_OFF in options else UNSUPPORTED
    return ThinkingLevels(support, binary)


def _verdict(record: Any) -> str:
    if record is None:
        return UNKNOWN
    if mc.CAP_TOOL_CALL in record.capability.capabilities:
        return YES
    # A reader that recognised the payload reports provider confidence; one
    # that didn't returns an unknown capability, which says nothing about tools.
    if record.capability.confidence == mc.CONFIDENCE_PROVIDER_REPORTED:
        return NO
    return UNKNOWN


def _client() -> httpx.AsyncClient:
    # The provider stream's shared client: probes and turns share one
    # transport (TLS overrides, and tests' mock transport).
    return chat_completions._get_http_client()
