"""What an (endpoint, model) can do, as its server reports it (spec #89).

Answers native tool support: YES, NO or UNKNOWN. UNKNOWN means the server
reports nothing (or the probe failed); callers treat it as "try native tools".
Values come from a lazy in-process cache: server-reported ones live 60s,
values learned from a rejection live until the endpoint's model-list refresh
or a restart. No model-name lists, no user override, no DB.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Iterable, Optional, Tuple
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
from src.chat_agent.providers import chat_completions
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

NATIVE_TOOLS_REJECTED = "This model doesn't support native tools; the next message uses legacy."

PROBE_TIMEOUT_SECONDS = 1.0
SERVER_REPORTED_TTL_SECONDS = 60.0
# Ollama's wording; the legacy matcher predates Ollama's tool check.
_OLLAMA_TOOLS_REJECTED = "does not support tools"
# Local servers on these ports have no capability endpoint (vLLM, SGLang).
_SILENT_LOCAL_VENDORS = frozenset({VENDOR_VLLM, VENDOR_SGLANG})

_Key = Tuple[Tuple[str, str, Optional[int]], str]
# (origin, model) -> (verdict, expiry); expiry None = learned, kept until refresh.
_verdicts: Dict[_Key, Tuple[str, Optional[float]]] = {}
_now = time.monotonic


async def native_tools(endpoint_url: str, model: str, headers: Any = None) -> str:
    """YES / NO / UNKNOWN for this (endpoint, model); never raises."""
    key = (_origin(endpoint_url), model)
    cached = _verdicts.get(key)
    if cached is not None and (cached[1] is None or cached[1] > _now()):
        return cached[0]
    try:
        # A hard bound: per-request timeouts don't cover a body that trickles.
        reported = await asyncio.wait_for(_probe(endpoint_url, model, headers), PROBE_TIMEOUT_SECONDS)
    except (httpx.HTTPError, ValueError, asyncio.TimeoutError) as exc:
        logger.info("chat_agent: capability probe for %s failed (%s)", model, type(exc).__name__)
        return UNKNOWN
    if not reported:
        return UNKNOWN
    expiry = _now() + SERVER_REPORTED_TTL_SECONDS
    for listed_model, verdict in reported.items():
        listed = _verdicts.get((key[0], listed_model))
        if listed is None or listed[1] is not None:  # a learned "no" outranks the server
            _verdicts[(key[0], listed_model)] = (verdict, expiry)
    verdict = reported.get(model, UNKNOWN)
    logger.info("chat_agent: %s reports native tools %s", model, verdict)
    return verdict


def learn_no_native_tools(endpoint_url: str, model: str) -> None:
    _verdicts[(_origin(endpoint_url), model)] = (NO, None)


def forget_endpoint(endpoint_url: str) -> None:
    """Drop everything known about this endpoint's server (model-list refresh)."""
    origin = _origin(endpoint_url)
    # Runs on the model-refresh thread: copy the keys, pop tolerantly.
    for key in list(_verdicts):
        if key[0] == origin:
            _verdicts.pop(key, None)
    parsed = urlparse(endpoint_url)
    # Same key _probe_lmstudio_models stores under: hostname and explicit port.
    _lmstudio_models_cache.pop((parsed.hostname or "", parsed.port), None)


def is_tools_rejection(status: Optional[int], message: str) -> bool:
    if status != 400:
        return False
    return _native_tool_transport_rejected(message) or _OLLAMA_TOOLS_REJECTED in (message or "").lower()


async def _probe(endpoint_url: str, model: str, headers: Any) -> Dict[str, str]:
    """Verdicts the server reports, by model; empty when it reports nothing."""
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
        return {model: _verdict(record)}
    lmstudio_models = await asyncio.to_thread(_probe_lmstudio_models, endpoint_url)
    if lmstudio_models:
        # LM Studio reports trained_for_tool_use only inside "capabilities".
        return _verdicts_by_model(
            lmstudio.record_from_native_model(item) for item in lmstudio_models
            if isinstance(item.get("capabilities"), dict)
        )
    response = await _client().get(
        f"{base}/props?model={quote(model, safe='')}", timeout=PROBE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    props = response.json()
    if not isinstance(props, dict) or not isinstance(props.get("chat_template_caps"), dict):
        return {}
    # /props names the loaded file, not the id the chat uses: it answers for `model`.
    return {model: _verdict(llamacpp.record_from_props_payload(props))}


async def _probe_catalog(provider: str, endpoint_url: str, headers: Any) -> Dict[str, str]:
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
        return _verdicts_by_model(
            openrouter.record_from_model(item) for item in items
            if isinstance(item.get("supported_parameters"), list)
        )
    verdicts = {}
    for item in items:
        flags = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
        if provider == "copilot":
            # Same field copilot.fetch_models reads for the picker.
            flags = flags.get("supports") if isinstance(flags.get("supports"), dict) else {}
            flag = flags.get("tool_calls")
        else:
            flag = flags.get("function_calling")
        if isinstance(flag, bool):
            verdicts[item["id"]] = YES if flag else NO
    return verdicts


def _verdicts_by_model(records: Iterable[Any]) -> Dict[str, str]:
    return {record.model_id: _verdict(record) for record in records if record is not None}


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
