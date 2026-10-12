"""Pins for the legacy helpers the chat_agent runtime imports (R2).

Each test states the one assumption the runtime relies on, so an upstream
change to a private helper fails here instead of misrouting a turn.
Session.add_message stamping ``_db_id`` is pinned by
tests/test_session_manager_persist_guard.py.
"""

import pytest

from src.agent_loop import _agent_route_tool_mode
from src.chat_agent.providers import Endpoint
from src.chat_agent.providers.chat_completions import build_payload
from src.endpoint_resolver import normalize_base
from src.llm_core import (
    _detect_provider,
    _format_upstream_error,
    _host_match,
    _normalize_openai_chat_url,
    _provider_headers,
)


@pytest.mark.parametrize(
    ("url", "xai"),
    [
        ("https://api.x.ai/v1", True),
        ("https://x.ai", True),
        ("https://max.ai/v1", False),
        ("https://proxy.example/x.ai/v1", False),
    ],
)
def test_host_match_compares_hostnames_not_substrings(url, xai):
    assert _host_match(url, "x.ai") is xai


@pytest.mark.parametrize(
    "url",
    ["https://api.x.ai/v1", "https://api.x.ai/v1/", "https://api.x.ai/v1/chat/completions",
     "https://api.x.ai/v1/responses"],
)
def test_normalize_base_strips_dialect_suffixes_to_the_api_base(url):
    assert normalize_base(url) == "https://api.x.ai/v1"


@pytest.mark.parametrize(
    ("url", "provider"),
    [
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://integrate.api.nvidia.com/v1", "nvidia"),
        ("http://192.168.1.20:1234/v1", "openai"),
        ("https://api.x.ai/v1", "openai"),
        ("https://api.anthropic.com/v1", "anthropic"),
        ("http://localhost:11434", "ollama"),
        ("https://chatgpt.com/backend-api/codex", "chatgpt-subscription"),
    ],
)
def test_detect_provider_maps_endpoint_url_to_dialect(url, provider):
    assert _detect_provider(url) == provider


@pytest.mark.parametrize(
    ("url", "model", "native"),
    [
        ("https://openrouter.ai/api/v1", "x-ai/grok-4.1-fast", True),
        ("http://192.168.1.20:1234/v1", "qwen3-30b-a3b", True),
        ("http://localhost:11434/v1", "qwen3:8b", False),
        ("https://llm.example/v1", "gpt-oss-120b", False),
    ],
)
def test_agent_route_tool_mode_first_value_is_native_tool_decision(url, model, native):
    assert _agent_route_tool_mode(url, model, "alice", None)[0] is native


def test_openai_chat_url_appends_chat_completions_once():
    assert _normalize_openai_chat_url("https://openrouter.ai/api/v1/") == (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    assert _normalize_openai_chat_url("http://h/v1/chat/completions") == "http://h/v1/chat/completions"


def test_provider_headers_keep_session_auth_and_add_json_content_type():
    headers = _provider_headers("openrouter", {"Authorization": "Bearer k"})
    assert headers["Authorization"] == "Bearer k"
    assert headers["Content-Type"] == "application/json"


def test_upstream_error_is_a_readable_sentence():
    message = _format_upstream_error(429, b'{"error": {"message": "slow down"}}', "https://openrouter.ai/api/v1")
    assert "429" in message and "slow down" in message


def test_payload_follows_per_model_temperature_and_token_rules():
    plain = build_payload(Endpoint("https://openrouter.ai/api/v1", "x-ai/grok-4.1-fast", None),
                          [], temperature=0.7, max_tokens=0)
    assert plain["temperature"] == 0.7
    assert "max_tokens" not in plain and "max_completion_tokens" not in plain

    reasoning = build_payload(Endpoint("https://api.openai.com/v1", "gpt-5", None),
                              [], temperature=0.7, max_tokens=500)
    assert "temperature" not in reasoning
    assert reasoning["max_completion_tokens"] == 500


@pytest.mark.asyncio
async def test_maybe_compact_without_persist_returns_system_summary_then_kept_half(monkeypatch):
    import src.context_compactor as context_compactor

    async def summarize(*args, **kwargs):
        return "summary text"

    monkeypatch.setattr(context_compactor, "get_context_length", lambda url, model: 1_000)
    monkeypatch.setattr(context_compactor, "resolve_endpoint", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(context_compactor, "llm_call_async", summarize)
    monkeypatch.setattr(
        context_compactor, "_update_session_history",
        lambda *a, **k: pytest.fail("persist=False must not write session history"),
    )
    system = {"role": "system", "content": "sys"}
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i} " + "w " * 400}
             for i in range(6)]
    state = {}

    result, _, compacted = await context_compactor.maybe_compact(
        object(), "http://h/v1", "m", [system, *turns], persist=False, compaction_state=state,
    )

    assert compacted is True
    assert result[0] is system
    assert result[1]["role"] == "system" and "summary text" in result[1]["content"]
    assert result[2:] == turns[state["split_point"]:]
