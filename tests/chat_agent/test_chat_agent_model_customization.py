"""Per-model generation settings (upstream #7061) on the new runtime.

Decided for devConflictSolving 2026-11-18: temperature resolves as chat
override > model setting > built-in default on every candidate; sampling
fields reach Chat Completions payloads only; the per-model thinking choice
does not map onto the chat's thinking level.
"""

import pytest

from src import model_customization
from src.chat_agent.providers import Endpoint, responses
from tests.chat_agent.test_chat_agent_generation_settings import _stream
from tests.chat_agent.test_chat_agent_route import OPENROUTER, _Request, _sse_tool_calls, _switch_on
from tests.chat_agent.test_chat_agent_route import harness  # noqa: F401


def _model_settings(monkeypatch, settings):
    monkeypatch.setattr(model_customization, "lookup", lambda url, model: dict(settings))


@pytest.mark.asyncio
async def test_model_temperature_applies_when_the_chat_sets_none(harness, monkeypatch):
    _switch_on()
    _model_settings(monkeypatch, {"temperature": 0.3})
    harness.provider_rounds = [_sse_tool_calls(("no_such_tool", {})), _stream([{"content": "Done."}])]

    await harness.send(_Request("agent"))

    assert [p["temperature"] for p in harness.captured["provider_requests"]] == [0.3, 0.3]


@pytest.mark.asyncio
async def test_chat_temperature_outranks_the_model_setting(harness, monkeypatch):
    _switch_on()
    _model_settings(monkeypatch, {"temperature": 0.3})
    harness.session.temperature_override = 0.6
    harness.provider_rounds = [_stream([{"content": "Done."}])]

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert payload["temperature"] == 0.6


@pytest.mark.asyncio
async def test_model_sampling_reaches_the_chat_completions_payload(harness, monkeypatch):
    _switch_on()
    _model_settings(monkeypatch, {"top_p": 0.8, "presence_penalty": 0.5, "top_k": 40})
    harness.provider_rounds = [_stream([{"content": "Done."}])]

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert payload["top_p"] == 0.8 and payload["presence_penalty"] == 0.5
    # OpenRouter is hosted-openai: top_k is not a field that format carries.
    assert "top_k" not in payload
    assert harness.session.endpoint_url == OPENROUTER


def test_responses_payload_carries_no_model_sampling(monkeypatch):
    _model_settings(monkeypatch, {"top_p": 0.8, "presence_penalty": 0.5})
    payload = responses.build_payload(
        Endpoint("https://api.x.ai/v1", "grok-4.1-fast", None), [], cache_key="s", temperature=None, max_tokens=0,
    )
    assert "top_p" not in payload and "presence_penalty" not in payload
