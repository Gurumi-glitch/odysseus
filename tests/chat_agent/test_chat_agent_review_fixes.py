"""Regressions for the spec #44 code-review findings."""
import pytest

from src.chat_agent import store
from tests.chat_agent.test_chat_agent_route import _Request, harness  # noqa: F401 (fixture)
from src.chat_agent.loop import Candidate, _TurnUsage
from src.chat_agent.providers import Endpoint, RoundEnd
from src.chat_agent.providers.chat_completions import build_payload as chat_completions_payload
from src.model_context import estimate_text_tokens


def test_estimated_output_tokens_use_the_shared_unicode_aware_estimate():
    usage = _TurnUsage(0.0, Candidate(Endpoint("http://x", "m", None), {}))
    text = "你好世界" * 10
    usage.add_round(1, RoundEnd([{"type": "text", "text": text}], None), [])
    bucket = usage.buckets[0]
    assert bucket["usage_source"] == "estimated"
    assert bucket["output_tokens"] == estimate_text_tokens(text)


def _chat_completions(url, model, mode):
    return chat_completions_payload(
        Endpoint(url, model, None), [{"role": "user", "content": "hi"}],
        temperature=0.2, max_tokens=0, thinking_mode=mode,
    )


def test_openrouter_payload_carries_the_thinking_switch_as_legacy():
    url, model = "https://openrouter.ai/api/v1", "qwen/qwen3-32b"
    assert _chat_completions(url, model, "on")["reasoning"] == {"enabled": True}
    assert _chat_completions(url, model, "off")["reasoning"] == {"effort": "none"}
    assert "reasoning" not in _chat_completions(url, model, None)


def test_local_qwen_payload_carries_enable_thinking_as_legacy():
    payload = _chat_completions("http://localhost:8000/v1", "qwen3-8b", "off")
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_the_sessions_thinking_mode_reaches_the_chat_agent_request(harness):
    store.set_chat_runtime("alice", store.RUNTIME_NEW)
    harness.session.model = "qwen/qwen3-32b"
    harness.session.thinking_mode = "on"

    await harness.send(_Request("agent"))

    assert harness.captured["provider_requests"][0]["reasoning"] == {"enabled": True}
