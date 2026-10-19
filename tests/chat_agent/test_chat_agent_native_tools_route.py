"""Routing by reported native tool support, through the chat route (ticket #90).

Capability probes and the provider stream are served by one mock transport;
asserts the runtime a turn takes, what the browser sees and the next turn.
"""

import json

import httpx
import pytest

import src.foreground_model_routing as foreground_model_routing
from tests.chat_agent.test_chat_agent_capabilities import (
    BONSAI,
    BONSAI_PROPS,
    LLAMA,
    _fresh_caches,  # noqa: F401  (autouse: clean capability caches)
    serve_probes,
)
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _switch_on,
)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path)
    # A hand-added llama-server: no endpoint row, so supports_tools is NULL
    # and the legacy name heuristics don't know the model.
    harness.session.endpoint_url = LLAMA
    harness.session.model = BONSAI
    harness.session.headers = {}
    return harness


def _chat(harness, answers):
    def answer(request):
        harness.captured["provider_requests"].append(json.loads(request.content))
        status, body = answers.pop(0)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": body}})
        return httpx.Response(200, content=_sse_events(body), headers={"content-type": "text/event-stream"})
    return answer


@pytest.mark.asyncio
async def test_hand_added_llama_server_reporting_tools_runs_new_runtime(harness, monkeypatch):
    _switch_on()
    serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS}, chat=_chat(harness, [(200, ["Hi"])]))

    response, body = await harness.send(_Request("agent"))

    assert response.headers["X-Odysseus-Runtime"] == "chat_agent"
    assert harness.captured["legacy_agent"] == 0
    [payload] = harness.captured["provider_requests"]
    assert payload["tools"]
    assert [e["delta"] for e in _data_events(body) if "delta" in e] == ["Hi"]


@pytest.mark.asyncio
async def test_server_reporting_no_tools_stays_legacy(harness, monkeypatch):
    _switch_on()
    props = {**BONSAI_PROPS, "chat_template_caps": {"supports_tools": False, "supports_tool_calls": False}}
    serve_probes(monkeypatch, {"GET /props": props}, chat=_chat(harness, []))

    response, _ = await harness.send(_Request("agent"))

    assert response.headers["X-Odysseus-Runtime"] == "legacy"
    assert harness.captured["legacy_agent"] == 1


@pytest.mark.asyncio
async def test_server_reporting_nothing_runs_new_runtime(harness, monkeypatch):
    """A name the legacy keyword list routes to text tools (#79: names ignored)."""
    _switch_on()
    harness.session.endpoint_url = "https://llm.example/v1"
    harness.session.model = "gpt-oss-120b"
    seen = serve_probes(monkeypatch, {}, chat=_chat(harness, [(200, ["Hi"])]))

    response, _ = await harness.send(_Request("agent"))

    assert response.headers["X-Odysseus-Runtime"] == "chat_agent"
    assert seen == ["POST /v1/chat/completions"]


@pytest.mark.asyncio
async def test_tools_rejection_ends_turn_and_next_message_uses_legacy(harness, monkeypatch):
    _switch_on()
    # A fallback is configured; a capability problem must not use it.
    monkeypatch.setattr(foreground_model_routing, "_load_policy_preferences", lambda owner=None: {
        "foreground_fallback_enabled": True,
        "foreground_model_fallbacks": [{"endpoint_id": "backup-ep", "model": "backup-model"}],
    })
    monkeypatch.setattr(
        foreground_model_routing, "resolve_fallback_entries",
        lambda *args, **kwargs: [("https://backup.example/v1", "backup-model", {})],
    )
    monkeypatch.setattr(
        foreground_model_routing, "resolve_route_descriptor_by_id",
        lambda *args, **kwargs: {"endpoint_id": "primary-ep", "endpoint_label": "Primary"},
    )
    monkeypatch.setattr(foreground_model_routing, "endpoint_cost_tracked", lambda url: False)
    seen = serve_probes(monkeypatch, {"GET /props": 404}, chat=_chat(harness, [
        (400, "registry.ollama.ai/library/bonsai does not support tools"),
    ]))

    response, body = await harness.send(_Request("agent"))

    assert response.headers["X-Odysseus-Runtime"] == "chat_agent"
    assert [p["model"] for p in harness.captured["provider_requests"]] == [BONSAI]
    assert "event: error" in body
    assert "doesn't support native tools; the next message uses legacy" in body
    assert not [e for e in _data_events(body) if e.get("type") == "fallback"]
    assert harness.saved == []

    probes_before = len(seen)
    response, _ = await harness.send(_Request("agent"))

    assert response.headers["X-Odysseus-Runtime"] == "legacy"
    assert harness.captured["legacy_agent"] == 1
    # The learned "no" answers without probing again.
    assert len(seen) == probes_before
