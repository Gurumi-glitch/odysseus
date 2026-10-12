"""Chat Completions adapter passes reasoning back unchanged (ticket #51).

Streams follow OpenRouter's documented reasoning_details deltas and the
reasoning_content field of DeepSeek/LM Studio style servers.
"""

import json

import httpx
import pytest

from src.chat_agent.providers import Endpoint, RoundEnd, chat_completions

OPENROUTER = Endpoint("https://openrouter.ai/api/v1", "anthropic/claude-opus-5.5", None)
DETAILS = [
    {"type": "reasoning.text", "text": "The user wants ", "format": "anthropic-claude-v1", "index": 0},
    {"type": "reasoning.text", "text": "a greeting.", "signature": "sig-abc", "format": "anthropic-claude-v1", "index": 0},
    {"type": "reasoning.encrypted", "data": "ENC==", "id": "rd_1", "format": "anthropic-claude-v1", "index": 1},
]


def _stream(deltas):
    lines = ["data: " + json.dumps({"choices": [{"delta": delta}]}) for delta in deltas]
    return ("\n\n".join(lines + ["data: [DONE]"]) + "\n\n").encode()


class _Provider:
    def __init__(self, monkeypatch, body):
        self.payloads = []
        client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        self._body = body
        monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)

    def _handle(self, request):
        self.payloads.append(json.loads(request.content))
        return httpx.Response(200, content=self._body, headers={"content-type": "text/event-stream"})


async def _run(conversation, endpoint=OPENROUTER):
    return [
        event
        async for event in chat_completions.stream_round(
            endpoint, conversation, temperature=0.2, max_tokens=0,
        )
    ]


@pytest.mark.asyncio
async def test_streamed_reasoning_details_are_kept_in_order_and_replayed_unchanged(monkeypatch):
    provider = _Provider(monkeypatch, _stream([
        {"reasoning_details": DETAILS[:2]},
        {"reasoning_details": DETAILS[2:]},
        {"content": "Hi!"},
    ]))
    question = [{"role": "user", "content": "hello"}]

    [end] = [e for e in await _run(question) if isinstance(e, RoundEnd)]
    blocks = json.loads(json.dumps(end.blocks))
    await _run([*question, {"role": "assistant", "blocks": blocks}, {"role": "user", "content": "again"}])

    assert blocks[0] == {
        "type": "reasoning", "provider": "openrouter", "model": "anthropic/claude-opus-5.5",
        "reasoning_details": DETAILS,
    }
    assert provider.payloads[1]["messages"][1] == {
        "role": "assistant", "content": "Hi!", "reasoning_details": DETAILS,
    }


@pytest.mark.asyncio
async def test_reasoning_content_is_replayed_unchanged(monkeypatch):
    provider = _Provider(monkeypatch, _stream([
        {"reasoning_content": "think "}, {"reasoning_content": "hard"}, {"content": "Done."},
    ]))
    local = Endpoint("http://192.168.1.20:1234/v1", "qwen3-30b-a3b", None)
    question = [{"role": "user", "content": "hello"}]

    [end] = [e for e in await _run(question, local) if isinstance(e, RoundEnd)]
    await _run([*question, {"role": "assistant", "blocks": end.blocks}], local)

    assert provider.payloads[1]["messages"][1] == {
        "role": "assistant", "content": "Done.", "reasoning_content": "think hard",
    }


@pytest.mark.asyncio
async def test_another_models_reasoning_is_not_replayed(monkeypatch):
    provider = _Provider(monkeypatch, _stream([{"content": "ok"}]))
    blocks = [
        {"type": "reasoning", "provider": "openrouter", "model": "x-ai/grok-4.1-fast", "reasoning_details": DETAILS},
        {"type": "text", "text": "Hi!"},
    ]

    await _run([{"role": "user", "content": "hello"}, {"role": "assistant", "blocks": blocks}])

    assert provider.payloads[0]["messages"][1] == {"role": "assistant", "content": "Hi!"}


@pytest.mark.asyncio
async def test_tool_rounds_serialize_as_tool_calls_and_tool_messages(monkeypatch):
    provider = _Provider(monkeypatch, _stream([{"content": "ok"}]))
    conversation = [
        {"role": "user", "content": "fetch it"},
        {"role": "assistant", "blocks": [
            {"type": "tool_call", "provider": "xai", "call_id": "call_1", "name": "web_fetch",
             "arguments": "{\"url\":\"https://x.test\"}", "item": {"type": "function_call"}},
        ]},
        {"role": "tool", "call_id": "call_1", "ok": False, "content": "HTTP 404", "error": "failed"},
    ]

    await _run(conversation)

    _, assistant, tool = provider.payloads[0]["messages"]
    assert assistant == {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://x.test\"}"}}]}
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_1"
    assert json.loads(tool["content"]) == {"error": "HTTP 404", "status": "failed"}
