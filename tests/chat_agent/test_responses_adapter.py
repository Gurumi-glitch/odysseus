"""Responses adapter against scripted provider streams (ticket #51, ADR 0004).

The xai_* fixtures under tests/fixtures/chat_agent/ are OpenAI-Responses-shaped
streams written from xAI's docs; a live capture replaces them in #69. The
chatgpt_* fixtures are trimmed from a live Codex backend recording.
"""

import json
from pathlib import Path

import httpx
import pytest

from src.chat_agent.providers import Endpoint, ProviderError, RoundEnd, TextDelta, Usage, responses

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "chat_agent"
XAI = Endpoint("https://api.x.ai/v1", "grok-4.7", {"Authorization": "Bearer key"})


def _fixture(name):
    return (FIXTURES / name).read_bytes()


def _output_items(raw):
    """The output items exactly as the provider sent them, from the done events."""
    items = []
    for line in raw.decode().splitlines():
        if line.startswith("data: {"):
            event = json.loads(line[6:])
            if event["type"] == "response.output_item.done":
                items.append(event["item"])
    return items


class _Provider:
    def __init__(self, monkeypatch, *replies):
        self.requests = []
        self._responses = list(replies)
        client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        monkeypatch.setattr(responses, "_get_http_client", lambda: client)

    def _handle(self, request):
        self.requests.append(request)
        status, body = self._responses.pop(0)
        return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})

    def payload(self, index):
        return json.loads(self.requests[index].content)


async def _run(conversation, endpoint=XAI):
    return [
        event
        async for event in responses.stream_round(
            endpoint, conversation, cache_key="session-1", temperature=0.2, max_tokens=0,
        )
    ]


CONVERSATION = [
    {"role": "system", "content": "You are Odysseus."},
    {"role": "user", "content": "add note for me: dinner time with mom"},
]


@pytest.mark.asyncio
async def test_round_two_replays_round_one_output_items_byte_identical(monkeypatch):
    round1 = _fixture("xai_responses_round1_tool_call.sse")
    provider = _Provider(monkeypatch, (200, round1), (200, _fixture("xai_responses_round2_text.sse")))

    [end] = [e for e in await _run(CONVERSATION) if isinstance(e, RoundEnd)]
    # The blocks survive a JSON store round trip (agent_v2 lives in a JSON column).
    blocks = json.loads(json.dumps(end.blocks))
    second = [
        *CONVERSATION,
        {"role": "assistant", "blocks": blocks},
        {"role": "tool", "call_id": "call_61514723", "ok": True, "content": "Note created.", "error": None},
    ]
    await _run(second)

    # httpx encodes compactly; same keys, order and values as received, byte for byte.
    replayed = ",".join(
        json.dumps(item, separators=(",", ":"), ensure_ascii=False) for item in _output_items(round1)
    ).encode()
    assert replayed in provider.requests[1].content
    assert provider.payload(1)["input"][-1] == {
        "type": "function_call_output", "call_id": "call_61514723", "output": "Note created.",
    }


@pytest.mark.asyncio
async def test_request_is_stateless_cache_keyed_and_asks_for_encrypted_reasoning(monkeypatch):
    provider = _Provider(monkeypatch, (200, _fixture("xai_responses_round2_text.sse")))

    await _run(CONVERSATION)

    [request] = provider.requests
    assert str(request.url) == "https://api.x.ai/v1/responses"
    assert request.headers["Authorization"] == "Bearer key"
    payload = provider.payload(0)
    assert payload["model"] == "grok-4.7"
    assert payload["store"] is False
    assert payload["stream"] is True
    assert payload["prompt_cache_key"] == "session-1"
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in payload
    assert "tools" not in payload and "parallel_tool_calls" not in payload
    assert payload["input"] == [
        {"role": "system", "content": "You are Odysseus."},
        {"role": "user", "content": "add note for me: dinner time with mom"},
    ]


@pytest.mark.asyncio
async def test_stream_yields_text_live_and_blocks_tagged_with_provider_and_model(monkeypatch):
    raw = _fixture("xai_responses_round2_text.sse")
    _Provider(monkeypatch, (200, raw))

    events = await _run(CONVERSATION)

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Added the note ", '"Dinner".']
    [end] = [e for e in events if isinstance(e, RoundEnd)]
    reasoning, text = end.blocks
    assert reasoning == {
        "type": "reasoning", "provider": "xai", "model": "grok-4.7", "item": _output_items(raw)[0],
    }
    assert text["type"] == "text" and text["text"] == 'Added the note "Dinner".'
    assert end.usage == Usage(input_tokens=1990, cached_input_tokens=1792, output_tokens=40, reasoning_tokens=22)


@pytest.mark.asyncio
async def test_function_call_becomes_a_tool_call_block(monkeypatch):
    _Provider(monkeypatch, (200, _fixture("xai_responses_round1_tool_call.sse")))

    [end] = [e for e in await _run(CONVERSATION) if isinstance(e, RoundEnd)]

    call = end.blocks[1]
    assert (call["type"], call["call_id"], call["name"]) == ("tool_call", "call_61514723", "manage_notes")
    assert json.loads(call["arguments"]) == {"action": "create", "title": "Dinner", "content": "dinner time with mom"}


def _history_with_reasoning(model):
    return [
        *CONVERSATION,
        {"role": "assistant", "blocks": [
            {"type": "reasoning", "provider": "xai", "model": model,
             "item": {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "ENC", "status": "completed"}},
            {"type": "text", "provider": "xai", "text": "Done.", "item": {
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "Done.", "logprobs": [], "annotations": []}]}},
        ]},
        {"role": "user", "content": "thanks"},
    ]


@pytest.mark.asyncio
async def test_other_models_reasoning_is_dropped_and_its_text_kept(monkeypatch):
    provider = _Provider(monkeypatch, (200, _fixture("xai_responses_round2_text.sse")))

    await _run(_history_with_reasoning("grok-4.5"))

    types = [item.get("type") for item in provider.payload(0)["input"]]
    assert "reasoning" not in types
    assert "message" in types


@pytest.mark.asyncio
async def test_rejected_reasoning_is_dropped_and_retried_once(monkeypatch):
    rejection = json.dumps({"error": {"message": "invalid encrypted_content"}}).encode()
    provider = _Provider(
        monkeypatch, (400, rejection), (200, _fixture("xai_responses_round2_text.sse")),
    )

    events = await _run(_history_with_reasoning("grok-4.7"))

    assert not [e for e in events if isinstance(e, ProviderError)]
    first, retry = (provider.payload(0)["input"], provider.payload(1)["input"])
    assert "reasoning" in [item.get("type") for item in first]
    assert retry == [item for item in first if item.get("type") != "reasoning"]


@pytest.mark.asyncio
async def test_second_rejection_is_reported(monkeypatch):
    rejection = json.dumps({"error": {"message": "bad input"}}).encode()
    provider = _Provider(monkeypatch, (400, rejection), (400, rejection))

    events = await _run(_history_with_reasoning("grok-4.7"))

    assert len(provider.requests) == 2
    [error] = events
    assert isinstance(error, ProviderError) and error.status == 400 and "bad input" in error.message


@pytest.mark.asyncio
async def test_rejection_without_replayed_reasoning_is_not_retried(monkeypatch):
    rejection = json.dumps({"error": {"message": "bad input"}}).encode()
    provider = _Provider(monkeypatch, (400, rejection))

    [error] = await _run(CONVERSATION)

    assert len(provider.requests) == 1
    assert isinstance(error, ProviderError)


@pytest.mark.asyncio
async def test_reasoning_that_fails_the_shape_check_is_not_replayed(monkeypatch):
    provider = _Provider(monkeypatch, (200, _fixture("xai_responses_round2_text.sse")))
    conversation = _history_with_reasoning("grok-4.7")
    del conversation[2]["blocks"][0]["item"]["encrypted_content"]

    await _run(conversation)

    types = [item.get("type") for item in provider.payload(0)["input"]]
    assert "reasoning" not in types and "message" in types


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "denied", "cancelled"])
async def test_failed_tool_result_is_sent_as_json_error_output(monkeypatch, status):
    provider = _Provider(monkeypatch, (200, _fixture("xai_responses_round2_text.sse")))
    conversation = [
        *CONVERSATION,
        {"role": "assistant", "blocks": [
            {"type": "tool_call", "call_id": "call_1", "name": "web_fetch", "arguments": "{}"},
        ]},
        {"role": "tool", "call_id": "call_1", "ok": False, "content": "HTTP 404", "error": status},
    ]

    await _run(conversation)

    *_, call, output = provider.payload(0)["input"]
    assert call == {"type": "function_call", "call_id": "call_1", "name": "web_fetch", "arguments": "{}"}
    assert output["type"] == "function_call_output" and output["call_id"] == "call_1"
    assert json.loads(output["output"]) == {"error": "HTTP 404", "status": status}


@pytest.mark.asyncio
async def test_failed_response_event_is_a_provider_error(monkeypatch):
    failed = (
        "data: " + json.dumps({"type": "response.failed", "response": {
            "status": "failed", "error": {"code": "server_error", "message": "upstream exploded"}}})
        + "\n\ndata: [DONE]\n\n"
    ).encode()
    _Provider(monkeypatch, (200, failed))

    [error] = await _run(CONVERSATION)

    assert isinstance(error, ProviderError) and "upstream exploded" in error.message


@pytest.mark.asyncio
async def test_stream_that_ends_before_completion_is_a_provider_error(monkeypatch):
    raw = _fixture("xai_responses_round2_text.sse")
    truncated = raw[: raw.index(b'data: {"type": "response.completed"')]
    _Provider(monkeypatch, (200, truncated))

    events = await _run(CONVERSATION)

    assert isinstance(events[-1], ProviderError)
    assert not [e for e in events if isinstance(e, RoundEnd)]


@pytest.mark.asyncio
async def test_image_parts_use_responses_input_types(monkeypatch):
    provider = _Provider(monkeypatch, (200, _fixture("xai_responses_round2_text.sse")))
    conversation = [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]

    await _run(conversation)

    assert provider.payload(0)["input"] == [{"role": "user", "content": [
        {"type": "input_text", "text": "what is this"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]}]


def test_chat_completions_tool_schemas_are_sent_flat_with_auto_choice():
    schema = {"type": "function", "function": {
        "name": "manage_notes", "description": "Notes", "parameters": {"type": "object"}}}

    payload = responses.build_payload(
        XAI, CONVERSATION, cache_key="s", temperature=0.2, max_tokens=0, tools=[schema],
    )

    assert payload["tools"] == [{
        "type": "function", "name": "manage_notes", "description": "Notes", "parameters": {"type": "object"},
        "strict": False}]
    assert payload["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in payload


# --- ChatGPT Subscription (Codex backend) ------------------------------------

CHATGPT = Endpoint("https://chatgpt.com/backend-api/codex", "gpt-6.1-sol", {"Authorization": "Bearer token"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    ["https://chatgpt.com/backend-api/codex", "https://chatgpt.com/backend-api/codex/",
     "https://chatgpt.com/backend-api/codex/responses"],
)
async def test_chatgpt_request_goes_to_codex_responses_without_temperature(monkeypatch, url):
    provider = _Provider(monkeypatch, (200, _fixture("chatgpt_responses_round2_text.sse")))

    # _run passes temperature=0.2; the Codex backend answers any temperature with 400.
    await _run(CONVERSATION, Endpoint(url, CHATGPT.model, CHATGPT.headers))

    [request] = provider.requests
    assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
    assert request.headers["Authorization"] == "Bearer token"
    payload = provider.payload(0)
    assert "temperature" not in payload
    assert payload["store"] is False and payload["stream"] is True
    assert payload["prompt_cache_key"] == "session-1"
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["input"][0] == {"role": "system", "content": "You are Odysseus."}


def test_chatgpt_payload_sends_max_output_tokens_when_set_and_never_temperature():
    payload = responses.build_payload(CHATGPT, CONVERSATION, cache_key="s", temperature=1.0, max_tokens=256)

    assert payload["max_output_tokens"] == 256
    assert "temperature" not in payload
    assert "max_output_tokens" not in responses.build_payload(
        CHATGPT, CONVERSATION, cache_key="s", temperature=None, max_tokens=0,
    )


def test_chatgpt_tools_opt_out_of_strict_so_optional_fields_stay_optional():
    # Strict mode made gpt-6.1 send manage_calendar every field, local_start={"date": ""}
    # and dtstart="" included, and the tool refused all of them.
    schema = {"type": "function", "function": {
        "name": "manage_calendar", "description": "Calendar",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string"}, "dtstart": {"type": "string"}}, "required": ["action"]}}}

    [tool] = responses.build_payload(
        CHATGPT, CONVERSATION, cache_key="s", temperature=None, max_tokens=0, tools=[schema],
    )["tools"]

    assert tool["strict"] is False
    assert tool["parameters"]["required"] == ["action"]


@pytest.mark.asyncio
async def test_chatgpt_blocks_are_tagged_with_its_provider(monkeypatch):
    raw = _fixture("chatgpt_responses_round2_text.sse")
    _Provider(monkeypatch, (200, _fixture("chatgpt_responses_round1_tool_call.sse")), (200, raw))

    [call_round] = [e for e in await _run(CONVERSATION, CHATGPT) if isinstance(e, RoundEnd)]
    events = await _run(CONVERSATION, CHATGPT)

    [call] = call_round.blocks
    assert (call["type"], call["provider"], call["call_id"], call["name"]) == (
        "tool_call", "chatgpt-subscription", "call_r1_0", "get_weather")
    assert json.loads(call["arguments"]) == {"city": "Taipei"}
    assert call_round.usage == Usage(input_tokens=73, cached_input_tokens=0, output_tokens=19, reasoning_tokens=0)
    [end] = [e for e in events if isinstance(e, RoundEnd)]
    [text] = end.blocks
    answer = "The current weather in Taipei is cloudy with a temperature of 24°C."
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == answer
    assert (text["type"], text["provider"], text["text"]) == ("text", "chatgpt-subscription", answer)
    assert text["item"] == _output_items(raw)[0]


@pytest.mark.asyncio
async def test_chatgpt_parallel_calls_become_one_tool_call_block_each(monkeypatch):
    _Provider(monkeypatch, (200, _fixture("chatgpt_responses_parallel_calls.sse")))

    [end] = [e for e in await _run(CONVERSATION, CHATGPT) if isinstance(e, RoundEnd)]

    assert [(b["type"], b["call_id"]) for b in end.blocks] == [
        ("tool_call", "call_r4_0"), ("tool_call", "call_r4_1"), ("tool_call", "call_r4_2"),
    ]


@pytest.mark.asyncio
async def test_chatgpt_http_error_names_chatgpt_subscription(monkeypatch):
    rejection = json.dumps({"error": {"message": "Unsupported parameter: temperature"}}).encode()
    _Provider(monkeypatch, (400, rejection))

    [error] = await _run(CONVERSATION, CHATGPT)

    assert isinstance(error, ProviderError) and error.status == 400
    assert error.message == "ChatGPT Subscription returned HTTP 400: Unsupported parameter: temperature"


@pytest.mark.asyncio
async def test_chatgpt_stream_failures_name_chatgpt_subscription(monkeypatch):
    raw = _fixture("chatgpt_responses_round2_text.sse")
    failed = (
        "data: " + json.dumps({"type": "response.failed", "response": {
            "status": "failed", "error": {"code": "server_error", "message": "upstream exploded"}}})
        + "\n\n"
    ).encode()
    _Provider(monkeypatch, (200, raw[: raw.index(b'data: {"type": "response.completed"')]), (200, failed))

    [ended] = [e for e in await _run(CONVERSATION, CHATGPT) if isinstance(e, ProviderError)]
    [stopped] = await _run(CONVERSATION, CHATGPT)

    assert ended.message == "ChatGPT Subscription ended the stream before the response completed."
    assert stopped.message == "ChatGPT Subscription stopped the response: upstream exploded"


@pytest.mark.asyncio
async def test_chatgpt_rejected_reasoning_is_dropped_and_retried_once(monkeypatch):
    reasoning_round = _fixture("chatgpt_responses_reasoning.sse")
    rejection = json.dumps({"error": {"message": "invalid encrypted_content"}}).encode()
    provider = _Provider(
        monkeypatch, (200, reasoning_round), (400, rejection), (200, _fixture("chatgpt_responses_round2_text.sse")),
    )
    [end] = [e for e in await _run(CONVERSATION, CHATGPT) if isinstance(e, RoundEnd)]
    history = [*CONVERSATION, {"role": "assistant", "blocks": json.loads(json.dumps(end.blocks))},
               {"role": "user", "content": "thanks"}]

    events = await _run(history, CHATGPT)

    assert not [e for e in events if isinstance(e, ProviderError)]
    first, retry = provider.payload(1)["input"], provider.payload(2)["input"]
    assert _output_items(reasoning_round)[0] in first
    assert retry == [item for item in first if item.get("type") != "reasoning"]
