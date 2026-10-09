"""A dropped tool preamble must not reach the stream or the saved reply (fork #14)."""
import json

import src.agent_loop as al
from routes.chat_routes import _AgentRenderState
from tests.test_agent_rounds_exhausted import _collect, _patch_common, _types


PREAMBLE = "I'll look up cats on the web and share what comes back."
ANSWER = (
    "搜尋結果主要指向兩篇關於貓咪的文章。\n\n"
    "Cats.com 則介紹了七種常見的貓叫聲與其含義：https://cats.com/cat-vocalizations"
)


def _run_preamble_then_answer(monkeypatch):
    _patch_common(monkeypatch)
    rounds = 0

    async def _fake_stream(_candidates, messages, **kwargs):
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            yield "data: " + json.dumps({"delta": PREAMBLE}) + "\n\n"
            yield "data: " + json.dumps({
                "type": "tool_calls",
                "calls": [{
                    "id": "search",
                    "name": "web_search",
                    "arguments": json.dumps({"query": "cat"}),
                }],
            }) + "\n\n"
        else:
            yield "data: " + json.dumps({"delta": ANSWER}) + "\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    return _types(_collect(al.stream_agent_loop(
        "http://x/v1",
        "grok-test",
        [{"role": "user", "content": "let us try web search about cat"}],
        max_rounds=4,
        relevant_tools={"web_search"},
    )))


def test_dropped_tool_preamble_is_not_streamed_or_saved(monkeypatch):
    events = _run_preamble_then_answer(monkeypatch)

    assert any(event.get("type") == "tool_start" for event in events), events
    assert not any(PREAMBLE in json.dumps(event, ensure_ascii=False) for event in events
                   if event.get("type") != "metrics"), events

    render = _AgentRenderState()
    for event in events:
        render.consume(event)
    assert render.content.strip() == ANSWER

    metrics = next(event["data"] for event in events if event.get("type") == "metrics")
    assert [text for text in metrics["round_texts"] if text.strip()] == [ANSWER]


def test_gate_retracts_a_preamble_split_across_deltas_and_keeps_reasoning():
    from src.agent_runtime.completion import with_completion_gate

    def frame(data):
        return "data: " + json.dumps(data) + "\n\n"

    @with_completion_gate
    async def stream(messages):
        yield frame({"delta": "Earlier answer. I'll look up "})
        yield frame({"delta": "Weighing sources.", "thinking": True})
        yield frame({"delta": "cats now."})
        yield frame({"type": "retract_answer", "content": "I'll look up cats now."})
        yield frame({"delta": "Cats purr."})
        yield "data: [DONE]\n\n"

    events = _types(_collect(stream([])))
    answer = "".join(e["delta"] for e in events if "delta" in e and not e.get("thinking"))
    assert answer == "Earlier answer. Cats purr."
    assert any(e.get("thinking") and e["delta"] == "Weighing sources." for e in events)
    assert not any(e.get("type") == "retract_answer" for e in events)
