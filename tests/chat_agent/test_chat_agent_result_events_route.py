"""Tool results drive the chat UI panels on the new runtime (ticket #53).

Driven through the chat route with a scripted provider; the executor's inner
worker returns handler-shaped results so the real wrapper and adapter run.
"""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.tool_execution as tool_execution
import src.tool_security as tool_security
from core import database
from src.agent_loop import _compute_final_metrics, _usage_bucket, _usage_bucket_summary
from src.tool_utils import _truncate
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _sse_tool_calls,
    _switch_on,
)

SOURCES = [{"title": "Example", "url": "https://example.com", "snippet": "s"}]


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'odysseus.db'}")
    database.Base.metadata.create_all(engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine))


@pytest.fixture
def harness(monkeypatch, tmp_path, db):
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)
    harness = _Harness(monkeypatch, tmp_path)
    _switch_on()
    return harness


@pytest.fixture
def results(monkeypatch):
    """Results the executor's worker returns, by tool; progress payloads it reports."""
    by_tool, progress = {}, []

    async def worker(block, *args, progress_cb=None, **kwargs):
        for payload in progress:
            await progress_cb(payload)
        return f"{block.tool_type}: ran", dict(by_tool[block.tool_type])

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", worker)
    return by_tool, progress


def _of(events, kind):
    return [e for e in events if e.get("type") == kind]


def _tool_messages(payload):
    return [m for m in payload["messages"] if m["role"] == "tool"]


@pytest.mark.asyncio
async def test_web_search_shows_sources_and_the_model_never_sees_the_marker(harness, results):
    results[0]["web_search"] = {
        "output": "Results here.\n\n<!-- SOURCES:" + json.dumps(SOURCES) + " -->", "exit_code": 0,
    }
    harness.provider_rounds = [
        _sse_tool_calls(("web_search", {"query": "odysseus"})),
        _sse_events(["Found it."]),
    ]

    _, body = await harness.send(_Request("agent", allow_web_search="true"))

    events = _data_events(body)
    assert _of(events, "web_sources") == [{"type": "web_sources", "data": SOURCES}]
    [output] = _of(events, "tool_output")
    assert output["output"] == "Results here."
    _, second = harness.captured["provider_requests"]
    [tool_message] = _tool_messages(second)
    assert "SOURCES" not in tool_message["content"]
    assert "Results here." in tool_message["content"]
    [row] = harness.saved
    assert row.metadata["web_sources"] == SOURCES


@pytest.mark.asyncio
async def test_create_document_streams_into_the_editor(harness, results):
    content = "# Plan\n\nShip it."
    results[0]["create_document"] = {
        "action": "create", "doc_id": "doc-1", "title": "plan.md", "language": "markdown",
        "content": content, "version": 1, "exit_code": 0,
    }
    harness.provider_rounds = [
        _sse_tool_calls(("create_document", {"title": "plan.md", "language": "markdown", "content": content})),
        _sse_events(["Created."]),
    ]

    _, body = await harness.send(_Request("agent"))

    kinds = [e.get("type") for e in _data_events(body)
             if e.get("type") in {"doc_stream_open", "doc_stream_delta", "doc_update", "tool_output"}]
    assert kinds == ["doc_stream_open", "doc_stream_delta", "doc_update", "tool_output", "doc_update"]
    [output] = _of(_data_events(body), "tool_output")
    assert (output["command"], output["output"], output["doc_id"]) == ("plan.md", 'Document created: "plan.md" (v1)', "doc-1")
    [row] = harness.saved
    [event] = row.metadata["tool_events"]
    assert (event["doc_id"], event["doc_title"], event["status"]) == ("doc-1", "plan.md", "done")


@pytest.mark.asyncio
async def test_update_plan_opens_the_plan_window_and_generate_image_shows_the_image(harness, results):
    results[0]["update_plan"] = {
        "plan_update": {"plan": "- [x] one"}, "output": "Plan updated (1/1 steps complete).", "exit_code": 0,
    }
    results[0]["generate_image"] = {
        "results": "Generated image for: a lighthouse", "image_url": "/api/generated-image/a.png",
        "image_id": "img-1", "image_prompt": "a lighthouse",
    }
    harness.provider_rounds = [
        _sse_tool_calls(("update_plan", {"plan": "- [x] one"}), ("generate_image", {"prompt": "a lighthouse"})),
        _sse_events(["Done."]),
    ]

    _, body = await harness.send(_Request("agent"))

    events = _data_events(body)
    assert _of(events, "plan_update") == [{"type": "plan_update", "data": {"plan": "- [x] one"}}]
    [image] = _of(events, "generated_image")
    assert (image["url"], image["image_id"]) == ("/api/generated-image/a.png", "img-1")
    [row] = harness.saved
    assert row.metadata["tool_events"][1]["image_url"] == "/api/generated-image/a.png"


@pytest.mark.asyncio
async def test_long_tool_progress_streams_before_its_output(harness, results):
    results[0]["web_search"] = {"output": "done", "exit_code": 0}
    results[1].extend([{"elapsed_s": 2, "tail": "searching"}, {"elapsed_s": 4, "tail": "fetching"}])
    harness.provider_rounds = [_sse_tool_calls(("web_search", {"query": "q"})), _sse_events(["Ok."])]

    _, body = await harness.send(_Request("agent", allow_web_search="true"))

    tool_events = [e for e in _data_events(body) if e.get("type", "").startswith("tool_")]
    assert [e["type"] for e in tool_events] == ["tool_start", "tool_progress", "tool_progress", "tool_output"]
    assert tool_events[1] == {"type": "tool_progress", "tool": "web_search", "round": 1,
                              "elapsed_s": 2, "tail": "searching"}


@pytest.mark.asyncio
async def test_each_round_starts_with_agent_step_and_metrics_carry_cumulative_usage(harness, results):
    results[0]["update_plan"] = {"plan_update": {"plan": "- [x] a"}, "output": "ok", "exit_code": 0}
    harness.provider_rounds = [_sse_tool_calls(("update_plan", {"plan": "- [x] a"})), _sse_events(["Done."])]
    after = []
    harness.monkeypatch.setattr(
        "routes.chat_routes.run_post_response_tasks", lambda *a, **k: after.append(k),
    )

    _, body = await harness.send(_Request("agent"))

    events = _data_events(body)
    assert _of(events, "agent_step") == [{"type": "agent_step", "round": 1}, {"type": "agent_step", "round": 2}]
    [metrics] = _of(events, "metrics")
    data = metrics["data"]
    assert data["model"] == "x-ai/grok-4.1-fast"
    assert data["context_length"] == 4096
    assert [b["round"] for b in data["usage_buckets"]] == [1, 2]
    assert data["input_tokens"] == sum(b["input_tokens"] for b in data["usage_buckets"])
    assert data["round_texts"] == ["", "Done."]
    assert not {"completion_decision", "evidence_events", "completion_requirements"} & set(data)
    kinds = [e.get("type") for e in events]
    assert kinds.index("metrics") < kinds.index("message_saved")
    assert after and after[0]["last_metrics"]["input_tokens"] == data["input_tokens"]


@pytest.mark.asyncio
async def test_a_provider_failure_still_reports_metrics(harness):
    import httpx
    from src.chat_agent.providers import chat_completions
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(400, text="bad")))
    harness.monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)

    _, body = await harness.send(_Request("agent"))

    assert _of(_data_events(body), "metrics")
    assert "event: error" in body


# Pins for the legacy metrics helpers the loop imports (R2).

def test_compute_final_metrics_uses_real_usage_and_estimates_cost_from_inputs_only():
    metrics = _compute_final_metrics(
        [{"role": "user", "content": "hi"}], "hello", 2.0, 0.5, 1000, 100, 20, True, [], ["hello"],
        model="m", round_models=["m"], round_endpoint_ids=[None], round_endpoint_labels=["Selected route"],
        last_round_input_tokens=100,
    )
    assert (metrics["input_tokens"], metrics["output_tokens"], metrics["usage_source"]) == (100, 20, "real")
    assert metrics["context_percent"] == 10.0
    assert metrics["round_texts"] == ["hello"]


def test_usage_buckets_sum_and_keep_per_round_attribution():
    buckets = [
        _usage_bucket(round_num=n, model="m", endpoint_id="ep", endpoint_label="L", endpoint_cost_tracked=True,
                      input_tokens=10 * n, output_tokens=n, usage_source="real")
        for n in (1, 2)
    ]
    summary = _usage_bucket_summary(buckets)
    assert (summary["input_tokens"], summary["output_tokens"], summary["usage_source"]) == (30, 3, "real")
    assert summary["usage_buckets"][1]["endpoint_cost_tracked"] is True


def test_truncate_always_returns_text():
    assert _truncate(None) == ""
    assert _truncate("abc", limit=2).startswith("ab\n... (truncated")
