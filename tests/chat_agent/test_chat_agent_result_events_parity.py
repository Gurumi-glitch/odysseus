"""result_events parity (spec #44 Testing Decisions, ticket #53).

The same tool results run through the legacy loop's inline adapter and through
``result_events.adapt``; the chat UI events and the model-facing tool text must
match. Results are shaped like the real handlers' (web_tools.WebSearchTool,
ai_interaction image generation, interaction_tools update_plan, documents).
"""

import asyncio
import copy
import json

import pytest

import src.agent_loop as agent_loop
import src.tool_capabilities as tool_capabilities
import src.tool_security as tool_security
from src.chat_agent import result_events, tools
from src.tool_execution import format_tool_result
from src.tool_schemas import function_call_to_tool_block

# The events the legacy adapter emits for one tool result.
ADAPTER_EVENTS = {
    "web_sources", "doc_stream_open", "doc_stream_delta", "doc_update", "doc_suggestions",
    "ui_control", "plan_update", "tool_output", "generated_image",
}
# Fields the runtime defines itself (spec #44: exit codes, call_id) or drops (no reader).
RUNTIME_FIELDS = {"exit_code", "execution_attempted", "blocked", "call_id", "tool_call_id", "action_receipt"}

SOURCES = [{"title": "Example", "url": "https://example.com", "snippet": "Example snippet"}]
LONG_SEARCH = (
    "Query: odysseus release\n\nSEARCH RESULTS SUMMARY:\n----------\n1. Example — Example snippet\n"
    "\n==========\nFETCHED PAGE CONTENT:\n" + ("page text " * 600)
)
CONTENT = "# Plan\n\nShip the runtime."

FIXTURES = {
    "web_search": [(
        ("web_search", {"query": "odysseus release"}),
        ("web_search: odysseus release", {
            "output": LONG_SEARCH + "\n\n<!-- SOURCES:" + json.dumps(SOURCES) + " -->",
            "exit_code": 0, "evidence_status": "available",
        }),
    )],
    "create_document": [(
        ("create_document", {"title": "plan.md", "language": "markdown", "content": CONTENT}),
        ("create_document: plan.md", {
            "action": "create", "doc_id": "doc-1", "title": "plan.md", "language": "markdown",
            "content": CONTENT, "version": 1, "exit_code": 0,
        }),
    )],
    "suggest_document": [(
        ("suggest_document", {"doc_id": "doc-1", "suggestions": [{"find": "Ship", "replace": "Release"}]}),
        ("suggest_document: doc-1", {
            "action": "suggest", "doc_id": "doc-1",
            "suggestions": [{"find": "Ship", "replace": "Release"}], "exit_code": 0,
        }),
    )],
    "update_plan": [(
        ("update_plan", {"plan": "- [x] one\n- [ ] two"}),
        ("update_plan: 1/2 done", {
            "plan_update": {"plan": "- [x] one\n- [ ] two"},
            "output": "Plan updated (1/2 steps complete).", "exit_code": 0,
        }),
    )],
    "generate_image": [(
        ("generate_image", {"prompt": "a lighthouse"}),
        ("generate_image: a lighthouse", {
            "results": "Generated image for: a lighthouse",
            "image_url": "/api/generated-image/abc.png", "image_id": "img-1",
            "image_prompt": "a lighthouse", "image_model": "gpt-image-1",
            "image_size": "1024x1024", "image_quality": "medium",
        }),
    )],
    "draft_email": [(
        ("draft_email", {"to": "bob@example.com", "subject": "Hi", "body": "Hello"}),
        ("mcp__email__draft_email", {
            "output": "Draft saved. Open it: [Hi](#document-0123abcd-4567)", "exit_code": 0,
        }),
    )],
    "calendar_open_inherits_range": [
        (
            ("manage_calendar", {"action": "list_events", "start": "2026-09-01", "end": "2026-10-01"}),
            ("manage_calendar: list_events", {
                "response": "2 events", "events": [{"uid": "e1"}, {"uid": "e2"}], "exit_code": 0,
            }),
        ),
        (
            ("ui_control", {"action": "open_panel", "panel": "calendar"}),
            ("ui_control: open_panel", {
                "ui_event": "open_panel", "panel": "calendar", "output": "Opening calendar", "exit_code": 0,
            }),
        ),
    ],
    "bash": [(
        ("bash", {"command": "echo hi"}),
        ("bash: echo hi", {"stdout": "hi\n", "stderr": "", "exit_code": 0}),
    )],
}


# Legacy only offers these tools in this state.
LEGACY_LOOP_ARGS = {
    "update_plan": {"approved_plan": "- [ ] one\n- [ ] two"},
    "bash": {"workspace": "/tmp"},
}


def _legacy(monkeypatch, steps, **loop_args):
    """Run the legacy loop over one round of these calls; return its adapter
    events and the tool texts it sends back to the model."""
    results = iter([copy.deepcopy(result) for _, result in steps])
    requests = []

    async def fake_stream(_candidates, messages, **kwargs):
        requests.append(copy.deepcopy(messages))
        if len(requests) == 1:
            calls = [{"id": f"call_{i}", "name": name, "arguments": json.dumps(args)}
                     for i, ((name, args), _) in enumerate(steps)]
            yield f'data: {json.dumps({"type": "tool_calls", "calls": calls})}\n\n'
        else:
            yield f'data: {json.dumps({"delta": "Done."})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        return next(results)

    formatted = []

    def spy_format(desc, result):
        formatted.append(format_tool_result(desc, result))
        return formatted[-1]

    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)
    monkeypatch.setattr(agent_loop, "format_tool_result", spy_format)
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)
    # The adapter is under test, not legacy's approval gate.
    monkeypatch.setattr(tool_capabilities, "TOOL_APPROVAL_GATE_ENABLED", False)

    async def run():
        return [chunk async for chunk in agent_loop.stream_agent_loop(
            "https://api.openai.com/v1", "gpt-4o",
            [{"role": "user", "content": "odysseus release"}],
            relevant_tools={name for (name, _), _ in steps}, _is_teacher_run=True,
            forced_tools={name for (name, _), _ in steps}, owner="alice", **loop_args,
        )]

    events = [json.loads(c[6:]) for c in asyncio.run(run())
              if c.startswith("data: ") and not c.startswith("data: [DONE]")]
    # Legacy ends some turns after the tool round (deterministic summaries);
    # then the formatted result is what it would have sent.
    model_texts = ([m["content"] for m in requests[1] if m.get("role") == "tool"]
                   if len(requests) > 1 else formatted)
    return [e for e in events if e.get("type") in ADAPTER_EVENTS], model_texts


def _port(steps):
    events, model_texts, tool_events = [], [], []
    for index, ((name, args), (desc, result)) in enumerate(steps):
        block = function_call_to_tool_block(name, json.dumps(args))
        outcome = tools.ToolOutcome(
            f"call_{index}", block.tool_type, block.content, tools.STATUS_DONE, tools.EXIT_DONE,
            "", desc=desc, result=copy.deepcopy(result),
        )
        adapted = result_events.adapt(
            outcome, block, round_number=1, model="gpt-4o", prior_tool_events=tool_events,
        )
        events.extend(adapted.events)
        model_texts.append(adapted.model_text)
        tool_events.append(adapted.tool_event)
    return events, model_texts


def _comparable(events):
    return [{k: v for k, v in e.items() if not (e["type"] == "tool_output" and k in RUNTIME_FIELDS)}
            for e in events]


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_port_matches_the_legacy_adapter(monkeypatch, name):
    steps = FIXTURES[name]
    legacy_events, legacy_texts = _legacy(monkeypatch, steps, **LEGACY_LOOP_ARGS.get(name, {}))
    port_events, port_texts = _port(steps)

    assert _comparable(port_events) == _comparable(legacy_events)
    assert port_texts == legacy_texts


def test_fixtures_exercise_every_kept_branch(monkeypatch):
    kinds = {e["type"] for steps in FIXTURES.values() for e in _port(steps)[0]}
    assert kinds == ADAPTER_EVENTS


def test_sources_marker_never_reaches_the_model_and_search_text_is_compacted():
    events, [text] = _port(FIXTURES["web_search"])
    assert {"type": "web_sources", "data": SOURCES} in events
    assert "SOURCES" not in text
    assert len(text) <= 3200


def test_tool_event_keeps_what_history_reload_renders():
    (call, (desc, result)), = FIXTURES["generate_image"]
    block = function_call_to_tool_block(call[0], json.dumps(call[1]))
    outcome = tools.ToolOutcome("call_0", block.tool_type, block.content, tools.STATUS_DONE,
                                tools.EXIT_DONE, "", desc=desc, result=dict(result))
    event = result_events.adapt(outcome, block, round_number=2, model="m", prior_tool_events=[]).tool_event
    assert (event["round"], event["status"], event["exit_code"], event["call_id"]) == (2, "done", 0, "call_0")
    assert event["output"] == "Generated image for: a lighthouse"
    assert event["image_url"] == "/api/generated-image/abc.png"


def test_calendar_open_inherits_the_listed_range_and_draft_gets_its_document():
    events, _ = _port(FIXTURES["calendar_open_inherits_range"])
    opened = [e for e in events if e["type"] == "tool_output" and e["tool"] == "ui_control"][0]
    assert (opened["target_date"], opened["view"]) == ("2026-09-01", "month")
    events, [text] = _port(FIXTURES["draft_email"])
    [output] = [e for e in events if e["type"] == "tool_output"]
    assert (output["doc_id"], output["document_language"]) == ("0123abcd-4567", "email")
    assert "0123abcd-4567" in text
