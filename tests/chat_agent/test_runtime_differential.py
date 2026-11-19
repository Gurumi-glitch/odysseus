"""Differential test: the legacy loop is the oracle for the new runtime's stream.

Both runtimes are driven through POST /api/chat_stream with the same scripted
provider on the wire (they share ``src.llm_core._get_http_client``) and the same
tool worker. Legacy event shapes are the chat UI contract and change only
additively (ADR 0005), so every event type legacy emits, and every field it puts
on a shared type, must appear on the new runtime too, unless listed below with
a reason. An upstream merge that adds a legacy event (as #6225 added
``user_message_saved``) fails here until the new runtime emits it or the
difference is approved.
"""

import json

import pytest

import routes.chat_routes as chat_routes
import src.agent_loop as agent_loop
import src.llm_core as llm_core
import src.tool_capabilities as tool_capabilities
import src.tool_execution as tool_execution
import src.tool_security as tool_security
from src.chat_agent import store
from src.chat_agent.providers import chat_completions
from core.models import ChatMessage
from tests.chat_agent.test_chat_agent_route import (
    _Harness,
    _Request,
    _data_events,
    _sse_events,
    _sse_tool_calls,
)

REAL_LEGACY_LOOP = chat_routes.stream_agent_loop

# Legacy-only event types and (type, field) pairs, each with why it is allowed.
# APPROVED: deliberate, nothing in the chat UI reads it.
# KNOWN_GAPS: the UI reads it and the new runtime should send it; listed so the
# suite stays green while the gap is open. Fix it, then delete the entry.
# An entry no scenario produces any more is stale and fails
# test_every_listed_difference_still_occurs, so the lists only shrink by hand.
APPROVED = {
    "turn_contract": "request authority / turn contract are legacy-only (ADR 0001, 0005); no UI reader",
    "tool_resolution_audit": "legacy tool-routing diagnostics; no UI reader",
    "tool_routing_audit": "legacy tool-routing diagnostics; no UI reader",
    ("delta", "round"): "no UI reader on delta events",
    ("delta", "render_owner"): "absent means 'streamed' (static/js/turnRendering.js accepts)",
    ("tool_start", "tool_call_id"): "new runtime sends call_id instead (spec #44); no UI reader",
    ("tool_output", "tool_call_id"): "new runtime sends call_id instead (spec #44); no UI reader",
    ("tool_output", "action_receipt"): "dropped by the result_events port; no UI reader",
    ("tool_output", "ask_user"): "legacy repeats the card on tool_output; chat.js reads the ask_user event, "
                                 "which the new runtime sends, and the card is saved on the tool event (#139)",
    ("final_response", "render_owner"): "legacy ends its turn at the question and marks that reply as "
                                        "replacing the turn; the new runtime's reply comes after the answer",
    ("final_response", "replacement_scope"): "same as final_response.render_owner (#139)",
}
KNOWN_GAPS = {}

def _sse_reasoning_tool_call(reasoning, call):
    """A scripted round whose `reasoning` delta comes before one tool call."""
    name, arguments = call
    chunks = [
        {"choices": [{"delta": {"reasoning": reasoning}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_0", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]},
    ]
    return ("\n\n".join(["data: " + json.dumps(c) for c in chunks] + ["data: [DONE]"]) + "\n\n").encode()


SCENARIOS = {
    "text_reply": {
        "request": {},
        "rounds": [_sse_events(["Hel", "lo!"])],
        "results": {},
    },
    # The preset's character name rides on model_info; chat.js hides thinking for it (#131).
    "persona": {
        "request": {},
        "rounds": [_sse_events(["<think>hidden</think>", "Ahoy."])],
        "results": {},
        "character_name": "Penelope",
    },
    # What the real context builder does: save this turn's user row first (#6225).
    "user_row_saved": {
        "request": {},
        "rounds": [_sse_events(["Hi."])],
        "results": {},
        "user_row_id": "user-row-7",
    },
    "web_search_round": {
        "request": {"allow_web_search": "true"},
        "rounds": [
            _sse_tool_calls(("web_search", {"query": "odysseus release"})),
            _sse_events(["Found it."]),
        ],
        "results": {"web_search": {"output": "Odysseus 2.0 shipped.", "exit_code": 0}},
    },
    # Reasoning in the `reasoning` field before a tool round (#142); the new runtime
    # replays it on the second request, which this test does not compare.
    "reasoning_tool": {
        "request": {"allow_web_search": "true"},
        "rounds": [
            _sse_reasoning_tool_call("Need to search.", ("web_search", {"query": "odysseus release"})),
            _sse_events(["Found it."]),
        ],
        "results": {"web_search": {"output": "Odysseus 2.0 shipped.", "exit_code": 0}},
    },
    # A question next to another call (#139). Non-interactive, so the new runtime's wait
    # is answered "deny" at once; the legacy loop has no wait and ends its turn at the question.
    "ask_user_plus_tool": {
        "request": {"interactive": "false"},
        "rounds": [
            _sse_tool_calls(
                ("manage_notes", {"action": "add", "title": "Groceries", "content": "milk"}),
                ("ask_user", {"question": "Which list?", "options": [{"label": "Work"}, {"label": "Home"}]}),
            ),
            _sse_events(["Noted."]),
        ],
        "results": {
            "manage_notes": {"output": "saved", "exit_code": 0},
            "ask_user": {
                "output": "", "exit_code": 0,
                "ask_user": {"question": "Which list?", "options": [{"label": "Work"}, {"label": "Home"}]},
            },
        },
    },
}


def _event_key(event):
    return event.get("type") or ("delta" if "delta" in event else "<untyped>")


def _shapes(events):
    """Event type -> union of the field names seen on it."""
    shapes = {}
    for event in events:
        shapes.setdefault(_event_key(event), set()).update(event)
    return shapes


def _differences(legacy_events, new_events):
    """Legacy-only event types, plus (type, field) for fields legacy puts on a shared type."""
    legacy, new = _shapes(legacy_events), _shapes(new_events)
    types = set(legacy) - set(new)
    fields = {(kind, field) for kind in set(legacy) & set(new) for field in legacy[kind] - new[kind]}
    return types | fields


async def _run(monkeypatch, tmp_path, runtime, scenario):
    """One turn on one runtime. Relies on _Harness stubbing only the legacy loop
    (restored here) and routing every provider request through its one mock client."""
    harness = _Harness(monkeypatch, tmp_path / runtime)
    monkeypatch.setattr(chat_routes, "stream_agent_loop", REAL_LEGACY_LOOP)
    monkeypatch.setattr(llm_core, "_get_http_client", chat_completions._get_http_client)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(tool_security, "owner_is_admin_or_single_user", lambda owner: True)
    # Approval gates differ by design (ADR 0001); the stream contract is under test.
    monkeypatch.setattr(tool_capabilities, "TOOL_APPROVAL_GATE_ENABLED", False)

    async def worker(block, *args, **kwargs):
        return f"{block.tool_type}: ran", dict(scenario["results"][block.tool_type])

    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", worker)
    if "user_row_id" in scenario:
        build = chat_routes.build_chat_context

        async def saving_build_context(*args, **kwargs):
            harness.session.history.append(
                ChatMessage("user", "hello", metadata={"_db_id": scenario["user_row_id"]})
            )
            return await build(*args, **kwargs)

        monkeypatch.setattr(chat_routes, "build_chat_context", saving_build_context)
    if "character_name" in scenario:
        build = chat_routes.build_chat_context

        async def persona_build_context(*args, **kwargs):
            ctx = await build(*args, **kwargs)
            ctx.preset.character_name = scenario["character_name"]
            return ctx

        monkeypatch.setattr(chat_routes, "build_chat_context", persona_build_context)
    store.set_chat_runtime("alice", store.RUNTIME_NEW if runtime == "new" else store.RUNTIME_LEGACY)
    harness.provider_rounds = list(scenario["rounds"])
    request = scenario["request"]
    response, body = await harness.send(_Request("agent", **request))
    assert response.headers["X-Odysseus-Runtime"] == ("chat_agent" if runtime == "new" else "legacy")
    return _data_events(body), harness


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(SCENARIOS))
async def test_new_runtime_emits_every_legacy_event_shape(monkeypatch, tmp_path, name):
    scenario = SCENARIOS[name]
    legacy_events, _ = await _run(monkeypatch, tmp_path, "legacy", scenario)
    new_events, _ = await _run(monkeypatch, tmp_path, "new", scenario)
    unexplained = sorted(_differences(legacy_events, new_events) - set(APPROVED) - set(KNOWN_GAPS), key=str)
    assert unexplained == [], (
        f"legacy sends {unexplained} and the new runtime does not; emit it, or list it in "
        f"APPROVED / KNOWN_GAPS with a reason.\nlegacy: {legacy_events}\nnew: {new_events}"
    )


@pytest.mark.asyncio
async def test_every_listed_difference_still_occurs(monkeypatch, tmp_path):
    seen = set()
    for name, scenario in SCENARIOS.items():
        legacy_events, _ = await _run(monkeypatch, tmp_path / name, "legacy", scenario)
        new_events, _ = await _run(monkeypatch, tmp_path / name, "new", scenario)
        seen |= _differences(legacy_events, new_events)
    stale = sorted((set(APPROVED) | set(KNOWN_GAPS)) - seen, key=str)
    assert stale == [], f"no longer differs, delete from APPROVED / KNOWN_GAPS: {stale}"
