"""Entry point of the chat_agent runtime: run one agent turn as SSE events.

The turn runs rounds until a reply has no tool calls or a shared cap is hit.
Each round's tool calls run one at a time in call order; errors and denials go
back to the model as tool results. The turn also stops at the round and
tool-call caps, when the repeat guard stops it, and after Finish (one more
round with no tools). Messages the user sends meanwhile (steers) join the
turn at the next round boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from contextlib import aclosing
from dataclasses import asdict, dataclass, replace
from typing import (
    Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence,
    Union,
)

# Relies on: _configured_model_tool_surface returns the per-model tool surface
# ("none" | "compact" | "full" | "") for this owner's endpoint row and model.
from src.agent_loop import _configured_model_tool_surface
# Relies on: _compute_final_metrics builds the legacy metrics dict (tokens,
# speed, context %, estimated cost) from these inputs alone; _usage_bucket and
# _usage_bucket_summary build and sum per-round usage attribution.
from src.agent_loop import _compute_final_metrics, _usage_bucket, _usage_bucket_summary
from src.agent_tools import MAX_AGENT_ROUNDS
# Relies on: agent_runs.should_finish(session_id) is True only while that
# session's registered run is running and Finish was requested for it; a run
# that replaces it starts with the flag clear.
from src import agent_runs
from src.chat_agent import (
    asks, classifier, permissions, prompt, repeat_guard, result_events, store, tools, transcript, turn_context,
)
from src.chat_agent.providers import (
    XAI,
    Endpoint,
    ProviderError,
    RoundEnd,
    RoundEvent,
    TextDelta,
    chat_completions,
    provider_key,
    xai_responses,
)
from src.chat_agent.routing import ComposerToggles
from src import settings
from src.llm_core import LLMConfig
from src.model_context import estimate_tokens
# Relies on: function_call_to_tool_block converts a native call into the
# executor's block (bare email names become mcp__email__*), or returns None
# when the arguments are unusable.
from src.tool_schemas import function_call_to_tool_block

logger = logging.getLogger(__name__)

TOOL_CALL_CAP_REASON = "Not run: this message reached its tool-call limit."
ASK_EXPIRED_REASON = "Not run: the user did not answer the request in time."
FINISH_REASON = "Not run: the user asked to finish with the results so far."
REPEAT_STOP_REASON = (
    f"Not run: this exact call was repeated {repeat_guard.STOP_STREAK} times in a row, "
    "so the turn stopped."
)
REPEAT_STOP_MESSAGE = (
    f"The agent repeated the same tool call {repeat_guard.STOP_STREAK} times in a row "
    "without progress, so this turn stopped."
)
_NOT_RUN_REASONS = {
    transcript.STOP_BUDGET: TOOL_CALL_CAP_REASON,
    transcript.STOP_INTERRUPTED: FINISH_REASON,
    transcript.STOP_REPEAT_GUARD: REPEAT_STOP_REASON,
    transcript.STOP_ASK_EXPIRED: ASK_EXPIRED_REASON,
}
# With no fallback configured, one retry on these (the legacy foreground
# availability retry, spec #44 Providers).
SINGLE_CANDIDATE_RETRY_STATUSES = frozenset({502, 503, 504, 529})


@dataclass(frozen=True)
class Candidate:
    endpoint: Endpoint
    # Its build_foreground_route_descriptors entry (endpoint_id,
    # endpoint_label, endpoint_cost_tracked) for the legacy fallback event.
    route: Dict[str, Any]


@dataclass(frozen=True)
class Fallback:
    """A fallback candidate produced output; the rest of the turn uses it."""

    candidate: Candidate
    # The request it answered, compacted when the history exceeded its window.
    conversation: List[Dict[str, Any]]
    event: Dict[str, Any]


@dataclass(frozen=True)
class ChatAgentTurn:
    session_id: str
    sess: Any
    # The signed-in user the route resolved (routing refuses turns without one).
    owner: str
    toggles: ComposerToggles
    auto_escalated: bool
    temperature: float
    max_tokens: int
    # The route's build_chat_context bound to this request; prompt picks the flags.
    build_context: Callable[..., Awaitable[Any]]
    privileges: Mapping[str, Any]
    delegated_credential: bool
    active_email_open: bool
    client_runtime_context: Optional[Dict[str, Any]]
    # The foreground candidate list, the session's own route first.
    candidates: Sequence[Candidate]
    # The foreground policy's eligible_statuses.
    fallback_statuses: FrozenSet[int]
    # Moves the session's stored model and endpoint (fallback is sticky).
    stick_route: Callable[[Candidate], None]
    # What the browser has open: the reader's email (uid/folder/account and
    # cached subject/from/preview), the visible editor tab's document id, and
    # the approved plan it sends back while that plan runs.
    active_email: Optional[Dict[str, str]] = None
    active_document_id: str = ""
    approved_plan: str = ""
    # The route's run_post_response_tasks bound to this request (memory and
    # skill extraction, webhook, auto-name); the loop adds the turn's results.
    after_turn: Optional[Callable[..., None]] = None
    # False when nobody can answer an ask (interactive:false or an API token):
    # every ask then resolves at once as a denial.
    interactive: bool = True
    # The route's SkillsManager, and the skill a /name invocation started
    # this turn (its text is already in the message; its tools load here).
    skills_manager: Any = None
    invoked_skill: str = ""
    # The composer's auto mode toggle as sent (#68); the classifier module
    # decides whether this user and turn may use it.
    auto_mode: bool = False
    # The user's message as sent, for auto mode's classifier.
    message: str = ""


@dataclass(frozen=True)
class Caps:
    """The shared per-message caps ("max steps", "max tool calls"; 0 = unlimited)."""

    max_rounds: int
    max_tool_calls: int


def _event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _error_event(message: str, status: Optional[int] = None) -> str:
    return f"event: error\ndata: {json.dumps({'error': message, 'status': status})}\n\n"


def _setting_int(key: str, default: int) -> int:
    # settings.json can be hand-edited past the admin endpoint's validation.
    try:
        return int(settings.get_setting(key, default))
    except (TypeError, ValueError):
        return default


def read_caps() -> Caps:
    return Caps(
        max_rounds=max(1, _setting_int("agent_max_rounds", MAX_AGENT_ROUNDS)),
        max_tool_calls=max(0, _setting_int("agent_max_tool_calls", 0)),
    )


def _stream_round(
    turn: ChatAgentTurn, provider: str, endpoint: Endpoint, conversation: List[Dict[str, Any]],
    tool_schemas: Sequence[Dict[str, Any]],
) -> AsyncIterator[RoundEvent]:
    if provider == XAI:
        return xai_responses.stream_round(
            endpoint, conversation, cache_key=turn.session_id,
            temperature=turn.temperature, max_tokens=turn.max_tokens, tools=tool_schemas,
        )
    return chat_completions.stream_round(
        endpoint, conversation, temperature=turn.temperature, max_tokens=turn.max_tokens,
        tools=tool_schemas,
    )


async def _round_with_fallback(
    turn: ChatAgentTurn, endpoint: Endpoint, conversation: List[Dict[str, Any]],
    tool_schemas: Sequence[Dict[str, Any]], history_start: int, current: Dict[str, Any],
) -> AsyncIterator[Union[RoundEvent, Fallback]]:
    """One round on the turn's current candidate; a failure before any output
    with an eligible status moves to the next candidate (spec #44 Providers).
    A failure after output passes through: switching would repeat streamed text.
    A fallback whose window the history exceeds gets it compacted once
    (``history_start`` and ``current``: see transcript.fit_request)."""
    candidates = turn.candidates
    start = next(i for i, candidate in enumerate(candidates) if candidate.endpoint == endpoint)
    failures: List[Dict[str, Any]] = []
    reason = ""
    for index in range(start, len(candidates)):
        candidate = candidates[index]
        request = conversation if index == start else await transcript.fit_request(
            conversation, history_start, current, candidate.endpoint, turn.owner,
        )
        retry_left = len(candidates) == 1
        while True:
            failure = None
            committed = False
            held: List[RoundEvent] = []
            provider = provider_key(candidate.endpoint.url)
            async with aclosing(_stream_round(turn, provider, candidate.endpoint, request, tool_schemas)) as events:
                async for event in events:
                    if not committed:
                        if isinstance(event, ProviderError):
                            failure = event
                            break
                        if isinstance(event, TextDelta) and not event.text.strip():
                            held.append(event)
                            continue
                        committed = True
                        if index != start:
                            turn.stick_route(candidate)
                            yield Fallback(candidate, request, _fallback_event(
                                candidates[0], candidate, index, reason, failures,
                            ))
                        for event_held in held:
                            yield event_held
                    yield event
            if failure is None:
                return
            if retry_left and failure.status in SINGLE_CANDIDATE_RETRY_STATUSES:
                retry_left = False
                logger.warning("chat_agent: retrying %s after pre-output HTTP %s",
                               candidate.endpoint.model, failure.status)
                await asyncio.sleep(LLMConfig.RETRY_DELAY)
                continue
            break
        if failure.status not in turn.fallback_statuses or index == len(candidates) - 1:
            yield failure
            return
        logger.warning("chat_agent: %s failed before output (HTTP %s); trying the next fallback",
                       candidate.endpoint.model, failure.status)
        reason = reason or failure.message
        failures.append({"candidate_index": index, "model": candidate.endpoint.model, "status": failure.status})


def _fallback_event(
    selected: Candidate, answered: Candidate, index: int, reason: str, failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """The legacy ``fallback`` event (llm_core.stream_llm_with_fallback)."""
    return {
        "type": "fallback",
        "selected_model": selected.endpoint.model,
        "answered_by": answered.endpoint.model,
        "selected_endpoint_id": selected.route.get("endpoint_id"),
        "selected_endpoint_label": selected.route.get("endpoint_label"),
        "selected_endpoint_cost_tracked": selected.route.get("endpoint_cost_tracked"),
        "answered_by_endpoint_id": answered.route.get("endpoint_id"),
        "answered_by_endpoint_label": answered.route.get("endpoint_label"),
        "answered_by_endpoint_cost_tracked": answered.route.get("endpoint_cost_tracked"),
        "candidate_index": index,
        "reason": reason,
        "failures": failures,
    }


def _round_record(end: RoundEnd) -> Dict[str, Any]:
    if end.usage is None:
        return {"blocks": end.blocks}
    return {"blocks": end.blocks, "usage": asdict(end.usage)}


async def run_chat_agent_turn(turn: ChatAgentTurn) -> AsyncGenerator[str, None]:
    yield _event({"type": "turn_mode", "mode": "agent", "auto_escalated": turn.auto_escalated})
    sess = turn.sess
    endpoint = turn.candidates[0].endpoint
    provider = provider_key(endpoint.url)
    record = transcript.TurnRecord(provider=provider, model=endpoint.model)
    writer = transcript.TurnWriter(sess, turn.session_id, record, incognito=turn.toggles.incognito)
    contexts = []

    async def build_context(**flags):
        # Keeps the context so the row projects what the route resolved.
        contexts.append(await turn.build_context(**flags))
        return contexts[-1]

    streamed: List[str] = []
    usage = _TurnUsage(time.monotonic(), turn.candidates[0])

    def hold_ask(call: Dict[str, Any], tool: str, command: str, card: dict) -> None:
        # The card is on the row before anyone can answer it, so a restart
        # while it waits can mark it expired.
        record.pending_ask = {
            "round": len(record.rounds), "model": endpoint.model, "tool": tool, "command": command,
            "status": "pending", "call_id": call["call_id"], "ask_user": card,
        }
        writer.checkpoint()

    steers = asks.SteerQueue(turn.session_id, turn.owner)
    try:
        # Before anything reads the history: the context builder, and the
        # egress provenance the summary row carries over (#67). Nobody-mode
        # history is a request-local transcript, never compacted.
        if not turn.toggles.incognito:
            window = await transcript.compact_history(sess, turn.session_id, endpoint, turn.owner)
            if window:
                yield _event({"type": "compacted", "context_length": window})
        caps = read_caps()
        rules = permissions.Rules.from_lists(store.user_rules(turn.owner))
        # The open email preview is both external content and a private read (ADR 0002).
        provenance = permissions.Provenance(
            external=turn.active_email_open, private=turn.active_email_open,
            origins=("active email preview",) if turn.active_email_open else (),
        ).after_history(getattr(turn.sess, "history", None) or ())
        # Before the context builder saves this turn's user row.
        auto = classifier.AutoMode.for_turn(turn)
        if auto.state != classifier.OFF:
            yield _event(auto.event())
        layers = permissions.build_deny_layers(
            owner=turn.owner,
            toggles=turn.toggles,
            privileges=turn.privileges,
            delegated_credential=turn.delegated_credential,
            active_email_open=turn.active_email_open,
            global_disabled_tools=settings.get_setting("disabled_tools", []),
            rules=rules,
        )
        surface = _configured_model_tool_surface(
            endpoint.url, endpoint.model, turn.owner, headers=endpoint.headers,
        )
        upfront = tools.upfront_schemas(layers.denied_names, surface)
        # Non-admins never see MCP tools (spec #44, managed deny layer).
        catalog = tools.deferred_catalog(
            layers.denied_names, include_mcp=not layers.hide_all_mcp,
        ) if upfront else []
        # The reader's denials come and go with the open email, so the turn
        # context names them; the system prompt stays the same across turns.
        reader_denied = permissions.EMAIL_READER_DENIED if turn.active_email_open else frozenset()
        unavailable = [
            name for name in tools.upfront_tool_names()
            if name in layers.disabled and name not in reader_denied
        ]
        skills = turn.skills_manager if upfront and turn.skills_manager is not None and tools.skills_enabled(
            turn.owner, sess, turn.toggles.incognito,
        ) else None
        invoked_tools = () if turn.toggles.incognito else tools.invoked_skill_tools(
            turn.skills_manager, turn.owner, turn.invoked_skill, catalog,
        )
        document = turn_context.load_document(turn.owner, turn.active_document_id)
        conversation, history_start = await prompt.build_messages(
            build_context,
            [tools.unavailable_line(unavailable), tools.catalog_section(catalog)] if upfront else [],
            owner=turn.owner,
            toggles=turn.toggles,
            model=endpoint.model,
            open_items=turn_context.OpenItems(
                document=document, email=turn.active_email,
                approved_plan=turn.approved_plan, denied=layers.denied_names,
            ),
            skill_index=prompt.skill_index_section(
                skills.index_for(owner=turn.owner),
                _setting_int("skill_index_max_chars", settings.DEFAULT_SETTINGS["skill_index_max_chars"]),
            ) if skills is not None else "",
        )
        # The turn's user message: a fallback's compaction keeps it and what follows.
        current = conversation[-1]
        tool_schemas, offered = _offered(upfront, catalog, conversation, surface, skills, invoked_tools)
        scope = tools.ExecutionScope(
            owner=turn.owner,
            session_id=turn.session_id,
            workspace=turn.toggles.workspace,
            client_runtime_context=turn.client_runtime_context,
            denied_names=layers.denied_names,
            active_document_id=document.id if document is not None else None,
        )
        record.context = _context_projections(contexts[-1])
        tool_calls_used = 0
        guard = repeat_guard.RepeatGuard()
        round_tools = tool_schemas
        # Tool images live for one round: the next request shows them, the one
        # after drops them, so frames never pile up in the request.
        visual = None
        while True:
            end = None
            streamed.clear()
            yield _event({"type": "agent_step", "round": len(record.rounds) + 1})
            async for event in _round_with_fallback(
                turn, endpoint, conversation, round_tools, history_start, current,
            ):
                if isinstance(event, Fallback):
                    endpoint, conversation = event.candidate.endpoint, event.conversation
                    usage.route = event.candidate
                    record.provider, record.model = provider_key(endpoint.url), endpoint.model
                    yield _event(event.event)
                elif isinstance(event, ProviderError):
                    _finish_failed_turn(writer)
                    yield _event({"type": "metrics", "data": usage.metrics(
                        record, conversation, contexts[-1].context_length,
                    )})
                    yield _error_event(event.message, event.status)
                    yield "data: [DONE]\n\n"
                    return
                if isinstance(event, TextDelta):
                    usage.mark_output()
                    streamed.append(event.text)
                    yield _event({"delta": event.text})
                elif isinstance(event, RoundEnd):
                    usage.mark_output()
                    end = event
            streamed.clear()
            round_record = _round_record(end)
            record.rounds.append(round_record)
            usage.add_round(len(record.rounds), end, conversation)
            writer.checkpoint()
            calls = [b for b in end.blocks if b["type"] == "tool_call"]
            if not calls or record.stop_reason == transcript.STOP_INTERRUPTED:
                # Finish's tool-less round is the turn's last. A steer sent
                # during a plain answer gets one more round to be answered.
                if calls or record.stop_reason or not steers.texts or len(record.rounds) >= caps.max_rounds:
                    break
                conversation.append({"role": "assistant", "blocks": end.blocks})
                _join_steers(record, conversation, auto, steers.take())
                writer.checkpoint()
                continue
            conversation.append({"role": "assistant", "blocks": end.blocks})
            results = round_record["results"] = []
            round_outcomes = []
            for call in calls:
                # tool_search and load_skill are the runtime's own tools: they
                # are answered here, never converted for the legacy executor.
                own_tool = call["name"] in (tools.TOOL_SEARCH, tools.LOAD_SKILL) and call["name"] in offered
                block = None if own_tool else function_call_to_tool_block(call["name"], call["arguments"])
                reminder = None
                # The last card this call raised, stored on its tool event.
                ask = None
                if record.stop_reason is None and agent_runs.should_finish(turn.session_id):
                    record.stop_reason = transcript.STOP_INTERRUPTED
                if record.stop_reason is None and caps.max_tool_calls and tool_calls_used >= caps.max_tool_calls:
                    record.stop_reason = transcript.STOP_BUDGET
                if record.stop_reason is not None:
                    # Every call still gets a result, so the round replays
                    # as a complete call/result pair.
                    decision = permissions.Decision(permissions.DENY, _NOT_RUN_REASONS[record.stop_reason])
                else:
                    tool_calls_used += 1
                    signal = guard.observe(call["name"], call["arguments"])
                    if signal.stop:
                        ask = _Ask(turn, asks.repeat_guard_card(
                            turn.session_id, call["name"], call["arguments"], signal.streak,
                        ))
                        hold_ask(call, call["name"], call["arguments"], ask.card)
                        async for event in ask.events():
                            yield event
                        if ask.reply is None:
                            record.stop_reason = transcript.STOP_ASK_EXPIRED
                        elif ask.reply.decision != asks.ONCE:
                            record.stop_reason = transcript.STOP_REPEAT_GUARD
                    if record.stop_reason is not None:
                        decision = permissions.Decision(permissions.DENY, _NOT_RUN_REASONS[record.stop_reason])
                    else:
                        reminder = signal.reminder
                        decision = (permissions.Decision(permissions.ALLOW) if own_tool
                                    else _gate(turn, call, block, layers, offered, rules, provenance, auto))
                if decision.outcome == permissions.ASK and decision.auto and auto.active:
                    verdict = await auto.judge(
                        block.tool_type, block.content, chat_endpoint=endpoint,
                        provenance=provenance, tool_events=record.tool_events,
                    )
                    if verdict is not None:
                        decision = permissions.Decision(permissions.ALLOW) if verdict.allow else classifier.denial(verdict)
                        if not auto.active:
                            yield _event(auto.event())
                if decision.outcome == permissions.ASK:
                    rule = permissions.session_rule(block.tool_type, block.content) if decision.chat_scope else None
                    task = (permissions.task_card(turn.owner, block.content)
                            if permissions.policy_name(block.tool_type) == "manage_tasks" else None)
                    ask = _Ask(turn, asks.permission_card(
                        turn.session_id, block.tool_type, block.content, rule.text if rule else None, task,
                    ))
                    hold_ask(call, block.tool_type, block.content, ask.card)
                    async for event in ask.events():
                        yield event
                    if ask.reply is None:
                        record.stop_reason = transcript.STOP_ASK_EXPIRED
                    decision = _permission_decision(turn, rule, ask.reply)
                    if decision.outcome == permissions.ALLOW and auto.approved():
                        yield _event(auto.event())
                adapted = None
                if decision.outcome == permissions.ALLOW:
                    tool, command = (call["name"], call["arguments"]) if own_tool else (block.tool_type, block.content)
                    yield _event({
                        "type": "tool_start", "tool": tool,
                        "command": command if own_tool else result_events.command_display(block),
                        "full_command": command.strip(), "round": len(record.rounds),
                        "call_id": call["call_id"],
                    })
                    if own_tool and call["name"] == tools.LOAD_SKILL:
                        outcome = tools.run_load_skill(call, skills, turn.owner, catalog)
                    elif own_tool:
                        outcome = tools.run_search(call, catalog)
                    else:
                        # Only a task the user approved on its own card gets its prompt's grants.
                        approved = (ask is not None and ask.card["ask_kind"] == asks.PERMISSION
                                    and ask.reply is not None and ask.reply.decision == asks.ONCE)
                        call_scope = replace(scope, approved_task=approved)
                        async for item in _execute_with_progress(call["call_id"], block, call_scope, len(record.rounds)):
                            if isinstance(item, tools.ToolOutcome):
                                outcome = item
                            else:
                                yield item
                        adapted = result_events.adapt(
                            outcome, block, round_number=len(record.rounds), model=endpoint.model,
                            prior_tool_events=record.tool_events,
                        )
                        outcome.model_text = adapted.model_text
                        provenance = provenance.after(block.tool_type)
                    question = outcome.result.get("ask_user") if outcome.ok else None
                    if question:
                        ask = _Ask(turn, asks.question_card(turn.session_id, question))
                        hold_ask(call, block.tool_type, block.content, ask.card)
                        async for event in ask.events():
                            yield event
                        if ask.reply is None:
                            record.stop_reason = transcript.STOP_ASK_EXPIRED
                        outcome.model_text = _question_result(ask.reply)
                else:
                    outcome = tools.refused_outcome(
                        call, block, decision.reason,
                        denied=decision.outcome != permissions.INVALID,
                    )
                if adapted is None:
                    output_event = _tool_output_event(outcome, attempted=decision.outcome == permissions.ALLOW)
                    tool_event = outcome.tool_event(len(record.rounds), endpoint.model)
                    if decision.by:
                        # The card shows who denied it, live and on reload.
                        output_event["denied_by"] = tool_event["denied_by"] = decision.by
                    yield _event(output_event)
                else:
                    for ui_event in adapted.events:
                        if ui_event["type"] == "web_sources":
                            # The reply's sources box on reload (legacy keeps the last search's).
                            record.context["web_sources"] = ui_event["data"]
                        yield _event(ui_event)
                    tool_event = adapted.tool_event
                if ask is not None:
                    tool_event["ask_user"] = asks.resolved_card(ask.card, ask.reply)
                record.pending_ask = None
                record.tool_events.append(tool_event)
                record.link_sources.append({**outcome.result, **tool_event})
                round_outcomes.append(outcome)
                envelope = outcome.envelope()
                if reminder:
                    # The reminder is for the model only; the tool card keeps the output.
                    envelope["content"] += reminder
                results.append(envelope)
                conversation.append({"role": "tool", **envelope})
                # Per call, not per round: a restart must not mark a call
                # that already ran as cancelled (it would replay as not run).
                writer.checkpoint()
            conversation = [message for message in conversation if message is not visual]
            visual = tools.visual_evidence([o.result for o in round_outcomes], endpoint.model)
            if visual is not None:
                conversation.append(visual)
            joined = steers.take()
            if joined:
                _join_steers(record, conversation, auto, joined)
                writer.checkpoint()
            # Tools loaded by this round's tool_search join the next round.
            tool_schemas, offered = _offered(upfront, catalog, conversation, surface, skills, invoked_tools)
            round_tools = tool_schemas
            if record.stop_reason == transcript.STOP_ASK_EXPIRED:
                # Nobody is there to answer; keep the work done so far.
                break
            if record.stop_reason == transcript.STOP_BUDGET:
                yield _event({"type": "budget_exceeded", "kind": "tool_calls",
                              "limit": caps.max_tool_calls, "used": tool_calls_used})
                break
            if record.stop_reason == transcript.STOP_REPEAT_GUARD:
                yield _event({"type": "loop_breaker_triggered", "reason": transcript.STOP_REPEAT_GUARD,
                              "message": REPEAT_STOP_MESSAGE})
                break
            if record.stop_reason is None and agent_runs.should_finish(turn.session_id):
                record.stop_reason = transcript.STOP_INTERRUPTED
            if record.stop_reason == transcript.STOP_INTERRUPTED:
                # Finish: one more round with no tools, so the model answers.
                round_tools = ()
                continue
            if len(record.rounds) >= caps.max_rounds:
                record.stop_reason = transcript.STOP_MAX_ROUNDS
                yield _event({"type": "rounds_exhausted", "rounds": caps.max_rounds})
                break
        # Steers that came after the last boundary stay in the record, so the
        # next turn's request carries them.
        unused = steers.close()
        if unused:
            record.add_steers(unused)
        # The last round's text with its titles linked, once (spec #44, Stream events).
        yield _event({"type": "final_response", "content": record.round_texts[-1]})
        message_id = writer.finish()
        metrics = usage.metrics(record, conversation, contexts[-1].context_length)
        _after_turn(turn, record, contexts[-1], message_id, tool_calls_used, metrics)
    except (asyncio.CancelledError, GeneratorExit):
        # Stop: keep the completed blocks and the text streamed so far; partial
        # reasoning is dropped and unfinished calls get cancelled results.
        if streamed:
            record.rounds.append({"blocks": [{"type": "text", "text": "".join(streamed)}]})
        unused = steers.close()
        if record.rounds:
            if unused:
                record.add_steers(unused)
            record.stop_reason = transcript.STOP_INTERRUPTED
            writer.finish()
        raise
    except Exception:
        # A crash ends the turn visibly; it never falls back to legacy, which
        # could run side effects twice (ADR 0005).
        logger.exception("chat_agent turn failed for session %s", turn.session_id)
        _finish_failed_turn(writer)
        yield _error_event("The chat agent runtime failed on this turn. See the server log.")
        yield "data: [DONE]\n\n"
        return
    finally:
        steers.close()
    yield _event({"type": "metrics", "data": metrics})
    if message_id:
        yield _event({"type": "message_saved", "id": message_id})
    yield "data: [DONE]\n\n"


def _finish_failed_turn(writer: transcript.TurnWriter) -> None:
    """A row already checkpointed must not stay ``in_progress`` after an error."""
    if writer.started:
        writer.record.stop_reason = transcript.STOP_ERROR
        writer.finish()


def _context_projections(ctx: Any) -> Dict[str, Any]:
    """What save_assistant_response projects from the route's context."""
    return {
        "character_name": ctx.preset.character_name,
        "web_sources": ctx.web_sources,
        "rag_sources": ctx.rag_sources,
        "memories_used": ctx.used_memories,
    }


def _after_turn(
    turn: ChatAgentTurn, record: transcript.TurnRecord, ctx: Any, message_id: Optional[str], tool_calls: int,
    metrics: Dict[str, Any],
) -> None:
    """Post-response tasks and the SFT trace, once per completed turn."""
    content = record.content()
    if turn.after_turn is not None:
        turn.after_turn(
            full_response=content, last_metrics=metrics, uprefs=ctx.uprefs,
            character_name=ctx.preset.character_name,
            agent_rounds=len(record.rounds), agent_tool_calls=tool_calls,
        )
    if not turn.toggles.incognito:
        # Relies on: _append_sft_trace_record writes only for owners with SFT
        # capture enabled and never raises.
        from routes.chat_helpers import _append_sft_trace_record
        _append_sft_trace_record(
            owner=turn.owner, session_id=turn.session_id, sess=turn.sess,
            assistant_content=content, metadata=record.metadata(), message_id=message_id,
        )


async def _execute_with_progress(
    call_id: str, block: Any, scope: tools.ExecutionScope, round_number: int,
) -> AsyncIterator[Union[str, tools.ToolOutcome]]:
    """Run the call, yielding its ``tool_progress`` events (long bash/python,
    web_search) as they arrive, then its outcome."""
    progress: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(tools.execute(call_id, block, scope, progress_cb=progress.put))
    task.add_done_callback(lambda _done: progress.put_nowait(None))
    try:
        while (payload := await progress.get()) is not None:
            yield _event({"type": "tool_progress", "tool": block.tool_type, "round": round_number, **payload})
        yield task.result()
    finally:
        # Stop or a client disconnect must not leave the tool (and any
        # subprocess it spawned) running with nothing awaiting it.
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


class _TurnUsage:
    """Cumulative usage of the turn, as the legacy final metrics report it."""

    def __init__(self, started: float, route: Candidate) -> None:
        self.started = started
        self.first_output: Optional[float] = None
        self.requested = route
        self.route = route
        self.buckets: List[Dict[str, Any]] = []

    def mark_output(self) -> None:
        if self.first_output is None:
            self.first_output = time.monotonic()

    def add_round(self, number: int, end: RoundEnd, request: List[Dict[str, Any]]) -> None:
        if end.usage is not None:
            tokens, source = (end.usage.input_tokens, end.usage.output_tokens), "real"
        else:
            text = "".join(block.get("text", "") for block in end.blocks if block["type"] == "text")
            tokens, source = (estimate_tokens(request), len(text) // 4), "estimated"
        self.buckets.append(_usage_bucket(
            round_num=number, model=self.route.endpoint.model,
            endpoint_id=self.route.route.get("endpoint_id"),
            endpoint_label=self.route.route.get("endpoint_label"),
            endpoint_cost_tracked=self.route.route.get("endpoint_cost_tracked"),
            input_tokens=tokens[0], output_tokens=tokens[1], usage_source=source,
        ))

    def metrics(
        self, record: transcript.TurnRecord, conversation: List[Dict[str, Any]], context_length: int,
    ) -> Dict[str, Any]:
        elapsed = time.monotonic() - self.started
        real = [r["usage"] for r in record.rounds if "usage" in r]
        endpoint = self.route.endpoint
        metrics = _compute_final_metrics(
            conversation, record.content(), elapsed,
            (self.first_output - self.started) if self.first_output else 0,
            context_length,
            sum(u["input_tokens"] for u in real), sum(u["output_tokens"] for u in real), bool(real),
            record.tool_events, record.round_texts, model=endpoint.model,
            round_models=[b["model"] for b in self.buckets],
            round_endpoint_ids=[b["endpoint_id"] for b in self.buckets],
            round_endpoint_labels=[b["endpoint_label"] for b in self.buckets],
            last_round_input_tokens=real[-1]["input_tokens"] if real else 0,
            endpoint_url=endpoint.url,
        )
        metrics.update({
            "requested_model": self.requested.endpoint.model,
            "endpoint_id": self.route.route.get("endpoint_id"),
            "endpoint_label": self.route.route.get("endpoint_label"),
            "requested_endpoint_id": self.requested.route.get("endpoint_id"),
            "requested_endpoint_label": self.requested.route.get("endpoint_label"),
        })
        if isinstance(self.route.route.get("endpoint_cost_tracked"), bool):
            metrics["endpoint_cost_tracked"] = self.route.route["endpoint_cost_tracked"]
        summary = _usage_bucket_summary(self.buckets)
        if summary:
            # As legacy: the totals are the buckets' and context % is the last
            # request's prompt.
            metrics.update(summary)
            if elapsed > 0:
                metrics["tokens_per_second"] = round(summary["output_tokens"] / elapsed, 2)
            if context_length:
                metrics["context_percent"] = min(
                    round(self.buckets[-1]["input_tokens"] / context_length * 100, 1), 100.0,
                )
        return metrics


def _offered(upfront, catalog, conversation, surface, skills, invoked_tools):
    schemas = tools.offered_schemas(
        upfront, catalog, [*invoked_tools, *tools.loaded_names(conversation)], surface,
        skills=skills is not None,
    )
    return schemas, frozenset(tools.schema_name(schema) for schema in schemas)


class _Ask:
    """One ask inside the turn: emit its card, wait for the answer, emit the outcome."""

    def __init__(self, turn: ChatAgentTurn, card: dict) -> None:
        self.turn = turn
        self.card = card
        self.reply: Optional[asks.Answer] = None

    async def events(self) -> AsyncIterator[str]:
        yield _event({"type": "ask_user", "data": self.card})
        if self.turn.interactive:
            self.reply = await asks.wait(self.turn.session_id, self.turn.owner, self.card)
        else:
            self.reply = asks.NO_INTERACTIVE_USER
        yield _event(asks.resolution_event(self.card, self.reply))


def _gate(
    turn: ChatAgentTurn, call: Dict[str, Any], block: Any, layers: permissions.DenyLayers,
    offered: frozenset, rules: permissions.Rules, provenance: permissions.Provenance,
    auto: classifier.AutoMode,
) -> permissions.Decision:
    """Admit only a tool this turn offered, then decide on the exact block the
    executor will run (the converter may rename it, e.g. a URL read_file)."""
    if call["name"] not in offered:
        return permissions.Decision(
            permissions.DENY, f"Tool '{call['name']}' is not available in this chat.",
        )
    if block is None:
        return permissions.Decision(
            permissions.INVALID, f"{call['name']}: the arguments are not valid for this tool.",
        )
    # Read per call: "Allow for this chat" earlier in this turn applies at once.
    session = [permissions.rule_from_record(r) for r in store.session_rules(turn.owner, turn.session_id)]
    return permissions.decide(
        layers, block.tool_type, block.content, rules=rules, session=session, provenance=provenance,
        owner=turn.owner, workspace=turn.toggles.workspace, auto_mode=auto.on,
    )


def _permission_decision(
    turn: ChatAgentTurn, rule: Optional[permissions.Rule], reply: Optional[asks.Answer],
) -> permissions.Decision:
    if reply is None:
        return permissions.Decision(permissions.DENY, ASK_EXPIRED_REASON)
    if reply.by == asks.BY_RUNTIME:
        return permissions.Decision(permissions.DENY, reply.text)
    if reply.decision == asks.DENY:
        reason = "Denied by the user."
        return permissions.Decision(permissions.DENY, f"{reason} Reason: {reply.text}" if reply.text else reason)
    if reply.decision == asks.CHAT and rule is not None:
        store.add_session_rule(turn.owner, turn.session_id, rule.record())
    return permissions.Decision(permissions.ALLOW)


def _join_steers(
    record: transcript.TurnRecord, conversation: List[Dict[str, Any]], auto: classifier.AutoMode, texts: List[str],
) -> None:
    record.add_steers(texts)
    conversation.extend({"role": "user", "content": text} for text in texts)
    # Auto mode judges by every message the user sent (spec #44, Auto mode).
    auto.user_messages.extend(texts)


def _question_result(reply: Optional[asks.Answer]) -> str:
    if reply is None:
        return "The user did not answer in time."
    if reply.by == asks.BY_RUNTIME:
        return reply.text
    if reply.decision == asks.DENY:
        return "The user declined to answer."
    return f"The user answered: {reply.text}"


def _tool_output_event(outcome: tools.ToolOutcome, *, attempted: bool) -> dict:
    return {
        "type": "tool_output", "tool": outcome.tool, "command": outcome.command,
        "output": outcome.model_text, "exit_code": outcome.exit_code,
        "execution_attempted": attempted, "blocked": outcome.status == tools.STATUS_DENIED,
        "call_id": outcome.call_id,
    }
