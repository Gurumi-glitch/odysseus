"""Entry point of the chat_agent runtime: run one agent turn as SSE events.

The turn runs rounds until a reply has no tool calls or a shared cap is hit.
Each round's tool calls run one at a time in call order; errors and denials go
back to the model as tool results. The turn also stops at the round and
tool-call caps, when the repeat guard stops it, and after Finish (one more
round with no tools). Steering arrives in #63.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

# Relies on: _configured_model_tool_surface returns the per-model tool surface
# ("none" | "compact" | "full" | "") for this owner's endpoint row and model.
from src.agent_loop import _configured_model_tool_surface
from src.agent_tools import MAX_AGENT_ROUNDS
# Relies on: agent_runs.should_finish(session_id) is True only while that
# session's registered run is running and Finish was requested for it; a run
# that replaces it starts with the flag clear.
from src import agent_runs
from src.chat_agent import asks, permissions, prompt, repeat_guard, session_rules, tools, transcript
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
    # The route's run_post_response_tasks bound to this request (memory and
    # skill extraction, webhook, auto-name); the loop adds the turn's results.
    after_turn: Optional[Callable[..., None]] = None


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


def _round_record(end: RoundEnd) -> Dict[str, Any]:
    if end.usage is None:
        return {"blocks": end.blocks}
    return {"blocks": end.blocks, "usage": asdict(end.usage)}


async def run_chat_agent_turn(turn: ChatAgentTurn) -> AsyncGenerator[str, None]:
    yield _event({"type": "turn_mode", "mode": "agent", "auto_escalated": turn.auto_escalated})
    sess = turn.sess
    endpoint = Endpoint(sess.endpoint_url, sess.model, sess.headers)
    provider = provider_key(endpoint.url)
    record = transcript.TurnRecord(provider=provider, model=endpoint.model)
    writer = transcript.TurnWriter(sess, turn.session_id, record, incognito=turn.toggles.incognito)
    contexts = []

    async def build_context(**flags):
        # Keeps the context so the row projects what the route resolved.
        contexts.append(await turn.build_context(**flags))
        return contexts[-1]

    streamed: List[str] = []

    def hold_ask(call: Dict[str, Any], tool: str, command: str, card: dict) -> None:
        # The card is on the row before anyone can answer it, so a restart
        # while it waits can mark it expired.
        record.pending_ask = {
            "round": len(record.rounds), "model": endpoint.model, "tool": tool, "command": command,
            "status": "pending", "call_id": call["call_id"], "ask_user": card,
        }
        writer.checkpoint()

    try:
        caps = read_caps()
        layers = permissions.build_deny_layers(
            owner=turn.owner,
            toggles=turn.toggles,
            privileges=turn.privileges,
            delegated_credential=turn.delegated_credential,
            active_email_open=turn.active_email_open,
            global_disabled_tools=settings.get_setting("disabled_tools", []),
        )
        surface = _configured_model_tool_surface(
            endpoint.url, endpoint.model, turn.owner, headers=endpoint.headers,
        )
        tool_schemas = tools.upfront_schemas(layers.denied_names, surface)
        offered = frozenset(tools.schema_name(schema) for schema in tool_schemas)
        unavailable = [name for name in tools.upfront_tool_names() if name in layers.disabled]
        conversation = await prompt.build_messages(
            build_context,
            [tools.unavailable_line(unavailable)] if tool_schemas else [],
            owner=turn.owner,
            toggles=turn.toggles,
        )
        scope = tools.ExecutionScope(
            owner=turn.owner,
            session_id=turn.session_id,
            workspace=turn.toggles.workspace,
            client_runtime_context=turn.client_runtime_context,
            denied_names=layers.denied_names,
        )
        record.context = _context_projections(contexts[-1])
        tool_calls_used = 0
        guard = repeat_guard.RepeatGuard()
        round_tools = tool_schemas
        while True:
            end = None
            streamed.clear()
            async for event in _stream_round(turn, provider, endpoint, conversation, round_tools):
                if isinstance(event, ProviderError):
                    _finish_failed_turn(writer)
                    yield _error_event(event.message, event.status)
                    yield "data: [DONE]\n\n"
                    return
                if isinstance(event, TextDelta):
                    streamed.append(event.text)
                    yield _event({"delta": event.text})
                elif isinstance(event, RoundEnd):
                    end = event
            streamed.clear()
            round_record = _round_record(end)
            record.rounds.append(round_record)
            writer.checkpoint()
            calls = [b for b in end.blocks if b["type"] == "tool_call"]
            if not calls or record.stop_reason == transcript.STOP_INTERRUPTED:
                # Finish's tool-less round is the turn's last.
                break
            conversation.append({"role": "assistant", "blocks": end.blocks})
            results = round_record["results"] = []
            for call in calls:
                block = function_call_to_tool_block(call["name"], call["arguments"])
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
                        decision = _gate(call, block, layers, offered)
                if decision.outcome == permissions.ASK and session_rules.allows(
                    turn.owner, turn.session_id, block.tool_type, block.content,
                ):
                    decision = permissions.Decision(permissions.ALLOW)
                if decision.outcome == permissions.ASK:
                    rule = session_rules.rule_for(block.tool_type, block.content)
                    ask = _Ask(turn, asks.permission_card(
                        turn.session_id, block.tool_type, block.content, rule.label,
                    ))
                    hold_ask(call, block.tool_type, block.content, ask.card)
                    async for event in ask.events():
                        yield event
                    if ask.reply is None:
                        record.stop_reason = transcript.STOP_ASK_EXPIRED
                    decision = _permission_decision(turn, rule, ask.reply)
                if decision.outcome == permissions.ALLOW:
                    yield _event({
                        "type": "tool_start", "tool": block.tool_type, "command": block.content,
                        "full_command": block.content, "round": len(record.rounds),
                        "call_id": call["call_id"],
                    })
                    outcome = await tools.execute(call["call_id"], block, scope)
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
                yield _event(_tool_output_event(outcome, attempted=decision.outcome == permissions.ALLOW))
                tool_event = outcome.tool_event(len(record.rounds), endpoint.model)
                if ask is not None:
                    tool_event["ask_user"] = asks.resolved_card(ask.card, ask.reply)
                record.pending_ask = None
                record.tool_events.append(tool_event)
                envelope = outcome.envelope()
                if reminder:
                    # The reminder is for the model only; the tool card keeps the output.
                    envelope["content"] += reminder
                results.append(envelope)
                conversation.append({"role": "tool", **envelope})
                # Per call, not per round: a restart must not mark a call
                # that already ran as cancelled (it would replay as not run).
                writer.checkpoint()
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
        message_id = writer.finish()
        _after_turn(turn, record, contexts[-1], message_id, tool_calls_used)
    except (asyncio.CancelledError, GeneratorExit):
        # Stop: keep the completed blocks and the text streamed so far; partial
        # reasoning is dropped and unfinished calls get cancelled results.
        if streamed:
            record.rounds.append({"blocks": [{"type": "text", "text": "".join(streamed)}]})
        if record.rounds:
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
) -> None:
    """Post-response tasks and the SFT trace, once per completed turn."""
    content = record.content()
    if turn.after_turn is not None:
        turn.after_turn(
            full_response=content, last_metrics=None, uprefs=ctx.uprefs,
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


class _Ask:
    """One ask inside the turn: emit its card, wait for the answer, emit the outcome."""

    def __init__(self, turn: ChatAgentTurn, card: dict) -> None:
        self.turn = turn
        self.card = card
        self.reply: Optional[asks.Answer] = None

    async def events(self) -> AsyncIterator[str]:
        yield _event({"type": "ask_user", "data": self.card})
        self.reply = await asks.wait(self.turn.session_id, self.turn.owner, self.card)
        yield _event(asks.resolution_event(self.card, self.reply))


def _gate(
    call: Dict[str, Any], block: Any, layers: permissions.DenyLayers, offered: frozenset,
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
    return permissions.decide(layers, block.tool_type, block.content)


def _permission_decision(
    turn: ChatAgentTurn, rule: session_rules.SessionRule, reply: Optional[asks.Answer],
) -> permissions.Decision:
    if reply is None:
        return permissions.Decision(permissions.DENY, ASK_EXPIRED_REASON)
    if reply.decision == asks.DENY:
        reason = "Denied by the user."
        return permissions.Decision(permissions.DENY, f"{reason} Reason: {reply.text}" if reply.text else reason)
    if reply.decision == asks.CHAT:
        session_rules.grant(turn.owner, turn.session_id, rule)
    return permissions.Decision(permissions.ALLOW)


def _question_result(reply: Optional[asks.Answer]) -> str:
    if reply is None:
        return "The user did not answer in time."
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
