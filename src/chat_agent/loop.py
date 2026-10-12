"""Entry point of the chat_agent runtime: run one agent turn as SSE events.

This ticket's loop is one round with no tools. Rounds, caps, the repeat
guard, steering and finish arrive in later tickets (#50, #54, #63).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing
from dataclasses import asdict, dataclass
from typing import (
    Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Dict, FrozenSet, List, Optional, Sequence, Union,
)

from src.chat_agent import prompt, transcript
from src.chat_agent.providers import (
    XAI,
    Endpoint,
    ProviderError,
    RoundEnd,
    RoundEvent,
    TextDelta,
    chat_completions,
    provider_key,
    tool_output_text,
    xai_responses,
)
from src.chat_agent.routing import ComposerToggles
# Relied on: with persist=False maybe_compact never writes session history.
# When it compacts it returns the system messages, then one summary message,
# then the kept non-system messages; compaction_state["split_point"] is the
# index into the non-system messages where the kept half starts.
from src.context_compactor import maybe_compact
from src.llm_core import LLMConfig

logger = logging.getLogger(__name__)

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
    # The foreground candidate list, the session's own route first.
    candidates: Sequence[Candidate]
    # The foreground policy's eligible_statuses.
    fallback_statuses: FrozenSet[int]
    # Moves the session's stored model and endpoint (fallback is sticky).
    stick_route: Callable[[Candidate], None]


def _event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _error_event(message: str, status: Optional[int] = None) -> str:
    return f"event: error\ndata: {json.dumps({'error': message, 'status': status})}\n\n"


def _stream_round(
    turn: ChatAgentTurn, provider: str, endpoint: Endpoint, conversation: List[Dict[str, Any]],
) -> AsyncIterator[RoundEvent]:
    if provider == XAI:
        return xai_responses.stream_round(
            endpoint, conversation, cache_key=turn.session_id,
            temperature=turn.temperature, max_tokens=turn.max_tokens,
        )
    return chat_completions.stream_round(
        endpoint, conversation, temperature=turn.temperature, max_tokens=turn.max_tokens,
    )


async def _round_with_fallback(
    turn: ChatAgentTurn, endpoint: Endpoint, conversation: List[Dict[str, Any]],
) -> AsyncIterator[Union[RoundEvent, Fallback]]:
    """One round on the turn's current candidate; a failure before any output
    with an eligible status moves to the next candidate (spec #44 Providers).
    A failure after output passes through: switching would repeat streamed text."""
    candidates = turn.candidates
    start = next(i for i, candidate in enumerate(candidates) if candidate.endpoint == endpoint)
    failures: List[Dict[str, Any]] = []
    reason = ""
    for index in range(start, len(candidates)):
        candidate = candidates[index]
        request = conversation if index == start else await _fit_window(turn, candidate.endpoint, conversation)
        retry_left = len(candidates) == 1
        while True:
            failure = None
            committed = False
            held: List[RoundEvent] = []
            provider = provider_key(candidate.endpoint.url)
            async with aclosing(_stream_round(turn, provider, candidate.endpoint, request)) as events:
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


async def _fit_window(
    turn: ChatAgentTurn, endpoint: Endpoint, conversation: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compact once when the history exceeds the fallback's window, with the
    existing compaction until #67. Request-local: the next turn's context
    builder compacts the stored history at the (now sticky) fallback window."""
    view = [{"role": message["role"], "content": _plain_text(message)} for message in conversation]
    state: Dict[str, Any] = {}
    compacted_view, _, compacted = await maybe_compact(
        turn.sess, endpoint.url, endpoint.model, view, endpoint.headers,
        owner=turn.owner, persist=False, compaction_state=state,
    )
    if not compacted:
        return conversation
    system = [message for message in conversation if message["role"] == "system"]
    rest = [message for message in conversation if message["role"] != "system"]
    split = state["split_point"]
    # A kept tool result needs its call: start the kept half at the call's round.
    while split > 0 and rest[split]["role"] == "tool":
        split -= 1
    return [*system, compacted_view[len(system)], *rest[split:]]


def _plain_text(message: Dict[str, Any]) -> Any:
    """What the legacy compactor can measure and summarize: rounds as text."""
    if "blocks" in message:
        return "\n".join(
            block["text"] if block["type"] == "text" else f"{block['name']}({block['arguments']})"
            for block in message["blocks"] if block["type"] in ("text", "tool_call")
        )
    if message["role"] == "tool":
        return tool_output_text(message)
    return message.get("content") or ""


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
    try:
        conversation = await prompt.build_messages(
            turn.build_context, owner=turn.owner, toggles=turn.toggles,
        )
        async for event in _round_with_fallback(turn, endpoint, conversation):
            if isinstance(event, Fallback):
                endpoint, conversation = event.candidate.endpoint, event.conversation
                record.provider, record.model = provider_key(endpoint.url), endpoint.model
                yield _event(event.event)
            elif isinstance(event, ProviderError):
                yield _error_event(event.message, event.status)
                yield "data: [DONE]\n\n"
                return
            if isinstance(event, TextDelta):
                yield _event({"delta": event.text})
            elif isinstance(event, RoundEnd):
                record.rounds.append(_round_record(event))
        message_id = transcript.save_turn(sess, turn.session_id, record, incognito=turn.toggles.incognito)
    except Exception:
        # A crash ends the turn visibly; it never falls back to legacy, which
        # could run side effects twice (ADR 0005).
        logger.exception("chat_agent turn failed for session %s", turn.session_id)
        yield _error_event("The chat agent runtime failed on this turn. See the server log.")
        yield "data: [DONE]\n\n"
        return
    if message_id:
        yield _event({"type": "message_saved", "id": message_id})
    yield "data: [DONE]\n\n"
