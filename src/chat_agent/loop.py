"""Entry point of the chat_agent runtime: run one agent turn as SSE events.

This ticket's loop is one round with no tools. Rounds, caps, the repeat
guard, steering and finish arrive in later tickets (#50, #54, #63).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Dict, List, Optional

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
    xai_responses,
)
from src.chat_agent.routing import ComposerToggles

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatAgentTurn:
    session_id: str
    sess: Any
    toggles: ComposerToggles
    auto_escalated: bool
    temperature: float
    max_tokens: int
    # The route's build_chat_context bound to this request; prompt picks the flags.
    build_context: Callable[..., Awaitable[Any]]


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
    try:
        conversation = await prompt.build_messages(turn.build_context)
        async for event in _stream_round(turn, provider, endpoint, conversation):
            if isinstance(event, ProviderError):
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
