"""Entry point of the chat_agent runtime: run one agent turn as SSE events.

This ticket's loop is one round with no tools. Rounds, caps, the repeat
guard, steering and finish arrive in later tickets (#50, #54, #63).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

from src.chat_agent import prompt, transcript
from src.chat_agent.providers import chat_completions
from src.chat_agent.routing import ComposerToggles
from src.llm_core import _detect_provider

logger = logging.getLogger(__name__)


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


def _event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _error_event(message: str, status: Optional[int] = None) -> str:
    return f"event: error\ndata: {json.dumps({'error': message, 'status': status})}\n\n"


async def run_chat_agent_turn(turn: ChatAgentTurn) -> AsyncGenerator[str, None]:
    yield _event({"type": "turn_mode", "mode": "agent", "auto_escalated": turn.auto_escalated})
    sess = turn.sess
    endpoint = chat_completions.Endpoint(sess.endpoint_url, sess.model, sess.headers)
    record = transcript.TurnRecord(provider=_detect_provider(endpoint.url), model=endpoint.model)
    try:
        messages = await prompt.build_messages(
            turn.build_context, owner=turn.owner, toggles=turn.toggles,
        )
        text = ""
        async for event in chat_completions.stream_round(
            endpoint, messages, temperature=turn.temperature, max_tokens=turn.max_tokens,
        ):
            if isinstance(event, chat_completions.ProviderError):
                yield _error_event(event.message, event.status)
                yield "data: [DONE]\n\n"
                return
            text += event.text
            yield _event({"delta": event.text})
        record.round_texts.append(text)
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
