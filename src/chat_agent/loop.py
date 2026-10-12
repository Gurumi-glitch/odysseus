"""Entry point of the chat_agent runtime: run one agent turn as SSE events.

The turn runs rounds until a reply has no tool calls. Each round's tool calls
run one at a time in call order; errors and denials go back to the model as
tool results. Caps, the repeat guard, steering and finish arrive in later
tickets (#54, #63).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Mapping, Optional

# Relies on: _configured_model_tool_surface returns the per-model tool surface
# ("none" | "compact" | "full" | "") for this owner's endpoint row and model.
from src.agent_loop import _configured_model_tool_surface
from src.chat_agent import permissions, prompt, tools, transcript
from src.chat_agent.providers import chat_completions
from src.chat_agent.routing import ComposerToggles
from src.llm_core import _detect_provider
from src.settings import get_setting
# Relies on: function_call_to_tool_block converts a native call into the
# executor's block (bare email names become mcp__email__*), or returns None
# when the arguments are unusable.
from src.tool_schemas import function_call_to_tool_block

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatAgentTurn:
    session_id: str
    sess: Any
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
        layers = permissions.build_deny_layers(
            owner=turn.owner,
            toggles=turn.toggles,
            privileges=turn.privileges,
            delegated_credential=turn.delegated_credential,
            active_email_open=turn.active_email_open,
            global_disabled_tools=get_setting("disabled_tools", []),
        )
        surface = _configured_model_tool_surface(
            endpoint.url, endpoint.model, turn.owner, headers=endpoint.headers,
        )
        tool_schemas = tools.upfront_schemas(layers.denied_names, surface)
        unavailable = [name for name in tools.upfront_tool_names() if name in layers.disabled]
        messages = await prompt.build_messages(
            turn.build_context, [tools.unavailable_line(unavailable)] if tool_schemas else [],
        )
        scope = tools.ExecutionScope(
            owner=turn.owner,
            session_id=turn.session_id,
            workspace=turn.toggles.workspace,
            client_runtime_context=turn.client_runtime_context,
            denied_names=layers.denied_names,
        )
        while True:
            round_ = transcript.Round()
            record.rounds.append(round_)
            async for event in chat_completions.stream_round(
                endpoint, messages, temperature=turn.temperature,
                max_tokens=turn.max_tokens, tools=tool_schemas,
            ):
                if isinstance(event, chat_completions.ProviderError):
                    yield _error_event(event.message, event.status)
                    yield "data: [DONE]\n\n"
                    return
                if isinstance(event, chat_completions.ToolCall):
                    round_.calls.append(event)
                    continue
                round_.text += event.text
                yield _event({"delta": event.text})
            if not round_.calls:
                break
            messages.append(chat_completions.assistant_message(round_.text, round_.calls))
            for call in round_.calls:
                block = function_call_to_tool_block(call.name, call.arguments)
                decision = _gate(call, block, layers)
                if decision.outcome == permissions.ALLOW:
                    yield _event({
                        "type": "tool_start", "tool": block.tool_type, "command": block.content,
                        "full_command": block.content, "round": len(record.rounds),
                        "call_id": call.call_id,
                    })
                    outcome = await tools.execute(call.call_id, block, scope)
                else:
                    outcome = tools.refused_outcome(call, block, decision)
                yield _event(_tool_output_event(outcome, attempted=decision.outcome == permissions.ALLOW))
                round_.outcomes.append(outcome)
                messages.append(chat_completions.tool_message(call.call_id, outcome.model_content()))
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


def _gate(call, block, layers) -> permissions.Decision:
    if block is None:
        return permissions.Decision(
            permissions.INVALID, f"{call.name}: the arguments are not valid for this tool.",
        )
    return permissions.decide(layers, block.tool_type, block.content)


def _tool_output_event(outcome: tools.ToolOutcome, *, attempted: bool) -> dict:
    return {
        "type": "tool_output", "tool": outcome.tool, "command": outcome.command,
        "output": outcome.model_text, "exit_code": outcome.exit_code,
        "execution_attempted": attempted, "blocked": outcome.status == tools.STATUS_DENIED,
        "call_id": outcome.call_id,
    }
