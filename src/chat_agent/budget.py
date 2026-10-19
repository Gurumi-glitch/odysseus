"""Fit the new runtime's request to a measured context window (spec #89).

Two decisions, both only when the window was measured (a DEFAULT_CONTEXT
fallback proves nothing about the model): the tool surface for the turn, and
whether a request may be sent at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Relies on: transcript.over_threshold(tokens, window) is True when tokens
# reach auto_compact_threshold_percent of a non-zero window; text_view gives
# the role/content view the estimates read.
from src.chat_agent import store, transcript
# Relies on: estimate_tool_schema_tokens estimates the native tool definitions
# alone; estimate_request_tokens adds them to the role/content messages and an
# image reserve, the same estimate the output-budget clamp uses.
from src.generation_budget import estimate_request_tokens, estimate_tool_schema_tokens

# Tool schemas are the one fixed cost sent again every round; past this share
# of the window they leave too little room for the conversation (#82).
TOOL_SHARE_PERCENT = 25
COMPACT = "compact"
OVERFLOW_MESSAGE = "request ~{tokens} tokens exceeds the {window}-token window"
TRIMMED_MESSAGE = "dropped {count} messages to fit the window"


def tool_surface(configured: str, schemas: Sequence[Dict[str, Any]], window: int, measured: bool) -> str:
    """The configured surface, or ``compact`` when none is set and the full
    schemas take more than TOOL_SHARE_PERCENT of a measured window."""
    if configured or not measured:
        return configured
    if estimate_tool_schema_tokens(list(schemas)) * 100 > window * TOOL_SHARE_PERCENT:
        return COMPACT
    return configured


def overflow(
    conversation: List[Dict[str, Any]], tool_schemas: Sequence[Dict[str, Any]], window: int, measured: bool,
) -> Optional[str]:
    """The error for a request over the auto-compact threshold of a measured
    window, else None."""
    if not measured:
        return None
    tokens = estimate_request_tokens(transcript.text_view(conversation), list(tool_schemas))
    if transcript.over_threshold(tokens, window):
        return OVERFLOW_MESSAGE.format(tokens=tokens, window=window)
    return None


def gate(
    policy: str, first_request: bool, conversation: List[Dict[str, Any]], tool_schemas: Sequence[Dict[str, Any]],
    window: int, measured: bool, history_start: int, current: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    """The chat's Context overflow policy on one round's request: the request
    to send, how many messages it dropped, and the error that ends the turn
    instead. Auto checks only the turn's first request; the trimming policies
    drop whole messages from a copy, so the stored chat is never touched."""
    if policy == store.OVERFLOW_AUTO and not first_request:
        return conversation, 0, None
    refused = overflow(conversation, tool_schemas, window, measured)
    if refused is None or policy not in (store.OVERFLOW_ROLLING, store.OVERFLOW_TRUNCATE_MIDDLE):
        return conversation, 0, refused
    dropped = set()
    for unit in _droppable(conversation, history_start, current, policy == store.OVERFLOW_TRUNCATE_MIDDLE):
        dropped.update(unit)
        request = [message for index, message in enumerate(conversation) if index not in dropped]
        refused = overflow(request, tool_schemas, window, measured)
        if refused is None:
            return request, len(dropped), None
    return conversation, 0, refused


def _droppable(
    conversation: List[Dict[str, Any]], history_start: int, current: Dict[str, Any], keep_oldest_user: bool,
) -> List[List[int]]:
    """Message indexes in drop order, one list per unit: earlier turns, then
    this turn's tool rounds before the latest. A tool call goes with its
    results. System messages (prompt, instructions, compact summary), the
    turn's user message, its steers and the latest round never appear."""
    # Relies on: the loop's conversation holds ``current`` itself (by
    # identity) after the replayed history, as transcript.fit_request does.
    end = next(index for index, message in enumerate(conversation) if message is current)
    oldest_user = next(
        (index for index in range(history_start, end) if conversation[index]["role"] == "user"), None,
    ) if keep_oldest_user else None
    latest = max(
        (index for index in range(end + 1, len(conversation)) if conversation[index]["role"] == "assistant"),
        default=end,
    )
    units = []
    for start, stop in ((history_start, end), (end + 1, latest)):
        index = start
        while index < stop:
            following = index + 1
            while following < stop and conversation[following]["role"] == "tool":
                following += 1
            role = conversation[index]["role"]
            if role != "system" and index != oldest_user and (start == history_start or role == "assistant"):
                units.append(list(range(index, following)))
            index = following
    return units
