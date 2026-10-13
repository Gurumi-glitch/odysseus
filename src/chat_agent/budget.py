"""Fit the new runtime's request to a measured context window (spec #89).

Two decisions, both only when the window was measured (a DEFAULT_CONTEXT
fallback proves nothing about the model): the tool surface for the turn, and
whether a request may be sent at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Relies on: transcript.over_threshold(tokens, window) is True when tokens
# reach auto_compact_threshold_percent of a non-zero window; text_view gives
# the role/content view the estimates read.
from src.chat_agent import transcript
# Relies on: estimate_tool_schema_tokens estimates the native tool definitions
# alone; estimate_request_tokens adds them to the role/content messages and an
# image reserve, the same estimate the output-budget clamp uses.
from src.generation_budget import estimate_request_tokens, estimate_tool_schema_tokens

# Tool schemas are the one fixed cost sent again every round; past this share
# of the window they leave too little room for the conversation (#82).
TOOL_SHARE_PERCENT = 25
COMPACT = "compact"
OVERFLOW_MESSAGE = "request ~{tokens} tokens exceeds the {window}-token window"


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
