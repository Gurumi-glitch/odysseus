# Copyright 2025 Moonshot AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for Odysseus (2026): ported from Kimi CLI
# src/kimi_cli/soul/toolset.py (MoonshotAI/kimi-cli @ 9ab1286b). Kept: the
# reminder texts, the 3/5/8/12 ladder and the canonical-argument call key.
# Changed: the streak counts every consecutive identical call in call order
# (no "seen in an earlier step" condition, no same-step dedup), lives for one
# turn, and is a standalone class instead of KimiToolset state; telemetry
# removed. See NOTICE in this package.
"""Repeat guard: escalating reminders for identical consecutive tool calls.

A call is identical when the tool name and its canonical arguments (JSON with
sorted keys) match the previous call. At 3, 5 and 8 in a row the model gets a
reminder appended to that call's result; at 12 the loop asks whether to stop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Tuple

REMINDER_1_STREAK = 3
REMINDER_2_STREAK = 5
REMINDER_3_STREAK = 8
STOP_STREAK = 12

_REMINDER_TEXT_1 = (
    "\n\n<system-reminder>\n"
    "You are repeating the exact same tool call with identical parameters."
    " Please carefully analyze the previous result. If the task is not yet complete,"
    " try a different method or parameters instead of repeating the same call."
    "\n</system-reminder>"
)


def _reminder_text_2(tool_name: str, repeat_count: int, canonical_args: str) -> str:
    return (
        "\n\n<system-reminder>\n"
        "You have repeatedly called the same tool with identical parameters many times.\n"
        "Repeated tool call detected:\n"
        f"- tool: {tool_name}\n"
        f"- repeated_times: {repeat_count}\n"
        f"- arguments: {canonical_args}\n"
        "The previous repeated calls did not make progress. Do not call this exact same tool "
        "with the exact same arguments again.\n"
        "Carefully inspect the latest tool result and choose a different next action, "
        "different parameters, or finish the task if enough evidence has been gathered."
        "\n</system-reminder>"
    )


_REMINDER_TEXT_3 = (
    "\n\n<system-reminder>\n"
    "You are stuck in a dead end and have repeatedly made the same function call without "
    "progress.\n"
    "Stop all function calls immediately. Do not call any tool in your next response.\n"
    "In analysis, review the current execution state and identify why progress is blocked.\n"
    "Then return a text-only summary to the user that reports the current problem, what has "
    "already been tried, and what information or decision is needed next."
    "\n</system-reminder>"
)


@dataclass(frozen=True)
class RepeatSignal:
    streak: int
    # Text appended to the call's model-facing result, or None.
    reminder: Optional[str]
    # True at STOP_STREAK: the loop must ask whether to stop the turn.
    stop: bool


def _sort_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_sort_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _sort_json_value(value[key]) for key in sorted(value)}
    return value


def _canonical_arguments(arguments: str) -> str:
    try:
        parsed = json.loads(arguments, strict=False)
    except json.JSONDecodeError:
        return arguments
    return json.dumps(_sort_json_value(parsed), ensure_ascii=False, separators=(",", ":"))


class RepeatGuard:
    """Tracks one turn's consecutive identical calls."""

    def __init__(self) -> None:
        self._key: Optional[Tuple[str, str]] = None
        self._streak = 0

    def observe(self, tool_name: str, arguments: str) -> RepeatSignal:
        """Count this call (name + raw JSON arguments text) and say what to do."""
        canonical = _canonical_arguments(arguments)
        key = (tool_name, canonical)
        self._streak = self._streak + 1 if key == self._key else 1
        self._key = key
        streak = self._streak
        if streak >= STOP_STREAK:
            return RepeatSignal(streak, _REMINDER_TEXT_3, stop=True)
        if streak >= REMINDER_3_STREAK:
            return RepeatSignal(streak, _REMINDER_TEXT_3, stop=False)
        if streak >= REMINDER_2_STREAK:
            return RepeatSignal(streak, _reminder_text_2(tool_name, streak, canonical), stop=False)
        if streak >= REMINDER_1_STREAK:
            return RepeatSignal(streak, _REMINDER_TEXT_1, stop=False)
        return RepeatSignal(streak, None, stop=False)
