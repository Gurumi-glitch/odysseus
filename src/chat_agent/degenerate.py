"""The per-round guard that stops a model stuck repeating tokens (spec #89).

Answer and thinking deltas feed one instance per round. It trips on any of the
legacy guard's rules, or on a long run of output with no letter or digit
(``////`` and ``/ / /``), which the legacy guard does not see because it only
counts word tokens.
"""

from __future__ import annotations

import logging

# Relies on: _DegenerateStreamGuard.check returns None unless one of its rules
# fired (and logs the reason itself); its returned SSE text is not used here.
from src.llm_core import _DegenerateStreamGuard

logger = logging.getLogger(__name__)

# Far above markdown rules and box-drawing lines, far below the 16k-token
# `////` run that motivated it.
SYMBOL_RUN_LIMIT = 1000

MESSAGE = (
    "Stopped: {model} produced degenerate output (repeating tokens). "
    "Try lower temperature or reload the model."
)


class RoundGuard:
    def __init__(self, model: str):
        self._model = model
        self._legacy = _DegenerateStreamGuard(model)
        self._symbol_run = 0

    def tripped(self, text: str) -> bool:
        for char in text:
            # Whitespace neither counts nor resets, so `/ / /` is one run.
            if char.isspace():
                continue
            self._symbol_run = 0 if char.isalnum() else self._symbol_run + 1
            if self._symbol_run >= SYMBOL_RUN_LIMIT:
                logger.warning("[degenerate-stream] aborting model=%s reason=%d characters without a letter or digit",
                               self._model, self._symbol_run)
                return True
        return bool(self._legacy.check(text))
