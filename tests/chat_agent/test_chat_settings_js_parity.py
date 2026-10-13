"""The frontend's per-chat setting options mirror store.CHAT_SETTINGS (spec #89):
a value the store adds, drops or reorders must change the dropdowns too."""

import re
from pathlib import Path

import pytest

from src.chat_agent import store

SOURCE = (Path(__file__).resolve().parents[2] / "static" / "js" / "chatAgentSettings.js").read_text(encoding="utf-8")


def _option_values(constant):
    block = re.search(rf"export const {constant} = \[(.*?)\];", SOURCE, re.DOTALL)
    assert block, f"{constant} not found"
    return tuple(re.findall(r"value: '([a-z_]+)'", block.group(1)))


@pytest.mark.parametrize(
    ("constant", "setting"),
    [("OVERFLOW_OPTIONS", store.CONTEXT_OVERFLOW), ("THINKING_LEVEL_OPTIONS", store.THINKING_LEVEL)],
)
def test_js_options_are_the_store_values_in_order(constant, setting):
    assert _option_values(constant) == store.CHAT_SETTINGS[setting]
