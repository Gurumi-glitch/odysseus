"""Browser execution of the chat_agent ask card functions (spec #44 asks, #60)."""
import subprocess
from pathlib import Path


def test_chat_agent_ask_cards_browser_suite():
    result = subprocess.run(
        ["node", "--test", "tests/chatAgentAskCards.test.mjs"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
