"""Browser execution of the chat_agent round rendering (spec #44 Transcript, #65)."""
import subprocess
from pathlib import Path


def test_chat_agent_rounds_browser_suite():
    result = subprocess.run(
        ["node", "--test", "tests/chatAgentRoundsRendering.test.mjs"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
