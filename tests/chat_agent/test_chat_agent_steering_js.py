"""Browser execution of the composer steer routing (spec #44 steering, #63)."""
import subprocess
from pathlib import Path


def test_chat_agent_steering_browser_suite():
    result = subprocess.run(
        ["node", "--test", "tests/chatAgentSteering.test.mjs"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
