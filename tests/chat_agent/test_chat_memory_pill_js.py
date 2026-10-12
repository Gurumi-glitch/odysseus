"""The memory pill survives a multi-round turn and a resume (footer target carry)."""
import subprocess
from pathlib import Path


def test_chat_memory_pill_footer_suite():
    result = subprocess.run(
        ["node", "--test", "tests/chatMemoryPillFooter.test.mjs"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
