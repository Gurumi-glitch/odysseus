"""A new chat before its first message (issue #103): the node suites, so CI runs them."""
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("suite", ["tests/chatPendingChatSettings.test.mjs", "tests/chatPendingChatTransition.test.mjs"])
def test_pending_chat_node_suite(suite):
    result = subprocess.run(
        ["node", "--test", suite],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
