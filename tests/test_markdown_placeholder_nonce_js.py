"""Real-Chromium regression: forged stash placeholders stay literal text."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _playwright_installed():
    if shutil.which("node") is None:
        return False
    probe = subprocess.run(
        ["node", "-e", "require.resolve('playwright')"], cwd=_REPO,
        capture_output=True, text=True, timeout=15,
    )
    return probe.returncode == 0


def test_forged_placeholder_in_attribute_does_not_break_out():
    # The pytest Node harness stubs `document`, so only a real DOM shows the
    # attribute the restore step would have broken out of.
    if not _playwright_installed():
        pytest.skip("playwright package not installed (npm ci)")
    result = subprocess.run(
        ["node", "tests/markdown_placeholder_nonce_browser.cjs"], cwd=_REPO,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"flagged": []}
