"""Real-Chromium regression: autolinks never splice attributes into markup."""
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


def test_autolink_output_is_sanitized_with_the_tag_it_lands_in():
    # The pytest Node harness stubs `document`, so sanitizeAllowedHtml is a
    # no-op there; this is the only check that runs the real sanitizer.
    if not _playwright_installed():
        pytest.skip("playwright package not installed (npm ci)")
    result = subprocess.run(
        ["node", "tests/markdown_autolink_sanitizer_browser.cjs"], cwd=_REPO,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"flagged": []}
