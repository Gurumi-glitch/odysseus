"""Runs the slash autocomplete skill-refresh suite under pytest.

Behaviour lives in tests/slashAutocompleteSkillRefresh.test.mjs (node:test).
This wrapper only exists so the JS suite runs in the normal pytest job.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SUITE = Path(__file__).with_name("slashAutocompleteSkillRefresh.test.mjs")
_HAS_NODE = shutil.which("node") is not None


@pytest.mark.skipif(not _HAS_NODE, reason="node binary not on PATH")
def test_slash_autocomplete_skill_refresh_behavior():
    result = subprocess.run(
        ["node", "--experimental-vm-modules", "--test", str(_SUITE)],
        cwd=_REPO,
        capture_output=True,
        timeout=30,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"node --test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
