from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_installs_bubblewrap_for_shell_containment():
    # src/containment.py selects its "bubblewrap" mechanism only when `bwrap`
    # is on PATH; without it, enforcing mode refuses every bash/python call.
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "bubblewrap" in dockerfile
