"""Guard for fork #47: the Docker seccomp profile that lets bubblewrap run.

The profile must stay Docker's stock default plus one extra rule. Any other
edit, or a Docker default refresh without re-measuring, shows up here.
"""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "docker" / "seccomp-odysseus.json"
COMPOSE = ROOT / "docker-compose.yml"

# sha256 of vendor/github.com/moby/profiles/seccomp/default.json at moby tag
# docker-v29.9.0 (moby/profiles/seccomp v0.2.4), byte-for-byte.
DOCKER_DEFAULT_SHA256 = "785b2429264afba4d594320337cb17f144f3c7d51585f9805eef72e28f4f9334"
# Measured by removing each candidate in turn (clone3 and setns were not needed).
BWRAP_SYSCALLS = ["clone", "mount", "pivot_root", "umount2", "unshare"]


def _profile():
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_profile_is_docker_default_plus_one_bwrap_rule():
    profile = _profile()
    extra = profile["syscalls"].pop()

    assert extra["names"] == BWRAP_SYSCALLS
    assert extra["action"] == "SCMP_ACT_ALLOW"
    assert set(extra) == {"names", "action", "comment"}, "the extra rule must be unconditional and cap-free"
    # moby ships default.json tab-indented without a trailing newline.
    base = json.dumps(profile, indent="\t").encode("utf-8")
    assert hashlib.sha256(base).hexdigest() == DOCKER_DEFAULT_SHA256


def test_compose_app_service_uses_profile():
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    opts = compose["services"]["odysseus"]["security_opt"]

    assert "seccomp=./docker/seccomp-odysseus.json" in opts
    assert (COMPOSE.parent / "docker" / "seccomp-odysseus.json").is_file()
    assert "systempaths=unconfined" in opts
    assert "cap_add" not in compose["services"]["odysseus"]
