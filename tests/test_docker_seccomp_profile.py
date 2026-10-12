"""Guard for fork #47: the Docker seccomp profile that lets bubblewrap run.

The profile must stay Docker's stock default plus exactly the bwrap rules. Any
other edit (a wider flag set, clone3/setns), or a Docker default refresh
without re-measuring, shows up here.
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
CLONE_NEWNS, CLONE_NEWUSER, CLONE_NEWPID = 0x20000, 0x10000000, 0x20000000
DOCKER_CLONE_NS_MASK = 0x7E020000
# Flags from strace of the bwrap argv containment builds; each rule was shown
# necessary by removing it (bwrap then refuses and bash/python are refused).
BWRAP_RULES = [
    {"names": ["clone"], "args": [{"index": 0, "value": DOCKER_CLONE_NS_MASK,
                                    "valueTwo": CLONE_NEWNS | CLONE_NEWUSER | CLONE_NEWPID,
                                    "op": "SCMP_CMP_MASKED_EQ"}]},
    {"names": ["unshare"], "args": [{"index": 0, "value": CLONE_NEWUSER, "op": "SCMP_CMP_EQ"}]},
    {"names": ["mount", "pivot_root", "umount2"]},
]


def _profile():
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_profile_is_docker_default_plus_only_the_bwrap_rules():
    profile = _profile()
    added = profile["syscalls"][-len(BWRAP_RULES):]
    del profile["syscalls"][-len(BWRAP_RULES):]

    for rule, expected in zip(added, BWRAP_RULES):
        assert rule["action"] == "SCMP_ACT_ALLOW"
        assert rule["names"] == expected["names"]
        assert rule.get("args") == expected.get("args")
        assert "includes" not in rule, "rules must not depend on added capabilities"
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
