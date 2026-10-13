"""Every reference to static/js/skills.js carries one cache-busting version.

The browser keys ES modules by full URL, so two `?v=` values load two copies
of the module, each with its own module-level state. A refresh started from
Memory or a chat link then updates a copy the Skills page isn't showing.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "static"

# A quoted URL/specifier ending in skills.js with an optional ?v= token:
# <script src>, the sw.js precache list, and import('./skills.js?v=...').
_REF = re.compile(r"""['"][^'"\s]*?/skills\.js(?:\?v=([^'"\s]+))?['"]""")


def _skills_refs() -> list[tuple[str, str | None]]:
    refs = []
    for path in sorted(STATIC.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in {".js", ".html"} or "/vendor/" in rel:
            continue
        for m in _REF.finditer(path.read_text(encoding="utf-8", errors="ignore")):
            refs.append((rel, m.group(1)))
    return refs


def test_every_skills_module_reference_shares_one_version():
    refs = _skills_refs()
    # Page tag, sw.js precache, chatRenderer.js, memory.js: guards the scan
    # from passing vacuously if the regex stops matching.
    assert {rel for rel, _ in refs} >= {
        "static/index.html",
        "static/sw.js",
        "static/js/chatRenderer.js",
        "static/js/memory.js",
    }
    versions = {}
    for rel, version in refs:
        versions.setdefault(version, []).append(rel)
    assert len(versions) == 1 and None not in versions, (
        f"skills.js is referenced with different versions: {versions}"
    )
