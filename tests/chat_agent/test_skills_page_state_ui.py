"""Skills page shows the backend `state` and offers Publish on draft and
awaiting-audit cards (#49). Pins the static contract of static/js/skills.js
the way tests/test_skills_saved_turns_ui.py does."""

import re
import subprocess
from pathlib import Path

from tests.helpers.stylesheets import app_css


ROOT = Path(__file__).resolve().parents[2]
SKILLS_JS = (ROOT / "static/js/skills.js").read_text(encoding="utf-8")
METRICS_JS = (ROOT / "static/js/skillsMetrics.js").read_text(encoding="utf-8")
STYLE = app_css()


def _fn(src: str, name: str) -> str:
    start = src.index(f"function {name}(")
    end = src.index("\nfunction ", start + 1) if "\nfunction " in src[start + 1:] else len(src)
    return src[start:end]


def test_card_status_comes_only_from_the_backend_state():
    pill = _fn(SKILLS_JS, "_statusPill")
    assert "sk.state" in pill
    assert "sk.status" not in pill
    assert 'data-status="awaiting_audit"' in pill
    assert ".skill-status-awaiting" in STYLE
    # The frontend no longer judges approval from verdict + confidence.
    for sym in ("skillIsApproved(sk, _skillApprovalThreshold)", "skillNeedsReview("):
        assert sym not in SKILLS_JS
    assert "skillNeedsReview" not in METRICS_JS
    assert re.search(r"card\.dataset\.skillStatus = sk\.state", SKILLS_JS)
    assert "const isPublished = (sk.state === 'published');" in SKILLS_JS


def test_audit_and_approval_badges_report_without_changing_the_status():
    audit = _fn(SKILLS_JS, "_auditModelPills")
    assert "audit_summary" in audit and "audited_at" in audit and "audit_by_teacher" in audit
    assert "skillNeedsReview" not in audit
    stats = _fn(SKILLS_JS, "_skillStatsHtml")
    assert "user_approved" in stats and "user_approval_at" in stats
    assert "approved by you" in stats and "approval lapsed" in stats


def test_publish_reports_a_refused_write_instead_of_a_success_toast():
    setter = _fn(SKILLS_JS, "_setSkillStatus")
    assert "res.ok" in setter and "data.detail" in setter
    bulk = _fn(SKILLS_JS, "_bulkApprove")
    assert "sk.state === 'published'" in bulk
    assert "showError" in bulk


def test_user_only_toggle_writes_disable_model_invocation():
    assert "JSON.stringify({ disable_model_invocation:" in SKILLS_JS
    assert "sk.disable_model_invocation" in SKILLS_JS
    assert "user-only" in SKILLS_JS


def test_metrics_module_reads_state_not_threshold():
    assert "approvalThreshold" not in METRICS_JS
    script = """
        import assert from 'node:assert/strict';
        import { skillsSummaryMetrics, filterSkillsByQuickFilter } from './static/js/skillsMetrics.js';
        const rows = [
          { name: 'ok', state: 'published', status: 'published' },
          { name: 'waiting', state: 'awaiting_audit', status: 'published', audit_verdict: 'pass', confidence: 0.99 },
          { name: 'drafty', state: 'draft', status: 'draft' },
        ];
        assert.equal(skillsSummaryMetrics(rows).approved, 1);
        assert.deepEqual(filterSkillsByQuickFilter(rows, 'approved').map(s => s.name), ['ok']);
        assert.deepEqual(filterSkillsByQuickFilter(rows, 'draft').map(s => s.name), ['waiting', 'drafty']);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
