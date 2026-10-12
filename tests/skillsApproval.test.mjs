import { test } from 'node:test';
import assert from 'node:assert/strict';
import { skillsSummaryMetrics, filterSkillsByQuickFilter, skillIsApproved, skillIsProven } from '../static/js/skillsMetrics.js';

// The backend's `state` (GET /api/skills) is the only status the page shows (#49).
// A published-but-unaudited skill is `awaiting_audit` there, whatever its
// confidence or verdict say.
test('approval reads the backend state, not status, verdict or confidence', () => {
  assert.equal(skillIsApproved({ state: 'published', status: 'published' }), true);
  assert.equal(skillIsApproved({ state: 'awaiting_audit', status: 'published', audit_verdict: 'pass', confidence: 0.99 }), false);
  assert.equal(skillIsApproved({ state: 'draft', status: 'draft', audit_verdict: 'pass', confidence: 0.99 }), false);
  assert.equal(skillIsApproved({ state: 'binned', status: 'binned' }), false);
  assert.equal(skillIsApproved({ status: 'published', audit_verdict: 'pass', confidence: 0.99 }), false);
});

test('summary and quick filters count published states only as approved', () => {
  const rows = [
    { name: 'ok', state: 'published', status: 'published', saved_turns: 2 },
    { name: 'waiting', state: 'awaiting_audit', status: 'published', audit_verdict: 'pass', confidence: 0.99 },
    { name: 'drafty', state: 'draft', status: 'draft' },
    { name: 'bad', state: 'binned', status: 'binned' },
  ];
  assert.equal(skillsSummaryMetrics(rows).approved, 1);
  assert.deepEqual(filterSkillsByQuickFilter(rows, 'approved').map(s => s.name), ['ok']);
  assert.deepEqual(filterSkillsByQuickFilter(rows, 'draft').map(s => s.name), ['waiting', 'drafty', 'bad']);
});

test('proven keeps the efficiency evidence on top of the backend state', () => {
  assert.equal(skillIsProven({ state: 'published', saved_turns: 2 }), true);
  assert.equal(skillIsProven({ state: 'published', baseline_verdict: 'worse' }), false);
  assert.equal(skillIsProven({ state: 'awaiting_audit', saved_turns: 2 }), false);
});

test('built-ins and drafts have distinct filters and archived skills remain recoverable', () => {
  const rows = [{ name: 'shipped', source: 'builtin', state: 'published', status: 'published' }, { name: 'bad', state: 'binned', status: 'binned' }];
  assert.deepEqual(filterSkillsByQuickFilter(rows, 'builtin').map(s => s.name), ['shipped']);
  assert.deepEqual(filterSkillsByQuickFilter(rows, 'draft').map(s => s.name), ['bad']);
});
