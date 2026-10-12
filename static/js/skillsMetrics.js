export function auditNumber(sk, key) {
  if (!sk) return null;
  const verdict = sk.verdict && typeof sk.verdict === 'object' ? sk.verdict : null;
  const candidates = [sk[key], verdict && verdict[key]];
  for (const value of candidates) {
    if (value === null || value === undefined || value === '') continue;
    const n = Number(value);
    if (Number.isFinite(n)) return Math.trunc(n);
  }
  return null;
}

export function skillSavedTurnsValue(sk) {
  const turns = auditNumber(sk, 'saved_turns');
  return turns === null ? 0 : turns;
}

export function necessityKind(sk) {
  const nec = sk && sk.necessity;
  // The recommended keeper is the duplicate group winner, not a review item.
  // Lower-priority members remain marked as duplicates for bin/review flows.
  if (sk && sk._duplicateGroup && !sk._duplicateKeep) return 'duplicate';
  if (sk && sk._duplicateGroup && sk._duplicateKeep) {
    const reason = String(nec?.reason || '').toLowerCase();
    const redundant = (nec?.redundant_with || []).filter(Boolean);
    if (redundant.length || /duplicat|redundan|overlap|same skill|same procedure/.test(reason)) return null;
  }
  if (!nec || nec.necessary !== false) return null;
  const reason = String(nec.reason || '').toLowerCase();
  const redundant = (nec.redundant_with || []).filter(Boolean);
  if (redundant.length || /duplicat|redundan|overlap|same skill|same procedure/.test(reason)) return 'duplicate';
  if (/trivial|generic|capable assistant|without a saved|not need|unnecessary/.test(reason)) return 'trivial';
  return 'irrelevant';
}

export function skillNeedsAudit(sk) {
  if (sk && sk.source === 'builtin') return false;
  return !(sk && sk.audit_verdict);
}

export function skillIsProven(sk) {
  if (!skillIsApproved(sk)) return false;
  if (sk.source === 'builtin') return true;
  const baseline = String(sk.baseline_verdict || '').toLowerCase();
  const savedTurns = skillSavedTurnsValue(sk);
  const savedCallsRaw = auditNumber(sk, 'saved_tool_calls');
  const savedCalls = savedCallsRaw === null ? 0 : savedCallsRaw;
  const usefulness = Number(sk.usefulness ?? 0);
  const necessity = necessityKind(sk);
  if (necessity === 'duplicate' || necessity === 'irrelevant') return false;
  if (baseline === 'worse') return false;
  return !(baseline === 'same' && usefulness < 0.5 && savedTurns <= 0 && savedCalls <= 0);
}

// `state` is computed by GET /api/skills from the one usability predicate
// (services.memory.skills.is_usable, #49); the page never re-derives it.
export function skillIsApproved(sk) {
  return !!sk && sk.state === 'published';
}

export function skillsSummaryMetrics(skills) {
  const list = Array.isArray(skills) ? skills : [];
  const total = list.length;
  const approved = list.filter(skillIsApproved).length;
  const proven = list.filter(skillIsProven).length;
  const needsAudit = list.filter(skillNeedsAudit).length;
  const audited = list.filter(sk => !!sk?.audit_verdict).length;
  const awaitingAudit = list.filter(sk => sk?.state === 'awaiting_audit').length;
  const drafts = list.filter(sk => sk?.state === 'draft').length;
  const binned = list.filter(sk => sk?.state === 'binned').length;
  const savedTurns = list.reduce((sum, sk) => sum + Math.max(0, skillSavedTurnsValue(sk)), 0);
  return { total, approved, proven, needsAudit, audited, awaitingAudit, drafts, binned, savedTurns };
}

export function filterSkillsByQuickFilter(skills, quickFilter) {
  const list = Array.isArray(skills) ? skills : [];
  if (quickFilter === 'all') return list;
  if (quickFilter === 'approved') return list.filter(sk => sk.source !== 'builtin' && skillIsApproved(sk));
  if (quickFilter === 'builtin') return list.filter(sk => sk.source === 'builtin');
  if (quickFilter === 'draft') return list.filter(sk => sk.source !== 'builtin' && !skillIsApproved(sk));
  if (quickFilter === 'proven') return list.filter(skillIsProven);
  if (quickFilter === 'audited') return list.filter(sk => !!sk?.audit_verdict);
  if (quickFilter === 'saved') return list.filter(sk => skillSavedTurnsValue(sk) > 0);
  if (quickFilter === 'unaudited') return list.filter(skillNeedsAudit);
  if (quickFilter === 'awaiting') return list.filter(sk => sk?.state === 'awaiting_audit');
  if (quickFilter === 'drafts') return list.filter(sk => sk?.state === 'draft');
  if (quickFilter === 'binned') return list.filter(sk => sk?.state === 'binned');
  return list;
}
