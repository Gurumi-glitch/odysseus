// Temperature / Max tokens chosen on a new chat before its first message.
// There is no session to save them on yet (issue #103): they are held here and
// written to the session that first message creates (sessions.js materializePendingSession).

let held = {};

// change: {temperature_override?, max_tokens_override?}; null means Default.
export function holdPendingGeneration(change) {
  held = { ...held, ...change };
}

export function pendingGeneration() {
  return { ...held };
}

// A new chat starts at Default.
export function dropPendingGeneration() {
  held = {};
}

// Posts only the held keys: the generation-settings route keeps every key a
// body leaves out, so the session's own thinking mode is untouched. -> ok.
export async function applyPendingGeneration(sid) {
  const change = held;
  held = {};
  if (!Object.keys(change).length) return true;
  try {
    const res = await fetch(`/api/session/${encodeURIComponent(sid)}/generation-settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(change),
    });
    return res.ok;
  } catch (_) {
    return false;
  }
}
