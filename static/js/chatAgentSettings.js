// Per-chat settings of the chat_agent runtime, shown in the chat context
// panel. The routes are owned by routes/chat_agent_routes.py.

// Values and order mirror store.CHAT_SETTINGS["context_overflow"]; the first is the default.
export const OVERFLOW_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: 'rolling_window', label: 'Rolling window' },
  { value: 'truncate_middle', label: 'Truncate middle' },
  { value: 'stop_at_limit', label: 'Stop at limit' },
];

async function getJson(url) {
  try {
    const res = await fetch(url, { credentials: 'same-origin' });
    return res.ok ? await res.json() : null;
  } catch (_) {
    return null;
  }
}

// -> {available, chat_runtime, context_overflow?, thinking_level?} or null.
export function fetchChatAgentSettings(sid) {
  return getJson(`/api/chat_agent/settings?session_id=${encodeURIComponent(sid)}`);
}

// -> {route: 'chat_agent'|'legacy', reason, thinking: {available, binary, levels}} or null.
// Shared with the thinking-level control (#101): one place for the fetch.
export function fetchChatAgentCapabilities(sid) {
  return getJson(`/api/chat_agent/capabilities/${encodeURIComponent(sid)}`);
}

// Returns the new value on success, null when the save was refused.
export async function saveChatAgentSetting(sid, name, value) {
  try {
    const res = await fetch('/api/chat_agent/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ session_id: sid, [name]: value }),
    });
    return res.ok ? value : null;
  } catch (_) {
    return null;
  }
}

// onChange(value) -> Promise<boolean>; false keeps the old value.
// update({value, legacy}) fills the row once the routes answered; until then
// (and on a legacy chat) the select is greyed, not hidden.
export function createOverflowRow({ doc = document, onChange }) {
  const el = doc.createElement('div');
  el.className = 'chat-context-toggle-row chat-context-overflow-row';
  const copy = doc.createElement('div');
  copy.className = 'chat-context-toggle-copy';
  const label = doc.createElement('span');
  label.textContent = 'Context overflow';
  copy.appendChild(label);

  const sel = doc.createElement('select');
  sel.className = 'chat-context-select';
  sel.setAttribute('aria-label', 'Context overflow for this chat');
  for (const { value, label: text } of OVERFLOW_OPTIONS) {
    const opt = doc.createElement('option');
    opt.value = value;
    opt.textContent = text;
    sel.appendChild(opt);
  }
  sel.value = OVERFLOW_OPTIONS[0].value;
  sel.disabled = true;
  el.className += ' disabled';

  let current = sel.value;
  sel.addEventListener('change', async () => {
    const next = sel.value;
    sel.disabled = true;
    const ok = await onChange(next);
    if (ok) current = next; else sel.value = current;
    sel.disabled = false;
  });

  el.appendChild(copy);
  el.appendChild(sel);

  const update = ({ value, legacy }) => {
    current = OVERFLOW_OPTIONS.some(o => o.value === value) ? value : OVERFLOW_OPTIONS[0].value;
    sel.value = current;
    sel.disabled = !!legacy;
    el.className = 'chat-context-toggle-row chat-context-overflow-row' + (legacy ? ' disabled' : '');
    if (legacy) el.setAttribute('title', 'Applies to chats on the new agent runtime');
  };
  return { el, update };
}

// Values and order mirror store.CHAT_SETTINGS["thinking_level"]; the first is the default.
export const THINKING_LEVEL_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: 'off', label: 'Off' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
  { value: 'xhigh', label: 'XHigh' },
  { value: 'max', label: 'Max' },
];

const GREY_REASONS = {
  legacy: 'Thinking level applies to chats on the new agent runtime',
  noLevel: "This model's API can't take a thinking level",
};

// The <select> beside the model picker. onChange(value) -> Promise<boolean>; false
// keeps the old value. update({value, caps}) fills it from the settings value and
// the capabilities payload; only a level the route calls "unsupported" is greyed
// ("unknown" stays selectable, the loop learns from the model's answer).
// refresh(sid) fetches both routes; an answer for an older call is dropped.
export function createThinkingLevelSelect({
  doc = document, onChange, fetchSettings = fetchChatAgentSettings, fetchCapabilities = fetchChatAgentCapabilities,
}) {
  const el = doc.createElement('select');
  el.className = 'thinking-level-select';
  el.setAttribute('aria-label', 'Thinking level');
  const optionEls = new Map();
  for (const { value, label } of THINKING_LEVEL_OPTIONS) {
    const opt = doc.createElement('option');
    opt.value = value;
    opt.textContent = label;
    el.appendChild(opt);
    optionEls.set(value, opt);
  }

  let current = THINKING_LEVEL_OPTIONS[0].value;
  el.value = current;
  el.disabled = true;

  el.addEventListener('change', async () => {
    const next = el.value;
    el.disabled = true;
    const ok = await onChange(next);
    if (ok) current = next; else el.value = current;
    el.disabled = false;
  });

  const grey = reason => {
    el.disabled = true;
    el.title = reason;
  };

  const update = ({ value, caps }) => {
    current = THINKING_LEVEL_OPTIONS.some(o => o.value === value) ? value : THINKING_LEVEL_OPTIONS[0].value;
    el.value = current;
    const levels = caps?.thinking?.levels || {};
    for (const { value: v } of THINKING_LEVEL_OPTIONS) {
      const o = optionEls.get(v);
      const refused = levels[v] === 'unsupported';
      o.disabled = refused;
      o.title = refused ? 'Not supported by this model' : '';
    }
    if (!caps) return grey('');
    if (caps.route === 'legacy') return grey(GREY_REASONS.legacy);
    if (!caps.thinking?.available) return grey(GREY_REASONS.noLevel);
    el.disabled = false;
    el.title = '';
  };

  let seq = 0;
  const refresh = async sid => {
    const mine = ++seq;
    if (!sid) { update({ value: undefined, caps: null }); return; }
    const [settings, caps] = await Promise.all([fetchSettings(sid), fetchCapabilities(sid)]);
    if (mine !== seq) return;
    update({ value: settings?.thinking_level, caps: settings ? caps : null });
  };

  return { el, update, refresh };
}
