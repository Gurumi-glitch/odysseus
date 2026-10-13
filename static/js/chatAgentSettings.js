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
// No sid (a new chat before its first message): the owner's last-used values.
export function fetchChatAgentSettings(sid) {
  return getJson(sid ? `/api/chat_agent/settings?session_id=${encodeURIComponent(sid)}` : '/api/chat_agent/settings');
}

// -> {route: 'chat_agent'|'legacy', reason, thinking: {available, binary, levels}} or null.
// Shared with the thinking-level control (#101): one place for the fetch.
// No sid: keyed by the pending chat's pick ({url, modelId, endpointId}), as its first message will send it.
export async function fetchChatAgentCapabilities(sid, pending = null) {
  if (sid) return getJson(`/api/chat_agent/capabilities/${encodeURIComponent(sid)}`);
  if (!pending?.modelId) return null;
  const query = new URLSearchParams({
    model: pending.modelId, endpoint_id: pending.endpointId || '', endpoint_url: pending.url || '',
  });
  return getJson(`/api/chat_agent/capabilities?${query}`);
}

// Returns the new value on success, null when the save was refused.
// No sid: only the owner's last-used value, which the new chat inherits.
export async function saveChatAgentSetting(sid, name, value) {
  try {
    const res = await fetch('/api/chat_agent/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(sid ? { session_id: sid, [name]: value } : { [name]: value }),
    });
    return res.ok ? value : null;
  } catch (_) {
    return null;
  }
}

// Sent when a turn may have taught the server side something new about the
// chat's model (a refused level, or no native tools): controls re-read the routes.
export const CAPABILITIES_CHANGED = 'odysseus:chat-agent-capabilities-changed';

function knownValue(options, value) {
  return options.some(o => o.value === value) ? value : options[0].value;
}

// One <option> per entry of `options` in `sel`, by value; `sel` starts greyed on the first.
function fillSelect(doc, sel, options) {
  const optionEls = new Map();
  for (const { value, label } of options) {
    const opt = doc.createElement('option');
    opt.value = value;
    opt.textContent = label;
    sel.appendChild(opt);
    optionEls.set(value, opt);
  }
  sel.value = options[0].value;
  sel.disabled = true;
  return optionEls;
}

// A choice is saved through onChange(value) -> Promise<boolean>; false puts
// back the value saved last. Returns the holder of that value for update().
function saveOnChange(sel, onChange) {
  const saved = { value: sel.value };
  sel.addEventListener('change', async () => {
    const next = sel.value;
    sel.disabled = true;
    const ok = await onChange(next);
    if (ok) saved.value = next; else sel.value = saved.value;
    sel.disabled = false;
  });
  return saved;
}

// refresh(sid, pending) reads both routes and hands them to apply(settings, caps);
// an answer for an older call is dropped. Without sid the pending chat's pick
// is asked; with neither, (null, null) applies.
function latestRoutes(fetchSettings, fetchCapabilities, apply) {
  let seq = 0;
  return async (sid, pending = null) => {
    const mine = ++seq;
    if (!sid && !pending) { apply(null, null); return; }
    const [settings, caps] = await Promise.all([fetchSettings(sid || ''), fetchCapabilities(sid || '', sid ? null : pending)]);
    if (mine === seq) apply(settings, caps);
  };
}

// update({value, legacy}) fills the row once the routes answered; until then
// (and on a legacy chat) the select is greyed, not hidden. refresh(sid): see latestRoutes.
export function createOverflowRow({
  doc = document, onChange, fetchSettings = fetchChatAgentSettings, fetchCapabilities = fetchChatAgentCapabilities,
}) {
  const el = doc.createElement('div');
  el.className = 'chat-context-toggle-row chat-context-overflow-row disabled';
  const copy = doc.createElement('div');
  copy.className = 'chat-context-toggle-copy';
  const label = doc.createElement('span');
  label.textContent = 'Context overflow';
  copy.appendChild(label);

  const sel = doc.createElement('select');
  sel.className = 'chat-context-select';
  sel.setAttribute('aria-label', 'Context overflow for this chat');
  fillSelect(doc, sel, OVERFLOW_OPTIONS);
  const saved = saveOnChange(sel, onChange);

  el.appendChild(copy);
  el.appendChild(sel);

  const update = ({ value, legacy }) => {
    saved.value = sel.value = knownValue(OVERFLOW_OPTIONS, value);
    sel.disabled = !!legacy;
    el.className = 'chat-context-toggle-row chat-context-overflow-row' + (legacy ? ' disabled' : '');
    if (legacy) el.setAttribute('title', 'Applies to chats on the new agent runtime');
  };
  // Grey unless both routes say the new runtime answers this chat.
  const refresh = latestRoutes(fetchSettings, fetchCapabilities, (settings, caps) => update({
    value: settings?.context_overflow,
    legacy: !settings || settings.context_overflow === undefined || !caps || caps.route === 'legacy',
  }));
  return { el, update, refresh };
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
// refresh(sid): see latestRoutes.
export function createThinkingLevelSelect({
  doc = document, onChange, fetchSettings = fetchChatAgentSettings, fetchCapabilities = fetchChatAgentCapabilities,
}) {
  const el = doc.createElement('select');
  el.className = 'thinking-level-select';
  el.setAttribute('aria-label', 'Thinking level');
  const optionEls = fillSelect(doc, el, THINKING_LEVEL_OPTIONS);
  const saved = saveOnChange(el, onChange);

  const grey = reason => {
    el.disabled = true;
    el.title = reason;
  };

  const update = ({ value, caps }) => {
    saved.value = el.value = knownValue(THINKING_LEVEL_OPTIONS, value);
    const levels = caps?.thinking?.levels || {};
    for (const [v, o] of optionEls) {
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

  const refresh = latestRoutes(fetchSettings, fetchCapabilities, (settings, caps) => update({
    value: settings?.thinking_level, caps: settings ? caps : null,
  }));
  return { el, update, refresh };
}
