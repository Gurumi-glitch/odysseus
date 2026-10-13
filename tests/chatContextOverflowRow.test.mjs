import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  OVERFLOW_OPTIONS, CAPABILITIES_CHANGED, createOverflowRow, fetchChatAgentSettings, fetchChatAgentCapabilities,
} from '../static/js/chatAgentSettings.js';

// Context overflow row of the chat context panel (spec #89, ticket #100).
// No browser: a minimal fake document is enough for one label and one <select>.
function fakeDoc() {
  const make = tag => {
    const attrs = {};
    const node = {
      tag, children: [], className: '', textContent: '', value: '', disabled: false, attrs,
      listeners: {},
      appendChild(c) { node.children.push(c); return c; },
      setAttribute(k, v) { attrs[k] = String(v); },
      addEventListener(type, fn) { node.listeners[type] = fn; },
    };
    return node;
  };
  return { createElement: make };
}
const select = row => {
  const find = n => n.tag === 'select' ? n : n.children.map(find).find(Boolean);
  return find(row);
};

test('options are the four store values, Auto first, with the spec labels', () => {
  assert.deepEqual(OVERFLOW_OPTIONS.map(o => o.value), ['auto', 'rolling_window', 'truncate_middle', 'stop_at_limit']);
  assert.deepEqual(OVERFLOW_OPTIONS.map(o => o.label), ['Auto', 'Rolling window', 'Truncate middle', 'Stop at limit']);
});

test('update shows the chat value and enables the select on the new runtime', () => {
  const { el, update } = createOverflowRow({ doc: fakeDoc(), onChange: async () => true });
  update({ value: 'truncate_middle', legacy: false });
  const sel = select(el);
  assert.equal(sel.value, 'truncate_middle');
  assert.equal(sel.disabled, false);
});

test('a legacy chat greys the row (disabled, still present)', () => {
  const { el, update } = createOverflowRow({ doc: fakeDoc(), onChange: async () => true });
  update({ value: 'auto', legacy: true });
  assert.equal(select(el).disabled, true);
  assert.match(el.className, /disabled/);
});

test('before any data arrives the row is disabled on Auto', () => {
  const { el } = createOverflowRow({ doc: fakeDoc(), onChange: async () => true });
  assert.equal(select(el).disabled, true);
  assert.equal(select(el).value, 'auto');
});

test('choosing a value saves it; a failed save puts the old value back', async () => {
  const saved = [];
  let ok = true;
  const { el, update } = createOverflowRow({ doc: fakeDoc(), onChange: async v => { saved.push(v); return ok; } });
  update({ value: 'auto', legacy: false });
  const sel = select(el);

  sel.value = 'rolling_window';
  await sel.listeners.change();
  assert.deepEqual(saved, ['rolling_window']);
  assert.equal(sel.value, 'rolling_window');

  ok = false;
  sel.value = 'stop_at_limit';
  await sel.listeners.change();
  assert.equal(sel.value, 'rolling_window');
  assert.equal(sel.disabled, false);
});

test('refresh(sid) reads both routes: the new runtime enables the row, a legacy route greys it', async () => {
  let route = 'chat_agent';
  const fetchers = {
    fetchSettings: async () => ({ available: true, context_overflow: 'rolling_window' }),
    fetchCapabilities: async () => ({ route, reason: '' }),
  };
  const { el, refresh } = createOverflowRow({ doc: fakeDoc(), onChange: async () => true, ...fetchers });
  await refresh('s1');
  assert.equal(select(el).value, 'rolling_window');
  assert.equal(select(el).disabled, false);
  // The turn just learned the model takes no native tools: the next one is legacy.
  route = 'legacy';
  await refresh('s1');
  assert.equal(select(el).disabled, true);
});

test('fetchChatAgentSettings reads the GET settings route for the chat', async () => {
  const calls = [];
  globalThis.fetch = async (url, opts) => { calls.push(url); return { ok: true, json: async () => ({ available: true, context_overflow: 'rolling_window' }) }; };
  assert.deepEqual(await fetchChatAgentSettings('s 1'), { available: true, context_overflow: 'rolling_window' });
  assert.equal(calls[0], '/api/chat_agent/settings?session_id=s%201');
});

test('fetchChatAgentCapabilities returns the route payload, null on failure', async () => {
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ route: 'legacy', reason: 'no_adapter' }) });
  assert.equal((await fetchChatAgentCapabilities('s1')).route, 'legacy');
  globalThis.fetch = async () => ({ ok: false, status: 404 });
  assert.equal(await fetchChatAgentCapabilities('s1'), null);
  globalThis.fetch = async () => { throw new Error('offline'); };
  assert.equal(await fetchChatAgentCapabilities('s1'), null);
});

// chat.js wiring: row sits above Memory extraction; trimmed event gets a toast.
const chat = await readFile(new URL('../static/js/chat.js', import.meta.url), 'utf8');
test('chat.js mounts the overflow row before the Memory extraction row', () => {
  const mount = chat.indexOf('// Above Memory extraction.\n    _appendOverflowRow(popup);');
  const memory = chat.indexOf('popup.appendChild(memoryRow)');
  assert.ok(mount > 0 && memory > mount);
});
test('chat.js shows a toast for the live trimmed event', () => {
  assert.match(chat, /json\.type === 'trimmed'[\s\S]{0,300}showToast\(json\.message/);
});
test('chat.js re-reads the open row when a turn changes what the model takes', () => {
  assert.match(chat, /const reread = [^\n]*overflow\.refresh[\s\S]{0,300}const rereadOverflow = [\s\S]{0,200}reread\(\)[\s\S]{0,100}addEventListener\(CAPABILITIES_CHANGED, rereadOverflow\)/);
  assert.equal(CAPABILITIES_CHANGED, 'odysseus:chat-agent-capabilities-changed');
});
