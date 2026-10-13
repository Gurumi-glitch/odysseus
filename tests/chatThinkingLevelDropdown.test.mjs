import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { THINKING_LEVEL_OPTIONS, createThinkingLevelSelect } from '../static/js/chatAgentSettings.js';

// Thinking level dropdown next to the model picker (spec #89, ticket #101).
// No browser: a minimal fake document is enough for one <select> of <option>s.
function fakeDoc() {
  const make = tag => {
    const attrs = {};
    const node = {
      tag, children: [], className: '', textContent: '', value: '', disabled: false, title: '', attrs,
      listeners: {},
      appendChild(c) { node.children.push(c); return c; },
      setAttribute(k, v) { attrs[k] = String(v); },
      removeAttribute(k) { delete attrs[k]; },
      addEventListener(type, fn) { node.listeners[type] = fn; },
    };
    return node;
  };
  return { createElement: make };
}
const opt = (sel, v) => sel.children.find(o => o.value === v);
const levels = over => ({ off: 'supported', low: 'supported', medium: 'supported', high: 'supported', xhigh: 'supported', max: 'supported', ...over });
const caps = (over = {}, extra = {}) => ({ route: 'chat_agent', reason: '', thinking: { available: true, binary: false, levels: levels(over) }, ...extra });

test('options: Auto first, then the six levels with the spec labels', () => {
  assert.deepEqual(THINKING_LEVEL_OPTIONS.map(o => o.value), ['auto', 'off', 'low', 'medium', 'high', 'xhigh', 'max']);
  assert.deepEqual(THINKING_LEVEL_OPTIONS.map(o => o.label), ['Auto', 'Off', 'Low', 'Medium', 'High', 'XHigh', 'Max']);
});

test('before any data arrives the control is greyed on Auto', () => {
  const { el } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async () => true });
  assert.equal(el.disabled, true);
  assert.equal(el.value, 'auto');
});

test('shows the chat level; unsupported levels are greyed with a tooltip, not removed', () => {
  const { el, update } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async () => true });
  update({ value: 'low', caps: caps({ high: 'unsupported', max: 'unsupported', xhigh: 'unknown' }) });
  assert.equal(el.disabled, false);
  assert.equal(el.value, 'low');
  assert.equal(el.children.length, 7);
  assert.equal(opt(el, 'high').disabled, true);
  assert.match(opt(el, 'high').title, /not supported/i);
  assert.equal(opt(el, 'max').disabled, true);
  assert.equal(opt(el, 'xhigh').disabled, false);
  assert.equal(opt(el, 'auto').disabled, false);
});

test('a stored level the model now refuses stays shown, greyed', () => {
  const { el, update } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async () => true });
  update({ value: 'high', caps: caps({ high: 'unsupported' }) });
  assert.equal(el.value, 'high');
  assert.equal(opt(el, 'high').disabled, true);
});

test('legacy route greys the whole control, with a reason', () => {
  const { el, update } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async () => true });
  update({ value: 'auto', caps: caps({}, { route: 'legacy', reason: 'runtime_off' }) });
  assert.equal(el.disabled, true);
  assert.ok(el.title);
});

test('a dialect that cannot carry a level greys the whole control', () => {
  const { el, update } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async () => true });
  update({ value: 'auto', caps: { route: 'chat_agent', reason: '', thinking: { available: false, binary: false, levels: {} } } });
  assert.equal(el.disabled, true);
});

test('unknown value falls back to Auto', () => {
  const { el, update } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async () => true });
  update({ value: 'bogus', caps: caps() });
  assert.equal(el.value, 'auto');
});

test('choosing a level saves it; a failed save puts the old value back', async () => {
  const saved = [];
  let ok = true;
  const { el, update } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async v => { saved.push(v); return ok; } });
  update({ value: 'auto', caps: caps() });
  el.value = 'medium';
  await el.listeners.change();
  assert.deepEqual(saved, ['medium']);
  assert.equal(el.value, 'medium');
  ok = false;
  el.value = 'max';
  await el.listeners.change();
  assert.equal(el.value, 'medium');
  assert.equal(el.disabled, false);
});

test('refresh(sid) reads both routes; a late answer for an older chat is dropped', async () => {
  const resolvers = {};
  const fetchers = {
    fetchSettings: sid => new Promise(r => { resolvers[sid] = r; }),
    fetchCapabilities: async () => caps({ high: 'unsupported' }),
  };
  const { el, refresh } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async () => true, ...fetchers });
  const first = refresh('a');
  const second = refresh('b');
  resolvers.b({ available: true, thinking_level: 'low' });
  await second;
  assert.equal(el.value, 'low');
  resolvers.a({ available: true, thinking_level: 'max' });
  await first;
  assert.equal(el.value, 'low');
});

test('refresh with no chat yet (pending) or failed routes leaves the control greyed', async () => {
  const { el, refresh } = createThinkingLevelSelect({
    doc: fakeDoc(), onChange: async () => true,
    fetchSettings: async () => null, fetchCapabilities: async () => null,
  });
  await refresh(null);
  assert.equal(el.disabled, true);
  await refresh('s1');
  assert.equal(el.disabled, true);
});

// Wiring: the chat.js refusal event and the picker mount.
const chat = await readFile(new URL('../static/js/chat.js', import.meta.url), 'utf8');
const picker = await readFile(new URL('../static/js/modelPicker.js', import.meta.url), 'utf8');
test('chat.js turns thinking_level_refused into a notice in the turn and a refresh event', () => {
  assert.match(chat, /json\.type === 'thinking_level_refused'[\s\S]{0,900}json\.message[\s\S]{0,400}CAPABILITIES_CHANGED/);
});
test('chat.js also sends the refresh event when a turn ends on an error (a tools rejection is one)', () => {
  assert.match(chat, /_streamTerminalError = createTerminalStreamError\(json\);[\s\S]{0,300}dispatchEvent\(new CustomEvent\(CAPABILITIES_CHANGED\)\)/);
});
test('modelPicker mounts the control in the picker wrap and refreshes on picker update and the refresh event', () => {
  assert.match(picker, /createThinkingLevelSelect/);
  assert.match(picker, /addEventListener\(CAPABILITIES_CHANGED/);
});
