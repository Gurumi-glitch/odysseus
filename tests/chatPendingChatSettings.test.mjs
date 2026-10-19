import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  applyPendingAgentSettings, createOverflowRow, createThinkingLevelSelect, dropPendingAgentSettings,
  fetchChatAgentCapabilities, fetchChatAgentSettings, saveChatAgentSetting,
} from '../static/js/chatAgentSettings.js';
import {
  applyPendingGeneration, dropPendingGeneration, holdPendingGeneration, pendingGeneration,
} from '../static/js/pendingChatSettings.js';

// A new chat before its first message has no session (issue #103): the
// controls read the picker's (endpoint, model) and the owner's last-used
// values; Temperature / Max tokens are held until the session exists.
function fakeDoc() {
  const make = tag => {
    const node = {
      tag, children: [], className: '', textContent: '', value: '', disabled: false, title: '', listeners: {},
      appendChild(c) { node.children.push(c); return c; },
      setAttribute() {}, removeAttribute() {},
      addEventListener(type, fn) { node.listeners[type] = fn; },
    };
    return node;
  };
  return { createElement: make };
}

const PENDING = { url: 'http://host.docker.internal:1235/v1', modelId: 'Ternary Bonsai', endpointId: 'ep 1', source: 'manual' };
const LEVELS = { off: 'supported', low: 'supported', medium: 'supported', high: 'unsupported', xhigh: 'supported', max: 'unsupported' };
const CAPS = { route: 'chat_agent', reason: '', thinking: { available: true, binary: false, levels: LEVELS } };

let calls;
function serve(answers) {
  calls = [];
  globalThis.fetch = async (url, opts = {}) => {
    calls.push({ url, method: opts.method || 'GET', body: opts.body ? JSON.parse(opts.body) : undefined });
    const path = String(url).split('?')[0];
    const body = answers[path];
    return body === undefined ? { ok: false, status: 404 } : { ok: true, json: async () => body };
  };
}

beforeEach(() => { dropPendingGeneration(); dropPendingAgentSettings(); });

test('no chat yet: settings come from the owner (no session_id), capabilities from the picked endpoint and model', async () => {
  serve({ '/api/chat_agent/settings': { available: true, thinking_level: 'off' }, '/api/chat_agent/capabilities': CAPS });

  assert.equal((await fetchChatAgentSettings('')).thinking_level, 'off');
  assert.deepEqual(await fetchChatAgentCapabilities('', PENDING), CAPS);

  assert.equal(calls[0].url, '/api/chat_agent/settings');
  const query = new URL(calls[1].url, 'http://x').searchParams;
  assert.equal(new URL(calls[1].url, 'http://x').pathname, '/api/chat_agent/capabilities');
  assert.deepEqual(Object.fromEntries(query), {
    model: 'Ternary Bonsai', endpoint_id: 'ep 1', endpoint_url: 'http://host.docker.internal:1235/v1',
  });
});

test('no chat and no pick: nothing to ask, no request', async () => {
  serve({});
  assert.equal(await fetchChatAgentCapabilities('', null), null);
  assert.deepEqual(calls, []);
});

test('a save with no chat writes the owner\'s last-used value (no session_id)', async () => {
  serve({ '/api/chat_agent/settings': { available: true } });
  assert.equal(await saveChatAgentSetting('', 'thinking_level', 'off'), 'off');
  assert.deepEqual(calls[0], { url: '/api/chat_agent/settings', method: 'PUT', body: { thinking_level: 'off' } });
});

test('Thinking level: selectable on a pending chat, and still selectable once its first message created the session', async () => {
  serve({
    '/api/chat_agent/settings': { available: true, thinking_level: 'off' },
    '/api/chat_agent/capabilities': CAPS,
    '/api/chat_agent/capabilities/s1': CAPS,
  });
  const { el, refresh } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async () => true });

  await refresh('', PENDING);
  assert.equal(el.disabled, false);
  assert.equal(el.value, 'off');
  assert.equal(el.children.find(o => o.value === 'high').disabled, true);

  await refresh('s1');
  assert.equal(el.disabled, false);
  assert.equal(calls.at(-1).url, '/api/chat_agent/capabilities/s1');
});

test('Thinking level: a pending pick the new runtime does not answer stays greyed', async () => {
  serve({
    '/api/chat_agent/settings': { available: true, thinking_level: 'auto' },
    '/api/chat_agent/capabilities': { ...CAPS, route: 'legacy', reason: 'runtime_off' },
  });
  const { el, refresh } = createThinkingLevelSelect({ doc: fakeDoc(), onChange: async () => true });
  await refresh('', PENDING);
  assert.equal(el.disabled, true);
});

test('Context overflow row: usable on a pending chat with the owner\'s last-used policy', async () => {
  serve({
    '/api/chat_agent/settings': { available: true, context_overflow: 'rolling_window' },
    '/api/chat_agent/capabilities': CAPS,
  });
  const { el, refresh } = createOverflowRow({ doc: fakeDoc(), onChange: async () => true });
  await refresh('', PENDING);
  const select = el.children.find(c => c.tag === 'select');
  assert.equal(select.disabled, false);
  assert.equal(select.value, 'rolling_window');
});

test('held Temperature / Max tokens go to the session the first message creates, then are gone', async () => {
  serve({ '/api/session/s%201/generation-settings': { status: 'success', temperature_override: 0.3 } });
  holdPendingGeneration({ temperature_override: 0.5 });
  holdPendingGeneration({ temperature_override: 0.3 });
  holdPendingGeneration({ max_tokens_override: null });
  assert.deepEqual(pendingGeneration(), { temperature_override: 0.3, max_tokens_override: null });

  assert.equal(await applyPendingGeneration('s 1'), true);

  // Only what was changed: the session keeps its own thinking mode.
  assert.deepEqual(calls, [{
    url: '/api/session/s%201/generation-settings', method: 'POST',
    body: { temperature_override: 0.3, max_tokens_override: null },
  }]);
  assert.deepEqual(pendingGeneration(), {});
});

test('nothing held: the new session is left at Default, no request', async () => {
  serve({});
  assert.equal(await applyPendingGeneration('s1'), true);
  assert.deepEqual(calls, []);
});

test('a new chat starts at Default: dropping clears what an earlier pending chat held', async () => {
  serve({ '/api/chat_agent/settings': { available: true } });
  holdPendingGeneration({ temperature_override: 1.2 });
  await saveChatAgentSetting('', 'thinking_level', 'high');
  dropPendingGeneration();
  dropPendingAgentSettings();
  assert.deepEqual(pendingGeneration(), {});

  serve({});
  assert.equal(await applyPendingAgentSettings('s1'), true);
  assert.deepEqual(calls, []);
});

// Issue #104: without this the chat keeps reading the owner's last-used value,
// so a pick made later in any other chat changes this one too.
test('Thinking level / Context overflow picked on a pending chat are stored on the session its first message creates', async () => {
  serve({ '/api/chat_agent/settings': { available: true } });
  await saveChatAgentSetting('', 'thinking_level', 'high');
  await saveChatAgentSetting('', 'context_overflow', 'stop_at_limit');
  await saveChatAgentSetting('', 'thinking_level', 'off');

  serve({ '/api/chat_agent/settings': { available: true } });
  assert.equal(await applyPendingAgentSettings('s 1'), true);
  assert.deepEqual(calls, [{
    url: '/api/chat_agent/settings', method: 'PUT',
    body: { session_id: 's 1', thinking_level: 'off', context_overflow: 'stop_at_limit' },
  }]);

  // Applied once: the next new chat does not get them again.
  serve({});
  assert.equal(await applyPendingAgentSettings('s2'), true);
  assert.deepEqual(calls, []);
});

test('nothing picked on a pending chat: no request, the chat keeps inheriting the last-used values', async () => {
  serve({});
  assert.equal(await applyPendingAgentSettings('s1'), true);
  assert.deepEqual(calls, []);
});

test('a save on an existing chat is not held for the next new chat', async () => {
  serve({ '/api/chat_agent/settings': { available: true } });
  await saveChatAgentSetting('s0', 'thinking_level', 'high');

  serve({});
  assert.equal(await applyPendingAgentSettings('s1'), true);
  assert.deepEqual(calls, []);
});
