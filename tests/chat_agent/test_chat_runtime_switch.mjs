// Settings switch for the chat_agent runtime (spec #44, ticket #48).
//
// Seam: initAgentSettings() in static/js/settings/agent.js, driven through
// the DOM ids the Settings markup gives it and the two /api/chat_agent/settings
// calls. The real agent.js and dom.js are loaded through Node's ESM linker;
// api.js is synthetic because its ui.js import needs a browser.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const JS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../static/js');
const AGENT_JS = path.join(JS, 'settings/agent.js');
const REAL_MODULES = new Set([AGENT_JS, path.join(JS, 'settings/dom.js')]);

class Element {
  constructor(id) {
    this.id = id;
    this.checked = false;
    this.disabled = false;
    this.textContent = '';
    this.style = {};
    this._listeners = {};
  }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  async dispatch(type) {
    for (const fn of this._listeners[type] || []) await fn({ target: this });
  }
}

function makeDom(ids) {
  const byId = new Map(ids.map(id => [id, new Element(id)]));
  return { byId, document: { getElementById: id => byId.get(id) || null } };
}

// Scripted fetch: each entry answers one call in order; records requests.
function makeFetch(responses) {
  const calls = [];
  const fetch = async (url, init = {}) => {
    calls.push({ url, method: init.method || 'GET', body: init.body ? JSON.parse(init.body) : null });
    const next = responses.shift();
    if (!next) throw new Error(`unexpected fetch ${init.method || 'GET'} ${url}`);
    return { ok: next.status < 400, status: next.status, json: async () => next.body };
  };
  return { fetch, calls };
}

async function loadAgentSettings({ ids, responses, window = {} }) {
  const dom = makeDom(ids);
  const net = makeFetch(responses);
  const context = vm.createContext({ document: dom.document, fetch: net.fetch, console, window });
  const cache = new Map();

  function load(filename) {
    if (cache.has(filename)) return cache.get(filename);
    let module;
    if (REAL_MODULES.has(filename)) {
      module = new vm.SourceTextModule(fs.readFileSync(filename, 'utf8'), { context, identifier: filename });
    } else if (filename === path.join(JS, 'settings/api.js')) {
      module = new vm.SyntheticModule(['postSettings'], function () {
        this.setExport('postSettings', async () => { throw new Error('postSettings is not the runtime switch path'); });
      }, { context, identifier: filename });
    } else {
      throw new Error(`unexpected module ${filename}`);
    }
    cache.set(filename, module);
    return module;
  }

  const agent = load(AGENT_JS);
  await agent.link((specifier, referrer) =>
    load(path.resolve(path.dirname(referrer.identifier), specifier.split('?')[0])));
  await agent.evaluate();
  await agent.namespace.initAgentSettings();
  return { ...dom, calls: net.calls };
}

const SWITCH_IDS = ['set-chatRuntimeCard', 'set-chatRuntimeToggle', 'set-chatRuntimeMsg'];

test('switch reflects the stored runtime and saves the new value through PUT', async () => {
  const { byId, calls } = await loadAgentSettings({
    ids: SWITCH_IDS,
    responses: [
      { status: 200, body: { chat_runtime: 'new', available: true } },
      { status: 200, body: { chat_runtime: 'legacy', available: true } },
    ],
  });
  const toggle = byId.get('set-chatRuntimeToggle');
  assert.equal(calls[0].url, '/api/chat_agent/settings');
  assert.equal(toggle.checked, true);
  assert.equal(byId.get('set-chatRuntimeCard').style.display, undefined);

  toggle.checked = false;
  await toggle.dispatch('change');

  assert.deepEqual(calls[1], { url: '/api/chat_agent/settings', method: 'PUT', body: { chat_runtime: 'legacy' } });
  assert.equal(toggle.checked, false);
  assert.match(byId.get('set-chatRuntimeMsg').textContent, /next message/);
});

test('a failing composer icon sync never undoes a save that succeeded', async () => {
  const { byId } = await loadAgentSettings({
    ids: SWITCH_IDS,
    responses: [
      { status: 200, body: { chat_runtime: 'legacy', available: true } },
      { status: 200, body: { chat_runtime: 'new', available: true } },
    ],
    window: { __odysseusSyncAutoModeButton() { throw new Error('icon sync broke'); } },
  });
  const toggle = byId.get('set-chatRuntimeToggle');
  toggle.checked = true;
  await toggle.dispatch('change').catch(() => {});

  assert.equal(toggle.checked, true);
  assert.match(byId.get('set-chatRuntimeMsg').textContent, /next message/);
});

test('a refused save reverts the switch and shows the server reason in red', async () => {
  const { byId } = await loadAgentSettings({
    ids: SWITCH_IDS,
    responses: [
      { status: 200, body: { chat_runtime: 'legacy', available: true } },
      { status: 403, body: { detail: 'Tool requests cannot change the chat runtime.' } },
    ],
  });
  const toggle = byId.get('set-chatRuntimeToggle');
  toggle.checked = true;
  await toggle.dispatch('change');

  assert.equal(toggle.checked, false);
  const msg = byId.get('set-chatRuntimeMsg');
  assert.equal(msg.textContent, 'Tool requests cannot change the chat runtime.');
  assert.equal(msg.style.color, 'var(--red)');
});

test('the card is hidden when the runtime is unavailable (auth disabled)', async () => {
  const { byId } = await loadAgentSettings({
    ids: SWITCH_IDS,
    responses: [{ status: 200, body: { chat_runtime: 'legacy', available: false } }],
  });
  assert.equal(byId.get('set-chatRuntimeCard').style.display, 'none');
});

test('pages without the switch markup still initialise the other agent settings', async () => {
  // set-agentMaxTools absent too: initAgentSettings must return without fetching.
  const { calls } = await loadAgentSettings({ ids: [], responses: [] });
  assert.deepEqual(calls, []);
});
