import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { chromium } from 'playwright';

// A new chat before its first message (issue #103), with the real sessions.js,
// modelPicker.js and chat.js modules in Chromium and a recording fetch: the
// Thinking level and the chat-settings pill work while the chat is pending,
// and the session its first message creates takes over (the dropdown used to
// stay greyed after that transition until a turn ended in an error).
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PAGE = `<!doctype html><html><body>
<div id="current-meta"></div><button id="chat-context-pill" hidden></button>
<div id="model-picker-wrap"><button id="model-picker-btn"><span id="model-picker-label"></span></button>
<div id="model-picker-menu" class="hidden"><div class="model-picker-search-row"><input id="model-picker-search"></div><div id="model-picker-list"></div></div></div>
<main id="chat-history"></main><textarea id="message"></textarea>
<script type="module">
import * as sessions from '/static/js/sessions.js';
import '/static/js/chat.js';
window.sessions = sessions;
window.ready = true;
</script></body></html>`;

let browser;
let server;
let base;
before(async () => {
  server = http.createServer(async (req, res) => {
    const { pathname } = new URL(req.url, 'http://x');
    if (pathname === '/') { res.setHeader('content-type', 'text/html'); res.end(PAGE); return; }
    try {
      const body = await readFile(path.join(ROOT, pathname));
      res.setHeader('content-type', pathname.endsWith('.js') ? 'text/javascript' : 'text/plain');
      res.end(body);
    } catch { res.statusCode = 404; res.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true });
});
after(async () => { await browser?.close(); server?.close(); });

async function setup() {
  const page = await browser.newPage();
  await page.addInitScript(() => {
    const caps = { route: 'chat_agent', reason: '', thinking: { available: true, binary: false,
      levels: { off: 'supported', low: 'supported', medium: 'supported', high: 'unsupported', xhigh: 'supported', max: 'unsupported' } } };
    const answers = {
      'GET /api/chat_agent/settings': { available: true, chat_runtime: 'new', thinking_level: 'off', context_overflow: 'auto' },
      'GET /api/chat_agent/capabilities': caps,
      'GET /api/chat_agent/capabilities/s1': caps,
      'POST /api/session': { id: 's1' },
      'POST /api/session/s1/generation-settings': { status: 'success' },
      'GET /api/sessions': [],
    };
    window.calls = [];
    const staticFetch = window.fetch.bind(window);
    window.fetch = async (url, init = {}) => {
      const u = new URL(String(url), location.href);
      if (u.pathname.startsWith('/static/')) return staticFetch(url, init);
      const key = `${init.method || 'GET'} ${u.pathname}`;
      window.calls.push({ key, search: u.search, body: typeof init.body === 'string' ? JSON.parse(init.body) : undefined });
      return new Response(JSON.stringify(answers[key] ?? {}), { status: 200, headers: { 'content-type': 'application/json' } });
    };
  });
  await page.goto(base + '/');
  await page.waitForFunction(() => window.ready);
  await page.evaluate(() => window.sessions.createDirectChat('http://host.docker.internal:1235/v1', 'Bonsai', 'ep1'));
  await page.waitForFunction(() => document.querySelector('.thinking-level-select') && !document.getElementById('chat-context-pill').hidden);
  await page.waitForTimeout(50);
  return page;
}

const thinking = page => page.evaluate(() => {
  const el = document.querySelector('.thinking-level-select');
  return { disabled: el.disabled, value: el.value };
});

async function openPopup(page) {
  await page.click('#chat-context-pill');
  await page.waitForSelector('.chat-context-popup');
  await page.waitForTimeout(50);
  return page.evaluate(() => {
    const popup = document.querySelector('.chat-context-popup');
    const sliders = [...popup.querySelectorAll('.chat-context-threshold-row')].map(row => [...row.querySelectorAll('.chat-context-threshold-top span')].map(span => span.textContent).join(' '));
    return { text: popup.innerText, overflowDisabled: popup.querySelector('.chat-context-select').disabled, sliders };
  });
}

test('pending chat: Thinking level is selectable for the picked endpoint and model', async () => {
  const page = await setup();
  try {
    assert.deepEqual(await thinking(page), { disabled: false, value: 'off' });
    const asked = await page.evaluate(() => window.calls.find(c => c.key === 'GET /api/chat_agent/capabilities'));
    assert.equal(new URLSearchParams(asked.search).get('endpoint_id'), 'ep1');
  } finally { await page.close(); }
});

test('pending chat: the pill opens Context overflow, Temperature and Max tokens only', async () => {
  const page = await setup();
  try {
    const popup = await openPopup(page);
    assert.equal(popup.overflowDisabled, false);
    assert.deepEqual(popup.sliders, ['Temperature Default', 'Max tokens No limit']);
    assert.doesNotMatch(popup.text, /Memory extraction|Usage|Auto compact/);
  } finally { await page.close(); }
});

test('first message: the new session gets the held Temperature before anything reads it; Thinking level stays selectable', async () => {
  const page = await setup();
  try {
    await openPopup(page);
    await page.evaluate(() => {
      const slider = document.querySelector('.chat-context-popup input[type=range]');
      slider.value = '0.4';
      slider.dispatchEvent(new Event('change'));
      window.calls.length = 0;
    });

    assert.equal(await page.evaluate(() => window.sessions.materializePendingSession()), true);
    await page.waitForTimeout(100);

    const calls = await page.evaluate(() => window.calls);
    const keys = calls.map(c => c.key);
    const save = keys.indexOf('POST /api/session/s1/generation-settings');
    assert.equal(keys[0], 'POST /api/session');
    assert.equal(save, 1);
    assert.deepEqual(calls[save].body, { temperature_override: 0.4 });
    assert.ok(keys.indexOf('GET /api/chat_agent/capabilities/s1') > save);
    assert.deepEqual(await thinking(page), { disabled: false, value: 'off' });
  } finally { await page.close(); }
});

test('a later new chat starts at Default again', async () => {
  const page = await setup();
  try {
    await openPopup(page);
    await page.evaluate(() => {
      const slider = document.querySelector('.chat-context-popup input[type=range]');
      slider.value = '0.4';
      slider.dispatchEvent(new Event('change'));
    });
    await page.keyboard.press('Escape');
    await page.evaluate(() => window.sessions.createDirectChat('http://host.docker.internal:1235/v1', 'Bonsai', 'ep1'));
    await page.waitForTimeout(50);

    assert.equal((await openPopup(page)).sliders[0], 'Temperature Default');
  } finally { await page.close(); }
});
