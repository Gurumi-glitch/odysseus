import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

// Execute the production composer-queue function in Chromium with a recording
// fetch: a send while a chat_agent run is live steers that run (spec #44, #63).
// Behavioral DOM tests, not source-text assertions.
const chat = await readFile(new URL('../static/js/chat.js', import.meta.url), 'utf8');
const renderer = await readFile(new URL('../static/js/chatRenderer.js', import.meta.url), 'utf8');
const queueFn = chat
  .slice(chat.indexOf('export function queueStreamingComposerRequest('), chat.indexOf('\n  function _drainQueuedAgentRequests'))
  .replace('export ', '');
const steerBubble = renderer
  .slice(renderer.indexOf('export function renderSteerBubble('), renderer.indexOf('\n}\n', renderer.indexOf('export function renderSteerBubble(')) + 3)
  .replace('export ', '');
let browser;
before(async () => { browser = await chromium.launch({ headless: true }); });
after(async () => { await browser?.close(); });

async function setup({ runtime = 'chat_agent', steerStatus = 200 } = {}) {
  const page = await browser.newPage();
  await page.setContent('<main id="chat-history"></main><textarea id="message">look at the tests dir too</textarea><button class="send-btn">Send</button>');
  await page.evaluate(([queueFn, steerBubble, runtime, steerStatus]) => {
    const esc = value => { const node = document.createElement('div'); node.textContent = String(value ?? ''); return node.innerHTML; };
    Object.assign(window, {
      posts: [], queued: [], errors: [], pendingFiles: 0,
      isStreaming: true, API_BASE: '',
      uiModule: { esc, el: id => document.getElementById(id), showError: e => window.errors.push(e), scrollHistory: () => {} },
      fileHandlerModule: { getPendingCount: () => window.pendingFiles },
      sessionModule: { getCurrentSessionId: () => 's1' },
      _streamRuntimes: new Map([['s1', runtime]]),
      _queueAgentRequest: text => { window.queued.push(text); return true; },
    });
    window.fetch = async (url, init) => {
      window.posts.push({ url: String(url), body: JSON.parse(init.body) });
      return new Response(JSON.stringify(steerStatus === 200 ? { accepted: true } : { detail: 'No running turn in this chat takes steering messages.' }), { status: steerStatus });
    };
    window.renderSteerBubble = new Function(steerBubble + '\nreturn renderSteerBubble;')();
    window.chatRenderer = { renderSteerBubble: window.renderSteerBubble };
    window.queueStreamingComposerRequest = new Function(queueFn + '\nreturn queueStreamingComposerRequest;')();
  }, [queueFn, steerBubble, runtime, steerStatus]);
  return page;
}

const state = () => ({
  composer: document.getElementById('message').value,
  steers: [...document.querySelectorAll('#chat-history .msg-steer')].map(n => ({ text: n.querySelector('.body').textContent, user: n.classList.contains('msg-user') })),
  posts: window.posts, queued: window.queued, errors: window.errors,
});

test('a send during a live chat_agent run posts a steer and draws it inline, nothing is queued', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async state => {
      const consumed = queueStreamingComposerRequest();
      await new Promise(r => setTimeout(r, 20));
      return { consumed, ...(0, eval)('(' + state + ')')() };
    }, String(state));
    assert.equal(result.consumed, true);
    assert.deepEqual(result.posts, [{ url: '/api/chat/steer/s1', body: { text: 'look at the tests dir too' } }]);
    assert.deepEqual(result.steers, [{ text: 'look at the tests dir too', user: false }]);
    assert.equal(result.composer, '');
    assert.deepEqual(result.queued, []);
  } finally { await page.close(); }
});

test('409 (no run takes steers) falls back to the existing queue', async () => {
  const page = await setup({ steerStatus: 409 });
  try {
    const result = await page.evaluate(async state => {
      queueStreamingComposerRequest();
      await new Promise(r => setTimeout(r, 20));
      return (0, eval)('(' + state + ')')();
    }, String(state));
    assert.equal(result.posts.length, 1);
    assert.deepEqual(result.queued, ['look at the tests dir too']);
    assert.deepEqual(result.steers, []);
    assert.equal(result.composer, '');
  } finally { await page.close(); }
});

test('a legacy run keeps the queue path; attachments still refuse', async () => {
  const page = await setup({ runtime: 'legacy' });
  try {
    const result = await page.evaluate(async state => {
      const consumed = queueStreamingComposerRequest();
      window.pendingFiles = 1;
      document.getElementById('message').value = 'with a file';
      const consumedWithFile = queueStreamingComposerRequest();
      await new Promise(r => setTimeout(r, 20));
      return { consumed, consumedWithFile, ...(0, eval)('(' + state + ')')() };
    }, String(state));
    assert.equal(result.consumed, true);
    assert.deepEqual(result.queued, ['look at the tests dir too']);
    assert.deepEqual(result.posts, []);
    assert.equal(result.consumedWithFile, true);
    assert.equal(result.errors.length, 1);
    assert.equal(result.composer, 'with a file');
  } finally { await page.close(); }
});

test('while a chat_agent ask waits in this chat, Enter is not consumed: the submit answers the ask (#60 gap)', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async state => {
      const card = document.createElement('div');
      card.className = 'ask-user-card';
      card.dataset.askId = 'ask-1';
      card._askUser = { ask_id: 'ask-1', session_id: 's1' };
      document.getElementById('chat-history').appendChild(card);
      const consumed = queueStreamingComposerRequest();
      card._askUser.session_id = 'other';
      const consumedOtherChat = queueStreamingComposerRequest();
      await new Promise(r => setTimeout(r, 20));
      return { consumed, consumedOtherChat, ...(0, eval)('(' + state + ')')() };
    }, String(state));
    assert.equal(result.consumed, false);
    assert.equal(result.consumedOtherChat, true);
    assert.equal(result.posts.length, 1);
    assert.equal(result.composer, '');
  } finally { await page.close(); }
});
