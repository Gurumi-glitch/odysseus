import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

// Execute the production card functions in Chromium with a recording fetch.
// Behavioral DOM tests for spec #44 asks (#60), not source-text assertions.
const renderer = await readFile(new URL('../static/js/chatRenderer.js', import.meta.url), 'utf8');
const cards = renderer
  .slice(renderer.indexOf('/** Remove any unanswered multiple-choice cards'), renderer.indexOf('/**\n * Add a message to the chat history.'))
  .replaceAll('export async function', 'async function')
  .replaceAll('export function', 'function');
let browser;
before(async () => { browser = await chromium.launch({ headless: true }); });
after(async () => { await browser?.close(); });

const permissionCard = {
  kind: 'tool_approval', approval_id: 'ask-1', ask_id: 'ask-1', ask_kind: 'permission', session_id: 's1',
  question: 'Allow this action?', description: 'The agent wants to run this action and needs your approval.',
  options: [
    { label: 'Allow once', value: 'once', description: 'Run this action now; ask again next time.' },
    { label: 'Allow for this chat', value: 'chat', description: 'Run it and allow bash for the rest of this chat.' },
    { label: 'Deny', value: 'deny', description: 'Do not run this action.' },
  ],
  action: { tool: 'bash', content: 'rm -rf build' },
};
const questionCard = {
  ask_id: 'ask-2', ask_kind: 'question', session_id: 's1', question: 'Which target?',
  options: [{ label: 'staging' }, { label: 'production' }],
};

async function setup() {
  const page = await browser.newPage();
  await page.setContent('<main id="chat-history"></main><textarea id="message"></textarea><button class="send-btn">Send</button>');
  await page.evaluate(cards => {
    const esc = value => { const node = document.createElement('div'); node.textContent = String(value ?? ''); return node.innerHTML; };
    Object.assign(window, {
      posts: [], toasts: [], sends: 0, askResponse: { ok: true, status: 200, body: { accepted: true } },
      uiModule: { esc, el: id => document.getElementById(id), showToast: value => window.toasts.push(value) },
      svgifyEmoji: value => value,
    });
    window.fetch = async (url, init) => {
      window.posts.push({ url: String(url), body: JSON.parse(init.body) });
      const r = window.askResponse;
      return new Response(JSON.stringify(r.body), { status: r.status });
    };
    document.querySelector('.send-btn').addEventListener('click', () => { window.sends++; });
    const exported = new Function(cards + '\nreturn { renderAskUserCard, resolveAskUserCard, answerChatAsk, removeAskUserCards, answerPendingAskFromComposer, hasPendingAsk };')();
    Object.assign(window, exported);
  }, cards);
  return page;
}

const cardState = () => {
  const card = document.querySelector('.ask-user-card');
  return {
    answered: card.classList.contains('ask-user-answered'),
    buttons: card.querySelectorAll('button:not(:disabled)').length,
    inputs: card.querySelectorAll('input:not(:disabled)').length,
    label: card.querySelector('.ask-user-answer-label')?.textContent || '',
    value: card.querySelector('.ask-user-answer-value')?.textContent || '',
    askId: card.dataset.askId || '',
  };
};

test('permission card: Deny posts the typed reason to the ask endpoint, nothing goes through the composer', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async ([card, cardState]) => {
      renderAskUserCard(card, { runtime: 'chat_agent' });
      const input = document.querySelector('.ask-user-other-input');
      const placeholder = input.placeholder;
      input.value = 'use a note instead';
      [...document.querySelectorAll('.ask-user-option')].find(b => b.textContent.includes('Deny')).click();
      await new Promise(r => setTimeout(r, 20));
      return { posts: window.posts, sends: window.sends, placeholder, state: (0, eval)('(' + cardState + ')')() };
    }, [permissionCard, String(cardState)]);
    assert.deepEqual(result.posts, [{ url: '/api/chat/ask/s1', body: { ask_id: 'ask-1', decision: 'deny', text: 'use a note instead' } }]);
    assert.equal(result.sends, 0);
    assert.match(result.placeholder, /reason/i);
    assert.deepEqual(result.state, { answered: true, buttons: 0, inputs: 0, label: 'Answered', value: 'Deny — use a note instead', askId: 'ask-1' });
  } finally { await page.close(); }
});

test('permission card: Enter in the reason box denies with that reason; Allow once posts once', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async card => {
      renderAskUserCard(card, { runtime: 'chat_agent' });
      const input = document.querySelector('.ask-user-other-input');
      input.value = 'not on prod';
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      renderAskUserCard({ ...card, ask_id: 'ask-9', approval_id: 'ask-9' }, { runtime: 'chat_agent' });
      [...document.querySelectorAll('.ask-user-card:not(.ask-user-answered) .ask-user-option')].find(b => b.textContent.includes('Allow once')).click();
      await new Promise(r => setTimeout(r, 20));
      return window.posts.map(p => p.body);
    }, permissionCard);
    assert.deepEqual(result, [
      { ask_id: 'ask-1', decision: 'deny', text: 'not on prod' },
      { ask_id: 'ask-9', decision: 'once', text: '' },
    ]);
  } finally { await page.close(); }
});

test('question card: an option or typed text posts decision answer', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async card => {
      renderAskUserCard(card, { runtime: 'chat_agent' });
      [...document.querySelectorAll('.ask-user-option')].find(b => b.textContent.includes('staging')).click();
      renderAskUserCard({ ...card, ask_id: 'ask-3' }, { runtime: 'chat_agent' });
      const input = document.querySelector('.ask-user-card:not(.ask-user-answered) .ask-user-other-input');
      input.value = 'the eu cluster';
      document.querySelector('.ask-user-card:not(.ask-user-answered) .ask-user-other-send').click();
      await new Promise(r => setTimeout(r, 20));
      return { posts: window.posts.map(p => p.body), sends: window.sends };
    }, questionCard);
    assert.deepEqual(result.posts, [
      { ask_id: 'ask-2', decision: 'answer', text: 'staging' },
      { ask_id: 'ask-3', decision: 'answer', text: 'the eu cluster' },
    ]);
    assert.equal(result.sends, 0);
  } finally { await page.close(); }
});

test('legacy cards keep the composer path when no runtime is given', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async card => {
      renderAskUserCard({ question: card.question, options: card.options });
      document.querySelector('.ask-user-option').click();
      await new Promise(r => setTimeout(r, 20));
      return { posts: window.posts.length, sends: window.sends, message: document.getElementById('message').value };
    }, questionCard);
    assert.deepEqual(result, { posts: 0, sends: 1, message: 'staging' });
  } finally { await page.close(); }
});

test('ask_resolved marks the live card by ask id; ask_expired says why', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([card, cardState]) => {
      const state = (0, eval)('(' + cardState + ')');
      renderAskUserCard(card, { runtime: 'chat_agent' });
      resolveAskUserCard('other-ask', { type: 'ask_resolved', decision: 'deny' });
      const untouched = state();
      resolveAskUserCard('ask-1', { type: 'ask_resolved', decision: 'chat', by: 'user' });
      const resolved = state();
      document.querySelector('#chat-history').replaceChildren();
      renderAskUserCard({ ...card, ask_id: 'ask-4' }, { runtime: 'chat_agent' });
      resolveAskUserCard('ask-4', { type: 'ask_expired', cause: 'restart' });
      return { untouched: untouched.answered, resolved, expired: state() };
    }, [permissionCard, String(cardState)]);
    assert.equal(result.untouched, false);
    assert.deepEqual(result.resolved, { answered: true, buttons: 0, inputs: 0, label: 'Answered', value: 'Allow for this chat', askId: 'ask-1' });
    assert.deepEqual(result.expired, { answered: true, buttons: 0, inputs: 0, label: 'Expired', value: 'the server restarted', askId: 'ask-4' });
  } finally { await page.close(); }
});

test('history: resolved and expired chat_agent cards render read-only, legacy resolved cards stay hidden', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([card, cardState]) => {
      const state = (0, eval)('(' + cardState + ')');
      const legacy = renderAskUserCard({ ...card, ask_id: undefined, resolved: 'approve' });
      renderAskUserCard({ ...card, resolved: 'once' }, { runtime: 'chat_agent', focus: false, scroll: false });
      const once = state();
      document.querySelector('#chat-history').replaceChildren();
      renderAskUserCard({ ...card, resolved: 'expired', expired_cause: 'timeout' }, { runtime: 'chat_agent' });
      const expired = state();
      return { legacy: legacy === null, once, expired, posts: window.posts.length };
    }, [permissionCard, String(cardState)]);
    assert.equal(result.legacy, true);
    assert.deepEqual(result.once, { answered: true, buttons: 0, inputs: 0, label: 'Answered', value: 'Allow once', askId: 'ask-1' });
    assert.deepEqual(result.expired, { answered: true, buttons: 0, inputs: 0, label: 'Expired', value: 'no answer in time', askId: 'ask-1' });
    assert.equal(result.posts, 0);
  } finally { await page.close(); }
});

// The composer path (submit handler, queue) answers through these two
// functions instead of reading the card (chat-agent-arch 06).
test('composer: text answers a pending question ask and settles its card', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async ([card, cardState]) => {
      renderAskUserCard(card, { runtime: 'chat_agent' });
      const before = hasPendingAsk('s1');
      const answered = answerPendingAskFromComposer('s1', 'the eu cluster');
      await new Promise(r => setTimeout(r, 20));
      return { before, answered, after: hasPendingAsk('s1'), posts: window.posts, sends: window.sends, state: (0, eval)('(' + cardState + ')')() };
    }, [questionCard, String(cardState)]);
    assert.equal(result.before, true);
    assert.equal(result.answered, true);
    assert.equal(result.after, false);
    assert.deepEqual(result.posts, [{ url: '/api/chat/ask/s1', body: { ask_id: 'ask-2', decision: 'answer', text: 'the eu cluster' } }]);
    assert.equal(result.sends, 0);
    assert.deepEqual(result.state, { answered: true, buttons: 0, inputs: 0, label: 'Answered', value: 'the eu cluster', askId: 'ask-2' });
  } finally { await page.close(); }
});

test('composer: text on a permission ask denies it with that reason', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async ([card, cardState]) => {
      renderAskUserCard(card, { runtime: 'chat_agent' });
      const answered = answerPendingAskFromComposer('s1', 'use a note instead');
      await new Promise(r => setTimeout(r, 20));
      return { answered, after: hasPendingAsk('s1'), posts: window.posts, sends: window.sends, state: (0, eval)('(' + cardState + ')')() };
    }, [permissionCard, String(cardState)]);
    assert.equal(result.answered, true);
    assert.equal(result.after, false);
    assert.deepEqual(result.posts, [{ url: '/api/chat/ask/s1', body: { ask_id: 'ask-1', decision: 'deny', text: 'use a note instead' } }]);
    assert.equal(result.sends, 0);
    assert.deepEqual(result.state, { answered: true, buttons: 0, inputs: 0, label: 'Answered', value: 'Deny — use a note instead', askId: 'ask-1' });
  } finally { await page.close(); }
});

test('composer: nothing is answered without a pending ask, on an answered card, or with an empty composer', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async ([card, cardState]) => {
      const none = { pending: hasPendingAsk('s1'), answered: answerPendingAskFromComposer('s1', 'hello') };
      renderAskUserCard(card, { runtime: 'chat_agent' });
      const empty = answerPendingAskFromComposer('s1', '');
      resolveAskUserCard('ask-1', { type: 'ask_resolved', decision: 'once', by: 'user' });
      const settled = { pending: hasPendingAsk('s1'), answered: answerPendingAskFromComposer('s1', 'hello') };
      await new Promise(r => setTimeout(r, 20));
      return { none, empty, settled, posts: window.posts, state: (0, eval)('(' + cardState + ')')() };
    }, [permissionCard, String(cardState)]);
    assert.deepEqual(result.none, { pending: false, answered: false });
    assert.equal(result.empty, false);
    assert.deepEqual(result.settled, { pending: false, answered: false });
    assert.deepEqual(result.posts, []);
    assert.equal(result.state.value, 'Allow once');
  } finally { await page.close(); }
});

test('composer: a pending ask in another chat is neither reported nor answered', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async ([card, cardState]) => {
      renderAskUserCard(card, { runtime: 'chat_agent' });
      const answered = answerPendingAskFromComposer('other', 'hello');
      await new Promise(r => setTimeout(r, 20));
      return { other: hasPendingAsk('other'), own: hasPendingAsk('s1'), answered, posts: window.posts, state: (0, eval)('(' + cardState + ')')() };
    }, [questionCard, String(cardState)]);
    assert.equal(result.other, false);
    assert.equal(result.own, true);
    assert.equal(result.answered, false);
    assert.deepEqual(result.posts, []);
    assert.equal(result.state.answered, false);
  } finally { await page.close(); }
});

test('a refused answer surfaces the server detail as a toast', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(async () => {
      window.askResponse = { status: 409, body: { detail: 'No open ask matches this answer.' } };
      await answerChatAsk('s1', 'ask-1', 'once', '');
      return window.toasts;
    });
    assert.deepEqual(result, ['No open ask matches this answer.']);
  } finally { await page.close(); }
});
