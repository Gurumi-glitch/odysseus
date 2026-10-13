import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

// chat_agent rows render from their rounds: history renderer and resume
// reconcile (spec #44 Transcript, #65). Same harness as historyResumeRendering.
const renderer = await readFile(new URL('../static/js/chatRenderer.js', import.meta.url), 'utf8');
const chat = await readFile(new URL('../static/js/chat.js', import.meta.url), 'utf8');
const addMessage = renderer.slice(renderer.indexOf('export function addMessage('), renderer.indexOf('\nconst chatRenderer =')).replace('export ', '');
const askCards = renderer
  .slice(renderer.indexOf('/**\n * A steering message'), renderer.indexOf('/**\n * Add a message to the chat history.'))
  .replaceAll('export async function', 'async function')
  .replaceAll('export function', 'function');
const resume = chat.slice(chat.indexOf('export async function resumeStream('), chat.indexOf('\n  /**\n   * Check for background streams')).replace('export ', '');
// The auto mode denial line on a tool card (#68).
const toolDenial = renderer.slice(renderer.indexOf('export function toolDenialHtml('), renderer.indexOf('function _suppressRawToolOutput(')).replace('export ', '');
let browser;
before(async () => { browser = await chromium.launch({ headless: true }); });
after(async () => { await browser?.close(); });

async function setup() {
  const page = await browser.newPage();
  await page.route('http://render.test/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/') return route.fulfill({ contentType: 'text/html', body: '<main id="chat-history"></main><textarea id="message"></textarea><button class="send-btn">Send</button>' });
    if (path === '/static/js/ui.js') return route.fulfill({ contentType: 'text/javascript', body: 'export default window.uiModule;' });
    if (!path.startsWith('/static/')) return route.abort();
    const source = await readFile(new URL('..' + path, import.meta.url), 'utf8');
    return route.fulfill({ contentType: 'text/javascript', body: source });
  });
  await page.goto('http://render.test/');
  await page.evaluate(async ({ addMessage, resume, askCards, toolDenial }) => {
    const noop = () => {};
    const esc = value => { const node = document.createElement('div'); node.textContent = String(value ?? ''); return node.innerHTML; };
    Object.assign(window, {
      uiModule: { esc, scrollHistory: noop, showToast: noop, showError: error => { throw Error(error); }, el: id => document.getElementById(id), captureHistoryScroll: () => ({ y: scrollY }), restoreHistoryScroll: s => scrollTo(0, s.y) },
      svgifyEmoji: x => x, CHAT_AGENT_RUNTIME: 'chat_agent', resumeRuntime: 'chat_agent', posts: [], sends: 0,
      hideWelcomeScreen: noop, resolveDocumentPlaceholderLinks: x => x,
      replyModelPair: () => ({ actualModel: 'test', requestedModel: 'test' }),
      modelRouteLabel: () => 'test', sameModelName: () => true, applyModelColor: noop,
      roleTimestamp: () => document.createElement('span'),
      createMsgFooter: () => { const node = document.createElement('div'); node.className = 'msg-footer'; node.textContent = 'Copy'; return node; },
      displayMetrics: noop,
      _suppressRawToolOutput: () => false, safeToolScreenshotSrc: () => '', _isPrivateBrowserTool: () => false,
      _toolDisplayInfo: () => ({}), renderToolIcon: () => '',
      buildSourcesBox: () => '', buildFindingsBox: () => '', buildRagSourcesBox: () => '',
      API_BASE: '', hasActiveStream: () => false,
      _streamRunIds: new Map(), _streamRuntimes: new Map(), _resumingStreams: new Set(), _backgroundStreams: new Map(),
      updateSubmitButton: noop, _shortModel: x => x, _applyModelColor: noop,
      _streamDisplayText: x => x, _finishDocumentWritingStatus: noop, _setRoleModelLabel: noop,
      spinnerModule: { create: () => { const node = document.createElement('span'); return { createElement: () => node, start: noop, destroy: () => node.remove() }; } },
      currentSession: 's', savedHistory: [],
      _metricsCostRecordId: () => 'test-run',
      sessionModule: { getCurrentSessionId: () => window.currentSession, getSessions: () => [{ id: 's', model: 'test' }], markStreaming: noop, clearStreaming: noop },
    });
    document.querySelector('.send-btn').addEventListener('click', () => { window.sends++; });
    window.markdownModule = (await import('/static/js/markdown.js')).default;
    markdownModule.renderMermaid = undefined;
    window.createTurnRendering = (await import('/static/js/turnRendering.js')).createTurnRendering;
    window.startsContinuationRound = (await import('/static/js/turnRendering.js')).startsContinuationRound;
    window.applyModelRouteEventState = (await import('/static/js/chatModelProvenance.js')).applyModelRouteEventState;
    window.createTerminalStreamError = (await import('/static/js/chatStreamErrors.js')).createTerminalStreamError;
    window.generatedImageResult = (await import('/static/js/generatedImageResult.js')).generatedImageResult;
    const cards = new Function(askCards + '\nreturn { renderSteerBubble, renderAskUserCard, resolveAskUserCard, answerChatAsk, removeAskUserCards, renderTurnStopNote, renderTurnNote, takePendingTurnSource };')();
    Object.assign(window, cards);
    window.addMessage = (0, eval)('(' + addMessage + ')');
    window.resumeStream = (0, eval)('(' + resume + ')');
    window.toolDenialHtml = (0, eval)('(' + toolDenial + ')');
    window.chatRenderer = { addMessage: window.addMessage, recordSessionMetricsCost: noop, buildSourcesBox, buildFindingsBox, buildRagSourcesBox, toolDenialHtml: window.toolDenialHtml, ...cards };
    let controller;
    const stream = new ReadableStream({ start(value) { controller = value; } });
    window.send = event => controller.enqueue(new TextEncoder().encode('data: ' + (typeof event === 'string' ? event : JSON.stringify(event)) + '\n\n'));
    window.fetch = async (url) => {
      if (String(url).includes('/api/chat/resume/')) return new Response(stream, { headers: { 'X-Odysseus-Run-Id': 'run-1', 'X-Odysseus-Runtime': window.resumeRuntime } });
      return new Response(JSON.stringify({ history: window.savedHistory }));
    };
  }, { addMessage, resume, askCards, toolDenial });
  return page;
}

// What the chat page shows for one turn, in document order.
const timeline = () => [...document.querySelector('#chat-history').children].map(node => {
  if (node.classList.contains('msg-ai')) return 'ai:' + node.querySelector('.body').textContent.trim();
  if (node.classList.contains('msg-steer')) return 'steer:' + node.querySelector('.body').textContent.trim();
  if (node.classList.contains('agent-thread')) return 'tools:' + [...node.querySelectorAll('.agent-thread-node')].map(n => n.querySelector('.agent-thread-status')?.textContent + (n.classList.contains('error') ? '!' : '')).join(',');
  if (node.classList.contains('stopped-indicator')) return 'stop:' + node.querySelector('span').textContent + (node.querySelector('button') ? '+continue' : '');
  return node.className;
});

const threeRounds = {
  _fromHistory: true, _db_id: 7, runtime: 'chat_agent', model: 'test',
  round_texts: ['Let me look at the workspace.', 'Now the tests.', 'All done: two files.'],
  tool_events: [
    { round: 1, tool: 'bash', command: 'ls', output: 'a b', exit_code: 0, status: 'done', call_id: 'c1' },
    { round: 2, tool: 'bash', command: 'rm -rf build', output: 'denied', exit_code: 126, status: 'denied', call_id: 'c2' },
  ],
  steers: [{ round: 1, text: 'also check the tests dir' }],
};
const content = 'Let me look at the workspace.\n\n> ↳ also check the tests dir\n\nNow the tests.\n\nAll done: two files.';

test('history: a chat_agent row draws one bubble per round, the steer after its round, tool status as stored, no duplicated text', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([meta, content, timeline]) => {
      addMessage('assistant', content, 'test', meta);
      const text = document.querySelector('#chat-history').textContent;
      return { timeline: (0, eval)('(' + timeline + ')')(), count: text.split('Let me look').length - 1, text };
    }, [threeRounds, content, String(timeline)]);
    assert.deepEqual(result.timeline, [
      'ai:Let me look at the workspace.', 'tools:done', 'steer:also check the tests dir',
      'ai:Now the tests.', 'tools:denied!', 'ai:All done: two files.',
    ]);
    assert.equal(result.count, 1);
    assert.doesNotMatch(result.text, /↳/);
  } finally { await page.close(); }
});

test('history: a pending tool reads pending, a cancelled one cancelled; neither is a tick', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(timeline => {
      addMessage('assistant', 'Checking', 'test', { _fromHistory: true, runtime: 'chat_agent', in_progress: true, round_texts: ['Checking'],
        tool_events: [{ round: 1, tool: 'bash', command: 'ls', status: 'pending', call_id: 'c1' }, { round: 1, tool: 'bash', command: 'pwd', status: 'cancelled', exit_code: 130, call_id: 'c2' }] });
      return { timeline: (0, eval)('(' + timeline + ')')(), ticks: document.querySelectorAll('.agent-thread-icon').length ? [...document.querySelectorAll('.agent-thread-icon')].map(n => n.textContent) : [] };
    }, String(timeline));
    assert.deepEqual(result.timeline, ['ai:Checking', 'tools:pending,cancelled!']);
    assert.ok(!result.ticks.includes('✓'), 'no tick: ' + result.ticks.join(','));
  } finally { await page.close(); }
});

test('history: stop reasons — max_rounds offers Continue as a new turn, interrupted only marks', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([meta, content, timeline]) => {
      addMessage('assistant', content, 'test', { ...meta, stop_reason: 'max_rounds' });
      const capped = (0, eval)('(' + timeline + ')')();
      document.querySelector('.stopped-indicator button').click();
      const composer = document.getElementById('message').value;
      const sends = window.sends;
      document.querySelector('#chat-history').replaceChildren();
      addMessage('assistant', 'Half', 'test', { _fromHistory: true, runtime: 'chat_agent', round_texts: ['Half'], stop_reason: 'interrupted',
        tool_events: [{ round: 1, tool: 'bash', command: 'ls', status: 'cancelled', exit_code: 130 }] });
      return { capped: capped.at(-1), composer, sends, interrupted: (0, eval)('(' + timeline + ')')() };
    }, [threeRounds, content, String(timeline)]);
    assert.match(result.capped, /^stop:.*step limit.*\+continue$/);
    assert.ok(result.composer.length > 0);
    assert.equal(result.sends, 1);
    assert.deepEqual(result.interrupted, ['ai:Half', 'tools:cancelled!', 'stop:[Message interrupted]']);
  } finally { await page.close(); }
});

const CONTINUE_TEXT = 'Continue from exactly where you left off and keep going until the task is done. Do not repeat work already done.';

test('history: Continue on a capped row marks the next send as a continuation (source=continue), as the live note does', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([meta, content]) => {
      addMessage('assistant', content, 'test', { ...meta, stop_reason: 'max_rounds' });
      const before = takePendingTurnSource();
      document.querySelector('.stopped-indicator .continue-btn').click();
      // The marker belongs to the one send Continue triggers: read once, then gone.
      return { before, source: takePendingTurnSource(), again: takePendingTurnSource(), sends: window.sends, composer: document.getElementById('message').value, notes: document.querySelectorAll('.stopped-indicator').length };
    }, [threeRounds, content]);
    assert.deepEqual(result, { before: '', source: 'continue', again: '', sends: 1, composer: CONTINUE_TEXT, notes: 0 });
  } finally { await page.close(); }
});

test('one renderer: the live round-cap note, the live budget note and the history notes are the same DOM apart from the label words', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([meta, content]) => {
      const box = document.querySelector('#chat-history');
      const history = stopReason => { box.replaceChildren(); addMessage('assistant', content, 'test', { ...meta, stop_reason: stopReason }); return box.querySelector('.stopped-indicator'); };
      // The live stream passes the cap event (loop.py: rounds_exhausted carries rounds, budget_exceeded used/limit).
      const notes = {
        liveRounds: renderTurnStopNote('max_rounds', { type: 'rounds_exhausted', rounds: 10 }),
        liveBudget: renderTurnStopNote('budget', { type: 'budget_exceeded', kind: 'tool_calls', used: 8, limit: 8 }),
        historyRounds: history('max_rounds'),
        historyBudget: history('budget'),
      };
      const shape = note => { const copy = note.cloneNode(true); copy.querySelector('.rounds-exhausted-label').textContent = '*'; return copy.outerHTML; };
      const shapes = Object.values(notes).map(shape);
      const labels = Object.fromEntries(Object.entries(notes).map(([k, n]) => [k, n.querySelector('.rounds-exhausted-label').textContent]));
      // Continue on the live-shaped note takes the same path as history: marker, composer text, send.
      box.replaceChildren(notes.liveBudget);
      notes.liveBudget.querySelector('.continue-btn').click();
      return { shape: shapes[0], same: shapes.every(s => s === shapes[0]), labels, source: takePendingTurnSource(), composer: document.getElementById('message').value, sends: window.sends };
    }, [threeRounds, content]);
    assert.equal(result.shape, '<div class="stopped-indicator rounds-exhausted"><span class="rounds-exhausted-label">*</span><button class="continue-btn" title="Continue the task in a new turn">Continue ▸</button></div>');
    assert.ok(result.same);
    assert.deepEqual(result.labels, {
      liveRounds: 'Reached the 10-step limit — not finished.',
      liveBudget: 'Tool budget reached (8/8 calls) — not finished.',
      historyRounds: 'Reached the step limit — not finished.',
      historyBudget: 'Reached the tool budget — not finished.',
    });
    assert.deepEqual({ source: result.source, composer: result.composer, sends: result.sends }, { source: 'continue', composer: CONTINUE_TEXT, sends: 1 });
  } finally { await page.close(); }
});

test('history: a stopped chat_agent row never offers the legacy Resume (the merge-last-assistant path); a legacy row still does', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([meta, content]) => {
      // The generic (single-bubble) renderer path the bare and legacy rows take; the harness only wires the rounds path.
      Object.assign(window, { stripToolBlocks: x => x, _renderAssistantBody: x => markdownModule.processWithThinking(x) });
      const render = (text, metadata) => {
        document.querySelector('#chat-history').replaceChildren();
        addMessage('assistant', text, 'test', metadata);
        return {
          markers: [...document.querySelectorAll('.stopped-indicator')].map(n => n.querySelector('span').textContent),
          resume: document.querySelectorAll('.resume-btn').length,
        };
      };
      return {
        // What transcript.TurnWriter stores for an interrupted turn: stop_reason plus the legacy `stopped` marker.
        stored: render(content, { ...meta, stop_reason: 'interrupted', stopped: true }),
        // The same marker on a chat_agent row with no rounds to replay falls through to the generic renderer.
        bare: render('Half', { _fromHistory: true, runtime: 'chat_agent', stop_reason: 'interrupted', stopped: true }),
        legacy: render('Half', { _fromHistory: true, stopped: true }),
      };
    }, [threeRounds, content]);
    assert.deepEqual(result.stored, { markers: ['[Message interrupted]'], resume: 0 });
    assert.deepEqual(result.bare, { markers: ['[Message interrupted]'], resume: 0 });
    assert.deepEqual(result.legacy, { markers: ['[Message interrupted]'], resume: 1 });
  } finally { await page.close(); }
});

test('history: a stored turn error replaces the generic marker, the output-limit notice sits under the kept text, rows without either render as before (#98)', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([meta, content, timeline]) => {
      Object.assign(window, { stripToolBlocks: x => x, _renderAssistantBody: x => markdownModule.processWithThinking(x) });
      const render = (text, metadata) => {
        document.querySelector('#chat-history').replaceChildren();
        addMessage('assistant', text, 'test', metadata);
        return {
          markers: [...document.querySelectorAll('.stopped-indicator')].map(n => n.textContent),
          timeline: (0, eval)('(' + timeline + ')')(),
          resume: document.querySelectorAll('.resume-btn').length,
        };
      };
      const failed = { _fromHistory: true, runtime: 'chat_agent', stop_reason: 'error', stopped: true, error: 'Stopped: m produced degenerate output (repeating tokens).' };
      return {
        // A failure after a few rounds: the rounds replay, the note carries the error text.
        rounds: render(content, { ...meta, ...failed }),
        // A round-1 failure saves a row with no rounds and no content (generic renderer path).
        bare: render('', { ...failed, round_texts: [], agent_v2: { rounds: [] } }),
        // An error row that carries no text keeps the generic wording.
        noText: render(content, { ...meta, stop_reason: 'error', stopped: true }),
        // Mid-answer cut: kept text plus the notice, no stop marker.
        notice: render('Half an answ', { _fromHistory: true, runtime: 'chat_agent', round_texts: ['Half an answ'], notice: 'stopped at output limit' }),
        plain: render('Fine', { _fromHistory: true, runtime: 'chat_agent', round_texts: ['Fine'] }),
        legacy: render('Half', { _fromHistory: true, stopped: true, error: 'x', notice: 'y' }),
      };
    }, [threeRounds, content, String(timeline)]);
    const err = '[Stopped: m produced degenerate output (repeating tokens).]';
    assert.deepEqual(result.rounds.markers, ['Stopped: m produced degenerate output (repeating tokens).']);
    assert.equal(result.rounds.timeline.at(-1), 'stop:Stopped: m produced degenerate output (repeating tokens).');
    assert.deepEqual(result.bare, { markers: ['Stopped: m produced degenerate output (repeating tokens).'], timeline: ['ai:Stopped: m produced degenerate output (repeating tokens).'], resume: 0 }, err);
    assert.deepEqual(result.noText.markers, ['[The turn failed]']);
    assert.deepEqual(result.notice.markers, ['stopped at output limit']);
    assert.deepEqual(result.notice.timeline, ['ai:Half an answ', 'stop:stopped at output limit']);
    assert.deepEqual(result.plain.markers, []);
    assert.deepEqual(result.legacy.markers, ['[Message interrupted]Resume']);
  } finally { await page.close(); }
});

test('history: a Shorter/Simpler rewrite (chosen variant) replaces the last round after a reload', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([meta, content, timeline]) => {
      addMessage('assistant', content, 'test', { ...meta, variants: [{ raw: 'All done: two files.', label: 'original' }, { raw: 'Shorter: two files.', label: 'shorter' }], variantIndex: 1 });
      return (0, eval)('(' + timeline + ')')();
    }, [threeRounds, content, String(timeline)]);
    assert.deepEqual(result, [
      'ai:Let me look at the workspace.', 'tools:done', 'steer:also check the tests dir',
      'ai:Now the tests.', 'tools:denied!', 'ai:Shorter: two files.',
    ]);
  } finally { await page.close(); }
});

test('history: an edited chat_agent row renders its content once', async () => {
  const page = await setup();
  try {
    const result = await page.evaluate(([meta, timeline]) => {
      addMessage('assistant', 'Edited by hand', 'test', { ...meta, edited: true });
      return { timeline: (0, eval)('(' + timeline + ')')(), edited: !!document.querySelector('.edited-indicator') };
    }, [threeRounds, String(timeline)]);
    assert.equal(result.timeline.filter(x => x.startsWith('ai:')).length, 1);
    assert.ok(result.timeline.includes('ai:Edited by hand'), result.timeline.join(' | '));
  } finally { await page.close(); }
});

test('resume on a chat_agent run replaces the in-progress history row and rebuilds the turn from its rounds at the end', async () => {
  const page = await setup();
  try {
    await page.evaluate(() => {
      const user = document.createElement('div');
      user.className = 'msg msg-user';
      user.innerHTML = '<div class="body">list the files</div>';
      document.querySelector('#chat-history').appendChild(user);
      addMessage('assistant', 'Let me look at the workspace.', 'test', { _fromHistory: true, _db_id: 7, runtime: 'chat_agent', in_progress: true,
        round_texts: ['Let me look at the workspace.'], tool_events: [{ round: 1, tool: 'bash', command: 'ls', status: 'pending', call_id: 'c1' }] });
      window.running = resumeStream('s');
    });
    await page.waitForSelector('.msg-ai .stream-content', { state: 'attached' });
    await page.evaluate(() => {
      send({ delta: 'Let me look at the workspace.' });
      send({ type: 'tool_start', tool: 'bash', command: 'ls', round: 1, call_id: 'c1' });
      send({ type: 'tool_output', tool: 'bash', command: 'ls', output: 'a b', exit_code: 0, call_id: 'c1' });
      send({ type: 'agent_step', round: 2 });
      send({ delta: 'Now the tests.' });
    });
    await page.waitForFunction(() => document.querySelector('#chat-history').textContent.includes('Now the tests.'));
    const during = await page.evaluate(() => document.querySelector('#chat-history').textContent.split('Let me look').length - 1);
    assert.equal(during, 1, 'the history copy of the in-progress row is gone while the replay runs');
    const result = await page.evaluate(async ([meta, content, timeline]) => {
      window.savedHistory = [{ role: 'assistant', content, metadata: { ...meta, _fromHistory: undefined } }];
      send({ type: 'final_response', content: 'All done: two files.' });
      send({ type: 'message_saved', id: 7 });
      send('[DONE]');
      await running;
      return { timeline: (0, eval)('(' + timeline + ')')(), streaming: document.querySelectorAll('.streaming').length };
    }, [threeRounds, content, String(timeline)]);
    assert.deepEqual(result.timeline, [
      'msg msg-user', 'ai:Let me look at the workspace.', 'tools:done', 'steer:also check the tests dir',
      'ai:Now the tests.', 'tools:denied!', 'ai:All done: two files.',
    ]);
    assert.equal(result.streaming, 0);
  } finally { await page.close(); }
});
