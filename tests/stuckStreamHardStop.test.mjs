import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// A chat whose server run is still live is rejoined through resumeStream,
// which raises the module-wide isStreaming flag without a stream registry
// entry. Leaving that chat (New Chat, another chat, delete) must drop the
// flag, or every send in the next chat queues behind the stuck one; and the
// queued bubble's "send now" must stop through the submit handler, not the
// send button, whose idle look is the New Chat button.
// Same slice-by-marker harness as chatMemoryPillFooter.test.mjs; no browser.
const chat = await readFile(new URL('../static/js/chat.js', import.meta.url), 'utf8');
const sessions = await readFile(new URL('../static/js/sessions.js', import.meta.url), 'utf8');

function slice(src, startMarker, endMarker) {
  const start = src.indexOf(startMarker);
  const end = src.indexOf(endMarker, start);
  assert.ok(start >= 0 && end > start, `block not found: ${startMarker}`);
  return src.slice(start, end).replace(/^\s*export /gm, '');
}

const chatSource = [
  slice(chat, '  function updateSubmitButton(state, submitBtn)', '  // -----------------------------------------------------------------------\n  // Slash commands'),
  slice(chat, '  function _getForegroundStreamState()', '  function _touchStreamActivity('),
  slice(chat, '  export function abortCurrentRequest(', '  // ── Stall watchdog'),
  // detachCurrentStream, and forgetSessionStreams once it exists (absent on the
  // pre-fix source, so those tests fail there instead of the whole suite).
  slice(chat, '  export function detachCurrentStream(', '  // _notifyStreamComplete and _insertStreamDoneToast'),
  slice(chat, '  function _queuedSessionId()', '  function _restoreQueuedRequestsForCurrentSession()'),
  slice(chat, '  function _removeQueuedRequest(id)', '  // A fallback answered and the server moved the session onto it'),
  slice(chat, '  function _promoteQueuedRequest(id)', '  function _queueAgentRequest(message)'),
].join('\n');

function fakeButton() {
  return { classList: { add() {}, remove() {} }, dataset: {}, innerHTML: '', title: '', offsetWidth: 0, addEventListener() {} };
}

function fakeStorage() {
  const map = new Map();
  return { getItem: k => (map.has(k) ? map.get(k) : null), setItem: (k, v) => map.set(k, String(v)), removeItem: k => map.delete(k) };
}

// Builds the sliced chat.js functions over fresh module state. Free
// identifiers resolve to the `var`s declared here, so assignments inside the
// production code (isStreaming = ..., _streamSessionId = null) land on them.
function buildChat(overrides = {}) {
  const sendBtn = fakeButton();
  const calls = [];
  const scope = {
    sendBtn,
    calls,
    currentSessionId: null,
    activeStreams: new Map(),
    backgroundStreams: new Map(),
    sendStates: new Map(),
    queued: [],
    localStorage: fakeStorage(),
    ...overrides,
  };
  const body = `
    var isStreaming = false, currentAbort = null, currentHolder = null, currentAccumulated = '', _sendInFlight = false;
    var forgetSessionStreams;
    var _streamSessionId = null, _lastReaderActivity = 0, _stallWatchdog = null, _queuedBubbleHost = null;
    var _activeStreams = scope.activeStreams, _backgroundStreams = scope.backgroundStreams, _sendStates = scope.sendStates;
    var _resumingStreams = new Set(), _terminalSavedStreams = new Set();
    var _streamRunIds = new Map(), _streamRuntimes = new Map();
    var _queuedAgentRequests = scope.queued;
    var QUEUED_AGENT_REQUESTS_KEY = 'odysseus-queued-agent-requests-v1';
    var localStorage = scope.localStorage;
    var setTimeout = () => 0;
    var CustomEvent = function (type, init) { this.type = type; this.detail = init && init.detail; };
    var window = { dispatchEvent() {}, _odysseusBtnIcons: null };
    var document = {
      querySelector: sel => (sel === '.send-btn' ? scope.sendBtn : null),
      querySelectorAll: () => [],
      getElementById: () => null,
      body: { classList: { toggle() {} } },
    };
    var uiModule = { el: () => null, showToast: () => {}, showError: () => {} };
    var sessionModule = { getCurrentSessionId: () => scope.currentSessionId, markStreaming() {}, clearStreaming() {} };
    var _setForegroundChatBusy = active => scope.calls.push(['busy', !!active]);
    var _startStallWatchdog = () => scope.calls.push(['watchdog', 'start']);
    var _stopStallWatchdog = () => scope.calls.push(['watchdog', 'stop']);
    var _stopExactRun = () => { throw new Error('server stop must not be posted here'); };
    var handleChatSubmit = scope.handleChatSubmit || (async () => scope.calls.push(['submit']));
    var _sendQueuedWhenIdle = item => scope.calls.push(['send-when-idle', item && item.message]);
    var console = { error() {} };
    ${chatSource}
    return {
      updateSubmitButton, detachCurrentStream, forgetSessionStreams, _promoteQueuedRequest,
      get isStreaming() { return isStreaming; },
      get streamSessionId() { return _streamSessionId; },
      resumingStreams: _resumingStreams, streamRunIds: _streamRunIds, streamRuntimes: _streamRuntimes,
      setStreamSessionId(v) { _streamSessionId = v; },
    };
  `;
  const mod = new Function('scope', body)(scope);
  return { mod, scope, sendBtn, calls };
}

test('leaving a chat rejoined by resumeStream drops the foreground streaming flag', () => {
  const { mod, scope, sendBtn, calls } = buildChat({ currentSessionId: 'A' });
  // resumeStream raises the flag this way and registers nothing in _activeStreams.
  mod.updateSubmitButton('streaming', sendBtn);
  assert.equal(mod.isStreaming, true);
  calls.length = 0;

  mod.detachCurrentStream('A');   // createDirectChat / selectSession leave path

  assert.equal(mod.isStreaming, false, 'the next chat would queue every send behind the old one');
  assert.equal(sendBtn.dataset.mode, '');
  assert.deepEqual(calls.filter(c => c[0] === 'busy').pop(), ['busy', false]);
  assert.ok(calls.some(c => c[0] === 'watchdog' && c[1] === 'stop'));
  assert.equal(scope.backgroundStreams.size, 0, 'nothing to background: no registered reader');
});

test('leaving a chat with a registered POST stream still backgrounds it', () => {
  const { mod, scope, sendBtn } = buildChat({ currentSessionId: 'A' });
  const abortCtrl = new AbortController();
  scope.activeStreams.set('A', { abortCtrl, holder: null });
  mod.updateSubmitButton('streaming', sendBtn);

  mod.detachCurrentStream('A');

  assert.equal(scope.backgroundStreams.get('A')?.status, 'running');
  assert.equal(abortCtrl._reason, 'detach');
  assert.equal(abortCtrl.signal.aborted, true);
  assert.equal(mod.isStreaming, false);
});

test('forgetting a deleted chat aborts its readers, drops its queue and idles an emptied foreground', () => {
  const { mod, scope, sendBtn } = buildChat({ currentSessionId: null });
  const bgCtrl = new AbortController();
  const sendCtrl = new AbortController();
  scope.backgroundStreams.set('A', { abortCtrl: bgCtrl, status: 'running' });
  scope.sendStates.set('A', { generation: 3, abortCtrl: sendCtrl });
  mod.resumingStreams.add('A');
  mod.streamRunIds.set('A', 'run-1');
  mod.streamRuntimes.set('A', 'chat_agent');
  mod.setStreamSessionId('A');
  const bubble = { parentNode: {}, remove() { this.removed = true; } };
  scope.queued.push({ id: 'q1', message: 'still waiting', createdAt: 1, sessionId: 'A', el: bubble });
  scope.queued.push({ id: 'q2', message: 'other chat', createdAt: 2, sessionId: 'B', el: null });
  scope.localStorage.setItem('odysseus-queued-agent-requests-v1', JSON.stringify({ A: [{ id: 'q1', message: 'still waiting', createdAt: 1 }] }));
  mod.updateSubmitButton('streaming', sendBtn);   // stale flag from the deleted chat's resumed reader

  mod.forgetSessionStreams('A');

  assert.equal(bgCtrl.signal.aborted, true);
  assert.equal(bgCtrl._reason, 'user-stop');
  assert.equal(sendCtrl.signal.aborted, true);
  assert.equal(scope.backgroundStreams.has('A'), false);
  assert.equal(scope.sendStates.has('A'), false);
  assert.equal(mod.resumingStreams.has('A'), false);
  assert.equal(mod.streamRunIds.has('A'), false);
  assert.equal(mod.streamRuntimes.has('A'), false);
  assert.equal(mod.streamSessionId, null);
  assert.deepEqual(scope.queued.map(q => q.id), ['q2'], 'only the deleted chat loses its queued messages');
  assert.equal(bubble.removed, true);
  assert.deepEqual(
    JSON.parse(scope.localStorage.getItem('odysseus-queued-agent-requests-v1')),
    { B: [{ id: 'q2', message: 'other chat', createdAt: 2 }] },
    'the deleted chat leaves no persisted queue row to restore',
  );
  assert.equal(mod.isStreaming, false);
  assert.equal(sendBtn.dataset.mode, '');
});

test('forgetting another chat leaves the streaming foreground chat alone', () => {
  const { mod, scope, sendBtn } = buildChat({ currentSessionId: 'B' });
  scope.activeStreams.set('B', { abortCtrl: new AbortController(), holder: null });
  scope.backgroundStreams.set('A', { abortCtrl: new AbortController(), status: 'running' });
  mod.updateSubmitButton('streaming', sendBtn);

  mod.forgetSessionStreams('A');

  assert.equal(scope.backgroundStreams.has('A'), false);
  assert.equal(scope.activeStreams.has('B'), true);
  assert.equal(mod.isStreaming, true);
  assert.equal(sendBtn.dataset.mode, 'streaming');
});

test('send-now on a queued bubble stops through the submit handler, never the send button', async () => {
  const { mod, scope, sendBtn, calls } = buildChat({ currentSessionId: '' });
  let clicked = 0;
  sendBtn.click = () => { clicked += 1; };   // in newchat mode this click wipes the chat
  scope.queued.push({ id: 'q1', message: 'hello', createdAt: 1, sessionId: '', el: null });
  mod.updateSubmitButton('streaming', sendBtn);
  calls.length = 0;

  mod._promoteQueuedRequest('q1');
  await Promise.resolve();

  assert.equal(clicked, 0);
  assert.deepEqual(calls.filter(c => c[0] !== 'busy' && c[0] !== 'watchdog'), [['submit'], ['send-when-idle', 'hello']]);
  assert.equal(scope.queued.length, 0);
});

test('removing a deleted session from local state forgets its streams first', () => {
  const src = slice(sessions, 'function _removeSessionFromLocalState(sid)', 'function _normalizeSessionsList(');
  const calls = [];
  const run = new Function('calls', `
    var sessions = [{ id: 'A' }, { id: 'B' }];
    var _selectedIds = new Set(['A']);
    var Storage = { get: () => null, set() {} };
    var document = { querySelectorAll: () => [] };
    var window = { chatModule: { forgetSessionStreams: sid => calls.push(['forget', sid]) } };
    var _deselectCurrentSession = sid => calls.push(['deselect', sid]);
    var console = { warn() {} };
    ${src}
    _removeSessionFromLocalState('A');
    return sessions;
  `);
  const remaining = run(calls);
  assert.deepEqual(calls, [['forget', 'A'], ['deselect', 'A']]);
  assert.deepEqual(remaining, [{ id: 'B' }]);
});
