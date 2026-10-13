import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// A resumed chat_agent run (GET /api/chat/resume) whose SSE connection ends
// before [DONE] while the run is still going on the server must re-attach,
// not finalize the partial turn: the user saw the view cut off at a running
// tool card, with the answer never drawn (the live reader already recovers
// through _tryAutoRecover). Same slice-by-marker harness as
// stuckStreamHardStop.test.mjs; the whole resumeStream function runs over a
// small fake DOM, no browser.
const chat = await readFile(new URL('../static/js/chat.js', import.meta.url), 'utf8');

function slice(src, startMarker, endMarker) {
  const start = src.indexOf(startMarker);
  const end = src.indexOf(endMarker, start);
  assert.ok(start >= 0 && end > start, `block not found: ${startMarker}`);
  return src.slice(start, end).replace(/^\s*export /gm, '');
}

const resumeSource = slice(chat, '  export async function resumeStream(sessionId, replaceHolder = null) {', '  /**\n   * Check for background streams when switching to a session.');

// ── fake DOM: enough of Element for the resume renderer ──────────────────
function el(tag) {
  const n = {
    tag, nodeType: 1, children: [], parentNode: null, className: '', style: {}, dataset: {},
    isConnected: false, _inner: '',
    classList: {
      add(c) { if (!n.classList.contains(c)) n.className = (n.className + ' ' + c).trim(); },
      remove(c) { n.className = n.className.split(' ').filter(x => x !== c).join(' '); },
      contains: c => n.className.split(' ').includes(c),
    },
    get innerHTML() { return n._inner; },
    // The renderer builds its bubbles from HTML strings; materialize the few
    // classed children it later queries.
    set innerHTML(v) {
      n._inner = v; n.children = [];
      for (const cls of ['role', 'body', 'agent-thread-content', 'agent-thread-header']) {
        if (v.includes(`class="${cls}"`)) {
          const child = el('div'); child.className = cls; n.appendChild(child);
          if (cls === 'body' && v.includes('class="stream-content"')) {
            const sc = el('div'); sc.className = 'stream-content'; child.appendChild(sc);
          }
        }
      }
    },
    appendChild(c) { c.parentNode = n; setConnected(c, n.isConnected); n.children.push(c); return c; },
    insertAdjacentHTML() {},
    remove() { if (n.parentNode) { n.parentNode.children = n.parentNode.children.filter(x => x !== n); n.parentNode = null; setConnected(n, false); } },
    querySelector: sel => n.querySelectorAll(sel)[0] || null,
    querySelectorAll(sel) {
      const cls = sel.replace(/^\./, '');
      const out = [];
      (function walk(node) { for (const c of node.children || []) { if (c.classList?.contains(cls)) out.push(c); walk(c); } })(n);
      return out;
    },
    get nextSibling() { const p = n.parentNode; if (!p) return null; return p.children[p.children.indexOf(n) + 1] || null; },
    get textContent() { return n._text || ''; },
    set textContent(v) { n._text = v; },
    compareDocumentPosition: () => 0,
  };
  return n;
}
function setConnected(node, value) { node.isConnected = value; for (const c of node.children || []) setConnected(c, value); }
function comment(data) { return { nodeType: 8, data, parentNode: null, children: [], remove() { if (this.parentNode) { this.parentNode.children = this.parentNode.children.filter(x => x !== this); this.parentNode = null; } }, get nextSibling() { const p = this.parentNode; return p ? p.children[p.children.indexOf(this) + 1] || null : null; } }; }

// ── SSE response stub: each `chunks` entry is one network read ───────────
function sseResponse(chunks, runtime = 'chat_agent') {
  const encoder = new TextEncoder();
  let i = 0;
  return {
    ok: true,
    headers: { get: name => (name === 'X-Odysseus-Run-Id' ? 'run-1' : name === 'X-Odysseus-Runtime' ? runtime : '') },
    body: { getReader: () => ({
      read: async () => (i < chunks.length ? { done: false, value: encoder.encode(chunks[i++]) } : { done: true }),
      cancel: async () => {},
    }) },
  };
}
const ev = obj => `data: ${JSON.stringify(obj)}\n\n`;

function build({ responses, history }) {
  const box = el('div'); box.isConnected = true;
  const user = el('div'); user.className = 'msg msg-user'; box.appendChild(user);
  const calls = { fetch: [], added: [], selected: [] };
  const responders = [...responses];
  const scope = {
    document: {
      getElementById: id => (id === 'chat-history' ? box : null),
      querySelector: () => null,
      createElement: el,
      createComment: comment,
    },
    fetch: async (url) => {
      calls.fetch.push(url);
      if (url.includes('/api/history/')) return { ok: true, json: async () => ({ history }) };
      const next = responders.shift();
      return next || { ok: false, status: 404, body: null, headers: { get: () => '' } };
    },
    sessionModule: {
      getSessions: () => [{ id: 's1', model: 'm' }],
      getCurrentSessionId: () => 's1',
      markStreaming() {}, clearStreaming() {},
      selectSession: async id => { calls.selected.push(id); },
    },
    uiModule: { esc: s => String(s), scrollHistory() {}, el: () => null },
    spinnerModule: { create: () => ({ createElement: () => el('div'), start() {}, destroy() {} }) },
    createTurnRendering: () => ({ accepts: () => true, settle() {}, render() {} }),
    markdownModule: { normalizeThinkingMarkup: s => s, squashOutsideCode: s => s, mdToHtml: s => s, processWithThinking: s => s },
    chatRenderer: {
      toolDenialHtml: () => '', renderAskUserCard() {}, resolveAskUserCard() {}, recordSessionMetricsCost() {},
      addMessage: (role, content) => { calls.added.push({ role, content }); const b = el('div'); b.className = 'msg msg-ai'; box.appendChild(b); },
    },
    renderToolIcon: () => '', hasActiveStream: () => false,
    _streamRunIds: new Map(), _streamRuntimes: new Map(), _resumingStreams: new Set(), _backgroundStreams: new Map(),
    updateSubmitButton() {}, _shortModel: m => m || '', _applyModelColor() {}, _streamDisplayText: t => t,
    _showDocumentWritingStatus() {}, _finishDocumentWritingStatus() {}, startsContinuationRound: () => true,
    displayMetrics() {}, createMsgFooter: () => { const f = el('div'); f.className = 'msg-footer'; return f; },
    _metricsCostRecordId: () => '', _applyAutoModeState() {}, _setStoredPlan() {}, documentModule: null,
    createTerminalStreamError: p => new Error(p.error || 'error'),
    API_BASE: '', Node: { DOCUMENT_POSITION_FOLLOWING: 4 },
  };
  const body = `
    var _autoNudges = 0;
    const _AUTO_NUDGE_CAP = 3;
    ${resumeSource}
    return { resumeStream, nudges: () => _autoNudges };
  `;
  const api = new Function(...Object.keys(scope), body)(...Object.values(scope));
  return { ...api, box, calls };
}

const TOOL_START = ev({ type: 'tool_start', tool: 'bash', command: 'stat /workspace', call_id: 'c2' });
const REST = [
  ev({ type: 'tool_output', tool: 'bash', command: 'stat /workspace', output: 'ok', exit_code: 0, call_id: 'c2' }),
  ev({ delta: 'The workspace is already tidy.' }),
  ev({ type: 'message_saved', id: 'm1' }),
  'data: [DONE]\n\n',
];
const SAVED = [{ role: 'assistant', content: 'The workspace is already tidy.', metadata: { _db_id: 'm1', runtime: 'chat_agent' } }];

test('resume: an SSE that ends before [DONE] re-attaches and draws the rest of the turn', async () => {
  const h = build({
    responses: [sseResponse([TOOL_START]), sseResponse([TOOL_START, ...REST])],
    history: SAVED,
  });
  assert.equal(await h.resumeStream('s1'), true);
  assert.deepEqual(h.calls.fetch, ['/api/chat/resume/s1', '/api/chat/resume/s1', '/api/history/s1?limit=12']);
  assert.equal(h.nudges(), 1);
  assert.deepEqual(h.calls.added, [{ role: 'assistant', content: 'The workspace is already tidy.' }]);
  // The first reader's running tool card did not survive next to the saved row.
  assert.equal(h.box.querySelectorAll('agent-thread-node').length, 0);
  assert.equal(h.box.querySelectorAll('msg-ai').length, 1);
});

test('resume: a stream that reached [DONE] is not re-attached', async () => {
  const h = build({ responses: [sseResponse([TOOL_START, ...REST])], history: SAVED });
  assert.equal(await h.resumeStream('s1'), true);
  assert.deepEqual(h.calls.fetch, ['/api/chat/resume/s1', '/api/history/s1?limit=12']);
  assert.equal(h.nudges(), 0);
});

test('resume: when the run ended during the gap (404), the chat is reloaded for its saved answer', async () => {
  const h = build({ responses: [sseResponse([TOOL_START])], history: SAVED });
  assert.equal(await h.resumeStream('s1'), true);
  assert.deepEqual(h.calls.fetch, ['/api/chat/resume/s1', '/api/chat/resume/s1']);
  assert.deepEqual(h.calls.selected, ['s1']);
  assert.equal(h.box.querySelectorAll('agent-thread-node').length, 0);
});

test('resume: re-attaching stops at the auto-recovery cap', async () => {
  const cut = () => sseResponse([TOOL_START]);
  const h = build({ responses: [cut(), cut(), cut(), cut(), cut()], history: SAVED });
  assert.equal(await h.resumeStream('s1'), true);
  assert.equal(h.calls.fetch.filter(u => u.includes('/resume/')).length, 4);
  assert.equal(h.nudges(), 3);
  // The last reader keeps what it drew instead of looping forever.
  assert.equal(h.box.querySelectorAll('agent-thread-node').length, 1);
});
