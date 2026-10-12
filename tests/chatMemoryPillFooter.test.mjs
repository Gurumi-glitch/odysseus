import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// The "N pinned, M recalled" pill is drawn by createMsgFooter from
// `_memoriesUsed` on the bubble that gets the footer. The memories_used
// event lands on the first bubble (holder) before any tool round exists, so
// a multi-round turn must carry it onto the round bubble the footer goes on.
// Same slice-by-marker harness as chatEditorProgress.test.mjs; no browser.
const chat = await readFile(new URL('../static/js/chat.js', import.meta.url), 'utf8');

const streamEndStart = chat.indexOf('        // Attach footer to the last visible bubble');
const streamEndEnd = chat.indexOf('        if (_generatedImagesForTurn.length', streamEndStart);
assert.ok(streamEndStart > 0 && streamEndEnd > streamEndStart, 'stream-end footer block not found');
const streamEndFooter = new Function(
  'roundHolder', 'holder', 'createMsgFooter',
  chat.slice(streamEndStart, streamEndEnd) + '\nreturn footerTarget;',
);

const resumeStart = chat.indexOf('    const target = roundHolder?.isConnected && roundHolder.style.display !== \'none\'');
const resumeEnd = chat.indexOf('        const body = target.querySelector(\'.body\');', resumeStart);
assert.ok(resumeStart > 0 && resumeEnd > resumeStart, 'resume footer block not found');
const resumeFooter = new Function(
  'roundHolder', 'currentToolThread', 'replayMessageId', 'roundText', 'metricsData', 'createMsgFooter',
  chat.slice(resumeStart, resumeEnd) + '}}\nreturn target;',
);

const MEMORIES = [{ type: 'pinned', text: 'Likes tea' }, { type: 'recalled', text: 'Lives in Taipei' }];

function bubble(cls = 'msg-ai') {
  const children = [];
  return {
    isConnected: true,
    style: { display: '' },
    dataset: {},
    classList: { contains: c => c === cls, remove() {} },
    querySelector: sel => children.find(n => '.' + n.className === sel) || null,
    appendChild: n => { children.push(n); return n; },
  };
}

// createMsgFooter stub: records the bubble it was given at the moment it ran.
function footerStub() {
  const calls = [];
  const fn = target => { calls.push({ target, memories: target._memoriesUsed }); return { className: 'msg-footer' }; };
  return { fn, calls };
}

test('stream end: a multi-round turn puts the memories on the round bubble that gets the footer', () => {
  const holder = bubble();
  holder._memoriesUsed = MEMORIES;
  const roundHolder = bubble();
  const footer = footerStub();

  const target = streamEndFooter(roundHolder, holder, footer.fn);

  assert.equal(target, roundHolder);
  assert.equal(footer.calls.length, 1);
  assert.equal(footer.calls[0].target, roundHolder);
  assert.deepEqual(footer.calls[0].memories, MEMORIES);
});

test('stream end: a single-round turn keeps the footer and memories on holder', () => {
  const holder = bubble();
  holder._memoriesUsed = MEMORIES;
  const footer = footerStub();

  const target = streamEndFooter(holder, holder, footer.fn);

  assert.equal(target, holder);
  assert.deepEqual(footer.calls[0].memories, MEMORIES);
});

test('stream end: a hidden round bubble falls back to holder and nothing leaks onto it', () => {
  const holder = bubble();
  holder._memoriesUsed = MEMORIES;
  const roundHolder = bubble();
  roundHolder.style.display = 'none';
  const footer = footerStub();

  const target = streamEndFooter(roundHolder, holder, footer.fn);

  assert.equal(target, holder);
  assert.equal(roundHolder._memoriesUsed, undefined);
  assert.deepEqual(footer.calls[0].memories, MEMORIES);
});

test('stream end: the footer is built once per bubble', () => {
  const holder = bubble();
  holder._memoriesUsed = MEMORIES;
  const roundHolder = bubble();
  const footer = footerStub();

  streamEndFooter(roundHolder, holder, footer.fn);
  streamEndFooter(roundHolder, holder, footer.fn);

  assert.equal(footer.calls.length, 1);
});

test('resume: memories_used from the saved metadata reaches the footer bubble', () => {
  const roundHolder = bubble();
  const footer = footerStub();

  const target = resumeFooter(roundHolder, null, 'db-1', 'answer', { memories_used: MEMORIES }, footer.fn);

  assert.equal(target, roundHolder);
  assert.equal(target.dataset.dbId, 'db-1');
  assert.deepEqual(footer.calls[0].memories, MEMORIES);
});

test('resume: no memories leaves the footer bubble untouched', () => {
  const roundHolder = bubble();
  const footer = footerStub();

  resumeFooter(roundHolder, null, 'db-1', 'answer', { rag_sources: [] }, footer.fn);

  assert.equal(footer.calls[0].memories, undefined);
});
