// Slash autocomplete picks up skill catalog changes made after page load.
//
// Seam: initSlashAutocomplete(textarea) in static/js/slashAutocomplete.js,
// driven through the textarea's input/keydown events and the
// /api/skills/slash-catalog fetch. The real module is loaded through Node's
// ESM linker; slashCommands.js is synthetic (its imports need a browser).
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const JS = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../static/js');
const AC_JS = path.join(JS, 'slashAutocomplete.js');

const COMMANDS = {
  help: { handler: true, category: 'General', help: 'Show help' },
  jev: { handler: true, category: 'General', help: 'Built-in jev' },
};

class Textarea {
  constructor() { this.value = ''; this._listeners = {}; }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  dispatch(type, event = {}) { for (const fn of this._listeners[type] || []) fn(event); }
  getBoundingClientRect() { return { left: 0, top: 600, width: 400, bottom: 640 }; }
  dispatchEvent() {}
  focus() {}
  setSelectionRange() {}
}

class Popup {
  constructor() { this.style = {}; this.innerHTML = ''; }
  setAttribute() {}
  querySelector() { return null; }
  rows() { return [...this.innerHTML.matchAll(/data-token="([^"]*)"/g)].map(m => m[1]); }
  selected() { return this.innerHTML.match(/slash-ac-row slash-ac-row-sel[^>]*data-token="([^"]*)"/)?.[1] ?? null; }
}

// Scripted fetch: each entry answers one /api/skills/slash-catalog call in
// order, as the catalog's skills array.
function makeFetch(catalogs) {
  const fetch = async (url) => {
    assert.equal(url, '/api/skills/slash-catalog');
    const skills = catalogs.shift();
    if (!skills) throw new Error('unexpected catalog fetch');
    return { ok: true, status: 200, json: async () => ({ skills }) };
  };
  return fetch;
}

const settle = () => new Promise(resolve => setImmediate(resolve));

async function loadAutocomplete(catalogs) {
  const popup = new Popup();
  const textarea = new Textarea();
  const document = {
    getElementById: () => null,
    createElement: () => popup,
    body: { appendChild() {} },
    addEventListener() {},
  };
  const window = { innerHeight: 900, addEventListener() {} };
  const context = vm.createContext({ document, window, fetch: makeFetch(catalogs), console, Event: class {} });

  const ac = new vm.SourceTextModule(fs.readFileSync(AC_JS, 'utf8'), { context, identifier: AC_JS });
  await ac.link((specifier) => {
    assert.equal(specifier.split('?')[0], './slashCommands.js');
    return new vm.SyntheticModule(['COMMANDS', 'LEGACY_ALIASES'], function () {
      this.setExport('COMMANDS', COMMANDS);
      this.setExport('LEGACY_ALIASES', {});
    }, { context, identifier: 'slashCommands.js' });
  });
  await ac.evaluate();
  ac.namespace.initSlashAutocomplete(textarea);
  await settle();

  const type = async (value) => { textarea.value = value; textarea.dispatch('input'); await settle(); };
  return { popup, textarea, type };
}

const JEV = { token: '/jev-classify', help: 'Classify with jev', usage: '/jev-classify <request>' };

test('a skill approved after page load appears once the popup reopens', async () => {
  const { popup, type } = await loadAutocomplete([[], [], [JEV]]);

  await type('/');       // opens: fetch #2, still without the skill
  await type('/jev');    // only the built-in matches
  assert.deepEqual(popup.rows(), ['/jev']);
  await type('/jev-');   // nothing matches: popup closes
  assert.equal(popup.style.display, 'none');

  await type('/');       // reopens: fetch #3 now carries the skill
  await type('/jev');
  assert.deepEqual(popup.rows(), ['/jev', '/jev-classify']);
});

test('an unpublished skill leaves the list once the popup reopens', async () => {
  const { popup, type } = await loadAutocomplete([[JEV], [JEV], []]);

  await type('/');
  await type('/jev-');
  assert.deepEqual(popup.rows(), ['/jev-classify']);
  await type('');        // closes
  assert.equal(popup.style.display, 'none');

  await type('/');       // reopens: fetch #3 says the skill is gone
  await type('/jev-');
  assert.equal(popup.style.display, 'none');
  await type('/jev');
  assert.deepEqual(popup.rows(), ['/jev']);
});

test('a skill whose token collides with a built-in still loses to the built-in', async () => {
  const skill = { token: '/help', help: 'Skill help' };
  const { popup, type } = await loadAutocomplete([[skill], [skill]]);
  await type('/help');
  assert.deepEqual(popup.rows(), ['/help']);
  assert.match(popup.innerHTML, /Show help/);
  assert.doesNotMatch(popup.innerHTML, /Skill help/);
});

test('an unchanged catalog arriving mid-navigation keeps the selected row', async () => {
  const { popup, textarea, type } = await loadAutocomplete([[JEV], [JEV]]);
  textarea.value = '/';  // rows: /jev, /help, /jev-classify (shorter token first)
  textarea.dispatch('input');  // reopen fetch is in flight until we yield
  textarea.dispatch('keydown', { key: 'ArrowDown', preventDefault() {} });
  assert.equal(popup.selected(), '/help');
  await settle();        // reopen reply lands; a re-render would reset to /jev
  assert.equal(popup.selected(), '/help');
});
