// Agent settings panel.
// Extracted verbatim from static/js/settings.js. Behaviour is unchanged: the
// functions moved, their bodies did not. settings.js imports them and calls
// them from the same places, so load order and init sequence are untouched.

import { byId as el } from './dom.js';
import { postSettings as _postSettings } from './api.js';

export async function initAgentSettings() {
  var runtimeToggle = el('set-chatRuntimeToggle');
  var runtimeCard = el('set-chatRuntimeCard');
  var runtimeMsg = el('set-chatRuntimeMsg');
  if (runtimeToggle) {
    // Per-user switch in the chat_agent store (spec #44), not /api/auth/settings.
    try {
      var rtRes = await fetch('/api/chat_agent/settings', { credentials: 'same-origin' });
      var rt = await rtRes.json();
      if (!rtRes.ok) throw new Error(rt.detail || 'Failed to load');
      if (rt.available) runtimeToggle.checked = rt.chat_runtime === 'new';
      else if (runtimeCard) {
        runtimeCard.style.display = 'none';
        // The classifier only serves the new runtime (#68).
        var classifierCard = el('set-classifierCard');
        if (classifierCard) classifierCard.style.display = 'none';
      }
    } catch (e) { runtimeMsg.textContent = e.message || 'Failed to load'; runtimeMsg.style.color = 'var(--red)'; }

    runtimeToggle.addEventListener('change', async function() {
      try {
        var res = await fetch('/api/chat_agent/settings', {
          method: 'PUT', credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ chat_runtime: runtimeToggle.checked ? 'new' : 'legacy' }),
        });
        var data = await res.json().catch(function() { return {}; });
        if (!res.ok) throw new Error(data.detail || 'Failed to save');
        runtimeToggle.checked = data.chat_runtime === 'new';
        runtimeMsg.textContent = 'Saved. Applies from your next message.'; runtimeMsg.style.color = 'var(--fg)';
        // The composer's auto mode icon exists only on the new runtime (#68).
        if (window.__odysseusSyncAutoModeButton) window.__odysseusSyncAutoModeButton();
      } catch (e) {
        // Keep the switch honest: a refused save leaves the server state unchanged.
        runtimeToggle.checked = !runtimeToggle.checked;
        runtimeMsg.textContent = e.message || 'Failed to save'; runtimeMsg.style.color = 'var(--red)';
      }
    });
  }

  var insCard = el('set-chatInstructionsCard');
  var insToggle = el('set-chatInstructionsToggle');
  var insText = el('set-chatInstructionsText');
  var insPath = el('set-chatInstructionsPath');
  var insCount = el('set-chatInstructionsCount');
  var insSave = el('set-chatInstructionsSave');
  var insMsg = el('set-chatInstructionsMsg');
  if (insText) {
    var insMax = 0;

    function showInstructions(data) {
      insToggle.checked = !!data.enabled;
      insText.value = data.content || '';
      insPath.textContent = data.path || '';
      insMax = data.max_chars || 0;
      countInstructions();
    }

    function countInstructions() {
      var n = insText.value.length;
      insCount.textContent = insMax ? n + ' / ' + insMax : String(n);
      // The server refuses anything over max_chars (400); say so before the click.
      insCount.style.color = insMax && n > insMax ? 'var(--red)' : '';
    }

    async function putInstructions(body, okText) {
      try {
        var res = await fetch('/api/chat_agent/instructions', {
          method: 'PUT', credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        var data = await res.json().catch(function() { return {}; });
        if (!res.ok) throw new Error(data.detail || 'Failed to save');
        showInstructions(data);
        insMsg.textContent = okText; insMsg.style.color = 'var(--fg)';
        return true;
      } catch (e) {
        insMsg.textContent = e.message || 'Failed to save'; insMsg.style.color = 'var(--red)';
        return false;
      }
    }

    try {
      var insRes = await fetch('/api/chat_agent/instructions', { credentials: 'same-origin' });
      var ins = await insRes.json();
      if (!insRes.ok) throw new Error(ins.detail || 'Failed to load');
      if (ins.available) showInstructions(ins);
      else insCard.style.display = 'none';
    } catch (e) { insMsg.textContent = e.message || 'Failed to load'; insMsg.style.color = 'var(--red)'; }

    insText.addEventListener('input', countInstructions);
    insSave.addEventListener('click', function() {
      putInstructions({ content: insText.value }, 'Saved. Applies from your next message.');
    });
    insToggle.addEventListener('change', async function() {
      var wanted = insToggle.checked;
      var ok = await putInstructions({ enabled: wanted }, wanted ? 'Instructions on.' : 'Instructions off.');
      // Keep the switch honest: a refused save leaves the server state unchanged.
      if (!ok) insToggle.checked = !wanted;
    });
  }

  var rulesCard = el('set-chatRulesCard');
  var rulesMsg = el('set-chatRulesMsg');
  if (rulesCard) {
    // Persistent allow/ask/deny rules of the new runtime (spec #44, #64).
    // PUT replaces all three lists, so every add/remove sends the whole set.
    var rules = { allow: [], ask: [], deny: [] };
    var ruleSections = rulesCard.querySelectorAll('[data-chat-rules]');

    function showRules(data) {
      rules = data.rules || { allow: [], ask: [], deny: [] };
      ruleSections.forEach(function(section) {
        var name = section.dataset.chatRules;
        var list = section.querySelector('[data-chat-rules-list]');
        list.textContent = '';
        if (!rules[name] || !rules[name].length) {
          var empty = document.createElement('div');
          empty.className = 'admin-empty';
          empty.style.textAlign = 'left';
          empty.style.padding = '2px 8px 6px';
          empty.textContent = 'No ' + name + ' rules';
          list.appendChild(empty);
          return;
        }
        rules[name].forEach(function(text) {
          var row = document.createElement('div');
          row.className = 'admin-rag-item';
          var label = document.createElement('span');
          label.className = 'admin-rag-item-name';
          label.title = text;
          label.textContent = text;
          var remove = document.createElement('button');
          remove.type = 'button';
          remove.className = 'admin-btn-delete';
          remove.textContent = 'Remove';
          remove.addEventListener('click', function() {
            var next = Object.assign({}, rules);
            next[name] = rules[name].filter(function(t) { return t !== text; });
            remove.disabled = true;
            putRules(next, 'Removed ' + text + '.').then(function(ok) { if (!ok) remove.disabled = false; });
          });
          row.appendChild(label);
          row.appendChild(remove);
          list.appendChild(row);
        });
      });
    }

    async function putRules(next, okText) {
      try {
        var res = await fetch('/api/chat_agent/rules', {
          method: 'PUT', credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rules: next }),
        });
        var data = await res.json().catch(function() { return {}; });
        // The 400 detail is the rule parser's own message; show it as is.
        if (!res.ok) throw new Error(data.detail || 'Failed to save');
        showRules(data);
        rulesMsg.textContent = okText; rulesMsg.style.color = 'var(--fg)';
        return true;
      } catch (e) {
        rulesMsg.textContent = e.message || 'Failed to save'; rulesMsg.style.color = 'var(--red)';
        return false;
      }
    }

    try {
      var rulesRes = await fetch('/api/chat_agent/rules', { credentials: 'same-origin' });
      var rulesData = await rulesRes.json();
      if (!rulesRes.ok) throw new Error(rulesData.detail || 'Failed to load');
      if (rulesData.available) showRules(rulesData);
      else rulesCard.style.display = 'none';
    } catch (e) { rulesMsg.textContent = e.message || 'Failed to load'; rulesMsg.style.color = 'var(--red)'; }

    ruleSections.forEach(function(section) {
      var name = section.dataset.chatRules;
      var form = section.querySelector('form');
      var input = form.querySelector('input');
      form.addEventListener('submit', async function(ev) {
        ev.preventDefault();
        var text = input.value.trim();
        if (!text) return;
        var next = Object.assign({}, rules);
        next[name] = (rules[name] || []).concat([text]);
        // A refused rule stays in the box so it can be corrected.
        if (await putRules(next, 'Added ' + text + ' to ' + name + '. Applies from your next message.')) input.value = '';
      });
    });
  }

  var toolsInput = el('set-agentMaxTools');
  var roundsInput = el('set-agentMaxRounds');
  var supInput = el('set-agentSupervisorLadder');
  var msg = el('set-agentMsg');
  if (!toolsInput) return;

  try {
    var res = await fetch('/api/auth/settings', { credentials: 'same-origin' });
    var settings = await res.json();
    if (settings.agent_max_tool_calls) toolsInput.value = settings.agent_max_tool_calls;
    if (roundsInput && settings.agent_max_rounds) roundsInput.value = settings.agent_max_rounds;
    if (supInput) supInput.checked = !!settings.agent_supervisor_ladder;
  } catch (e) {}

  // Clamp + coerce a raw input to an int in [lo, hi]; falls back to `dflt`
  // when blank/non-numeric. Mirrors the server-side validation.
  function clampInt(raw, lo, hi, dflt) {
    var n = parseInt(raw, 10);
    if (isNaN(n)) return dflt;
    return Math.max(lo, Math.min(n, hi));
  }

  async function save() {
    var tools = clampInt(toolsInput.value, 0, 1000, 0);
    var rounds = roundsInput ? clampInt(roundsInput.value, 1, 200, 20) : null;
    toolsInput.value = tools;                       // reflect the clamped value
    if (roundsInput) roundsInput.value = rounds;
    var payload = { agent_max_tool_calls: tools };
    if (rounds != null) payload.agent_max_rounds = rounds;
    if (supInput) payload.agent_supervisor_ladder = !!supInput.checked;
    try {
      await _postSettings(payload);
      msg.textContent = (tools > 0 ? 'Limit: ' + tools + ' tool calls' : 'Unlimited tool calls') +
        (rounds != null ? ' · ' + rounds + ' steps/message' : '') +
        (supInput && supInput.checked ? ' · supervisor on' : '');
      msg.style.color = 'var(--fg)';
    } catch (e) { msg.textContent = e.status === 403 ? 'Admin access is required to change these settings.' : 'Failed to save'; msg.style.color = 'var(--red)'; }
  }

  toolsInput.addEventListener('change', save);
  if (roundsInput) roundsInput.addEventListener('change', save);
  if (supInput) supInput.addEventListener('change', save);
  var cur = parseInt(toolsInput.value, 10) || 0;
  var curR = roundsInput ? (parseInt(roundsInput.value, 10) || 20) : null;
  msg.textContent = (cur > 0 ? 'Limit: ' + cur + ' tool calls' : 'Unlimited tool calls') +
    (curR != null ? ' · ' + curR + ' steps/message' : '') +
    (supInput && supInput.checked ? ' · supervisor on' : '');

}
