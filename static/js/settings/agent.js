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
      else if (runtimeCard) runtimeCard.style.display = 'none';
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
