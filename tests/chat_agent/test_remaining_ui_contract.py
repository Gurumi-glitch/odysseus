"""The form fields the browser sends the chat_agent runtime (spec #44).

Source pins in the style of tests/test_tool_approval_task_scope.py: each
field is read by routes/chat_routes.py, so the sender and the reader must
agree on its name and value.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_unwatched_senders_say_so():
    # Notes agent-solve and group panes have no ask card (#61); compare
    # synthesis stays on the legacy runtime and sends nothing.
    assert "fd.append('interactive', 'false');" in _source("frontend/src/features/notes/application/agentActions.js")
    assert "fd.append('interactive', 'false');" in _source("static/js/group.js")
    assert "interactive" not in _source("static/js/compare/stream.js")
    assert 'str(form_data.get("interactive", "")).lower() != "false"' in _source("routes/chat_routes.py")


def test_slash_invoked_skill_rides_the_next_send_once():
    slash = _source("static/js/slashCommands.js")
    chat = _source("static/js/chat.js")
    assert "window.__odysseusPendingInvokedSkill = data.name || '';" in slash
    assert "fd.append('invoked_skill', window.__odysseusPendingInvokedSkill);" in chat
    assert chat.count("window.__odysseusPendingInvokedSkill = '';") == 1
    assert 'invoked_skill=str(form_data.get("invoked_skill") or "").strip()' in _source("routes/chat_routes.py")


def test_continue_after_a_capped_chat_agent_turn_names_its_source():
    # The renderer's note sets the marker (live and history alike); the
    # submit handler reads it once into the form.
    chat = _source("static/js/chat.js")
    assert "_pendingTurnSource = 'continue';" in _source("static/js/chatRenderer.js")
    assert "const _turnSource = chatRenderer.takePendingTurnSource();" in chat
    assert "if (_turnSource) fd.append('source', _turnSource);" in chat
    assert 'source="continue" if form_data.get("source") == "continue" else None' in _source("routes/chat_routes.py")


# upstream/dev's rounds_exhausted block, verbatim: the legacy note keeps its
# text and merge-continue, and the chat_agent case lives in its own branch.
UPSTREAM_ROUNDS_EXHAUSTED_BLOCK = """              } else if (json.type === 'rounds_exhausted') {
                // The agent hit the per-turn step limit while still working.
                // Offer a Continue button instead of stalling silently.
                // NOTE: append to the chat-history container (bottom), NOT the
                // message body — the body innerHTML is re-rendered at stream
                // finalize, which would wipe a note placed inside it.
                const _chatBox = document.getElementById('chat-history');
                if (!_isBg && _chatBox) {
                  // Drop any prior box so repeated cap-hits each get a fresh
                  // Continue at the bottom (multiple continues in a row).
                  const _old = _chatBox.querySelector('.rounds-exhausted');
                  if (_old) _old.remove();
                  const note = document.createElement('div');
                  note.className = 'stopped-indicator rounds-exhausted';
                  const label = document.createElement('span');
                  label.className = 'rounds-exhausted-label';
                  label.textContent = `Reached the ${json.rounds || ''}-step limit — not finished.`;
                  note.appendChild(label);
                  const contBtn = document.createElement('button');
                  contBtn.className = 'continue-btn';
                  contBtn.title = 'Continue the task';
                  contBtn.textContent = 'Continue ▸';
                  const _holder = holder;
                  contBtn.addEventListener('click', () => {
                    note.remove();
                    _hideUserBubble = true;
                    _pendingContinue = _holder;
                    const msgInput = uiModule.el('message');
                    if (msgInput) {
                      msgInput.value = 'You hit the step limit before finishing — the task is not complete. Continue from exactly where you left off and keep going until it is done. Do NOT repeat work already done.';
                      const sb = document.querySelector('.send-btn');
                      if (sb) sb.click();
                    }
                  });
                  note.appendChild(contBtn);
                  _chatBox.appendChild(note);
                  try { note.scrollIntoView({ block: 'end', behavior: 'smooth' }); } catch (_) { uiModule.scrollHistory && uiModule.scrollHistory(); }
                }
              } else if (json.type === 'model_actual') {"""


def test_legacy_rounds_exhausted_block_is_upstreams_and_the_runtime_has_one_branch_before_it():
    chat = _source("static/js/chat.js")
    assert UPSTREAM_ROUNDS_EXHAUSTED_BLOCK in chat
    runtime_branch = "} else if (streamRuntime === 'chat_agent' && (json.type === 'rounds_exhausted' || json.type === 'budget_exceeded')) {"
    assert chat.count(runtime_branch) == 1
    assert chat.index(runtime_branch) < chat.index(UPSTREAM_ROUNDS_EXHAUSTED_BLOCK)
    assert "chatRenderer.renderTurnStopNote(json.type === 'budget_exceeded' ? 'budget' : 'max_rounds', json)" in chat
