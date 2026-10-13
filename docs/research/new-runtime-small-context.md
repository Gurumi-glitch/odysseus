# New chat agent runtime with a small-context local model

Research for fork issue #77 (map #74). Question: what does the new chat agent
runtime (`src/chat_agent/`) do today with a local model on a 16k context?

Code is at `origin/devGuru` 3505020a. Each fact carries a tag:

- **measured**: observed by running code or querying the server (method below).
- **verified-from-code**: read in the source at the cited line.
- **inferred**: follows from the above, but nobody ran it end to end.

## Method

- **Request capture (measured).** The real chat route
  (`routes/chat_routes.py::chat_stream`) and the real runtime
  (`loop.run_chat_agent_turn`) ran in a throwaway `odysseus-local:dev`
  container. The test harness from `tests/chat_agent/test_chat_agent_route.py`
  (`_Harness`) supplied the session, and an `httpx.MockTransport` stood in for
  the OpenAI-compatible server. That stub recorded the exact JSON body, played
  back scripted SSE streams, and the run captured the SSE events and saved rows.
- **Simulated session.** Endpoint `http://host.docker.internal:1235/v1`, model
  `Ternary-Bonsai-2-27B`, window patched to 16384, first turn, message
  "Hello". The route preface was the real one for the default preset: no preset
  system prompt, only `UNTRUSTED_CONTEXT_POLICY`, plus the real date/time
  message. No memory, RAG, instruction files, skills or MCP.
- **Forced routing.** `routing.endpoint_has_adapter` was forced to True. In
  reality this endpoint falls back to legacy (#74, point 1), so these numbers
  describe what the new runtime would send once routing lets the endpoint in.
- **Privileges.** The harness user `alice` is not an admin, so
  `blocked_tools_for_owner` hid 55 tools (`src/chat_agent/permissions.py:232`).
  The measured tool list is therefore a non-admin surface. The admin upper
  bound was measured separately from the full upfront table.
- **Token counts (measured).** Counted with the real tokenizer and chat
  template of the running llama-server (port 1235, Ternary-Bonsai-2-27B,
  `n_ctx` 16384), using only its read-only `/apply-template` and `/tokenize`
  endpoints. Nothing was generated, and the server was not restarted.
  Server-side defaults come from `GET /props`.
- **Scratch files.** The measuring test was a scratch file and is not
  committed.

## 1. Prompt token footprint of a first round

### Measured: first request, non-admin surface, no composer toggles

| Part | Tokens | Where it comes from |
|---|---:|---|
| Template wrapper (role tags, injected "Reasoning effort is set to xhigh..." line, opening `<think>`) | 53 | chat template of the server |
| System: `UNTRUSTED_CONTEXT_POLICY` | 101 | `src/prompt_security.py:8`, added by `src/chat_processor.py` `build_context_preface` |
| System: behaviour section | 59 | `src/chat_agent/prompt.py:40-45` |
| System: unavailable-tools line (5 names) | 35 | `src/chat_agent/tools.py` `unavailable_line`, `src/chat_agent/loop.py:362-365` |
| System: deferred-tool index (15 built-in names, no MCP) | 80 | `src/chat_agent/tools.py` `catalog_section`, `loop.py:356-358` |
| **System message total** | **278** | assembled at `src/chat_agent/prompt.py:212-216` |
| User: backend runtime context block | 201 | `src/chat_agent/turn_context.py:177` (`_backend_runtime_context_message`; present because the app runs in Docker) |
| User: date/time block | 207 | `src/user_time.py:221` |
| User: "Hello" | 1 | |
| **User message total** | **410** | `prompt.py:220-225` |
| Instructions block | 0 | none in this run (`prompt.py:149-163`) |
| **Messages total** | **741** | |
| Tool schemas, 18 tools, full descriptions | **7,417** | `tools.upfront_schemas` (`src/chat_agent/tools.py:134-141`) + `tool_search` |
| **Request total** | **8,158** (50% of 16,384) | |

Per-tool cost in that request (JSON tokens; measured): `inspect_media` 1,506,
`ui_control` 1,415, `manage_notes` 817, `transcribe_media` 518, `ask_user`
380, `edit_document` 327, `create_document` 315, `edit_image` 251. The other
10 tools cost 101 to 238 each. `inspect_media` and `ui_control` alone are 39%
of the tool tokens.

### Measured: other surfaces (same messages)

| Surface | Tools | Tool tokens | Request total |
|---|---:|---:|---:|
| Non-admin, no toggles, full schemas | 18 | 7,417 | 8,158 |
| Non-admin, no toggles, `compact` surface | 18 | 3,066 | 3,807 |
| Non-admin, web search + bash toggles on, full | 23 | 8,704 | 9,409 |
| Non-admin, web search + bash toggles on, compact | 23 | 3,716 | 4,421 |
| Every upfront tool (57 + `tool_search`), full | 58 | **17,240** | ~17,950 (inferred: 17,240 + ~705 message tokens) |
| Every upfront tool, compact | 58 | 7,144 | ~7,850 (inferred) |

Biggest schemas in the full upfront table (measured): `inspect_media` 1,506,
`ui_control` 1,415, `manage_calendar` 1,271, `manage_tasks` 1,076,
`manage_notes` 817.

### Facts

- **inferred.** With every upfront tool offered and full descriptions, the
  first request alone (~17.9k tokens) is larger than the 16,384-token window.
  The admin set the user actually gets depends on privileges, composer toggles
  (`permissions.build_deny_layers`), a connected email account and a picked
  workspace, so it falls between the 18-tool and 58-tool rows.
- **verified-from-code.** The tool surface comes from the endpoint row's
  per-model setting, through `_configured_model_tool_surface`
  (`src/chat_agent/loop.py:351-354`, `src/agent_loop.py:8470`).
  - `"compact"` strips descriptions and `"none"` sends no tools
    (`src/agent_loop.py:386-395`).
  - A hand-started server with no endpoint row gets `""`, which means full
    schemas.
  - The runtime never picks a surface from the window size.
- **verified-from-code.** The deferred split is one fixed table
  (`DEFERRED_TOOLS`, `src/chat_agent/tools.py:47-61`).
  - The 31 deferred built-ins cost only their names in the system prompt: 137
    tokens for the whole index when none is denied (measured).
  - An admin also gets every enabled MCP tool name listed there
    (`tools.py:158-176`). That was not measured, because the test had no MCP
    manager.
- **verified-from-code.** No step checks whether the first request fits the
  window.
  - `transcript.compact_history` runs at turn start, only when the chat already
    has history (`src/chat_agent/transcript.py:502-505`).
  - It compares history tokens (reported usage, or the legacy estimate) with
    85% of the window (`transcript.py:419-427`;
    `src/context_compactor.py:94,99-105`).
  - Tool schemas are not part of the estimate on a turn with no reported usage.
    After a turn that reported usage, `prompt_tokens` does include them.
  - `transcript.fit_request` runs only when the turn falls back to another
    candidate (`src/chat_agent/loop.py:221-223`).
- **verified-from-code.** The runtime reads the window through
  `get_context_length`. For llama.cpp, that comes from `/slots` or `/props`
  `n_ctx` (`src/model_context.py:440-454`), so it would see 16384 here.
- **Not measured (variable per user).** These parts were not in the run:
  - Preset system prompt.
  - Saved memory and RAG blocks (`chat_processor.build_context_preface`).
  - Instruction files: up to 12k characters each
    (`src/chat_agent/prompt.py:115`). At the measured 4.36 characters per
    token, that is roughly 2.7k tokens per file (inferred).
  - Skill index: names plus up to 8,000 characters of descriptions
    (`prompt.py:37`), roughly 1.8k+ tokens (inferred).
  - The writing-rules system section (`turn_context.py:156-159`), added only
    when an email or document style is saved.

## 2. Generation settings

The request body (measured, Ternary-Bonsai, no session overrides) is exactly
`model`, `messages`, `stream: true`, `stream_options: {include_usage: true}`,
`tools`, `tool_choice: "auto"` and `temperature: 1.0`. It has no `max_tokens`,
`top_p`, `top_k`, `min_p`, `chat_template_kwargs` or `reasoning_effort`.

With the session overrides `max_tokens_override=8192` and
`temperature_override=0.6`, the body (measured) gains `max_tokens: 8192` and
carries `temperature: 0.6`. No other key changes.

| Setting | Value sent | Source | Tag |
|---|---|---|---|
| `temperature` | 1.0 | 1. request form `temperature` (`routes/chat_routes.py:2482-2488`); 2. session `temperature_override` overrides it (`chat_routes.py:2771-2772`); 3. otherwise `LLMConfig.DEFAULT_TEMPERATURE` = 1.0 (`chat_routes.py:3057-3061`, `src/llm_core.py:183`). Omitted only for models in `_omit_temperature` (`src/chat_agent/providers/chat_completions.py:80-81`) | verified-from-code + measured |
| `max_tokens` | not sent | `ChatAgentTurn.max_tokens = sess.max_tokens_override or 0` (`chat_routes.py:3062`); `build_payload` sends the key only when > 0 (`chat_completions.py:82-84`). The UI slider caps at 8192, and anything above that is saved as `null` = "No limit" (`static/js/chat.js:529,533`) | verified-from-code + measured |
| Preset `temperature` / `max_tokens` (Custom preset: 1.0 / 32768, `src/preset_manager.py:56-60`) | ignored | The runtime never reads `ctx.preset.temperature` / `.max_tokens`; `loop.py` reads only `preset.character_name` (`loop.py:671`) | verified-from-code |
| `top_p`, `top_k`, `min_p`, penalties | not sent | `build_payload` has no such keys (`chat_completions.py:60-88`). The server then uses its own defaults: top_p 0.95, top_k 20, min_p 0.05, temperature 1.0, `n_predict` -1 (unlimited) | verified-from-code; server defaults measured via `/props` |
| Thinking switch | not sent | The route forces `thinking_mode="off"` for models without a user thinking toggle (`chat_routes.py:2765-2766`, list in `src/model_profiles.py:50-62`). `_apply_local_qwen_thinking_mode` then does nothing for a model `_supports_thinking` doesn't know (`src/llm_core.py:1217-1228`) | verified-from-code + measured (no `chat_template_kwargs` in body) |
| Effective thinking on the server | on, effort xhigh | The rendered prompt contains "Reasoning effort is set to xhigh..." and ends with `<|im_start|>assistant\n<think>` | measured (`/apply-template`) |
| Stream read timeout | 300 s per read | `_stream_timeout(LLMConfig.STREAM_TIMEOUT)` (`chat_completions.py:149`, `llm_core.py:187,209-211`). It limits silence between reads, not total time | verified-from-code |

Clamping:

- **verified-from-code.** `max_tokens` is **not** clamped to the remaining
  context. The runtime never calls `generation_budget.fit_output_token_budget`.
  The legacy loop calls it per route at `src/agent_loop.py:26198-26214`; the
  function is at `src/generation_budget.py:124-152`, and even there it clamps
  only a positive value.
- **inferred.** With no override, generation length is bounded only by the
  server: `n_predict` -1, which means until the 16,384-token context is full.
  In the measured minimal case that is 16,384 − 8,158 = 8,226 tokens for
  thinking plus answer.

## 3. Live answer and thinking deltas

- **measured.** Answer deltas reach the browser live.
  - A stub streamed 2 `reasoning_content` chunks and then 3 `content` chunks,
    0.5 s apart.
  - The SSE subscriber received `delta` "A" at 1.51 s, "B" at 2.01 s and "C"
    at 2.51 s: one event per provider chunk, with no buffering.
  - Path: `TextDelta` (`chat_completions.py:186-188`) → `{"delta": ...}`
    (`loop.py:429-432`) → `agent_runs.subscribe` (`src/agent_runs.py:223`).
  - The legacy `with_completion_gate` is applied only to the legacy agent loop
    (`src/agent_loop.py:21049`), so #74 point 2 does not happen on this path.
- **verified-from-code + measured.** Two exceptions delay deltas:
  - **Whitespace-only deltas** before the first real output are held until
    something non-whitespace arrives (`loop.py:236-238`). In the measured
    50-newline stream, all 50 were released at round end.
  - **Reasoning gets no live event.** `reasoning_content` is accumulated into
    a string (`chat_completions.py:185`) and only becomes a `reasoning` block
    at round end (`chat_completions.py:219-229`).
    - Measured: the 1.0 s of reasoning produced no event.
    - The runtime never emits the `{"delta": ..., "thinking": true}` event
      that legacy emits (`src/llm_core.py:418-422,3670-3676`) and the browser
      wraps in `<think>` (`static/js/chat.js:4186-4193`).
    - While the model thinks, the browser shows nothing new after
      `agent_step`.
- **measured + verified-from-code.** Reasoning shows as a thinking block in
  history.
  - The saved row carries `metadata.thinking` = the joined `reasoning_content`
    (`src/chat_agent/transcript.py:139-141,216-229`).
  - The history renderer prepends it as `<think>` before round 1's text
    (`static/js/chatRenderer.js:3563-3571`).
  - It is not shown live, only after a reload (or a history re-render).
- **inferred.** The server-side split is not confirmed. `/props` reports
  `reasoning_format: "none"` in its default slot params. The rendered prompt
  opens `<think>`, so the server might stream thinking inside `content`
  instead of `reasoning_content`.
- **measured.** If thinking arrives inside `content`, it streams live as plain
  deltas: `<think>`, "plan", `</think>`, "Hi" produced 4 deltas and was saved
  verbatim as content `<think>plan</think>Hi`.
  - The runtime has no think-tag or template-artifact stripping (no
    `_strip_visible_chat_template_artifacts`, no `think>` handling in
    `src/chat_agent/`).
  - The browser's own `<think>` detection would render it as thinking
    (`static/js/chat.js:4220-4224`; inferred).
  - The model's own record then holds the thinking as answer text, and a later
    turn replays it to the model as assistant content
    (`prompt.py:258-275`; inferred).

## 4. A round with no answer text

Measured with three streams: reasoning only, an empty stream, and whitespace
only.

- **The turn ends normally.** A round with no tool calls ends the turn
  (`loop.py:441-446`). It emits `final_response` with `content: ""`
  (`loop.py:627`), then `metrics` and `message_saved`.
- **No error.** No `event: error` is sent. The runtime emits errors only for a
  provider error (`loop.py:423-428`) or a crash (`loop.py:645-652`).
- **A message is saved.** An assistant row with `content ""` and
  `round_texts [""]` is saved, with no `stop_reason`. Writer:
  `transcript.TurnWriter`, `transcript.py:286-302`; the first checkpoint is at
  `loop.py:440`.
  - Reasoning only: the row also carries `metadata.thinking` and the reasoning
    block in `agent_v2.rounds`.
  - Empty stream: the row has `rounds: [{"blocks": []}]`.
  - Whitespace only: the row content is the 50 newlines.
- **verified-from-code.** The browser ignores an empty `final_response`
  (`if (!finalText.trim()) continue;`, `static/js/chat.js:4115`). The runtime
  sends no "empty answer" marker.
- **inferred.** The user sees the agent step end with no answer bubble text
  and no error. A thinking-only answer becomes visible only after a reload.
  The browser's end-of-stream rendering of an empty bubble was not traced.

## 5. Degenerate-output guard

- **verified-from-code.** No degenerate-output guard runs on this path.
  - `src/chat_agent/` has no reference to `_DegenerateStreamGuard`.
    `chat_completions.stream_round` reads the stream directly
    (`chat_completions.py:131-197`), not through legacy `stream_llm`, where the
    guard lives (`src/llm_core.py:3037,3674-3685`).
  - `repeat_guard.py` watches only repeated identical tool calls (stop at 12),
    never text.
- **measured.** A 5,000-delta stream of `/` and a 5,000-delta stream of
  `"the "` both streamed every delta to the browser and were saved whole: 5,000
  and 20,000 characters, with no error and no stop reason.
- **measured.** For comparison, the legacy guard fed one `/` per call never
  trips, even after 20,000 deltas.
  - It counts only `[A-Za-z0-9_` + Greek + Cyrillic`]` runs
    (`src/llm_core.py:425,452-453`) and returns early when a delta has none.
  - Fed `"the "`, it trips after 28 deltas (`same_run >= 28`,
    `llm_core.py:469`).
  - So `////` output would pass the guard even if the runtime adopted it as
    is.
- **inferred.** A server stuck emitting `/` (#74 point 3) would, on this path,
  stream `/` live until llama-server stops at the full context.
  - Nothing in the runtime ends it: no max_tokens, no guard, and the read
    timeout only fires on silence.
  - The turn then saves a row of slashes with no error. Unlike legacy, the
    browser would show the slashes rather than a blank screen.

## Implications for the map

Facts only.

- **Fits the window.** On a 16,384 window:
  - With full schemas, the measured non-admin first request uses 8,158 tokens
    (50%), 91% of it tool schemas.
  - The full upfront tool table alone (17,240 tokens) exceeds the window.
  - The `compact` surface cuts tool tokens by 57-59%.
  - The surface is a per-endpoint-row setting, never derived from the window.
- **Output length.** The runtime sends no `max_tokens` unless the session has
  an override. It never clamps one to the remaining context, and it ignores
  the preset's `max_tokens` and `temperature`.
- **Thinking control.** No thinking control reaches a model whose name is not
  on the thinking lists. Ternary-Bonsai's template then thinks at effort
  `xhigh` by default.
- **Thinking display.**
  - `reasoning_content` is invisible live and appears only on reload.
  - Thinking inside `content` streams live and is stored as answer text.
- **Liveness.** Answer deltas are live, with no completion gate.
- **Empty answers.** A turn with no answer text saves an empty assistant row
  and shows no error.
- **Degenerate output.** The path has no degenerate-output guard. The legacy
  guard would not catch punctuation-only output anyway.
- **Routing.** Every number above assumes routing lets the endpoint onto the
  new runtime. Today it does not (#74, point 1).
