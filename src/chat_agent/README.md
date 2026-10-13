# chat_agent — the opt-in agent runtime for the main chat

Spec: fork issue #44. Decisions: ADR 0001–0008 (local). This package sits beside
upstream's `src/agent_runtime/` and never edits the legacy loop
(`src/agent_loop.py`).

## How a turn gets here

1. `routes/chat_routes.py::chat_stream` parses the request and snapshots the raw
   composer toggles (`routing.ComposerToggles`) before the legacy wording-based
   promotion rewrites them.
2. After the session, privileges and auth headers are resolved, and before
   compact-preview selection, context prefetching, deny-set assembly and
   turn-contract binding, the route calls `routing.choose_runtime`.
3. On `CHAT_AGENT` the route starts a detached run
   (`agent_runs.start(..., runtime="chat_agent")`) over
   `loop.run_chat_agent_turn`, outside the execution-bridge wrapper, and
   returns `X-Odysseus-Run-Id` + `X-Odysseus-Runtime: chat_agent`. Resume sends
   the same header. Legacy runs send `X-Odysseus-Runtime: legacy`.
4. `run_chat_agent_turn` is imported into the chat route as a module attribute,
   so route tests patch `routes.chat_routes.run_chat_agent_turn`.

## Modules

| Module | Job | Status |
|---|---|---|
| `store.py` | Per-user JSON store `CHAT_AGENT_DIR/<percent-encoded lowercased user>.json` (confined). Holds `chat_runtime: legacy\|new` and `instructions_enabled`. Owns the user instruction folder `INSTRUCTIONS_DIR/<same name>/` (`confine()`d). `rename_owner` / `delete_owner` are called from the user rename/delete routes. `rules` (allow/ask/deny lists of rule text, `user_rules` / `set_user_rules`) and `session_rules` (chat id -> `[tool, kind, value]` records, `session_rules` / `add_session_rule`); changing `chat_runtime` drops every session rule. | #48, #52, #62 |
| `routing.py` | `ComposerToggles` snapshot; `choose_runtime(TurnFacts)` → `legacy` or `chat_agent`. | #48 |
| `prompt.py` | Calls the route's context builder with `prefetch=False, agent_mode=False` (no URL/web/YouTube prefetch, no legacy skills index) and assembles: one system message (route system messages, then the behaviour section, then runtime sections such as the unavailable-tools line) → instructions block (user role: user folder unless disabled, then workspace ancestors up to home, `AGENTS.md` + `CLAUDE.md`, 12k chars each, labelled by path; none in Nobody mode) → history → current user message carrying the turn context (`turn_context.messages`, then memory, RAG, date/time). The system prompt never holds per-turn state, so it is byte-identical across a chat's turns. | #48, #50, #52, #59 |
| `turn_context.py` | What is open and asked for this turn, from the request (no wording heuristics): plan / approved-plan / guide-only directives, backend runtime context, workspace coding rules, the open editor document (plain with line numbers, PDF form, or email draft; owner-checked by the id the browser sends), the email reader and the saved email style (`email_style_context`), recent email ids from legacy rows only, integrations (when `api_call` is allowed), uploads and the image-edit hint (when `edit_image` is allowed). `system_sections()` adds only the static writing rules. `bind_process_state` sets the process-wide active document and model each turn. | #59 |
| `loop.py` | `ChatAgentTurn` + `run_chat_agent_turn`: rounds until a reply has no tool calls; each round's `tool_call` blocks are gated, then run one at a time in call order; their envelopes go back as `tool` messages. Emits `turn_mode`, `rag_sources`/`memories_used` (when non-empty, before the first round), `delta`, `tool_start`/`tool_output` (legacy shape + `call_id`), `message_saved`, `event: error`, `[DONE]`. Stops: `agent_max_rounds` → `rounds_exhausted`; `agent_max_tool_calls` → `budget_exceeded{kind: tool_calls}`; Finish (`agent_runs.should_finish`) → calls not yet run get a not-run result, then one round with no tools; repeat guard stop → a `repeat_guard` ask; Stop → `loop_breaker_triggered{reason: repeat_guard}`. An `ask` decision not covered by a session rule raises a `permission` ask; the model's `ask_user` tool raises a `question` ask; each waits in place (`_Ask`), and an expired ask ends the turn after its round (`stop_reason: ask_expired`). Writes `metadata.stop_reason` (`transcript.STOP_*`). Model fallback (`_round_with_fallback` around each round's provider call): the route passes the foreground candidates (`Candidate` = endpoint + route descriptor), the policy's eligible statuses and `stick_route`; a pre-output failure moves to the next candidate (legacy `fallback` event, session model/endpoint persisted, rest of the turn stays there), one retry on 502/503/504/529 with a single candidate, request-local compaction once when the fallback's window is smaller; a failure after output passes through. A crash ends the turn with a visible error; it never falls back to legacy. | #48, #50, #54, #55, #56 |
| `repeat_guard.py` | Ported from Kimi CLI (Apache-2.0 header, "modified" note, `NOTICE`). Consecutive identical calls (tool + canonical JSON args) in call order, per turn; reminders appended to the model-facing result at 3/5/8, stop signal at 12 (constants at the top). | #54 |
| `permissions.py` | `build_deny_layers` (managed, raw toggles, request state, plan mode incl. MCP writes, bare user deny rules; `hidden` never reaches the model, `disabled` is listed as unavailable), `POSTURE` table for every native/email tool (four levels: allow, ask, sensitive, protected). Rules: `parse_rule` (`tool` / `tool(specifier)`: bash/host_shell/python take the bare name only (a specifier is a ValueError, see Not decided yet), gitignore-style path glob for read/write/edit_file, `domain:` (IDNA) for web_fetch, `action:` for manage_*, `to:` for send_email; ValueError otherwise). Matching fails closed (HIT/MISS/UNSURE): an allow never covers a path that climbs out or resolves (workspace symlinks, `/workspace` alias) outside the glob; a deny or ask rule asks when it cannot read the call. `policy_name` folds every `mcp__email__` casing to the bare name. `session_rule` = what "Allow for this chat" saves (shell tools: the whole tool; manage_tasks: the exact call). `is_sensitive`, `is_protected(owner=, workspace=)` (a write_file/edit_file whose target resolves to, or cannot be placed off, an `AGENTS.md`/`CLAUDE.md`, or an apply_patch naming one, is protected, and so is a known workspace-writing tool whose target path can't be read, `download_attachment` exempt (no path argument); reads the call with `tools.system.manage_tasks_args` and the stored task with `tools.system.find_task`, do_manage_tasks' own parsing and lookup; unparsable or not found = protected), `task_card`, `task_args`, `Provenance` (egress legs, seeded from earlier turns' tool events). `decide()`: deny layers > deny rules > asks; asks from an ask rule, an unsure deny, the egress gate, a protected action or a manage_tasks create/edit/resume/run are `chat_scope=False` (no "Allow for this chat", session rules don't lift them); other asks a session rule lifts. | #50, #62 |
| `tools.py` | `DEFERRED_TOOLS` (the one split table), `upfront_schemas`, `unavailable_line`, `execute()`: the per-call wrapper (fresh `RequestAuthority` with one grant, casefolded owner, real session/workspace, backends sealed, `NO_TOOL_SECURITY_CONTEXT`, `tool_policy=None`, runtime deny set). `ToolOutcome` → the shared envelope and a tool event (exit code: done 0, failed code or 1, denied 126). `ExecutionScope.active_document_id` binds document tools to the open document. A `manage_tasks` create/edit approved with Allow once on its own card (`ExecutionScope.approved_task`) adds `create_request_authority(prompt)`'s grants, so the sealed task gets what a UI-created one does; any other admission adds none. `visual_evidence` turns a round's tool images (first image-bearing result, at most 3 evenly spaced frames) into one user message the next round sees and the one after drops. | #50, #59 |
| `providers/__init__.py` | Canonical conversation and block shapes both adapters serialize (`text`, `tool_call`, `reasoning` tagged with provider+model, tool-result envelope `{call_id, ok, content, error}`); `provider_key(url)` (`"xai"` for x.ai hosts, else `_detect_provider`); round events `TextDelta`, `RoundEnd(blocks, usage)`, `ProviderError`. | #51 |
| `providers/xai_responses.py` | xAI Responses: `store:false`, `prompt_cache_key` = chat id, `include: reasoning.encrypted_content`; each output item kept on its block as `item` and replayed unchanged; reasoning only to the same model; a 400 on replayed reasoning retries once without it; usage incl. cached tokens. | #51 |
| `providers/chat_completions.py` | Chat Completions dialect. Imports legacy transport/headers/URL helpers from `src.llm_core`; never the legacy message sanitizer. Streams `reasoning_details` / `reasoning_content` into a reasoning block and passes them back unchanged to the same model. Sends `tools` + `tool_choice: auto` when tools exist; streamed `delta.tool_calls` fragments become `tool_call` blocks. | #48, #50, #51 |
| `transcript.py` | `TurnRecord` → assistant row with `metadata.agent_v2` (`turn_id`, `provider`, `model`, `rounds[]` = `{blocks, usage?, results?}`, `content_digest`) plus projections `content`, `round_texts`, `tool_events` (`round`, `tool`, `command`, `output`, `status`, `exit_code`, `call_id`, `error`), `thinking`, `character_name`, `web_sources`, `rag_sources`, `memories_used`, `model`, `stop_reason`, `runtime: "chat_agent"`. `TurnWriter` inserts the row `in_progress` at the first round, rewrites it after each round end and each tool result (`Session.update_message`, found by `turn_id`), and finishes it; Stop keeps completed blocks and streamed text, unfinished calls get `cancelled` (130). `content_digest` = HMAC (app key) over persisted content + record; `trusted_rounds` replays only a verifying, un-`edited`, finished row. `sweep_interrupted_turns` (app startup) marks leftover `in_progress` rows `interrupted` and expires their open asks (`ask_user.resolved = "expired"`, `expired_cause`). Nobody mode writes the request-local incognito transcript once, at finish. The last `round_texts` entry (and so `content` and `final_response`) has note/email/calendar titles linked from `link_sources` (run calls' result fields + tool events, not persisted); `agent_v2` keeps the model's words. | #48, #50, #51, #57 |
| `asks.py` | Ask registry: a Future per ask keyed by (session, run, ask id), bound to the owner, `ASK_TIMEOUT_SECONDS` (30 min). Flags the run `awaiting_user` while waiting (`interactive_gate` skips such runs). Card builders in the legacy `ask_user` shape (`kind: tool_approval` + `approval_id` = `ask_id` for permission and repeat-guard asks; question asks as the `ask_user` tool's payload) plus `ask_id`/`ask_kind`; `ask_resolved` / `ask_expired` events; stored cards carry `resolved`. Answered only via `POST /api/chat/ask/{sid}` (`{ask_id, decision: once|chat|deny|answer, text?}`; 403 loopback/bearer, 409 when no open ask of that session's live run and owner matches). | #56 |
| `result_events.py` | Port of the legacy inline result-to-event adapter (`adapt(outcome, block, ...)` → events, model text, tool event). Keeps branches with a frontend reader: `web_sources` (SOURCES marker stripped; the row's `web_sources` is the last search's), editor `doc_stream_open`/`doc_stream_delta`, `doc_update`/`doc_suggestions` (+ the post-output re-activating `doc_update`), `ui_control`, `plan_update`, `tool_output` (legacy fields + `call_id`; runtime exit code/blocked), `generated_image`; model text = `format_tool_result` after the draft-email document id and the calendar open range are added, `web_search` compacted to 3200 chars. Pinned by `tests/chat_agent/test_chat_agent_result_events_parity.py` against the legacy loop. The loop also emits `agent_step` at each round start, `tool_progress` (`progress_cb`), and `metrics` (legacy `_compute_final_metrics` + usage buckets, no completion fields) at turn end and on a provider failure; the same metrics go to post-response tasks (session token totals). | #53 |
| `classifier.py` | Auto mode (ADR 0006). Request form field `auto_mode=true` → `ChatAgentTurn.auto_mode`; `AutoMode.for_turn` makes it `unavailable` for non-interactive turns and users without the `can_use_auto_mode` privilege, else `active` or `paused`. `decide(auto_mode=)` drops shell (bash/host_shell/python) allow and session rules while it is on; an ask with `Decision.auto` (not an explicit ask rule, unsure deny, protected action or manage_tasks create/edit/resume/run) goes to one classifier call (no tools, no cache, `TIMEOUT_SECONDS` = `LLMConfig.DEFAULT_TIMEOUT`) on the `classifier_endpoint_id`/`classifier_model` setting when this user may use that model, else the chat's model. Input JSON: real user rows (`interaction_mode` set, stamped only on interactive non-token turns) of the replayed history (Nobody mode: the request-local transcript) + this message, instruction files (read once at turn start), non-read-only tool-call payloads of the chat, `Provenance` facts (external origins, private read, whether each recipient/host appears in the user's messages), the judged call (JSON arguments parsed as the executor reads them; a repeated key asks the user). Allow runs it; deny → error result `Denied by auto mode: <reason>` with `denied_by: "auto_mode"` on `tool_output` and the stored tool event; a failed call or any answer that is not exactly one JSON verdict (fences allowed, duplicate keys refused) raises the normal ask and is not counted. `PAUSE_AFTER_DENIALS_IN_A_ROW` (3) / `PAUSE_AFTER_DENIALS_IN_CHAT` (20), stored per chat in `store.auto_mode_denials`; a paused chat asks until an ask is approved, which resets both. Events: `auto_mode {state: active\|paused\|unavailable}` at turn start (when requested) and on each change. | #68 |

Control-plane route: `routes/chat_agent_routes.py`
(`GET/PUT /api/chat_agent/settings`, `GET/PUT /api/chat_agent/instructions`, `GET/PUT /api/chat_agent/rules` `{rules: {allow, ask, deny}}`, normalized rule text, 400 on a rule outside the table; `GET /api/chat_agent/auto_mode/{sid}` → `{available, paused}`). Context info (`GET /api/session/{sid}/context_info?cwd=`) lists the loaded instruction files as `instructions: [{path, truncated}]`. `require_interactive_user` refuses
auth-disabled instances, internal-tool loopback and bearer tokens; reuse it for
the ask/steer/rules routes.

## Not decided yet

- Shell-command specifiers (Claude-Code-style `bash(git:*)`) are not supported
  (2026-10-13): `PUT /api/chat_agent/rules` refuses them with a 400. Shell is
  governed by the composer Shell toggle (off = managed deny) and bare-name
  rules on `bash` / `host_shell` / `python` (default posture ask; "Allow for
  this chat" allows the whole tool in that chat). Whether to add best-effort
  specifiers later is open.

## Routing: what stays on legacy

Switch off; auth disabled or no owner; not an agent turn after the legacy
promotion (or `can_use_agent` off); compare mode or `[CMP]` sessions; any
`client_runtime_context.surface` (TUI/native, bridges, client tools);
image-generation sessions; the research path (`use_research` or
`research_pending`); legacy approval continuations (`tool_approval_id`);
internal-tool loopback requests; providers without an adapter (`anthropic`,
`ollama`, `chatgpt-subscription` per `llm_core._detect_provider`); endpoints the
legacy predicate `agent_loop._agent_route_tool_mode` runs on text tools.

## Conventions for later tickets

- Every imported private legacy helper is listed with its relied-on assumption
  in a comment at the import and pinned in
  `tests/chat_agent/test_chat_agent_legacy_pins.py`.
- Persisted paths come from `src/constants.py` (`CHAT_AGENT_DIR`); directories
  are created on first write.
- `prompt._replay` turns a history row with `agent_v2.rounds` into
  `assistant{blocks}` + `tool{...}` entries (round `results` are the tool-result
  envelopes) only when `transcript.trusted_rounds` verifies it; other rows
  (incl. client-written or edited `agent_v2`) replay as text, and adjacent
  same-role text rows merge (blank line, or concatenated parts).
- Tests live in `tests/chat_agent/` with unique basenames. The primary seam is
  the chat route with a scripted HTTP transport: patch
  `providers.chat_completions._get_http_client` with an `httpx.MockTransport`
  client (see `test_chat_agent_route.py::_Harness`).
