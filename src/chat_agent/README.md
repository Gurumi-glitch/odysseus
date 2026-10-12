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
| `store.py` | Per-user JSON store `CHAT_AGENT_DIR/<percent-encoded lowercased user>.json` (confined). Holds `chat_runtime: legacy\|new` and `instructions_enabled`. Owns the user instruction folder `INSTRUCTIONS_DIR/<same name>/` (`confine()`d). `rename_owner` / `delete_owner` are called from the user rename/delete routes. Rules and session rules join here. | #48, #52 |
| `routing.py` | `ComposerToggles` snapshot; `choose_runtime(TurnFacts)` → `legacy` or `chat_agent`. | #48 |
| `prompt.py` | Calls the route's context builder with `prefetch=False, agent_mode=False` (no URL/web/YouTube prefetch, no legacy skills index) and assembles: one system message (route system messages, then the behaviour section, then runtime sections such as the unavailable-tools line) → instructions block (user role: user folder unless disabled, then workspace ancestors up to home, `AGENTS.md` + `CLAUDE.md`, 12k chars each, labelled by path; none in Nobody mode) → history → current user message carrying the turn context (memory, RAG, date/time). | #48, #50, #52 |
| `loop.py` | `ChatAgentTurn` + `run_chat_agent_turn`: rounds until a reply has no tool calls; each round's `tool_call` blocks are gated, then run one at a time in call order; their envelopes go back as `tool` messages. Emits `turn_mode`, `delta`, `tool_start`/`tool_output` (legacy shape + `call_id`), `message_saved`, `event: error`, `[DONE]`. Stops: `agent_max_rounds` → `rounds_exhausted`; `agent_max_tool_calls` → `budget_exceeded{kind: tool_calls}`; Finish (`agent_runs.should_finish`) → calls not yet run get a not-run result, then one round with no tools; repeat guard stop → a `repeat_guard` ask; Stop → `loop_breaker_triggered{reason: repeat_guard}`. An `ask` decision not covered by a session rule raises a `permission` ask; the model's `ask_user` tool raises a `question` ask; each waits in place (`_Ask`), and an expired ask ends the turn after its round (`stop_reason: ask_expired`). Writes `metadata.stop_reason` (`transcript.STOP_*`). Model fallback (`_round_with_fallback` around each round's provider call): the route passes the foreground candidates (`Candidate` = endpoint + route descriptor), the policy's eligible statuses and `stick_route`; a pre-output failure moves to the next candidate (legacy `fallback` event, session model/endpoint persisted, rest of the turn stays there), one retry on 502/503/504/529 with a single candidate, request-local compaction once when the fallback's window is smaller; a failure after output passes through. A crash ends the turn with a visible error; it never falls back to legacy. | #48, #50, #54, #55, #56 |
| `repeat_guard.py` | Ported from Kimi CLI (Apache-2.0 header, "modified" note, `NOTICE`). Consecutive identical calls (tool + canonical JSON args) in call order, per turn; reminders appended to the model-facing result at 3/5/8, stop signal at 12 (constants at the top). | #54 |
| `permissions.py` | `build_deny_layers` (managed, raw toggles, request state, plan mode incl. MCP writes; `hidden` never reaches the model, `disabled` is listed as unavailable), `POSTURE` table for every native/email tool plus call-level overlays (destructive actions, `manage_tasks` create/edit/resume/run, loosening `ui_control`), `decide()`. | #50 |
| `tools.py` | `DEFERRED_TOOLS` (the one split table), `upfront_schemas`, `unavailable_line`, `execute()`: the per-call wrapper (fresh `RequestAuthority` with one grant, casefolded owner, real session/workspace, backends sealed, `NO_TOOL_SECURITY_CONTEXT`, `tool_policy=None`, runtime deny set). `ToolOutcome` → the shared envelope and a tool event (exit code: done 0, failed code or 1, denied 126). | #50 |
| `providers/__init__.py` | Canonical conversation and block shapes both adapters serialize (`text`, `tool_call`, `reasoning` tagged with provider+model, tool-result envelope `{call_id, ok, content, error}`); `provider_key(url)` (`"xai"` for x.ai hosts, else `_detect_provider`); round events `TextDelta`, `RoundEnd(blocks, usage)`, `ProviderError`. | #51 |
| `providers/xai_responses.py` | xAI Responses: `store:false`, `prompt_cache_key` = chat id, `include: reasoning.encrypted_content`; each output item kept on its block as `item` and replayed unchanged; reasoning only to the same model; a 400 on replayed reasoning retries once without it; usage incl. cached tokens. | #51 |
| `providers/chat_completions.py` | Chat Completions dialect. Imports legacy transport/headers/URL helpers from `src.llm_core`; never the legacy message sanitizer. Streams `reasoning_details` / `reasoning_content` into a reasoning block and passes them back unchanged to the same model. Sends `tools` + `tool_choice: auto` when tools exist; streamed `delta.tool_calls` fragments become `tool_call` blocks. | #48, #50, #51 |
| `transcript.py` | `TurnRecord` → assistant row with `metadata.agent_v2` (`turn_id`, `provider`, `model`, `rounds[]` = `{blocks, usage?, results?}`) plus projections `content`, `round_texts`, `tool_events` (`round`, `tool`, `command`, `output`, `status`, `exit_code`, `call_id`, `error`), `model`, `runtime: "chat_agent"`. Nobody mode writes the request-local incognito transcript. | #48, #50, #51 (write at turn end) |
| `asks.py` | Ask registry: a Future per ask keyed by (session, run, ask id), bound to the owner, `ASK_TIMEOUT_SECONDS` (30 min). Flags the run `awaiting_user` while waiting (`interactive_gate` skips such runs). Card builders in the legacy `ask_user` shape (`kind: tool_approval` + `approval_id` = `ask_id` for permission and repeat-guard asks; question asks as the `ask_user` tool's payload) plus `ask_id`/`ask_kind`; `ask_resolved` / `ask_expired` events; stored cards carry `resolved`. Answered only via `POST /api/chat/ask/{sid}` (`{ask_id, decision: once|chat|deny|answer, text?}`; 403 loopback/bearer, 409 when no open ask of that session's live run and owner matches). | #56 |
| `session_rules.py` | Interim "Allow for this chat": in-memory `tool(specifier)` rules bound to owner + chat (bash/host_shell command prefix, every compound part must match, no substitution/redirection; file path; `action:`; email recipient; MCP qualified name; anything else only the exact call). #62 moves them into the store. | #56 |
| `classifier.py`, `result_events.py` | Not yet built. | #53–#68 |

Control-plane route: `routes/chat_agent_routes.py`
(`GET/PUT /api/chat_agent/settings`, `GET/PUT /api/chat_agent/instructions`). Context info (`GET /api/session/{sid}/context_info?cwd=`) lists the loaded instruction files as `instructions: [{path, truncated}]`. `require_interactive_user` refuses
auth-disabled instances, internal-tool loopback and bearer tokens; reuse it for
the ask/steer/rules routes.

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
  envelopes); rows without it replay as text.
- Tests live in `tests/chat_agent/` with unique basenames. The primary seam is
  the chat route with a scripted HTTP transport: patch
  `providers.chat_completions._get_http_client` with an `httpx.MockTransport`
  client (see `test_chat_agent_route.py::_Harness`).
