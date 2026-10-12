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
| `store.py` | Per-user JSON store `CHAT_AGENT_DIR/<percent-encoded casefolded user>.json` (confined). Holds `chat_runtime: legacy\|new`. Rules, session rules and rename/delete hooks join here. | #48 (switch only) |
| `routing.py` | `ComposerToggles` snapshot; `choose_runtime(TurnFacts)` → `legacy` or `chat_agent`. | #48 |
| `prompt.py` | Calls the route's context builder with `prefetch=False, agent_mode=False` (no URL/web/YouTube prefetch, no legacy skills index) and assembles: one system message (route system messages, then the behaviour section) → history → current user message carrying the turn context (memory, RAG, date/time). | #48 (minimal) |
| `loop.py` | `ChatAgentTurn` + `run_chat_agent_turn`: rounds until a reply has no tool calls; each call is gated, then run one at a time in call order. Emits `turn_mode`, `delta`, `tool_start`/`tool_output` (legacy shape + `call_id`), `message_saved`, `event: error`, `[DONE]`. A crash ends the turn with a visible error; it never falls back to legacy. | #50 (no caps yet: #54) |
| `permissions.py` | `build_deny_layers` (managed, raw toggles, request state, plan mode incl. MCP writes; `hidden` never reaches the model, `disabled` is listed as unavailable), `POSTURE` table for every native/email tool plus call-level overlays (destructive actions, `manage_tasks` create/edit/resume/run, loosening `ui_control`), `decide()`. Until #56 every `ask` returns a denial result. | #50 |
| `tools.py` | `DEFERRED_TOOLS` (the one split table), `upfront_schemas`, `unavailable_line`, `execute()`: the per-call wrapper (fresh `RequestAuthority` with one grant, casefolded owner, real session/workspace, backends sealed, `NO_TOOL_SECURITY_CONTEXT`, `tool_policy=None`, runtime deny set). `ToolOutcome` → model envelope (`ok` text, or JSON `{error, status}`) and exit code (done 0, failed code or 1, denied 126). | #50 |
| `providers/chat_completions.py` | Chat Completions dialect: text deltas, then the round's assembled tool calls; `tools` + `tool_choice: auto` when tools exist. Imports legacy transport/headers/URL helpers from `src.llm_core`; never the legacy message sanitizer. | #48, #50 |
| `transcript.py` | `TurnRecord.rounds[]` (text, tool calls, outcomes) → assistant row with `metadata.agent_v2` (`rounds[].blocks[]`: `text`, `tool_call`, `tool_result` with uncut model text) plus projections `content`, `round_texts` (one per round), `tool_events` (`round`, `status`, `exit_code`, `call_id`, `error`), `model`, `runtime: "chat_agent"`. | #48, #50 (write at turn end) |
| `classifier.py`, `result_events.py`, `asks.py`, `providers/xai_responses.py` | Not yet built. | #51–#68 |

Control-plane route: `routes/chat_agent_routes.py`
(`GET/PUT /api/chat_agent/settings`). `require_interactive_user` refuses
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
- Tests live in `tests/chat_agent/` with unique basenames. The primary seam is
  the chat route with a scripted HTTP transport: patch
  `providers.chat_completions._get_http_client` with an `httpx.MockTransport`
  client (see `test_chat_agent_route.py::_Harness`).
