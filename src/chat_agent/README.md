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
| `store.py` | Per-user JSON store `CHAT_AGENT_DIR/<percent-encoded casefolded user>.json` (confined). Holds `chat_runtime: legacy\|new` and `instructions_enabled`. Owns the user instruction folder `INSTRUCTIONS_DIR/<same name>/` (`confine()`d). `rename_owner` / `delete_owner` are called from the user rename/delete routes. Rules and session rules join here. | #48, #52 |
| `routing.py` | `ComposerToggles` snapshot; `choose_runtime(TurnFacts)` → `legacy` or `chat_agent`. | #48 |
| `prompt.py` | Calls the route's context builder with `prefetch=False, agent_mode=False` (no URL/web/YouTube prefetch, no legacy skills index) and assembles: one system message (route system messages, then the behaviour section) → instructions block (user role: user folder unless disabled, then workspace ancestors up to home, `AGENTS.md` + `CLAUDE.md`, 12k chars each, labelled by path; none in Nobody mode) → history → current user message carrying the turn context (memory, RAG, date/time). | #48, #52 |
| `loop.py` | `ChatAgentTurn` + `run_chat_agent_turn`: emits `turn_mode`, `delta`, `message_saved`, `event: error`, `[DONE]`. A crash ends the turn with a visible error; it never falls back to legacy. | #48 (one round, no tools) |
| `providers/chat_completions.py` | Chat Completions dialect. Imports legacy transport/headers/URL helpers from `src.llm_core`; never the legacy message sanitizer. | #48 (text deltas only) |
| `transcript.py` | `TurnRecord` → assistant row with `metadata.agent_v2` (`turn_id`, `provider`, `model`, `rounds[].blocks[]`) plus projections `content`, `round_texts`, `model`, `runtime: "chat_agent"`. Nobody mode writes the request-local incognito transcript. | #48 (write at turn end) |
| `permissions.py`, `classifier.py`, `tools.py`, `result_events.py`, `asks.py`, `providers/xai_responses.py` | Not yet built. | #50–#68 |

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
- Tests live in `tests/chat_agent/` with unique basenames. The primary seam is
  the chat route with a scripted HTTP transport: patch
  `providers.chat_completions._get_http_client` with an `httpx.MockTransport`
  client (see `test_chat_agent_route.py::_Harness`).
