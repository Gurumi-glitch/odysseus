# Claude Code agent loop and clean reference code

Research for wayfinder ticket #21 (map #20). Sources checked 2026-10-10. Only primary sources were used: official Claude Code and Agent SDK docs, the Claude API docs, and open-source agent CLIs read at the commits pinned below. No leaked source was used.

## Answer summary

- **Loop shape (Claude Code / Agent SDK).** Call the model, run every requested tool, append the results, and call again. The loop ends on the first response with no tool calls. A "turn" is one tool-use round trip. There is no default cap: `max_turns` and `max_budget_usd` are opt-in, and hitting one ends with `error_max_turns` or `error_max_budget_usd` ([agent-loop]).
- **Stop conditions.** The documented result subtypes are `success`, `error_max_turns`, `error_max_budget_usd`, `error_during_execution` (crash or cancel) and `error_max_structured_output_retries`. `stop_reason` carries the API value (`end_turn`, `max_tokens`, `refusal`) ([agent-loop]).
- **Tool errors and denials are ordinary tool results.** A denied call reaches the model as the tool result, and Claude "typically attempts a different approach or reports that it couldn't proceed" ([agent-loop]). The API convention is `tool_result` with `is_error: true` and an instructive message. Claude retries an invalid call 2-3 times before giving up ([handle-tool-calls]). No documented Claude Code path strips the tools to force an answer.
- **Odysseus contrast.** Odysseus has 33 `_force_answer = True` sites in `src/agent_loop.py`. Each makes the next round run with no tools. `_failed_tool_round_limit` (line 9596) allows only 2 failed rounds outside terminal-agent mode (3-8 inside it). Claude Code relies on the model recovering from error results under a turn or budget cap instead.
- **Doom-loop guards in OSS CLIs.** None of these CLIs force-answer on the first error:
  - OpenCode asks the user through a `doom_loop` permission after 3 identical calls.
  - Kimi CLI escalates reminders at 3, 5 and 8 repeats and force-stops at 12.
  - Codex returns every non-fatal tool error as `success:false` output.
- **"Done" is model-decided.** A text-only response ends the loop. External checks can veto it: a `Stop` hook returning `decision:"block"` with a `reason` keeps Claude working ([hooks]). Codex runs the same Stop-hook gate in `run_turn`.
- **Todo/plan tool is optional and model-dependent.** `TaskCreate`/`TaskUpdate`/`TaskGet`/`TaskList` (or legacy `TodoWrite`) are on by default only for Claude 3.x, Opus 4-4.7, Sonnet 4-4.6 and Haiku 4.5. Newer models "track multi-step work without a written todo list" and need an opt-in ([todo-tracking]). The statuses are pending, in_progress, completed and deleted. A reminder nudges Claude when the list goes stale.
- **Plan tools elsewhere.** Codex's `update_plan` and Kimi's `SetTodoList` are model-written status lists, rendered by the UI, that never gate completion.
- **Interruption.** Esc stops the current response or tool call and keeps the work done so far. A message typed mid-turn is queued and delivered after the current tool calls, within the same turn ([interactive-mode]). The SDK exposes `interrupt()`, and a permission deny can carry `interrupt=True` (Python SDK `types.py`/`_internal/query.py`).
- **Prompt layout.** The system prompt holds the static preset: identity, tool-usage and safety rules. Tool schemas travel in the API `tools` field, and some guidance lives in tool descriptions; for example, the git commit rules are part of the Bash description. CLAUDE.md, skills list, hook output, todo nudges and file-change notes arrive as `<system-reminder>` text inside user messages, not in the system prompt ([modifying-system-prompts]).
- **Prompt layout elsewhere.** Codex does the same: base instructions go in `instructions`, while AGENTS.md and the environment are user-role fragments. OpenCode puts provider prompt, environment and instructions into `system` each step.
- **Compaction.** Near the context limit the SDK summarizes older history and emits `compact_boundary`. System prompt, CLAUDE.md, memory and MCP tools reload, up to 5 recently edited files are re-read, and invoked skills are re-injected (capped at 5k tokens each) ([context-window], [agent-loop]). Codex, OpenCode, Kimi and Goose all auto-compact on a token threshold with a "handoff summary" prompt.
- **Deferred tools.** Tool search is on by default in the SDK. Deferred schemas are withheld, Claude searches, and up to 5 matches load and stay loaded until compaction. Core tools (Bash/Read/Edit) always load upfront. `auto:N` switches modes at N% of the context window ([tool-search]). The Claude API notes that accuracy degrades above 30-50 loaded tools.
- **Deferred tools vs Odysseus.** Codex also ships a client-side `tool_search` (`tools/handlers/tool_search.rs`). This model-driven search contrasts with Odysseus's server-side k=8 retrieval per turn.
- **Clean code.** Python Agent SDK (MIT) is only a subprocess/control-protocol wrapper around the bundled Claude Code binary, so it contains no loop. The TypeScript SDK repo is "All rights reserved" under Anthropic Commercial Terms, so it is not reusable code. The portable loop references are Codex (Apache-2.0), OpenCode (MIT), Kimi CLI (Apache-2.0) and Goose (Apache-2.0). Kimi CLI is the closest to Odysseus because it is Python, and it is the one the user can run hands-on: see §6 for a deep dive and a side-by-side observation checklist. Codex and OpenCode were studied from source only.

## 1. Loop shape: turns, rounds and stop conditions

### Claude Code / Agent SDK (docs)

- The cycle runs: receive prompt, evaluate, execute tools, repeat, return result. "Each full cycle is one turn. Claude continues calling tools and processing results until it produces a response with no tool calls." ([agent-loop] "The loop at a glance")
- `max_turns` counts tool-use turns only. The default is no limit, and the same holds for `max_budget_usd`. The budget is checked after each response arrives, so spend can overshoot by one response ([agent-loop] "Turns and budget").
- Read-only tools (and MCP tools with `readOnlyHint`) can run in parallel. Mutating tools (Edit, Write, Bash) run sequentially ([agent-loop] "Parallel tool execution").
- Result subtypes and `stop_reason` are listed in the summary ([agent-loop] "Handle the result").
- The conceptual phases are gather context, take action and verify results. "You can interrupt at any point to steer" ([how-claude-code-works]).

### Open-source comparables

- **Codex.** See `codex-rs/core/src/session/turn.rs` `run_turn`; its doc comment states the contract. On a function call it executes the call and samples again. When the model returns only an assistant message, the turn is complete. Inside the `loop {}`:
  - `needs_follow_up = model_needs_follow_up || has_pending_input`, so queued user input also continues the turn.
  - On `token_limit_reached` it runs `run_auto_compact(..., CompactionPhase::MidTurn)` and then `continue`s.
  - When `!needs_follow_up` it runs Stop hooks, which can block the stop and inject a continuation prompt.
- **OpenCode.** See `packages/opencode/src/session/prompt.ts` `runLoop`, a `while (true)` loop:
  - It exits when the last assistant `finish` is not `tool-calls`/`unknown` and the message has no tool parts. The code comments that some providers return `"stop"` even when tool calls are present.
  - When `agent.steps` (default `Infinity`) is reached, it appends `MAX_STEPS_PROMPT` (`packages/core/src/session/runner/max-steps.ts`). That prompt disables tools and demands a summary, the remaining work and next steps.
  - The processor returns `"compact" | "stop" | "continue"` (`session/processor.ts`).
- **Kimi CLI.** See `src/kimi_cli/soul/kimisoul.py` `_agent_loop`:
  - Each step runs, in order: the step guard (`max_steps_per_turn`, default 1000, in `config.py`), auto-compaction, a checkpoint, then `_step()`.
  - Step stop reasons are `no_tool_calls | tool_rejected | tool_call_repeat`.
  - Pending "steers" (mid-turn user input) force another step before the turn resolves.
- **Goose.** See `crates/goose/src/agents/agent.rs`: `DEFAULT_MAX_TURNS = 1000` and `MaxTurnsOperation`.
- **Aider.** Aider is not a tool-call loop. It uses edit formats with a "reflection" retry capped at `max_reflections = 3` (`aider/coders/base_coder.py`). It is included only for contrast.

## 2. Tool errors and denials: how they reach the model

### Claude Code / Claude API (docs)

- A tool execution error is sent back as `tool_result` content with `"is_error": true`. The docs advise instructive messages that say what went wrong and what to try next. On invalid calls, "Claude will retry 2-3 times with corrections before apologizing to the user." ([handle-tool-calls])
- A permission denial becomes the tool result: "When a tool is denied, Claude receives a rejection message as the tool result and typically attempts a different approach or reports that it couldn't proceed." ([agent-loop] "Tool permissions")
- Permission evaluation order is hooks, then deny rules, ask rules, permission mode, allow rules and finally the `canUseTool` callback ([permissions]).
  - `canUseTool` can deny with a `message` that tells Claude why ([user-input]).
  - The Python SDK's `PermissionResultDeny(message, interrupt=False)` (`src/claude_agent_sdk/types.py`) is serialized to `{"behavior":"deny","message":...,"interrupt":...}` in `_internal/query.py` `_handle_control_request`. `interrupt=True` also stops the turn.
- A `PreToolUse` hook deny also feeds back as the tool result ([agent-loop] "Hooks").
- **No forced-answer mechanism is documented.** The only hard stops are the turn and budget caps, after which the SDK returns an error result and the loop does not synthesize an answer.

### Open-source comparables

- **Codex.**
  - `codex-rs/core/src/tools/parallel.rs`: `handle_tool_call` maps any non-`Fatal` `FunctionCallError` through `failure_response` to a `FunctionCallOutput { success: Some(false), body: <error text> }`, so the model sees it and samples again. Only `FunctionCallError::Fatal` aborts the turn.
  - User aborts become `"aborted by user after Ns"` outputs.
  - Approval or sandbox rejections are `ToolError::Rejected(reason)` in `tools/orchestrator.rs`.
  - Argument-parse errors use `RespondToModel` (`tools/handlers/plan.rs` `parse_update_plan_arguments`).
- **OpenCode.**
  - `session/processor.ts` `failToolCall` marks the tool part `status:"error"`. `session/message-v2.ts` converts it to an `output-error` part with `errorText`.
  - Interrupted or dangling calls get `"[Tool execution was interrupted]"`, because the Anthropic API requires every `tool_use` to have a result.
  - Malformed calls route to the `invalid` tool (`tool/invalid.ts`), which returns "The arguments provided to the tool are invalid: ...".
  - A permission or question rejection sets `blocked`, which stops the loop, unless `experimental.continue_loop_on_deny` is set.
  - Doom loop: 3 identical consecutive calls trigger `permission.ask({permission:"doom_loop"})` (`DOOM_LOOP_THRESHOLD = 3`), which hands the decision to the user.
- **Kimi CLI.**
  - `soul/toolset.py` returns `ToolParseError`, `ToolError` and `ToolRuntimeError` as tool results.
  - Repeated identical calls get escalating reminder text appended (`_REPEAT_REMINDER_1_START = 3`, `_2 = 5`, `_3 = 8`). At `_REPEAT_FORCE_STOP_STREAK = 12` the turn ends with `tool_call_repeat`.
  - A rejection *with* user feedback is returned as "The tool call is rejected by the user. User feedback: ..." and the loop continues.
  - A pure rejection without feedback ends the root turn (`tool_rejected`). Subagents instead get "Try a different approach ... Do not retry the same tool call" (`soul/approval.py` `ApprovalResult.rejection_error`, `kimisoul.py` `_step`).

### Contrast with Odysseus

- `src/agent_loop.py`, on `devGuru` at the time of writing:
  - `_force_answer` is declared at line 24845 ("set by loop-breaker → next round runs with NO tools") and set to `True` at 33 sites.
  - `_failed_tool_round_limit` (line 9596) returns 2 unless the context is a terminal agent; terminal agents get 3-8, default 5. It is used at line 36975.
  - When a forced round still emits native calls, they are discarded (around line 29751).
- `src/agent_runtime/completion.py` `with_completion_gate` (line 241) wraps the loop with an evidence-ledger completion gate. That is roughly where a Claude Code `Stop` hook would sit.
- **The structural difference.** In the reference loops a failure is information for the next model call, and termination comes from the model, a cap, a user rejection or an interrupt. Odysseus instead removes tools after a small number of failures or after heuristic triggers.

## 3. Done, progress and interruption

- **Done (Claude Code).** Done means the first assistant response with no tool calls; the [agent-loop] doc gives no other completion criterion.
- **Vetoing "done".** A `Stop` hook can veto it: `decision:"block"` "prevents Claude from stopping", `reason` is required and tells Claude why it should continue, and `additionalContext` gives non-error feedback ([hooks] "Stop decision control"). `SubagentStop` works the same way for subagents.
- **Codex.** The same gate: `run_turn_stop_hooks`, then `should_block` records a hook prompt message and continues.
- **Todo tools (Claude Code).**
  - The default set depends on the model, as described in the summary ([todo-tracking] "Model availability").
  - The lifecycle is pending, in_progress, completed and deleted (`TaskUpdate status:"deleted"`).
  - Claude uses the tools for tasks of 3+ steps, for user-supplied lists and on explicit request.
  - The SDK stream shows them as ordinary `tool_use` blocks, which the UI renders itself.
  - A "task list nudge" reminder fires when the list hasn't been touched for several turns ([modifying-system-prompts] "Reminders").
- **Codex plan tool.** `update_plan` (`tools/handlers/plan_spec.rs` `create_update_plan_tool`) takes an `explanation` and `plan[{step, status: pending|in_progress|completed}]`, with "at most one step in_progress".
- **Kimi plan tool.** `SetTodoList` (`tools/todo/__init__.py`) uses `pending|in_progress|done`.
- **OpenCode plan tool.** `todowrite` (`tool/todo.ts`) plus a separate plan *mode*, whose plan-file reminders are injected by `session/reminders.ts`.
- **Interruption (Claude Code).**
  - Esc interrupts the current response or tool call and keeps the work so far. On a permission prompt, Esc equals "No".
  - Messages typed mid-turn are queued and passed to Claude once the running tool calls finish, within the same turn. Ctrl+Enter sends them immediately; this interrupts if Claude is only writing ([interactive-mode]).
  - SDK: `ClaudeSDKClient.interrupt()` sends a `{"subtype":"interrupt"}` control request (`_internal/query.py` `interrupt`).
- **Interruption elsewhere.** Codex and Kimi implement the same steer/queue semantics (`input_queue` in `run_turn`; `_steer_queue`/`_consume_pending_steers` in Kimi).

## 4. Prompt layout

- **Claude Code system prompt.** The `claude_code` preset holds identity, tool usage instructions, and security and safety instructions. `append` adds to the end ([modifying-system-prompts]).
- **Not in the system prompt.**
  - CLAUDE.md and environment details (cwd, platform, shell, OS) "don't affect the system prompt cache, because Claude Code delivers them in the conversation, not the system prompt".
  - `excludeDynamicSections` moves the remaining per-user section (the auto-memory location) into the first user message for cross-user caching.
- **System reminders.** These are added to the conversation, wrapped in `<system-reminder>` inside a user message, or on some models sent as a separate `system`-role message. They carry:
  - CLAUDE.md, with a line saying the instructions override defaults
  - output style
  - commit attribution
  - hook `additionalContext`
  - the available skills and subagents lists
  - task-list nudges
  - file-changed notes
- **Custom prompts.** A custom system prompt should say that reminders come from the application, not the user.
- **Tool descriptions.** The tool definitions themselves carry behavior guidance. For example, the commit/PR instructions are "part of the Bash tool's description" ([modifying-system-prompts] "Turn off the context your agent replaces").
- **Caching.** The static prefix (system prompt, tool definitions, CLAUDE.md) is prompt-cached. The TS SDK `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` splits a custom prompt into a static block and a per-request block ([agent-loop] "The context window", [modifying-system-prompts]).
- **Codex.**
  - `build_prompt` in `session/turn.rs` sends `base_instructions` plus `tools` plus `input`.
  - AGENTS.md is a **user-role** fragment wrapped in `# AGENTS.md instructions ... <INSTRUCTIONS>` (`core/src/context/user_instructions.rs`).
  - Environment, permissions, time reminders and similar are separate contextual fragments under `core/src/context/`.
- **OpenCode.**
  - The per-provider base prompt comes from `session/system.ts` `provider()`, which picks anthropic/gpt/gemini/default `.txt` files.
  - Each step builds `system = [...env, ...instructions, mcpInstructions, skills]` (`prompt.ts` `runLoop`).
  - Mode reminders are injected into messages (`session/reminders.ts`).

## 5. Context: compaction and deferred tool loading

### Compaction

- **Claude Code/SDK, automatic.** Near the limit, older history is summarized, and a `compact_boundary` system message is emitted ([agent-loop] "Automatic compaction").
- **What the summary keeps.** It keeps requests and intent, key technical concepts, files examined or modified with important snippets, errors and fixes, pending tasks and current work ([context-window]).
- **After compaction.**
  - The system prompt, CLAUDE.md, memory and MCP tools reload.
  - Up to 5 most-recently-modified files are re-read, and invoked skills are re-injected (capped at 5k tokens each).
  - The skills *listing* is not re-injected.
  - Deferred tools found by tool search must be searched again ([tool-search]).
- **Customization.** CLAUDE.md may contain summarization instructions, a `PreCompact` hook can run first (`trigger: manual|auto`), and `/compact` triggers compaction manually. The docs advise keeping persistent rules in CLAUDE.md, because they are re-injected every request while early prompt instructions may be lost.
- **Codex.**
  - The trigger is checked post-sampling: `token_limit_reached` calls `run_auto_compact(... MidTurn)`, and there is also a pre-turn `run_pre_sampling_compact`.
  - Prompt: `codex-rs/prompts/templates/compact/prompt.md` ("CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM ..."). The summary is prefixed with `summary_prefix.md`.
  - Recent user messages are kept up to `COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000` (`core/src/compact.rs` `build_compacted_history`).
- **OpenCode.**
  - `session/overflow.ts` `isOverflow` fires when tokens reach (input limit − reserved, default `COMPACTION_BUFFER = 20_000`).
  - `session/compaction.ts` adds `prune`: walking backwards, tool outputs beyond the most recent `PRUNE_PROTECT = 40_000` tokens are cleared. This runs only if more than `PRUNE_MINIMUM = 20_000` would be freed, and `skill` outputs are protected.
- **Kimi CLI.** `soul/compaction.py` `should_auto_compact` fires when tokens ≥ 0.85 × window, or tokens + 50_000 reserved ≥ window (`config.py`).
- **Goose.** `agents/agent.rs` uses `context_mgmt::auto_compact_threshold` and tool-pair compaction.

### Deferred tool loading

- **Claude Code/SDK.**
  - Tool search is on by default. Definitions are withheld, and Claude gets a summary and searches. Up to 5 tools load per search and persist until compaction.
  - Core built-ins load upfront. The feature falls back to upfront loading for non-first-party base URLs, because proxies drop `tool_reference` blocks.
  - `ENABLE_TOOL_SEARCH=auto:N` activates search at N% of the context window ([tool-search]).
  - Rationale: 50 tools can use 10-20K tokens, and "tool selection accuracy degrades with more than 30-50 tools loaded at once". With fewer than ~10 tools, upfront loading is faster.
  - The API-level mechanism is `defer_loading` plus the `tool_search_tool_regex` / `tool_search_tool_bm25` server tools ([tool-search-tool]).
- **Codex.** A client-side `tool_search` handler (`core/src/tools/handlers/tool_search.rs`) uses an embedder/scorer over a search index (`codex-rs/tools/src/tool_search.rs`) with `TOOL_SEARCH_DEFAULT_LIMIT`. MCP tools can be marked deferred (`router.rs` `deferred_tool_namespaces`).
- **Kimi CLI.** MCP tools load in the background at turn start (`_agent_loop` step 1a), but schemas are offered in full.
- **Odysseus.** Retrieves k=8 of ~88 schemas per turn server-side (map #20 baseline), so the model cannot ask for a tool outside the retrieved set.

## 6. Kimi CLI in depth (the one CLI available hands-on)

Codex and OpenCode were studied from source and docs only, because running them needs a ChatGPT subscription or similar. Kimi CLI with the Kimi API is the one CLI the user can run, so this section goes deeper. All paths are in [MoonshotAI/kimi-cli] @ `9ab1286b8f` (Apache-2.0).

### Loop, by lifecycle phase

- **Turn entry (`soul/kimisoul.py` `KimiSoul.run`).**
  - Sets an approval source ContextVar and refreshes OAuth.
  - Runs the `UserPromptSubmit` hook, then `_turn` and `_agent_loop`.
  - Maps exits to `interrupt_reason`: `MaxStepsReached` gives `"max_steps"`, `asyncio.CancelledError` gives `"user_cancelled"`, and any other exception gives `"error"`.
- **`_agent_loop`.** Its docstring lists the phases:
  - Turn init (discard stale steers; wait for background MCP loading).
  - Step guard (`max_steps_per_turn`, default 1000; raises `MaxStepsReached`).
  - `StepBegin`.
  - Auto-compaction check.
  - `_checkpoint()`.
  - `_step()`.
  - Error handling, in two branches:
    - `BackToTheFuture` reverts to a checkpoint and injects a "D-Mail" message.
    - Any other exception emits `StepInterrupted`, fires the `StopFailure` hook and re-raises. That ends the turn with no forced answer.
  - Outcome resolution: when a step returns a stop reason, `_consume_pending_steers()` runs first. Queued user input forces another step instead of ending.
- **`_step`.** Its docstring lists:
  - notifications
  - dynamic injections (`soul/dynamic_injection.py`; plan mode and AFK providers under `soul/dynamic_injections/`)
  - merging adjacent user messages
  - the LLM call (`kosong.step`)
  - waiting for all tool results
  - appending the assistant and tool messages
  - outcome resolution
- **Transport retry.** `tenacity` with `stop_after_attempt(max_retries_per_step=3)` and `wait_exponential_jitter(initial=0.3, max=5)`. Only `_is_retryable_error` errors are retried: connection, timeout, empty response, and HTTP 429/500/502/503/504. This retries *transport* failures only; tool failures are never retried by the loop.
- **Step outcomes** (`StepStopReason`):
  - `no_tool_calls` means done, and `final_message` is the assistant text.
  - `tool_rejected` means a pure user rejection at the root agent.
  - `tool_call_repeat` means the repeat force-stop fired.
  - Returning `None` means continue.
- **Stop hook (`run`).** After the turn, a `Stop` hook returning `action == "block"` with a `reason` re-runs `_turn` once, with the reason as a user message. `_stop_hook_active` guards against infinite re-trigger; this is the same contract as Claude Code's Stop hook.
- **Ralph mode.** `loop_control.max_ralph_iterations` (default 0, -1 means unlimited) wraps the prompt in a `FlowRunner.ralph_loop`. The same prompt is fed repeatedly with "Only choose STOP when the task is fully complete ... choose CONTINUE", and a `tool_rejected` outcome stops the flow. It is an opt-in "keep going until done" mode outside the core loop.

### Error feedback

- **Results, not exceptions.** `soul/toolset.py` returns `ToolParseError` (bad JSON args), `ToolError` and `ToolRuntimeError` (exception text) as tool results, so the model sees them on the next step.
- **Repeat guard (`_build_repeat_reminder`).** It counts the projected streak of identical canonical calls:
  - At 3 it appends "You are repeating the exact same tool call with identical parameters ... try a different method or parameters".
  - At 5 and 8 it sends stronger texts, including `repeated_times` and "did not make progress".
  - At 12 it sets `force_stop_turn`, and the step returns `tool_call_repeat`.

### Approval

- **Tool-initiated, not a central policy engine.**
  - A tool calls `Approval.request(sender, action, description, display)`; for example the shell tool uses `action="run command"` (`tools/shell/__init__.py`).
  - A call from outside a tool call raises `RuntimeError`.
- **Auto-approval.** `is_auto_approve()` is `yolo or afk` (`soul/approval.py`). `afk` (persisted) and `runtime_afk` (`--afk`/`--print`) mean no user is present.
- **Responses.**
  - `approve`
  - `approve_for_session` adds the `action` string to `auto_approve_actions`. The granularity is the action label, so "run command" covers all shell commands.
  - `reject`, optionally with feedback.
  - cancel
- **Rejection text (`ApprovalResult.rejection_error`).**
  - With feedback: "The tool call is rejected by the user. User feedback: …", and the loop continues.
  - Subagent without feedback: "Try a different approach … Do not retry the same tool call, and do not attempt to bypass this restriction".
  - Root without feedback: a bare `ToolRejectedError`, and `_step` ends the turn with `tool_rejected`.

### Compaction

- `soul/compaction.py` `SimpleCompaction(max_preserved_messages=2)` summarizes with `COMPACTION_SYSTEM_PROMPT` and prefixes the result with "Previous context has been compacted. Here is the compaction output:".
- The trigger is `should_auto_compact`: 0.85 ratio or 50k reserved.
- Injection providers are notified through `on_context_compacted`, so they can re-inject state.

### Worth observing in a hands-on side-by-side

These run Kimi CLI on the Kimi API against the same scenario in Odysseus chat.

1. **Failing command recovery.** Ask for a task whose first obvious command fails (missing binary, wrong path). Check whether the error text arrives as a tool result and whether the next step changes approach. Count steps until success or give-up, and compare with Odysseus forcing an answer after 2 failed rounds.
2. **Same-call repetition.** Provoke a call that keeps returning the same thing, such as polling a file that never appears. Check whether the reminders appear at 3, 5 and 8, whether behavior changes, and whether the force-stop lands at 12. Then compare where Odysseus's loop-breaker fires.
3. **Reject with and without feedback.** Reject a shell call with no comment: the turn should end. Reject one with a comment such as "use rg instead": the model should adapt in the same turn. Note how the approval panel (`ui/shell/visualize/_approval_panel.py`) presents once versus session.
4. **`approve_for_session` scope.** After approving one shell command for the session, check whether unrelated, riskier commands also skip the prompt. This is input for the allow/ask/deny rule-granularity decision.
5. **Mid-turn steering.** Type a correction while tools are running. Check whether it is consumed before the turn ends (`_consume_pending_steers`) and whether the model changes course without restarting.
6. **Done without tools.** Check how often the model stops with a text-only answer while work remains, for example in a multi-file task. Compare an ad-hoc Stop hook that blocks with a reason against Odysseus's completion gate.
7. **Todo list use.** On a multi-step task, check whether the model calls `SetTodoList` unprompted, how often it updates it, and whether the list stays accurate.
8. **Long-session compaction.** Push context past 85%. Inspect the summary quality, which facts survive (files, decisions, pending work), and whether the model repeats finished work afterwards.
9. **Interrupt (Ctrl+C / cancel).** Check what the transcript keeps for an interrupted tool call and how the next turn resumes.
10. **Max steps.** Set a small `max_steps_per_turn`. Kimi raises `MaxStepsReached` with no summary answer, unlike OpenCode's tools-off summary prompt. Note what the user sees.

## Comparison table

| | Claude Code / Agent SDK (docs) | Codex CLI | OpenCode | Kimi CLI | Goose | Odysseus (today) |
|---|---|---|---|---|---|---|
| Loop shape | sample, run tools, repeat until no tool calls; parallel read-only tools | `run_turn` loop; `needs_follow_up` incl. queued input | `runLoop` `while(true)`; exit on finish≠tool-calls and no tool parts | `_agent_loop` steps; stop reasons no_tool_calls/tool_rejected/tool_call_repeat | agent loop with `MaxTurnsOperation` | `stream_agent_loop` rounds plus many heuristic branches |
| Turn cap default | none (`max_turns`, `max_budget_usd` opt-in) | none found in `run_turn` | `agent.steps` default ∞; last step gets `MAX_STEPS_PROMPT` (tools off) | `max_steps_per_turn=1000` | `DEFAULT_MAX_TURNS=1000` | small failed-round limit (2; 3-8 terminal) |
| Error → model | `tool_result is_error:true`; deny message as tool result | `FunctionCallOutput success:false` with error text; only `Fatal` aborts | `output-error` part with `errorText`; `invalid` tool for bad args | `ToolError`/`ToolRuntimeError`/`ToolParseError` results | declined → "This request was declined." result | errors fed back, but force-answer strips tools after limits/triggers |
| Repeat / stuck handling | none documented beyond caps | none specific found | 3 identical calls → `doom_loop` permission ask | reminders at 3/5/8, force stop at 12 | not checked | loop-breaker → `_force_answer` |
| Approval / ask | hooks → deny → ask → mode → allow → `canUseTool`; modes default/acceptEdits/plan/dontAsk/auto/bypass | approval + sandbox orchestrator; `ToolError::Rejected` | permission rules allow/ask/deny; deny stops loop by default | approval runtime; reject w/o feedback stops root turn, with feedback continues | permission modes (not examined) | regex request authority; approval cards after tainted context |
| Done criteria | no-tool-call response; `Stop` hook can block | same plus Stop hooks | finish reason | `no_tool_calls` | no tool calls | completion gate with evidence ledger |
| Plan / todo | `TaskCreate/Update/Get/List` or `TodoWrite`; default off on newest models | `update_plan` | `todowrite` + plan mode | `SetTodoList` + plan mode | not examined | none |
| Compaction | auto near limit; structured summary; re-inject CLAUDE.md, ≤5 files, skills | auto mid-turn + pre-turn; handoff-summary prompt; keep 20k user tokens | overflow at limit−20k; prune old tool outputs (keep 40k) | 0.85 ratio or 50k reserved | `auto_compact_threshold` | not specified (map fog) |
| Deferred tools | tool search default; ≤5 per search | client `tool_search` | no (full set per agent) | no (background MCP load) | not examined | server-side k=8 retrieval |
| Licence | docs only; Python SDK MIT, TS SDK proprietary | Apache-2.0 | MIT | Apache-2.0 | Apache-2.0 | n/a |

Cursor is closed source, so no Cursor code was used or cited.

## Clean code references

All repositories were read at the commit listed. The paths are relative to the repository root.

| Repo @ commit | Path / symbol | What it shows | Licence |
|---|---|---|---|
| [openai/codex](https://github.com/openai/codex) @ `4bad6d78e9` | `codex-rs/core/src/session/turn.rs` `run_turn`, `run_auto_compact`, `build_prompt` | turn loop, follow-up rule, mid-turn compaction, Stop-hook continuation | Apache-2.0 |
| openai/codex | `codex-rs/core/src/tools/parallel.rs` `handle_tool_call`, `failure_response`, `aborted_response` | non-fatal tool errors become `success:false` outputs | Apache-2.0 |
| openai/codex | `codex-rs/core/src/tools/orchestrator.rs` | approval + sandbox, `ToolError::Rejected` | Apache-2.0 |
| openai/codex | `codex-rs/core/src/tools/handlers/plan_spec.rs` `create_update_plan_tool`; `plan.rs` | plan tool schema and handler | Apache-2.0 |
| openai/codex | `codex-rs/core/src/compact.rs`; `codex-rs/prompts/templates/compact/prompt.md`, `summary_prefix.md` | compaction history rebuild and prompts | Apache-2.0 |
| openai/codex | `codex-rs/core/src/tools/handlers/tool_search.rs`; `codex-rs/tools/src/tool_search.rs` | client-side deferred-tool search | Apache-2.0 |
| openai/codex | `codex-rs/core/src/context/user_instructions.rs` | AGENTS.md as user-role fragment | Apache-2.0 |
| [anomalyco/opencode](https://github.com/anomalyco/opencode) (was sst/opencode) @ `055d95bb7e` | `packages/opencode/src/session/prompt.ts` `runLoop` | loop exit rule, max-steps, per-step system assembly | MIT |
| anomalyco/opencode | `packages/opencode/src/session/processor.ts` (`failToolCall`, `DOOM_LOOP_THRESHOLD`) | error parts, doom-loop permission ask, stop/compact/continue | MIT |
| anomalyco/opencode | `packages/opencode/src/session/message-v2.ts` | error and interrupted tool parts sent to the model | MIT |
| anomalyco/opencode | `packages/opencode/src/session/compaction.ts`, `overflow.ts` | prune + summary compaction, overflow threshold | MIT |
| anomalyco/opencode | `packages/core/src/session/runner/max-steps.ts` | last-step "tools disabled, summarize" prompt | MIT |
| anomalyco/opencode | `packages/opencode/src/tool/invalid.ts`, `tool/todo.ts`, `session/system.ts`, `session/reminders.ts` | invalid-args tool, todo tool, provider prompts, mode reminders | MIT |
| [MoonshotAI/kimi-cli](https://github.com/MoonshotAI/kimi-cli) @ `9ab1286b8f` | `src/kimi_cli/soul/kimisoul.py` `_agent_loop`, `_step` | Python step loop, steer queue, rejection/repeat stop reasons | Apache-2.0 |
| MoonshotAI/kimi-cli | `src/kimi_cli/soul/toolset.py` (`_build_repeat_reminder`) | graduated repeat reminders, error results | Apache-2.0 |
| MoonshotAI/kimi-cli | `src/kimi_cli/soul/approval.py` `ApprovalResult.rejection_error` | rejection messages with/without user feedback | Apache-2.0 |
| MoonshotAI/kimi-cli | `src/kimi_cli/soul/compaction.py` `should_auto_compact`; `src/kimi_cli/config.py` | compaction trigger, loop defaults | Apache-2.0 |
| MoonshotAI/kimi-cli | `src/kimi_cli/tools/todo/__init__.py` `SetTodoList` | todo tool | Apache-2.0 |
| [aaif-goose/goose](https://github.com/aaif-goose/goose) @ `3bd8520029` | `crates/goose/src/agents/agent.rs` | max turns, auto-compact threshold | Apache-2.0 |
| [Aider-AI/aider](https://github.com/Aider-AI/aider) @ `5dc9490bb3` | `aider/coders/base_coder.py` (`max_reflections`) | reflection retry (non-tool-call loop, contrast only) | Apache-2.0 |
| [anthropics/claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python) @ `b6e9d12fe1` | `src/claude_agent_sdk/types.py` `PermissionResultDeny`; `_internal/query.py` `_handle_control_request`, `interrupt`; `_internal/transport/subprocess_cli.py` | deny/interrupt wire format; SDK spawns the CLI (no loop inside) | MIT |
| anthropics/claude-agent-sdk-typescript | (repo) | examples/changelog only; `LICENSE.md` = "All rights reserved", Commercial Terms | proprietary, do not port |

## Implications for Odysseus

These are facts for later decisions, not decisions.

- Every reference loop treats a tool error, denial or abort as a tool result and lets the model choose the next step. None strips tools after 2 failed rounds. Their hard stops are caps (turns/steps/budget, defaults none or 1000), a pure user rejection, repeated identical calls (OpenCode at 3, which asks the user; Kimi at 12) or an interrupt.
- When a cap is hit, there are two documented patterns. OpenCode sends a final tools-off "summarize progress and remaining work" prompt (`MAX_STEPS_PROMPT`). The Claude SDK returns an error result without synthesizing an answer.
- External veto of "done" exists in both Claude Code (`Stop` hook block + reason) and Codex (Stop hooks). That is the same seam `with_completion_gate` occupies.
- Claude Code keeps instruction files and reminders out of the system prompt, in user-role `<system-reminder>` blocks, and re-injects them after compaction. Codex does the same for AGENTS.md.
- Claude Code now enables the todo tools by default only on older models.
- Kimi CLI is a Python, Apache-2.0 loop that maps closely onto a provider-agnostic re-implementation. Its `kimisoul.py` and `toolset.py` are the most directly portable references.
- The TypeScript Agent SDK repo is not open-source code. Only the docs and the MIT Python SDK wrapper are usable from Anthropic.

[MoonshotAI/kimi-cli]: https://github.com/MoonshotAI/kimi-cli
[agent-loop]: https://code.claude.com/docs/en/agent-sdk/agent-loop
[todo-tracking]: https://code.claude.com/docs/en/agent-sdk/todo-tracking
[permissions]: https://code.claude.com/docs/en/agent-sdk/permissions
[user-input]: https://code.claude.com/docs/en/agent-sdk/user-input
[modifying-system-prompts]: https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts
[tool-search]: https://code.claude.com/docs/en/agent-sdk/tool-search
[how-claude-code-works]: https://code.claude.com/docs/en/how-claude-code-works
[context-window]: https://code.claude.com/docs/en/context-window
[interactive-mode]: https://code.claude.com/docs/en/interactive-mode
[hooks]: https://code.claude.com/docs/en/hooks
[handle-tool-calls]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
[tool-search-tool]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
