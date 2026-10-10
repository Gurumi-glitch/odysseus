# Provider native tool calling: what Odysseus wires today

Research for wayfinder ticket #23 (map #20). Code citations are against `origin/devGuru`
(`e8887aaf`, 2026-10-10). Provider facts come from official docs, fetched 2026-10-10.

## Answer summary

- **Provider detection.** `_detect_provider` has dedicated branches for `anthropic` and
  `openrouter`. xAI and OpenAI fall through to the generic `"openai"` Chat Completions path,
  because there is no `x.ai` branch (`src/llm_core.py:1069-1109`).
- **Native versus text tool calls.** All four targets resolve to native tool calls. Their hosts
  are in `_API_HOSTS` (`src/agent_loop.py:8283-8292`), and `_agent_route_tool_mode` treats a
  listed host or a `claude`/`gpt-5` model keyword as native (`src/agent_loop.py:8446-8470`).
  When a round returns no native calls, a text parser runs as a fallback. For native models,
  that parser skips fenced code blocks (`src/agent_loop.py:15406-`).
- **ChatGPT-subscription exception.** This route strips `tools`, `tool_choice` and
  `parallel_tool_calls` and uses a text protocol only (`src/llm_core.py:1505-1530`).
- **`tool_choice` and `parallel_tool_calls`.** The provider layer never sends
  `parallel_tool_calls`. It sends `tool_choice` only as `"none"`, and only when no tools are
  attached (`src/llm_core.py:2992-2996`). The Anthropic builder never sends `tool_choice`
  (`src/llm_core.py:1822-1907`). Every provider therefore runs on its default, which is
  "auto, parallel allowed".
- **Errors in tool results.** Errors are never marked structurally. The OpenAI dialect has no
  error flag at all, and Anthropic's `is_error` is never set. A failure is just text in the
  result, such as `**Error:** ...` or a non-zero `exit_code` (`src/tool_execution.py:2485-2532`;
  `src/llm_core.py:1829-1838`).
- **Anthropic parallel results.** Each tool result becomes its own `user` message
  (`src/llm_core.py:1829-1838`). Anthropic documents this as the wrong format, one that
  "teaches" Claude to avoid parallel calls.
- **Anthropic thinking.** Thinking is dropped. The stream parser handles only `text_delta` and
  `input_json_delta` (`src/llm_core.py:3298-3313`), so `thinking` and `signature` blocks are
  never replayed. Current Claude models (Opus 5.5, Fable 5.1) have adaptive thinking always on.
  Anthropic says thinking blocks are **required** within a tool-use turn, and the API can
  silently disable thinking when the history is incompatible.
- **OpenAI-dialect reasoning.** Reasoning travels only as `reasoning_content`, and only on the
  newest assistant turn. DeepSeek is the exception and keeps every turn
  (`src/agent_loop.py:16003-16018`). It is not echoed at all when external tool schemas are in
  play (`src/agent_loop.py:36347-`). OpenRouter `reasoning_details` is dropped by the sanitizer's
  allow-list (`src/llm_core.py:1986`).
- **OpenAI endpoint.** Odysseus uses only Chat Completions for OpenAI (`src/llm_core.py:726-735`).
  OpenAI says Chat Completions does not support function calling with GPT-6 Astra or
  GPT-6.1 Sol, and reasoning items carry across tool calls only on the Responses API.
- **Context windows are understated.**
  - `grok-4*` resolves to 131,072 (`src/model_context.py:177`), but xAI lists 500k–1M.
  - Claude 5.x and GPT-6 match no entry in the known-model table, so they fall back to the
    128,000 default (`src/model_context.py:106`). Anthropic and OpenAI list about 1M.
  - Anthropic's `/v1/models` field `max_input_tokens` is not among the fields that are read
    (`src/model_context.py:332-338`).
- **Bottom line.** Native tool transport works for all four targets. What a long, unscripted
  loop lacks today is correct parallel-result framing for Claude, structured error signaling,
  reasoning continuity across rounds, OpenAI Responses support, and accurate context limits.

## Matrix

"As wired" means the Odysseus code path. "Provider" means the official API capability.

| | xAI Grok (`api.x.ai`) | Anthropic (`api.anthropic.com`) | OpenAI / generic compatible | OpenRouter (`openrouter.ai`) |
|---|---|---|---|---|
| **Wire dialect, as wired** | Generic `"openai"` Chat Completions; no xAI branch (`llm_core.py:1069-1109`, `:2976-3022`) | Native Messages; OpenAI-shaped history converted in `_build_anthropic_payload` (`llm_core.py:1822-1907`) | Chat Completions only (`llm_core.py:726-735`, `:2976-3022`) | Chat Completions plus `reasoning` toggle (`llm_core.py:1239-1246`) |
| **Native or text tools, as wired** | Native: host in `_API_HOSTS` (`agent_loop.py:8289`) | Native: host plus `claude` keyword (`agent_loop.py:8284`, `:8446`) | Native for `api.openai.com` / `gpt-*`; other hosts depend on `supports_tools` or keyword (`agent_loop.py:8446-8470`) | Native: host in `_API_HOSTS` (`agent_loop.py:8285`) |
| **Text fallback** | Yes, when a round has no native calls; fenced blocks are ignored for native models (`agent_loop.py:15406-`) | Same | Same | Same |
| **Parallel calls, provider** | On by default; `parallel_tool_calls: false` disables ([xAI][xai-fc]) | On by default; `tool_choice.disable_parallel_tool_use` disables ([parallel][a-par]) | Parallel on supported models from GPT-5; `parallel_tool_calls: false` gives zero or one call ([OpenAI FC][o-fc]) | "default is true for most models"; `parallel_tool_calls: false` ([OR tools][or-tc]) |
| **Parallel calls, as wired** | Parsed and executed: index-keyed accumulator (`llm_core.py:3745-3800`); never disabled | Parsed per `content_block` index (`llm_core.py:3288-3313`, `:3356-3366`); **results returned as separate user messages** (`llm_core.py:1829-1838`) | Same as xAI | Same as xAI |
| **`tool_choice`, provider** | `auto` / `required` / `none` / named function ([xAI][xai-fc]) | `auto` / `any` / `tool` / `none`; Opus 5.5, Sonnet 5.5 and Fable 5.1 reject `any` and `tool` with a 400 ([define tools][a-def]) | `auto` / `required` / `none` / forced / `allowed_tools` ([OpenAI FC][o-fc]) | `auto` / `none` / named function ([OR tools][or-tc]) |
| **`tool_choice`, as wired** | Only `"none"`, and only when `tools` is empty (`llm_core.py:2992-2996`) | Never sent | Same as xAI | Same as xAI |
| **Reasoning across rounds, provider** | Chat Completions "has no field for the ciphertext"; encrypted reasoning round-trips only through the Responses API ([xAI reasoning][xai-r]) | Thinking blocks with `signature` must be passed back unmodified within a tool-use turn ([thinking][a-th]) | Responses: "any reasoning items returned … with tool calls must also be passed back"; Chat Completions not covered ([OpenAI FC][o-fc], [reasoning][o-r]) | Pass `reasoning_details` back unmodified and in order ([OR reasoning][or-r]) |
| **Reasoning across rounds, as wired** | `reasoning_content` deltas are captured (`llm_core.py:3655`); newest assistant turn only (`agent_loop.py:16003-16018`) | **Dropped**: thinking and signature deltas are not parsed (`llm_core.py:3298-3313`); no `thinking` parameter is sent | Same as xAI. GPT-5 with tools is forced to `reasoning_effort:"none"` (`llm_core.py:1433-1441`, `:1376-1379`) | `reasoning` is read as `reasoning_content`; `reasoning_details` is stripped (`llm_core.py:1986`) |
| **Tool-result format, provider** | `function_call_output` + `call_id` (Responses); no error flag; the example returns `{"error": ...}` JSON ([xAI][xai-fc]) | `tool_result{tool_use_id, content, is_error}`; all results in one user message, placed first ([handle][a-handle]) | `role: tool` + `tool_call_id`; Responses uses `function_call_output` + `call_id`; no error flag ([OpenAI FC][o-fc]) | `role: tool` + `tool_call_id`; no error flag ([OR tools][or-tc]) |
| **Tool-result format, as wired** | `{"role":"tool","tool_call_id","content":<markdown text>}` (`agent_loop.py:16053-16058`) | One `user` message per result with `tool_result{tool_use_id, content}`; **no `is_error`** (`llm_core.py:1829-1838`) | Same as xAI | Same as xAI |
| **Error marking, as wired** | Text only: `**Error:** …`, `Error: …`, `exit_code` (`tool_execution.py:2497-2532`) | Same; `is_error` is never set | Same | Same |
| **Context limit, provider** | grok-4.5 to 4.7: 500k; grok-4.3 and 4.20: 1M ([xAI models][xai-m]) | Current lineup: 1M; max output 128K ([models][a-models]) | GPT-6 Astra, 6.1 Sol and 6 Luna: 1.05M; max output 128K ([OpenAI models][o-m]) | Per model: `context_length` from `/api/v1/models` (`specs/model-providers/openrouter.md`) |
| **Context limit, as wired** | `grok-4` → 131,072 by substring match (`model_context.py:177`, `:305-319`) | `claude-*-4*` / `claude-3*` → 200k; Claude 5.x matches nothing → 128,000 default (`model_context.py:106`, `:114-124`); `max_input_tokens` is not read (`model_context.py:332-338`) | `gpt-5` → 400k; `gpt-4.1` → 1,047,576; GPT-6 matches nothing → 128,000 (`model_context.py:127-130`) | `context_length` is read from the catalog (`model_context.py:332`) |

## Gaps a new runtime must fill in the provider layer

1. **Structured tool-result envelope.**
   - Today a tool result reaches the provider as a markdown string (`tool_execution.py:2485`), with
     success or failure visible only in its text.
   - Anthropic supports `is_error: true`, and recommends it for every call that is not executed
     ([handle][a-handle], [parallel][a-par]).
   - The OpenAI-dialect providers define no error flag. There, the convention is error content in
     the output string ([xAI][xai-fc], [OR tools][or-tc]).
   - The runtime needs a canonical result object, such as `{call_id, ok, content, error}`, that the
     adapter maps to `is_error` for Anthropic and to a consistent error prefix or JSON for OpenAI.
2. **Anthropic parallel-result batching.**
   - `_build_anthropic_payload` emits one `user` message per `role:"tool"` message
     (`llm_core.py:1829-1838`).
   - Anthropic says all `tool_result` blocks must go in a single user message, placed before any
     text. Separate messages "teach" Claude to stop making parallel calls ([parallel][a-par]).
3. **Anthropic thinking round-trip.**
   - The parser drops `thinking`, `redacted_thinking` and `signature_delta` (`llm_core.py:3298-3313`),
     and the builder rebuilds assistant turns from only text and `tool_use`
     (`llm_core.py:1840-1858`).
   - Opus 5.5 and Fable 5.1 have adaptive thinking always on ([models][a-models]). Anthropic says
     passing thinking blocks back within a tool-use turn is required, and the API may strip or
     disable thinking when the history is incompatible ([thinking][a-th]).
   - The runtime needs a provider-opaque "assistant turn blocks" slot that survives across rounds
     and is replayed verbatim.
4. **OpenRouter `reasoning_details` passthrough.**
   - The sanitizer allow-list keeps only `reasoning_content` (`llm_core.py:1986`).
   - OpenRouter's documented round-trip field is `reasoning_details`, which must stay unmodified and
     in order ([OR reasoning][or-r]). This matters most for Claude served through OpenRouter.
5. **OpenAI Responses API adapter.**
   - The only Responses path is the ChatGPT-subscription route, and it forbids tools
     (`llm_core.py:1505-1530`).
   - GPT-6 Astra and 6.1 Sol have no Chat Completions function calling ([OpenAI reasoning][o-r]).
     Reasoning items must accompany function-call outputs on Responses ([OpenAI FC][o-fc]).
   - xAI's encrypted reasoning also round-trips only through Responses ([xAI reasoning][xai-r]).
6. **Tool-control knobs.**
   - The runtime needs `tool_choice` (`auto`/`required`/`none`/named) and a parallel on/off switch,
     mapped per dialect. OpenAI dialect uses `parallel_tool_calls`; Anthropic uses
     `tool_choice.disable_parallel_tool_use` ([parallel][a-par]).
   - Anthropic's restriction must be modelled per model: no forced tool use on Opus 5.5, Sonnet 5.5
     or Fable 5.1, and forcing skips thinking ([define tools][a-def]).
7. **Context-window truth.**
   - Read the provider's own catalog fields: Anthropic `max_input_tokens` ([models][a-models]) and
     OpenRouter `context_length`.
   - Refresh the stale table, where `grok-4` is 131,072 and Claude 5.x and GPT-6 fall to the 128k
     default (`model_context.py:106-179`). Without this, compaction and budget logic trim a
     1M-token model at about 128k.
8. **First-class xAI adapter.**
   - xAI is classified as generic `"openai"` (`llm_core.py:1069-1109`).
   - `reasoning_effort` is supported on grok-4.5, 4.6 and 4.7, and reasoning cannot be disabled
     ([xAI reasoning][xai-r]). No Odysseus code maps `thinking_mode` or effort for xAI, except an
     OpenRouter-only guard for `grok-4.5` (`llm_core.py:1250-1254`).
9. **Tool count per request.**
   - xAI caps a request at 350 tools ([xAI][xai-fc]). The map's baseline of k=8 retrieved schemas
     per turn is far under this limit.
   - Moving to a larger or static tool set is a provider-side option, but Anthropic's prompt-cache
     breakpoint sits on the last tool (`llm_core.py:1896-1906`), so the tool list should stay
     stable across rounds to keep cache hits.

[xai-fc]: https://docs.x.ai/docs/guides/function-calling
[xai-r]: https://docs.x.ai/docs/guides/reasoning
[xai-m]: https://docs.x.ai/docs/models
[a-def]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools
[a-handle]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
[a-par]: https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use
[a-th]: https://platform.claude.com/docs/en/build-with-claude/thinking
[a-models]: https://platform.claude.com/docs/en/about-claude/models/overview
[o-fc]: https://developers.openai.com/api/docs/guides/function-calling
[o-r]: https://developers.openai.com/api/docs/guides/reasoning
[o-m]: https://developers.openai.com/api/docs/models
[or-tc]: https://openrouter.ai/docs/guides/features/tool-calling
[or-r]: https://openrouter.ai/docs/use-cases/reasoning-tokens
