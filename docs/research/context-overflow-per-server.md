# Context overflow behaviour per server

Research for issue #85 (map #74). Tags: **[src]** verified from primary source (URL or file), **[inf]** inferred, **[unk]** not found in primary sources. Checked 2026-10-13 against each project's default branch.

## Summary table

| Server | Overflow default | Per-request control on OpenAI-style endpoint? | Readable policy? |
|---|---|---|---|
| LM Studio | Policy `stopAtLimit` / `truncateMiddle` / `rollingWindow`; default **[unk]** | No documented way. SDK-only (`contextOverflowPolicy`); `/api/v1/chat` has `context_length` but no policy | Not documented |
| llama.cpp `llama-server` | Prompt >= n_ctx: HTTP 400 `exceed_context_size`. Generation hits n_ctx: stop, `truncated=true` (shift off by default) | `n_keep`, `n_discard` per request; `--context-shift` is server-start only | Partly: `/props` shows `n_ctx`; shift flag not confirmed in `/props` |
| Ollama | `/api/chat`: truncate oldest messages (keeps system + latest). Generation: shift | `truncate`, `shift` on native `/api/chat`, `/api/generate` only; not on `/v1` | No |
| vLLM | Reject (400) when prompt > max_model_len | `truncate_prompt_tokens` + `truncation_side` as extra body fields | `max_model_len` via `/v1/models` **[inf]** |
| OpenAI | Chat Completions: 400 **[inf]**. Responses: `truncation` `disabled` (400) or `auto` | `truncation` on Responses API | n/a |
| Anthropic | Input > window: 400 "prompt is too long". Input + max_tokens > window: accepted on 4.5+; stops with `model_context_window_exceeded` | Server-side compaction / context editing (beta) | n/a |
| Gemini | **[unk]** (long-context doc is silent) | **[unk]** | n/a |

## LM Studio

- **[src]** Enum values and meaning: `stopAtLimit` stops, `stopReason` = `contextLengthReached`; `truncateMiddle` "Keep the system prompt and the first user message, truncate middle"; `rollingWindow` "Maintain a rolling window and truncate past messages". https://github.com/lmstudio-ai/docs/blob/main/2_typescript/7_api-reference/llm-prediction-config-input.md (line 28-31)
- **[src]** The policy applies "Once the context length is exceeded"; it is part of the SDK prediction config. `.../2_typescript/7_api-reference/llm-load-model-config.md` lines 19-22.
- **[src]** The OpenAI-compat docs list supported params (model, top_p, top_k, messages, temperature, max_tokens, stream, stop, presence/frequency_penalty, logit_bias, repeat_penalty, seed) and mention no overflow field. https://lmstudio.ai/docs/developer/openai-compat/chat-completions
- **[src]** `POST /api/v1/chat` has `context_length` ("Number of tokens to consider as context") and no overflow policy field; overflow behaviour is not documented there. https://lmstudio.ai/docs/developer/rest/chat
- **[src]** An undocumented `"lmstudio": {"context_overflow_policy": ...}` body field was tried on `/api/v0/chat/completions` in 0.3.13; a number fails schema validation, the string `rollingWindow` fails with "Invalid context overflow policy". Issue still open. https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/532
- **[unk]** Default policy: not in the docs repo (grepped all 172 .md files). Community posts say the app UI has a per-model setting; treat the default as unknown and read it from the app.
- Note: the local "LM Studio compatible" llama-server (port 1235) is llama.cpp behind the LM Studio protocol shape, so the llama.cpp rules apply there. `GET /props` on :1235 was attempted and the port was not reachable (curl exit 7), so nothing measured.

## llama.cpp (`llama-server`, forks inherit unless they diverge)

- **[src]** `--context-shift` / `--no-context-shift`: default **disabled**; server-start flag only (`common/arg.cpp:1742-1749`, README "default: disabled"). Auto-disabled for multimodal / unsupported contexts (`tools/server/server-context.cpp:1315-1329`).
- **[src]** Per-request fields `n_keep` (-1 = keep whole prompt, default from `--keep`, 0) and `n_discard` (0 = half of what is left after n_keep): `tools/server/server-schema.cpp:54-60`. Used only when shift is on: `server-context.cpp:3298-3315`. `n_keep` excludes BOS (README); code adds 1 when a BOS is added.
- **[src]** Prompt already >= slot n_ctx: request fails with error type `exceed_context_size` ("request (N tokens) exceeds the available context size (M tokens), try increasing it"), body includes `n_prompt_tokens` and `n_ctx`; no truncation. `server-context.cpp:3560-3583`, `server-task.cpp:1506-1510`. HTTP 400 mapping not read; **[inf]** from the error type name.
- **[src]** Generation reaches n_ctx with shift disabled: slot stops with `STOP_TYPE_LIMIT`, `truncated = true`, response flag "context size was exceeded during generation" (`server-context.cpp:2029-2033`, README `truncated`).
- **[src]** `/props` exposes `default_generation_settings.n_ctx` and `params.n_keep/n_discard` (README). Whether it reports the `ctx_shift` flag: **[unk]**; check a live `/props` before relying on it.
- **[inf]** The `n_keep` and `n_discard` schema fields are also accepted on `/v1/chat/completions` since the OpenAI route feeds the same task parser; not tested (no generation requests allowed).

## Ollama

- **[src]** `/api/chat` and `/api/generate` request types have `truncate` ("truncates the chat history messages if the rendered prompt exceeds the context length limit") and `shift` ("shifts the chat history when hitting the context length limit instead of erroring"). `api/types.go:114-120, 166-172`.
- **[src]** Both default to true when unset: `req.Truncate == nil || *req.Truncate`, `req.Shift == nil || *req.Shift`. `server/routes.go:642-649, 756-757`.
- **[src]** Truncation keeps system messages and the latest message, drops older ones. `server/prompt.go:21-22, 35-77`.
- **[src]** `/api/embed` `truncate=false` returns an error on overflow. docs/api.md.
- **[src]** Context length default depends on VRAM: <24 GiB 4k, 24-48 GiB 32k, >=48 GiB 256k; set via `OLLAMA_CONTEXT_LENGTH` or app slider. https://github.com/ollama/ollama/blob/main/docs/context-length.mdx
- **[src]** The OpenAI-compat doc says "The OpenAI API does not have a way of setting the context size"; use `PARAMETER num_ctx` in a Modelfile. `docs/api/openai-compatibility.mdx:384-393`. `/v1` does not pass `truncate`/`shift` (no mention in `middleware/openai.go`). `num_ctx` is settable per request only through native `options`.
- **[inf]** `ollama ps` shows allocated `CONTEXT`; `/api/show` shows model parameters including `num_ctx`.

## vLLM

- **[src]** Prompt longer than the limit is rejected with `VLLMValidationError`: "This model's maximum context length is N tokens. However, your prompt contains M input tokens, which exceeds this limit even without any output tokens." `vllm/renderers/params.py:381-400, 520-535`.
- **[src]** Chat requests accept `truncate_prompt_tokens` (int, -1 allowed) and `truncation_side` ("left" keeps the last N tokens, "right" keeps the first N). `vllm/entrypoints/openai/chat_completion/protocol.py:283-290`. These are extra-body fields on the OpenAI endpoint.
- **[src]** `max_tokens` is clamped downstream by `get_max_tokens()` (`params.py:214`), so prompt + max_tokens over the window does not fail by itself. Exact clamp not read; **[inf]**.
- **[inf]** Server limit (`--max-model-len`) is listed as `max_model_len` in `/v1/models`; not verified here.

## Cloud APIs

- **[src]** Anthropic: input alone over the window gives 400 `invalid_request_error` "prompt is too long". On Claude 4.5 and newer, input + `max_tokens` over the window is accepted and generation stops with `stop_reason: "model_context_window_exceeded"`. Compaction (beta) and context editing summarize or clear server-side. https://platform.claude.com/docs/en/build-with-claude/context-windows
- **[src]** OpenAI Responses API `truncation`: `auto` drops items from the beginning of the conversation; `disabled` (default) fails with 400. https://github.com/openai/openai-openapi/blob/master/openapi.yaml (lines ~47599-47603). `context_management` with type `compaction` and `compact_threshold` also exists (https://developers.openai.com/api/reference/resources/responses/methods/create).
- **[unk]** OpenAI Chat Completions overflow error and Gemini behaviour: not found in the pages read. Chat Completions has no truncation field in the schema section read **[inf]**.

## What Odysseus sends today

`src/chat_agent/providers/chat_completions.py:60-88` (`build_payload`): `model`, `messages`, `stream`, `stream_options.include_usage` (not for openrouter/groq), `tools`/`tool_choice`, `temperature`, `max_tokens` or `max_completion_tokens`, plus thinking-mode fields. No overflow-related field at all. `xai_responses.py` has no `truncation` field either.

Per-request field each server would need (all would be added in `build_payload`, keyed by provider):

| Server | Field to send | Where |
|---|---|---|
| LM Studio | none available on `/v1`; policy stays app-side | n/a |
| llama.cpp | `n_keep` (and `n_discard`) in body; shift itself needs a server flag | extra body field |
| Ollama | `/v1`: nothing; native `/api/chat` `truncate`, `shift` | needs native endpoint |
| vLLM | `truncate_prompt_tokens`, `truncation_side` | extra body field |
| OpenAI Responses | `truncation: "auto" \| "disabled"` | Responses only |
| Anthropic | compaction / context-editing params | Messages API only |

## Implication for "Auto"

Only llama.cpp, vLLM, Ollama (native API) and OpenAI Responses allow per-request choice. LM Studio and Ollama `/v1` cannot be set from the request, so "Auto" there means "whatever the server is configured to do". The only server-side policy value found readable from a client is llama.cpp's `n_ctx` via `/props`; reading the active policy generally needs a live probe per server (open).
