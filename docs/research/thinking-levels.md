# Thinking-level support and overrides per provider

Research for #76 (map #74). Written 2026-10-13 against `origin/devGuru` at `3505020a`.

Fact markers:
- **[doc]**: verified from the provider's official docs (URL given; fetched 2026-10-13).
- **[code]**: verified from source code (Odysseus file:line, or upstream server source at the commit date given).
- **[inferred]**: my reading. Not stated by a primary source.

APIs in this area change monthly. Where a doc page carries a date, it is quoted.

## TL;DR

1. **Every cloud API now has a graded effort control, but the level sets differ per model, not per provider.** Four providers expose the per-model level list through an API: Anthropic (`GET /v1/models` → `capabilities.effort.{low..max}`, `capabilities.thinking.types.disabled`), OpenRouter (`GET /api/v1/models` → `reasoning.supported_efforts`, `mandatory`), LM Studio (`GET /api/v1/models` → `capabilities.reasoning.allowed_options`) and Ollama (`/api/show` → `thinking.values`). OpenAI, xAI, Gemini, DeepSeek, Mistral and NVIDIA document levels only in their docs.
2. **"Off" does not exist everywhere.** xAI Grok 4.5–4.7, Gemini 2.5 Pro and Gemini 3.x, Claude Opus 5.5 and OpenRouter `mandatory` models all refuse to turn thinking off. Several OpenAI models reject `none`.
3. **llama.cpp (upstream, master 2026-10-13) lets a request override the server's thinking settings.** The request's `chat_template_kwargs` are merged over `--chat-template-kwargs`, `--reasoning on|off` and `--reasoning-effort`, and the request wins. Request-level `reasoning_effort` is supported too (`none` turns thinking off; any other value goes to the template). The one thing a request cannot do is lift a server `--reasoning-budget` back to unlimited: `-1` in the request means "use the server default". vLLM documents the same rule: request kwargs always win.
4. **llama.cpp `/props` does not say whether a template supports thinking.** `chat_template_caps` has `supports_reasoning_effort` and `supports_preserve_reasoning`, but no thinking on/off flag, and the server's default kwargs are not exposed. `POST /apply-template` renders the prompt through the same request parser, so rendering with `enable_thinking` true and false (or with different `reasoning_effort` values) and comparing the two prompts detects whether a control has any effect.
5. **Odysseus today:** thinking is a binary `on`/`off` keyed on model-name lists. On the new runtime, only `_apply_local_qwen_thinking_mode` and `_apply_hosted_thinking_mode` run (OpenRouter and Kimi only). The DeepSeek, Mistral and Ollama branches, the OpenAI tools scrub and ChatGPT `reasoning_effort` are legacy-only. xAI on the new runtime sends no reasoning control at all.
6. **Two existing bugs matter for the spec.** (a) Any model not on the `supports_user_thinking_toggle` list is forced to `thinking_mode="off"`, so on OpenRouter every such model gets `reasoning: {effort: "none"}` on both runtimes. That can 400 on mandatory-reasoning models. (b) Legacy sends `think: false` to Ollama's `/v1` endpoint, but Ollama's `/v1` request struct has no `think` field, so the value is silently ignored. Only `reasoning_effort: "none"` or `reasoning.effort` works there.

## Per-provider table

"Off" means the value that turns thinking off. "Detect" means how a client learns which levels a model accepts.

| Provider / server | Parameter (Chat Completions unless noted) | Accepted values | Detect per-model support | Unsupported value | Can the server refuse a client override? |
|---|---|---|---|---|---|
| **xAI** (Responses API, used by the new runtime) | `reasoning: {effort}`. `reasoning_effort` is read only when `reasoning` is unset. [doc] | `low`, `medium`, `high` (default), `xhigh` (grok-4.6+). On grok-4.20-multi-agent the value sets agent count, not depth. [doc] | Docs only. No models-endpoint field found. [doc/inferred] | `xhigh` on grok-4.5 is "treated as `high`" (clamped). [doc] Behaviour for `none` is undocumented. | "Reasoning cannot be disabled" on grok-4.5/4.6/4.7, so Off is unsupported. [doc] |
| **OpenAI** | `reasoning_effort` (Chat), `reasoning.effort` (Responses). [doc] | Model-dependent subset of `none, minimal, low, medium, high, xhigh, max`. Most current models default to `medium`. [doc] | Docs only (per-model pages). `/v1/models` has no level list. [doc] | HTTP 400 for `none` on a model without it (for example GPT-6 Astra, GPT-6.1 Sol). [doc] Odysseus code says GPT-5.x rejects `reasoning_effort` together with tools on Chat Completions (`src/llm_core.py:1376-1378`). [code] | N/A |
| **Anthropic** (Messages API; no new-runtime adapter) | `thinking: {type: adaptive \| enabled+budget_tokens \| disabled \| between_tools}` plus `output_config: {effort}`. [doc] | effort: `low, medium, high, xhigh, max`, as a model-dependent subset. Default is `high`, or `medium` on Opus 5.5 and Haiku 5.5. [doc] | **API**: `GET /v1/models/{id}` → `capabilities.effort.<level>.supported` and `capabilities.thinking.types.{adaptive,enabled,disabled}.supported`. `disabled` is "False exactly when a request that sends it gets a 400". [doc] | 400: `type: enabled` on 4.7+; `disabled` on Opus 5.5 at any level; `disabled` at `xhigh`/`max` on Opus 5 and Haiku 5.5. [doc] | N/A. Opus 5.5 cannot turn thinking off. [doc] |
| **Gemini** (OpenAI-compat endpoint; Odysseus classifies it as `openai`) | `reasoning_effort`. Alternative: `extra_body.google.thinking_config` (`thinking_level`/`thinking_budget`). The two can't be combined. [doc] | `minimal, low, medium, high`, mapped per family (for example 3.1 Pro: minimal→low). `none` disables thinking on 2.5 models only. [doc, page updated 2026-09-02] | Docs only (the thinking page has a per-model table, updated 2026-10-09). [doc] | "Reasoning cannot be turned off for Gemini 2.5 Pro or 3 models". The error is undocumented. [doc] | N/A |
| **OpenRouter** | `reasoning: {effort \| max_tokens, enabled, exclude}`. [doc] | `max, xhigh, high, medium, low, minimal, none`. [doc] | **API**: `GET /api/v1/models` → `reasoning.supported_efforts` (`null` = all levels), `default_effort`, `default_enabled`, `mandatory`, `supports_max_tokens`. `reasoning` is omitted for non-reasoning models. [doc] | "map your requested effort to the nearest supported level" (clamp; stated in the Gemini section). Non-reasoning models omit the field. `mandatory: true` → "effort `none` is rejected". [doc] | Mandatory models refuse Off. [doc] |
| **NVIDIA NIM** | Per model: system prompt `detailed thinking on/off` (Nemotron Super/Ultra v1), `/no_think` (Super v1.5), `chat_template_kwargs.enable_thinking` (Nemotron 3, MiMo), `reasoning_effort` low/medium/high (gpt-oss only). [doc, updated 2026-08-21] | See left. | Docs only. [doc] | Undocumented. [doc] | Undocumented. |
| **DeepSeek** | `thinking: {type: enabled \| disabled}` plus `reasoning_effort`. [doc] | Effort `low, high, max`. The server maps minimal/low→low, medium/high/xhigh→high, max/ultra→max. Thinking is on by default with effort `high`. [doc] | Docs only. [doc] | Clamped by the mapping above. [doc] | N/A |
| **Mistral** | `reasoning_effort`. [doc] | `high`, `none` (zai-glm-5-3: `low, high, max`, no `none`). Supported on mistral-small-latest, mistral-medium-3-5, mistral-large-4-0. Magistral is deprecated. [doc] | Docs only. [doc] | Undocumented. A third-party SDK says non-reasoning Mistral models reject the parameter. [inferred] | N/A |
| **llama.cpp** (upstream master, 2026-10-13) | `chat_template_kwargs` (for example `enable_thinking`), `reasoning_effort`, `reasoning_budget_tokens`/`thinking_budget_tokens`, `reasoning_format`. [code `server-common.cpp` `oaicompat_chat_params_parse`] | `reasoning_effort: "none"` → thinking off. Any other non-empty string is passed to the template as `reasoning_effort`/`reasoning_strength`. [code] `enable_thinking` must be a boolean; a string returns 400 "invalid type". [code] | `/props.chat_template_caps.supports_reasoning_effort` (the template reads the variable). [code `common/jinja/caps.cpp`] There is no thinking-support flag in `/props`. Use `POST /apply-template` with and without the kwarg and compare the prompts. [code + inferred] | No validation: an unknown effort string goes to the template, which may ignore it. [code] | **Mostly no.** Request kwargs merge over `--chat-template-kwargs`, `--reasoning`, `--reasoning-effort`, and the request wins. [code] Exceptions: `--no-jinja` (no kwargs apply); a request budget of `-1` falls back to `--reasoning-budget`; `--skip-chat-parsing` puts reasoning in content. [code] Server defaults are not in `/props`. [code] |
| **LM Studio** | Native `POST /api/v1/chat`: `reasoning`. OpenAI-compat `/v1/chat/completions`: not documented. [doc] | `off, low, medium, high, on`. [doc] | **API**: `GET /api/v1/models` → `capabilities.reasoning.{allowed_options, default}`. Absent when the model exposes no reasoning config. [doc] | "Will error if the model ... does not support the reasoning setting". [doc] | Per request; no server lock documented. |
| **Ollama** | Native: `think` (bool or level name). `/v1`: `reasoning_effort` or `reasoning.effort` only. [doc + code `openai/openai.go`, main 2026-10-13] | Native: values from `/api/show` → `thinking.values`. `/v1`: `none`→false, `minimal`→`low`, `xhigh`/`ultra`→`max`. Without model metadata, values outside `minimal..max/none` return an error. [code] | **API**: `/api/show` → `thinking.values` and `thinking.default`. `values: [false]` = no thinking. [doc] | Native: an unsupported name → model default. [doc] `/v1` on a boolean-only model: any effort → `true`. [code] | Per request. |
| **vLLM** (docs dated 2026-10-01) | `chat_template_kwargs` (`enable_thinking` or `thinking`, depending on template), `reasoning_effort`, `thinking_token_budget`. [doc] | `reasoning_effort` low/medium/high → `enable_thinking=true`; `none` → false. An explicit kwarg beats `reasoning_effort`. [doc] | No capability API. Only the parser name in the supported-models table. [doc] | A kwarg the template doesn't declare is "harmlessly filtered out". [doc] | No: "Request-level `chat_template_kwargs` always take priority over server defaults" (`--default-chat-template-kwargs`). [doc] |

Sources:
- xAI: https://docs.x.ai/docs/guides/reasoning, https://docs.x.ai/docs/api-reference
- OpenAI: https://developers.openai.com/api/docs/guides/reasoning
- Anthropic: https://platform.claude.com/docs/en/build-with-claude/effort, https://platform.claude.com/docs/en/build-with-claude/extended-thinking, https://platform.claude.com/docs/en/api/models/retrieve
- Gemini: https://ai.google.dev/gemini-api/docs/openai, https://ai.google.dev/gemini-api/docs/thinking
- OpenRouter: https://openrouter.ai/docs/use-cases/reasoning-tokens, https://openrouter.ai/docs/api/api-reference/models/get-models
- NVIDIA NIM: https://docs.nvidia.com/nim/large-language-models/1.15.0/reasoning-model.html
- DeepSeek: https://api-docs.deepseek.com/guides/thinking_mode
- Mistral: https://docs.mistral.ai/studio/conversations/reasoning
- llama.cpp: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md, plus `tools/server/server-common.cpp`, `tools/server/server-context.cpp`, `common/arg.cpp`, `common/chat.cpp`, `common/jinja/caps.cpp` (master 2026-10-13)
- LM Studio: https://lmstudio.ai/docs/developer/rest/list, https://lmstudio.ai/docs/developer/rest/chat
- Ollama: https://docs.ollama.com/capabilities/thinking, https://github.com/ollama/ollama/blob/main/openai/openai.go, https://github.com/ollama/ollama/issues/14820
- vLLM: https://docs.vllm.ai/en/latest/features/reasoning_outputs.html

**PriSM fork caveat:** the acceptance model runs on the llama.cpp PriSM fork. Everything in the llama.cpp row is from upstream master and has **not** been checked against the fork. A live `/props` probe on port 1235 was not possible because the server was down. [inferred]

## Mapping Auto / Off / Low / Medium / High / XHigh / Max

General rule [inferred]: **Auto = send nothing.** Every provider above has a model or server default, so Auto works everywhere. A level is "supported" only when a detection source lists it, or, for docs-only providers, when a hand-kept per-provider table lists it. Everything else is greyed out.

| Level | xAI | OpenAI | Anthropic* | Gemini compat | OpenRouter | DeepSeek | Mistral | llama.cpp / vLLM | LM Studio (native) | Ollama /v1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Auto | omit | omit | omit | omit | omit | omit | omit | omit | omit | omit |
| Off | ✗ | `none` (if model has it) | `thinking.type=disabled` (if `types.disabled`) | `none` (2.5 Flash only) | `effort:none` (✗ if `mandatory`) | `thinking.type=disabled` | `none` | `reasoning_effort:none` or `enable_thinking:false` | `off` (if allowed) | `none` |
| Low | `low` | `low` | `low` | `low` | `low` | `low` | ✗ | `low` † | `low` (if allowed) | `low` |
| Medium | `medium` | `medium` | `medium` | `medium` | `medium` | → high (server clamp) | ✗ | `medium` † | `medium` | `medium` |
| High | `high` | `high` | `high` | `high` | `high` | `high` | `high` | `high` † | `high` | `high` |
| XHigh | `xhigh` (4.6+; 4.5 clamps) | `xhigh` (model-dep.) | `xhigh` (if listed) | ✗ | `xhigh` | → high (clamp) | ✗ | `xhigh` † | ✗ | → `max` (server alias) |
| Max | ✗ | `max` (model-dep.) | `max` (if listed) | ✗ | `max` | `max` | ✗ | `max` † | ✗ | `max` |

\* Anthropic has no new-runtime adapter (`src/chat_agent/routing.py:19`), so on the new runtime the control is greyed out for Anthropic anyway.
† llama.cpp passes any string to the template. Only templates with `supports_reasoning_effort: true` read it, and the template decides which names mean anything. Without that cap, the only reliable local controls are Auto, and Off via `enable_thinking:false` if `/apply-template` shows the kwarg changes the prompt. [code + inferred]

Mapping notes [inferred]:
- **Clamping providers** (xAI `xhigh` on 4.5, DeepSeek, OpenRouter, Ollama aliases) accept a level they don't really have. Showing it as supported would mislead. Grey out any level the provider is known to clamp, and show the effective level instead.
- **Off and the effort scale are separate axes** on Anthropic (`thinking.type` vs `output_config.effort`) and DeepSeek (`thinking.type` vs `reasoning_effort`). On OpenAI, OpenRouter, Ollama, vLLM, llama.cpp and Mistral, Off is just the `none` effort.
- **Gemini via `reasoning_effort` has no xhigh/max.** Native `thinking_level` exists, but Odysseus talks to Gemini only through the OpenAI-compat URL.

## Can a server refuse an override, and how to detect it

- **llama.cpp upstream:** no refusal path for `enable_thinking` or `reasoning_effort`. The request merges last. [code `server-common.cpp` ~L1402-1427: `inputs.chat_template_kwargs = opt.chat_template_kwargs; for item in request: inputs.chat_template_kwargs[k] = v`, then `enable_thinking` and `reasoning_effort` are parsed from the merged map/body]
  - `--reasoning off` only sets the server default (`params.default_template_kwargs["enable_thinking"]="false"`, `common/arg.cpp` ~L3710-3725), and a request kwarg overrides it.
  - Budget: the request's `reasoning_budget_tokens`/`thinking_budget_tokens` win, except `-1`, which falls back to `--reasoning-budget`. [code ~L1441-1445]
  - `--no-jinja`: thinking is never enabled (`server-context.cpp` ~L1604-1605). [code]
  - Detection: `/props` does **not** expose the server's default kwargs or budget. `default_generation_settings` and `chat_template_caps` are exposed; `default_template_kwargs` is not. [code `server-context.cpp` ~L5140-5165] A client can detect that a control has no effect with `POST /apply-template` (same parser, `server-context.cpp` ~L5603) by rendering the same message with two settings and comparing `prompt`. After the turn, the presence or absence of `reasoning_content` in the stream is the ground truth. [inferred]
- **vLLM:** no refusal. Request kwargs always win. [doc]
- **Cloud providers:** the refusal is a model property (mandatory reasoning). It is detected from metadata on Anthropic and OpenRouter, from docs elsewhere, or after the fact from a 400 response.

## Odysseus code-path inventory

`thinking_mode` today is `"on"`, `"off"` or `""`/None (`_normalize_thinking_mode`, `src/llm_core.py:1199-1201`). There is no level scale except the ChatGPT-subscription `effort:<level>` session value. [code]

### Where the mode comes from (route)

| Step | Location | Reached by new runtime? |
|---|---|---|
| Request form `thinking_mode` (on/off only) and `reasoning_effort` | `routes/chat_routes.py:2478-2481` | yes |
| Session fallback: `session_mode = sess.thinking_mode or "off"`, used when the request sent none | `routes/chat_routes.py:2756-2763` | yes |
| Force `"off"` if `not supports_user_thinking_toggle(model)` (model-name list, `src/model_profiles.py:50-62`) | `routes/chat_routes.py:2764-2766`, again at `2895-2896` | yes |
| `effort:<level>` session value → `reasoning_effort`, validated against ChatGPT model metadata | `routes/chat_routes.py:2767-2770`, `src/chatgpt_subscription.py:226-241` | parsed, but **not passed** to `ChatAgentTurn` |
| Session setter: accepts `""/on/off/effort:x`, forces `off` for unlisted models | `routes/history/history_routes.py:1050-1115` | n/a (settings) |
| DB default `'off'` | `core/database.py:199`, `:1049` | n/a |
| `ChatAgentTurn(thinking_mode=...)` | `routes/chat_routes.py:3063`; field `src/chat_agent/loop.py:146` | yes |

Consequence [code]: `""` (Auto) cannot survive the route. `sess.thinking_mode or "off"` turns it into `"off"`, and any model off the name list is forced to `"off"`. So "model default" is reachable only when the request explicitly sends nothing *and* the session value is not on/off, which in practice never happens.

### Where the mode is applied (payload)

| Path | Location | What it does | New runtime? |
|---|---|---|---|
| `_apply_local_qwen_thinking_mode` | `src/llm_core.py:1204-1228` | `chat_template_kwargs.enable_thinking = (mode=="on")`, only if `is_local_endpoint(url)` **and** the model name matches `_THINKING_MODEL_PATTERNS` (`:1734-1745`) or the Odysseus Qwen router | **yes**, via `src/chat_agent/providers/chat_completions.py:86` |
| `_apply_hosted_thinking_mode` | `src/llm_core.py:1231-1250` | OpenRouter: on → `reasoning:{enabled:true}`, off → `reasoning:{effort:"none"}` (drops it for grok-4.5); Moonshot/Kimi: `thinking:{type}` | **yes**, `chat_completions.py:87` |
| DeepSeek V4 defaults | `src/llm_core.py:1444-1463` (`_DEEPSEEK_REASONING_EFFORT` `:1727-1731`) | `thinking:{type}` plus `reasoning_effort` high/none, only for model names matching `deepseek-v4` | no |
| Mistral branch | `src/llm_core.py:2344-2345`, `:2734-2735`, `:3001-3002` | `reasoning_effort = ODYSSEUS_MISTRAL_REASONING_EFFORT` (default `high`) when the name is in the list; ignores `thinking_mode` | no |
| Ollama `/v1` `think: False` | `src/llm_core.py:2728-2733`, `:3006-3011` | sends top-level `think:false` for listed thinking models; ignores `thinking_mode` | no |
| OpenAI GPT-5.x tools scrub | `src/llm_core.py:1432-1440`, called at `:3018` | forces `reasoning_effort:"none"` on api.openai.com with tools | no |
| ChatGPT subscription `reasoning.effort` | `src/llm_core.py:1564-1565` | only path that sends a graded level | no (no adapter, `routing.py:19`) |
| Anthropic builder | `src/llm_core.py:1822` | sends no thinking or effort at all | no (no adapter) |
| Ollama native builder | `src/llm_core.py:834` | sends no `think` | no (no adapter) |
| xAI Responses | `src/chat_agent/providers/xai_responses.py` (`loop.py:195-199` passes no `thinking_mode`) | sends no `reasoning` → server default `high` | new runtime only |
| Legacy agent route policy `_thinking_mode_for_route` | `src/agent_loop.py:186-215` (used at `:22134`, `:24547`) | Qwen3.5/3.8-router-only policy; `ODYSSEUS_QWEN_ROUTE_THINKING` | no |
| Clean-agent progressive thinking | `src/clean_agent_preview.py:1402`, `:5880` | fine-tuned Odysseus model only | no |
| Fixed `thinking_mode='off'` callers (email, background jobs, auth, agent_loop internals) | `src/email_reply_stream.py:29`, `src/background_tool_jobs.py:120`, `routes/email/email_routes.py:6473`, `src/agent_loop.py:27741/31978/37312`, `src/email_task_intent.py:176` | internal calls | no |

So the new runtime reaches exactly: the route resolution, `_apply_local_qwen_thinking_mode`, and `_apply_hosted_thinking_mode`. [code]

### Capability plumbing that exists but is unused [code]

- `src/model_capabilities.py:183-210` defines `REASONING_CONTROL_*` mechanisms (template kwarg, native bool, structured object, budget, effort) and values (on/off/auto). A grep finds no consumer outside the module.
- `src/model_capability_readers/`: `llamacpp.py:152-164` reads `/props` `chat_template_caps` for tools only, not `supports_reasoning_effort`. `lmstudio.py:77` reads `capabilities.reasoning` as a bool and drops `allowed_options`. `openrouter.py:40-42` maps `supported_parameters` → `CAP_REASONING` but not `reasoning.supported_efforts`/`mandatory`. `ollama.py:30` maps the `thinking` capability token. These readers feed the gallery/model pages (`src/ai_interaction.py:1291`, `routes/gallery/gallery_routes.py:1323`), not the chat payload.

## Things that change the plan

1. **OpenRouter Off-by-default bug [code + doc]:** any OpenRouter model not on `supports_user_thinking_toggle`'s list is forced `off` and sent `reasoning:{effort:"none"}`. The only exception is grok-4.5. OpenRouter says mandatory-reasoning models reject `none`, so these requests can fail with 400, on legacy and new runtime alike. The thinking-level spec replaces this with Auto = omit.
2. **Ollama `/v1` `think:false` is a no-op [code]:** Ollama's `ChatCompletionRequest` (`openai/openai.go`, main 2026-10-13) has no `think` field, so legacy's suppression never applied. Use `reasoning_effort:"none"`.
3. **DeepSeek name regex is stale [doc + inferred]:** the DeepSeek docs' examples use `deepseek-flash`. `_apply_deepseek_v4_reasoning_defaults` matches only `deepseek-v4`, so it may not fire for current model ids. (Legacy-only, so it doesn't matter for the new runtime.)
4. **llama.cpp overrides are cheap and safe upstream:** a request can set `reasoning_effort` (including `none`) and `chat_template_kwargs` and will win over the launch flags. "The server refuses overrides" is not a real upstream state, apart from `--no-jinja` and the `-1` budget fallback. Whether the PriSM fork matches is unverified. Probe the fork's `/props` and `/apply-template` when it is up.
5. **Detection sources by quality:** API metadata (Anthropic, OpenRouter, LM Studio native, Ollama native) > template probe (llama.cpp `/apply-template` + `supports_reasoning_effort`) > a hand-kept docs table (OpenAI, xAI, Gemini, DeepSeek, Mistral, NVIDIA, vLLM). The map's "no model-name keyword lists" note collides with the last group: those providers expose nothing at runtime. Either accept a per-provider docs table keyed on provider and model family, or show only Auto plus the levels that can be probed. That is a decision for the grilling ticket.
