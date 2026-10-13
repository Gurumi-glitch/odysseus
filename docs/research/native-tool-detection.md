# How servers report native tool-call support

Research for #75 (map #74). Written 2026-10-13 against `origin/devGuru` at `3505020a`. Companion to `docs/research/thinking-levels.md` (#76, branch `research/thinking-levels`) and the #78 probe.

Fact markers:
- **[doc]**: from the vendor's official docs (URL given, fetched 2026-10-13).
- **[code]**: from source. Odysseus as `file:line`; upstream servers as `repo/path` on `main`/`master`, 2026-10-13.
- **[probed]**: observed on this machine on 2026-10-13 with read-only GETs (LM Studio `:1234`, llama-server PriSM fork `:1235`).
- **[inferred]**: my reading, not stated by a primary source.

## TL;DR

1. **Local servers mostly do report tool support, but each in a different place, and only from the chat template.** llama.cpp: `GET /props` → `chat_template_caps.supports_tools` / `supports_tool_calls`. LM Studio: `GET /api/v1/models` → `capabilities.trained_for_tool_use`. Ollama: `POST /api/show` → `capabilities` contains `"tools"`. vLLM and SGLang: no endpoint at all; tool support is a launch flag (`--enable-auto-tool-choice` + `--tool-call-parser`) that is not exposed.
2. **"Reports support" and "accepts a `tools` request" are different things.** LM Studio accepts `tools` for every model (a "default tool use" prompt for models without native support). llama.cpp's docs say it accepts tools for every template (generic fallback). Ollama returns HTTP 400 `does not support tools`. vLLM returns HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`. So only Ollama and vLLM fail loudly; LM Studio and llama.cpp degrade silently.
3. **The OpenAI-compatible `/v1/models` never carries a tool flag on any local server.** llama.cpp hardcodes `capabilities: ["completion"]` (or `"multimodal"`). LM Studio's `/v1/models` lists only loaded models and has no capability field.
4. **Cloud:** OpenRouter (`supported_parameters` contains `tools`), Mistral (`capabilities.function_calling`) and GitHub Copilot (`capabilities.supports.tool_calls`) report it per model. OpenAI, Anthropic, Gemini and xAI model lists do not; for those, tool calling is a property of the API itself.
5. **Odysseus today decides with a hand-written host list plus a model-name keyword list** (`src/agent_loop.py:8443-8466`), overridable by the per-endpoint `ModelEndpoint.supports_tools` column. That one value is also the new runtime's routing gate (`src/chat_agent/routing.py:78`). Nothing reads a server capability signal at chat time. The capability readers that parse `/props`, LM Studio and Ollama payloads exist but have **no production caller**.
6. **`supports_tools` is per endpoint, not per model.** One llama-server router, LM Studio or Ollama endpoint serves many models with different templates, so a detected per-model value does not fit the existing column. `model_tool_modes` (per-model JSON on the same row) shows the existing per-model shape.

## Per-server table

"No-tools template" = what the signal says, and what a `tools` request does, when the model's chat template has no tool support.

| Server | Signal (endpoint → field) | How it's computed | Reliability | No-tools template | False positive / negative risks |
|---|---|---|---|---|---|
| **llama.cpp upstream** (`llama-server`, `--jinja` on by default) | `GET /props` → `chat_template_caps.supports_tools`, `.supports_tool_calls`, `.supports_parallel_tool_calls`. In router mode: `GET /props?model=<id>` (400 without it). [code `tools/server/server-context.cpp` ~L5157; doc `tools/server/README.md` "router mode"] | Static template analysis: render the template with a dummy tool and dummy `tool_calls`; a cap is cleared when the template never reads `tools[0].function.name` / `messages[1].tool_calls`. All three default to `true`. [code `common/jinja/caps.cpp` ~L300-410, `caps.h` L11-14] When the GGUF has a separate `tool_use` template, caps come from that one. [code `common/chat.cpp` `common_chat_templates_get_caps`] | Good for "does the template know about tools". It says nothing about whether the model was trained for them. | Caps `false`. A `tools` request is still accepted: the docs say a generic tool-call format is used when the template isn't recognized. Current master routes unknown templates through a "differential autoparser"; a template without tool support only logs a warning when `supports_tool_calls && !supports_tools`. [doc `docs/function-calling.md`; code `common/chat.cpp` ~L1351-1360] What a fully tool-less template does on master is **not verified**. [inferred] | **FP:** defaults are `true`, so a template the analyzer can't execute keeps `true` for the first check. [code] **FN:** tool support added through `--chat-template-file` is reflected (caps are computed from the active template), so a FN needs a template that handles tools in a way the analyzer doesn't trace. [inferred] `--no-jinja`: `tools` → error `tools param requires --jinja flag`. [code `server-common.cpp` L1286] |
| **llama.cpp `/v1/models`** | `models[].capabilities` | Hardcoded `["completion"]` or `["completion","multimodal"]`. [code `server-context.cpp` ~L5107] | **Not a tool signal.** | Same as with tools. | Always "no tools" → false negatives for every model. |
| **llama.cpp PriSM fork** (`:1235`, build `b10772-e8fd209c9`) | Same `/props` shape. [probed] | Same fields present: `supports_tools`, `supports_tool_calls`, `supports_parallel_tool_calls` all `true` for Bonsai; `/v1/models` `capabilities: ["completion"]`. [probed + #78] | Matches upstream shape. #78 confirmed real tool calls (including 2 parallel) work. [probed] | Not tested (only one model loaded). | Same as upstream. |
| **LM Studio** | Native `GET /api/v1/models` → `models[].capabilities.trained_for_tool_use` (bool). Lists all downloaded models, loaded or not. "Absent for embedding and decision models." [doc https://lmstudio.ai/docs/developer/rest/list; probed] | From the model's metadata; the docs only say "Whether the model was trained for tool/function calling". [doc] A bug report shows it can disagree with the model's `model.yaml` (`trainedForToolUse: true` but API `false`). [https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2388] | Good per model, keyed by `key`. | `trained_for_tool_use: false`, but **a `tools` request still works**: "All models that don't have native tool use support will have default tool use support" (custom system prompt + default call format; "Results will vary by model"). [doc https://lmstudio.ai/docs/developer/openai-compat/tools] | **FN** possible per the bug report above. No hard failure ever, so a wrong value is never surfaced by the server. Probed: 8 LLMs, 7 `true`, `jev-style-qwen3.5-2b-decision` `false` (Odysseus's name list would say yes because of `qwen3`). [probed] `/v1/models` returned `{"data":[]}` with nothing loaded. [probed] |
| **Ollama** | `POST /api/show` → `capabilities: [...]` contains `"tools"`. [doc https://docs.ollama.com/api-reference/show-model-details lists the field; values from code] | Go templates: the template uses the `.Tools` variable. Jinja chat templates: **substring match** — the template text contains `"tools"` or `"tool_call"`. [code `ollama/server/images.go` `chatTemplateHasToolSupport`, `goTemplateCapabilities`] | Good for Go templates; the Jinja substring test is crude. | No `"tools"`. A chat request with tools → HTTP 400 `"<model>" does not support tools` (`CheckCapabilities`). [code `server/images.go` ~L565-600, `server/routes.go` ~L3157] | **FP:** a Jinja template that merely mentions the word "tools" counts as tool-capable. [code] |
| **vLLM** | **None.** `/v1/models` has no capability field; docs list no endpoint. [doc https://docs.vllm.ai/en/latest/features/tool_calling/] | Launch flags: `--enable-auto-tool-choice` ("mandatory") + `--tool-call-parser`. [doc] | n/a | Without the flags, a request with `tools` defaults to `tool_choice: auto` and gets HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`. [third-party reports, e.g. https://vlaicu.io/posts/dgx-vllm-playbook/; Odysseus already matches this text, `src/llm_core.py:4006-4022`] With the flags, response parsing depends on the parser, not the template. [code `vllm/entrypoints/openai/chat_completion/serving.py` ~L1027] | Only detectable by attempting a call (or, for Cookbook-launched servers, from the launch command). |
| **SGLang** | None found; Odysseus has only a placeholder vendor id (`src/model_capability_readers/__init__.py:39-46`). [code; not researched further] | — | — | — | — |

## Cloud providers

| Provider | Model list exposes tool support? | Source |
|---|---|---|
| OpenRouter | **Yes.** `GET /api/v1/models` → `supported_parameters` contains `tools` / `tool_choice`; filter `?supported_parameters=tools`. [doc https://openrouter.ai/docs/guides/overview/models] Probed: 459 models, 391 list `tools`; the filtered query returned 369. [probed] | per model |
| Mistral | **Yes.** `GET /v1/models` → `capabilities.function_calling`. [doc https://docs.mistral.ai/api/endpoint/models] | per model |
| GitHub Copilot | **Yes.** `capabilities.supports.tool_calls`, already read by `src/copilot.py:199-203`. [code] | per model |
| Anthropic | **No client-tool flag.** `capabilities` covers `server_tools`, `structured_outputs`, `effort`, `thinking`, etc. [doc https://platform.claude.com/docs/en/api/models/retrieve] Tool use is part of the Messages API for all models. [inferred] | — |
| OpenAI | **No.** Model object: `id, created, object, owned_by, shutdown_date`. [doc https://developers.openai.com/api/reference/resources/models/methods/list] | — |
| Gemini | **No.** Model resource has `supportedGenerationMethods`, `thinking`, limits; nothing on function calling. [doc https://ai.google.dev/api/models] | — |
| xAI | Not found in the API reference. [doc https://docs.x.ai/docs/api-reference, negative result] | — |
| NVIDIA, Groq, Cerebras, Moonshot, Featherless | Not checked. | — |

How Odysseus decides for cloud today [code `src/agent_loop.py:8280-8289`, `8443-8466`]: native tools when the URL contains a host in `_API_HOSTS` (OpenAI, Anthropic, OpenRouter, Groq, Mistral, Cohere, DeepSeek, Together, Fireworks, Perplexity, xAI, ollama.com, Venice, Kimi, Copilot) **or** the model name matches the keyword list. Gemini (`generativelanguage.googleapis.com`), NVIDIA, Cerebras, Moonshot, Featherless and OpenCode hosts are **not** in `_API_HOSTS`, so they get native tools only through the model-name list (`gemini`, `kimi`, …). ChatGPT subscription is forced `False` (`routes/chatgpt_subscription_routes.py:194`, `routes/model_routes.py:2308-2309`, `:2896-2897`); Copilot sets it from its model list (`routes/copilot_routes.py:58`, `:80`).

## Odysseus inventory

Index: codebase-memory `home-gurumi-Projects-odysseus` (rebuilt 2026-10-13), `check_index_coverage` clean for every file below; static/website/specs via grep.

### Every reader and writer of `supports_tools`

| Kind | Location | What it does |
|---|---|---|
| Column | `core/database.py:585` | `ModelEndpoint.supports_tools`, nullable bool, **per endpoint**. Comment: NULL = unknown → keyword heuristic. Migration `:1288-1304`. |
| Read (routing) | `src/agent_loop.py:8383-8466` `_agent_route_tool_mode` | Looks up the endpoint row(s) by base URL (+ owner, + header match), reads `endpoint.supports_tools` (`:8437`). `True` → native; `False` / `deepseek-r1` / `gpt-oss` / Ollama native or `:11434/v1` → text; `None` → host list or model keyword list. Odysseus fine-tune names short-circuit to native (`:8395-8396`). |
| Read (callers of above) | `src/agent_loop.py:24137` (legacy route state) and `src/chat_agent/routing.py:78` `endpoint_has_adapter` → `choose_runtime` → `routes/chat_routes.py:3017` | Graph trace: these are the only two callers. On the new runtime, `False` from this function means **the turn goes to legacy**. |
| Read (API out) | `routes/model_routes.py:2117` (`GET /api/model-endpoints`), `:2904` (PATCH response), `companion/routes.py:162` | Exposes the value. |
| Write (create) | `routes/model_routes.py:2146`, `:2305-2331` (`POST /api/model-endpoints` form field `"true"/"false"/""`) | User or Cookbook supplied. |
| Write (patch) | `routes/model_routes.py:2844-2846` (`PATCH /api/model-endpoints/{id}`) | Manual override; documented for users in `website/setup.md:345-359`. No UI control found in `static/`. |
| Write (forced) | `routes/model_routes.py:2308-2309`, `:2896-2897`; `routes/chatgpt_subscription_routes.py:194` | ChatGPT subscription → `False`. |
| Write (Copilot) | `routes/copilot_routes.py:58`, `:80` | `True` if any model reports `tool_calls`, or the model fetch failed. |
| Write (Cookbook) | `routes/cookbook_routes.py:1888`, `:1920-1921`, `:1971` | `True` if the serve command has `--enable-auto-tool-choice`, else unchanged/None. |
| Write (Cookbook UI) | `static/js/cookbookRunning.js:4372`, `:4399` | Sends `supports_tools=true` on `--enable-auto-tool-choice` **or a JS model-name regex** (`deepseek|gpt-[45o]|claude|gemini|qwen3|…`). |
| Read (capability readers) | `src/model_capability_readers/llamacpp.py:152-164`, `:206` | Maps `/props` caps → `CAP_TOOL_CALL`. No production caller (see below). |
| Fixture | `scripts/compare_schema_thinking.mjs:186` | Test harness inserts `supports_tools=1`. |
| Docs | `specs/agent-tools.md:139`, `specs/llm-models.md:97,117`, `specs/model-providers/ollama.md:37` | Describe the opt-in/override behaviour. |
| Tests | `tests/test_tool_support_heuristic.py` (copies the keyword list), `test_model_routes.py`, `test_copilot_routes.py`, `test_chatgpt_*`, `test_companion_readonly.py`, `test_featherless_provider.py`, `test_model_capability_readers.py` | — |

Related per-model state on the same row: `model_tool_modes` (`core/database.py:588`, JSON `{model_id: none|compact|full}`), read by `_configured_model_tool_surface` (`src/agent_loop.py:8470`) and by the new runtime at `src/chat_agent/loop.py:351`; listed per model by `GET /model-endpoints/{id}/models` (`routes/model_routes.py:2513-2523`).

Reactive fallback (legacy only): `_native_tool_transport_rejected` (`src/llm_core.py:4006-4022`) matches error text (`enable-auto-tool-choice`, `tools are not supported`, …); on a match, `_candidate_capability_recovery` (`src/agent_loop.py:26241-26253`) disables native tools for that `(url, model)` for 15 min (`:8299-8319`, checked at `:24153`). It does **not** match Ollama's `does not support tools`. `src/chat_agent/` never reads this cache.

### Where servers are already probed

| Path | Location | What it fetches | Keeps capability data? |
|---|---|---|---|
| Model refresh (background, picker) | `routes/model_routes.py:1576-1656` `_refresh_caches_bg`, triggered by `GET /api/models?background|refresh` (`:1746-1805`) | `_probe_endpoint` (`:1012`) → `/v1/models` (or Anthropic/Google/ChatGPT lists) | **No.** Returns ids only; writes `cached_models`. |
| Manual refresh | `routes/model_routes.py:2473-2525` `GET /model-endpoints/{id}/models?refresh=1` | same `_probe_endpoint` | No. |
| Endpoint create | `routes/model_routes.py:2295` | same | No. |
| Per-model probe | `routes/model_routes.py:2423-2470` `GET /model-endpoints/{id}/probe` | a tiny chat call per model (`_probe_single_model`), sets `hidden_models` | No tools in the probe. |
| `probe-local` | `routes/model_routes.py:1813-1890` | `_ping_endpoint` reachability only, 8 s cache | No. |
| Discovery fingerprint | `src/model_discovery.py:153-185` `_fingerprint_provider` | LM Studio `/api/v1/models`, llama.cpp `/props` | Only to name the server; payload discarded. |
| Context length | `src/model_context.py:430-460` | llama.cpp `/slots`, then `/props` `n_ctx` | Only `n_ctx`. |
| Vision (chat time) | `src/chat_helpers.py:110-180` `_probe_lmstudio_models`, `lmstudio_supports_vision`, `model_supports_vision` | LM Studio `/api/v1/models`, short-TTL cache, local hosts only | Reads `capabilities.vision` per model live; falls back to names. **A working precedent for a live per-model capability read.** |
| Capability readers | `src/model_capability_readers/*` | parse `/props`, LM Studio, Ollama `/api/show`, OpenRouter payloads | **Unused:** `records_from_payload` has no non-test caller (graph trace, inbound = 0). Only `base.detect_vendor` is used, for image URLs (`src/ai_interaction.py:1291`, `:1492`, `routes/gallery/gallery_routes.py:1323`). This corrects #76's note that the readers feed gallery pages. |

## Implications for the map (facts only)

1. A detect-only design has a per-server signal for llama.cpp (`/props`), LM Studio (`/api/v1/models`), Ollama (`/api/show`), OpenRouter, Mistral and Copilot. vLLM/SGLang expose nothing; for them the only sources are the Cookbook launch command (already used) or the server's 400 on a tools request.
2. OpenAI, Anthropic, Gemini and xAI expose no per-model flag; today they are decided by `_API_HOSTS` and the model-name list. Gemini, NVIDIA, Cerebras, Moonshot, Featherless and OpenCode depend on the model-name list alone.
3. The acceptance model (Bonsai) reports `supports_tools: true` on `/props` and its id on llama-server is a `.gguf` path; on LM Studio it reports `trained_for_tool_use: true`. Neither id matches the keyword list, so today both route as "no native tools" → legacy, unless `supports_tools=true` is set by hand. [code + probed]
4. `ModelEndpoint.supports_tools` is a per-endpoint, user-writable override (PATCH API, documented in `website/setup.md`). The map's "no user setting, no override" conflicts with that API and with `static/js/cookbookRunning.js:4372`'s name regex.
5. llama.cpp's and LM Studio's flags describe the template/metadata, and both servers still accept `tools` when the flag is false; Ollama and vLLM reject with 400. So a flag of `false` means "no native format", not "request will fail", on the first two.
6. `src/chat_agent/` has no reactive fallback: the legacy rejection cache and error-text matcher are not reached by the new runtime.
7. Existing hooks where a detected value could be computed: the background model refresh (`_refresh_caches_bg`), which already hits each endpoint but keeps only ids; or a live, cached per-turn read like `model_supports_vision`. Both `_agent_route_tool_mode` and `endpoint_has_adapter` take `(endpoint_url, model, owner, headers)`, so a per-model value is reachable from both.

## Sources

- llama.cpp (master, 2026-10-13): `common/jinja/caps.cpp`, `common/jinja/caps.h`, `common/chat.cpp`, `tools/server/server-context.cpp`, `tools/server/server-common.cpp`, `tools/server/README.md`, `docs/function-calling.md` — https://github.com/ggml-org/llama.cpp
- LM Studio: https://lmstudio.ai/docs/developer/rest/list, https://lmstudio.ai/docs/developer/openai-compat/tools, https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2388
- Ollama: https://docs.ollama.com/api-reference/show-model-details, https://docs.ollama.com/capabilities/tool-calling, `server/images.go`, `server/routes.go` — https://github.com/ollama/ollama (main, 2026-10-13)
- vLLM: https://docs.vllm.ai/en/latest/features/tool_calling/, `vllm/entrypoints/openai/chat_completion/serving.py` (main)
- OpenRouter: https://openrouter.ai/docs/guides/overview/models
- Mistral: https://docs.mistral.ai/api/endpoint/models
- Anthropic: https://platform.claude.com/docs/en/api/models/retrieve
- OpenAI: https://developers.openai.com/api/reference/resources/models/methods/list
- Gemini: https://ai.google.dev/api/models
- xAI: https://docs.x.ai/docs/api-reference
