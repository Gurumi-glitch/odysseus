# Z.AI Provider Shape

Last updated: integrate/model-support@718209b9 | 2026-11-19

## Scope

Canonical provider ID `zai`; Z.AI/GLM OpenAI-compatible endpoints including
coding-plan variants; curated discovery in `routes/model_routes.py` and prior
vision/reasoning fixes such as #664.

## Shape And Observations

Use general identity-only inventory mapping. Some working coding-plan models may be
absent from `/models`, so pinned/curated IDs are availability compatibility,
not capability truth. GLM reasoning controls have appeared as structured
objects or serving-template kwargs depending on direct cloud versus local
engine (#3031). Keep those scopes separate.

## Documented API Facts

Sources, all fetched as markdown and accessed 2026-11-19 (no live API call;
anything marked unconfirmed has not been seen on the wire). Endpoints:
standard `https://api.z.ai/api/paas/v4` (examples in the thinking guide),
Coding Plan `https://api.z.ai/api/coding/paas/v4`
(<https://docs.z.ai/devpack/faq.md>, <https://docs.z.ai/guides/llm/glm-5.3.md>).

- **`tool_choice` is auto-only.** The schema enum is `[auto]`: "the default
  value is auto, and only auto is supported". The request schema lists no
  `parallel_tool_calls` and no `stream_options`. Whether a value or field outside
  the schema is rejected or ignored is unconfirmed (#145).
  (<https://docs.z.ai/api-reference/llm/chat-completion.md>)
- **Two thinking controls.** `thinking: {type: "enabled"|"disabled",
  clear_thinking: bool}` (GLM-4.6+, default `enabled`) and a top-level
  `reasoning_effort` that is "only supported by GLM-5.2 and above" (default
  `max`). On the standard API GLM-5.3 accepts only `max|high|low` ("any other
  input will result in an error") and its migration note says `disabled` makes
  the request fail; GLM-5.2 accepts `max, xhigh, high, medium, low, minimal,
  none` (`none`/`minimal` skip thinking). The same pages disagree on whether
  GLM-4.7 can switch thinking per request. Behaviour of `reasoning_effort` on
  GLM-4.7/4.6 is unconfirmed (#145).
  (<https://docs.z.ai/guides/capabilities/thinking.md>,
  <https://docs.z.ai/guides/capabilities/thinking-mode.md>,
  <https://docs.z.ai/guides/llm/glm-5.3.md>)
- **Coding Plan maps efforts instead of rejecting them.** On
  `/api/coding/paas/v4` GLM-5.3 (and GLM-5.3-Flash) maps `none|minimal|low` to
  `low`, `medium|high` to `high` and `xhigh|max` to `max`, so an effort of
  `none` does not turn thinking off. On the standard API GLM-5.3 errors on
  `none`, so one request means different things on the two endpoints. GLM-5.2
  behaves the same on both.
  (<https://docs.z.ai/guides/capabilities/thinking.md>; mainland mirror
  <https://docs.bigmodel.cn/cn/guide/capabilities/thinking.md>)
- **`clear_thinking` defaults differ per endpoint.** The standard API defaults to
  `true` (earlier turns' `reasoning_content` is ignored server-side); preserved
  thinking is "enabled by default on the Coding Plan endpoint and disabled by
  default on the standard API endpoint". With `clear_thinking: false` the full,
  unmodified `reasoning_content`, in the original order, must be sent back.
  (<https://docs.z.ai/api-reference/llm/chat-completion.md> for the `true`
  default, <https://docs.z.ai/guides/capabilities/thinking-mode.md> for the
  per-endpoint defaults)
- **Extra `finish_reason` values.** Beyond `stop`, `tool_calls` and `length`:
  `sensitive`, `network_error`, `model_context_window_exceeded`. When a
  streamed request fails during inference the error codes are not returned;
  the failure is reported through `finish_reason` only. HTTP errors use
  `{"error": {"code": "1214", "message": ...}}` with 1214 = "Parameter `${field}`
  is invalid" (`code` is a string in the example, an integer in the Error
  schema). The exact text of a thinking-parameter 400 is unconfirmed (#145).
  (<https://docs.z.ai/api-reference/llm/chat-completion.md>,
  <https://docs.z.ai/api-reference/api-code.md>)

## Current Runtime Behaviour

Code, not documentation: `src/chat_agent/providers/chat_completions.py` and
`src/chat_agent/loop.py` (new runtime only; the legacy loop is unchanged).
There is no `zai` provider key: `_detect_provider` returns `openai` for Z.ai
hosts, and the host (`z.ai`, `bigmodel.cn`) is matched in the payload builder.

- Thinking Off sends `thinking: {type: "disabled"}` and no `reasoning_effort`,
  on both endpoints; graded levels send `reasoning_effort` unchanged. A
  pre-output 4xx/5xx naming `effort` or the standalone word `thinking` makes the
  runtime learn the level as unsupported for that endpoint and model, emit
  `thinking_level_refused` and re-send once without a level. No per-model rule
  is coded for GLM-5.3 (forced thinking, `low|high|max`); it is learned from the
  rejection. Whether Z.ai's 400 text matches is unconfirmed (#145).
- The new runtime sends `tool_choice: "auto"` only with tools. The legacy loop
  sends `tool_choice: "none"` when it has none (`src/llm_core.py:3097-3098`),
  a value outside Z.ai's documented enum; its effect is unconfirmed (#145).
- `finish_reason` `sensitive` ends the turn as refused, and `network_error`,
  `model_context_window_exceeded` as an error, with a saved error naming the
  model and the raw value; the round's tool calls are not run.
- `clear_thinking` is never sent; `reasoning_content` is replayed unchanged to
  the same provider and model, so the endpoint default applies.

## Fallback And Current Gaps

Exact `*.z.ai` or explicit endpoint kind preserves Z.AI identity. Never infer
vision/reasoning/tool support from `glm` in a name. A rich official model-card
reader and direct-versus-coding-plan schema split are still missing.
