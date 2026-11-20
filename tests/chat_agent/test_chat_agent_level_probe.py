"""llama.cpp thinking-level probe and hand-back on a refused level (ticket #97, spec #89).

The probe runs through the capabilities module's interface with the recorded
Bonsai /apply-template answers (PriSM llama-server, 2026-10-13); the hand-back
through a full new-runtime turn on the chat route with the network mocked.
"""

import json
from pathlib import Path

import httpx
import pytest

from src.chat_agent import capabilities
from tests.chat_agent.test_chat_agent_capabilities import (
    BONSAI,
    BONSAI_PROPS,
    LLAMA,
    _fresh_caches,  # noqa: F401  (autouse: clean capability caches)
    serve_probes,
)
# The fallback harness: one backup candidate, so a fallback would show.
from tests.chat_agent.test_chat_agent_fallback import harness as _fallback_harness  # noqa: F401
from tests.chat_agent.test_chat_agent_thinking_level import api  # noqa: F401  (the settings/capabilities app)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "chat_agent"
# POST /apply-template {"messages": [user "hi"], "reasoning_effort": <level>}
# per level ("auto" = no reasoning_effort): {"status", "body"} as the server answered.
BONSAI_RENDERS = json.loads((FIXTURES / "llamacpp_apply_template_bonsai.json").read_text())

GRADED = ("low", "medium", "high", "xhigh", "max")
S, U, N = "supported", "unsupported", "unknown"


def _support(off, *graded):
    return {"off": off, **dict(zip(GRADED, graded))}


def _renders(answers, seen=None):
    """An /apply-template handler answering by the request's reasoning_effort."""

    def handler(request):
        effort = json.loads(request.content).get("reasoning_effort", "auto")
        if seen is not None:
            seen.append(effort)
        answer = answers[effort]
        return httpx.Response(answer["status"], json=answer["body"])

    return handler


def _prompt(text):
    return {"status": 200, "body": {"prompt": text}}


# --- Detection --------------------------------------------------------------

@pytest.mark.asyncio
async def test_bonsai_renders_grey_the_levels_its_template_refuses(monkeypatch):
    rendered = []
    serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS, "POST /apply-template": _renders(BONSAI_RENDERS, rendered)})

    levels = await capabilities.thinking_levels(LLAMA, BONSAI)

    # xhigh renders exactly like Auto (it is the template default) and stays selectable.
    assert levels.support == _support(S, S, S, U, S, U)
    assert levels.binary is False
    assert await capabilities.native_tools(LLAMA, BONSAI) == capabilities.YES
    # Auto and each level rendered once; the 60s cache answers the second ask.
    assert sorted(rendered) == sorted(["auto", "none", *GRADED])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("none_render", "support"),
    [
        # Only Off changes the prompt: the template ignores effort.
        ("<think>\n\n</think>", _support(S, U, U, U, U, U)),
        # Nothing changes the prompt: no level has an effect.
        ("same", _support(U, U, U, U, U, U)),
    ],
    ids=["only-off-has-effect", "nothing-has-effect"],
)
async def test_graded_levels_are_greyed_when_every_one_renders_like_auto(monkeypatch, none_render, support):
    answers = {level: _prompt("same") for level in ("auto", *GRADED)}
    answers["none"] = _prompt(none_render)
    serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS, "POST /apply-template": _renders(answers)})

    assert (await capabilities.thinking_levels(LLAMA, BONSAI)).support == support


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "apply_template",
    [
        404,  # a llama-server without the route
        _renders({**BONSAI_RENDERS, "auto": {"status": 500, "body": {"error": "template"}}}),
        _renders({**BONSAI_RENDERS, "low": {"status": 200, "body": {"no": "prompt"}}}),
        httpx.ConnectError("down"),
    ],
    ids=["no-route", "auto-refused", "no-prompt", "unreachable"],
)
async def test_a_failed_level_probe_reports_nothing_and_keeps_the_tools_verdict(monkeypatch, apply_template):
    serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS, "POST /apply-template": apply_template})

    assert (await capabilities.thinking_levels(LLAMA, BONSAI)).support == _support(N, N, N, N, N, N)
    assert await capabilities.native_tools(LLAMA, BONSAI) == capabilities.YES


@pytest.mark.asyncio
async def test_renders_are_redone_after_the_ttl_and_after_a_refresh(monkeypatch):
    rendered = []
    serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS, "POST /apply-template": _renders(BONSAI_RENDERS, rendered)})
    clock = [1000.0]
    monkeypatch.setattr(capabilities, "_now", lambda: clock[0])

    await capabilities.thinking_levels(LLAMA, BONSAI)
    clock[0] += 59
    await capabilities.thinking_levels(LLAMA, BONSAI)
    assert len(rendered) == 7

    clock[0] += 2
    await capabilities.thinking_levels(LLAMA, BONSAI)
    assert len(rendered) == 14

    capabilities.forget_endpoint(LLAMA)
    await capabilities.thinking_levels(LLAMA, BONSAI)
    assert len(rendered) == 21


@pytest.mark.asyncio
async def test_a_learned_refusal_outranks_the_servers_report_until_a_manual_refresh(monkeypatch):
    serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS, "POST /apply-template": _renders(BONSAI_RENDERS)})
    clock = [1000.0]
    monkeypatch.setattr(capabilities, "_now", lambda: clock[0])

    capabilities.learn_unsupported_level(LLAMA, BONSAI, "low")
    clock[0] += 3600  # the server report expires and is fetched again

    assert (await capabilities.thinking_levels(LLAMA, BONSAI)).support["low"] == U

    capabilities.forget_endpoint(LLAMA)
    assert (await capabilities.thinking_levels(LLAMA, BONSAI)).support["low"] == U

    capabilities.forget_learned_levels(LLAMA)
    assert (await capabilities.thinking_levels(LLAMA, BONSAI)).support["low"] == S


# --- Hand-back, through a new-runtime turn on the chat route -----------------

# The PriSM fork's answer to a level its template refuses (recorded 2026-10-13).
REFUSED_HIGH = BONSAI_RENDERS["high"]["body"]
TOOLS_AND_EFFORT = {"error": {"message": "model does not support tools; unexpected reasoning effort"}}


@pytest.fixture
def harness(_fallback_harness):
    # The stub of the acceptance run: a llama-server that reports no levels.
    _fallback_harness.session.endpoint_url = LLAMA
    return _fallback_harness


def _provider(harness, monkeypatch, answers):
    """Answer each chat request from ``answers`` in order: (status, JSON body) or text chunks."""
    from tests.chat_agent.test_chat_agent_route import _sse_events

    def chat(request):
        harness.captured["provider_requests"].append(json.loads(request.content))
        answer = answers.pop(0)
        if isinstance(answer, tuple):
            return httpx.Response(answer[0], json=answer[1])
        return httpx.Response(200, content=_sse_events(answer), headers={"content-type": "text/event-stream"})

    serve_probes(monkeypatch, {}, chat=chat)


async def _turn(harness, level):
    from src.chat_agent import store
    from tests.chat_agent.test_chat_agent_route import _Request, _data_events, _switch_on

    _switch_on()
    store.set_chat_settings("alice", "session-1", {store.THINKING_LEVEL: level})
    response, body = await harness.send(_Request("agent"))
    assert response.headers["X-Odysseus-Runtime"] == "chat_agent"
    return _data_events(body), body


def _sent(harness):
    return [(p["model"], p.get("reasoning_effort")) for p in harness.captured["provider_requests"]]


@pytest.mark.asyncio
async def test_refused_level_is_resent_once_as_auto_without_fallback(harness, monkeypatch):
    from tests.chat_agent.test_chat_agent_fallback import PRIMARY_MODEL

    _provider(harness, monkeypatch, [(500, REFUSED_HIGH), ["Hi"]])

    events, body = await _turn(harness, "high")

    assert _sent(harness) == [(PRIMARY_MODEL, "high"), (PRIMARY_MODEL, None)]
    [notice] = [e for e in events if e.get("type") == "thinking_level_refused"]
    assert notice["level"] == "high"
    assert "high" in notice["message"] and PRIMARY_MODEL in notice["message"]
    assert not [e for e in events if e.get("type") == "fallback"]
    assert "event: error" not in body
    [row] = harness.saved
    assert row.content == "Hi"
    url = harness.session.endpoint_url
    assert (await capabilities.thinking_levels(url, PRIMARY_MODEL)).support["high"] == U


@pytest.mark.asyncio
async def test_a_refusal_that_also_hits_the_resend_learns_no_level(harness, monkeypatch):
    from tests.chat_agent.test_chat_agent_fallback import PRIMARY_MODEL

    _provider(harness, monkeypatch, [(400, TOOLS_AND_EFFORT), (400, TOOLS_AND_EFFORT)])

    events, body = await _turn(harness, "max")

    # The re-send without the level fails the same way: the tools are the cause.
    assert _sent(harness) == [(PRIMARY_MODEL, "max"), (PRIMARY_MODEL, None)]
    assert not [e for e in events if e.get("type") == "thinking_level_refused"]
    assert capabilities.NATIVE_TOOLS_REJECTED in body
    url = harness.session.endpoint_url
    assert (await capabilities.thinking_levels(url, PRIMARY_MODEL)).support["max"] == N
    assert await capabilities.native_tools(url, PRIMARY_MODEL) == capabilities.NO


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "answers", "sent"),
    [
        # Auto carried no level: nothing to re-send without it.
        ("auto", [(500, REFUSED_HIGH), ["From backup"]], [None, None]),
        # The re-send without the level fails too: not the level's fault.
        ("high", [(500, {"error": {"message": "upstream 500"}})] * 2 + [["From backup"]],
         ["high", None, "high"]),
    ],
    ids=["auto", "other-500"],
)
async def test_other_failures_still_fall_back(harness, monkeypatch, level, answers, sent):
    from tests.chat_agent.test_chat_agent_fallback import BACKUP_MODEL, PRIMARY_MODEL

    _provider(harness, monkeypatch, answers)

    events, _ = await _turn(harness, level)

    models = [PRIMARY_MODEL] * (len(sent) - 1) + [BACKUP_MODEL]
    assert _sent(harness) == list(zip(models, sent))
    assert not [e for e in events if e.get("type") == "thinking_level_refused"]
    assert [e for e in events if e.get("type") == "fallback"]
    url = harness.session.endpoint_url
    assert (await capabilities.thinking_levels(url, PRIMARY_MODEL)).support["high"] == N


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "message"),
    [
        # LM Studio /v1 refusing a value it doesn't list (#80).
        (400, "Invalid value for 'reasoning_effort': 'max'."),
        (400, "Reasoning is mandatory for this endpoint and cannot be disabled."),
        # xAI, live 2026-11-20: grok-4.7 at max.
        (400, "Invalid reasoning effort."),
        # Wording that names no parameter: only the re-send tells.
        (400, "Request rejected."),
    ],
    ids=["lmstudio-effort", "openrouter-mandatory", "xai-max", "no-parameter-named"],
)
async def test_a_refusal_the_resend_clears_is_learned_whatever_it_says(harness, monkeypatch, status, message):
    from tests.chat_agent.test_chat_agent_fallback import PRIMARY_MODEL

    _provider(harness, monkeypatch, [(status, {"error": {"message": message}}), ["Hi"]])

    events, _ = await _turn(harness, "high")

    assert _sent(harness) == [(PRIMARY_MODEL, "high"), (PRIMARY_MODEL, None)]
    assert [e["level"] for e in events if e.get("type") == "thinking_level_refused"] == ["high"]
    learned = (await capabilities.thinking_levels(harness.session.endpoint_url, PRIMARY_MODEL)).support["high"]
    assert learned == U


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 429])
async def test_failures_no_level_refusal_answers_with_are_not_resent(harness, monkeypatch, status):
    from tests.chat_agent.test_chat_agent_fallback import PRIMARY_MODEL

    _provider(harness, monkeypatch, [(status, {"error": {"message": "no"}}), ["From backup"]])

    events, _ = await _turn(harness, "high")

    assert _sent(harness)[0] == (PRIMARY_MODEL, "high")
    assert (PRIMARY_MODEL, None) not in _sent(harness)
    assert not [e for e in events if e.get("type") == "thinking_level_refused"]


ZAI_PAAS = "https://api.z.ai/api/paas/v4"
# Z.ai's answer to glm-5.3-flash at off, medium and xhigh (live, 2026-11-20).
ZAI_LEVEL_REFUSED = {"error": {
    "code": "1210",
    "message": "This model always engages in thinking and cannot be disabled; please use low, high, or max",
}}


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["off", "medium", "xhigh"])
async def test_zai_refused_level_is_learned_and_resent_once_without_it(harness, monkeypatch, level):
    from tests.chat_agent.test_chat_agent_fallback import PRIMARY_MODEL

    harness.session.endpoint_url = ZAI_PAAS
    _provider(harness, monkeypatch, [(400, ZAI_LEVEL_REFUSED), ["Hi"]])

    events, body = await _turn(harness, level)

    refused, resent = harness.captured["provider_requests"]
    assert "thinking" in refused or refused["reasoning_effort"] == level
    assert "thinking" not in resent and "reasoning_effort" not in resent
    [notice] = [e for e in events if e.get("type") == "thinking_level_refused"]
    assert notice["level"] == level
    assert not [e for e in events if e.get("type") == "fallback"]
    assert "event: error" not in body
    assert (await capabilities.thinking_levels(ZAI_PAAS, PRIMARY_MODEL)).support[level] == U


@pytest.mark.asyncio
async def test_a_failed_resend_is_not_resent_again(harness, monkeypatch):
    from tests.chat_agent.test_chat_agent_fallback import BACKUP_MODEL, PRIMARY_MODEL

    _provider(harness, monkeypatch, [(500, REFUSED_HIGH), (500, REFUSED_HIGH), ["From backup"]])

    events, _ = await _turn(harness, "high")

    assert _sent(harness) == [(PRIMARY_MODEL, "high"), (PRIMARY_MODEL, None), (BACKUP_MODEL, "high")]
    assert not [e for e in events if e.get("type") == "thinking_level_refused"]


# --- Learned levels on disk ---------------------------------------------------

@pytest.mark.asyncio
async def test_a_learned_level_outlives_the_process_caches():
    capabilities.learn_unsupported_level(ZAI_PAAS, "glm-5.3-flash", "medium")
    # What a restart loses.
    capabilities._reported.clear()
    capabilities._learned.clear()

    levels = await capabilities.thinking_levels(ZAI_PAAS, "glm-5.3-flash")

    assert levels.support["medium"] == U
    assert levels.support["high"] == N


@pytest.mark.asyncio
async def test_a_model_list_probe_keeps_learned_levels_and_a_manual_refresh_drops_them():
    capabilities.learn_unsupported_level(ZAI_PAAS, "glm-5.3-flash", "medium")
    capabilities.learn_unsupported_level(LLAMA, BONSAI, "low")

    capabilities.forget_endpoint(ZAI_PAAS)
    assert (await capabilities.thinking_levels(ZAI_PAAS, "glm-5.3-flash")).support["medium"] == U

    capabilities.forget_learned_levels(ZAI_PAAS)
    assert (await capabilities.thinking_levels(ZAI_PAAS, "glm-5.3-flash")).support["medium"] == N
    # Another server's levels stay.
    assert (await capabilities.thinking_levels(LLAMA, BONSAI)).support["low"] == U


@pytest.mark.asyncio
async def test_an_unreadable_learned_store_reports_nothing_learned():
    from src import constants

    path = Path(constants.CHAT_AGENT_DIR) / "learned" / "thinking_levels.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")

    levels = await capabilities.thinking_levels(ZAI_PAAS, "glm-5.3-flash")

    assert levels.support["medium"] == N
    capabilities.learn_unsupported_level(ZAI_PAAS, "glm-5.3-flash", "medium")
    assert (await capabilities.thinking_levels(ZAI_PAAS, "glm-5.3-flash")).support["medium"] == U


@pytest.mark.asyncio
async def test_capabilities_route_greys_a_learned_refusal(api, monkeypatch):
    from tests.chat_agent.test_chat_agent_thinking_level import _capabilities, _switch_on

    _switch_on()
    serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS, "POST /apply-template": _renders(BONSAI_RENDERS)})
    # The chat's model refused Low in a turn (a template change since the probe, say).
    capabilities.learn_unsupported_level(LLAMA, BONSAI, "low")

    levels = (await _capabilities(api)).json()["thinking"]["levels"]

    assert levels == _support(S, U, S, U, S, U)
