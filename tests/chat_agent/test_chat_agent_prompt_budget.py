"""The new runtime's prompt on a small context window (ticket #94, spec #89).

Route tests drive POST /api/chat_stream with a scripted provider and assert the
outgoing request (tool surface, messages) and the error a request that cannot
fit gets before anything is sent.
"""

import json
import logging

import pytest

from src.chat_agent import loop, transcript
from src.generation_budget import estimate_tool_schema_tokens
from tests.chat_agent.test_chat_agent_route import _Request, _switch_on
from tests.chat_agent.test_chat_agent_route import harness  # noqa: F401

# Bonsai's llama-server window (spec #89 acceptance setup).
SMALL_WINDOW = 16_384
LARGE_WINDOW = 131_072


def _window(monkeypatch, window, measured):
    monkeypatch.setattr(loop, "get_context_length_known", lambda url, model: (window, measured))


def _configured_surface(monkeypatch, surface):
    # The endpoint row's per-model setting (a DB lookup) is the boundary here.
    monkeypatch.setattr(loop, "_configured_model_tool_surface", lambda *a, **k: surface)


def _error(body):
    lines = body.splitlines()
    return json.loads(lines[lines.index("event: error") + 1][len("data: "):])


def _compact(payload):
    return all("description" not in tool["function"] for tool in payload["tools"])


@pytest.mark.asyncio
async def test_full_schemas_over_a_quarter_of_a_small_measured_window_switch_to_compact(
    harness, monkeypatch, caplog,
):
    _switch_on()
    _window(monkeypatch, SMALL_WINDOW, True)

    with caplog.at_level(logging.INFO, logger=loop.__name__):
        await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert payload["tools"] and _compact(payload)
    assert "tool surface compact" in caplog.text


@pytest.mark.asyncio
async def test_the_harness_tool_set_is_bonsai_size(harness, monkeypatch):
    # Pins the premise of the small-window tests: the full set is over 25% of
    # 16k (Bonsai: 7.4k tokens, 45%) and under 25% of the large window.
    _switch_on()
    _window(monkeypatch, LARGE_WINDOW, True)

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert not _compact(payload)
    assert SMALL_WINDOW // 4 < estimate_tool_schema_tokens(payload["tools"]) < LARGE_WINDOW // 4


@pytest.mark.asyncio
async def test_unmeasured_window_keeps_the_full_surface(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch, SMALL_WINDOW, False)

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    assert not _compact(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["full", "compact", "none"])
async def test_an_explicit_surface_is_kept_on_a_small_window(harness, monkeypatch, surface):
    _switch_on()
    _window(monkeypatch, SMALL_WINDOW, True)
    _configured_surface(monkeypatch, surface)

    await harness.send(_Request("agent"))

    [payload] = harness.captured["provider_requests"]
    if surface == "none":
        assert "tools" not in payload
    else:
        assert payload["tools"] and _compact(payload) == (surface == "compact")


@pytest.mark.asyncio
async def test_the_switch_changes_only_the_tools_not_the_messages(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch, LARGE_WINDOW, True)
    await harness.send(_Request("agent"))
    _window(monkeypatch, SMALL_WINDOW, True)
    await harness.send(_Request("agent"))

    large, small = harness.captured["provider_requests"]
    assert small["messages"] == large["messages"]
    assert _compact(small) and not _compact(large)


# "word " estimates at ~1.6 tokens; the harness's compact tool set adds ~2.6k.
FITS = "word " * 5_000  # ~10.8k of 16k with the tools: 66%
TOO_LARGE = "word " * 8_000  # ~15.5k of 16k: over 85%


@pytest.mark.asyncio
async def test_a_first_request_over_the_threshold_is_refused_before_sending(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch, SMALL_WINDOW, True)
    harness.context.current = {"role": "user", "content": TOO_LARGE}

    response, body = await harness.send(_Request("agent"))

    assert harness.captured["provider_requests"] == []
    error = _error(body)["error"]
    assert error.startswith("request ~") and error.endswith(f" tokens exceeds the {SMALL_WINDOW}-token window")
    assert SMALL_WINDOW * 85 // 100 < int(error.split("~")[1].split(" ")[0]) < SMALL_WINDOW
    # Kept on reload, like the runtime's other stops.
    [row] = harness.saved
    assert row.metadata["error"] == error


@pytest.mark.asyncio
async def test_a_request_that_fits_is_sent(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch, SMALL_WINDOW, True)
    harness.context.current = {"role": "user", "content": FITS}

    await harness.send(_Request("agent"))

    assert len(harness.captured["provider_requests"]) == 1


@pytest.mark.asyncio
async def test_the_gate_uses_the_auto_compact_threshold(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch, SMALL_WINDOW, True)
    monkeypatch.setattr(transcript, "auto_compact_threshold_percent", lambda: 50)
    harness.context.current = {"role": "user", "content": FITS}

    response, body = await harness.send(_Request("agent"))

    assert harness.captured["provider_requests"] == []
    assert _error(body)["error"].endswith(f" tokens exceeds the {SMALL_WINDOW}-token window")


@pytest.mark.asyncio
async def test_unmeasured_window_never_refuses(harness, monkeypatch):
    _switch_on()
    _window(monkeypatch, SMALL_WINDOW, False)
    harness.context.current = {"role": "user", "content": TOO_LARGE}

    await harness.send(_Request("agent"))

    assert len(harness.captured["provider_requests"]) == 1
