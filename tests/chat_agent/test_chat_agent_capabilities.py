"""Native tool detection per (endpoint, model) (ticket #90, spec #89).

Drives the capabilities module through its public interface with recorded
server payloads (Bonsai on llama-server, an LM Studio model list) served by
one httpx.MockTransport, the same transport the provider stream uses.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import src.chat_helpers as chat_helpers
from src.chat_agent import capabilities
from src.chat_agent.providers import chat_completions

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "chat_agent"
BONSAI_PROPS = json.loads((FIXTURES / "llamacpp_props_bonsai.json").read_text())
LMSTUDIO_MODELS = json.loads((FIXTURES / "lmstudio_api_v1_models.json").read_text())

LLAMA = "http://localhost:1235/v1"
BONSAI = "Ternary-Bonsai-2-27B"


def serve_probes(monkeypatch, routes, *, chat=None):
    """Serve every outgoing request from one mock transport.

    ``routes`` maps "METHOD path" to a JSON body or an int status; unrouted
    probe paths answer 404. ``chat`` answers POST .../chat/completions.
    Returns the list of "METHOD path?query" strings seen.
    """
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.query.decode() if isinstance(request.url.query, bytes) else request.url.query
        seen.append(f"{request.method} {request.url.path}" + (f"?{query}" if query else ""))
        if request.url.path.endswith("/chat/completions") and chat is not None:
            return chat(request)
        answer = routes.get(f"{request.method} {request.url.path}", 404)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return httpx.Response(answer, json={"error": "nope"})
        return httpx.Response(200, json=answer)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)
    # The LM Studio fingerprint probe is chat_helpers' own synchronous GET.
    monkeypatch.setattr(chat_helpers, "httpx", SimpleNamespace(get=httpx.Client(transport=transport).get))
    return seen


@pytest.fixture(autouse=True)
def _fresh_caches():
    caches = (capabilities._reported, capabilities._learned, chat_helpers._lmstudio_models_cache)
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


def _props(**caps):
    return {**BONSAI_PROPS, "chat_template_caps": caps}


async def _ask(url=LLAMA, model=BONSAI, headers=None):
    return await capabilities.native_tools(url, model, headers)


@pytest.mark.asyncio
async def test_bonsai_props_report_native_tools(monkeypatch):
    seen = serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS})

    assert await _ask() == capabilities.YES
    # Router-mode llama-server answers per model.
    assert f"GET /props?model={BONSAI}" in seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caps", "verdict"),
    [
        ({"supports_tools": True, "supports_tool_calls": False}, "yes"),
        ({"supports_tools": False, "supports_tool_calls": True}, "yes"),
        ({"supports_tools": False, "supports_tool_calls": False}, "no"),
    ],
    ids=["tools-only", "tool-calls-only", "neither"],
)
async def test_llamacpp_either_template_flag_means_native_tools(monkeypatch, caps, verdict):
    # Pins the llama.cpp reader: either flag true -> tool_call capability.
    serve_probes(monkeypatch, {"GET /props": _props(**caps)})

    assert await _ask() == verdict


@pytest.mark.asyncio
async def test_props_without_template_caps_report_nothing(monkeypatch):
    props = {k: v for k, v in BONSAI_PROPS.items() if k != "chat_template_caps"}
    serve_probes(monkeypatch, {"GET /props": props})

    assert await _ask() == capabilities.UNKNOWN


@pytest.mark.asyncio
async def test_lmstudio_list_answers_every_model_from_one_fetch(monkeypatch):
    seen = serve_probes(monkeypatch, {"GET /api/v1/models": LMSTUDIO_MODELS})
    url = "http://localhost:1234/v1"

    assert await _ask(url, "ternary-bonsai-2-27b-ptq1_0-mtp") == capabilities.YES
    assert await _ask(url, "jev-style-qwen3.5-2b-decision") == capabilities.NO
    assert seen == ["GET /api/v1/models"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caps", "verdict"),
    [(["completion", "tools"], "yes"), (["completion"], "no"), (None, "unknown")],
    ids=["tools", "no-tools", "old-ollama"],
)
async def test_ollama_show_capabilities(monkeypatch, caps, verdict):
    show = {"details": {"family": "qwen3"}}
    if caps is not None:
        show["capabilities"] = caps
    seen = serve_probes(monkeypatch, {"POST /api/show": show})

    assert await _ask("http://localhost:11434/v1", "qwen3:8b") == verdict
    assert seen == ["POST /api/show"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "verdict"),
    [(["tools", "tool_choice", "temperature"], "yes"), (["temperature"], "no")],
)
async def test_openrouter_supported_parameters(monkeypatch, params, verdict):
    catalog = {"data": [{
        "id": "vendor/model", "architecture": {"modality": "text->text"},
        "supported_parameters": params,
    }]}
    serve_probes(monkeypatch, {"GET /api/v1/models": catalog})

    assert await _ask("https://openrouter.ai/api/v1", "vendor/model") == verdict


@pytest.mark.asyncio
async def test_mistral_and_copilot_per_model_flags(monkeypatch):
    serve_probes(monkeypatch, {
        "GET /v1/models": {"data": [{"id": "codestral", "capabilities": {"function_calling": False}}]},
        "GET /models": {"data": [{"id": "gpt-4.1", "capabilities": {"supports": {"tool_calls": True}}}]},
    })

    assert await _ask("https://api.mistral.ai/v1", "codestral") == capabilities.NO
    assert await _ask("https://api.githubcopilot.com", "gpt-4.1") == capabilities.YES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    ["https://api.openai.com/v1", "https://api.x.ai/v1", "http://localhost:8000/v1", "https://llm.example/v1"],
    ids=["openai", "xai", "vllm", "remote-unknown"],
)
async def test_servers_known_to_report_nothing_are_not_probed(monkeypatch, url):
    seen = serve_probes(monkeypatch, {})

    assert await _ask(url, "any-model") == capabilities.UNKNOWN
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [500, httpx.ConnectTimeout("slow")], ids=["http-500", "timeout"],
)
async def test_failed_probe_is_unknown_and_not_cached(monkeypatch, failure):
    routes = {"GET /props": failure}
    seen = serve_probes(monkeypatch, routes)

    assert await _ask() == capabilities.UNKNOWN
    routes["GET /props"] = BONSAI_PROPS
    assert await _ask() == capabilities.YES
    assert len([s for s in seen if s.startswith("GET /props")]) == 2


@pytest.mark.asyncio
async def test_server_reported_value_is_cached_for_60s(monkeypatch):
    seen = serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS})
    now = [1000.0]
    monkeypatch.setattr(capabilities, "_now", lambda: now[0])

    await _ask()
    now[0] += 59
    await _ask()
    assert len([s for s in seen if s.startswith("GET /props")]) == 1
    now[0] += 2
    await _ask()
    assert len([s for s in seen if s.startswith("GET /props")]) == 2


@pytest.mark.asyncio
async def test_learned_no_has_no_ttl_and_refresh_clears_it(monkeypatch):
    seen = serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS})
    now = [1000.0]
    monkeypatch.setattr(capabilities, "_now", lambda: now[0])

    capabilities.learn_no_native_tools(LLAMA, BONSAI)
    now[0] += 10_000
    assert await _ask() == capabilities.NO
    assert seen == []

    capabilities.forget_endpoint("http://localhost:1235")
    assert await _ask() == capabilities.YES


@pytest.mark.asyncio
async def test_refresh_clears_server_reported_values_and_lmstudio_fingerprint(monkeypatch):
    seen = serve_probes(monkeypatch, {"GET /api/v1/models": LMSTUDIO_MODELS})
    url = "http://localhost:1234/v1"

    await _ask(url, "ternary-bonsai-2-27b-ptq1_0-mtp")
    capabilities.forget_endpoint("http://localhost:1234/v1/")
    await _ask(url, "ternary-bonsai-2-27b-ptq1_0-mtp")

    assert seen == ["GET /api/v1/models", "GET /api/v1/models"]


@pytest.mark.parametrize(
    ("status", "message", "rejected"),
    [
        (400, 'Ollama returned HTTP 400: registry.ollama.ai/library/gemma3:4b does not support tools', True),
        (400, 'provider returned HTTP 400: "auto" tool choice requires --enable-auto-tool-choice', True),
        (400, "provider returned HTTP 400: tools are not supported by this model", True),
        (500, "provider returned HTTP 500: does not support tools", False),
        (400, "provider returned HTTP 400: messages must alternate", False),
    ],
    ids=["ollama", "vllm", "generic", "not-400", "other-400"],
)
def test_tools_rejection_matcher(status, message, rejected):
    assert capabilities.is_tools_rejection(status, message) is rejected


@pytest.mark.asyncio
async def test_model_list_refresh_forgets_the_endpoint(monkeypatch):
    import routes.model_routes as model_routes
    import src.llm_core as llm_core

    seen = serve_probes(monkeypatch, {"GET /props": BONSAI_PROPS})
    capabilities.learn_no_native_tools(LLAMA, BONSAI)
    monkeypatch.setattr(
        llm_core, "httpx_get_kimi_aware",
        lambda url, headers, **kwargs: httpx.Response(
            200, json={"data": [{"id": BONSAI}]}, request=httpx.Request("GET", url),
        ),
    )

    assert model_routes._probe_endpoint("http://localhost:1235/v1") == [BONSAI]

    assert await _ask() == capabilities.YES
    assert f"GET /props?model={BONSAI}" in seen


@pytest.mark.asyncio
async def test_list_refill_keeps_a_learned_no(monkeypatch):
    serve_probes(monkeypatch, {"GET /api/v1/models": LMSTUDIO_MODELS})
    url = "http://localhost:1234/v1"

    capabilities.learn_no_native_tools(url, "ternary-bonsai-2-27b-ptq1_0-mtp")
    assert await _ask(url, "qwythos-9b-openthoughts-i1") == capabilities.YES

    assert await _ask(url, "ternary-bonsai-2-27b-ptq1_0-mtp") == capabilities.NO


@pytest.mark.asyncio
async def test_probe_that_never_finishes_is_unknown_within_the_timeout(monkeypatch):
    import asyncio

    async def trickle():
        yield b"{"
        await asyncio.Event().wait()

    def handler(request):
        if request.url.path == "/props":
            return httpx.Response(200, content=trickle())
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(chat_completions, "_get_http_client", lambda: client)
    monkeypatch.setattr(chat_helpers, "httpx", SimpleNamespace(get=httpx.Client(transport=transport).get))
    monkeypatch.setattr(capabilities, "PROBE_TIMEOUT_SECONDS", 0.2)

    assert await asyncio.wait_for(_ask(), 2) == capabilities.UNKNOWN
