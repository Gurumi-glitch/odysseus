"""The Codex bridge calls the /api/email/list handler as a plain function, so
FastAPI never resolves its ``Query(...)`` defaults. Any parameter the bridge
omits arrives as a truthy ``FieldInfo``; ``date_from`` then reached
``strptime`` as a ``Query`` object, raised TypeError, and the bearer-token
middleware reported it as a 401 "Invalid API token".
"""
import inspect
from types import SimpleNamespace

import pytest
from fastapi.params import Depends
from starlette.requests import Request

import routes.codex_routes as codex_routes
from routes.email.email_routes import setup_email_routes


def _real_list_signature() -> inspect.Signature:
    for route in setup_email_routes().routes:
        if route.path == "/api/email/list" and "GET" in route.methods:
            return inspect.signature(route.endpoint)
    raise AssertionError("GET /api/email/list route not found")


def _email_read_request() -> Request:
    request = Request(
        {"type": "http", "method": "GET", "path": "/api/codex/emails", "headers": [], "state": {}}
    )
    request.state.api_token = True
    request.state.api_token_owner = "alice"
    request.state.api_token_scopes = ["email:read"]
    return request


@pytest.mark.asyncio
async def test_codex_email_list_passes_every_query_param_explicitly():
    calls = []

    async def spy(**kwargs):
        calls.append(kwargs)
        return {"emails": [], "total": 0}

    route = SimpleNamespace(path="/api/email/list", methods={"GET"}, endpoint=spy)
    router = codex_routes.setup_codex_routes(email_router=SimpleNamespace(routes=[route]))
    endpoint = next(
        r.endpoint for r in router.routes if r.path == "/api/codex/emails" and "GET" in r.methods
    )

    await endpoint(_email_read_request(), folder="INBOX", limit=8, offset=0, filter="all")

    query_params = {
        name
        for name, param in _real_list_signature().parameters.items()
        if not isinstance(param.default, Depends)
    }
    assert query_params <= set(calls[-1])
    assert not [k for k, v in calls[-1].items() if type(v).__module__.startswith("fastapi")]
