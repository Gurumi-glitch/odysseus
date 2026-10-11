"""app_api refuses blocked routes by the path the server will route (fork #46)."""

import json

import httpx
import pytest

from src.constants import internal_api_base
from src.tool_implementations import do_app_api


class _RecordingAsyncClient:
    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def request(self, method, url, **kwargs):
        type(self).calls.append((method, url))
        return httpx.Response(200, json={"ok": True})


@pytest.fixture
def loopback(monkeypatch):
    _RecordingAsyncClient.calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _RecordingAsyncClient)
    return _RecordingAsyncClient.calls


async def _call(method, path):
    return await do_app_api(
        json.dumps({"action": "call", "method": method, "path": path, "body": {}}),
        owner="admin",
    )


@pytest.mark.parametrize(
    "method,path",
    [
        pytest.param("GET", "/x/../api/auth/status", id="dot-segments-prefix"),
        pytest.param("POST", "/api/x/../users", id="dot-segments-users"),
        pytest.param("GET", "/api/./tokens", id="single-dot-segment"),
        pytest.param("GET", "/%61pi/admin/x", id="percent-escape-prefix"),
        pytest.param("POST", "/api/%73hell/exec", id="percent-escape-shell"),
        pytest.param("POST", "/api/cookbook/x/../state", id="dot-segments-method-path"),
        pytest.param("POST", "/api/cookbook/%6Bill-pid", id="percent-escape-method-path"),
        pytest.param("POST", "/api/import", id="backup-import"),
        pytest.param("PUT", "/api/prefs/some-key", id="prefs-write"),
        pytest.param("POST", "/api/session/s1/update-last-meta", id="session-meta-write"),
        pytest.param("POST", "/api/session/s1/message", id="session-inject"),
        pytest.param("POST", "/api/session/s1/edit-message", id="session-edit"),
        pytest.param("POST", "/api/session/s1/merge-last-assistant", id="session-merge"),
        pytest.param("POST", "/api/session/s1/truncate", id="session-truncate"),
        pytest.param("POST", "/api/session/s1/delete-messages", id="session-delete-messages"),
        pytest.param("POST", "/api/chat_stream", id="chat-stream"),
        pytest.param("POST", "/api/chat/stop/s1", id="chat-run-control"),
        pytest.param("PUT", "/api/skills/my-skill", id="skill-publish-or-user-only-toggle"),
        pytest.param("POST", "/api/skills/my-skill/markdown", id="skill-markdown-save"),
        pytest.param("PUT", "/api/skills/x/../my-skill", id="skill-write-dot-segments"),
    ],
)
async def test_app_api_refuses_blocked_route(loopback, method, path):
    result = await _call(method, path)

    assert result["exit_code"] == 1
    assert "blocked" in result["error"].lower() or "don't" in result["error"].lower()
    assert loopback == []


@pytest.mark.parametrize(
    "method,path",
    [
        pytest.param("GET", "/api/prefs/some-key", id="prefs-read"),
        pytest.param("GET", "/api/session/s1/context", id="session-read"),
        pytest.param("GET", "/api/cookbook/gpus", id="plain-read"),
        pytest.param("GET", "/api/documents/foo%20bar", id="escaped-read"),
    ],
)
async def test_app_api_sends_allowed_route_from_internal_base(loopback, method, path):
    result = await _call(method, path)

    assert result["exit_code"] == 0
    assert loopback == [(method, httpx.URL(f"{internal_api_base()}{path}"))]
