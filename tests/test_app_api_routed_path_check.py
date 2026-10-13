"""app_api checks the path the server will route (fork #46) against its allowlist (fork #71)."""

import json
import re

import httpx
import pytest

from src.constants import internal_api_base
from src.tool_implementations import do_app_api
from src.tools.system import _APP_API_WRITES


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
        pytest.param("GET", "/%61pi/auth/settings", id="percent-escape-prefix"),
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
        pytest.param("POST", "/api/chat", id="chat-non-streaming"),
        pytest.param("POST", "/api/inject_context/s1", id="session-context-inject"),
        pytest.param("PATCH", "/api/session/s1", id="session-patch"),
        pytest.param("DELETE", "/api/session/s1", id="session-delete"),
        pytest.param("DELETE", "/api/sessions/all", id="sessions-delete-all"),
        pytest.param("POST", "/api/sessions/bulk-delete", id="sessions-bulk-delete"),
        pytest.param("POST", "/api/tools", id="tool-enablement"),
        pytest.param("PATCH", "/api/mcp/servers/m1/tools", id="mcp-tool-enablement"),
        pytest.param("PATCH", "/api/mcp/servers/m1", id="mcp-server-toggle"),
        pytest.param("POST", "/api/cookbook/install-system-deps", id="host-package-install"),
        pytest.param("POST", "/api/cookbook/setup", id="host-remote-setup"),
        pytest.param("POST", "/api/cookbook/ssh-key", id="host-ssh-key"),
        pytest.param("POST", "/api/cookbook/test-ssh", id="host-ssh-probe"),
        pytest.param("POST", "/api/codex/cookbook/serve", id="serve-alias"),
        pytest.param("POST", "/api/codex/cookbook/preset/p1", id="serve-preset-alias"),
        pytest.param("POST", "/api/codex/calendar/events", id="calendar-alias"),
        pytest.param("DELETE", "/api/codex/calendar/events/e1", id="calendar-delete-alias"),
        pytest.param("POST", "/api/mcp/oauth/exchange/m1", id="mcp-sign-in"),
        pytest.param("POST", "/api/copilot/device/start", id="provider-sign-in"),
        pytest.param("POST", "/api/chatgpt-subscription/device/poll", id="provider-sign-in-alias"),
        pytest.param("PUT", "/api/skills/my-skill", id="skill-publish-or-user-only-toggle"),
        pytest.param("POST", "/api/skills/my-skill/markdown", id="skill-markdown-save"),
        pytest.param("PUT", "/api/skills/x/../my-skill", id="skill-write-dot-segments"),
        pytest.param("POST", "/api/session", id="session-create"),
        pytest.param("GET", "/api/prefs/some-key", id="prefs-read"),
        pytest.param("GET", "/api/mcp/servers/m1/tools", id="mcp-read"),
        # Allowlist audit classes (fork #71).
        pytest.param("POST", "/api/companion/pair", id="credential-mint"),
        pytest.param("POST", "/api/email/send", id="outbound-send-gate"),
        pytest.param("POST", "/api/codex/emails/send", id="outbound-send-gate-alias"),
        pytest.param("POST", "/api/email/schedule", id="outbound-schedule-gate"),
        pytest.param("POST", "/api/email/pending/s1/approve", id="outbound-approval-gate"),
        pytest.param("POST", "/api/email/accounts", id="account-credential-create"),
        pytest.param("PUT", "/api/email/accounts/a1", id="account-credential-update"),
        pytest.param("DELETE", "/api/email/accounts/a1", id="account-credential-delete"),
        pytest.param("POST", "/api/calendar/config/accounts", id="account-credential-create-sibling"),
        pytest.param("POST", "/api/calendar/config", id="account-config-write"),
        pytest.param("PUT", "/api/email/config", id="account-config-write-sibling"),
        pytest.param("PUT", "/api/contacts/config", id="account-config-write-alias"),
        pytest.param("GET", "/api/export", id="secret-read-backup"),
        pytest.param("GET", "/api/tasks/t1", id="secret-read-trigger-token"),
        pytest.param("GET", "/api/mcp/servers", id="secret-read-server-env"),
        pytest.param("GET", "/api/mcp/oauth/authorize/m1", id="sign-in-flow-read"),
        pytest.param("GET", "/api/email/oauth/google/authorize", id="sign-in-flow-read-sibling"),
        pytest.param("GET", "/api/webhooks", id="secret-read-hook-url"),
        pytest.param("GET", "/api/prefs", id="secret-read-stored-passwords"),
        pytest.param("DELETE", "/api/skills/s1", id="persistent-prompt-delete"),
        pytest.param("DELETE", "/api/skills/builtin/b1", id="persistent-prompt-reset"),
        pytest.param("POST", "/api/presets/custom", id="persistent-prompt-write"),
        pytest.param("POST", "/api/presets/groups", id="persistent-prompt-write-sibling"),
        pytest.param("POST", "/api/presets/templates", id="persistent-prompt-write-alias"),
        pytest.param("DELETE", "/api/mcp/servers/m1", id="control-sibling-delete"),
        pytest.param("POST", "/api/mcp/servers/m1/reconnect", id="control-sibling-reconnect"),
        pytest.param("PATCH", "/api/webhooks/w1", id="hook-sibling-toggle"),
        pytest.param("DELETE", "/api/webhooks/w1", id="hook-sibling-delete"),
        pytest.param("POST", "/api/webhooks/w1/test", id="hook-sibling-fire"),
        pytest.param("PATCH", "/api/model-endpoints/e1", id="endpoint-sibling-toggle"),
        pytest.param("PATCH", "/api/model-endpoints/e1/models", id="endpoint-sibling-models"),
        pytest.param("POST", "/api/tasks", id="scheduled-create"),
        pytest.param("POST", "/api/tasks/t1/webhook-regenerate", id="scheduled-token-mint"),
        pytest.param("POST", "/api/tasks/t1/run", id="scheduled-run"),
        pytest.param("POST", "/api/tasks/t1/pause", id="scheduled-pause"),
        pytest.param("POST", "/api/tasks/t1/resume", id="scheduled-resume"),
        pytest.param("POST", "/api/tasks/t1/revert", id="scheduled-revert"),
        pytest.param("PUT", "/api/tasks/t1", id="scheduled-update"),
        pytest.param("DELETE", "/api/tasks/t1", id="scheduled-delete"),
        pytest.param("POST", "/api/assistant/run/t1", id="scheduled-run-alias"),
        pytest.param("POST", "/api/cleanup", id="history-bulk-cleanup"),
        pytest.param("POST", "/api/sessions/auto-sort", id="history-bulk-sort"),
        pytest.param("POST", "/api/research/spinoff/r1", id="history-spinoff"),
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
        pytest.param("GET", "/api/session/s1/context", id="session-read"),
        pytest.param("GET", "/api/cookbook/gpus", id="plain-read"),
        pytest.param("GET", "/api/gallery/library", id="gallery-read"),
        pytest.param("POST", "/api/gallery/albums", id="gallery-album-create"),
        pytest.param("PUT", "/api/gallery/albums/a1", id="gallery-album-rename"),
        pytest.param("POST", "/api/gallery/albums/a1/add", id="gallery-album-add"),
        pytest.param("POST", "/api/gallery/albums/a1/remove", id="gallery-album-remove"),
        pytest.param("POST", "/api/gallery/i1/favorite", id="gallery-favorite"),
        pytest.param("POST", "/api/gallery/i1/rename", id="gallery-rename"),
        pytest.param("PATCH", "/api/gallery/i1", id="gallery-patch"),
        pytest.param("GET", "/api/documents/foo%20bar", id="escaped-read"),
    ],
)
async def test_app_api_sends_allowed_route_from_internal_base(loopback, method, path):
    result = await _call(method, path)

    assert result["exit_code"] == 0
    assert loopback == [(method, httpx.URL(f"{internal_api_base()}{path}"))]


def _concrete(template):
    return re.sub(r"\{[^}]+\}", "x1", template)


def _app_routes():
    import app as app_module
    from fastapi.routing import iter_route_contexts

    for route in iter_route_contexts(app_module.app.routes):
        for method in route.methods or ():
            yield method, route.path


async def test_app_api_sends_no_write_outside_the_allowlist(loopback):
    routes = set(_app_routes())
    allowed = set(_APP_API_WRITES)
    assert allowed <= routes, f"allowlisted writes with no route: {allowed - routes}"

    sent = []
    for method, template in sorted(routes):
        if method in ("GET", "HEAD", "OPTIONS") or (method, template) in allowed:
            continue
        result = await _call(method, _concrete(template))
        if result["exit_code"] == 0:
            sent.append((method, template))

    assert sent == [], "new write routes need an app_api allowlist decision (fork #71)"
    assert loopback == []
