"""Pins for the legacy helpers the chat_agent runtime imports (R2).

Each test states the one assumption the runtime relies on, so an upstream
change to a private helper fails here instead of misrouting a turn.
Session.add_message stamping ``_db_id`` is pinned by
tests/test_session_manager_persist_guard.py.
"""

import json

import pytest

from src.agent_loop import _agent_route_tool_mode, _apply_tool_surface_to_schemas
from src.agent_runtime.authority import active_request_authority
from src.agent_tools import FUNCTION_TOOL_SCHEMAS
from src.mcp_manager import McpManager
from src.tool_capabilities import (
    _action_from_content,
    tool_result_is_successful,
    tool_result_was_denied,
)
from src.tool_execution import format_tool_result
from src.tool_policy import WEB_ACCESS_TOOL_NAMES
from src.tool_schemas import function_call_to_tool_block
from src.turn_contract import canonical_tool
from src.chat_agent.providers.chat_completions import Endpoint, build_payload
from src.llm_core import (
    _detect_provider,
    _format_upstream_error,
    _normalize_openai_chat_url,
    _provider_headers,
)


@pytest.mark.parametrize(
    ("url", "provider"),
    [
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://integrate.api.nvidia.com/v1", "nvidia"),
        ("http://192.168.1.20:1234/v1", "openai"),
        ("https://api.x.ai/v1", "openai"),
        ("https://api.anthropic.com/v1", "anthropic"),
        ("http://localhost:11434", "ollama"),
        ("https://chatgpt.com/backend-api/codex", "chatgpt-subscription"),
    ],
)
def test_detect_provider_maps_endpoint_url_to_dialect(url, provider):
    assert _detect_provider(url) == provider


@pytest.mark.parametrize(
    ("url", "model", "native"),
    [
        ("https://openrouter.ai/api/v1", "x-ai/grok-4.1-fast", True),
        ("http://192.168.1.20:1234/v1", "qwen3-30b-a3b", True),
        ("http://localhost:11434/v1", "qwen3:8b", False),
        ("https://llm.example/v1", "gpt-oss-120b", False),
    ],
)
def test_agent_route_tool_mode_first_value_is_native_tool_decision(url, model, native):
    assert _agent_route_tool_mode(url, model, "alice", None)[0] is native


def test_openai_chat_url_appends_chat_completions_once():
    assert _normalize_openai_chat_url("https://openrouter.ai/api/v1/") == (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    assert _normalize_openai_chat_url("http://h/v1/chat/completions") == "http://h/v1/chat/completions"


def test_provider_headers_keep_session_auth_and_add_json_content_type():
    headers = _provider_headers("openrouter", {"Authorization": "Bearer k"})
    assert headers["Authorization"] == "Bearer k"
    assert headers["Content-Type"] == "application/json"


def test_upstream_error_is_a_readable_sentence():
    message = _format_upstream_error(429, b'{"error": {"message": "slow down"}}', "https://openrouter.ai/api/v1")
    assert "429" in message and "slow down" in message


def test_payload_follows_per_model_temperature_and_token_rules():
    plain = build_payload(Endpoint("https://openrouter.ai/api/v1", "x-ai/grok-4.1-fast", None),
                          [], temperature=0.7, max_tokens=0)
    assert plain["temperature"] == 0.7
    assert "max_tokens" not in plain and "max_completion_tokens" not in plain

    reasoning = build_payload(Endpoint("https://api.openai.com/v1", "gpt-5", None),
                              [], temperature=0.7, max_tokens=500)
    assert "temperature" not in reasoning
    assert reasoning["max_completion_tokens"] == 500


# --- #50: tools, permissions and the execution wrapper ---------------------


def test_native_email_call_becomes_the_email_mcp_block_and_bad_args_none():
    block = function_call_to_tool_block("send_email", json.dumps({"to": "a@b.c", "subject": "s", "body": "b"}))
    assert block.tool_type == "mcp__email__send_email"
    assert canonical_tool(block.tool_type) == "send_email"
    assert function_call_to_tool_block("web_search", "{}") is None


def test_tool_surface_none_sends_nothing_and_compact_shrinks_schemas():
    assert _apply_tool_surface_to_schemas(FUNCTION_TOOL_SCHEMAS, "none") == []
    compact = _apply_tool_surface_to_schemas(FUNCTION_TOOL_SCHEMAS, "compact")
    assert len(compact) == len(FUNCTION_TOOL_SCHEMAS)
    assert len(json.dumps(compact)) < len(json.dumps(FUNCTION_TOOL_SCHEMAS))


def test_no_request_authority_is_bound_outside_the_legacy_decorators():
    assert active_request_authority() is None


def test_result_classification_and_model_text():
    assert tool_result_is_successful({"output": "ok", "exit_code": 0})
    assert not tool_result_is_successful({"error": "HTTP 404", "exit_code": 1})
    assert tool_result_was_denied({"error": "x", "blocked": True, "failure_kind": "resource_identity_denied"})
    assert not tool_result_was_denied({"error": "HTTP 404", "exit_code": 1})
    assert "HTTP 404" in format_tool_result("web_fetch: x", {"error": "web_fetch: x: HTTP 404", "exit_code": 1})


def test_manage_tasks_action_includes_the_implicit_create():
    assert _action_from_content("manage_tasks", json.dumps({"task": "news"})) == "create"
    assert _action_from_content("manage_notes", json.dumps({"action": "create"})) == "add"


def test_web_access_names_are_the_spec_web_family():
    assert WEB_ACCESS_TOOL_NAMES == {"web_search", "web_fetch", "get_weather", "youtube_tool", "private_browser"}


def test_plan_mode_blocked_mcp_returns_qualified_write_names_second():
    manager = McpManager()
    manager._tools = {"github": [
        {"name": "create_issue", "annotations": {"readOnlyHint": False}},
        {"name": "list_issues", "annotations": {"readOnlyHint": True}},
    ]}
    _, qualified = manager.plan_mode_blocked_mcp()
    assert qualified == {"mcp__github__create_issue"}
