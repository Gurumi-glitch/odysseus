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
from src.chat_agent.providers import Endpoint
from src.chat_agent.providers.chat_completions import build_payload
from src.endpoint_resolver import normalize_base
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
from src.llm_core import (
    _detect_provider,
    _format_upstream_error,
    _host_match,
    _normalize_openai_chat_url,
    _provider_headers,
)


@pytest.mark.parametrize(
    ("url", "xai"),
    [
        ("https://api.x.ai/v1", True),
        ("https://x.ai", True),
        ("https://max.ai/v1", False),
        ("https://proxy.example/x.ai/v1", False),
    ],
)
def test_host_match_compares_hostnames_not_substrings(url, xai):
    assert _host_match(url, "x.ai") is xai


@pytest.mark.parametrize(
    "url",
    ["https://api.x.ai/v1", "https://api.x.ai/v1/", "https://api.x.ai/v1/chat/completions",
     "https://api.x.ai/v1/responses"],
)
def test_normalize_base_strips_dialect_suffixes_to_the_api_base(url):
    assert normalize_base(url) == "https://api.x.ai/v1"


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


@pytest.mark.asyncio
async def test_maybe_compact_without_persist_returns_system_summary_then_kept_half(monkeypatch):
    import src.context_compactor as context_compactor

    async def summarize(*args, **kwargs):
        return "summary text"

    monkeypatch.setattr(context_compactor, "get_context_length", lambda url, model: 1_000)
    monkeypatch.setattr(context_compactor, "resolve_endpoint", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(context_compactor, "llm_call_async", summarize)
    monkeypatch.setattr(
        context_compactor, "_update_session_history",
        lambda *a, **k: pytest.fail("persist=False must not write session history"),
    )
    system = {"role": "system", "content": "sys"}
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i} " + "w " * 400}
             for i in range(6)]
    state = {}

    result, _, compacted = await context_compactor.maybe_compact(
        object(), "http://h/v1", "m", [system, *turns], persist=False, compaction_state=state,
    )

    assert compacted is True
    assert result[0] is system
    assert result[1]["role"] == "system" and "summary text" in result[1]["content"]
    assert result[2:] == turns[state["split_point"]:]


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


@pytest.mark.asyncio
async def test_finish_flag_belongs_to_the_running_run_and_a_replacement_starts_clear():
    import asyncio

    from src import agent_runs

    release = asyncio.Event()

    async def waiting_run():
        await release.wait()
        yield "data: [DONE]\n\n"

    session = "pin-finish-session"
    first = agent_runs.start(session, waiting_run())
    await asyncio.sleep(0)
    assert not agent_runs.should_finish(session)
    assert agent_runs.request_finish(session, first.run_id)
    assert agent_runs.should_finish(session)

    second = agent_runs.start(session, waiting_run())
    await asyncio.sleep(0)
    assert not agent_runs.should_finish(session)

    release.set()
    await second.task
    assert not agent_runs.should_finish(session)


# --- Turn context helpers (ticket #59) ---


def test_empty_plan_workspace_and_uploads_add_nothing():
    from src.agent_loop import _uploaded_files_context_message, _workspace_coding_rules, build_active_plan_note

    assert build_active_plan_note("  ") == ""
    assert _workspace_coding_rules(None) == ""
    assert _uploaded_files_context_message([]) is None


def test_backend_runtime_context_is_untrusted_or_absent():
    from src.agent_loop import _backend_runtime_context_message

    message = _backend_runtime_context_message()
    assert message is None or (message["role"] == "user" and message["metadata"]["trusted"] is False)


def test_compact_email_draft_keeps_headers_and_the_users_text():
    from src.agent_loop import _compact_email_draft_context

    raw = ("To: bob@example.com\nSubject: Re: hi\n---\nThanks!\n\n"
           "---------- Previous message ----------\n" + "x" * 5000)
    compact = _compact_email_draft_context(raw)
    assert compact.startswith("To: bob@example.com\nSubject: Re: hi\n---\nThanks!")
    assert len(compact) < len(raw)


def test_email_reader_message_needs_a_uid():
    from src.clean_agent_preview import active_email_context_message

    assert active_email_context_message({"folder": "INBOX"}) is None
    message = active_email_context_message({"uid": "9", "account": "work"})
    assert message["metadata"]["trusted"] is False and "Message UID: 9" in message["content"]


def test_integrations_prompt_is_empty_without_enabled_integrations(monkeypatch):
    import src.integrations as integrations

    monkeypatch.setattr(integrations, "load_integrations", lambda: [{"id": "a", "enabled": False}])
    assert integrations.get_integrations_prompt() == ""


def test_pdf_source_markers_mark_form_documents():
    from src.pdf_form_doc import find_source_upload_id

    upload = "b" * 32
    assert find_source_upload_id(f'<!-- pdf_form_source upload_id="{upload}" -->') == upload
    assert find_source_upload_id(f'<!-- pdf_source upload_id="{upload}" -->') == upload
    assert find_source_upload_id("plain text") is None


def test_guide_only_detection_reads_the_users_own_words():
    from src.tool_policy import detect_guide_only_turn

    assert detect_guide_only_turn("don't use tools, just explain")
    assert detect_guide_only_turn("use the tools you need") is None


def test_visual_evidence_is_off_only_for_text_only_qwen_routers():
    from src.agent_loop import _allow_visual_tool_evidence_for_model

    assert _allow_visual_tool_evidence_for_model("x-ai/grok-4.1-fast")
    assert not _allow_visual_tool_evidence_for_model("qwen38-tool-router")


def test_linkify_helpers_link_titles_from_successful_tool_events():
    from src.chat_agent.transcript import linkify

    events = [
        {"tool": "mcp", "command": "mcp__email__list_emails {}", "exit_code": 0,
         "output": "1. **Lunch plans**\n   UID: 77"},
        {"tool": "manage_calendar", "exit_code": 0, "output": "",
         "events": [{"uid": "ev-1", "summary": "Dentist visit"}]},
        {"tool": "manage_notes", "exit_code": 1, "error": True, "note_id": "n1", "note_title": "Failed note"},
    ]
    text = linkify("Lunch plans\nDentist visit\nFailed note", events)
    assert text == "[Lunch plans](#email-77)\n[Dentist visit](#event-ev-1)\nFailed note"
