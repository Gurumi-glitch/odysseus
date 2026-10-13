"""Control-plane routes of the chat_agent runtime (spec #44)."""

import os

from fastapi import APIRouter, HTTPException, Request

from core.atomic_io import atomic_write_text
from core.exceptions import SessionNotFoundError
from routes.chat_helpers import _enforce_chat_privileges
from routes.chat_routes import _resolve_selected_endpoint
from routes.session_routes import _verify_session_owner
from src import constants
from src.agent_runtime.authority import is_internal_tool_request
from src.auth_helpers import effective_user, is_delegated_credential, require_user
from src.chat_agent import asks, classifier, permissions, routing, store
from src.owner_identity import auth_disabled


def require_interactive_user(request: Request) -> str:
    """Only a signed-in person may change the runtime's control plane.

    ``require_user`` already refuses bearer tokens (delegated credentials);
    internal-tool loopback impersonates the owner, so it is refused here.
    """
    if auth_disabled():
        raise HTTPException(403, "The chat runtime switch needs authentication enabled.")
    if is_internal_tool_request(request):
        raise HTTPException(403, "Tool requests cannot change the chat runtime.")
    user = require_user(request)
    if not user:
        raise HTTPException(403, "The chat runtime switch needs a signed-in user.")
    return user


def _require_interactive_answerer(request: Request, action: str) -> None:
    """Asks and steering speak for the person at the keyboard: tool loopback
    and API tokens act for them without being them (spec #44, Asks).

    Deliberately not ``require_interactive_user``: this one lets auth-disabled
    and empty-user callers through to the owner check, and words its refusal
    per action. The outcome table is pinned in test_chat_agent_ask_steer_authority.
    """
    if is_internal_tool_request(request) or is_delegated_credential(request):
        raise HTTPException(403, f"Only you, in an interactive session, can {action}.")


def _settings(user: str, session_id: str = "") -> dict:
    available = not auth_disabled()
    runtime = store.chat_runtime(user) if available else store.RUNTIME_LEGACY
    settings = {"chat_runtime": runtime, "available": available}
    if available:
        # No session_id (a new chat before its first message): the owner's
        # last-used values, which that chat inherits.
        for name in store.CHAT_SETTINGS:
            settings[name] = store.chat_setting(user, session_id, name)
    return settings


def setup_chat_agent_routes(session_manager) -> APIRouter:
    # Full paths, not a router prefix: ask/steer keep their /api/chat paths.
    router = APIRouter(tags=["chat_agent"])

    @router.get("/api/chat_agent/settings")
    async def get_settings(request: Request, session_id: str = "") -> dict:
        """The user's runtime switch and that chat's settings (``session_id``),
        else the owner's last-used ones."""
        return _settings(require_user(request), session_id)

    @router.put("/api/chat_agent/settings")
    async def put_settings(request: Request, body: dict) -> dict:
        user = require_interactive_user(request)
        chat_settings = {name: body[name] for name in store.CHAT_SETTINGS if name in body}
        if "chat_runtime" not in body and not chat_settings:
            raise HTTPException(400, "Nothing to change.")
        # Checked before the chat settings are written: a bad value changes nothing.
        if "chat_runtime" in body and body["chat_runtime"] not in store.CHAT_RUNTIME_VALUES:
            raise HTTPException(400, "chat_runtime must be 'legacy' or 'new'.")
        session_id = body.get("session_id") or ""
        if chat_settings:
            # The store is the owner's own file: a foreign chat id only keys
            # an entry no other user reads, so no owner check is needed.
            # Without one, only the owner's last-used values change.
            if not isinstance(session_id, str):
                raise HTTPException(400, "session_id must be a string.")
            try:
                store.set_chat_settings(user, session_id, chat_settings)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        if "chat_runtime" in body:
            store.set_chat_runtime(user, body["chat_runtime"])
        return _settings(user, session_id)

    @router.get("/api/chat_agent/capabilities/{session_id}")
    async def get_capabilities(request: Request, session_id: str) -> dict:
        """What the chat's next agent turn would run on, and the thinking
        levels its (endpoint, model) takes. The composer asks after a model
        change and after a turn that learned a rejection."""
        user = require_user(request)
        _verify_session_owner(request, session_id, session_manager)
        try:
            sess = session_manager.get_session(session_id)
        except (KeyError, SessionNotFoundError):
            raise HTTPException(404, "Session not found")
        return await routing.chat_capabilities(user, sess.endpoint_url or "", sess.model or "", sess.headers)

    @router.get("/api/chat_agent/capabilities")
    async def get_pending_capabilities(
        request: Request, model: str = "", endpoint_id: str = "", endpoint_url: str = "",
    ) -> dict:
        """The same answer for a new chat that has no session yet, keyed by
        the picker's (endpoint, model)."""
        user = require_user(request)
        # Relies on: the chat route's own resolver, so a pick is resolved (and
        # refused) exactly as the first message's selected_endpoint_* would be.
        resolved = (
            _resolve_selected_endpoint(user, endpoint_id, endpoint_url)
            if model and (endpoint_id or endpoint_url) else None
        )
        if not resolved:
            raise HTTPException(404, "Endpoint not found")
        url, headers, _ = resolved
        return await routing.chat_capabilities(user, url, model, headers)

    @router.get("/api/chat_agent/instructions")
    async def get_instructions(request: Request) -> dict:
        if auth_disabled():
            return {"available": False}
        return _instructions(require_user(request))

    @router.put("/api/chat_agent/instructions")
    async def put_instructions(request: Request, body: dict) -> dict:
        user = require_interactive_user(request)
        content = body.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise HTTPException(400, "content must be text.")
            if len(content) > constants.INSTRUCTION_FILE_MAX_CHARS:
                raise HTTPException(
                    400,
                    f"Instructions are limited to {constants.INSTRUCTION_FILE_MAX_CHARS} characters.",
                )
            atomic_write_text(store.user_instructions_path(user), content)
        if "enabled" in body:
            store.set_instructions_enabled(user, bool(body["enabled"]))
        return _instructions(user)

    @router.get("/api/chat_agent/auto_mode/{session_id}")
    async def get_auto_mode(request: Request, session_id: str) -> dict:
        """Whether auto mode is paused in this chat, for the composer icon on
        load; a running turn reports changes as ``auto_mode`` events (#68)."""
        if auth_disabled():
            return {"available": False}
        user = require_user(request)
        return {"available": True, "paused": classifier.paused_in(user, session_id)}

    @router.get("/api/chat_agent/rules")
    async def get_rules(request: Request) -> dict:
        if auth_disabled():
            return {"available": False}
        return {"available": True, "rules": store.user_rules(require_user(request))}

    @router.put("/api/chat_agent/rules")
    async def put_rules(request: Request, body: dict) -> dict:
        user = require_interactive_user(request)
        rules = body.get("rules")
        if not isinstance(rules, dict) or set(rules) - set(store.RULE_LISTS):
            raise HTTPException(400, f"rules must be an object with lists {', '.join(store.RULE_LISTS)}.")
        written = {}
        for name in store.RULE_LISTS:
            texts = rules.get(name) or []
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise HTTPException(400, f"rules.{name} must be a list of rule strings.")
            try:
                written[name] = list(dict.fromkeys(permissions.parse_rule(text).text for text in texts))
            except ValueError as exc:
                raise HTTPException(400, f"rules.{name}: {exc}") from exc
        store.set_user_rules(user, written)
        return {"available": True, "rules": store.user_rules(user)}

    @router.post("/api/chat/ask/{session_id}")
    async def chat_ask(request: Request, session_id: str, body: dict) -> dict:
        """Answer a chat_agent ask; the waiting turn continues (spec #44, Asks)."""
        _require_interactive_answerer(request, "answer an ask")
        _verify_session_owner(request, session_id)
        decision = str(body.get("decision") or "")
        if decision not in asks.ALL_DECISIONS:
            raise HTTPException(400, "decision must be once, chat, deny or answer.")
        text = body.get("text") or ""
        if not isinstance(text, str):
            raise HTTPException(400, "text must be a string.")
        answered = asks.answer(
            session_id, effective_user(request), str(body.get("ask_id") or ""), decision, text,
        )
        if not answered:
            raise HTTPException(409, "No open ask matches this answer.")
        return {"accepted": True}

    @router.post("/api/chat/steer/{session_id}")
    async def chat_steer(request: Request, session_id: str, body: dict) -> dict:
        """Queue a message for the running chat_agent turn; it joins at the
        next round boundary (spec #44, Asks, steering, finish)."""
        _require_interactive_answerer(request, "steer a turn")
        _verify_session_owner(request, session_id)
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, "text must be a non-empty string.")
        try:
            sess = session_manager.get_session(session_id)
        except (KeyError, SessionNotFoundError):
            raise HTTPException(404, "Session not found")
        _enforce_chat_privileges(request, sess)
        if not asks.steer(session_id, effective_user(request), text):
            raise HTTPException(409, "No running turn in this chat takes steering messages.")
        return {"accepted": True}

    return router


def _instructions(user: str) -> dict:
    path = store.user_instructions_path(user)
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except FileNotFoundError:
        content = ""
    return {
        "available": True,
        "enabled": store.instructions_enabled(user),
        "content": content,
        # The host path differs anyway, and the server's layout is not the
        # user's business.
        "path": os.path.join(store.instructions_display_dir(user), store.USER_INSTRUCTIONS_FILENAME),
        "max_chars": constants.INSTRUCTION_FILE_MAX_CHARS,
    }
