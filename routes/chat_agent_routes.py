"""Control-plane routes of the chat_agent runtime (spec #44)."""

import os

from fastapi import APIRouter, HTTPException, Request

from core.atomic_io import atomic_write_text
from src import constants
from src.agent_runtime.authority import is_internal_tool_request
from src.auth_helpers import require_user
from src.chat_agent import store
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


def _settings(user: str) -> dict:
    available = not auth_disabled()
    runtime = store.chat_runtime(user) if available else store.RUNTIME_LEGACY
    return {"chat_runtime": runtime, "available": available}


def setup_chat_agent_routes() -> APIRouter:
    router = APIRouter(prefix="/api/chat_agent", tags=["chat_agent"])

    @router.get("/settings")
    async def get_settings(request: Request) -> dict:
        return _settings(require_user(request))

    @router.put("/settings")
    async def put_settings(request: Request, body: dict) -> dict:
        user = require_interactive_user(request)
        value = body.get("chat_runtime")
        if value not in store.CHAT_RUNTIME_VALUES:
            raise HTTPException(400, "chat_runtime must be 'legacy' or 'new'.")
        store.set_chat_runtime(user, value)
        return _settings(user)

    @router.get("/instructions")
    async def get_instructions(request: Request) -> dict:
        if auth_disabled():
            return {"available": False}
        return _instructions(require_user(request))

    @router.put("/instructions")
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
