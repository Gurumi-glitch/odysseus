"""Per-user store of the chat_agent runtime: one JSON file per user.

The file lives at ``CHAT_AGENT_DIR/<percent-encoded casefolded user>.json``.
This ticket stores only the runtime switch (``chat_runtime``); permission
rules, session rules and the rename/delete hooks join it in later tickets.
"""

from __future__ import annotations

import json
import logging
import os
from urllib.parse import quote

from core.atomic_io import atomic_write_json
from src import constants

logger = logging.getLogger(__name__)

RUNTIME_LEGACY = "legacy"
RUNTIME_NEW = "new"
CHAT_RUNTIME_VALUES = frozenset({RUNTIME_LEGACY, RUNTIME_NEW})


def store_path(owner: str) -> str:
    """Return the owner's store file, confined to ``CHAT_AGENT_DIR``."""
    name = quote(str(owner or "").strip().casefold(), safe="")
    if not name:
        raise ValueError("chat_agent store needs an owner")
    root = os.path.realpath(constants.CHAT_AGENT_DIR)
    path = os.path.realpath(os.path.join(root, name + ".json"))
    if os.path.dirname(path) != root:
        raise ValueError("chat_agent store path escapes its directory")
    return path


def load(owner: str) -> dict:
    try:
        with open(store_path(owner), encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # An unreadable store must not break chat; the switch reads as off.
        logger.warning("chat_agent store unreadable for owner: %s", type(exc).__name__)
        return {}
    return data if isinstance(data, dict) else {}


def chat_runtime(owner: str | None) -> str:
    if not owner:
        return RUNTIME_LEGACY
    value = load(owner).get("chat_runtime")
    return value if value in CHAT_RUNTIME_VALUES else RUNTIME_LEGACY


def set_chat_runtime(owner: str, value: str) -> None:
    if value not in CHAT_RUNTIME_VALUES:
        raise ValueError(f"chat_runtime must be one of {sorted(CHAT_RUNTIME_VALUES)}")
    data = load(owner)
    data["chat_runtime"] = value
    atomic_write_json(store_path(owner), data, indent=2)
