"""Per-user store of the chat_agent runtime: one JSON file per user.

The file lives at ``CHAT_AGENT_DIR/<percent-encoded lowercased user>.json``.
It holds the runtime switch (``chat_runtime``) and the user-level
instructions toggle (``instructions_enabled``). The user's instruction files
live in ``INSTRUCTIONS_DIR/<same encoded name>/``. User rename and delete move
or remove both (``rename_owner`` / ``delete_owner``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from urllib.parse import quote

from core.atomic_io import atomic_write_json
from src import constants
# Relies on: confine(allow_root=False) raises ValueError for the root itself
# and for anything that resolves outside it (symlinks followed).
from src.path_confinement import confine

logger = logging.getLogger(__name__)

RUNTIME_LEGACY = "legacy"
RUNTIME_NEW = "new"
CHAT_RUNTIME_VALUES = frozenset({RUNTIME_LEGACY, RUNTIME_NEW})


# The Settings Instructions panel edits this file; CLAUDE.md beside it loads too.
USER_INSTRUCTIONS_FILENAME = "AGENTS.md"


def _owner_name(owner: str) -> str:
    # strip().lower() is how AuthManager keys usernames; casefold() would merge
    # distinct accounts such as "straße" and "strasse" into one folder.
    name = quote(str(owner or "").strip().lower(), safe="")
    if not name:
        raise ValueError("chat_agent store needs an owner")
    return name


def store_path(owner: str) -> str:
    """Return the owner's store file, confined to ``CHAT_AGENT_DIR``."""
    name = _owner_name(owner)
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


def instructions_enabled(owner: str) -> bool:
    return load(owner).get("instructions_enabled", True) is not False


def set_instructions_enabled(owner: str, enabled: bool) -> None:
    data = load(owner)
    data["instructions_enabled"] = bool(enabled)
    atomic_write_json(store_path(owner), data, indent=2)


def instructions_dir(owner: str) -> str:
    """Return the owner's instruction folder, confined to ``INSTRUCTIONS_DIR``."""
    return confine(constants.INSTRUCTIONS_DIR, _owner_name(owner), allow_root=False)


def instructions_display_dir(owner: str) -> str:
    """The folder as shown to the user and the model: no server data path."""
    return os.path.join(os.path.basename(constants.INSTRUCTIONS_DIR), _owner_name(owner))


def user_instructions_path(owner: str) -> str:
    return os.path.join(instructions_dir(owner), USER_INSTRUCTIONS_FILENAME)


def rename_owner(old: str, new: str) -> None:
    # Whatever already sits under the new name belongs to no live account
    # (auth refuses taken names), so it is dropped, never inherited.
    for path_for in (store_path, instructions_dir):
        source, target = path_for(old), path_for(new)
        if source == target:
            continue
        _remove(target)
        if os.path.lexists(source):
            os.rename(source, target)


def delete_owner(owner: str) -> None:
    _remove(store_path(owner))
    _remove(instructions_dir(owner))


def _remove(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)
