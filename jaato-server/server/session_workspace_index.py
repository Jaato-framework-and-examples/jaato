"""Daemon-owned ``session_id → workspace_path`` index for the wake primitive.

The daemon is workspace-agnostic and serves sessions across many workspaces,
but a session record lives at ``<workspace>/.jaato/sessions/<id>.json`` — so
locating a COLD (unloaded) session's record requires knowing its workspace.
A LOADED session carries ``workspace_path`` on its in-memory ``Session``
object; a cold one does not.

This index closes that gap for :meth:`SessionManager.wake_session`: a caller
wakes a session by ``session_id`` alone and the daemon resolves the workspace
needed to revive it — WITHOUT the caller supplying any path.  That is a
deliberate security property: an authenticated-but-untrusted wake caller (e.g.
an HTTP shim in front of a public PR-review webhook) must not be able to point
revival at a weaker sandbox root.  Workspace resolution stays server-owned;
the sandbox root itself always comes from the persisted record.

**Collision handling (fail-loud, never wrong).**  Session ids are
second-granularity timestamps (``YYYYMMDD_HHMMSS`` — see
``shared.plugins.session.file_session.generate_session_id``) minted
per-workspace, so two sessions started the same second in different workspaces
share an id.  A flat id→workspace map is then ambiguous.  Rather than guess
(and risk waking the WRONG session), an id observed under more than one
workspace is marked AMBIGUOUS and :meth:`resolve` refuses it.  (Follow-up: add
entropy to ``generate_session_id`` to remove this ambiguity class entirely; an
ambiguous id stays un-wakeable-by-id until then — the conservative, safe
behavior.)

**Durability.**  Persisted to a daemon-owned location (``~/.jaato/`` by
default, the same home as ``ws.token`` and the apparmor cache) so wake survives
a daemon restart or a reboot — the "wake me hours/days later" case that the
runner-bound webhook listener cannot serve.
"""
from __future__ import annotations

import json
import logging
import pathlib
import threading
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# Durable, daemon-owned default location (mirrors ~/.jaato/ws.token).
_DEFAULT_INDEX_PATH = pathlib.Path.home() / ".jaato" / "session_workspace_index.json"


class SessionWorkspaceIndex:
    """Thread-safe, disk-backed ``session_id → workspace_path`` map that
    fails loud on cross-workspace id collisions.

    Lifecycle: constructed once by :class:`SessionManager`, updated on every
    ``_save_session`` (authoritative ``session.workspace_path``), queried by
    :meth:`SessionManager.wake_session` for cold sessions only (a loaded
    session's workspace comes straight off the ``Session`` object).
    """

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        self._path = pathlib.Path(path) if path is not None else _DEFAULT_INDEX_PATH
        self._lock = threading.Lock()
        self._map: Dict[str, str] = {}
        self._ambiguous: Set[str] = set()
        self._load()

    def _load(self) -> None:
        """Populate from disk.  A missing file is normal (fresh daemon); a
        corrupt/unreadable one must NOT break startup — it rebuilds as sessions
        are saved, and a wake of a not-yet-re-saved cold session fails loud
        until then."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning(
                "session-workspace index unreadable at %s: %s — starting empty",
                self._path, exc)
            return
        if not isinstance(raw, dict):
            return
        mapping = raw.get("map", {})
        ambiguous = raw.get("ambiguous", [])
        if isinstance(mapping, dict):
            self._map = {
                str(k): str(v) for k, v in mapping.items() if isinstance(v, str)
            }
        if isinstance(ambiguous, list):
            self._ambiguous = {str(x) for x in ambiguous}

    def _save_locked(self) -> None:
        """Atomically persist (caller holds ``self._lock``)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(
                json.dumps({"map": self._map, "ambiguous": sorted(self._ambiguous)}),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except OSError as exc:
            logger.warning(
                "session-workspace index save failed at %s: %s", self._path, exc)

    def record(self, session_id: str, workspace_path: str) -> None:
        """Record ``session_id → workspace_path``.

        If ``session_id`` is already mapped to a DIFFERENT workspace, mark it
        ambiguous (a cross-workspace id collision) so :meth:`resolve` refuses it
        rather than pick wrong.
        """
        if not session_id or not workspace_path:
            return
        with self._lock:
            existing = self._map.get(session_id)
            if existing is not None and existing != workspace_path:
                if session_id not in self._ambiguous:
                    logger.warning(
                        "session-workspace index: id %s seen under two workspaces "
                        "(%s vs %s) — marking AMBIGUOUS; wake-by-id will refuse it",
                        session_id, existing, workspace_path)
                self._ambiguous.add(session_id)
            self._map[session_id] = workspace_path
            self._save_locked()

    def resolve(self, session_id: str) -> Optional[str]:
        """Return the workspace for ``session_id``, or ``None`` if unknown or
        AMBIGUOUS.

        An ambiguous id (colliding across workspaces) is refused rather than
        guessed — the caller then fails loud instead of waking the wrong
        session.
        """
        with self._lock:
            if session_id in self._ambiguous:
                logger.warning(
                    "session-workspace index: refusing to resolve ambiguous id %s "
                    "(collides across workspaces)", session_id)
                return None
            return self._map.get(session_id)
