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

**Runner identity (#812).**  The index is also the one daemon-owned,
cross-workspace place an operator can look up a session id, so it carries a
second, independent section: ``identity``, mapping session id to the
``RunnerIdentity`` dict (runner pid, pool slot, cascade, AppArmor profile).
#812 reports an operator who found a session here, got its workspace, and had
nothing to act on — no runner, pid or slot anywhere in the index or the record.

The two sections are deliberately independent maps rather than one map of
richer values:

* ``map`` keeps its exact pre-#812 shape, so a file written by this daemon is
  still read correctly by an older one (which ignores the new key) and a file
  written by an older one loads here with no identities;
* an ambiguous id is refused for WAKE (waking the wrong session is
  unrecoverable) and still yields its identity, because two sessions sharing a
  timestamp is not a reason to withhold diagnostic information from an
  operator — the identity says which process, and the ambiguity is about which
  workspace.

**Group membership (session group messaging).**  A third independent section,
``membership``, maps session id to the two facts the group predicate
(``server.session_groups``) reads -- ``created_by`` and ``cascade_driver_id``
-- plus the cascade-scoped ``sibling_name``.  It is what lets a peer resolve a
COLD session across workspaces: "which sessions does user X own" and "which
sessions carry cascade Y" are questions a per-workspace listing cannot
answer, and the index is the one daemon-owned, cross-workspace place a
session id is already recorded.  Written from ``_save_session`` beside the
workspace mapping, so the two cannot disagree about which sessions exist.
A file written before the section existed loads with no memberships: a cold
session then joins no group until its next save, which is the safe
direction (a message to it is refused, never misrouted).
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


_MEMBERSHIP_FIELDS = ("created_by", "cascade_driver_id", "sibling_name")


def _membership_row(raw: Dict[str, object]) -> Dict[str, Optional[str]]:
    """Normalise a membership row to exactly the three known fields, each a
    non-empty string or ``None``.  An empty string is an absent fact, the
    same reading the group predicate gives it."""
    row: Dict[str, Optional[str]] = {}
    for key in _MEMBERSHIP_FIELDS:
        value = raw.get(key)
        row[key] = str(value) if isinstance(value, str) and value else None
    return row


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
        #: ``session_id -> RunnerIdentity.to_dict()`` (#812).  Independent of
        #: ``_map``: written by :meth:`record_identity`, read by
        #: :meth:`identity`, and never consulted by :meth:`resolve`.
        self._identity: Dict[str, Dict[str, object]] = {}
        #: ``session_id -> {"created_by", "cascade_driver_id", "sibling_name"}``
        #: (session group messaging).  Independent of ``_map`` and
        #: ``_identity``: written by :meth:`record_membership`, read by
        #: :meth:`membership` / :meth:`members`, never consulted by
        #: :meth:`resolve`.
        self._membership: Dict[str, Dict[str, Optional[str]]] = {}
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
        identity = raw.get("identity", {})
        if isinstance(mapping, dict):
            self._map = {
                str(k): str(v) for k, v in mapping.items() if isinstance(v, str)
            }
        if isinstance(ambiguous, list):
            self._ambiguous = {str(x) for x in ambiguous}
        # Absent on every file written before #812 -- an index with no
        # identities behaves exactly as it did, which is what makes this
        # section additive rather than a format change.
        if isinstance(identity, dict):
            self._identity = {
                str(k): v for k, v in identity.items() if isinstance(v, dict)
            }
        membership = raw.get("membership", {})
        if isinstance(membership, dict):
            self._membership = {
                str(k): _membership_row(v)
                for k, v in membership.items() if isinstance(v, dict)
            }

    def _save_locked(self) -> None:
        """Atomically persist (caller holds ``self._lock``)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(
                json.dumps({
                    "map": self._map,
                    "ambiguous": sorted(self._ambiguous),
                    "identity": self._identity,
                    "membership": self._membership,
                }),
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
            # Unchanged mapping — no state change, so skip the disk write.
            # ``record`` runs from ``_save_session`` on every turn, so for a
            # stable session this would otherwise rewrite the file each save.
            if existing == workspace_path:
                return
            if existing is not None and existing != workspace_path:
                if session_id not in self._ambiguous:
                    logger.warning(
                        "session-workspace index: id %s seen under two workspaces "
                        "(%s vs %s) — marking AMBIGUOUS; wake-by-id will refuse it",
                        session_id, existing, workspace_path)
                self._ambiguous.add(session_id)
            self._map[session_id] = workspace_path
            self._save_locked()

    def record_identity(
        self, session_id: str, identity: Optional[Dict[str, object]],
    ) -> None:
        """Record which process is executing ``session_id`` (#812).

        Independent of :meth:`record`: a session's workspace never changes
        while its runner does (a revive, or a pool slot handed on), so the two
        are written separately and an identity update does not touch the
        workspace map or its ambiguity marks.

        Skips the disk write when nothing changed — this runs from
        ``_save_session`` on every turn, and a stable session would otherwise
        rewrite the file each save (the same reasoning :meth:`record` gives).

        Args:
            session_id: The session.
            identity: ``RunnerIdentity.to_dict()``, or ``None`` to drop the
                entry (a session that no longer has a runner).
        """
        if not session_id:
            return
        with self._lock:
            existing = self._identity.get(session_id)
            if identity is None:
                if existing is None:
                    return
                self._identity.pop(session_id, None)
            else:
                if existing == identity:
                    return
                self._identity[session_id] = dict(identity)
            self._save_locked()

    def identity(self, session_id: str) -> Optional[Dict[str, object]]:
        """Return the recorded runner identity for ``session_id`` (#812).

        Unlike :meth:`resolve`, an AMBIGUOUS id is **not** refused: ambiguity
        is about which workspace a timestamp-colliding id belongs to, and
        refusing to say which process ran it would withhold exactly the
        diagnostic #812 was missing.  Nothing acts on this value — it names a
        process for an operator, and the daemon stops a session by id, never
        by pid.

        Args:
            session_id: The session to look up.

        Returns:
            The stored dict (a copy), or ``None`` when unknown.
        """
        with self._lock:
            found = self._identity.get(session_id)
            return dict(found) if found is not None else None

    def identities(self) -> Dict[str, Dict[str, object]]:
        """Snapshot of every recorded runner identity.

        Returns:
            A fresh ``session_id -> identity dict`` map, safe to iterate
            outside the lock.
        """
        with self._lock:
            return {k: dict(v) for k, v in self._identity.items()}

    def record_membership(
        self,
        session_id: str,
        *,
        created_by: Optional[str],
        cascade_driver_id: Optional[str],
        sibling_name: Optional[str],
    ) -> None:
        """Record the group facts for ``session_id`` (session group messaging).

        Independent of :meth:`record` and :meth:`record_identity`, and like
        them it skips the disk write when nothing changed -- this runs from
        ``_save_session`` on every turn.

        The facts are recorded even when all three are ``None``: a row that
        says "this session has no owner and no cascade" is the answer a peer
        resolving it needs (it joins no group, so the message is refused by
        name), where an ABSENT row is indistinguishable from an index written
        before the section existed.

        Args:
            session_id: The session.
            created_by: Its authenticated creator (record 2.9), or ``None``.
            cascade_driver_id: Its cascade (record 2.10), or ``None``.
            sibling_name: Its cascade-scoped address, or ``None``.
        """
        if not session_id:
            return
        row = _membership_row({
            "created_by": created_by,
            "cascade_driver_id": cascade_driver_id,
            "sibling_name": sibling_name,
        })
        with self._lock:
            if self._membership.get(session_id) == row:
                return
            self._membership[session_id] = row
            self._save_locked()

    def membership(self, session_id: str) -> Optional[Dict[str, Optional[str]]]:
        """The recorded group facts for ``session_id``, or ``None`` when the
        index holds no row for it.

        An AMBIGUOUS id is answered too: ambiguity is about which WORKSPACE
        a colliding id belongs to, and :meth:`resolve` already refuses the
        wake.  A caller that needs the workspace as well asks it separately
        and gets the refusal there.
        """
        with self._lock:
            row = self._membership.get(session_id)
            return dict(row) if row is not None else None

    def members(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Snapshot of every recorded membership row, keyed by session id.

        A fresh copy, safe to iterate outside the lock.  The caller filters
        by group key (``session_groups.group_keys_for``) -- the index stores
        facts, not a derived grouping, so a change to the predicate needs no
        index migration.
        """
        with self._lock:
            return {k: dict(v) for k, v in self._membership.items()}

    def forget(self, session_id: str) -> None:
        """Drop any mapping (and ambiguity mark) for ``session_id``.

        Also drops the #812 runner identity.  Called when a session is
        DELETED so its id→workspace entry doesn't outlive the session — a stale entry is otherwise only reaped never
        (the index has no TTL), and a later second-granularity id collision
        would resolve against a workspace for a session that no longer exists.
        Idempotent — a no-op for an unknown id.
        """
        if not session_id:
            return
        with self._lock:
            present = (
                session_id in self._map
                or session_id in self._ambiguous
                or session_id in self._identity
                or session_id in self._membership
            )
            self._map.pop(session_id, None)
            self._ambiguous.discard(session_id)
            # A deleted session is in no group: leaving the row would let a
            # later timestamp collision inherit a dead session's owner.
            self._membership.pop(session_id, None)
            # The identity goes with the mapping: a deleted session's pid is
            # not evidence about anything, and leaving it would let a later
            # timestamp collision inherit a dead process's record.
            self._identity.pop(session_id, None)
            if present:
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

    def workspaces(self) -> Set[str]:
        """Return the distinct workspace paths this index knows about.

        Lets ``SessionManager.list_sessions`` include the workspaces of
        cold, persisted sessions that no in-memory session or attached
        client currently references — the WS per-session-provisioned
        ``ws_<hash>`` dirs a workspace-pinless client can't surface on its
        own.  A snapshot copy (safe to iterate outside the lock).
        """
        with self._lock:
            return set(self._map.values())
