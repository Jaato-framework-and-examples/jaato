"""Daemon-owned ``wake_ref → binding`` registry for the mode-B wake ingress.

A binding is a SESSION's declaration: *"wake me (this ``session_id``, in this
``workspace_path``) about ``wake_ref``, trusting a signature from any of
``trust_keys``."*  It is written via the owner-guarded ``bind_wake`` command
(which runs AS the caller's session, so a caller can only bind ITSELF —
hijack-proof by construction) and resolved by the Stage-B verify shim.

**Ownership line** (the load-bearing invariant of the wake primitive):

- ``workspace`` / sandbox root is **SERVER-owned** — never caller-supplied
  (caller-supplied = sandbox escape).  It comes from the persisted session
  record via ``SessionWorkspaceIndex`` (#516).
- ``wake_ref`` + ``trust_keys`` are **SESSION-owned** — the callee's contract
  with its caller.  This registry holds that session-owned half and never lets
  one session overwrite another's binding.

So the daemon owns what protects the session *from* the caller (sandbox); the
session owns what invites the caller *in* (``wake_ref`` + ``trust_keys``).

**``wake_ref``** is an OPAQUE, session-supplied routing string (no
daemon-minting — the daemon persists/routes/revives but authors nothing).
Source-namespaced by convention (``<source>:<ref>`` — e.g.
``github-pr:owner/repo#42``) to avoid cross-source collisions, but treated as
opaque here.

**``trust_keys``** is a bounded SET (rotation overlap → ``[old, new]``; the
shim OR-verifies a signature against any of them).  Every key is validated as a
PEM public key at bind time (:func:`wake_verify.validate_trust_keys`).

**TTL**: each binding carries an expiry — a safety net for a forgotten
``unbind_wake`` (the clean path is an explicit unbind when the matter closes).
An expired binding is treated as absent: it stops resolving AND a new owner may
re-bind the same ``wake_ref``.

**Squat is denial, not hijack**: a rogue pre-binding a guessable ``wake_ref``
denies the legitimate owner its binding (the owner's ``bind_wake`` is rejected
— a signal it sees), but a real store-signed wake for that ref resolves to the
squatter's session and fails signature verification (the squatter cannot sign
as the store or forge).  A session that wants to remove even that denial risk
authors an unguessable (random-UUID) ``wake_ref``.

Persisted under ``~/.jaato/`` (durable across restart/reboot — a wake may
arrive days after the bind).
"""
from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .wake_verify import InvalidTrustKey, validate_trust_keys

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = pathlib.Path.home() / ".jaato" / "wake_bindings.json"
# Upper bound on trust_keys per binding — DoS hygiene so a pathological binding
# can't force many signature verifications per wake.  Rotation needs 2; a few
# co-signers is the realistic ceiling.
_MAX_TRUST_KEYS = 8
# Default binding lifetime (safety net for a forgotten unbind).  Generous: a
# review/wake can arrive days after the bind.  Overridable per bind_wake.
_DEFAULT_TTL_SECONDS = 30 * 24 * 3600


class BindOutcome(str, Enum):
    """Structured result of :meth:`WakeBindingRegistry.bind` / ``unbind``.

    Callers route on this enum (an HTTP/command layer maps it to a status /
    event), not a prose string.
    """
    OK = "ok"
    NO_SESSION = "no_session"        # caller has no active session to bind
    NO_KEYS = "no_keys"              # bind with an empty trust_keys set
    TOO_MANY_KEYS = "too_many_keys"  # trust_keys exceeds the cap
    MALFORMED_KEY = "malformed_key"  # a trust_key is not a valid PEM public key
    UNAUTHORIZED = "unauthorized"    # wake_ref owned by a DIFFERENT session
    UNKNOWN = "unknown"              # unbind of a nonexistent / expired wake_ref

    @property
    def is_ok(self) -> bool:
        return self is BindOutcome.OK


@dataclass
class WakeBinding:
    """A resolved, non-expired binding.  ``trust_keys`` are PEM public keys the
    Stage-B verifier OR-checks a wake signature against."""
    session_id: str
    workspace_path: str
    trust_keys: List[str]
    expires_at: float


class WakeBindingRegistry:
    """Thread-safe, disk-backed ``wake_ref → WakeBinding`` map with an
    owner-guarded idempotent upsert and TTL expiry.

    Lifecycle: constructed once by :class:`SessionManager`; written by the
    ``bind_wake`` / ``unbind_wake`` commands (owner = the caller's session);
    read by the Stage-B verify shim (``resolve``).
    """

    def __init__(
        self,
        path: Optional[pathlib.Path] = None,
        max_keys: int = _MAX_TRUST_KEYS,
        default_ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._path = pathlib.Path(path) if path is not None else _DEFAULT_REGISTRY_PATH
        self._max_keys = max_keys
        self._default_ttl = default_ttl_seconds
        self._lock = threading.Lock()
        self._bindings: Dict[str, WakeBinding] = {}
        self._load()

    # ---- persistence --------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning(
                "wake-binding registry unreadable at %s: %s — starting empty",
                self._path, exc)
            return
        if not isinstance(raw, dict):
            return
        for wake_ref, b in raw.items():
            if not isinstance(b, dict):
                continue
            try:
                self._bindings[str(wake_ref)] = WakeBinding(
                    session_id=str(b["session_id"]),
                    workspace_path=str(b["workspace_path"]),
                    trust_keys=[str(k) for k in b["trust_keys"]],
                    expires_at=float(b["expires_at"]),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "wake-binding registry: skipping malformed entry %r", wake_ref)

    def _save_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            payload = {
                ref: {
                    "session_id": b.session_id,
                    "workspace_path": b.workspace_path,
                    "trust_keys": b.trust_keys,
                    "expires_at": b.expires_at,
                }
                for ref, b in self._bindings.items()
            }
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:
            logger.warning(
                "wake-binding registry save failed at %s: %s", self._path, exc)

    # ---- operations ---------------------------------------------------------

    def bind(
        self,
        wake_ref: str,
        session_id: str,
        workspace_path: str,
        trust_keys: List[str],
        ttl_seconds: Optional[int] = None,
    ) -> BindOutcome:
        """Owner-guarded idempotent-upsert of ``wake_ref`` → this session.

        - ``wake_ref`` absent (or its binding expired) → create, owned by
          ``session_id``.
        - present and owned by ``session_id`` → refresh ``trust_keys`` + renew
          the TTL (the rotation path — re-call with ``[old, new]`` then
          ``[new]``).  ``session_id`` / ``workspace_path`` are NOT mutated.
        - present and owned by a DIFFERENT (live) session → ``UNAUTHORIZED``
          (the hijack / squat guard).
        """
        if not session_id or not workspace_path:
            return BindOutcome.NO_SESSION
        if not wake_ref:
            return BindOutcome.UNKNOWN
        if not trust_keys:
            return BindOutcome.NO_KEYS
        if len(trust_keys) > self._max_keys:
            return BindOutcome.TOO_MANY_KEYS
        try:
            validate_trust_keys(trust_keys)
        except InvalidTrustKey as exc:
            logger.info("bind_wake rejected malformed trust_key for %r: %s",
                        wake_ref, exc)
            return BindOutcome.MALFORMED_KEY

        now = time.time()
        with self._lock:
            existing = self._bindings.get(wake_ref)
            if (existing is not None
                    and existing.expires_at > now
                    and existing.session_id != session_id):
                # Live binding owned by another session — refuse (hijack/squat).
                return BindOutcome.UNAUTHORIZED
            ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
            self._bindings[wake_ref] = WakeBinding(
                session_id=session_id,
                workspace_path=workspace_path,
                trust_keys=list(trust_keys),
                expires_at=now + ttl,
            )
            self._save_locked()
        return BindOutcome.OK

    def unbind(self, wake_ref: str, session_id: str) -> BindOutcome:
        """Remove ``wake_ref`` — owner-guarded.

        ``UNKNOWN`` if absent/expired; ``UNAUTHORIZED`` if owned by a different
        session; ``OK`` on removal.  (The clean path on a matter's close; the
        TTL covers the forgotten case.)
        """
        now = time.time()
        with self._lock:
            existing = self._bindings.get(wake_ref)
            if existing is None or existing.expires_at <= now:
                return BindOutcome.UNKNOWN
            if existing.session_id != session_id:
                return BindOutcome.UNAUTHORIZED
            del self._bindings[wake_ref]
            self._save_locked()
        return BindOutcome.OK

    def resolve(self, wake_ref: str) -> Optional[WakeBinding]:
        """Return the live binding for ``wake_ref``, or ``None`` if absent or
        expired.  The Stage-B shim calls this, then verifies the wake signature
        against ``binding.trust_keys`` and drives ``binding.session_id``."""
        now = time.time()
        with self._lock:
            b = self._bindings.get(wake_ref)
            if b is None:
                return None
            if b.expires_at <= now:
                # Prune opportunistically so an expired ref frees its name.
                del self._bindings[wake_ref]
                self._save_locked()
                return None
            return b
