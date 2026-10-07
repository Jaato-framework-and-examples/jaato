"""Authenticating a session record the daemon wrote (#1529).

A session record lives at ``<workspace>/.jaato/sessions/<id>.json``, inside
the workspace, and so inside what a session's own code can write: the
workspace-wide ``rwkl`` grant covers it on an AppArmor host, the workspace
type covers it on an SELinux host, and under a dropping
``--runner-uid-policy`` the record is the runner account's own file (#1528).
A revive rebuilds the session from it, and several of the fields it reads
decide the boundary the revived session runs under: the profile snapshot
(plugins, plugin configs, AppArmor fragments, ``runtime_limits`` and its
seccomp knobs, the permission policy, secret ``env`` URIs), the saved
``always`` rules, the record's own ``sandbox_mode``, its workspace and
config root, and its owner.

This module is the half of the fix that lets the daemon tell its own record
from an edited one.  Every save the daemon makes is SEALED with an
HMAC-SHA256 under a key only the daemon holds; :func:`verify` answers
whether a record it reads back is byte-for-byte (canonically) what it
wrote.  The other half, what a revive does with a record that does not
verify, is :mod:`server.record_distrust`.

Why a MAC over the whole record rather than a list of trusted fields: the
security-relevant surface of a profile is not a closed list.  ``plugins``,
``plugin_configs.notebook.allow_uncontained_exec``,
``plugin_configs.cli.extra_paths``, ``plugin_configs.interactive_shell.
require_confinement`` and an MCP server table all widen a session, and a
per-field merge is the list that drifts the day someone adds a knob.  A seal
over everything the daemon wrote covers a field added later with no edit
here.

The key
-------
``~/.jaato/session-record.key`` in the DAEMON's home: 32 random bytes,
mode ``0600``, created on first use with ``O_EXCL``.  The runner cannot read
it on the hosts this protects:

* an AppArmor-confined runner is granted only the ``~/.jaato`` subtrees
  plugins declare (themes, agents, profiles, ...; #1465), never this file;
* a runner dropped to another uid (#1168) cannot read a ``0600`` file in
  the daemon account's home;
* an SELinux-confined runner reads no file of the daemon's home type.

On an UNCONFINED host whose runner shares the daemon's uid, the runner can
read the key, just as it can already edit the profile files and
``~/.jaato`` -- nothing here adds a boundary there, and nothing claims to.

A key that cannot be read or created (a read-only home, a file owned by
another account) makes every save UNSEALED and every record unverified:
revives then take the distrust path, which is always the narrower one.  It
is logged once at ERROR.  A key file readable by group or others is
tightened to ``0600`` (and said so) rather than refused, because refusing
would put every session of the daemon on the distrust path for a mode bit.

Canonical form
--------------
The MAC is computed over ``json.dumps(..., sort_keys=True,
separators=(",", ":"), ensure_ascii=False)`` of the record WITHOUT its
:data:`SEAL_FIELD`, after one JSON round-trip, so what is MAC-ed is exactly
what a reader parses back (tuples become lists, non-string keys become
strings, as on disk).  The seal itself is ``{"alg": "hmac-sha256",
"mac": "<hex>"}``.

Records written before this change carry no seal and read as unverified,
once: the first revive takes the distrust path and the save after it seals
the result.  An older daemon reading a sealed record ignores the extra key
(``deserialize_session_state`` reads named keys only), so no record
version bump is needed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pathlib
import stat
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: The record key carrying the seal.  Never MAC-ed itself.
SEAL_FIELD = "record_seal"

#: The one algorithm this module writes and accepts.
SEAL_ALG = "hmac-sha256"

#: File name of the key under the daemon's ``~/.jaato``.
KEY_FILE_NAME = "session-record.key"

_KEY_BYTES = 32

_lock = threading.Lock()
_cached: Dict[str, Optional[bytes]] = {}


def default_key_path() -> pathlib.Path:
    """``~/.jaato/session-record.key`` of the account running this process.

    Resolved per call (not at import) so a test that redirects ``HOME``
    gets its own key, and a daemon never shares a key with a test run.
    """
    return pathlib.Path.home() / ".jaato" / KEY_FILE_NAME


def _read_key(path: pathlib.Path) -> Optional[bytes]:
    """Read an existing key, tightening a group/other-readable mode."""
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        logger.error(
            "session record key %s is not a regular file; records will be "
            "written UNSEALED and every revive takes the distrust path "
            "(#1529)", path)
        return None
    if st.st_uid != os.geteuid():
        logger.error(
            "session record key %s is owned by uid %d, not this daemon "
            "(uid %d); records will be written UNSEALED and every revive "
            "takes the distrust path (#1529)", path, st.st_uid, os.geteuid())
        return None
    if st.st_mode & 0o077:
        logger.warning(
            "session record key %s had mode %o; tightened to 600 (#1529)",
            path, stat.S_IMODE(st.st_mode))
        os.chmod(path, 0o600)
    data = path.read_bytes()
    if len(data) < _KEY_BYTES:
        logger.error(
            "session record key %s is %d bytes, expected %d; records will be "
            "written UNSEALED (#1529)", path, len(data), _KEY_BYTES)
        return None
    return data


def _create_key(path: pathlib.Path) -> bytes:
    """Create a fresh key with ``O_EXCL``; a racing creator's key is read."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = os.urandom(_KEY_BYTES)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _read_key(path)
        if existing is None:
            raise OSError(f"session record key {path} is unusable")
        return existing
    with os.fdopen(fd, "wb") as f:
        f.write(key)
        f.flush()
        os.fsync(f.fileno())
    logger.info("created the session record key at %s (#1529)", path)
    return key


def load_key(path: Optional[pathlib.Path] = None) -> Optional[bytes]:
    """The daemon's record key, created on first use; ``None`` if unusable.

    Cached per path for the life of the process.  ``None`` is cached too,
    so an unusable key is reported once rather than on every save.
    """
    path = path or default_key_path()
    cache_key = str(path)
    with _lock:
        if cache_key in _cached:
            return _cached[cache_key]
        try:
            key = (_read_key(path) if os.path.lexists(path)
                   else _create_key(path))
        except OSError as exc:
            logger.error(
                "could not read or create the session record key %s (%s); "
                "records will be written UNSEALED and every revive takes the "
                "distrust path (#1529)", path, exc)
            key = None
        _cached[cache_key] = key
        return key


def reset_cache() -> None:
    """Forget cached keys (tests that redirect ``HOME``)."""
    with _lock:
        _cached.clear()


def _canonical(record: Dict[str, Any]) -> bytes:
    """The bytes the MAC covers: *record* minus its seal, canonical JSON."""
    body = {k: v for k, v in record.items() if k != SEAL_FIELD}
    round_tripped = json.loads(json.dumps(body, ensure_ascii=False))
    return json.dumps(
        round_tripped, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _mac(record: Dict[str, Any], key: bytes) -> str:
    return hmac.new(key, _canonical(record), hashlib.sha256).hexdigest()


def seal(record: Dict[str, Any], key: bytes) -> Dict[str, Any]:
    """Return *record* with a fresh :data:`SEAL_FIELD` (input not mutated)."""
    out = {k: v for k, v in record.items() if k != SEAL_FIELD}
    out[SEAL_FIELD] = {"alg": SEAL_ALG, "mac": _mac(out, key)}
    return out


def verify(record: Dict[str, Any], key: Optional[bytes]) -> bool:
    """Whether *record* carries a valid seal under *key*.

    ``False`` for: no key, no seal, an unknown algorithm, a malformed seal,
    or a MAC that does not match.  Compared with ``hmac.compare_digest``.
    """
    if not key or not isinstance(record, dict):
        return False
    sealed = record.get(SEAL_FIELD)
    if not isinstance(sealed, dict) or sealed.get("alg") != SEAL_ALG:
        return False
    mac = sealed.get("mac")
    if not isinstance(mac, str):
        return False
    return hmac.compare_digest(mac, _mac(record, key))
