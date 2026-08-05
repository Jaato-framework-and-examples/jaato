"""Session-id safety validation.

A session id is used to build filesystem paths (``<sessions>/<id>.json``), the
per-session cgroup directory name (``jaato-ws-<id>``), and the per-session
AppArmor profile name (``profile jaato-ws-<id>``).  A client can supply a
session id when routing / attaching (the IPC/WS transport reads
``event.session_id``), so an unvalidated id like ``../../../etc/foo`` would
traverse out of the sessions directory, and characters like ``/`` or newlines
could break the cgroup path / AppArmor profile grammar.

This module is the single source of truth for "is this id safe to interpolate".
It is enforced fail-closed at the sinks (session persistence, cgroups, apparmor)
so every caller is protected, and at the client-input boundary for a clean
early rejection.

Allowed: ASCII letters, digits, ``.`` ``_`` ``-`` (1-256 chars), with NO ``..``
sequence.  This accepts every server-generated form —
``20260704_192600`` (timestamp), ``ephemeral-<hex>``,
``<parent>__sub_<parent>.subagent_<n>`` (isolated subagent) — while rejecting
path separators, parent-dir traversal, whitespace, control characters, and the
AppArmor/cgroup metacharacters.
"""

from __future__ import annotations

import re

# Single lone dots are inert in a path segment; the traversal danger is the
# ``..`` sequence and path separators, which the charset + the explicit
# ``".." not in`` check below both exclude.
_SAFE_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,256}$")


def is_safe_session_id(session_id: str) -> bool:
    """True if ``session_id`` is safe to interpolate into a path / profile name.

    Rejects empty ids, ids with path separators (``/`` ``\\``), any ``..``
    parent-dir sequence, whitespace, control characters, and anything outside
    ``[A-Za-z0-9._-]``.
    """
    if not session_id or not isinstance(session_id, str):
        return False
    if ".." in session_id:
        return False
    return bool(_SAFE_SESSION_ID_RE.match(session_id))


def validate_session_id(session_id: str) -> str:
    """Return ``session_id`` unchanged if safe, else raise ``ValueError``.

    Use at the sinks (persistence, cgroups, apparmor) so an unsafe id can never
    reach a filesystem path or a kernel profile name.
    """
    if not is_safe_session_id(session_id):
        raise ValueError(
            f"unsafe session_id {session_id!r}: must match [A-Za-z0-9._-] "
            f"(1-256 chars) with no '..' — refusing to build a path / profile "
            f"name from it"
        )
    return session_id
