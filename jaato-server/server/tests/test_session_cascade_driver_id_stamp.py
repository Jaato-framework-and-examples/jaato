"""Pin test for Session.cascade_driver_id field (Bug B real root cause).

Server 0.6.164+.

PR-188 added _record_cid_session_activity at session_manager.py:4209
to update _cid_last_session_ts on every session create.  But the
update read ``getattr(session, "cascade_driver_id", None)`` against
the ``Session`` dataclass — which had no ``cascade_driver_id`` field
defined and no runtime assignment anywhere in the codebase.  So
the getattr always returned None and the update no-op'd.

Same root cause silently broke _dispatch_to_cascade_clients
(session_manager.py:3141): the dispatch path uses the same getattr
and early-returned on None.  Cascade-client IN-PROCESS callbacks
never fired for cascade sessions.

Empirical motivation (peer 7:1, retry-45 diagnostic, 2026-05-28):
GC reaper reaped observer registration at 22:07:25 despite
host_validator session spawn at 22:06:00 (85s before reap, well
inside the 300s threshold).  cid_last_session_ts had never been
updated because session.cascade_driver_id was None for every
downstream session.

Fix (this PR): add ``cascade_driver_id: Optional[str] = None`` to
the Session dataclass + stamp it at session-add site in
``create_session``.
"""

from __future__ import annotations

from dataclasses import fields

from server.session_manager import Session


def test_session_dataclass_has_cascade_driver_id_field():
    """The Session dataclass MUST define ``cascade_driver_id`` so
    ``getattr(session, "cascade_driver_id", None)`` returns the
    real value, not None-by-fallback."""
    field_names = {f.name for f in fields(Session)}
    assert "cascade_driver_id" in field_names, (
        "Session dataclass missing cascade_driver_id field — "
        "_dispatch_to_cascade_clients + _record_cid_session_activity "
        "both silently no-op without it.  See Bug B root cause "
        "diagnosis (server 0.6.164)."
    )


def test_session_default_cascade_driver_id_is_none():
    """Standalone sessions (no cascade) have cascade_driver_id=None
    by default — preserves the existing "cid is None ⇒ skip cascade
    dispatch / skip cid GC tracking" semantics."""
    s = Session(
        session_id="standalone-1",
        name="standalone",
        server=None,
        created_at="2026-05-28T00:00:00Z",
    )
    assert s.cascade_driver_id is None


def test_session_explicit_cascade_driver_id_stored():
    """Constructing a Session with cascade_driver_id=<x> persists
    the value on the object — both cascade dispatch and GC sweep
    can then read it via getattr."""
    s = Session(
        session_id="cascade-1",
        name="cascade",
        server=None,
        created_at="2026-05-28T00:00:00Z",
        cascade_driver_id="cid-abc-123",
    )
    assert s.cascade_driver_id == "cid-abc-123"
    # The getattr pattern both code paths use:
    assert getattr(s, "cascade_driver_id", None) == "cid-abc-123"
