"""Which sessions may message each other: the GROUP predicate.

A session belongs to a group by one of two facts it already carries, both
persisted on its record and both stamped by the daemon rather than by the
session itself:

* ``cascade_driver_id`` (record 2.10) -- the cascade it is a stage of.
  Group key ``cid:<id>``.
* ``created_by`` (record 2.9) -- the transport-authenticated user that
  created it (``SO_PEERCRED`` on IPC, a bound ticket on WS).  Group key
  ``user:<created_by>``.

Two sessions are in a common group when their key sets intersect.  That
is the whole predicate, and the module is stdlib-only for the reason
``shared/apparmor_label.py`` and ``shared/runtime_limits.py`` are: the one
question "may A reach B" must have exactly one answer, importable from the
daemon and from a test without dragging the session manager in.

Rules, each attached to a way the predicate goes wrong:

* **``None`` never matches ``None``.**  Two anonymous IPC sessions -- no
  cascade, no authenticated creator -- must not form a daemon-wide group.
  The same posture the ticket door takes about ``created_by=""`` (#1074):
  absence of identity is not an identity.  Empty strings are treated as
  absent for the same reason.
* **``created_by`` is already application-qualified** (``app:user``), so a
  user group never crosses an application boundary on a shared daemon.
* **A group is derived, never declared.**  There is no ``group_id`` a
  session can claim; both keys come from facts the daemon stamped.  A
  declared key, if one is ever wanted, is a third entry in the same set.
"""
from __future__ import annotations

from typing import Any, FrozenSet, Optional

#: Key prefixes, so a consumer rendering a key can say which fact it came
#: from without re-deriving it.
CASCADE_KEY_PREFIX = "cid:"
USER_KEY_PREFIX = "user:"


def group_keys_for(
    cascade_driver_id: Optional[str],
    created_by: Optional[str],
) -> FrozenSet[str]:
    """The group keys two facts yield.

    The value-level form of :func:`group_keys`, for callers that hold the
    facts rather than an object carrying them (a persisted record read off
    an index, a test).
    """
    keys = set()
    if cascade_driver_id:
        keys.add(CASCADE_KEY_PREFIX + str(cascade_driver_id))
    if created_by:
        keys.add(USER_KEY_PREFIX + str(created_by))
    return frozenset(keys)


def group_keys(session: Any) -> FrozenSet[str]:
    """The group keys a session (or any record duck-typed like one) belongs to.

    Reads ``cascade_driver_id`` and ``created_by`` off the object; a missing
    attribute counts as absent, so a persisted ``SessionInfo`` predating the
    owner field joins no user group -- the safe direction.
    """
    return group_keys_for(
        getattr(session, "cascade_driver_id", None),
        getattr(session, "created_by", None),
    )


def common_groups(a: Any, b: Any) -> FrozenSet[str]:
    """The keys *a* and *b* share.  Empty when they share none."""
    return group_keys(a) & group_keys(b)


def same_group(a: Any, b: Any) -> bool:
    """Whether *a* and *b* share at least one group key.

    Symmetric.  ``False`` for two anonymous sessions however alike they
    are, because neither carries a key.
    """
    return bool(common_groups(a, b))
