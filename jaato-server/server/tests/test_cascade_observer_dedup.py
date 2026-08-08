"""A client that is BOTH attached and cascade-registered gets each event ONCE.

``_emit_to_session`` fans out twice — directly to ``session.attached_clients``,
then again to cascade-clients registered for the session's cid.  A driver that
attaches AND registers as an observer for the same cascade (the canonical
"client of the API" pattern) therefore sits on both paths, and without a dedup
receives every event twice.

This is load-bearing and has silently broken once already: PR-207 (server
0.6.177) compared ``entry.client_id`` — the NAMESPACED registration id
``_cascade:{cid}:{conn}`` — instead of the raw connection id the callback
actually delivers to, so the skip never matched and the dedup was a no-op.
Falsified only by a kb-side report on 2026-06-03, not by the suite. Fixed in
0.6.178 by comparing ``delivery_target_id``.

These tests pin the comparand, both call sites, and the negative case, so the
next refactor cannot quietly reintroduce it.
"""

from types import SimpleNamespace

import pytest

from server.session_manager import SessionManager


class _Entry:
    """Stand-in for CascadeClientEntry with the fields dispatch reads."""

    def __init__(self, client_id, delivery_target_id, role="observer"):
        self.client_id = client_id
        self.delivery_target_id = delivery_target_id
        self.role = role
        self.last_event_ts = None
        self.received = []

    def event_type_match(self, event):
        return True

    def callback(self, event):
        self.received.append(event)


def _sm(cid, entries):
    import threading
    return SimpleNamespace(
        _cascade_clients={cid: entries},
        _cascade_clients_lock=threading.Lock(),
    )


def _dispatch(sm, cid, event, **kw):
    SessionManager._dispatch_to_cascade_clients_by_cid(sm, cid, event, **kw)


EVENT = object()


def test_attached_observer_is_skipped_post_bootstrap():
    """The steady-state path: the direct fan-out already delivered."""
    entry = _Entry("_cascade:c1:ipc_1", delivery_target_id="ipc_1")
    sm = _sm("c1", [entry])
    _dispatch(sm, "c1", EVENT, skip_client_ids={"ipc_1"})
    assert entry.received == []


def test_attached_observer_is_skipped_at_bootstrap():
    """The bootstrap path uses the singular skip_client_id."""
    entry = _Entry("_cascade:c1:ipc_1", delivery_target_id="ipc_1")
    sm = _sm("c1", [entry])
    _dispatch(sm, "c1", EVENT, skip_client_id="ipc_1")
    assert entry.received == []


def test_a_different_connection_still_receives_it():
    """Dedup must not silence genuine observers on other connections."""
    mine = _Entry("_cascade:c1:ipc_1", delivery_target_id="ipc_1")
    other = _Entry("_cascade:c1:ipc_2", delivery_target_id="ipc_2")
    sm = _sm("c1", [mine, other])
    _dispatch(sm, "c1", EVENT, skip_client_ids={"ipc_1"})
    assert mine.received == []
    assert other.received == [EVENT]


def test_skip_compares_delivery_target_not_the_namespaced_id():
    """THE PR-207 regression, pinned.

    The registration id is ``_cascade:{cid}:{conn}``; the callback delivers to
    ``{conn}``. Comparing the former against a set of raw connection ids never
    matches, so the dedup silently no-ops and every event doubles.
    """
    entry = _Entry("_cascade:c1:ipc_1", delivery_target_id="ipc_1")
    sm = _sm("c1", [entry])
    # The caller passes RAW connection ids (session.attached_clients).
    _dispatch(sm, "c1", EVENT, skip_client_ids={"ipc_1"})
    assert entry.received == [], (
        "dedup compared the wrong identifier — this is exactly how PR-207 "
        "shipped a no-op skip that doubled every event on the wire"
    )


def test_in_process_entry_without_a_target_is_never_skipped():
    """delivery_target_id is None for in-process callers (the premium reactor
    extension); they are not on the direct-IPC path, so they must still fire."""
    entry = _Entry("_cascade:c1", delivery_target_id=None)
    sm = _sm("c1", [entry])
    _dispatch(sm, "c1", EVENT, skip_client_ids={"ipc_1", "_cascade:c1"})
    assert entry.received == [EVENT]


def test_no_skip_delivers_to_everyone():
    entry = _Entry("_cascade:c1:ipc_1", delivery_target_id="ipc_1")
    sm = _sm("c1", [entry])
    _dispatch(sm, "c1", EVENT)
    assert entry.received == [EVENT]


def test_skipped_entry_does_not_have_its_liveness_bumped():
    """last_event_ts drives the GC sweep; a skipped entry received nothing,
    so bumping it would keep a dead registration alive on phantom traffic."""
    entry = _Entry("_cascade:c1:ipc_1", delivery_target_id="ipc_1")
    sm = _sm("c1", [entry])
    _dispatch(sm, "c1", EVENT, skip_client_ids={"ipc_1"})
    assert entry.last_event_ts is None


def test_owners_are_dispatched_before_observers():
    order = []
    owner = _Entry("_cascade:c1", None, role="owner")
    obs = _Entry("_cascade:c1:ipc_2", "ipc_2", role="observer")
    owner.callback = lambda e: order.append("owner")
    obs.callback = lambda e: order.append("observer")
    # registered observer-first to prove the sort, not the insertion order
    _dispatch(_sm("c1", [obs, owner]), "c1", EVENT)
    assert order == ["owner", "observer"]


def test_a_raising_observer_does_not_break_the_chain():
    boom = _Entry("_cascade:c1:ipc_1", "ipc_1")
    boom.callback = lambda e: (_ for _ in ()).throw(RuntimeError("boom"))
    after = _Entry("_cascade:c1:ipc_2", "ipc_2")
    _dispatch(_sm("c1", [boom, after]), "c1", EVENT)
    assert after.received == [EVENT]


def test_none_cid_is_a_noop():
    entry = _Entry("_cascade:c1:ipc_1", "ipc_1")
    _dispatch(_sm("c1", [entry]), None, EVENT)
    assert entry.received == []
