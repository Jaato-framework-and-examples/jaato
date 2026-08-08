"""Concurrent create_session must never issue the same id twice.

Session ids are second-resolution timestamps with _N appended on collision.
The check used to be plain check-then-act: a candidate was tested against
_sessions at allocation, but only inserted there after the runner spawned —
measured ~7.3s later. Every concurrent create inside that window saw the
same id free and took it, so three simultaneous creates were issued ONE id
between them and two of the three sessions never ran.

Invisible to sequential use: PoC #1 created one session, PoC #2 created
three strictly one after another. Concurrency is the first thing a fan-out
does.
"""

import threading
from types import SimpleNamespace

import pytest

from server.session_manager import SessionManager


class _SM:
    """Harness binding just the allocator to real locks + registries."""

    def __init__(self, persisted=(), live=()):
        self._lock = threading.RLock()
        self._sessions = {sid: object() for sid in live}
        self._reserved_session_ids = set()
        self._persisted = [SimpleNamespace(session_id=s) for s in persisted]

    def _get_persisted_sessions(self, workspace_path=None):
        return self._persisted

    def __getattr__(self, name):
        fn = getattr(SessionManager, name)
        return lambda *a, **k: fn(self, *a, **k)


def test_sequential_allocation_is_unique():
    sm = _SM()
    ids = [sm._allocate_session_id(None) for _ in range(5)]
    assert len(set(ids)) == 5


def test_concurrent_allocation_never_duplicates():
    """THE regression. Same second, many threads, no duplicates."""
    sm = _SM()
    out, errors = [], []
    lock = threading.Lock()

    def claim():
        try:
            sid = sm._allocate_session_id(None)
            with lock:
                out.append(sid)
        except Exception as exc:            # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=claim) for _ in range(32)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(out) == 32
    assert len(set(out)) == 32, (
        f"duplicate ids issued: {len(out) - len(set(out))} collisions — "
        "concurrent creates would drive the same session"
    )


def test_claim_survives_the_gap_before_registration():
    """The window that actually bit: allocation and registration are ~336
    lines and a full runner spawn apart. A second caller inside that gap
    must not be handed the same id."""
    sm = _SM()
    first = sm._allocate_session_id(None)
    # ... runner spawn happens here; _sessions is still empty ...
    assert sm._sessions == {}
    second = sm._allocate_session_id(None)
    assert second != first


def test_allocation_avoids_live_and_persisted_ids():
    sm = _SM(persisted=("20260808_120000",), live=("20260808_120000_1",))
    got = sm._allocate_session_id(None)
    assert got not in ("20260808_120000", "20260808_120000_1")


def test_release_frees_the_claim_for_reuse():
    sm = _SM()
    first = sm._allocate_session_id(None)
    sm._release_session_id(first)
    assert first not in sm._reserved_session_ids


def test_release_is_idempotent():
    sm = _SM()
    sid = sm._allocate_session_id(None)
    sm._release_session_id(sid)
    sm._release_session_id(sid)          # must not raise


def test_a_leaked_claim_only_costs_that_one_id():
    """Claims are never released on some exotic failure path? The cost is
    bounded: that id string is skipped, nothing else breaks."""
    sm = _SM()
    leaked = sm._allocate_session_id(None)
    nxt = sm._allocate_session_id(None)
    assert nxt != leaked
    assert leaked in sm._reserved_session_ids
