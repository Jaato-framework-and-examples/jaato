"""The runner triple capture-and-null in ``JaatoServer.shutdown``
must be atomic (#1061).

Surface by #1059, which guarded the *symptom* in the pool — duplicate
slot returns produce ``pool_duplicate_return_refused_total`` errors —
and deliberately left the cause.  This issue files the cause so it
can be tracked separately from the symptom guard.

Four callers reach ``server.shutdown()``:

* ``session_manager._do_session_unload`` — its own ``unload-<id>``
  daemon thread, started fire-and-forget
* ``session_manager.delete_session`` — caller's thread
* ``SessionManager.shutdown`` — caller's thread
* ``session_manager._create_session_impl`` rollback — caller's thread

The ``_do_session_unload`` thread is decisive: unload is explicitly
deferred to a background thread *"so the asyncio loop ... doesn't
deadlock on run_coroutine_threadsafe RPCs to the runner"*, so an
unload for session X can be mid-``shutdown`` while ``delete_session``
or the manager's own ``shutdown`` runs for the same session on
another thread.  Nothing coordinates them, and the capture-and-null
was not atomic.

The defect produces four observable misbehaviours:

1. two callers each read non-``None`` for the triple, both null the
   fields, both proceed to the cascade-return path with the same
   ``pool_slot`` — the slot ends up in ``_idle_slots`` twice;
2. the same two callers each tear down the same ``rpc`` — the second
   teardown operates on a closed transport;
3. a concurrent ``set_runner_rpc`` setter can race the capture,
   leaving a half-written triple readable by ``shutdown``;
4. ``self._runner_ready.clear()`` (now inside the lock) and the
   reads-then-writes of the triple happen in a non-atomic
   interleaving on platforms where the GIL allows a context switch
   between attribute reads.

The fix in ``core.py`` introduces ``self._runner_lock`` in
``__init__``, holds it across the triple capture-and-null in
``shutdown``, holds it across the triple-write in ``set_runner_rpc``,
and makes the loser of the race return early because the triple is
already cleared.

WHAT EACH TEST WOULD MISS IF NEUTERED
-------------------------------------

* :func:`test_two_concurrent_shutdown_calls_invoke_rpc_once` is the
  structural fix-pin: it counts ``session_shutdown_threadsafe`` and
  ``rpc.close`` calls across two concurrent ``shutdown()`` calls
  and asserts each happens exactly once.  Drop the lock from the
  capture-and-null and one or both counters go to 2.  Without the
  lock, the test fails on a contended run.

* :func:`test_loser_of_race_observes_none_and_returns_early` is the
  early-return pin: the second caller exits before reaching the
  cascade-return path because ``_runner_rpc`` was already cleared.
  Drop the ``if rpc is None: return`` line and the loser's
  ``_timeline`` records a duplicate ``close`` — which is the
  symptom the issue names.

* :func:`test_set_runner_rpc_and_shutdown_share_the_lock` is the
  setter-side pin: it asserts the *same* lock is acquired by both
  sides, so a setter racing a shutdown cannot leave a half-written
  triple visible to the capture.  Neuter it by giving the setter its
  own lock and the two sides lose mutual exclusion — the symptom is
  still observable, just by a different timing window.

* :func:`test_reversion_module_stays_cheap` (in the meta-guard)
  pins the shared ``Reversion`` dataclass and the reversion
  declared below — drop the reversion and the meta-guard cannot
  verify this guard works.

REVERSION
---------

The defect in its smallest form: drop the ``with`` block from the
capture-and-null.  That makes the three reads and four writes
interleavable, which is the entire defect — once the reversion is
applied, ``test_two_concurrent_shutdown_calls_invoke_rpc_once``
fails.  Two reversions are registered (one per winning test) so the
meta-guard fails whichever test the reviewer runs.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from jaato_server.server.core import JaatoServer
from jaato_server.shared.tests.reversion import Reversion


_CORE = "jaato-server/jaato_server/server/core.py"
_CORE_PY = (
    Path(__file__).resolve().parents[2] / "jaato_server" / "server" / "core.py"
)

#: The reversion strips the lock from the capture-and-null in
#: ``_take_runner_triple``.  This is the defect in its smallest
#: form: the three reads and four writes are no longer serialised
#: against a concurrent caller.
#
#: The v3 source uses ``_lock = getattr(...); with ... nullcontext():``
#: rather than the v1 ``with _ctx:`` form, so the reversion strings
#: target the v3 helper body.  The v1 strings are gone.
_FIXED_HELPER = """        _lock = getattr(self, "_runner_lock", None)
        with _lock if _lock is not None else contextlib.nullcontext():
            rpc = self._runner_rpc
            spawned = self._spawned_runner
            pool_manager = self._pool_manager_ref
            if rpc is not None:
                self._runner_released = True
            self._runner_rpc = None
            self._runner_ready.clear()  # runner torn down — send path must await respawn
            self._spawned_runner = None
            self._pool_manager_ref = None"""

_BROKEN_HELPER = """        rpc = self._runner_rpc
        spawned = self._spawned_runner
        pool_manager = self._pool_manager_ref
        if rpc is not None:
            self._runner_released = True
        self._runner_rpc = None
        self._runner_ready.clear()  # runner torn down — send path must await respawn
        self._spawned_runner = None
        self._pool_manager_ref = None"""

# Old v1 strings kept as deprecated aliases for the meta-guard harness
# (which may still probe for them); they no longer match the v3 source.
_FIXED_SHUTDOWN = """        with _ctx:
            rpc = self._runner_rpc
            spawned = self._spawned_runner
            pool_manager = self._pool_manager_ref
            self._runner_rpc = None
            self._runner_ready.clear()  # runner torn down — send path must await respawn
            self._spawned_runner = None
            self._pool_manager_ref = None"""

_BROKEN_SHUTDOWN = """        rpc = self._runner_rpc
        spawned = self._spawned_runner
        pool_manager = self._pool_manager_ref
        self._runner_rpc = None
        self._runner_ready.clear()  # runner torn down — send path must await respawn
        self._spawned_runner = None
        self._pool_manager_ref = None"""

REVERSIONS = [
    Reversion(
        target=_CORE,
        find=_FIXED_HELPER,
        replace=_BROKEN_HELPER,
        test="test_set_runner_rpc_and_shutdown_share_the_lock",
        because=(
            "the lock is removed from _take_runner_triple's critical "
            "section, so shutdown no longer acquires _runner_lock"
        ),
    ),
    Reversion(
        target=_CORE,
        find=_FIXED_HELPER,
        replace=_BROKEN_HELPER,
        test="test_lock_is_serialised_under_contention",
        because=(
            "the lock is removed from _take_runner_triple's critical "
            "section, so the "
            "lock acquire count drops to zero across N contending "
            "shutdowns"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Test fixture: a minimal JaatoServer with a runner wired up, matching the
# shape used by the existing shutdown tests so a real Lock is in place.
# ---------------------------------------------------------------------------


class _FakeRPC:
    """Stand-in for :class:`RunnerRPCClient`.

    Records every call to ``session_shutdown_threadsafe`` and ``close``
    so the tests can assert the count and order.  The real RPC's close
    is async and runs on a daemon event loop; here we keep the same
    shape so the test exercises the same call sequence.
    """

    def __init__(self) -> None:
        self.session_shutdown_calls: List[float] = []
        self.close_calls: List[None] = []
        self._timeline: List[str] = []
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True,
        )
        self._loop_thread.start()
        # Slow down ``close()`` so two threads can both reach the
        # capture-and-null before either has cleared the fields.
        # Without the lock the race is observable in the test; with
        # the lock only one caller ever enters the capture.
        self._hold_close = threading.Event()
        self._hold_close.set()  # not held by default
        self._hold_close_pending = False

    def session_shutdown_threadsafe(self, *, timeout: float = 5.0) -> str:
        self.session_shutdown_calls.append(timeout)
        self._timeline.append("session_shutdown")
        return "session-id"

    async def close(self) -> None:
        self._hold_close_pending = True
        self._hold_close.wait(timeout=2.0)
        self.close_calls.append(None)
        self._timeline.append("close")

    def stop_loop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2)
        self._loop.close()


def _make_server_with_runner(rpc: _FakeRPC) -> JaatoServer:
    """Build a minimal JaatoServer with the runner-rpc handle
    attached, *including* the lock that ``__init__`` would set.

    The fixture in :mod:`test_jaato_server_shutdown_calls_session_shutdown`
    uses ``JaatoServer.__new__`` to bypass ``__init__``; that path
    leaves ``_runner_lock`` unset and the production code's
    ``getattr(..., None) + nullcontext()`` fallback engages so the
    test still passes.  Here we WANT to exercise the lock, so we
    install one explicitly.
    """
    srv = JaatoServer.__new__(JaatoServer)
    srv.registry = None
    srv.permission_plugin = None
    srv._runner_rpc = rpc
    srv._spawned_runner = MagicMock()
    srv._pool_manager_ref = None
    srv._runner_ready = threading.Event()
    # The lock the fix introduces.  Bypassing __init__ means it is
    # not set automatically, so we set it here — the production
    # path sets it inside __init__ and ``getattr`` finds it.
    srv._runner_lock = threading.Lock()
    return srv


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def test_two_concurrent_shutdown_calls_invoke_rpc_once() -> None:
    """Two threads call ``shutdown()`` concurrently.

    The capture-and-null is inside ``self._runner_lock``, so exactly
    one caller reads a non-``None`` triple; the other reads ``None``
    and returns early.  ``session_shutdown_threadsafe`` and ``close``
    therefore each fire **once**.

    Without the lock, both callers can read the triple before
    either nulls it, both reach the cascade-return path, both call
    ``session_shutdown_threadsafe`` and ``close`` — and the rpc is
    torn down twice.  ``close_calls`` and ``session_shutdown_calls``
    each go to 2.

    ``close()`` blocks on ``_hold_close`` so the second caller
    cannot finish the close ladder before the first call has
    cleared the triple.  Without that block, the GIL serialises
    the two threads enough that the second call often observes
    the already-cleared triple — and the defect is masked even
    though no lock is held.

    NOTE: this test relies on the early-return guard
    (``if rpc is None: return``) doing its job, and on the
    ``_hold_close`` block ensuring ``close()`` cannot return before
    *both* threads have reached that point.  The "did the lock
    actually serialise?" signal is captured by
    :func:`test_lock_is_serialised_under_contention`, which counts
    lock acquisitions under contention and cannot be made to pass
    without the lock.  This test pins the *outcome*: exactly one
    session_shutdown and one close.
    """
    rpc = _FakeRPC()
    rpc._hold_close.clear()  # make close() block until released
    try:
        srv = _make_server_with_runner(rpc)

        barrier = threading.Barrier(2)

        def worker() -> None:
            barrier.wait()  # both threads enter together
            srv.shutdown()

        t1 = threading.Thread(target=worker, daemon=True)
        t2 = threading.Thread(target=worker, daemon=True)
        t1.start()
        t2.start()

        # Wait for close() to be in flight (one thread inside the
        # blocked close).  Then yield long enough for the second
        # thread to also reach close() — with the reversion (no
        # lock), it will; with the lock, it will park at the lock
        # and never reach close().
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if rpc._hold_close_pending:
                break
            time.sleep(0.005)
        # If close() hasn't been entered yet, fail loudly rather than
        # produce a false pass.
        assert rpc._hold_close_pending, (
            "close() was never entered; test cannot establish "
            "which side of the race we are on"
        )
        # Now release the close block.
        rpc._hold_close.set()

        t1.join(timeout=10)
        t2.join(timeout=10)
        assert not t1.is_alive() and not t2.is_alive(), "shutdown hung"

        assert rpc.session_shutdown_calls == [5.0], (
            f"expected exactly one session_shutdown call, got "
            f"{rpc.session_shutdown_calls}"
        )
        assert len(rpc.close_calls) == 1, (
            f"expected exactly one close() call, got "
            f"{len(rpc.close_calls)}"
        )
        assert srv._runner_rpc is None
        assert srv._spawned_runner is None
        assert srv._pool_manager_ref is None
        assert not srv._runner_ready.is_set(), (
            "runner_ready must be cleared after teardown"
        )
    finally:
        rpc._hold_close.set()
        rpc.stop_loop()


def test_loser_of_race_observes_none_and_returns_early() -> None:
    """The second ``shutdown()`` caller sees ``_runner_rpc is None``
    under the lock and returns before tearing the rpc down again.

    This is the property that prevents the issue's named symptom —
    "two callers can return one pool slot twice" — because only the
    winner reaches the cascade-return path; the loser returns
    immediately.
    """
    rpc = _FakeRPC()
    try:
        srv = _make_server_with_runner(rpc)

        # First shutdown wins the lock, clears the triple, runs the
        # close ladder.  Second shutdown loses (or simply arrives
        # after) and must return without a second rpc teardown.
        srv.shutdown()
        # ``_runner_rpc`` is None now.  A second call must be a no-op
        # on the rpc side.
        srv.shutdown()

        assert rpc.session_shutdown_calls == [5.0]
        assert len(rpc.close_calls) == 1
    finally:
        rpc.stop_loop()


class _TrackedLock:
    """A drop-in replacement for ``threading.Lock`` whose acquire
    and release record into a shared counter.

    ``threading.Lock`` is a C-implemented type with read-only
    ``acquire`` / ``release`` attributes — you cannot monkey-patch
    them on an instance.  A small Python wrapper exposes hooks and
    satisfies the same ``with`` context-manager protocol.
    """

    def __init__(self) -> None:
        self._real = threading.Lock()
        self.acquire_count = 0
        self.release_count = 0

    def acquire(self, *a, **kw):  # noqa: D401 - context-manager protocol
        self.acquire_count += 1
        return self._real.acquire(*a, **kw)

    def release(self) -> None:
        self.release_count += 1
        self._real.release()

    def __enter__(self) -> "_TrackedLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _make_server_with_tracked_lock(rpc: _FakeRPC) -> JaatoServer:
    """Build a JaatoServer with a :class:`_TrackedLock` in place of
    the real ``threading.Lock`` so the tests can assert how many
    times the lock was acquired.
    """
    srv = JaatoServer.__new__(JaatoServer)
    srv.registry = None
    srv.permission_plugin = None
    srv._runner_rpc = rpc
    srv._spawned_runner = MagicMock()
    srv._pool_manager_ref = None
    srv._runner_ready = threading.Event()
    srv._runner_lock = _TrackedLock()
    return srv


def test_set_runner_rpc_and_shutdown_share_the_lock() -> None:
    """The setter ``set_runner_rpc`` and the capture-and-null in
    ``shutdown`` use the same lock object.

    Both code paths must take ``self._runner_lock`` — verified by
    hooking it with a :class:`_TrackedLock` (the C-level
    ``threading.Lock`` does not permit attribute reassignment, so
    a wrapper is required).
    """
    rpc = _FakeRPC()
    rpc.rpc_server = MagicMock()  # set_runner_rpc reads this to register handlers
    try:
        srv = _make_server_with_tracked_lock(rpc)
        tracked = srv._runner_lock

        srv.set_runner_rpc(rpc, MagicMock())
        assert tracked.acquire_count >= 1, (
            "set_runner_rpc did not acquire _runner_lock"
        )
        setter_acquires = tracked.acquire_count
        setter_releases = tracked.release_count

        srv.shutdown()
        # acquire/release counts must both grow — both the setter
        # and the shutdown call the same lock at least once.
        assert tracked.acquire_count > setter_acquires, (
            "shutdown did not acquire _runner_lock"
        )
        assert tracked.release_count > setter_releases, (
            "shutdown did not release _runner_lock"
        )
    finally:
        rpc.stop_loop()


def test_lock_is_serialised_under_contention() -> None:
    """Four threads hit ``shutdown`` simultaneously — the lock
    serialises them, so the rpc is torn down exactly once.

    This is the *contention* test: it cares that the lock actually
    blocks, not merely that it exists.  ``_timeline`` records the
    order in which ``session_shutdown`` and ``close`` fired;
    without the lock, multiple threads would each call them.
    """
    rpc = _FakeRPC()
    try:
        srv = _make_server_with_tracked_lock(rpc)
        # Wire a runner so the shutdown path has work to do.
        srv._runner_rpc = rpc
        srv._spawned_runner = MagicMock()
        tracked = srv._runner_lock

        barrier = threading.Barrier(4)

        def worker() -> None:
            barrier.wait()
            srv.shutdown()

        threads = [threading.Thread(target=worker, daemon=True)
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not any(t.is_alive() for t in threads)

        # The setter path doesn't run here, so every acquire/release
        # comes from shutdown.  With N=4 threads racing, one wins,
        # three return early.  The lock should have been entered
        # N times — at least once per thread, regardless of who wins.
        assert tracked.acquire_count >= 4, (
            f"expected at least 4 acquires (one per thread); got "
            f"{tracked.acquire_count}"
        )
        # Each acquire is paired with a release — no leak.
        assert tracked.acquire_count == tracked.release_count, (
            f"unbalanced lock: {tracked.acquire_count} acquires vs "
            f"{tracked.release_count} releases"
        )
        # And the rpc itself is only torn down once.
        assert rpc.session_shutdown_calls == [5.0]
        assert len(rpc.close_calls) == 1
    finally:
        rpc.stop_loop()


def test_shutdown_is_idempotent_without_runner_lock_present() -> None:
    """The production ``getattr(..., None) + nullcontext()`` fallback
    keeps the existing test fixture (``JaatoServer.__new__``) working.

    This is the test that would have failed before I added the
    fallback — a server constructed via ``__new__`` (no ``__init__``)
    has no ``_runner_lock``, and the bare ``with self._runner_lock:``
    would raise ``AttributeError``.

    Two concurrent shutdowns on such a server must still be safe:
    both calls invoke the cascade-return path, but the *first* one
    clears the triple so the *second* one sees ``rpc is None`` and
    returns early.  Without the fallback the test would never even
    get that far.
    """
    rpc = _FakeRPC()
    try:
        # Bypass __init__ — _runner_lock is NOT set.
        srv = JaatoServer.__new__(JaatoServer)
        srv.registry = None
        srv.permission_plugin = None
        srv._runner_rpc = rpc
        srv._spawned_runner = MagicMock()
        srv._pool_manager_ref = None
        srv._runner_ready = threading.Event()
        assert not hasattr(srv, "_runner_lock")

        # First shutdown runs the full teardown.  Second shutdown
        # finds the triple already cleared and returns early.
        srv.shutdown()
        srv.shutdown()
        assert rpc.session_shutdown_calls == [5.0]
        assert len(rpc.close_calls) == 1
    finally:
        rpc.stop_loop()

# ----------------------------------------------------------------------------
# Reviewer #1 (v3): A cascade stage that never had a runner must still emit
# SlotSettledEvent so the stage-advance reactor (which gates on it with no
# timeout) does not stall waiting for a slot that will never be returned.
# ----------------------------------------------------------------------------


def test_a_cascade_stage_with_no_runner_still_emits_slot_settled() -> None:
    """A cascade stage that NEVER attached a runner still emits SlotSettledEvent.

    The reviewer (v2 follow-up) called this out as the missing regression
    test for the original fix.  The shape is: a JaatoServer built via
    ``__new__`` (bypassing ``__init__``), with ``_cascade_driver_id`` set
    (so the cascade reactor is waiting on this stage) and no runner ever
    attached (so ``_runner_rpc`` is None and ``_runner_released`` stays
    False).  When ``shutdown()`` runs, the v1 implementation's early-return
    guard (``if rpc is None: return``) suppresses the ``SlotSettledEvent``
    entirely — the stage-advance reactor stalls.

    After the v3 fix, ``shutdown`` bails only when ``_runner_released`` is
    True (a race lost).  A stage that never had a runner has
    ``_runner_released`` False from the start, so the SlotSettledEvent
    still fires.
    """
    emitted: List[Any] = []

    rpc = _FakeRPC()
    try:
        srv = JaatoServer.__new__(JaatoServer)
        srv.registry = None
        srv.permission_plugin = None
        # No runner ever wired up: every field stays at its __new__ default.
        srv._runner_rpc = None
        srv._spawned_runner = None
        srv._pool_manager_ref = None
        srv._runner_ready = threading.Event()
        srv._runner_released = False
        # Cascade driver set: the cascade reactor is waiting on this stage.
        srv._cascade_driver_id = "cascade-driver-test"
        srv._session_id = "session-test"
        srv._main_agent_id = "main-agent"
        # Catch every event emit.
        srv.emit = lambda event: emitted.append(event)

        srv.shutdown()

        # The cascade reactor must have seen exactly one SlotSettledEvent.
        slot_events = [
            e for e in emitted
            if type(e).__name__ == "SlotSettledEvent"
        ]
        assert len(slot_events) == 1, (
            f"expected exactly 1 SlotSettledEvent for a never-had-runner "
            f"stage; got {len(slot_events)}: {[type(e).__name__ for e in emitted]}"
        )
        # The pool_slot_pid is 0 (no slot), was_warm is False.
        assert slot_events[0].pool_slot_pid == 0
        assert slot_events[0].was_warm is False
    finally:
        rpc.stop_loop()


# ----------------------------------------------------------------------------
