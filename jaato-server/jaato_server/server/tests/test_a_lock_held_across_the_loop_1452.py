"""A thread holding ``SessionManager._lock`` never waits on the daemon loop (#1452).

The daemon loop stopped for 121 s inside ``_emit_to_session``, on ``with
self._lock:``, routing a runner's streamed output.  The holder was a
session listing: ``list_sessions`` held the lock while it asked every
loaded session's runner for its history (``turn_count``), and that ask is
``session_get_history_threadsafe`` -- a coroutine scheduled onto the very
loop that was waiting for the lock.  Each fetch broke only when its 15 s
outer timeout expired; eight loaded sessions made 121 s.  #1355 moved
listings off the loop thread, which is what turned a refused call into
this deadlock.

Covered here:

* the listing, driven on a worker thread with a real
  :class:`RunnerRPCClient` talking to a real :class:`RunnerRPC`, while the
  loop takes the lock at the moment the history fetch reaches it;
* an AST guard that no ``with self._lock:`` block in ``session_manager.py``
  calls a ``*_threadsafe`` wrapper or a sync entry point known to reach
  one (the #1355 list, reused);
* the lock instrumentation (ask 1): the profiled lock names its holder,
  the long-hold log fires, and the loop watchdog puts the holder in its
  ``LOOP_STALL`` dump and no longer files a thread waiting on a future as
  "blocked acquiring a lock";
* ``session.new`` phase timing (ask 2), up to the socket write;
* the loop-lag histogram (ask 3).
"""
from __future__ import annotations

import ast
import asyncio
import logging
import threading
import time
from typing import Any, List

import pytest

from jaato_sdk.events import SessionInfoEvent, SystemMessageEvent
from jaato_sdk.plugins.model_provider.types import Message, Part, Role
from jaato_server.server import session_new_timing
from jaato_server.server.lock_profile import ProfiledRLock, held_locks
from jaato_server.server.loop_watchdog import LoopWatchdog
from jaato_server.server.tests.test_a_save_on_the_loop_thread_1355 import (
    _SERVER_DIR, _Harness, _RunnerSession, _is_blocking_call,
)
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"
_WD = "jaato-server/jaato_server/server/loop_watchdog.py"
_LP = "jaato-server/jaato_server/server/lock_profile.py"
_IPC = "jaato-server/jaato_server/server/ipc.py"

REVERSIONS = [
    Reversion(
        target=_SM,
        find="""        with self._lock:
            loaded = list(self._sessions.values())
        for session in loaded:
""",
        replace="""        with self._lock:
          loaded = list(self._sessions.values())
          for session in loaded:
""",
        because="the listing holds the lock across every runner history "
                "fetch again, deadlocking against the loop until each "
                "fetch times out",
        test="test_a_listing_off_the_loop_does_not_hold_the_lock_on_the_runner",
    ),
    Reversion(
        target=_SM,
        find="""        if attached is not None:
            # #1452: emitted AFTER""",
        replace="""        with self._lock:
          if attached is not None:
            # #1452: emitted AFTER""",
        because="get_or_create_default asks the runner under the lock again",
        test="test_no_lock_block_calls_a_runner_blocking_entry_point",
    ),
    Reversion(
        target=_WD,
        find="""            if (self._is_waiting_for_a_lock(line)
                    and not self._is_a_condition_wait(frame)):""",
        replace="""            if self._is_waiting_for_a_lock(line):""",
        because="a thread holding a lock while it waits on a future is "
                "filed as blocked again, which is what hid the holder",
        test="test_a_holder_waiting_on_a_future_is_a_candidate",
    ),
    Reversion(
        target=_WD,
        find="""                stack or "<loop thread not found>", self._lock_owners(),""",
        replace="""                stack or "<loop thread not found>", "",""",
        because="the stall dump stops naming the lock's owner",
        test="test_a_stall_on_the_lock_names_its_holder",
    ),
    Reversion(
        target=_LP,
        find="""        if self._warn_after and held >= self._warn_after:""",
        replace="""        if False:""",
        because="a long hold is no longer logged",
        test="test_a_long_hold_is_logged_with_its_site",
    ),
    Reversion(
        target=_IPC,
        find="""        session_new_timing.note_written(event, "ipc")""",
        replace="""        pass""",
        because="nothing records when a session.new answer reached the socket",
        test="test_the_session_new_answer_is_timed_to_the_socket",
    ),
    Reversion(
        target=_WD,
        find="""                self.record_lag(time.monotonic() - self._beat - self._interval)""",
        replace="""                pass""",
        because="the heartbeat no longer feeds the loop-lag histogram",
        test="test_the_heartbeat_feeds_the_histogram",
    ),
]


# ----------------------------------------------------------------------
# The reported deadlock, with a real client and a real runner
# ----------------------------------------------------------------------


class _TwoMessageSession(_RunnerSession):
    """A runner history of one turn, so ``turn_count`` is 1 when read."""

    def get_history(self) -> List[Message]:
        return [Message(role=Role.USER, parts=[Part(text="q")]),
                Message(role=Role.MODEL, parts=[Part(text="a")])]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    h = _Harness(tmp_path)
    h.runner._session_host.session = _TwoMessageSession()
    return h


def _make_listable(server: Any) -> None:
    """The attributes the listing reads that the #1355 harness leaves unset."""
    for key, value in dict(
        _model_provider="p", _model_name="m", _model_running=False,
        _pending_permission_request_id=None, _pending_permission_since=None,
        _pending_clarification_request_id=None,
        _pending_clarification_since=None,
    ).items():
        setattr(server, key, value)


async def test_a_listing_off_the_loop_does_not_hold_the_lock_on_the_runner(
    harness,
) -> None:
    """The loop routes an event the moment the listing's fetch reaches it.

    The loop takes the lock inside the history coroutine, before it goes to
    the runner: exactly when a lock held across the fetch deadlocks.  With
    the fix the lock is free and the listing answers at once, with the
    runner's turn count; with the defect it answers after the 15 s outer
    timeout, from the empty daemon-side fallback.
    """
    await harness.start()
    try:
        _make_listable(harness.server)
        manager, client = harness.manager, harness.client
        original = client.session_get_history

        async def history_via_the_lock(**kwargs):
            # Runs on the loop thread: the ``_emit_to_session`` of the
            # incident's stack.
            manager._emit_to_session("s1", SystemMessageEvent(message="x"))
            return await original(**kwargs)

        client.session_get_history = history_via_the_lock
        started = time.monotonic()
        rows = await asyncio.to_thread(manager.list_sessions)
        elapsed = time.monotonic() - started
    finally:
        await harness.stop()
    assert elapsed < 5.0, f"listing took {elapsed:.1f}s: lock held on the loop"
    [row] = [r for r in rows if r.session_id == "s1"]
    assert row.turn_count == 1


def _blocking_calls_under_the_lock() -> List[str]:
    """Calls a ``with self._lock:`` body makes that reach the runner.

    Nested functions and lambdas are skipped (they run later, elsewhere).
    ``self._session_plugin.list_sessions`` is the file-session plugin's
    disk listing, which shares a name with the manager's and reaches no
    runner.
    """
    path = _SERVER_DIR / "session_manager.py"
    tree = ast.parse(path.read_text())
    found = []
    for block in ast.walk(tree):
        if not (isinstance(block, ast.With) and any(
                ast.unparse(i.context_expr) == "self._lock"
                for i in block.items)):
            continue
        stack = list(block.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.Lambda, ast.ClassDef)):
                continue
            stack.extend(ast.iter_child_nodes(node))
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)):
                continue
            receiver = ast.unparse(node.func.value)
            if receiver == "self._session_plugin":
                continue
            if _is_blocking_call(node):
                found.append(f"session_manager.py:{node.lineno} "
                             f"{receiver}.{node.func.attr}()")
    return found


def test_no_lock_block_calls_a_runner_blocking_entry_point() -> None:
    """Snapshot under the lock; ask the runner after releasing it."""
    assert _blocking_calls_under_the_lock() == []


# ----------------------------------------------------------------------
# Ask 1: the lock says who holds it
# ----------------------------------------------------------------------


def _hold(lock: ProfiledRLock, release: threading.Event,
          name: str) -> threading.Thread:
    """A thread that takes ``lock`` and waits on ``release`` holding it."""
    held = threading.Event()

    def run() -> None:
        with lock:
            held.set()
            release.wait(10)

    t = threading.Thread(target=run, name=name, daemon=True)
    t.start()
    assert held.wait(5)
    return t


def test_the_profiled_lock_is_a_reentrant_lock() -> None:
    lock = ProfiledRLock("t.reentrant", warn_after=0)
    with lock:
        with lock:
            assert lock.locked()
        assert lock.locked()
    assert not lock.locked()
    cond = threading.Condition(lock)
    with cond:
        assert not cond.wait(0.01)
        assert lock.locked()
    assert not lock.locked()
    with pytest.raises(RuntimeError):
        lock.release()


def test_the_profiled_lock_names_its_holder_and_what_it_is_doing() -> None:
    lock = ProfiledRLock("t.named", warn_after=0)
    release = threading.Event()
    t = _hold(lock, release, "holder-1452")
    try:
        [snap] = [h for h in held_locks() if h.name == "t.named"]
        assert snap.owner_name == "holder-1452"
        assert "release.wait(10)" in snap.stack
        assert "in run" in snap.site
    finally:
        release.set()
        t.join(5)
    assert not [h for h in held_locks() if h.name == "t.named"]


def test_a_long_hold_is_logged_with_its_site(caplog) -> None:
    lock = ProfiledRLock("t.long", warn_after=0.05)
    with caplog.at_level(logging.WARNING, "jaato_server.server.lock_profile"):
        with lock:
            time.sleep(0.1)
        with lock:
            pass
    lines = [r.getMessage() for r in caplog.records
             if "LOCK_HELD_LONG" in r.getMessage()]
    assert len(lines) == 1
    assert "'t.long'" in lines[0] and __file__ in lines[0]


def test_the_hold_threshold_reads_its_env_var(monkeypatch) -> None:
    from jaato_server.server.lock_profile import hold_warn_seconds
    monkeypatch.setenv("JAATO_LOCK_HOLD_WARN_MS", "250")
    assert hold_warn_seconds() == 0.25
    monkeypatch.setenv("JAATO_LOCK_HOLD_WARN_MS", "0")
    assert hold_warn_seconds() == 0.0
    monkeypatch.setenv("JAATO_LOCK_HOLD_WARN_MS", "nonsense")
    assert hold_warn_seconds() == 0.5


def test_a_holder_waiting_on_a_future_is_a_candidate() -> None:
    """The incident's holder was in ``Condition.wait``, filed as blocked."""
    lock = ProfiledRLock("t.future", warn_after=0)
    release = threading.Event()
    t = _hold(lock, release, "future-waiter-1452")
    try:
        dump = LoopWatchdog()._all_thread_stacks()
    finally:
        release.set()
        t.join(5)
    candidates, _, blocked = dump.partition("=== BLOCKED ACQUIRING A LOCK")
    assert "future-waiter-1452" in candidates
    assert "future-waiter-1452" not in blocked


async def test_a_stall_on_the_lock_names_its_holder(caplog) -> None:
    """A stalled loop's dump says who holds the lock it is waiting for."""
    lock = ProfiledRLock("t.stall", warn_after=0)
    release = threading.Event()
    t = _hold(lock, release, "stall-holder-1452")
    dog = LoopWatchdog(interval=0.05, threshold=0.1, resample_every=0.1,
                       all_threads_after=60)
    with caplog.at_level(logging.WARNING, "jaato_server.server.loop_watchdog"):
        dog.start()
        try:
            threading.Timer(0.6, release.set).start()
            with lock:        # blocks THE LOOP until the holder lets go
                pass
            await asyncio.sleep(0.2)
        finally:
            dog.stop()
            t.join(5)
    stalls = [r.getMessage() for r in caplog.records
              if r.getMessage().startswith("LOOP_STALL")]
    assert stalls
    assert any("'t.stall' is HELD by thread" in s
               and "stall-holder-1452" in s for s in stalls)


def test_the_session_manager_lock_is_profiled() -> None:
    from jaato_server.server.session_manager import SessionManager
    manager = SessionManager.__new__(SessionManager)
    SessionManager.__init__(manager)
    assert isinstance(manager._lock, ProfiledRLock)


# ----------------------------------------------------------------------
# Ask 2: session.new, phase by phase, to the socket
# ----------------------------------------------------------------------


class _Request:
    command = "session.new"
    payload = {"request_id": "rq-1452"}


async def test_the_session_new_answer_is_timed_to_the_socket(caplog) -> None:
    from jaato_server.server.ipc import JaatoIPCServer

    server = JaatoIPCServer.__new__(JaatoIPCServer)
    server._lock = asyncio.Lock()
    written = []

    class _Client:
        writer = object()

    server._clients = {"c1": _Client()}

    async def write(writer, data):
        written.append(data)

    server._write_message = write
    with caplog.at_level(logging.INFO,
                         "jaato_server.server.session_new_timing"):
        session_new_timing.note_request(_Request())
        await asyncio.to_thread(_create_on_a_worker)
        await server._send_to_client(
            "c1", SessionInfoEvent(session_id="s9", request_id="rq-1452"))
    phases = [r.getMessage().split("phase=")[1].split()[0]
              for r in caplog.records if "SESSION_NEW_PHASE" in r.getMessage()]
    assert written
    assert phases == ["received", "handler_started", "session_created",
                      "answer_written"]


def _create_on_a_worker() -> None:
    session_new_timing.begin("rq-1452")
    try:
        session_new_timing.mark("session_created", session_id="s9")
    finally:
        session_new_timing.end()


# ----------------------------------------------------------------------
# Ask 3: loop lag as a histogram
# ----------------------------------------------------------------------


def test_loop_lag_lands_in_its_bucket_and_is_reported(caplog) -> None:
    dog = LoopWatchdog(lag_report_every=0.01)
    for lag in (0.0005, 0.003, 0.2, 30.0):
        dog.record_lag(lag)
    t = dog.get_telemetry()
    assert t["loop_lag_le_1ms"] == 1 and t["loop_lag_le_5ms"] == 1
    assert t["loop_lag_le_500ms"] == 1 and t["loop_lag_inf"] == 1
    assert t["loop_lag_samples_total"] == 4 and t["loop_lag_max_ms"] == 30000.0
    time.sleep(0.02)
    with caplog.at_level(logging.INFO, "jaato_server.server.loop_watchdog"):
        dog._maybe_report_lag()
        dog._maybe_report_lag()          # nothing new: no second line
    assert [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("LOOP_LAG")] == [
        f"LOOP_LAG: {t}"]


async def test_the_heartbeat_feeds_the_histogram() -> None:
    dog = LoopWatchdog(interval=0.01, threshold=5)
    dog.start()
    try:
        await asyncio.sleep(0.1)
    finally:
        dog.stop()
    assert dog.get_telemetry()["loop_lag_samples_total"] >= 3
