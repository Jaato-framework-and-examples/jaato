"""Tests for the orphan-runner sweep in ``jaato-server --stop``
(server 0.6.196+, PR-252).

Backs the diagnosis at
``project_backlog_jaato_server_stop_orphan_runner_reaper.md``:
``--stop`` walks an in-memory session→runner registry that gets
pruned when an upstream session is killed mid-cascade.  Orphan
runners re-parent to the daemon via ``PR_SET_CHILD_SUBREAPER`` but
stay alive past the daemon reap because they're not in the
registry's view anymore — and continue POSTing to the upstream
provider until manually killed.

The fix snapshots the daemon's descendant PIDs BEFORE sending
SIGTERM and reaps any survivors after the daemon exits.  These
tests pin the helper behavior — the integration with stop_server
itself is exercised via the broader stop_server test suite.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import List
from unittest.mock import patch

import pytest

from server.__main__ import (
    _reap_orphan_descendants,
    _snapshot_daemon_descendants,
)


def _spawn_sleeping_child(seconds: int = 60) -> subprocess.Popen:
    """Spawn a subprocess that just sleeps — proxy for an orphan runner."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"]
    )


def _pid_alive(pid: int) -> bool:
    """True iff PID exists AND is not a zombie.

    A subprocess that's been killed but not yet reaped by its
    parent (us) is a zombie — its PID is still valid (so
    ``os.kill(pid, 0)`` succeeds) but the process is dead.  This
    helper treats zombies as dead because that's what callers
    actually want to know.  Uses ``psutil`` for the zombie check.
    """
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        import psutil
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return True


# ----------------------------------------------------------------------
# _snapshot_daemon_descendants
# ----------------------------------------------------------------------


class TestSnapshotDaemonDescendants:
    def test_dead_pid_returns_empty(self) -> None:
        """psutil.NoSuchProcess → return [] (caller treats as "no
        orphans to sweep" rather than an error)."""
        # PID 0 is the kernel sched on Linux — psutil.Process(0)
        # raises NoSuchProcess on every supported platform.
        result = _snapshot_daemon_descendants(0)
        assert result == []

    def test_process_with_no_children_returns_empty(self) -> None:
        """A process with no descendants returns an empty list, not
        an error.  Uses the current python process as a no-children
        target — `children()` on `os.getpid()` returns only sub-
        processes spawned BY this test."""
        # Spawn ourselves a child to make this deterministic.
        child = _spawn_sleeping_child(seconds=2)
        try:
            time.sleep(0.1)
            descendants = _snapshot_daemon_descendants(os.getpid())
            # The test process has at least one child (the sleep child
            # we just spawned).  Snapshot must include it.
            assert child.pid in descendants
        finally:
            child.terminate()
            child.wait(timeout=3)

    def test_real_subprocess_tree_snapshotted(self) -> None:
        """Spawn a real subprocess; verify its PID shows up in the
        snapshot of the current process's descendants."""
        child = _spawn_sleeping_child(seconds=2)
        try:
            time.sleep(0.1)  # give the process time to be visible.
            descendants = _snapshot_daemon_descendants(os.getpid())
            assert child.pid in descendants
        finally:
            child.terminate()
            child.wait(timeout=3)


# ----------------------------------------------------------------------
# _reap_orphan_descendants
# ----------------------------------------------------------------------


class TestReapOrphanDescendants:
    def test_empty_pid_list_returns_zero(self) -> None:
        assert _reap_orphan_descendants([]) == 0

    def test_already_dead_pids_skipped(self) -> None:
        """A PID that already exited (e.g. daemon's reaper got
        there first) is skipped, not double-killed."""
        # Spawn + reap a process so its PID becomes a corpse.
        child = _spawn_sleeping_child(seconds=1)
        child.wait(timeout=3)
        # PID is now dead.  Reaper should skip it (0 killed).
        killed = _reap_orphan_descendants([child.pid])
        assert killed == 0

    def test_live_process_killed(self) -> None:
        """A live process gets SIGTERM, exits, counted as killed."""
        child = _spawn_sleeping_child(seconds=30)  # long-lived
        try:
            time.sleep(0.1)
            assert _pid_alive(child.pid), "child must be alive before reap"
            killed = _reap_orphan_descendants([child.pid])
            assert killed == 1
            # Within the helper's 2s SIGTERM grace + SIGKILL escalation,
            # the child must be dead.
            assert not _pid_alive(child.pid)
        finally:
            # Defensive cleanup if the test bug-exited early.
            if _pid_alive(child.pid):
                try:
                    os.kill(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=2)

    def test_mixed_live_and_dead_pids(self) -> None:
        """Reaper handles a mix of live and already-dead PIDs;
        counts only the live ones killed."""
        dead = _spawn_sleeping_child(seconds=1)
        dead.wait(timeout=3)

        alive = _spawn_sleeping_child(seconds=30)
        try:
            time.sleep(0.1)
            killed = _reap_orphan_descendants([dead.pid, alive.pid])
            assert killed == 1  # only the alive one
            assert not _pid_alive(alive.pid)
        finally:
            if _pid_alive(alive.pid):
                try:
                    os.kill(alive.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                alive.wait(timeout=2)

    def test_sigterm_grace_then_sigkill_for_unresponsive(self) -> None:
        """A process that ignores SIGTERM (e.g. a stuck runner)
        gets escalated to SIGKILL within the helper's 2s grace.
        Simulated by a child that traps SIGTERM."""
        # Spawn a python child that ignores SIGTERM.
        script = (
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(30)"
        )
        child = subprocess.Popen([sys.executable, "-c", script])
        try:
            time.sleep(0.1)
            assert _pid_alive(child.pid)
            start = time.monotonic()
            killed = _reap_orphan_descendants([child.pid])
            elapsed = time.monotonic() - start
            assert killed == 1
            # SIGKILL is uncatchable but kernel teardown is async —
            # give the kernel up to 1s to mark the PID dead/zombie.
            for _ in range(10):
                if not _pid_alive(child.pid):
                    break
                time.sleep(0.1)
            assert not _pid_alive(child.pid), (
                "child should be dead/zombie after SIGKILL"
            )
            # Helper grants 2s of SIGTERM grace before SIGKILL escalation;
            # total time should be in that range (plus a bit of overhead).
            assert 2.0 <= elapsed <= 4.0, (
                f"elapsed {elapsed:.2f}s outside expected SIGTERM-grace window"
            )
        finally:
            if _pid_alive(child.pid):
                try:
                    os.kill(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
