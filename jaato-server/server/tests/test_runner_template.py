"""Tests for the pre-warm runner template lifecycle (pool PR 2).

Verifies that:
  1. ``TemplateManager.spawn()`` forks + exec's the runner with
     ``--template-mode``, leaving a healthy subprocess in the
     ``pid`` + ``control_sock`` attributes.
  2. The template subprocess actually imports runner-tier plugins
     without crashing (end-to-end spawn-and-survive check).
  3. ``TemplateManager.shutdown()`` cleanly terminates the template
     via the SHUTDOWN control command.
  4. ``is_alive()`` returns the right answer before/after both
     ``spawn()`` and ``shutdown()``.
  5. Idempotency: double-spawn and double-shutdown are safe no-ops.

These are end-to-end tests against the real ``python -m server.runner
--template-mode`` subprocess.  Slow (~3-5s each because we wait for
plugin imports), so a smaller unit-test layer would not catch real
issues like "the runner's plugin discovery crashes in template-mode".
"""

from __future__ import annotations

import os
import time

import pytest

from server.runner_template import TemplateManager


def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.05):
    """Poll ``predicate`` until True or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestTemplateManagerSpawn:
    def test_spawn_creates_subprocess(self):
        mgr = TemplateManager()
        try:
            mgr.spawn()
            assert mgr.pid is not None
            assert mgr.pid > 0
            assert mgr.control_sock is not None
            # The subprocess should be a valid process — /proc/<pid>
            # exists on Linux.
            assert os.path.isdir(f"/proc/{mgr.pid}"), (
                f"template subprocess pid={mgr.pid} not found in /proc"
            )
        finally:
            mgr.shutdown()

    def test_spawn_subprocess_survives_plugin_imports(self):
        """End-to-end pin: the template imports runner-tier plugins
        without crashing.  Waits long enough for the imports to
        complete (plugins logged at INFO level in the template's own
        stderr; we don't read the stderr here, just verify the
        process stays alive past the import-completion window)."""
        mgr = TemplateManager()
        try:
            mgr.spawn()
            # Give the template ~10 seconds to import.  If imports
            # crash, the process exits non-zero and ``is_alive``
            # flips False.
            still_alive = _wait_until(
                lambda: not mgr.is_alive(),
                timeout=10.0,
            )
            # ``still_alive`` is True when the predicate (NOT alive)
            # eventually fires — i.e., the template died.  We expect
            # the template to STILL be alive after 10s.
            assert not still_alive, (
                "template subprocess died within 10s — plugin "
                "imports may have crashed"
            )
            assert mgr.is_alive()
        finally:
            mgr.shutdown()

    def test_spawn_idempotent(self):
        """Double-spawn is a no-op + logs a warning."""
        mgr = TemplateManager()
        try:
            mgr.spawn()
            first_pid = mgr.pid
            mgr.spawn()  # second call — should be a no-op
            assert mgr.pid == first_pid
        finally:
            mgr.shutdown()


class TestTemplateManagerShutdown:
    def test_shutdown_terminates_subprocess(self):
        mgr = TemplateManager()
        mgr.spawn()
        pid = mgr.pid
        mgr.shutdown()
        assert mgr.pid is None
        assert mgr.control_sock is None
        # The process should be gone — /proc/<pid> no longer exists
        # OR is in zombie state (briefly).  Poll until gone.
        gone = _wait_until(
            lambda: not os.path.isdir(f"/proc/{pid}"),
            timeout=5.0,
        )
        assert gone, f"template pid={pid} still present in /proc after shutdown"

    def test_shutdown_idempotent(self):
        """Double-shutdown is a no-op + logs a warning."""
        mgr = TemplateManager()
        mgr.spawn()
        mgr.shutdown()
        mgr.shutdown()  # second call — should be a no-op
        assert mgr.pid is None

    def test_shutdown_before_spawn_is_noop(self):
        """Calling shutdown without ever spawning is safe."""
        mgr = TemplateManager()
        mgr.shutdown()  # no-op; logs warning
        assert mgr.pid is None


class TestTemplateManagerIsAlive:
    def test_is_alive_false_before_spawn(self):
        mgr = TemplateManager()
        assert not mgr.is_alive()

    def test_is_alive_true_after_spawn(self):
        mgr = TemplateManager()
        try:
            mgr.spawn()
            # Give the subprocess a moment to actually start.
            time.sleep(0.5)
            assert mgr.is_alive()
        finally:
            mgr.shutdown()

    def test_is_alive_false_after_shutdown(self):
        mgr = TemplateManager()
        mgr.spawn()
        mgr.shutdown()
        assert not mgr.is_alive()


class TestTemplateManagerRequestForkSlot:
    """Tests for the FORK_SLOT RPC (pool PR 3)."""

    def test_returns_none_before_spawn(self):
        """Before ``spawn()``, no template is running to fork from."""
        mgr = TemplateManager()
        assert mgr.request_fork_slot() is None

    def test_forks_a_real_slot(self):
        """End-to-end: spawn template, fork a slot, verify the slot
        process exists, then SHUTDOWN the slot.

        The slot inherits the template's warm imports (which is the
        whole point of the pool design) — but PR 3 just verifies the
        slot PROCESS exists and accepts SHUTDOWN.  Envelope handling
        is PR 4."""
        mgr = TemplateManager()
        mgr.spawn()
        # Let the template's discover() finish so it's ready to handle
        # FORK_SLOT.  Template-side discover took ~1.7s in PR 2; allow
        # 3s for safety.
        time.sleep(3.0)

        try:
            handle = mgr.request_fork_slot()
            assert handle is not None, (
                "request_fork_slot returned None — template should be "
                "ready by now"
            )
            slot_pid, daemon_end = handle
            assert slot_pid > 0
            # /proc/<slot_pid> should exist on Linux.
            assert os.path.isdir(f"/proc/{slot_pid}"), (
                f"forked slot pid={slot_pid} not in /proc"
            )
            # Send SHUTDOWN to the slot directly.
            daemon_end.sendall(b"SHUTDOWN\n")
            daemon_end.close()
            # Slot should exit cleanly + be reapable.  Note: the
            # slot's parent is the daemon (this test process) because
            # of fork inheritance, so we waitpid here.  A "dead but
            # not reaped" slot stays as a zombie in /proc — checking
            # /proc/<pid> existence is the wrong probe.
            reaped = False
            for _ in range(100):  # 5s budget at 50ms intervals
                try:
                    waited_pid, _ = os.waitpid(slot_pid, os.WNOHANG)
                    if waited_pid == slot_pid:
                        reaped = True
                        break
                except ChildProcessError:
                    reaped = True  # already reaped
                    break
                time.sleep(0.05)
            assert reaped, (
                f"slot pid={slot_pid} did not exit + reap within 5s "
                f"after SHUTDOWN"
            )
        finally:
            mgr.shutdown()

    def test_multiple_slots_distinct_pids(self):
        """Fork 3 slots, verify they're 3 distinct processes."""
        mgr = TemplateManager()
        mgr.spawn()
        time.sleep(3.0)

        handles = []
        try:
            for _ in range(3):
                h = mgr.request_fork_slot()
                assert h is not None
                handles.append(h)
            pids = [h[0] for h in handles]
            assert len(set(pids)) == 3, (
                f"expected 3 distinct slot PIDs; got {pids}"
            )
        finally:
            for slot_pid, sock in handles:
                try:
                    sock.sendall(b"SHUTDOWN\n")
                    sock.close()
                except OSError:
                    pass
                try:
                    os.waitpid(slot_pid, 0)
                except ChildProcessError:
                    pass
            mgr.shutdown()


def _send_recv_echo(sock, request_id: int = 1, payload=None) -> dict:
    """Drive one ``echo`` RPC against a slot's RPC dispatcher.

    Helper for PR 4 tests that verify a forked slot is in
    ``RunnerRPC.serve()`` mode (not the PR 3 command loop).  The
    slot's RPC accepts the same length-prefixed JSON framing the
    daemon-side ``RunnerRPCClient`` uses; this helper sidesteps that
    client (which is async + needs a daemon loop) and writes raw
    frames synchronously.

    Returns the decoded response payload dict.

    Raises:
        TimeoutError: slot didn't respond within 10s — usually means
            the slot is still in PR 3's command loop (which doesn't
            speak length-prefixed JSON) instead of PR 4's RPC mode.
    """
    import json
    from shared.framing import read_frame_sync, write_frame_sync

    payload = payload or {"hello": "slot"}
    request = json.dumps({
        "id": request_id,
        "kind": "request",
        "method": "echo",
        "args": payload,
    })
    sock.settimeout(10.0)
    try:
        write_frame_sync(sock, request)
        raw = read_frame_sync(sock)
    finally:
        sock.settimeout(None)
    if raw is None:
        raise RuntimeError(
            "slot closed the socket before responding to echo — slot "
            "may have crashed or PR 4's RPC adoption regressed"
        )
    return json.loads(raw)


class TestSlotServesRPC:
    """Pool PR 4: forked slots speak RunnerRPC framing on their socket.

    Pre-PR-4 (PR 3) slots ran a line-delimited command loop
    accepting ``SHUTDOWN\\n``.  PR 4 replaced that with the
    ``RunnerRPC.serve()`` body (same as session-mode runners after
    they self-confine).  These tests pin the new behavior end-to-end
    against a real subprocess.
    """

    def test_slot_responds_to_echo_rpc(self):
        """A freshly-forked slot answers an ``echo`` RPC.

        Demonstrates two PR 4 properties at once:
          - The slot's RunnerRPC.serve() loop is actually running.
          - The slot's worker pool dispatches requests + writes
            response frames over the slot socket.

        Without PR 4 the slot would either block on its command-
        loop ``recv`` (returning nothing within the test timeout)
        or send garbage back, both of which fail this assertion.
        """
        mgr = TemplateManager()
        mgr.spawn()
        time.sleep(3.0)
        try:
            handle = mgr.request_fork_slot()
            assert handle is not None
            slot_pid, daemon_end = handle

            try:
                response = _send_recv_echo(
                    daemon_end, request_id=42, payload={"x": 1, "y": "z"},
                )
            finally:
                # Close → slot's serve loop sees EOF → clean exit.
                daemon_end.close()
                try:
                    os.waitpid(slot_pid, 0)
                except ChildProcessError:
                    pass

            assert response.get("id") == 42
            assert response.get("kind") == "response"
            assert response.get("ok") is True
            # echo returns args verbatim under "result".
            assert response.get("result") == {"x": 1, "y": "z"}
        finally:
            mgr.shutdown()


class TestPoolManager:
    """Tests for PoolManager (pool PR 3 — daemon-side pool wrapper)."""

    def test_target_size_zero_disables_pool(self):
        """``target_size=0`` means no pool — ``spawn_initial_slots``
        returns 0 without touching the template."""
        from server.runner_pool import PoolManager

        # Template-manager mock not even needed because target_size=0
        # short-circuits.
        pool = PoolManager(template_manager=None, target_size=0)
        assert pool.spawn_initial_slots() == 0
        assert pool.idle_count() == 0

    def test_spawn_initial_slots_with_live_template(self):
        """End-to-end: spawn template, ask pool to fork 2 slots, verify
        idle_count == 2, then shutdown."""
        from server.runner_pool import PoolManager

        tm = TemplateManager()
        tm.spawn()
        time.sleep(3.0)
        try:
            pool = PoolManager(template_manager=tm, target_size=2)
            forked = pool.spawn_initial_slots()
            assert forked == 2
            assert pool.idle_count() == 2
            pool.shutdown_all()
            assert pool.idle_count() == 0
        finally:
            tm.shutdown()

    def test_acquire_slot_pops_idle(self):
        """``acquire_slot`` returns one idle slot per call; empty pool
        returns None."""
        from server.runner_pool import PoolManager

        tm = TemplateManager()
        tm.spawn()
        time.sleep(3.0)
        try:
            pool = PoolManager(template_manager=tm, target_size=2)
            pool.spawn_initial_slots()
            assert pool.idle_count() == 2

            h1 = pool.acquire_slot()
            assert h1 is not None
            assert pool.idle_count() == 1

            h2 = pool.acquire_slot()
            assert h2 is not None
            assert pool.idle_count() == 0

            h3 = pool.acquire_slot()
            assert h3 is None  # empty

            # Clean up the acquired slots ourselves (PR 4 will do this
            # via session bootstrap; PR 3 the test does it manually).
            for slot_pid, sock in [h1, h2]:
                sock.sendall(b"SHUTDOWN\n")
                sock.close()
                try:
                    os.waitpid(slot_pid, 0)
                except ChildProcessError:
                    pass
        finally:
            tm.shutdown()

    def test_shutdown_all_idempotent(self):
        """Calling ``shutdown_all()`` twice is safe."""
        from server.runner_pool import PoolManager

        tm = TemplateManager()
        tm.spawn()
        time.sleep(3.0)
        try:
            pool = PoolManager(template_manager=tm, target_size=1)
            pool.spawn_initial_slots()
            pool.shutdown_all()
            pool.shutdown_all()  # second call — no-op
            assert pool.idle_count() == 0
        finally:
            tm.shutdown()


class TestPoolReplenishment:
    """Pool PR 4: the background replenishment thread keeps the pool
    topped up between session acquisitions.

    Without replenishment a cascade of N sessions cold-spawns N - pool_size
    runners — defeating the pool.  With replenishment the pool refills
    fast enough that cascade steps past the initial window find a warm
    slot waiting.
    """

    def test_replenishment_refills_pool_after_acquire(self):
        """End-to-end: acquire a slot, verify replenishment forks a
        replacement within a few seconds + restores idle_count to
        target."""
        from server.runner_pool import PoolManager

        tm = TemplateManager()
        tm.spawn()
        time.sleep(3.0)
        try:
            pool = PoolManager(
                template_manager=tm,
                target_size=2,
                replenish_interval=0.1,  # tight loop for fast test
            )
            pool.spawn_initial_slots()
            assert pool.idle_count() == 2

            pool.start_replenishment()
            try:
                # Acquire a slot — pool drops to 1.
                handle = pool.acquire_slot()
                assert handle is not None
                assert pool.idle_count() == 1

                # Replenishment thread should refill within a few
                # seconds (template fork is sub-100ms; we allow 5s
                # to be tolerant of test-host scheduling jitter).
                refilled = _wait_until(
                    lambda: pool.idle_count() >= 2,
                    timeout=5.0,
                )
                assert refilled, (
                    f"replenishment did not refill pool back to target "
                    f"within 5s; idle_count={pool.idle_count()}"
                )
            finally:
                # Clean up the acquired slot.
                slot_pid, sock = handle
                sock.close()
                try:
                    os.waitpid(slot_pid, 0)
                except ChildProcessError:
                    pass
                pool.shutdown_all()
        finally:
            tm.shutdown()

    def test_stop_replenishment_clean(self):
        """``stop_replenishment`` joins the thread within timeout."""
        from server.runner_pool import PoolManager

        tm = TemplateManager()
        tm.spawn()
        time.sleep(3.0)
        try:
            pool = PoolManager(
                template_manager=tm,
                target_size=1,
                replenish_interval=0.1,
            )
            pool.start_replenishment()
            time.sleep(0.3)  # let the thread spin a couple of cycles
            pool.stop_replenishment(timeout=3.0)
            assert pool._replenish_thread is None
        finally:
            tm.shutdown()

    def test_start_replenishment_idempotent(self):
        """A second :meth:`start_replenishment` call is a no-op."""
        from server.runner_pool import PoolManager

        tm = TemplateManager()
        tm.spawn()
        time.sleep(3.0)
        try:
            pool = PoolManager(
                template_manager=tm,
                target_size=1,
                replenish_interval=0.5,
            )
            pool.start_replenishment()
            first_thread = pool._replenish_thread
            pool.start_replenishment()  # second call — no-op
            assert pool._replenish_thread is first_thread
            pool.stop_replenishment()
        finally:
            tm.shutdown()

    def test_target_size_zero_skips_thread(self):
        """``target_size=0`` means no replenishment thread spins up."""
        from server.runner_pool import PoolManager

        pool = PoolManager(
            template_manager=None,
            target_size=0,
            replenish_interval=0.1,
        )
        pool.start_replenishment()
        assert pool._replenish_thread is None
