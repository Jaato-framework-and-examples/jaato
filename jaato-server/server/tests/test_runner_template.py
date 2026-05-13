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
