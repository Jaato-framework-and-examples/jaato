"""Pool PR 5c: template-ready handshake tests.

PR 5c replaces the daemon's 2s ``time.sleep`` after template spawn
with a proper READY signal from the template.  Template sends
``"READY\\n"`` over the control socket after plugin discovery
completes; ``TemplateManager.spawn`` blocks for it (up to 30s)
before returning.

Tests:
1. ``_wait_for_ready`` returns promptly when the peer writes
   ``"READY\\n"`` to the other end of the control socket.
2. ``_wait_for_ready`` times out cleanly when no signal arrives.
3. ``_wait_for_ready`` handles peer-closed socket (template
   crashed during plugin imports).
4. End-to-end: spawn a real template + verify spawn() actually
   blocked until READY (no time.sleep needed for fork-slot).
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Optional

import pytest

from server.runner_template import TemplateManager


# ----------------------------------------------------------------------
# Unit tests: _wait_for_ready against a controlled socketpair
# ----------------------------------------------------------------------


def _make_tm_with_socket() -> tuple[TemplateManager, socket.socket]:
    """Build a TemplateManager whose control_sock is the daemon end of
    a socketpair.  Returns (mgr, peer_sock) — the peer_sock is the
    "template" side; tests write/close it to simulate template
    behavior."""
    mgr = TemplateManager()
    daemon_sock, peer_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )
    mgr.pid = 99999  # pretend a template is "running"
    mgr.control_sock = daemon_sock
    return mgr, peer_sock


def test_wait_for_ready_returns_promptly_on_ready_signal() -> None:
    """When the template (peer) writes 'READY\\n', ``_wait_for_ready``
    returns within milliseconds (not the full timeout)."""
    mgr, peer_sock = _make_tm_with_socket()
    try:
        # Write READY signal BEFORE the daemon-side wait starts —
        # recv() will pick it up immediately.
        peer_sock.sendall(b"READY\n")

        t0 = time.monotonic()
        mgr._wait_for_ready(timeout=10.0)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, (
            f"_wait_for_ready should return promptly (<1s) when peer "
            f"already sent READY; took {elapsed:.2f}s"
        )
    finally:
        peer_sock.close()
        if mgr.control_sock is not None:
            mgr.control_sock.close()


def test_wait_for_ready_times_out_cleanly() -> None:
    """When peer never sends anything, ``_wait_for_ready`` returns
    after the timeout without raising."""
    mgr, peer_sock = _make_tm_with_socket()
    try:
        t0 = time.monotonic()
        # Use a short timeout to keep the test fast.
        mgr._wait_for_ready(timeout=0.5)
        elapsed = time.monotonic() - t0

        assert 0.4 < elapsed < 1.5, (
            f"_wait_for_ready timeout wall-clock outside expected band; "
            f"observed {elapsed:.2f}s for 0.5s timeout"
        )
    finally:
        peer_sock.close()
        if mgr.control_sock is not None:
            mgr.control_sock.close()


def test_wait_for_ready_handles_peer_close() -> None:
    """When peer closes the socket (template crashed pre-READY),
    ``_wait_for_ready`` returns cleanly with empty recv."""
    mgr, peer_sock = _make_tm_with_socket()
    try:
        # Close the peer end immediately — daemon's recv() returns
        # empty bytes (EOF).
        peer_sock.close()

        t0 = time.monotonic()
        mgr._wait_for_ready(timeout=10.0)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, (
            f"_wait_for_ready should return promptly on peer close; "
            f"took {elapsed:.2f}s"
        )
    finally:
        if mgr.control_sock is not None:
            mgr.control_sock.close()


def test_wait_for_ready_no_control_sock_is_noop() -> None:
    """Defensive: if ``spawn`` didn't populate control_sock (e.g.
    pre-spawn or post-shutdown), ``_wait_for_ready`` returns
    immediately without raising."""
    mgr = TemplateManager()
    # pid + control_sock both None — defensive path.
    t0 = time.monotonic()
    mgr._wait_for_ready(timeout=10.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.1, (
        f"_wait_for_ready with no control_sock should be a no-op; "
        f"took {elapsed:.2f}s"
    )


def test_wait_for_ready_handles_unexpected_first_message() -> None:
    """If peer sends something OTHER than 'READY\\n' first (e.g. a
    FORK_FAILED from a broken template), ``_wait_for_ready`` logs a
    warning + returns.  Daemon proceeds; template may be degraded
    but spawn() doesn't hang."""
    mgr, peer_sock = _make_tm_with_socket()
    try:
        peer_sock.sendall(b"GARBAGE\n")

        t0 = time.monotonic()
        mgr._wait_for_ready(timeout=10.0)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, (
            f"_wait_for_ready should return promptly on unexpected "
            f"first message; took {elapsed:.2f}s"
        )
    finally:
        peer_sock.close()
        if mgr.control_sock is not None:
            mgr.control_sock.close()


# ----------------------------------------------------------------------
# End-to-end: real template + spawn() actually blocks for READY
# ----------------------------------------------------------------------


def test_spawn_blocks_until_template_ready() -> None:
    """End-to-end: ``spawn`` returns AFTER template's plugin
    discovery completes.  Verification: immediately-after-spawn
    ``request_fork_slot`` succeeds without an explicit sleep."""
    mgr = TemplateManager()
    try:
        t0 = time.monotonic()
        mgr.spawn()
        spawn_elapsed = time.monotonic() - t0

        # Template discover empirically takes ~1.5-2s on dev
        # workstations.  Spawn-blocks-for-READY should take at least
        # 0.5s (lower bound — discovery isn't instant) and well
        # under the 30s timeout.
        assert 0.5 < spawn_elapsed < 30.0, (
            f"spawn elapsed {spawn_elapsed:.2f}s — expected 0.5s "
            f"(discover lower-bound) < t < 30s (timeout upper-bound)"
        )

        # Immediately try fork-slot — should succeed because
        # template is guaranteed ready.
        handle = mgr.request_fork_slot()
        assert handle is not None, (
            "request_fork_slot returned None immediately after spawn; "
            "template-ready handshake didn't deliver — was the 2s "
            "sleep removed prematurely?"
        )

        slot_pid, slot_sock = handle
        # Close the slot so test cleanup is tidy.
        slot_sock.close()
        try:
            os.waitpid(slot_pid, 0)
        except ChildProcessError:
            pass
    finally:
        mgr.shutdown()


def test_spawn_idempotent_post_ready() -> None:
    """Double-spawn after the first ready-handshake is still a no-op
    + log warning (existing PR 2 contract preserved)."""
    mgr = TemplateManager()
    try:
        mgr.spawn()
        first_pid = mgr.pid
        mgr.spawn()  # no-op
        assert mgr.pid == first_pid
    finally:
        mgr.shutdown()
