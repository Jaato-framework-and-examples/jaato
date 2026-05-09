"""Tests for ``JaatoServer.set_terminal_width`` forwarding to the
runner-side JaatoSession (Phase 3 §3.3c migration).

After §3.3c precursors landed the ``session.set_terminal_width``
runner-RPC handler + wrapper, ``JaatoServer.set_terminal_width``
now also forwards to the runner so post-seat-flip the runner's
enrichment notification formatting uses the same width as the
daemon.

Tests pin:

- with a runner attached, the daemon-side setter forwards to
  ``runner_rpc.session_set_terminal_width_threadsafe(width, timeout=2)``
- without a runner (Phase 2 cli-only path), the setter doesn't
  attempt the forward (no AttributeError on missing runner_rpc)
- runner-side propagation failure logs but doesn't break the
  daemon-side flow (best-effort)
- backward compat: rpc instances without
  ``session_set_terminal_width_threadsafe`` skip gracefully
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple
from unittest.mock import MagicMock

import pytest

from server.core import JaatoServer


class _FakeRPC:
    def __init__(self) -> None:
        self.set_width_calls: List[Tuple[int, float]] = []
        self.raise_next: Any = None

    def session_set_terminal_width_threadsafe(
        self, width: int, *, timeout: float = 5.0,
    ) -> None:
        self.set_width_calls.append((width, timeout))
        if self.raise_next is not None:
            raise self.raise_next


def _make_server(rpc: Any = None) -> JaatoServer:
    """Build a minimal JaatoServer skeleton suitable for the
    set_terminal_width path.  We bypass the heavy __init__ and
    seed only the attrs the method touches."""
    srv = JaatoServer.__new__(JaatoServer)
    srv._terminal_width = 0
    srv._jaato = None  # no in-process JaatoClient for this test
    srv._formatter_pipeline = None
    srv._agents = {}
    srv._runner_rpc = rpc
    return srv


# ----------------------------------------------------------------------
# Forward path
# ----------------------------------------------------------------------


def test_set_terminal_width_forwards_to_runner_when_attached() -> None:
    rpc = _FakeRPC()
    srv = _make_server(rpc=rpc)

    srv.terminal_width = (120)
    assert srv._terminal_width == 120
    assert rpc.set_width_calls == [(120, 2.0)]


def test_set_terminal_width_uses_short_timeout() -> None:
    """Daemon-side setter passes timeout=2.0 to keep
    presentation-update latency bounded — operators changing
    terminal size shouldn't have the daemon stall on a stuck
    runner."""
    rpc = _FakeRPC()
    srv = _make_server(rpc=rpc)

    srv.terminal_width = (80)
    assert rpc.set_width_calls[0][1] == 2.0


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------


def test_set_terminal_width_runner_failure_does_not_block_daemon() -> None:
    """Forwarding failures (transport timeout, runner crash, etc.)
    must not block the daemon-side state update.  Best-effort —
    the daemon's terminal_width is the source of truth for the
    daemon-side formatter pipelines that are still in use."""
    rpc = _FakeRPC()
    rpc.raise_next = RuntimeError("rpc transport closed")
    srv = _make_server(rpc=rpc)

    # Should not raise.
    srv.terminal_width = (100)
    assert srv._terminal_width == 100  # daemon-side state still updated


def test_set_terminal_width_no_runner_attached_skips_forward() -> None:
    """Phase 2 cli-only paths (no runner spawned) don't attempt
    the forward.  No AttributeError, no spurious log."""
    srv = _make_server(rpc=None)
    # Should not raise.
    srv.terminal_width = (80)
    assert srv._terminal_width == 80


def test_set_terminal_width_old_rpc_without_method_skips_gracefully() -> None:
    """Forward-compat: older RunnerRPCClient instances without
    the new wrapper method don't break the daemon (rolling-
    upgrade scenario).  The daemon-side state still updates."""

    class _OldRPC:
        pass  # no session_set_terminal_width_threadsafe

    srv = _make_server(rpc=_OldRPC())
    srv.terminal_width = (80)
    assert srv._terminal_width == 80
