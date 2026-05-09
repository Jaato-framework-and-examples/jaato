"""Tests for ``JaatoServer.shutdown`` calling
``runner_rpc.session_shutdown`` BEFORE the transport-close ladder
(Phase 3 §3.3c migration).

Pre-§3.3c: shutdown closed the runner-RPC transport directly,
which SIGTERMed the runner process — plugin on_session_end hooks
ran (or didn't) racing against process termination.  Post-§3.3c
the daemon calls session.shutdown first, giving the runner-side
JaatoSession a clean close_session() chance to fire hooks before
the transport goes away.

Tests pin:

- shutdown calls session_shutdown_threadsafe BEFORE rpc.close
- session_shutdown crash doesn't block the transport close
  (best-effort — robustness over gracefulness)
- shutdown without an attached runner is a no-op (no session_shutdown
  attempt)
- order: session_shutdown precedes rpc.close in the call sequence
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from server.core import JaatoServer


class _FakeRPC:
    """Stand-in for :class:`RunnerRPCClient` exposing the methods
    JaatoServer.shutdown calls."""

    def __init__(self) -> None:
        self.session_shutdown_calls: List[float] = []
        self.close_calls: List[None] = []
        self.session_shutdown_outcome: Any = "sess-x"  # str or Exception
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True,
        )
        self._loop_thread.start()
        self._timeline: List[str] = []

    def session_shutdown_threadsafe(self, *, timeout: float = 10.0) -> str:
        self.session_shutdown_calls.append(timeout)
        self._timeline.append("session_shutdown")
        if isinstance(self.session_shutdown_outcome, Exception):
            raise self.session_shutdown_outcome
        return self.session_shutdown_outcome

    async def close(self) -> None:
        self.close_calls.append(None)
        self._timeline.append("close")

    def stop_loop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2)
        self._loop.close()


def _make_server_with_runner(rpc: _FakeRPC) -> JaatoServer:
    """Build a minimal JaatoServer with the runner-rpc handle
    attached but no real session — exercises the shutdown path."""
    srv = JaatoServer.__new__(JaatoServer)
    srv.registry = None
    srv.permission_plugin = None
    srv._runner_rpc = rpc
    srv._spawned_runner = MagicMock()
    return srv


# ----------------------------------------------------------------------
# Lifecycle ordering: session_shutdown before close
# ----------------------------------------------------------------------


def test_shutdown_calls_session_shutdown_before_transport_close() -> None:
    rpc = _FakeRPC()
    try:
        srv = _make_server_with_runner(rpc)
        srv.shutdown()
        assert rpc._timeline == ["session_shutdown", "close"]
    finally:
        rpc.stop_loop()


def test_shutdown_session_shutdown_called_with_short_timeout() -> None:
    """JaatoServer.shutdown passes timeout=5.0 to keep daemon
    teardown bounded (the runner's ``session.shutdown`` handler
    has its own internal close_session — the wrapper's default
    10s would compound)."""
    rpc = _FakeRPC()
    try:
        srv = _make_server_with_runner(rpc)
        srv.shutdown()
        assert rpc.session_shutdown_calls == [5.0]
    finally:
        rpc.stop_loop()


# ----------------------------------------------------------------------
# Robustness: session_shutdown failure doesn't block close
# ----------------------------------------------------------------------


def test_shutdown_session_shutdown_crash_still_closes_transport() -> None:
    """If session.shutdown raises (transport already half-closed,
    runner crashed mid-call), the daemon must still proceed to
    close the transport — robustness over gracefulness."""
    rpc = _FakeRPC()
    rpc.session_shutdown_outcome = RuntimeError("rpc crashed")
    try:
        srv = _make_server_with_runner(rpc)
        srv.shutdown()
        assert rpc._timeline == ["session_shutdown", "close"]
    finally:
        rpc.stop_loop()


# ----------------------------------------------------------------------
# Backward compat: rpc without session_shutdown_threadsafe still works
# ----------------------------------------------------------------------


def test_shutdown_rpc_without_session_shutdown_method_skips_gracefully() -> None:
    """Older RunnerRPCClient instances without the wrapper method
    don't break daemon shutdown — the daemon-side migration is
    forward-compatible.  Ensures rolling upgrades work."""

    class _OldRPC:
        def __init__(self) -> None:
            self.close_calls: List[None] = []
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, daemon=True,
            )
            self._loop_thread.start()

        async def close(self) -> None:
            self.close_calls.append(None)

        def stop_loop(self):
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=2)
            self._loop.close()

    rpc = _OldRPC()
    try:
        srv = _make_server_with_runner(rpc)
        srv.shutdown()
        # close was called even though session_shutdown_threadsafe
        # wasn't.
        assert rpc.close_calls == [None]
    finally:
        rpc.stop_loop()


# ----------------------------------------------------------------------
# No runner attached: no session_shutdown attempt
# ----------------------------------------------------------------------


def test_shutdown_without_runner_does_nothing_runner_related() -> None:
    """Sessions without a spawned runner (no apparmor opt-in) skip
    both session_shutdown and rpc.close.  Phase 2 cli-only paths
    don't break."""
    srv = JaatoServer.__new__(JaatoServer)
    srv.registry = None
    srv.permission_plugin = None
    srv._runner_rpc = None
    srv._spawned_runner = None
    # Should not raise.
    srv.shutdown()
