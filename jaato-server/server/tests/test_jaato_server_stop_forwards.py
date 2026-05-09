"""Tests for ``JaatoServer.stop`` forwarding cancellation to the
runner-side JaatoSession (Phase 3 §7b.1 migration).

After §3.3c precursors landed the ``session.request_stop`` runner-
RPC handler + wrapper, ``JaatoServer.stop`` now ALSO signals the
runner so whichever side holds the in-flight message receives
the cancel.  Pre-§7c the daemon-side is authoritative; post-§7c
the runner-side will be.

Tests pin:

- with both sides processing: cancel goes to both; returns True
- daemon-side cancels (running) + runner doesn't (no message):
  returns True
- daemon-side doesn't (idle) + runner cancels: returns True (the
  post-seat-flip case; runner is authoritative)
- neither side processing: returns False (no-op)
- runner forward failure: daemon-side outcome wins; no crash
- no runner attached: existing daemon-only path (Phase 2 cli)
- old rpc without method: skip gracefully
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from server.core import JaatoServer


class _FakeRPC:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, float]] = []
        self.cancel_outcome: bool = False
        self.raise_next: Any = None

    def session_request_stop_threadsafe(
        self, reason: str = "", *, timeout: float = 5.0,
    ) -> bool:
        self.calls.append((reason, timeout))
        if self.raise_next is not None:
            raise self.raise_next
        return self.cancel_outcome


class _FakeJaato:
    def __init__(self) -> None:
        self.is_processing: bool = False
        self.stop_called: bool = False
        self.stop_outcome: bool = False

    def stop(self) -> bool:
        self.stop_called = True
        return self.stop_outcome


def _make_server(rpc: Any = None, jaato: Any = None) -> JaatoServer:
    srv = JaatoServer.__new__(JaatoServer)
    srv._jaato = jaato
    srv._runner_rpc = rpc
    return srv


# ----------------------------------------------------------------------
# Both sides processing → cancel both
# ----------------------------------------------------------------------


def test_stop_cancels_both_when_both_processing() -> None:
    rpc = _FakeRPC()
    rpc.cancel_outcome = True
    jaato = _FakeJaato()
    jaato.is_processing = True
    jaato.stop_outcome = True

    srv = _make_server(rpc=rpc, jaato=jaato)
    result = srv.stop()

    assert result is True
    assert jaato.stop_called is True
    assert rpc.calls == [("user_stop", 2.0)]


def test_stop_returns_true_when_daemon_cancels_and_runner_does_not() -> None:
    """Pre-§7c case: daemon-side message is the live one; runner-
    side is idle (its session was bootstrapped but never
    exercised).  Daemon's cancel wins."""
    rpc = _FakeRPC()
    rpc.cancel_outcome = False  # runner reports no in-flight
    jaato = _FakeJaato()
    jaato.is_processing = True
    jaato.stop_outcome = True

    srv = _make_server(rpc=rpc, jaato=jaato)
    assert srv.stop() is True


def test_stop_returns_true_when_runner_cancels_and_daemon_does_not() -> None:
    """Post-seat-flip case: daemon-side JaatoSession is gone (or
    just idle); runner-side has the live message.  Runner's
    cancel wins."""
    rpc = _FakeRPC()
    rpc.cancel_outcome = True
    jaato = _FakeJaato()
    jaato.is_processing = False  # daemon idle
    jaato.stop_outcome = False  # would not fire if called

    srv = _make_server(rpc=rpc, jaato=jaato)
    result = srv.stop()
    assert result is True
    # Daemon's stop() NOT called because is_processing=False.
    assert jaato.stop_called is False
    # Runner's stop WAS called.
    assert rpc.calls == [("user_stop", 2.0)]


def test_stop_returns_false_when_neither_processing() -> None:
    rpc = _FakeRPC()
    rpc.cancel_outcome = False
    jaato = _FakeJaato()
    jaato.is_processing = False

    srv = _make_server(rpc=rpc, jaato=jaato)
    assert srv.stop() is False


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------


def test_stop_runner_failure_does_not_block_daemon() -> None:
    rpc = _FakeRPC()
    rpc.raise_next = RuntimeError("rpc transport closed")
    jaato = _FakeJaato()
    jaato.is_processing = True
    jaato.stop_outcome = True

    srv = _make_server(rpc=rpc, jaato=jaato)
    # Should not raise.
    result = srv.stop()
    # Daemon-side cancel still fires + reports True.
    assert jaato.stop_called is True
    assert result is True


def test_stop_no_runner_falls_back_to_daemon_only() -> None:
    """Phase 2 cli-only path: no runner attached.  Existing
    behavior preserved (return whatever daemon-side stop returns)."""
    jaato = _FakeJaato()
    jaato.is_processing = True
    jaato.stop_outcome = True

    srv = _make_server(rpc=None, jaato=jaato)
    assert srv.stop() is True
    assert jaato.stop_called is True


def test_stop_no_runner_no_daemon_message_returns_false() -> None:
    jaato = _FakeJaato()
    jaato.is_processing = False

    srv = _make_server(rpc=None, jaato=jaato)
    assert srv.stop() is False
    assert jaato.stop_called is False


def test_stop_old_rpc_without_method_skips_gracefully() -> None:
    """Forward-compat: rolling-upgrade scenario."""

    class _OldRPC:
        pass

    jaato = _FakeJaato()
    jaato.is_processing = True
    jaato.stop_outcome = True

    srv = _make_server(rpc=_OldRPC(), jaato=jaato)
    assert srv.stop() is True
    assert jaato.stop_called is True
