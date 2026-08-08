"""Tests for ``JaatoServer.stop`` forwarding cancellation to the
runner-side JaatoSession.

History (commit chronology):

  §7b.1 (commit fafe90a6): introduced write-both — daemon-side
    ``_jaato.stop()`` + runner-RPC forward.
  §7c step 6.3 (this version): daemon-side leg dropped.  The
    runner-side ``session.request_stop`` RPC is now the only
    source of truth for cancellation.

Tests pin the post-§7c-step-6.3 contract:

- runner reports cancelled (in-flight message): returns True
- runner reports no cancellation (idle): returns False
- runner forward failure: returns False (no crash)
- no runner attached: returns False (no daemon-side fallback)
- old rpc without method: skip gracefully → False
"""

from __future__ import annotations

from typing import Any, List, Tuple

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


def _make_server(rpc: Any = None) -> JaatoServer:
    srv = JaatoServer.__new__(JaatoServer)
    srv._jaato = None  # No daemon-side JaatoClient post-§7c step 6.3
    srv._runner_rpc = rpc
    return srv


# ----------------------------------------------------------------------
# Forward path
# ----------------------------------------------------------------------


def test_stop_returns_true_when_runner_cancels() -> None:
    """Runner has the live message and cancels it → True."""
    rpc = _FakeRPC()
    rpc.cancel_outcome = True

    srv = _make_server(rpc=rpc)
    assert srv.stop() is True
    assert rpc.calls == [("user_stop", 2.0)]


def test_stop_returns_false_when_runner_idle() -> None:
    """Runner reports no in-flight message → False (no-op)."""
    rpc = _FakeRPC()
    rpc.cancel_outcome = False

    srv = _make_server(rpc=rpc)
    assert srv.stop() is False
    # Forward STILL fired (the runner is the source of truth on
    # whether a message is in flight; we always ask).
    assert rpc.calls == [("user_stop", 2.0)]


def test_stop_uses_short_timeout() -> None:
    """timeout=2.0 — interactive operation, shouldn't stall."""
    rpc = _FakeRPC()
    srv = _make_server(rpc=rpc)
    srv.stop()
    assert rpc.calls[0][1] == 2.0


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------


def test_stop_runner_failure_does_not_propagate() -> None:
    """Runner-RPC failures log but don't propagate; method
    returns False."""
    rpc = _FakeRPC()
    rpc.raise_next = RuntimeError("rpc transport closed")

    srv = _make_server(rpc=rpc)
    # Should not raise.
    assert srv.stop() is False


def test_stop_no_runner_returns_false() -> None:
    """No runner attached (e.g. spawn failed) → False (no-op,
    no daemon-side fallback post-§7c step 6.3)."""
    srv = _make_server(rpc=None)
    assert srv.stop() is False


def test_stop_old_rpc_without_method_returns_false() -> None:
    """Forward-compat: rolling-upgrade scenario where the
    RunnerRPCClient predates ``session_request_stop_threadsafe``.
    Skip gracefully → False."""

    class _OldRPC:
        pass

    srv = _make_server(rpc=_OldRPC())
    assert srv.stop() is False
