"""Tests for ``JaatoServer.clear_history`` forwarding to the
runner-side JaatoSession (Phase 3 §7b.1 migration).

After §3.3c precursors landed the ``session.reset`` runner-RPC
handler + wrapper, ``JaatoServer.clear_history`` now also forwards
to the runner so post-seat-flip the runner's history state stays
in sync with the daemon's after a clear.

This is the 4th daemon-side migration in the seat-flip arc
(after shutdown, terminal_width, presentation_context).  Same
write-both pattern: update daemon-side state (unchanged), then
forward to runner with a short timeout, log + skip failures
gracefully.

Tests pin:

- with a runner attached, the daemon-side method forwards to
  ``runner_rpc.session_reset_threadsafe(timeout=2)``
- without a runner (Phase 2 cli-only path), no forward attempt
- runner-side propagation failure logs but doesn't break the
  daemon-side flow (best-effort)
- backward compat: rpc instances without
  ``session_reset_threadsafe`` skip gracefully
- daemon-side state (``_original_inputs`` clear, agent history
  reset) still happens regardless of forward outcome
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

from server.core import JaatoServer


class _FakeRPC:
    def __init__(self) -> None:
        self.reset_calls: List[float] = []
        self.raise_next: Any = None

    def session_reset_threadsafe(self, *, timeout: float = 5.0) -> None:
        self.reset_calls.append(timeout)
        if self.raise_next is not None:
            raise self.raise_next


class _FakeJaatoClient:
    """Stand-in for JaatoClient.  ``reset_session`` is the only
    call site clear_history hits."""

    def __init__(self) -> None:
        self.reset_called: bool = False

    def reset_session(self) -> None:
        self.reset_called = True


def _make_server(rpc: Any = None, jaato: Any = None) -> JaatoServer:
    """Minimal JaatoServer skeleton suitable for clear_history."""
    srv = JaatoServer.__new__(JaatoServer)
    srv._jaato = jaato
    srv._runner_rpc = rpc
    srv._original_inputs = ["sample-input"]
    srv._main_agent_id = "main"
    srv._agents = {}
    srv.emit = lambda event: None  # type: ignore[method-assign]
    return srv


# ----------------------------------------------------------------------
# Forward path
# ----------------------------------------------------------------------


def test_clear_history_forwards_to_runner_when_attached() -> None:
    rpc = _FakeRPC()
    jaato = _FakeJaatoClient()
    srv = _make_server(rpc=rpc, jaato=jaato)

    srv.clear_history()

    # Daemon-side reset still happens.
    assert jaato.reset_called is True
    # Runner-side forward fired with the bounded timeout.
    assert rpc.reset_calls == [2.0]
    # Daemon-side state cleared.
    assert srv._original_inputs == []


def test_clear_history_uses_short_timeout() -> None:
    """Daemon-side method passes timeout=2.0 — clear-history is
    interactive (operator-triggered); shouldn't stall the daemon
    on a stuck runner."""
    rpc = _FakeRPC()
    srv = _make_server(rpc=rpc, jaato=_FakeJaatoClient())

    srv.clear_history()
    assert rpc.reset_calls[0] == 2.0


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------


def test_clear_history_runner_failure_does_not_block_daemon() -> None:
    """Forwarding failures must not block the daemon-side state
    update."""
    rpc = _FakeRPC()
    rpc.raise_next = RuntimeError("runner stuck")
    jaato = _FakeJaatoClient()
    srv = _make_server(rpc=rpc, jaato=jaato)

    # Should not raise.
    srv.clear_history()
    # Daemon-side state STILL cleared.
    assert jaato.reset_called is True
    assert srv._original_inputs == []


def test_clear_history_no_runner_attached_skips_forward() -> None:
    """Phase 2 cli-only paths (no runner spawned) don't attempt
    the forward.  No AttributeError, no spurious log."""
    jaato = _FakeJaatoClient()
    srv = _make_server(rpc=None, jaato=jaato)

    srv.clear_history()
    assert jaato.reset_called is True
    assert srv._original_inputs == []


def test_clear_history_old_rpc_without_method_skips_gracefully() -> None:
    """Forward-compat: rolling-upgrade scenario where the
    RunnerRPCClient on the wire predates the session_reset
    wrapper.  Daemon-side clear still works."""

    class _OldRPC:
        pass  # no session_reset_threadsafe

    jaato = _FakeJaatoClient()
    srv = _make_server(rpc=_OldRPC(), jaato=jaato)

    srv.clear_history()
    assert jaato.reset_called is True
    assert srv._original_inputs == []


def test_clear_history_no_jaato_no_forward_no_crash() -> None:
    """If there's neither a JaatoClient nor a runner — the
    method is a clean no-op on the runtime side, but daemon-side
    state still clears."""
    srv = _make_server(rpc=None, jaato=None)
    srv.clear_history()
    assert srv._original_inputs == []
