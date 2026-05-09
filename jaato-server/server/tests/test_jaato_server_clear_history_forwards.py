"""Tests for ``JaatoServer.clear_history`` forwarding to the
runner-side JaatoSession.

History (commit chronology):

  §7b.1 (commit 8cbb8ba2): introduced write-both — daemon-side
    ``_jaato.reset_session()`` + runner-RPC forward.
  §7c step 6.3 (this version): daemon-side leg dropped.  The
    runner-side ``session.reset`` RPC is now the only source of
    truth for conversation-history state.

Tests pin the post-§7c-step-6.3 contract:

- with a runner attached, the daemon-side method forwards to
  ``runner_rpc.session_reset_threadsafe(timeout=2)``
- without a runner, no forward attempt + no crash
- runner-side propagation failure logs but doesn't break the
  daemon-side AgentState clear
- backward compat: rpc instances without
  ``session_reset_threadsafe`` skip gracefully
- daemon-side AgentState (``_original_inputs`` clear, agent
  history reset) still happens regardless of forward outcome
"""

from __future__ import annotations

from typing import Any, List

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


def _make_server(rpc: Any = None) -> JaatoServer:
    """Minimal JaatoServer skeleton suitable for clear_history."""
    srv = JaatoServer.__new__(JaatoServer)
    srv._jaato = None  # No daemon-side JaatoClient post-§7c step 6.3
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
    srv = _make_server(rpc=rpc)

    srv.clear_history()

    # Runner-side forward fired with the bounded timeout.
    assert rpc.reset_calls == [2.0]
    # Daemon-side AgentState cleared (independent of forward outcome).
    assert srv._original_inputs == []


def test_clear_history_uses_short_timeout() -> None:
    """Daemon-side method passes timeout=2.0 — clear-history is
    interactive (operator-triggered); shouldn't stall the daemon
    on a stuck runner."""
    rpc = _FakeRPC()
    srv = _make_server(rpc=rpc)

    srv.clear_history()
    assert rpc.reset_calls[0] == 2.0


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------


def test_clear_history_runner_failure_does_not_block_daemon() -> None:
    """Forwarding failures must not block the daemon-side
    AgentState clear."""
    rpc = _FakeRPC()
    rpc.raise_next = RuntimeError("runner stuck")
    srv = _make_server(rpc=rpc)

    # Should not raise.
    srv.clear_history()
    # Daemon-side AgentState STILL cleared.
    assert srv._original_inputs == []


def test_clear_history_no_runner_attached_skips_forward() -> None:
    """No runner attached (e.g. spawn failed) — clear_history is
    a clean no-op on the RPC side; daemon-side AgentState still
    clears."""
    srv = _make_server(rpc=None)

    srv.clear_history()
    assert srv._original_inputs == []


def test_clear_history_old_rpc_without_method_skips_gracefully() -> None:
    """Forward-compat: rolling-upgrade scenario where the
    RunnerRPCClient on the wire predates the session_reset
    wrapper.  Daemon-side AgentState still clears."""

    class _OldRPC:
        pass  # no session_reset_threadsafe

    srv = _make_server(rpc=_OldRPC())

    srv.clear_history()
    assert srv._original_inputs == []
