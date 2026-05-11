"""Tests for ``server.runner_spawn.dispatch_bootstrap_envelope``
(Phase 3 §7c step 2).

The helper is the shared bootstrap-dispatch primitive used by
both IPC + WS spawn paths.  Tests pin:

- Bootstrap fires on a normal server with a runner_rpc handle.
- Bootstrap is a no-op (no exception) when ``server.runner_rpc``
  is None (e.g., spawn failed before this helper ran).
- Bootstrap failure does NOT propagate (logged at WARNING).
- The helper threads the timeout kwarg through.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

from server.runner_spawn import dispatch_bootstrap_envelope


class _FakeRPC:
    def __init__(self, *, raise_on_bootstrap: Exception = None) -> None:
        self.bootstrap_calls: List[Any] = []
        self._raise = raise_on_bootstrap

    def bootstrap_session_threadsafe(self, envelope, *, timeout=30.0):
        self.bootstrap_calls.append((envelope, timeout))
        if self._raise is not None:
            raise self._raise
        return {"ok": True}


def _make_server(rpc: _FakeRPC = None) -> Any:
    """Build a minimal JaatoServer-shaped stub with the attrs the
    helper reads.  ``runner_rpc`` is the only required attr;
    ``_profile`` + ``config_root`` flow through to
    ``build_session_envelope`` which is exercised separately in
    ``test_build_session_envelope.py``."""
    return SimpleNamespace(
        runner_rpc=rpc,
        _profile=None,
        config_root=None,
    )


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_dispatch_fires_with_runner_rpc_set() -> None:
    rpc = _FakeRPC()
    server = _make_server(rpc=rpc)

    dispatch_bootstrap_envelope(
        server=server,
        session_id="sess-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
    )

    assert len(rpc.bootstrap_calls) == 1
    envelope, timeout = rpc.bootstrap_calls[0]
    assert envelope.session_id == "sess-1"
    assert envelope.workspace_path == "/tmp/ws"
    assert envelope.profile_name == "cli_test"
    assert timeout == 30.0


def test_dispatch_threads_timeout_kwarg() -> None:
    rpc = _FakeRPC()
    server = _make_server(rpc=rpc)

    dispatch_bootstrap_envelope(
        server=server,
        session_id="sess-2",
        workspace_path="/tmp/ws",
        profile_name="x",
        timeout=5.0,
    )

    _, timeout = rpc.bootstrap_calls[0]
    assert timeout == 5.0


# ----------------------------------------------------------------------
# Defensive: spawn failed (runner_rpc still None)
# ----------------------------------------------------------------------


def test_dispatch_noops_when_runner_rpc_is_none() -> None:
    """If ``server.runner_rpc`` is None (spawn helper failed before
    this was called), dispatch silently no-ops — no exception, no
    crash.  Caller's spawn-failure path already logged the spawn
    error; the bootstrap dispatch just has nothing to send to."""
    server = _make_server(rpc=None)

    # Should not raise.
    dispatch_bootstrap_envelope(
        server=server,
        session_id="sess-3",
        workspace_path="/tmp/ws",
        profile_name="x",
    )


# ----------------------------------------------------------------------
# Failure tolerance
# ----------------------------------------------------------------------


def test_dispatch_swallows_bootstrap_exception() -> None:
    """Bootstrap RPC failure logs at WARNING but does NOT propagate
    — the daemon-side JaatoSession is still authoritative during
    the §7c rollout window."""
    rpc = _FakeRPC(raise_on_bootstrap=RuntimeError("envelope rejected"))
    server = _make_server(rpc=rpc)

    # Should not raise.
    dispatch_bootstrap_envelope(
        server=server,
        session_id="sess-4",
        workspace_path="/tmp/ws",
        profile_name="x",
    )

    # Bootstrap was attempted exactly once.
    assert len(rpc.bootstrap_calls) == 1
