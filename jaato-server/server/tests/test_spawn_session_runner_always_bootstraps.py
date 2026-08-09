"""Regression test for Phase 3 §7c step 1.

The IPC ``_spawn_session_runner`` helper used to gate the
``rpc.bootstrap_session_threadsafe`` envelope dispatch on the
``JAATO_RUNNER_HOSTS_SESSION`` env-var.  Since §7c step 1 the
flag is removed and the bootstrap dispatches unconditionally.

Pins the unconditional dispatch + the failure-tolerant shape:

- Bootstrap fires regardless of env state.
- A bootstrap exception is logged but does not propagate
  (the daemon-side JaatoSession is still authoritative
  during the §7c rollout window).
- The flag, if set, has no effect (it's been removed).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from server.__main__ import _spawn_session_runner


class _FakeRPC:
    """Mock RunnerRPCClient — captures bootstrap calls."""

    def __init__(self, *, raise_on_bootstrap: Exception = None) -> None:
        self.bootstrap_calls: List[Any] = []
        self._raise = raise_on_bootstrap

    def bootstrap_session_threadsafe(self, envelope, *, timeout=30.0):
        self.bootstrap_calls.append((envelope, timeout))
        if self._raise is not None:
            raise self._raise
        return {"ok": True}


def _make_server() -> Any:
    """Minimal JaatoServer-shaped stub.  ``_spawn_session_runner``
    needs ``runner_rpc`` (set by spawn_session_runner), ``_profile`` /
    ``config_root`` (read by _build_session_envelope), and
    ``mark_runner_ready`` — which ``dispatch_bootstrap_envelope`` calls
    once the bootstrap RPC settles, on BOTH the success and the
    daemon-authoritative failure path, so a reused warm pool slot does
    not strand the send/push gates on a readiness timeout.

    ``mark_runner_ready`` records rather than no-ops: whether bootstrap
    signalled readiness is exactly the property these tests exist to pin.
    """
    srv = SimpleNamespace(
        runner_rpc=None,
        _profile=None,
        config_root=None,
        runner_ready_marks=0,
    )
    def _mark_runner_ready() -> None:
        srv.runner_ready_marks += 1
    srv.mark_runner_ready = _mark_runner_ready
    return srv


def _patch_spawn(monkeypatch, fake_rpc: _FakeRPC) -> Dict[str, Any]:
    """Patch ``server.runner_spawn.spawn_session_runner`` to wire
    ``server.runner_rpc`` to ``fake_rpc`` instead of forking a real
    runner subprocess.  Returns a dict capturing the spawn call's
    kwargs for assertion."""
    captured: Dict[str, Any] = {}

    def _stub_spawn(*, server, **kwargs):
        captured.update(kwargs)
        captured["server"] = server
        server.runner_rpc = fake_rpc

    monkeypatch.setattr(
        "server.runner_spawn.spawn_session_runner", _stub_spawn,
    )
    return captured


# ----------------------------------------------------------------------
# Unconditional dispatch
# ----------------------------------------------------------------------


def test_bootstrap_dispatched_when_flag_unset(monkeypatch) -> None:
    """Pre-§7c-step-1 the flag had to be set for bootstrap to fire.
    Post-§7c-step-1 it fires regardless."""
    monkeypatch.delenv("JAATO_RUNNER_HOSTS_SESSION", raising=False)
    fake_rpc = _FakeRPC()
    _patch_spawn(monkeypatch, fake_rpc)

    server = _make_server()
    _spawn_session_runner(
        server=server,
        session_id="sess-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        daemon_loop=None,
    )

    assert len(fake_rpc.bootstrap_calls) == 1
    envelope, timeout = fake_rpc.bootstrap_calls[0]
    assert envelope.session_id == "sess-1"
    assert envelope.workspace_path == "/tmp/ws"
    assert envelope.profile_name == "cli_test"
    assert timeout == 30.0


def test_bootstrap_dispatched_when_flag_set(monkeypatch) -> None:
    """The flag is dead — setting it has no effect; bootstrap
    fires the same way (i.e. once, with the same envelope)."""
    monkeypatch.setenv("JAATO_RUNNER_HOSTS_SESSION", "true")
    fake_rpc = _FakeRPC()
    _patch_spawn(monkeypatch, fake_rpc)

    server = _make_server()
    _spawn_session_runner(
        server=server,
        session_id="sess-2",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        daemon_loop=None,
    )

    assert len(fake_rpc.bootstrap_calls) == 1


@pytest.mark.parametrize("flag_value", ["", "false", "0", "no", "off"])
def test_bootstrap_dispatched_for_falsy_flag_values(
    monkeypatch, flag_value: str,
) -> None:
    """Pre-§7c-step-1 these values would suppress the bootstrap.
    Post-§7c-step-1 every value (including falsy ones) results in
    a bootstrap dispatch — the flag is no longer consulted."""
    monkeypatch.setenv("JAATO_RUNNER_HOSTS_SESSION", flag_value)
    fake_rpc = _FakeRPC()
    _patch_spawn(monkeypatch, fake_rpc)

    server = _make_server()
    _spawn_session_runner(
        server=server,
        session_id="sess-3",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        daemon_loop=None,
    )

    assert len(fake_rpc.bootstrap_calls) == 1


# ----------------------------------------------------------------------
# Failure tolerance
# ----------------------------------------------------------------------


def test_bootstrap_failure_does_not_propagate(monkeypatch) -> None:
    """Bootstrap exceptions are caught + logged; the spawn helper
    returns successfully so the daemon-side JaatoSession (still
    authoritative pre-seat-flip) can proceed."""
    monkeypatch.delenv("JAATO_RUNNER_HOSTS_SESSION", raising=False)
    fake_rpc = _FakeRPC(raise_on_bootstrap=RuntimeError("envelope rejected"))
    _patch_spawn(monkeypatch, fake_rpc)

    server = _make_server()
    # Should NOT raise.
    _spawn_session_runner(
        server=server,
        session_id="sess-4",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        daemon_loop=None,
    )

    # Bootstrap was attempted exactly once.
    assert len(fake_rpc.bootstrap_calls) == 1
