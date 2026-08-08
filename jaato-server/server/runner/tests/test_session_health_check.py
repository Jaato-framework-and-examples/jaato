"""Tests for the ``session.health_check`` runner-RPC handler
(Phase 3 §3.3c precursor).

The handler is the smallest non-bootstrap session-method RPC
that proves the bidirectional dispatch surface works for
read-only session probes (independent of the ``tool.execute``
path).  §3.3c will use the same surface to dispatch
``session.send_message`` / ``session.get_history`` / etc. once
the daemon-shell rewrite lands; this commit lays the
foundation by adding the smallest such handler + tests proving
the dispatch round-trips correctly.

Tests pin:

- No bootstrap → ``has_host=False`` + ``ready=False`` +
  ``session_id=""`` + ``tool_count=-1``.
- After successful bootstrap → ``has_host=True`` + matching
  session_id + ``tool_count`` reflects the registry's exposed
  tools.
- Unknown method dispatch is unchanged (sanity check that
  adding the new method didn't break the catch-all error path).
- The handler is read-only: invoking it does NOT mutate the
  session host (idempotent across repeated calls).
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    """Construct a RunnerRPC against a closed-side socket; we exercise
    ``_dispatch_method`` / ``_handle_session_health_check`` directly
    without driving the serve loop."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope(**overrides: Any) -> SessionInitEnvelope:
    base = dict(
        session_id="sess-hc-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )
    base.update(overrides)
    return SessionInitEnvelope(**base)


# ----------------------------------------------------------------------
# No bootstrap → empty status
# ----------------------------------------------------------------------


def test_health_check_returns_empty_status_before_bootstrap() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_health_check()
    assert ok is True
    assert result == {
        "has_host": False,
        "ready": False,
        "session_id": "",
        "tool_count": -1,
    }


def test_health_check_via_dispatch_method_before_bootstrap() -> None:
    """End-to-end through ``_dispatch_method`` (the route the
    actual RPC handler uses).  Verifies the new dispatch entry
    routes correctly, not just that the method exists."""
    rpc = _make_lone_runner()
    env = RequestEnvelope(id=1, method="session.health_check", args={})
    ok, result = rpc._dispatch_method(env)
    assert ok is True
    assert result["has_host"] is False
    assert result["session_id"] == ""


# ----------------------------------------------------------------------
# After bootstrap → populated status
# ----------------------------------------------------------------------


def test_health_check_after_bootstrap_reports_session_id() -> None:
    """When a session host is set, the status reflects it.  We
    install the host directly (bypassing the bootstrap RPC) so
    the test doesn't require provider connectivity / plugin
    discovery — we're testing the health-check handler, not the
    bootstrap path (which has its own tests)."""
    rpc = _make_lone_runner()
    envelope = _good_envelope(session_id="sess-hc-active")
    # session=None mimics the test-stub bootstrap path
    # (runtime_factory=None) where the host exists but the
    # JaatoSession was deliberately not constructed.
    rpc._session_host = RunnerSessionHost(
        envelope=envelope,
        runtime=None,
        session=None,
    )

    ok, result = rpc._handle_session_health_check()
    assert ok is True
    assert result["has_host"] is True
    assert result["ready"] is False  # session is None → not ready
    assert result["session_id"] == "sess-hc-active"
    assert result["tool_count"] == -1  # no session → can't enumerate


def test_health_check_with_session_reports_tool_count() -> None:
    """When the host has a JaatoSession with a runtime + registry,
    the handler enumerates the exposed tool schemas.  Tests the
    enumeration path against a stubbed registry."""
    rpc = _make_lone_runner()
    envelope = _good_envelope(session_id="sess-hc-tools")

    # Build the chain: session._runtime.registry.get_exposed_tool_schemas
    class _FakeRegistry:
        def get_exposed_tool_schemas(self):
            # Return three fake schemas; the count is what the
            # handler reports.
            return [object(), object(), object()]

    class _FakeRuntime:
        registry = _FakeRegistry()

    class _FakeSession:
        _runtime = _FakeRuntime()

    rpc._session_host = RunnerSessionHost(
        envelope=envelope,
        runtime=_FakeRuntime(),
        session=_FakeSession(),
    )

    ok, result = rpc._handle_session_health_check()
    assert ok is True
    assert result["has_host"] is True
    assert result["ready"] is True  # session is not None → ready
    assert result["session_id"] == "sess-hc-tools"
    assert result["tool_count"] == 3


def test_health_check_swallows_registry_enumeration_errors() -> None:
    """The probe must never raise — registry enumeration crashes
    return ``tool_count=-1`` rather than propagating.  Otherwise
    a buggy plugin breaks the daemon's ability to even probe the
    runner's status."""
    rpc = _make_lone_runner()
    envelope = _good_envelope(session_id="sess-hc-broken")

    class _FailingRegistry:
        def get_exposed_tool_schemas(self):
            raise RuntimeError("registry boom")

    class _FakeRuntime:
        registry = _FailingRegistry()

    class _FakeSession:
        _runtime = _FakeRuntime()

    rpc._session_host = RunnerSessionHost(
        envelope=envelope,
        runtime=_FakeRuntime(),
        session=_FakeSession(),
    )

    # Must not raise.
    ok, result = rpc._handle_session_health_check()
    assert ok is True
    assert result["tool_count"] == -1
    # Other fields still populated correctly.
    assert result["has_host"] is True
    assert result["ready"] is True


# ----------------------------------------------------------------------
# Read-only invariant
# ----------------------------------------------------------------------


def test_health_check_is_idempotent_across_repeated_calls() -> None:
    """The handler must not mutate the host on each call —
    repeated invocations return the same status (modulo any
    external mutation)."""
    rpc = _make_lone_runner()
    envelope = _good_envelope(session_id="sess-idem")
    rpc._session_host = RunnerSessionHost(
        envelope=envelope,
        runtime=None,
        session=None,
    )

    results = [rpc._handle_session_health_check() for _ in range(5)]
    # All five (ok, status) tuples are identical.
    assert all(r == results[0] for r in results)


def test_unknown_method_still_returns_error() -> None:
    """Sanity: adding session.health_check didn't break the
    catch-all unknown-method error path."""
    rpc = _make_lone_runner()
    env = RequestEnvelope(
        id=99, method="session.does_not_exist", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is False
    assert "unknown method" in result["error"]
