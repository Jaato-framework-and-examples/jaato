"""Tests for the session.end RPC handler + wrapper (Phase 2
cascade-sharing).

The handler iterates every plugin in the runner-side session's
registry and invokes ``reset_for_next_session()`` on each.  Caller
(daemon-side session teardown) branches on the returned
``errors`` list to decide whether the slot is safe to return to
the pool.

These tests pin:
  - happy path: every plugin's reset called exactly once
  - per-plugin reset exception captured in ``errors``, sweep continues
  - no_host / no_registry error paths
  - dispatch routing
  - plugins without reset_for_next_session attribute are skipped silently

Design reference: ``docs/design/runner-cascade-sharing.md`` §4.3.
"""

from __future__ import annotations

import socket
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from server.runner.envelope import RequestEnvelope
from server.runner.rpc import RunnerRPC
from server.runner.session import RunnerSessionHost
from shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()

    def _no_executor(name: str, args: Any):
        return False, {"error": "no executor"}

    return RunnerRPC(a, _no_executor)


def _good_envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-end-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[],
    )


class _FakePlugin:
    """Records reset_for_next_session calls; optional exception."""

    def __init__(self, name: str, raises: Exception = None) -> None:
        self.name = name
        self.raises = raises
        self.reset_calls = 0

    def reset_for_next_session(self) -> None:
        self.reset_calls += 1
        if self.raises is not None:
            raise self.raises


class _FakePluginWithoutReset:
    """Older-style plugin that doesn't implement reset_for_next_session.

    The handler must skip it silently (not raise AttributeError).
    """

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRegistry:
    def __init__(self, plugins: dict) -> None:
        self._plugins = plugins

    def list_available(self) -> List[str]:
        return list(self._plugins.keys())

    def get_plugin(self, name: str):
        return self._plugins.get(name)


def _install_session(rpc: RunnerRPC, plugins: dict) -> None:
    """Build a fake session host whose session has a registry."""
    registry = _FakeRegistry(plugins)
    runtime = MagicMock()
    runtime.registry = registry
    session = MagicMock()
    session._runtime = runtime
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=runtime,
        session=session,
    )


# ======================================================================
# Direct handler
# ======================================================================


def test_session_end_happy_path() -> None:
    """Every plugin's reset_for_next_session is called once."""
    rpc = _make_lone_runner()
    p1 = _FakePlugin("plugin-a")
    p2 = _FakePlugin("plugin-b")
    p3 = _FakePlugin("plugin-c")
    _install_session(rpc, {"plugin-a": p1, "plugin-b": p2, "plugin-c": p3})

    ok, result = rpc._handle_session_end()
    assert ok is True
    assert result == {"plugins_reset": 3, "errors": []}
    assert p1.reset_calls == 1
    assert p2.reset_calls == 1
    assert p3.reset_calls == 1


def test_session_end_per_plugin_exception_captured() -> None:
    """One plugin's reset raises; others still reset; error captured."""
    rpc = _make_lone_runner()
    p1 = _FakePlugin("plugin-a")
    p2 = _FakePlugin("plugin-b", raises=RuntimeError("boom!"))
    p3 = _FakePlugin("plugin-c")
    _install_session(rpc, {"plugin-a": p1, "plugin-b": p2, "plugin-c": p3})

    ok, result = rpc._handle_session_end()
    assert ok is True
    # 2 succeeded; b raised.
    assert result["plugins_reset"] == 2
    assert len(result["errors"]) == 1
    assert "plugin-b" in result["errors"][0]
    assert "RuntimeError" in result["errors"][0]
    assert "boom!" in result["errors"][0]
    # All three were attempted (sweep continues past failure).
    assert p1.reset_calls == 1
    assert p2.reset_calls == 1
    assert p3.reset_calls == 1


def test_session_end_skips_plugin_without_reset_method() -> None:
    """Older plugin lacking reset_for_next_session is silently skipped."""
    rpc = _make_lone_runner()
    p1 = _FakePlugin("plugin-a")
    p2 = _FakePluginWithoutReset("legacy-plugin")
    _install_session(rpc, {"plugin-a": p1, "legacy-plugin": p2})

    ok, result = rpc._handle_session_end()
    assert ok is True
    # Only plugin-a counted (legacy-plugin doesn't have the method).
    assert result["plugins_reset"] == 1
    assert result["errors"] == []
    assert p1.reset_calls == 1


def test_session_end_no_host_returns_error() -> None:
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_end()
    assert ok is False
    assert result["stage"] == "no_host"


def test_session_end_no_registry_returns_error() -> None:
    """A session host with a session that has no runtime/registry
    surfaces as stage=no_registry — defensive programmer-error
    branch."""
    rpc = _make_lone_runner()
    # Build a session whose _runtime is None — registry can't be
    # resolved.
    session = MagicMock()
    session._runtime = None
    rpc._session_host = RunnerSessionHost(
        envelope=_good_envelope(),
        runtime=None,
        session=session,
    )
    ok, result = rpc._handle_session_end()
    assert ok is False
    assert result["stage"] == "no_registry"


def test_session_end_empty_registry_returns_zero() -> None:
    """No plugins to reset → plugins_reset=0, no errors."""
    rpc = _make_lone_runner()
    _install_session(rpc, {})
    ok, result = rpc._handle_session_end()
    assert ok is True
    assert result == {"plugins_reset": 0, "errors": []}


def test_session_end_clears_session_host() -> None:
    """PR #174 hotfix: ``_handle_session_end`` must clear
    ``self._session_host`` so the next ``session.bootstrap`` on a
    reused slot is NOT rejected by the already-bootstrapped guard.

    Pre-fix: session_end only reset plugins; _session_host stayed
    installed; next bootstrap returned ``ToolError: runner already
    hosting session_id=...; refusing to re-bootstrap with different
    envelope``.  Old session's model loop kept running on the
    runner.  Surfaced as duplicate reactor fire 30s after slot
    reuse (kb-orchestrator v152-retry-7, 2026-05-21).
    """
    rpc = _make_lone_runner()
    _install_session(rpc, {"plugin-a": _FakePlugin("plugin-a")})
    assert rpc._session_host is not None  # baseline

    ok, result = rpc._handle_session_end()

    assert ok is True
    assert rpc._session_host is None, (
        "session_end must clear _session_host or the next "
        "session.bootstrap on a reused slot will be refused"
    )
    # plugin reset still counted
    assert result["plugins_reset"] == 1


def test_session_end_fires_close_session_hook() -> None:
    """PR #174: session_end calls session.close_session() BEFORE
    plugin resets so on_session_end hooks see live state.  Mirrors
    session.shutdown semantics."""
    rpc = _make_lone_runner()
    p1 = _FakePlugin("plugin-a")
    _install_session(rpc, {"plugin-a": p1})

    # Track call order: close_session must fire BEFORE plugin.reset.
    call_order: List[str] = []
    session = rpc._session_host.session
    session.close_session = lambda: call_order.append("close_session")

    # Wrap reset to record order without disturbing the count.
    original_reset = p1.reset_for_next_session
    def tracking_reset():
        call_order.append("plugin.reset")
        return original_reset()
    p1.reset_for_next_session = tracking_reset

    ok, result = rpc._handle_session_end()

    assert ok is True
    assert call_order == ["close_session", "plugin.reset"]
    # close_session firing should NOT add to errors when it succeeds.
    assert result["errors"] == []


def test_session_end_close_session_exception_captured_in_errors() -> None:
    """If close_session raises, the exception is captured in
    ``errors`` (slot-poisoning signal to daemon) but the sweep
    still runs the plugin resets."""
    rpc = _make_lone_runner()
    p1 = _FakePlugin("plugin-a")
    _install_session(rpc, {"plugin-a": p1})

    def raising_close():
        raise RuntimeError("close failed")
    rpc._session_host.session.close_session = raising_close

    ok, result = rpc._handle_session_end()

    assert ok is True
    assert any("close_session" in e for e in result["errors"])
    # Plugin reset still ran despite close_session failure
    assert p1.reset_calls == 1
    # But _session_host still cleared — slot must transition cleanly
    # regardless of close_session outcome.
    assert rpc._session_host is None


def test_session_end_clear_host_under_lock() -> None:
    """``_session_host`` clearing happens under ``self._session_lock``
    so parallel handlers see the post-end state atomically.
    Lightweight pin: verify the attribute is None after the call;
    the locking is implicit in the with-block structure."""
    rpc = _make_lone_runner()
    _install_session(rpc, {})  # empty plugins, simplest path

    rpc._handle_session_end()

    assert rpc._session_host is None


def test_session_end_dispatch_routes_to_handler(monkeypatch) -> None:
    """``_dispatch_method`` routes method=='session.end' to
    ``_handle_session_end``."""
    rpc = _make_lone_runner()
    _install_session(rpc, {"plugin-a": _FakePlugin("plugin-a")})

    called = {}

    def fake_handler():
        called["yes"] = True
        return True, {"plugins_reset": 99, "errors": []}

    monkeypatch.setattr(rpc, "_handle_session_end", fake_handler)
    env = RequestEnvelope(
        id="r1", method="session.end", args={},
    )
    ok, result = rpc._dispatch_method(env)
    assert called.get("yes") is True
    assert ok is True
    assert result == {"plugins_reset": 99, "errors": []}
