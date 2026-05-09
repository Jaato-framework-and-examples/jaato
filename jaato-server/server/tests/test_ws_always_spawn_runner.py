"""Tests for the WS-server apparmor-pre-init-hook always-spawn
behavior (Phase 3 §7a WS counterpart).

Pre-§7a: the WS hook gated runner-spawn behind apparmor
availability — when the kernel module was missing, no runner
spawned and tool execution stayed in-process.

Post-§7a: the hook spawns the runner unconditionally for any
WS-provisioned session (workspace under WS-server's root).
Apparmor confinement is layered atop iff the host has the
kernel module + provisioning succeeds.

This file exercises the WS hook's behavior under three apparmor
states (unavailable / available+success / available+failure) and
the gate paths (workspace not under WS root, no daemon loop).

The shape mirrors the IPC test ``test_always_spawn_runner.py``
but the surface is ``websocket.JaatoWSServer.set_command_router``
(which constructs + registers the hook) rather than the
SessionManager method.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _FakeAppArmor:
    """Stand-in for ``server.apparmor.AppArmorManager`` exposing
    only the methods the WS hook calls."""

    def __init__(self) -> None:
        self.available = True
        self.provision_outcome = True
        self.provision_calls: List[Tuple[str, str]] = []

    def is_available(self) -> bool:
        return self.available

    def provision_profile(
        self, session_id: str, workspace_path: str,
    ) -> bool:
        self.provision_calls.append((session_id, workspace_path))
        return self.provision_outcome

    def get_profile_name(self, session_id: str) -> str:
        return f"jaato-ws-{session_id}"


class _FakeSM:
    def __init__(self) -> None:
        self.pre_init_hooks: List[Any] = []
        self.session_hooks: List[Any] = []

    def add_pre_initialize_hook(self, hook: Any) -> None:
        self.pre_init_hooks.append(hook)

    def add_session_hook(self, hook: Any) -> None:
        self.session_hooks.append(hook)


class _FakeRouter:
    def __init__(self) -> None:
        self._session_manager = _FakeSM()


def _make_ws_server(
    workspace_root: str,
    apparmor: Optional[_FakeAppArmor],
    daemon_loop: Any = "<loop>",
):
    """Construct a JaatoWSServer skeleton with just the attrs the
    hook reads.  Bypasses the heavy __init__ since we're testing
    the hook in isolation."""
    from server.websocket import JaatoWSServer
    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._apparmor = apparmor
    ws._cgroups = None
    ws._workspace_root = workspace_root
    ws._event_loop = daemon_loop
    return ws


def _register_and_get_hook(ws: Any) -> Any:
    """Drive ``set_command_router`` to register the hook + return
    the registered closure for direct invocation."""
    from server.websocket import JaatoWSServer
    router = _FakeRouter()
    JaatoWSServer.set_command_router(ws, router)
    assert router._session_manager.pre_init_hooks, "hook not registered"
    return router._session_manager.pre_init_hooks[0]


@pytest.fixture(autouse=True)
def _patch_spawn():
    """Replace ``spawn_session_runner`` with a stub recorder."""
    spawn_calls: List[Dict[str, Any]] = []
    spawn_outcome: Dict[str, Any] = {"raise": None}

    def _fake_spawn(**kwargs: Any) -> None:
        spawn_calls.append(kwargs)
        if spawn_outcome["raise"] is not None:
            raise spawn_outcome["raise"]

    with patch("server.runner_spawn.spawn_session_runner", _fake_spawn):
        yield {
            "spawn_calls": spawn_calls,
            "spawn_outcome": spawn_outcome,
        }


def _server_stub() -> Any:
    return type("_FakeJaatoServer", (), {})()


# ----------------------------------------------------------------------
# Always-spawn invariant: WS hook spawns regardless of apparmor
# ----------------------------------------------------------------------


def test_ws_hook_spawns_unconfined_when_apparmor_unavailable(
    tmp_path, _patch_spawn,
) -> None:
    """§7a invariant: WS-provisioned session with apparmor
    unavailable still spawns a runner — disable_confine=True,
    empty profile_name."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    apparmor = _FakeAppArmor()
    apparmor.available = False  # kernel module missing
    ws = _make_ws_server(str(ws_root), apparmor)
    hook = _register_and_get_hook(ws)

    hook(_server_stub(), "s-no-aa", str(sess_dir), client_id=None)

    spawn_calls = _patch_spawn["spawn_calls"]
    assert len(spawn_calls) == 1, "spawn must fire on no-aa path"
    call = spawn_calls[0]
    assert call["session_id"] == "s-no-aa"
    assert call["workspace_path"] == str(sess_dir)
    assert call["disable_confine"] is True
    assert call["profile_name"] == ""
    # Apparmor was probed (is_available) but provision_profile
    # was NOT called.
    assert apparmor.provision_calls == []


def test_ws_hook_spawns_unconfined_when_no_apparmor_manager(
    tmp_path, _patch_spawn,
) -> None:
    """WS server constructed without an apparmor manager (e.g.,
    daemon configured `apparmor_mode: false`) — spawn STILL fires
    unconfined."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    ws = _make_ws_server(str(ws_root), apparmor=None)
    hook = _register_and_get_hook(ws)

    hook(_server_stub(), "s-no-aa-mgr", str(sess_dir), client_id=None)

    spawn_calls = _patch_spawn["spawn_calls"]
    assert len(spawn_calls) == 1
    assert spawn_calls[0]["disable_confine"] is True
    assert spawn_calls[0]["profile_name"] == ""


def test_ws_hook_spawns_confined_when_apparmor_available(
    tmp_path, _patch_spawn,
) -> None:
    """Existing apparmor-available path: spawn with the
    provisioned profile + disable_confine=False."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    apparmor = _FakeAppArmor()
    apparmor.available = True
    apparmor.provision_outcome = True
    ws = _make_ws_server(str(ws_root), apparmor)
    hook = _register_and_get_hook(ws)

    hook(_server_stub(), "s-aa", str(sess_dir), client_id=None)

    spawn_calls = _patch_spawn["spawn_calls"]
    assert len(spawn_calls) == 1
    call = spawn_calls[0]
    assert call["disable_confine"] is False
    assert call["profile_name"] == "jaato-ws-s-aa"
    assert apparmor.provision_calls == [("s-aa", str(sess_dir))]


def test_ws_hook_spawns_unconfined_when_provisioning_fails(
    tmp_path, _patch_spawn,
) -> None:
    """Apparmor available but profile provisioning fails — spawn
    STILL fires unconfined.  §7a: dispatch surface comes first;
    confinement is best-effort."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    apparmor = _FakeAppArmor()
    apparmor.available = True
    apparmor.provision_outcome = False  # provision returns False
    ws = _make_ws_server(str(ws_root), apparmor)
    hook = _register_and_get_hook(ws)

    hook(_server_stub(), "s-prov-fail", str(sess_dir), client_id=None)

    spawn_calls = _patch_spawn["spawn_calls"]
    assert len(spawn_calls) == 1
    assert spawn_calls[0]["disable_confine"] is True
    assert spawn_calls[0]["profile_name"] == ""


# ----------------------------------------------------------------------
# Skip paths preserved
# ----------------------------------------------------------------------


def test_ws_hook_skips_no_workspace_path(tmp_path, _patch_spawn) -> None:
    """No workspace_path → no spawn (no cwd for runner)."""
    ws = _make_ws_server(str(tmp_path), _FakeAppArmor())
    hook = _register_and_get_hook(ws)

    hook(_server_stub(), "s-no-ws", None, client_id=None)
    assert _patch_spawn["spawn_calls"] == []


def test_ws_hook_skips_workspace_outside_ws_root(
    tmp_path, _patch_spawn,
) -> None:
    """Workspace not under WS server's root → IPC / user-CWD
    session; the IPC hook handles those.  WS hook must skip."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    other = tmp_path / "elsewhere"
    other.mkdir()

    ws = _make_ws_server(str(ws_root), _FakeAppArmor())
    hook = _register_and_get_hook(ws)

    hook(_server_stub(), "s-other", str(other), client_id=None)
    assert _patch_spawn["spawn_calls"] == []


def test_ws_hook_skips_when_no_daemon_loop(tmp_path, _patch_spawn) -> None:
    """Defensive: if start() didn't capture the daemon loop
    (shouldn't happen but might during early boot), skip the
    spawn rather than crashing."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    ws = _make_ws_server(str(ws_root), _FakeAppArmor(), daemon_loop=None)
    hook = _register_and_get_hook(ws)

    hook(_server_stub(), "s-no-loop", str(sess_dir), client_id=None)
    assert _patch_spawn["spawn_calls"] == []


# ----------------------------------------------------------------------
# Spawn-failure tolerance
# ----------------------------------------------------------------------


def test_ws_hook_spawn_failure_logs_and_returns(
    tmp_path, _patch_spawn,
) -> None:
    """Spawn failure: hook logs the warning and returns; doesn't
    crash session creation.  The post-init hook downgrades to
    soft mode based on the missing runner_rpc handle."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    ws = _make_ws_server(str(ws_root), _FakeAppArmor())
    hook = _register_and_get_hook(ws)

    _patch_spawn["spawn_outcome"]["raise"] = RuntimeError("spawn boom")

    # Should not raise.
    hook(_server_stub(), "s-spawn-fail", str(sess_dir), client_id=None)
    # Spawn was attempted.
    assert len(_patch_spawn["spawn_calls"]) == 1
