"""Tests for ``SessionManager._provision_ipc_apparmor_and_spawn_runner``
(Phase 3 §3.13).

The method is the relocation target for the IPC apparmor pre-init
hook that lived in ``server/__main__.py`` through Phase 2 §2.3.
After §3.13, the logic is a normal SessionManager method called
inline from ``_bootstrap_session`` rather than a hook registered
via ``add_pre_initialize_hook``.

These tests pin its contract:

- No client_id → no-op (the disk-restore / ephemeral / WS paths
  follow §3.12's per-path migration; this method is IPC-only).
- Client without ``apparmor`` opt-in in client_config → no-op.
- No workspace_path with apparmor opt-in → return None +
  warning notification.
- Workspace under WS-server's root → no-op (the WS hook handles
  it; we don't double-provision).
- AppArmor unavailable on host → return ``"soft"`` + warning.
- Profile provisioning fails → return ``"soft"`` + warning.
- Runner spawn fails → return ``"soft"`` + warning.
- Successful provision + spawn → return ``"apparmor"`` + info.
- ``set_apparmor_dependencies`` propagates ws_server + daemon_loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest


# ----------------------------------------------------------------------
# Fixtures / fakes
# ----------------------------------------------------------------------


class _FakeAppArmorManager:
    """Stand-in for ``server.apparmor.AppArmorManager``."""

    instances: List["_FakeAppArmorManager"] = []

    def __init__(
        self,
        workspace_root: Any,
        loop: Any = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.loop = loop
        # Tests configure these to drive the provisioning paths.
        self.available = True
        self.provision_outcome = True
        self.provision_calls: List[Dict[str, Any]] = []
        _FakeAppArmorManager.instances.append(self)

    def is_available(self) -> bool:
        return self.available

    def provision_profile(
        self,
        session_id: str,
        workspace_path: str,
        config_root: Optional[str] = None,
        env_file: Optional[str] = None,
    ) -> bool:
        self.provision_calls.append({
            "session_id": session_id,
            "workspace_path": workspace_path,
            "config_root": config_root,
            "env_file": env_file,
        })
        return self.provision_outcome

    def get_profile_name(self, session_id: str) -> str:
        return f"jaato-ws-{session_id}"


class _FakeWSServer:
    """Stand-in for the WS server with a ``_workspace_root`` attribute
    used by the workspace-overlap precedence check."""

    def __init__(self, workspace_root: str) -> None:
        self._workspace_root = workspace_root


@pytest.fixture
def fake_session_manager(tmp_path):
    """Build a real SessionManager with patched apparmor manager +
    spawn helper, ready for the tests to call the relocated method
    directly."""
    from server.session_manager import SessionManager

    sm = SessionManager()
    # Track _emit_to_client invocations so tests can assert
    # warning / info notifications without a real IPC server.
    sm._emit_calls: List[Tuple[str, Any]] = []  # type: ignore[attr-defined]
    original_emit = sm._emit_to_client

    def _capture_emit(client_id: str, event: Any) -> None:
        sm._emit_calls.append((client_id, event))

    sm._emit_to_client = _capture_emit  # type: ignore[method-assign]
    return sm


@pytest.fixture(autouse=True)
def _patch_apparmor_and_spawn():
    """Replace AppArmorManager + spawn_session_runner with stubs so
    the tests don't actually exec or call apparmor_parser."""
    _FakeAppArmorManager.instances.clear()
    spawn_calls: List[Dict[str, Any]] = []
    spawn_outcome: Dict[str, Any] = {"raise": None}

    def _fake_spawn(**kwargs: Any) -> None:
        spawn_calls.append(kwargs)
        if spawn_outcome["raise"] is not None:
            raise spawn_outcome["raise"]

    with patch(
        "server.apparmor.AppArmorManager", _FakeAppArmorManager,
    ), patch(
        "server.runner_spawn.spawn_session_runner", _fake_spawn,
    ):
        # Expose the spawn-call recorder + outcome controller on the
        # patched module so tests can introspect / drive them.
        yield {
            "spawn_calls": spawn_calls,
            "spawn_outcome": spawn_outcome,
        }


def _server_stub() -> Any:
    """Minimal JaatoServer stand-in.  The relocated method calls
    ``server.set_runner_rpc(...)`` via the spawn helper (which is
    patched), so the real surface area we exercise is just an
    object identity."""
    return type("_FakeJaatoServer", (), {})()


# ----------------------------------------------------------------------
# Skips
# ----------------------------------------------------------------------


def test_returns_none_when_client_id_is_none(fake_session_manager) -> None:
    """Disk-restore / ephemeral / WS-standalone paths pass
    client_id=None; the method must no-op without provisioning."""
    sm = fake_session_manager
    result = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-1",
        workspace_path="/tmp/ws",
        client_id=None,
    )
    assert result is None
    assert _FakeAppArmorManager.instances == []


def test_returns_none_when_client_did_not_opt_in(fake_session_manager) -> None:
    """Default IPC clients don't opt into apparmor; the method
    must no-op."""
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": False}
    result = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-2",
        workspace_path="/tmp/ws",
        client_id="c-1",
    )
    assert result is None
    assert _FakeAppArmorManager.instances == []


def test_no_workspace_path_with_opt_in_returns_none_with_warning(
    fake_session_manager,
) -> None:
    """Apparmor opt-in but no workspace → can't provision; emit
    warning + return None."""
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    result = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-3",
        workspace_path=None,
        client_id="c-1",
    )
    assert result is None
    # A warning was emitted about no-workspace.
    styles = [getattr(ev, "style", None) for _, ev in sm._emit_calls]
    assert "warning" in styles


# ----------------------------------------------------------------------
# WS-overlap precedence
# ----------------------------------------------------------------------


def test_workspace_under_ws_root_skips_provisioning(
    fake_session_manager, tmp_path,
) -> None:
    """When the session's workspace is under the WS server's root,
    the WS hook handles confinement; this method must no-op."""
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sub = ws_root / "session_dir"
    sub.mkdir()
    sm.set_apparmor_dependencies(
        ws_server=_FakeWSServer(workspace_root=str(ws_root)),
        daemon_loop=None,
    )

    result = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-ws",
        workspace_path=str(sub),
        client_id="c-1",
    )
    assert result is None
    # No AppArmor manager constructed.
    assert _FakeAppArmorManager.instances == []


# ----------------------------------------------------------------------
# Apparmor unavailable / provisioning fails
# ----------------------------------------------------------------------


def test_apparmor_unavailable_returns_soft(
    fake_session_manager, tmp_path,
) -> None:
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop=None)

    workspace = str(tmp_path)

    # Hook into the manager construction so we can flip is_available.
    def _reduce_available():
        for inst in _FakeAppArmorManager.instances:
            inst.available = False

    # Pre-flip won't work; the manager is constructed lazily.  Patch
    # the class so ALL instances start unavailable.
    _FakeAppArmorManager.instances.clear()
    original_init = _FakeAppArmorManager.__init__

    def _unavailable_init(self, **kwargs: Any) -> None:
        original_init(self, **kwargs)
        self.available = False

    with patch.object(_FakeAppArmorManager, "__init__", _unavailable_init):
        result = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_stub(),
            session_id="s-noaa",
            workspace_path=workspace,
            client_id="c-1",
        )

    assert result == "soft"
    styles = [getattr(ev, "style", None) for _, ev in sm._emit_calls]
    assert "warning" in styles


def test_provisioning_failure_returns_soft(
    fake_session_manager, tmp_path,
) -> None:
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop=None)

    _FakeAppArmorManager.instances.clear()
    original_init = _FakeAppArmorManager.__init__

    def _failing_init(self, **kwargs: Any) -> None:
        original_init(self, **kwargs)
        self.provision_outcome = False

    with patch.object(_FakeAppArmorManager, "__init__", _failing_init):
        result = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_stub(),
            session_id="s-fail",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )

    assert result == "soft"


# ----------------------------------------------------------------------
# Successful provision + spawn
# ----------------------------------------------------------------------


def test_success_returns_apparmor_and_calls_spawn(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

    workspace = str(tmp_path)
    server = _server_stub()
    result = sm._provision_ipc_apparmor_and_spawn_runner(
        server=server,
        session_id="s-ok",
        workspace_path=workspace,
        client_id="c-1",
    )
    assert result == "apparmor"
    spawn_calls = _patch_apparmor_and_spawn["spawn_calls"]
    assert len(spawn_calls) == 1
    call = spawn_calls[0]
    assert call["server"] is server
    assert call["session_id"] == "s-ok"
    assert call["workspace_path"] == workspace
    assert call["profile_name"] == "jaato-ws-s-ok"
    assert call["daemon_loop"] == "<loop>"
    # An info-style notification was emitted.
    styles = [getattr(ev, "style", None) for _, ev in sm._emit_calls]
    assert "info" in styles


def test_spawn_failure_returns_soft(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

    _patch_apparmor_and_spawn["spawn_outcome"]["raise"] = RuntimeError(
        "spawn boom"
    )

    result = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-spawn-fail",
        workspace_path=str(tmp_path),
        client_id="c-1",
    )
    assert result == "soft"


# ----------------------------------------------------------------------
# Dependency wiring
# ----------------------------------------------------------------------


def test_set_apparmor_dependencies_stashes_refs() -> None:
    from server.session_manager import SessionManager

    sm = SessionManager()
    ws = _FakeWSServer("/tmp/ws")
    sm.set_apparmor_dependencies(ws_server=ws, daemon_loop="<loop>")

    assert sm._ws_server_ref is ws
    assert sm._daemon_loop == "<loop>"
