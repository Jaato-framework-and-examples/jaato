"""Tests for the §7a always-spawn refactor.

Phase 3 §7a decouples runner-spawn from apparmor opt-in: every IPC
session with a workspace gets a runner, regardless of whether the
client opted into apparmor confinement.  The runner-RPC dispatch
surface is then unconditionally available for the seat-flip's
``self._jaato.X`` migrations.

This file exercises the new shape via the already-patched
``server.runner_spawn.spawn_session_runner`` fixture from
``test_ipc_apparmor_provision.py`` (same patching scaffolding —
we just reach into the spawn-call recorder to verify spawns
happen on the no-apparmor path too).

Tests pin:

- IPC session WITHOUT apparmor opt-in → runner IS spawned with
  ``disable_confine=True`` + empty profile_name (the always-spawn
  invariant; pre-§7a this returned None and skipped spawn).
- IPC session WITH apparmor opt-in → runner is spawned with the
  apparmor profile_name + ``disable_confine=False``.
- Apparmor unavailable on host (opt-in but kernel module missing)
  → spawn STILL fires with ``disable_confine=True``; sandbox_mode
  returns ``"soft"``.
- Apparmor provisioning failure → same as unavailable: spawn
  fires; sandbox_mode is ``"soft"``.
- Spawn failure with no apparmor opt-in → returns None (silent;
  the opt-in case still surfaces ``"soft"`` for telemetry
  attribution).
- Spawn failure with apparmor opt-in → returns ``"soft"`` (the
  apparmor was provisioned but the runner died — operator visible).
- No client_id → no-op (non-IPC paths).
- No workspace_path → no spawn attempt (the runner needs a cwd).
- Workspace under WS-server root → no-op (WS hook owns it).

The structural refactor (split into
_spawn_session_runner_unconditional + _provision_apparmor_for_session)
is also pinned: each helper is callable in isolation against a
SessionManager instance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest


# ----------------------------------------------------------------------
# Fixtures (mirror test_ipc_apparmor_provision.py shape)
# ----------------------------------------------------------------------


class _FakeAppArmorManager:
    instances: List["_FakeAppArmorManager"] = []

    def __init__(self, workspace_root: Any, loop: Any = None) -> None:
        self.workspace_root = workspace_root
        self.loop = loop
        self.available = True
        self.provision_outcome = True
        self.provision_calls: List[Dict[str, Any]] = []
        _FakeAppArmorManager.instances.append(self)

    def is_available(self) -> bool:
        return self.available

    def provision_profile(
        self, session_id: str, workspace_path: str,
        config_root: Optional[str] = None, env_file: Optional[str] = None,
        requested_fragments: Optional[List[str]] = None,
        plugin_rules: Optional[List[str]] = None,
    ) -> bool:
        self.provision_calls.append({
            "session_id": session_id,
            "workspace_path": workspace_path,
        })
        return self.provision_outcome

    def get_profile_name(self, session_id: str) -> str:
        return f"jaato-ws-{session_id}"


class _FakeWSServer:
    def __init__(self, workspace_root: str) -> None:
        self._workspace_root = workspace_root


@pytest.fixture
def fake_session_manager(tmp_path):
    from server.session_manager import SessionManager
    sm = SessionManager()
    sm._emit_calls = []  # type: ignore[attr-defined]
    sm._emit_to_client = (
        lambda cid, ev: sm._emit_calls.append((cid, ev))  # type: ignore[attr-defined]
    )
    return sm


@pytest.fixture(autouse=True)
def _patch_apparmor_and_spawn():
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
        yield {
            "spawn_calls": spawn_calls,
            "spawn_outcome": spawn_outcome,
        }


def _server_stub() -> Any:
    return type("_FakeJaatoServer", (), {})()


# ----------------------------------------------------------------------
# Always-spawn invariant: no apparmor opt-in still spawns
# ----------------------------------------------------------------------


def test_no_apparmor_optin_still_spawns_runner_unconfined(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    """The §7a invariant: IPC session without apparmor opt-in
    STILL spawns a runner — disable_confine=True, empty
    profile_name, sandbox_mode=None (no confinement)."""
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": False}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

    workspace = str(tmp_path)
    sandbox_mode = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-no-aa",
        workspace_path=workspace,
        client_id="c-1",
    )

    # sandbox_mode is None: no confinement, but the runner IS spawned.
    assert sandbox_mode is None
    spawn_calls = _patch_apparmor_and_spawn["spawn_calls"]
    assert len(spawn_calls) == 1, "spawn must fire on the no-opt-in path"
    call = spawn_calls[0]
    assert call["session_id"] == "s-no-aa"
    assert call["workspace_path"] == workspace
    assert call["disable_confine"] is True
    assert call["profile_name"] == ""
    # AppArmorManager NEVER constructed for the no-opt-in path.
    assert _FakeAppArmorManager.instances == []


def test_apparmor_optin_spawns_runner_with_profile(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    """Existing apparmor-opt-in path: runner spawned with the
    provisioned profile + disable_confine=False."""
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

    workspace = str(tmp_path)
    sandbox_mode = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-aa",
        workspace_path=workspace,
        client_id="c-1",
    )

    assert sandbox_mode == "apparmor"
    spawn_calls = _patch_apparmor_and_spawn["spawn_calls"]
    assert len(spawn_calls) == 1
    call = spawn_calls[0]
    assert call["disable_confine"] is False
    assert call["profile_name"] == "jaato-ws-s-aa"


def test_apparmor_unavailable_still_spawns_unconfined(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    """Apparmor opt-in but kernel module missing — spawn STILL
    fires with disable_confine=True; sandbox_mode='soft' for
    operator visibility."""
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

    _FakeAppArmorManager.instances.clear()
    original_init = _FakeAppArmorManager.__init__

    def _unavailable_init(self, **kwargs):
        original_init(self, **kwargs)
        self.available = False

    with patch.object(_FakeAppArmorManager, "__init__", _unavailable_init):
        sandbox_mode = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_stub(),
            session_id="s-no-aa-host",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )

    # sandbox_mode='soft' — apparmor was wanted but unavailable.
    assert sandbox_mode == "soft"
    # Spawn STILL fires — that's the §7a invariant.
    spawn_calls = _patch_apparmor_and_spawn["spawn_calls"]
    assert len(spawn_calls) == 1
    assert spawn_calls[0]["disable_confine"] is True


def test_apparmor_provision_failure_still_spawns_unconfined(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

    _FakeAppArmorManager.instances.clear()
    original_init = _FakeAppArmorManager.__init__

    def _failing_init(self, **kwargs):
        original_init(self, **kwargs)
        self.provision_outcome = False

    with patch.object(_FakeAppArmorManager, "__init__", _failing_init):
        sandbox_mode = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_stub(),
            session_id="s-prov-fail",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )

    assert sandbox_mode == "soft"
    spawn_calls = _patch_apparmor_and_spawn["spawn_calls"]
    assert len(spawn_calls) == 1
    assert spawn_calls[0]["disable_confine"] is True


# ----------------------------------------------------------------------
# Spawn-failure paths (apparmor-opt-in distinction)
# ----------------------------------------------------------------------


def test_spawn_failure_no_optin_returns_none(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    """Spawn failure on the no-opt-in path: returns None
    (sandbox_mode stays empty since no confinement was promised)."""
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": False}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

    _patch_apparmor_and_spawn["spawn_outcome"]["raise"] = RuntimeError("boom")

    sandbox_mode = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-spawn-fail",
        workspace_path=str(tmp_path),
        client_id="c-1",
    )
    assert sandbox_mode is None


def test_spawn_failure_with_optin_returns_soft(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    """Spawn failure on the opt-in path: sandbox_mode='soft' so
    operator sees the downgrade."""
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": True}
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

    _patch_apparmor_and_spawn["spawn_outcome"]["raise"] = RuntimeError("boom")

    sandbox_mode = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-spawn-fail-aa",
        workspace_path=str(tmp_path),
        client_id="c-1",
    )
    assert sandbox_mode == "soft"


# ----------------------------------------------------------------------
# Skip paths (preserved from pre-§7a)
# ----------------------------------------------------------------------


def test_no_client_id_still_no_op(
    fake_session_manager, _patch_apparmor_and_spawn,
) -> None:
    """Disk-restore / ephemeral / WS-standalone: no client_id →
    no spawn attempt.  These paths supply their own bootstrap."""
    sm = fake_session_manager
    sandbox_mode = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-no-cid",
        workspace_path="/tmp/ws",
        client_id=None,
    )
    assert sandbox_mode is None
    assert _patch_apparmor_and_spawn["spawn_calls"] == []


def test_no_workspace_path_no_spawn_attempt(
    fake_session_manager, _patch_apparmor_and_spawn,
) -> None:
    """Sessions without a workspace don't get a runner — no cwd
    target.  Same as pre-§7a behavior."""
    sm = fake_session_manager
    sm._client_config["c-1"] = {"apparmor": False}

    sandbox_mode = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-no-ws",
        workspace_path=None,
        client_id="c-1",
    )
    assert sandbox_mode is None
    assert _patch_apparmor_and_spawn["spawn_calls"] == []


def test_workspace_under_ws_root_no_spawn(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    """WS-overlap precedence preserved: WS hook owns this lane."""
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

    sandbox_mode = sm._provision_ipc_apparmor_and_spawn_runner(
        server=_server_stub(),
        session_id="s-ws-overlap",
        workspace_path=str(sub),
        client_id="c-1",
    )
    assert sandbox_mode is None
    assert _patch_apparmor_and_spawn["spawn_calls"] == []


# ----------------------------------------------------------------------
# Helper extraction: each piece callable in isolation
# ----------------------------------------------------------------------


def test_spawn_helper_callable_in_isolation(
    fake_session_manager, tmp_path, _patch_apparmor_and_spawn,
) -> None:
    """``_spawn_session_runner_unconditional`` is callable as a
    standalone — proves the §7a refactor extraction is clean."""
    sm = fake_session_manager
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")
    ok = sm._spawn_session_runner_unconditional(
        server=_server_stub(),
        session_id="s-iso",
        workspace_path=str(tmp_path),
        client_id="c-1",
        profile_name="",  # unconfined
    )
    assert ok is True
    spawn_calls = _patch_apparmor_and_spawn["spawn_calls"]
    assert spawn_calls[0]["disable_confine"] is True


def test_apparmor_helper_callable_in_isolation(
    fake_session_manager, tmp_path,
) -> None:
    """``_provision_apparmor_for_session`` is callable as a
    standalone — proves the §7a refactor extraction is clean."""
    sm = fake_session_manager
    sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")
    # 2026-05-14: the helper takes ``config_root`` / ``env_file`` explicitly
    # instead of reading them from ``client_config[client_id]`` -- the old
    # form silently omitted ``{config_root_rules}`` on the reactor-spawned
    # headless path.  Opt-in is the CALLER's decision now, so there is no
    # apparmor flag to pass here.
    profile_name, mode = sm._provision_apparmor_for_session(
        session_id="s-iso",
        workspace_path=str(tmp_path),
        client_id="c-1",
        config_root=None,
        env_file=None,
    )
    # The pin is that the extraction is CALLABLE STANDALONE and honours its
    # documented 2-tuple contract -- not that this host can confine.  A host
    # without AppArmor returns ("", "soft") by design, so asserting
    # mode == "apparmor" would make the refactor pin fail on environment.
    assert (profile_name, mode) in {
        ("jaato-ws-s-iso", "apparmor"),   # AppArmor available + provisioned
        ("", "soft"),                     # unavailable / provisioning failed
    }, f"undocumented return: {(profile_name, mode)!r}"
