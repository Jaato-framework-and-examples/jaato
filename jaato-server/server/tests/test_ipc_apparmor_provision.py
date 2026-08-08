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
        requested_fragments: Optional[List[str]] = None,
        plugin_rules: Optional[List[str]] = None,
    ) -> bool:
        self.provision_calls.append({
            "session_id": session_id,
            "workspace_path": workspace_path,
            "config_root": config_root,
            "env_file": env_file,
            "requested_fragments": requested_fragments,
            "plugin_rules": plugin_rules,
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


# ----------------------------------------------------------------------
# PR-A (2026-05-14): precedence resolution across kwarg / client_config /
# profile.apparmor.  Three sources can express confinement intent; the
# helper must apply them in the order documented in
# ``project_backlog_apparmor_kwarg_for_headless_sessions``.
# ----------------------------------------------------------------------


def _profile_stub(apparmor: bool) -> Any:
    """Minimal profile stand-in carrying just ``apparmor``."""
    return type("_FakeProfile", (), {"apparmor": apparmor})()


def _server_with_profile(profile: Any) -> Any:
    """JaatoServer stand-in that exposes ``_profile``."""
    return type("_FakeJaatoServerWithProfile", (), {"_profile": profile})()


class TestAppArmorPrecedence:
    """Resolution order: kwarg override > client_config["apparmor"] >
    profile.apparmor > legacy False default.  PR-A surfaces all three
    sources but keeps the False default until PR-B flips it.
    """

    def test_kwarg_override_true_wins_over_falsy_sources(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Reactor / headless caller passes ``apparmor=True``; client
        config + profile both say False/absent; opt-in fires."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {"apparmor": False}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        result = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(_profile_stub(apparmor=False)),
            session_id="s-kwarg-true",
            workspace_path=str(tmp_path),
            client_id="c-1",
            apparmor_override=True,
        )
        # Spawn happens; apparmor provisioning is opted-in.
        assert result == "apparmor"
        assert len(_FakeAppArmorManager.instances) == 1

    def test_kwarg_override_false_wins_over_truthy_sources(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Caller passes ``apparmor=False``; client config + profile
        both say True; opt-in is suppressed.  This is the path TUI
        will use after PR-B flips the profile default to True — a
        defensive ``apparmor=False`` from the wire keeps interactive
        sessions unconfined regardless of profile choice."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {"apparmor": True}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        result = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(_profile_stub(apparmor=True)),
            session_id="s-kwarg-false",
            workspace_path=str(tmp_path),
            client_id="c-1",
            apparmor_override=False,
        )
        # Spawn happens unconfined; no AppArmor provisioning.
        assert result is None
        assert _FakeAppArmorManager.instances == []

    def test_client_config_signal_wins_over_profile_when_kwarg_unset(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """No kwarg override; client_config["apparmor"]=True opts in
        even when profile.apparmor=False (e.g. TUI user passes
        ``--apparmor`` against a profile that didn't declare it)."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {"apparmor": True}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        result = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(_profile_stub(apparmor=False)),
            session_id="s-cc-true",
            workspace_path=str(tmp_path),
            client_id="c-1",
            # apparmor_override absent
        )
        assert result == "apparmor"
        assert len(_FakeAppArmorManager.instances) == 1

    def test_profile_apparmor_consulted_when_no_kwarg_no_client_signal(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """No kwarg, no client_config opt-in; profile.apparmor=True
        drives confinement.  This is the path PR-B will rely on once
        the profile default flips to True — cascade-spawned headless
        sessions of confined profiles will automatically opt in."""
        sm = fake_session_manager
        # No "apparmor" key in client_config — None lookup.
        sm._client_config["c-1"] = {}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        result = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(_profile_stub(apparmor=True)),
            session_id="s-profile-true",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )
        assert result == "apparmor"
        assert len(_FakeAppArmorManager.instances) == 1

    def test_all_sources_falsy_leaves_unconfined(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Default state: no opt-in anywhere → no provisioning, spawn
        runs unconfined.  Preserves PR-A back-compat: existing callers
        that didn't set the field keep their current behaviour."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        result = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(_profile_stub(apparmor=False)),
            session_id="s-all-false",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )
        assert result is None
        assert _FakeAppArmorManager.instances == []

    def test_profile_apparmor_consulted_when_profile_attr_missing(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Older profile objects without an ``apparmor`` attribute (or
        a None ``_profile`` on the server) must default to False
        gracefully — no AttributeError, no opt-in by accident."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        # Server without _profile at all
        result = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_stub(),  # no _profile attribute
            session_id="s-no-profile",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )
        assert result is None
        assert _FakeAppArmorManager.instances == []


class TestConfigRootEnvelopeUnification:
    """2026-05-14 unification: ``config_root`` + ``env_file`` flow from
    the envelope (every entry point) into the policy generator, instead
    of being read from ``client_config[client_id]`` (IPC-only).  The
    pre-unification gap silently produced empty ``{config_root_rules}``
    on the reactor-spawned headless path, which crashed peer's v76
    cascade.  See
    ``project_backlog_apparmor_config_root_threading_for_headless``.
    """

    def test_envelope_config_root_override_reaches_policy_generator(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Headless path: ``client_config`` is empty (no
        ``ClientConfigRequest`` fires), but the envelope carries
        ``config_root`` from the caller (e.g. reactor /
        ``create_headless_session(config_root=...)``).  The policy
        generator must receive that value, not None."""
        sm = fake_session_manager
        sm._client_config["headless"] = {}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        result = sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(_profile_stub(apparmor=True)),
            session_id="s-headless-cr",
            workspace_path=str(tmp_path),
            client_id="headless",
            config_root_override="/srv/operator/.jaato",
            env_file_override="/srv/operator/.env",
        )
        assert result == "apparmor"
        assert len(_FakeAppArmorManager.instances) == 1
        provision_calls = _FakeAppArmorManager.instances[0].provision_calls
        assert provision_calls == [{
            "session_id": "s-headless-cr",
            "workspace_path": str(tmp_path),
            "config_root": "/srv/operator/.jaato",
            "env_file": "/srv/operator/.env",
            "requested_fragments": None,
            "plugin_rules": None,
        }]

    def test_envelope_override_wins_over_client_config(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Explicit envelope override beats whatever ``client_config``
        had.  The envelope is the source of truth post-unification;
        ``client_config`` is the fallback for legacy IPC paths."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {
            "config_root": "/legacy/.jaato",
            "env_file": "/legacy/.env",
        }
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(_profile_stub(apparmor=True)),
            session_id="s-override",
            workspace_path=str(tmp_path),
            client_id="c-1",
            config_root_override="/from-envelope/.jaato",
            env_file_override="/from-envelope/.env",
        )
        provision_calls = _FakeAppArmorManager.instances[0].provision_calls
        assert provision_calls[0]["config_root"] == "/from-envelope/.jaato"
        assert provision_calls[0]["env_file"] == "/from-envelope/.env"

    def test_client_config_fallback_when_no_envelope_override(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Existing IPC path (no envelope override passed) keeps
        reading ``client_config`` for back-compat.  Closes the
        regression risk that unification breaks the IPC flow."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {
            "apparmor": True,
            "config_root": "/ipc/.jaato",
            "env_file": "/ipc/.env",
        }
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(_profile_stub(apparmor=False)),
            session_id="s-ipc",
            workspace_path=str(tmp_path),
            client_id="c-1",
            # No overrides — pre-unification IPC behaviour preserved
        )
        provision_calls = _FakeAppArmorManager.instances[0].provision_calls
        assert provision_calls[0]["config_root"] == "/ipc/.jaato"
        assert provision_calls[0]["env_file"] == "/ipc/.env"

    def test_neither_source_falls_back_to_none(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """No envelope override, no ``client_config`` entry; policy
        generator receives ``None`` for both fields and renders
        ``{config_root_rules}`` empty.  This is the workspace-internal
        ``.jaato/`` layout — broad ``{workspace_path}/** rwkl`` grant
        covers reads, no config_root needed."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {"apparmor": True}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")

        sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(_profile_stub(apparmor=False)),
            session_id="s-no-cr",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )
        provision_calls = _FakeAppArmorManager.instances[0].provision_calls
        assert provision_calls[0]["config_root"] is None
        assert provision_calls[0]["env_file"] is None


def _profile_stub_with_fragments(
    apparmor: bool, apparmor_fragments: Any,
) -> Any:
    """Profile stand-in carrying both fields (Piece 1 wiring)."""
    return type("_FakeProfileWithFragments", (), {
        "apparmor": apparmor,
        "apparmor_fragments": apparmor_fragments,
    })()


class TestApparmorFragmentsEnvelopeThreading:
    """Piece 1 (2026-05-14): ``profile.apparmor_fragments`` reaches
    ``apparmor.provision_profile(requested_fragments=...)`` end-to-end
    through the spawn helper.
    """

    def test_profile_fragments_reach_provision_profile(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Profile declares ``apparmor_fragments=['host_validator']``;
        spawn helper threads that into the policy generator."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")
        profile = _profile_stub_with_fragments(
            apparmor=True, apparmor_fragments=["host_validator"],
        )

        sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(profile),
            session_id="s-fragments",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )
        provision_calls = _FakeAppArmorManager.instances[0].provision_calls
        assert provision_calls[0]["requested_fragments"] == ["host_validator"]

    def test_profile_empty_fragments_reach_provision_profile(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Profile declares explicit ``apparmor_fragments=[]`` (the
        maximally locked-down stage); empty list (not None) reaches
        the policy generator."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")
        profile = _profile_stub_with_fragments(
            apparmor=True, apparmor_fragments=[],
        )

        sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(profile),
            session_id="s-locked-down",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )
        provision_calls = _FakeAppArmorManager.instances[0].provision_calls
        # Empty list, not None — distinguishes from absent.
        assert provision_calls[0]["requested_fragments"] == []

    def test_profile_none_fragments_keeps_back_compat(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """Profile leaves ``apparmor_fragments`` absent (resolves to
        None on the dataclass); spawn helper passes None →
        provision_profile composes all fragments (back-compat)."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")
        profile = _profile_stub_with_fragments(
            apparmor=True, apparmor_fragments=None,
        )

        sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(profile),
            session_id="s-default",
            workspace_path=str(tmp_path),
            client_id="c-1",
        )
        provision_calls = _FakeAppArmorManager.instances[0].provision_calls
        assert provision_calls[0]["requested_fragments"] is None

    def test_kwarg_override_wins_over_profile_field(
        self, fake_session_manager, tmp_path,
    ) -> None:
        """``apparmor_fragments_override`` kwarg on the spawn helper
        wins over the profile's field — same precedence pattern as
        the other override kwargs (apparmor / config_root / env_file)."""
        sm = fake_session_manager
        sm._client_config["c-1"] = {}
        sm.set_apparmor_dependencies(ws_server=None, daemon_loop="<loop>")
        profile = _profile_stub_with_fragments(
            apparmor=True, apparmor_fragments=["host_validator"],
        )

        sm._provision_ipc_apparmor_and_spawn_runner(
            server=_server_with_profile(profile),
            session_id="s-kwarg-override",
            workspace_path=str(tmp_path),
            client_id="c-1",
            apparmor_fragments_override=["from-kwarg"],
        )
        provision_calls = _FakeAppArmorManager.instances[0].provision_calls
        # Override wins; profile's value is shadowed.
        assert provision_calls[0]["requested_fragments"] == ["from-kwarg"]
