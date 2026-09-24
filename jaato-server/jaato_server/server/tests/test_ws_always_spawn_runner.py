"""Tests for the WS-server apparmor-pre-init-hook always-spawn
behavior (Phase 3 §7a WS counterpart).

Pre-§7a: the WS hook gated runner-spawn behind apparmor
availability — when the kernel module was missing, no runner
spawned and tool execution stayed in-process.

Post-§7a: the hook spawns the runner unconditionally for any
WS-provisioned session (workspace under WS-server's root) when the
host has no AppArmor — an unconfined session runs in-process on spawn
failure, exactly as before.

#1253 refines the confined case: when the daemon holds an available
``AppArmorManager`` confinement is REQUIRED, and a provisioning failure
or a spawn failure REFUSES the session (recorded for
``initialize_or_refuse``) rather than serving model-driven work with no
kernel boundary.  Confinement is no longer "layered atop, best-effort";
it is confined or it does not run.

This file exercises the WS hook's behavior under three apparmor
states (unavailable / available+success / available+provisioning-failure)
and the gate paths (workspace not under WS root, no daemon loop).

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

from jaato_server.shared.apparmor_label import SANDBOX_MODE_SOFT
from jaato_server.shared.tests.reversion import Reversion


# ----------------------------------------------------------------------
# Reversions — read by
# ``shared/tests/test_every_guard_detects_its_own_reversion.py``.  The two
# #1253 Layer-1 (daemon WS hook) reversions live HERE, beside the tests they
# name, because the meta-guard resolves a reversion's ``test`` WITHIN the
# module that DECLARES it (#1065).  The runner-side gate and the envelope
# round-trip are guarded from ``test_confinement_before_serve_1253.py``,
# whose tests live there.
# ----------------------------------------------------------------------
_WS = "jaato-server/jaato_server/server/websocket.py"

REVERSIONS = [
    # Layer 1 — the daemon refuses a provisioning failure.  Falling through
    # (``return`` -> ``pass``) spawns an unconfined runner for a session that
    # required confinement: exactly the #1253 bypass.
    Reversion(
        target=_WS,
        find=(
            '                        "unconfined runner (#1253)",\n'
            "                    )\n"
            "                    return"
        ),
        replace=(
            '                        "unconfined runner (#1253)",\n'
            "                    )\n"
            "                    pass  # #1253 reversion: fall through to spawn"
        ),
        test="test_ws_hook_refuses_when_provisioning_fails",
        because=(
            "the WS hook again spawns an unconfined runner when profile "
            "provisioning fails instead of refusing the session"
        ),
    ),
    # Layer 1 — a confined session's spawn failure must refuse, not fall back
    # to unconfined in-process execution.  Neutering the gate in
    # ``_report_confined_spawn_failure`` restores the in-process fallback for
    # a session that required confinement.
    Reversion(
        target=_WS,
        find=(
            "    if confinement_required:\n"
            "        logger.warning(\n"
            '            "AppArmor pre-init: runner spawn failed for session %s "'
        ),
        replace=(
            "    if False:  # #1253 reversion\n"
            "        logger.warning(\n"
            '            "AppArmor pre-init: runner spawn failed for session %s "'
        ),
        test="test_ws_hook_spawn_failure_refuses_a_confined_session",
        because=(
            "a confined session whose runner spawn fails again falls back to "
            "unconfined in-process tool execution instead of being refused"
        ),
    ),
    # Point 3 — the POST-init resilience hook must not SILENTLY downgrade a
    # required-but-failed boundary to soft.  It runs after the runner already
    # spawned (so it cannot un-spawn — the pre-init hook fails that closed),
    # but a silent ``sandbox_mode=soft`` is exactly the invisible boundary loss
    # #1253 is about.  Removing the WARNING restores the silent downgrade.
    Reversion(
        target=_WS,
        find=(
            "                logger.warning(\n"
            "                    \"AppArmor confinement required for session %s (WS-\"\n"
            "                    \"provisioned, host supports it) but profile provisioning \"\n"
            "                    \"failed in the post-init hook — the runner is NOT kernel-\"\n"
            "                    \"confined; recording sandbox_mode=soft rather than \"\n"
            "                    \"silently claiming enforcement (#1253/#1014)\",\n"
            "                    session_id,\n"
            "                )"
        ),
        replace=(
            "                pass  # #1253 reversion: silent soft downgrade restored"
        ),
        test="test_ws_session_hook_announces_a_required_boundary_it_could_not_apply",
        because=(
            "the post-init hook again downgrades a WS-provisioned confined "
            "session to soft mode with no WARNING when provisioning fails"
        ),
    ),
]


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
        self._confinement_ids: Dict[str, str] = {}

    def is_available(self) -> bool:
        return self.available

    def provision_profile(
        self, session_id: str, workspace_path: str,
        **kwargs: Any,
    ) -> bool:
        self.provision_calls.append((session_id, workspace_path))
        cid = kwargs.get("confinement_id")
        if cid:
            self._confinement_ids[session_id] = cid
        return self.provision_outcome

    def get_profile_name(self, session_id: str) -> str:
        return f"jaato-ws-{self._confinement_ids.get(session_id, session_id)}"

    def confinement_id_for_boundary(
        self, workspace_path: str, **kwargs: Any,
    ) -> str:
        """#1033: the profile is named after the boundary, not the
        session, so the WS hook derives an id before provisioning."""
        from jaato_server.server.confinement_id import confinement_id
        return confinement_id(
            workspace_root=workspace_path,
            config_root=None,
            rendered_body=repr(sorted(kwargs.items(), key=str)),
        )


class _FakeSession:
    """The two attributes the post-init ``_apparmor_session_hook`` reads on
    the provision-failure path: a workspace under the WS root, and a
    ``sandbox_mode`` it writes the (truthful) ``soft`` verdict onto."""

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = workspace_path
        self.sandbox_mode: Optional[str] = None


class _FakeSM:
    def __init__(self) -> None:
        self.pre_init_hooks: List[Any] = []
        self.session_hooks: List[Any] = []
        self._sessions: Dict[str, _FakeSession] = {}

    def add_pre_initialize_hook(self, hook: Any) -> None:
        self.pre_init_hooks.append(hook)

    def add_session_hook(self, hook: Any) -> None:
        self.session_hooks.append(hook)

    def get_session(self, session_id: str) -> Optional[_FakeSession]:
        return self._sessions.get(session_id)


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
    from jaato_server.server.websocket import JaatoWSServer
    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._apparmor = apparmor
    ws._cgroups = None
    ws._workspace_root = workspace_root
    ws._event_loop = daemon_loop
    return ws


def _register_and_get_hook(ws: Any) -> Any:
    """Drive ``set_command_router`` to register the hook + return
    the registered closure for direct invocation."""
    from jaato_server.server.websocket import JaatoWSServer
    router = _FakeRouter()
    JaatoWSServer.set_command_router(ws, router)
    assert router._session_manager.pre_init_hooks, "hook not registered"
    return router._session_manager.pre_init_hooks[0]


def _register_and_get_session_hook(ws: Any) -> "Tuple[Any, _FakeSM]":
    """Register the hooks + return the POST-init ``_apparmor_session_hook``
    (the resilience re-run) and the fake SM, so the caller can install the
    fake session the hook resolves via ``sm.get_session``."""
    from jaato_server.server.websocket import JaatoWSServer
    router = _FakeRouter()
    JaatoWSServer.set_command_router(ws, router)
    sm = router._session_manager
    assert sm.session_hooks, "session hook not registered"
    return sm.session_hooks[0], sm


@pytest.fixture(autouse=True)
def _patch_spawn():
    """Replace ``spawn_session_runner`` + ``dispatch_bootstrap_envelope``
    with stub recorders so tests focus on the WS hook's spawn-side
    behavior without forking real runners or sending real RPC.

    Phase 3 §7c step 2 added the ``dispatch_bootstrap_envelope``
    call after spawn; the fixture stubs both helpers so tests
    written against the spawn-only contract stay green and new
    tests can assert the bootstrap dispatch fires too."""
    spawn_calls: List[Dict[str, Any]] = []
    spawn_outcome: Dict[str, Any] = {"raise": None}
    bootstrap_calls: List[Dict[str, Any]] = []

    def _fake_spawn(**kwargs: Any) -> None:
        spawn_calls.append(kwargs)
        if spawn_outcome["raise"] is not None:
            raise spawn_outcome["raise"]

    def _fake_bootstrap(**kwargs: Any) -> None:
        bootstrap_calls.append(kwargs)

    with patch("jaato_server.server.runner_spawn.spawn_session_runner", _fake_spawn), \
         patch("jaato_server.server.runner_spawn.dispatch_bootstrap_envelope", _fake_bootstrap):
        yield {
            "spawn_calls": spawn_calls,
            "spawn_outcome": spawn_outcome,
            "bootstrap_calls": bootstrap_calls,
        }


def _server_stub() -> Any:
    """A bare JaatoServer stand-in that RECORDS bootstrap-outcome notes.

    #1253: the WS pre-init hook fails a confinement-required session closed
    by calling ``server.note_runner_bootstrap_outcome(<reason>)`` and
    returning without spawning; ``session_manager.initialize_or_refuse``
    reads that back to refuse the session.  The stub records those calls
    into ``bootstrap_outcomes`` so a test can assert the refusal (a real
    ``JaatoServer`` stores it on ``_runner_bootstrap_error``)."""
    srv = type("_FakeJaatoServer", (), {})()
    srv.bootstrap_outcomes = []  # type: ignore[attr-defined]
    srv.note_runner_bootstrap_outcome = (  # type: ignore[attr-defined]
        lambda reason: srv.bootstrap_outcomes.append(reason)
    )
    return srv


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
    # #1033: boundary-derived, so the session id is NOT in the name.
    assert call["profile_name"].startswith("jaato-ws-")
    assert "s-aa" not in call["profile_name"]
    assert apparmor.provision_calls == [("s-aa", str(sess_dir))]


def test_ws_hook_refuses_when_provisioning_fails(
    tmp_path, _patch_spawn,
) -> None:
    """#1253: apparmor available (so confinement is REQUIRED) but profile
    provisioning fails — the hook REFUSES the session rather than spawning
    an unconfined runner.

    Pre-#1253 this spawned unconfined (``disable_confine=True``, empty
    ``profile_name``), leaving a session whose record claims
    ``sandbox_mode: apparmor`` serving model-driven work with no kernel
    boundary — the silent bypass #1100 deferred and #1253 measured live.
    The fail-closed contract: no runner is spawned, no bootstrap is
    dispatched, and ``note_runner_bootstrap_outcome`` records the reason so
    ``initialize_or_refuse`` refuses the session by name."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    apparmor = _FakeAppArmor()
    apparmor.available = True
    apparmor.provision_outcome = False  # provision returns False
    ws = _make_ws_server(str(ws_root), apparmor)
    hook = _register_and_get_hook(ws)

    server = _server_stub()
    hook(server, "s-prov-fail", str(sess_dir), client_id=None)

    # FAIL CLOSED: no unconfined runner, no bootstrap.
    assert _patch_spawn["spawn_calls"] == []
    assert _patch_spawn["bootstrap_calls"] == []
    # The refusal was recorded so the session is refused (not run unconfined).
    assert server.bootstrap_outcomes, (
        "provisioning failure must record a bootstrap outcome so "
        "initialize_or_refuse refuses the session"
    )
    assert "1253" in server.bootstrap_outcomes[-1]


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


def test_ws_hook_spawn_failure_refuses_a_confined_session(
    tmp_path, _patch_spawn,
) -> None:
    """Spawn failure on a confinement-required session: hook logs, records
    the refusal and returns without crashing session creation.

    #1253: an in-process fallback would run the model's tools in the daemon
    process with NO AppArmor boundary, which is the same silent bypass the
    provisioning-failure path refuses — so a confined session's spawn
    failure is recorded (``note_runner_bootstrap_outcome``) and the session
    is refused rather than falling back to unconfined in-process execution.
    ``_FakeAppArmor`` defaults available, so confinement is required here."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    ws = _make_ws_server(str(ws_root), _FakeAppArmor())
    hook = _register_and_get_hook(ws)

    _patch_spawn["spawn_outcome"]["raise"] = RuntimeError("spawn boom")

    server = _server_stub()
    # Should not raise.
    hook(server, "s-spawn-fail", str(sess_dir), client_id=None)
    # Spawn was attempted.
    assert len(_patch_spawn["spawn_calls"]) == 1
    # Bootstrap was NOT attempted — the hook returns after the
    # spawn failure, so dispatch_bootstrap_envelope never fires.
    assert len(_patch_spawn["bootstrap_calls"]) == 0
    # #1253: the confined session's spawn failure is recorded so
    # initialize_or_refuse refuses it rather than running it unconfined.
    assert server.bootstrap_outcomes
    assert "1253" in server.bootstrap_outcomes[-1]


# ----------------------------------------------------------------------
# §7c step 2: bootstrap dispatch fires after spawn (WS path)
# ----------------------------------------------------------------------


def test_ws_hook_dispatches_bootstrap_after_spawn(
    tmp_path, _patch_spawn,
) -> None:
    """§7c step 2 invariant: the WS hook calls
    ``dispatch_bootstrap_envelope`` after a successful spawn so the
    runner-side JaatoSession host gets populated.  Pre-§7c-step-2
    this dispatch was IPC-only — every WS session left the
    runner-side host NULL."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    ws = _make_ws_server(str(ws_root), _FakeAppArmor())
    hook = _register_and_get_hook(ws)

    hook(_server_stub(), "s-bootstrap", str(sess_dir), client_id=None)

    # Spawn fired exactly once + bootstrap fired exactly once.
    assert len(_patch_spawn["spawn_calls"]) == 1
    assert len(_patch_spawn["bootstrap_calls"]) == 1

    bootstrap = _patch_spawn["bootstrap_calls"][0]
    assert bootstrap["session_id"] == "s-bootstrap"
    assert bootstrap["workspace_path"] == str(sess_dir)
    # profile_name passed through from apparmor provisioning.
    assert bootstrap["profile_name"].startswith("jaato-ws-")
    assert "s-bootstrap" not in bootstrap["profile_name"]
    # #1253: a confined session carries the invariant to the runner gate.
    assert bootstrap["confinement_required"] is True


def test_ws_hook_dispatches_bootstrap_on_unconfined_path(
    tmp_path, _patch_spawn,
) -> None:
    """The unconfined path (apparmor unavailable / disabled) also
    bootstraps — the seat-flip needs the runner-side host populated
    regardless of confinement state."""
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    apparmor = _FakeAppArmor()
    apparmor.available = False
    ws = _make_ws_server(str(ws_root), apparmor)
    hook = _register_and_get_hook(ws)

    hook(_server_stub(), "s-no-aa-boot", str(sess_dir), client_id=None)

    assert len(_patch_spawn["spawn_calls"]) == 1
    assert len(_patch_spawn["bootstrap_calls"]) == 1
    # profile_name is empty on the unconfined path.
    assert _patch_spawn["bootstrap_calls"][0]["profile_name"] == ""
    # #1253: a genuinely-unconfined session (no AppArmor manager available)
    # carries confinement_required=False, so the runner gate stays inert.
    assert _patch_spawn["bootstrap_calls"][0]["confinement_required"] is False


# ----------------------------------------------------------------------
# #1253 point 3 — the post-init resilience hook announces a required boundary
# it could not apply, rather than silently downgrading to soft.
# ----------------------------------------------------------------------


def test_ws_session_hook_announces_a_required_boundary_it_could_not_apply(
    tmp_path, monkeypatch, caplog,
) -> None:
    """A WS-provisioned session on an AppArmor-available host whose profile
    FAILS to provision in the post-init hook must WARN (never a silent
    downgrade of a boundary — the #1014 posture) while recording the truthful
    ``soft`` mode.

    Reaching the provision-failure branch means confinement was REQUIRED: the
    host has an available ``AppArmorManager`` and the workspace is under the WS
    root.  The pre-init hook (#1260) fails that closed BEFORE spawn on the core
    path; this resilience re-run runs after the runner already spawned, so it
    cannot un-spawn — but it must not lie by omission.
    """
    ws_root = tmp_path / "ws_root"
    ws_root.mkdir()
    sess_dir = ws_root / "session_dir"
    sess_dir.mkdir()

    apparmor = _FakeAppArmor()
    apparmor.available = True
    apparmor.provision_outcome = False  # provisioning FAILS in the hook
    ws = _make_ws_server(str(ws_root), apparmor)

    # The hook imports resolve_plugin_apparmor_rules inside the function.
    monkeypatch.setattr(
        "jaato_server.server.apparmor.resolve_plugin_apparmor_rules",
        lambda **kwargs: None,
    )

    session_hook, sm = _register_and_get_session_hook(ws)
    sess = _FakeSession(str(sess_dir))
    sm._sessions["s-postinit-fail"] = sess

    server = _server_stub()  # no _profile -> no cgroup limits, skips that block

    with caplog.at_level("WARNING"):
        session_hook(server, "s-postinit-fail")

    # Truthful record: soft, not a false apparmor claim.
    assert sess.sandbox_mode == SANDBOX_MODE_SOFT
    # And it is AUDIBLE — a WARNING naming the issue, not a silent downgrade.
    warnings = [
        r.getMessage() for r in caplog.records if r.levelname == "WARNING"
    ]
    assert any("1253" in m for m in warnings), (
        "a required-but-failed boundary must be announced at WARNING, not "
        f"silently downgraded; got warnings: {warnings}"
    )
