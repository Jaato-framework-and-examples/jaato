"""Confinement is required on the SessionManager spawn path too (#1253).

PR #1260 armed ONE of the two spawn paths — the WS pre-init hook
(``websocket._apparmor_pre_init_hook``): it provisions the AppArmor profile
BEFORE spawn, stamps ``SessionInitEnvelope.confinement_required``, and refuses
(records a bootstrap outcome, does not spawn) when provisioning fails.  The
SECOND path — ``SessionManager._spawn_session_runner_unconditional``, driven by
``_provision_ipc_apparmor_and_spawn_runner``, which a WS ``session.new`` on a
live premium daemon actually took — passed **no** ``confinement_required`` and,
when ``profile_name == ""``, spawned an unconfined runner "by design" (the §7a
"always have a runner; confinement is layered" comment).  That §7a intent IS
the #1253 silent bypass: a runner spawns ``confined=False`` while the session
record still claims ``sandbox_mode: apparmor``, and the profile is provisioned
seconds later by the post-init resilience hook.

The invariant this pins, on the second path: **a session that opted into
AppArmor on a host that supports it must never serve model-driven work with an
empty profile** — it is confined, or it does not run.  Three points, fail
CLOSED:

1. ``_provision_ipc_apparmor_and_spawn_runner`` computes
   ``confinement_required = opt_in_apparmor AND host-has-AppArmor`` and threads
   it to ``_spawn_session_runner_unconditional`` → ``dispatch_bootstrap_envelope``,
   so #1260's runner-side raise arms on this path (defence in depth).
2. When ``confinement_required`` and ``profile_name == ""`` at the spawn step,
   the method REFUSES — records a bootstrap outcome the free function
   ``initialize_or_refuse`` reads (``RunnerBootstrapFailed``) — instead of
   spawning unconfined.

The discriminator is keyed on the session's OPT-IN, not host availability
alone: an IPC session that did not opt in (``IPCClient(apparmor=False)``, the
default) has ``confinement_required=False`` and spawns exactly as before.  And
a session that opted in on a host with NO AppArmor is genuinely unconfined
(nothing to enforce) — also ``confinement_required=False`` — so it, too, is
unchanged.

**No kernel here.** This container carries no AppArmor LSM (as #1023 / #1033 /
#1100 / #1260 all record).  These tests exercise the FRAMEWORK's decision — that
it refuses to serve an apparmor-opted session with no profile — from fabricated
manager / server state, never the kernel's behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.apparmor_label import SANDBOX_MODE_SOFT
from jaato_server.shared.tests.reversion import Reversion

_SM = "jaato-server/jaato_server/server/session_manager.py"


# ----------------------------------------------------------------------
# Reversions — read by
# ``shared/tests/test_every_guard_detects_its_own_reversion.py``.  Every
# ``test`` names a test declared IN THIS MODULE (#1065): the meta-guard
# resolves it as ``<this module>::<test>``, so a reversion here may only name
# a test that lives here.  Anchors are tight (the gate / the two threading
# sites), so an insertion nearby does not make one go stale.
# ----------------------------------------------------------------------
REVERSIONS = [
    # Point 2 — the fail-closed refusal.  Disabling the gate restores the
    # pre-#1253 behaviour on this path: an apparmor-opted session whose profile
    # did not provision falls through to the unconditional spawn and serves
    # work unconfined.
    Reversion(
        target=_SM,
        find="        if confinement_required and not profile_name:",
        replace="        if False:  # #1253 reversion: refusal gate disabled",
        test="test_refuses_when_confinement_required_and_no_profile",
        because=(
            "the SessionManager spawn path again spawns an unconfined runner "
            "for an apparmor-opted session when provisioning produced no "
            "profile, instead of refusing the session"
        ),
    ),
    # Point 1 — the invariant reaches the spawn helper.  Reading a fixed False
    # here means the runner-side gate can never see that confinement was
    # required on this path.
    Reversion(
        target=_SM,
        find=(
            "            # slot, a peer daemon), ``_maybe_self_confine`` RAISES rather than\n"
            "            # running the session unconfined (#1260).\n"
            "            confinement_required=confinement_required,"
        ),
        replace=(
            "            # slot, a peer daemon), ``_maybe_self_confine`` RAISES rather than\n"
            "            # running the session unconfined (#1260).\n"
            "            confinement_required=False,  # #1253 reversion"
        ),
        test="test_success_threads_confinement_required_to_spawn",
        because=(
            "confinement_required stops reaching _spawn_session_runner_"
            "unconditional, so the runner gate is never told the session was "
            "confined on this path"
        ),
    ),
    # Point 1 — the invariant reaches the runner over the envelope.  Reading a
    # fixed False on the bootstrap dispatch drops it before the runner gate.
    Reversion(
        target=_SM,
        find=(
            "                    # #1253: carry the confinement invariant to the runner\n"
            "                    # gate (defence in depth — see this method's docstring).\n"
            "                    confinement_required=confinement_required,"
        ),
        replace=(
            "                    # #1253: carry the confinement invariant to the runner\n"
            "                    # gate (defence in depth — see this method's docstring).\n"
            "                    confinement_required=False,  # #1253 reversion"
        ),
        test="test_spawn_threads_confinement_required_to_bootstrap",
        because=(
            "confinement_required no longer rides the bootstrap envelope, so "
            "the runner-side _maybe_self_confine gate can never see it"
        ),
    ),
]


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _FakeAppArmor:
    """Stand-in exposing only ``is_available`` — the one method
    ``SessionManager._apparmor_available`` reads."""

    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeProfile:
    def __init__(self, apparmor: bool) -> None:
        self.apparmor = apparmor
        self.apparmor_fragments = None
        self.runtime_limits = None
        self.budget_control = None


class _FakeServer:
    def __init__(self, *, apparmor: bool) -> None:
        self._profile = _FakeProfile(apparmor=apparmor)
        self._spawn_isolated_runner_handler = None
        self.client_tool_schemas: Dict[str, Any] = {}
        self.runner_rpc = object()
        self.bootstrap_outcomes: List[str] = []
        # scratch attributes the spawn path assigns
        self._cascade_budget_pool = None
        self._draws_on_parent_budget = True

    def note_runner_bootstrap_outcome(self, reason: str) -> None:
        self.bootstrap_outcomes.append(reason)


def _make_sm(
    *,
    apparmor_manager: Optional[_FakeAppArmor],
    provision_result: Tuple[str, Optional[str]],
) -> Tuple[SessionManager, List[Dict[str, Any]]]:
    """A bare SessionManager with the heavy dependencies of
    ``_provision_ipc_apparmor_and_spawn_runner`` stubbed.

    ``_apparmor_available`` and ``_record_bootstrap_refusal`` are the REAL
    methods (they are what the fix installs); everything else the provision
    path touches is stubbed to a recorder or a no-op.  Returns the manager and
    the list into which the stubbed ``_spawn_session_runner_unconditional``
    records its kwargs — empty means it was never called (the refusal path).
    """
    sm = SessionManager.__new__(SessionManager)
    sm._client_config = {}  # type: ignore[attr-defined]
    sm._pending_client_tools = {}  # type: ignore[attr-defined]
    sm._apparmor_manager = apparmor_manager  # type: ignore[attr-defined]

    spawn_calls: List[Dict[str, Any]] = []

    def _fake_spawn(**kwargs: Any) -> bool:
        spawn_calls.append(kwargs)
        return True

    sm._spawn_session_runner_unconditional = _fake_spawn  # type: ignore[assignment]
    sm._provision_apparmor_for_session = (  # type: ignore[assignment]
        lambda **kwargs: provision_result
    )
    sm._workspace_under_ws_root = lambda _p: False  # type: ignore[assignment]
    sm._notify_apparmor = lambda *a, **k: None  # type: ignore[assignment]
    return sm, spawn_calls


@pytest.fixture()
def _no_plugin_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_provision_ipc_apparmor_and_spawn_runner`` imports
    ``resolve_plugin_apparmor_rules`` inside the function; stub it so the
    provision path needs no real registry."""
    monkeypatch.setattr(
        "jaato_server.server.apparmor.resolve_plugin_apparmor_rules",
        lambda **kwargs: None,
    )


def _drive_provision(
    sm: SessionManager, server: _FakeServer, *, opt_in: bool,
) -> Optional[str]:
    """Run ``_provision_ipc_apparmor_and_spawn_runner`` for one client."""
    client_id = "client-1"
    if opt_in:
        sm._client_config[client_id] = {"apparmor": True}  # type: ignore[attr-defined]
    return sm._provision_ipc_apparmor_and_spawn_runner(
        server, "sess-1", "/ws/one", client_id,
    )


# ----------------------------------------------------------------------
# Point 2 — the fail-closed refusal (the load-bearing case)
# ----------------------------------------------------------------------


def test_refuses_when_confinement_required_and_no_profile(
    _no_plugin_rules: None,
) -> None:
    """Opted in + host supports AppArmor + provisioning produced NO profile
    → REFUSE (record a bootstrap outcome, do NOT spawn).

    This is the #1253 silent bypass on the SessionManager path.  The daemon
    turns the recorded outcome into a ``RunnerBootstrapFailed`` refusal via
    ``initialize_or_refuse``; the session serves no model-driven work rather
    than serving it unconfined.
    """
    sm, spawn_calls = _make_sm(
        apparmor_manager=_FakeAppArmor(available=True),
        provision_result=("", SANDBOX_MODE_SOFT),
    )
    server = _FakeServer(apparmor=True)

    result = _drive_provision(sm, server, opt_in=True)

    # FAIL CLOSED: no spawn, and the refusal was recorded for
    # initialize_or_refuse.
    assert spawn_calls == [], "an unconfined runner must not be spawned"
    assert server.bootstrap_outcomes, (
        "confinement-required provisioning failure must record a bootstrap "
        "outcome so initialize_or_refuse refuses the session"
    )
    assert "1253" in server.bootstrap_outcomes[-1]
    assert result == SANDBOX_MODE_SOFT


# ----------------------------------------------------------------------
# Genuinely-unconfined paths — must be unchanged (spawn, no refusal)
# ----------------------------------------------------------------------


def test_spawns_unconfined_when_host_lacks_apparmor(
    _no_plugin_rules: None,
) -> None:
    """Opted in but the host has no AppArmor → genuinely unconfined: spawn as
    before with ``confinement_required=False`` and NO refusal."""
    sm, spawn_calls = _make_sm(
        apparmor_manager=_FakeAppArmor(available=False),
        provision_result=("", SANDBOX_MODE_SOFT),
    )
    server = _FakeServer(apparmor=True)

    _drive_provision(sm, server, opt_in=True)

    assert len(spawn_calls) == 1, "a genuinely-unconfined session still spawns"
    assert spawn_calls[0]["confinement_required"] is False
    assert spawn_calls[0]["profile_name"] == ""
    assert server.bootstrap_outcomes == [], "no refusal for an unconfined host"


def test_spawns_as_before_without_opt_in(_no_plugin_rules: None) -> None:
    """A session that did NOT opt into AppArmor (the IPC default) spawns
    exactly as today: ``confinement_required=False``, no provisioning, no
    refusal.  ``_provision_apparmor_for_session`` is not even called."""
    sm, spawn_calls = _make_sm(
        apparmor_manager=_FakeAppArmor(available=True),
        provision_result=("should-not-be-used", "apparmor"),
    )
    # A real method that fails loudly if the no-opt-in path calls it.
    def _boom(**kwargs: Any) -> Tuple[str, Optional[str]]:
        raise AssertionError("provisioning must not run without opt-in")

    sm._provision_apparmor_for_session = _boom  # type: ignore[assignment]
    server = _FakeServer(apparmor=False)

    _drive_provision(sm, server, opt_in=False)

    assert len(spawn_calls) == 1
    assert spawn_calls[0]["confinement_required"] is False
    assert spawn_calls[0]["profile_name"] == ""
    assert server.bootstrap_outcomes == []


# ----------------------------------------------------------------------
# Point 1 — the invariant is threaded on the success path
# ----------------------------------------------------------------------


def test_success_threads_confinement_required_to_spawn(
    _no_plugin_rules: None,
) -> None:
    """Opted in + host supports AppArmor + provisioning SUCCEEDED (non-empty
    profile) → spawn with ``confinement_required=True`` and the profile, no
    refusal.  This is the defence-in-depth arming of the runner-side gate."""
    sm, spawn_calls = _make_sm(
        apparmor_manager=_FakeAppArmor(available=True),
        provision_result=("jaato-ws-boundary-abc", "apparmor"),
    )
    server = _FakeServer(apparmor=True)

    _drive_provision(sm, server, opt_in=True)

    assert len(spawn_calls) == 1
    assert spawn_calls[0]["confinement_required"] is True
    assert spawn_calls[0]["profile_name"] == "jaato-ws-boundary-abc"
    assert server.bootstrap_outcomes == []


# ----------------------------------------------------------------------
# Point 1 — the invariant rides the bootstrap envelope to the runner
# ----------------------------------------------------------------------


def test_spawn_threads_confinement_required_to_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_spawn_session_runner_unconditional(confinement_required=True)`` passes
    it to ``dispatch_bootstrap_envelope`` (which stamps
    ``SessionInitEnvelope.confinement_required`` for the runner gate).

    Drives the REAL ``_spawn_session_runner_unconditional`` with the two
    runner_spawn helpers stubbed, mirroring the WS-hook test's ``_patch_spawn``.
    """
    bootstrap_calls: List[Dict[str, Any]] = []

    monkeypatch.setattr(
        "jaato_server.server.runner_spawn.spawn_session_runner",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "jaato_server.server.runner_spawn.dispatch_bootstrap_envelope",
        lambda **kwargs: bootstrap_calls.append(kwargs),
    )

    sm = SessionManager.__new__(SessionManager)
    sm._daemon_loop = object()  # type: ignore[attr-defined]
    sm._pool_manager_ref = None  # type: ignore[attr-defined]
    sm.get_cascade_budget = lambda *a, **k: None  # type: ignore[assignment]
    sm._reconcile_cascade_pool = lambda *a, **k: None  # type: ignore[assignment]
    sm._record_runner_identity = lambda *a, **k: None  # type: ignore[assignment]
    sm._teardown_prior_apparmor_profile_after_transition = (  # type: ignore[assignment]
        lambda **k: None
    )
    sm._resolve_cgroups_manager = lambda: None  # type: ignore[assignment]
    sm._notify_apparmor = lambda *a, **k: None  # type: ignore[assignment]

    server = _FakeServer(apparmor=True)

    ok = sm._spawn_session_runner_unconditional(
        server=server,
        session_id="sess-1",
        workspace_path="/ws/one",
        client_id="client-1",
        profile_name="jaato-ws-boundary-abc",
        confinement_required=True,
    )

    assert ok is True
    assert len(bootstrap_calls) == 1
    assert bootstrap_calls[0]["confinement_required"] is True
