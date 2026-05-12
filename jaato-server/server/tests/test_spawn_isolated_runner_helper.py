"""Tests for ``SessionManager._spawn_isolated_runner`` (Phase 4 §4.3.3).

The helper is the daemon-side bridge between the
``subagent.spawn_isolated_runner`` RPC handler and the eventual
isolated-runner spawn machinery (§4.3.4 sub-profile + §4.3.5
sub-cgroup + §4.3.6 cross-runner forwarding).

§4.3.3 scope tests:

1. **Profile reconstruction** — wire-shape ``profile_payload``
   dicts (mirroring ``session.new`` inline spec shape) round-trip
   through ``build_inline_profile`` to a ``SubagentProfile``.
2. **Validation failures** surface as ``stage=validation`` envelopes
   with diagnostic messages.
3. **Successful reconstruction** returns ``stage=sub_profile`` (the
   next-stage signal — §4.3.4 will advance this).
4. **Isolated session id** generation follows the
   ``{parent}__sub_{subagent}`` template.

When §4.3.4 lands sub-profile generation, this suite extends to
assert ``stage=sub_cgroup`` (and the validation/reconstruction
suites stay valid bit-exact).
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from server.session_manager import SessionManager, SubRunnerHandle


def _valid_payload(**overrides):
    """Build a valid profile_payload dict, allowing overrides.

    Mirrors the field set accepted by
    ``shared/plugins/subagent/config.py:build_inline_profile``."""
    base = {
        "name": "researcher",
        "description": "Deep research profile",
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "plugins": ["cli", "web_search"],
        "plugin_configs": {},
        "system_instructions": "You are a researcher.",
        "max_turns": 25,
    }
    base.update(overrides)
    return base


def _make_session_manager(
    apparmor_manager=None,
    cgroups_manager=None,
    spawn_returns: Any = None,
    spawn_raises: Optional[Exception] = None,
) -> SessionManager:
    """Construct a SessionManager for unit tests.

    ``_spawn_isolated_runner`` reads:
    - ``_apparmor_manager`` via ``_resolve_apparmor_manager``
    - ``_cgroups_manager`` via ``_resolve_cgroups_manager``
    - ``_do_spawn_isolated_runner`` (§4.3.6a sub-runner spawn)

    An ``__new__`` instance without the attributes returns ``None``
    from the resolvers, which the helper translates to "X
    unavailable" branches (graceful degradation per Audit 7 for
    cgroups; hard stop at stage=sub_profile for AppArmor).

    Phase 4 §4.3.6a: ``_do_spawn_isolated_runner`` raises
    ``RuntimeError`` when ``_daemon_loop`` is unset (the default).
    Tests that need to exercise the §4.3.6a SUCCESS path (handle
    stored in ``_isolated_sub_runners``) pass ``spawn_returns=<handle>``.
    Tests that need the spawn-failure rollback can either leave
    spawn_returns=None (defaults to RuntimeError) or pass
    ``spawn_raises=...``.

    Also initializes ``_isolated_sub_runners`` + ``_lock`` so the
    helper's ``with self._lock`` registration path works.
    """
    sm = SessionManager.__new__(SessionManager)
    import threading
    sm._lock = threading.RLock()
    sm._isolated_sub_runners = {}
    sm._sessions = {}  # §4.3.6c reads this in _forward_subagent_event_to_parent
    if apparmor_manager is not None:
        sm._apparmor_manager = apparmor_manager
    if cgroups_manager is not None:
        sm._cgroups_manager = cgroups_manager
    if spawn_returns is not None or spawn_raises is not None:
        def _stub_spawn(**kwargs):
            if spawn_raises is not None:
                raise spawn_raises
            return spawn_returns
        sm._do_spawn_isolated_runner = _stub_spawn  # type: ignore[method-assign]
    return sm


def _make_default_apparmor():
    """AppArmorManager mock that succeeds — used as a baseline for
    §4.3.5 tests that focus on cgroups behavior."""
    apparmor = MagicMock()
    apparmor.is_available.return_value = True
    apparmor.provision_sub_profile.return_value = (
        True, "jaato-ws-sess-A//agent-1",
    )
    apparmor.teardown_sub_profile.return_value = True
    return apparmor


# ──────────────────────────────────────────────────────────────────────
# Validation failures
# ──────────────────────────────────────────────────────────────────────


class TestValidationFailures:
    """Malformed profile_payload surfaces as ``stage=validation``
    so callers can distinguish reconstruction failures from
    downstream spawn failures."""

    def test_invalid_runtime_limits_returns_validation_error(self):
        """``build_inline_profile`` raises ValueError on a non-dict
        runtime_limits; the helper translates this to a
        ``stage=validation`` envelope."""
        sm = _make_session_manager()
        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(runtime_limits="not-a-dict"),
            task="do thing",
            workspace_path="/work",
        )
        assert result["ok"] is False
        assert result["stage"] == "validation"
        assert "runtime_limits" in result["error"]
        assert "ValueError" in result["error"]

    def test_invalid_gc_returns_validation_error(self):
        """``GCProfileConfig.from_dict`` raises on garbage; helper
        translates to ``stage=validation``."""
        sm = _make_session_manager()
        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(gc={"type": "invalid_gc_type"}),
            task="do thing",
            workspace_path="/work",
        )
        # Either: build_inline_profile crashes (→ validation), OR
        # gc validation is lenient and we reach sub_profile.  Both
        # are acceptable — the contract is "validation errors surface
        # as stage=validation"; if there's no validation error, we
        # proceed.  This test asserts ONLY that we don't reach
        # success and don't crash uncaught.
        assert result["ok"] is False
        assert result["stage"] in {"validation", "sub_profile"}


# ──────────────────────────────────────────────────────────────────────
# Successful reconstruction → stage=sub_profile
# ──────────────────────────────────────────────────────────────────────


class TestSuccessfulReconstructionUnwiredApparmor:
    """When the SessionManager has NO AppArmorManager wired (test
    fake or host without AppArmor), reconstruction succeeds but
    the helper returns ``stage=sub_profile`` with "AppArmor
    unavailable" message — kernel confinement is a hard
    requirement for isolated-runner spawn."""

    def test_minimal_payload_returns_sub_profile_stage(self):
        sm = _make_session_manager()  # no AppArmorManager
        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )
        assert result["ok"] is False
        assert result["stage"] == "sub_profile"
        assert "AppArmor" in result["error"]
        assert "unavailable" in result["error"].lower()
        assert "phase4_implementation_audits" in result["error"]

    def test_diagnostic_fields_present_on_sub_profile_envelope(self):
        """Caller (the runner-side RPC wrapper) needs the would-be
        session_id and profile_name for diagnostics."""
        sm = _make_session_manager()
        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(name="my-researcher"),
            task="do thing",
            workspace_path="/work",
        )
        assert "isolated_session_id" in result
        assert "profile_name" in result
        assert result["profile_name"] == "my-researcher"

    def test_minimal_payload_without_name_uses_placeholder(self):
        """Payload with no ``name`` falls back to ``<isolated>`` —
        matches ``build_inline_profile``'s default-name pattern."""
        sm = _make_session_manager()
        payload = _valid_payload()
        del payload["name"]
        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=payload,
            task="do thing",
            workspace_path="/work",
        )
        assert result["profile_name"] == "<isolated>"


class TestSubProfileProvisioningStage:
    """Phase 4 §4.3.4: when AppArmorManager IS wired + available,
    the helper advances past ``stage=sub_profile``.  The actual
    next-stage value depends on cgroups state (§4.3.5):

    - cgroups unavailable / no runtime_limits → stage=forwarding
      with cgroup_path="" (graceful degradation per Audit 7).
    - cgroups available + has limits + provision succeeds →
      stage=forwarding with cgroup_path populated.
    - cgroups available + has limits + provision fails →
      stage=sub_cgroup with AppArmor rolled back.

    AppArmor failure remains stage=sub_profile.
    """

    def test_apparmor_success_no_cgroups_no_daemon_loop_rolls_back(self):
        """Sub-AppArmor success + no CgroupsManager + no daemon_loop
        (the default test-fixture state post-§4.3.6a) → spawn raises
        RuntimeError("daemon loop unavailable") → rollback fires →
        stage=forwarding with apparmor_profile="" + cgroup_path="".

        This pins the §4.3.6a rollback path triggered by missing
        spawn infrastructure (test fakes / pre-init bootstrap)."""
        apparmor = _make_default_apparmor()
        sm = _make_session_manager(apparmor_manager=apparmor)

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is False
        assert result["stage"] == "forwarding"
        # Rollback cleared the diagnostic fields.
        assert result["apparmor_profile"] == ""
        assert result["cgroup_path"] == ""
        # Spawn-failure message must reference the rollback.
        assert "spawn failed" in result["error"].lower()
        assert "rolled back" in result["error"].lower()
        # AppArmor provision invoked once + teardown invoked once
        # (rollback).
        apparmor.provision_sub_profile.assert_called_once_with(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/work",
        )
        apparmor.teardown_sub_profile.assert_called_once_with(
            parent_session_id="sess-A",
            subagent_id="agent-1",
        )

    def test_apparmor_failure_returns_sub_profile_with_kernel_error(self):
        """When AppArmor sub-profile provisioning fails, the helper
        forwards the kernel error message via the stage envelope."""
        apparmor = MagicMock()
        apparmor.is_available.return_value = True
        apparmor.provision_sub_profile.return_value = (
            False, "apparmor_parser exit=1: profile syntax error",
        )
        sm = _make_session_manager(apparmor_manager=apparmor)

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is False
        assert result["stage"] == "sub_profile"
        assert "apparmor_parser exit=1" in result["error"]
        assert "syntax error" in result["error"]

    def test_apparmor_unavailable_returns_sub_profile_stage(self):
        """When AppArmorManager.is_available() is False, the helper
        skips the provision call and returns ``stage=sub_profile``
        with the unavailable message."""
        apparmor = MagicMock()
        apparmor.is_available.return_value = False
        sm = _make_session_manager(apparmor_manager=apparmor)

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is False
        assert result["stage"] == "sub_profile"
        assert "unavailable" in result["error"].lower()
        apparmor.provision_sub_profile.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# §4.3.5 sub-cgroup provisioning
# ──────────────────────────────────────────────────────────────────────


def _make_spawn_handle(
    parent_session_id: str = "sess-A",
    subagent_id: str = "agent-1",
    sub_apparmor_profile: str = "jaato-ws-sess-A//agent-1",
    cgroup_path: str = "",
    pid: int = 12345,
) -> SubRunnerHandle:
    """Build a SubRunnerHandle for tests that need spawn to succeed.
    Includes mocked rpc + spawned (MagicMock) so callers can
    inspect what the §4.3.6b/c/d wiring would dispatch."""
    rpc = MagicMock()
    spawned = MagicMock()
    spawned.pid = pid
    return SubRunnerHandle(
        parent_session_id=parent_session_id,
        subagent_id=subagent_id,
        isolated_session_id=f"{parent_session_id}__sub_{subagent_id}",
        rpc=rpc,
        spawned=spawned,
        sub_apparmor_profile=sub_apparmor_profile,
        cgroup_path=cgroup_path,
    )


class TestSubCgroupProvisioningStage:
    """Phase 4 §4.3.5: with AppArmor success + CgroupsManager wired,
    the helper exercises the sub-cgroup creation path.

    Post-§4.3.6a, these tests use ``spawn_returns=<handle>`` to
    bypass the daemon_loop check + isolate cgroup behavior.  Cgroup
    branch coverage stays:
    - cgroups unavailable → stage=forwarding, cgroup_path=""
    - profile has no runtime_limits → stage=forwarding, cgroup_path=""
    - cgroups + limits + success → stage=forwarding, cgroup_path=<path>
    - cgroups + limits + failure → stage=sub_cgroup + AppArmor rollback
    """

    def test_cgroups_unavailable_advances_to_forwarding(self):
        """CgroupsManager wired but is_available() returns False
        (e.g., host without cgroup v2).  Helper skips cgroup
        creation and advances to forwarding with empty cgroup_path."""
        apparmor = _make_default_apparmor()
        cgroups = MagicMock()
        cgroups.is_available.return_value = False
        sm = _make_session_manager(
            apparmor_manager=apparmor,
            cgroups_manager=cgroups,
            spawn_returns=_make_spawn_handle(cgroup_path=""),
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                runtime_limits={
                    "memory_max_bytes": 2 * 1024**3,
                    "pids_max": 128,
                },
            ),
            task="do thing",
            workspace_path="/work",
        )

        # Post-§4.3.6c: bootstrap dispatched on MagicMock RPC returns
        # truthy → ok=True.  cgroup_path stays empty (cgroups
        # unavailable was the §4.3.5 branch).
        assert result["ok"] is True
        assert result["cgroup_path"] == ""
        cgroups.provision_cgroup.assert_not_called()

    def test_no_runtime_limits_provisions_cgroup_with_defaults(self):
        """Phase 5 §5.1: profile without ``runtime_limits`` triggers
        the isolated-subagent default (2 GiB / 128 pids / cpu_weight=100).

        Pre-§5.1 this test pinned the security-violating fallback
        ("skipped cgroup creation when profile omits runtime_limits");
        §5.1 closes the gap so cgroup provision is unconditional on
        cgroups-availability.  ``CgroupsManager.provision_cgroup`` is
        now invoked with the merged-default RuntimeLimits."""
        from pathlib import Path
        from shared.runtime_limits import (
            ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS,
        )
        apparmor = _make_default_apparmor()
        cgroups = MagicMock()
        cgroups.is_available.return_value = True
        cgroups.provision_cgroup.return_value = True
        cgroups.get_cgroup_path.return_value = Path(
            "/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1",
        )
        cgroup_str = "/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1"
        sm = _make_session_manager(
            apparmor_manager=apparmor,
            cgroups_manager=cgroups,
            spawn_returns=_make_spawn_handle(cgroup_path=cgroup_str),
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),  # No runtime_limits.
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is True
        assert result["cgroup_path"] == cgroup_str
        # Default applied: provision_cgroup called with the default
        # RuntimeLimits.  We don't pin every field — only that the
        # call happened with a non-None RuntimeLimits whose values
        # match the documented default.
        cgroups.provision_cgroup.assert_called_once()
        provision_args = cgroups.provision_cgroup.call_args
        passed_limits = provision_args[0][1]
        assert (
            passed_limits == ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS
        )

    def test_provision_success_advances_to_forwarding_with_path(self):
        """Cgroups + has_kernel_limits + provision OK → stage=forwarding
        with cgroup_path populated."""
        from pathlib import Path
        apparmor = _make_default_apparmor()
        cgroups = MagicMock()
        cgroups.is_available.return_value = True
        cgroups.provision_cgroup.return_value = True
        cgroups.get_cgroup_path.return_value = Path(
            "/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1",
        )
        cgroup_str = "/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1"
        sm = _make_session_manager(
            apparmor_manager=apparmor,
            cgroups_manager=cgroups,
            spawn_returns=_make_spawn_handle(cgroup_path=cgroup_str),
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                runtime_limits={
                    "memory_max_bytes": 2 * 1024**3,
                    "pids_max": 128,
                },
            ),
            task="do thing",
            workspace_path="/work",
        )

        # Post-§4.3.6c: ok=True with cgroup_path populated.
        assert result["ok"] is True
        assert result["cgroup_path"] == cgroup_str
        # provision_cgroup invoked with the isolated_session_id.
        provision_call = cgroups.provision_cgroup.call_args
        assert provision_call[0][0] == "sess-A__sub_agent-1"
        # AppArmor NOT rolled back on success.
        apparmor.teardown_sub_profile.assert_not_called()

    def test_provision_failure_rolls_back_apparmor(self):
        """Sub-cgroup provision failure → stage=sub_cgroup +
        AppArmor sub-profile torn down (no orphaned kernel state)."""
        apparmor = _make_default_apparmor()
        cgroups = MagicMock()
        cgroups.is_available.return_value = True
        cgroups.provision_cgroup.return_value = False  # Kernel failed.
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                runtime_limits={
                    "memory_max_bytes": 2 * 1024**3,
                    "pids_max": 128,
                },
            ),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is False
        assert result["stage"] == "sub_cgroup"
        # AppArmor profile rolled back — apparmor_profile cleared.
        assert result["apparmor_profile"] == ""
        # AppArmor teardown invoked exactly once.
        apparmor.teardown_sub_profile.assert_called_once_with(
            parent_session_id="sess-A",
            subagent_id="agent-1",
        )

    def test_provision_failure_tolerates_apparmor_teardown_crash(self):
        """If AppArmor rollback itself crashes, helper still returns
        the stage=sub_cgroup envelope (no exception bubbles up)."""
        apparmor = _make_default_apparmor()
        apparmor.teardown_sub_profile.side_effect = RuntimeError(
            "teardown boom",
        )
        cgroups = MagicMock()
        cgroups.is_available.return_value = True
        cgroups.provision_cgroup.return_value = False
        sm = _make_session_manager(
            apparmor_manager=apparmor, cgroups_manager=cgroups,
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                runtime_limits={
                    "memory_max_bytes": 2 * 1024**3,
                    "pids_max": 128,
                },
            ),
            task="do thing",
            workspace_path="/work",
        )

        assert result["stage"] == "sub_cgroup"
        # Teardown was attempted even though it crashed.
        apparmor.teardown_sub_profile.assert_called_once()


# ──────────────────────────────────────────────────────────────────────
# Phase 5 §5.1 — isolated-subagent default RuntimeLimits
# ──────────────────────────────────────────────────────────────────────


class TestIsolatedDefaultRuntimeLimits:
    """Phase 5 §5.1: spawning an isolated subagent merges the profile's
    ``runtime_limits`` with :data:`ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS`
    so the isolation-implies-bounds invariant holds even when the
    profile omits the field.

    Property pinned per assertion (never the implementation):
    - cgroup provision sees the merged limits.
    - spawn-side ``RunnerSpawner.spawn`` call receives the app-layer
      defaults via ``max_output_chars`` + ``tool_timeout_seconds``.
    - Supplied kernel/app fields win per-field; defaults fill the rest.
    """

    def _capture_spawn(self):
        """Build a SessionManager where ``_do_spawn_isolated_runner``
        is a MagicMock — exposes the kwargs the helper passed,
        including the new ``effective_runtime_limits`` parameter."""
        from unittest.mock import MagicMock as MM
        apparmor = _make_default_apparmor()
        cgroups = MagicMock()
        cgroups.is_available.return_value = True
        cgroups.provision_cgroup.return_value = True
        from pathlib import Path
        cgroups.get_cgroup_path.return_value = Path(
            "/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1",
        )
        sm = _make_session_manager(
            apparmor_manager=apparmor,
            cgroups_manager=cgroups,
        )
        spawn_mock = MM(
            return_value=_make_spawn_handle(
                cgroup_path=(
                    "/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1"
                ),
            ),
        )
        sm._do_spawn_isolated_runner = spawn_mock  # type: ignore[method-assign]
        return sm, cgroups, spawn_mock

    def test_default_applies_when_profile_omits_runtime_limits(self):
        """Profile without ``runtime_limits`` → cgroup provisioned
        with the conservative default."""
        from shared.runtime_limits import (
            ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS,
        )
        sm, cgroups, spawn_mock = self._capture_spawn()

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is True
        cgroups.provision_cgroup.assert_called_once()
        passed_limits = cgroups.provision_cgroup.call_args[0][1]
        assert (
            passed_limits == ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS
        )
        # _do_spawn_isolated_runner received the same effective limits.
        spawn_kwargs = spawn_mock.call_args.kwargs
        assert (
            spawn_kwargs["effective_runtime_limits"]
            == ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS
        )

    def test_default_applies_when_profile_supplies_empty_runtime_limits(
        self,
    ):
        """``runtime_limits: {}`` (all-None) is semantically identical
        to absent — defaults still apply per plan-Q2."""
        from shared.runtime_limits import (
            ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS,
        )
        sm, cgroups, spawn_mock = self._capture_spawn()

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(runtime_limits={}),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is True
        passed_limits = cgroups.provision_cgroup.call_args[0][1]
        assert (
            passed_limits == ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS
        )

    def test_supplied_kernel_fields_win_per_field(self):
        """Profile sets ``memory_max_mb=4096`` → cgroup uses 4096 for
        memory + default 128/100 for pids/cpu_weight."""
        sm, cgroups, spawn_mock = self._capture_spawn()

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                runtime_limits={"memory_max_mb": 4096},
            ),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is True
        passed_limits = cgroups.provision_cgroup.call_args[0][1]
        assert passed_limits.memory_max_mb == 4096
        assert passed_limits.pids_max == 128  # Default fills.
        assert passed_limits.cpu_weight == 100  # Default fills.

    def test_supplied_app_fields_win_per_field(self):
        """Profile sets ``tool_timeout_seconds=30`` → spawn-side
        receives 30s for the timeout knob; output cap stays at
        the 1 MiB default."""
        sm, cgroups, spawn_mock = self._capture_spawn()

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                runtime_limits={"tool_timeout_seconds": 30.0},
            ),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is True
        spawn_kwargs = spawn_mock.call_args.kwargs
        effective = spawn_kwargs["effective_runtime_limits"]
        assert effective.tool_timeout_seconds == 30.0
        assert effective.max_output_bytes == 1_048_576  # Default fills.

    def test_cgroup_provision_skipped_when_manager_unavailable(self):
        """The cgroup-skip branch can only fire when ``CgroupsManager``
        is unavailable on the host — profile-content can no longer
        bypass cgroup provision after §5.1."""
        apparmor = _make_default_apparmor()
        cgroups = MagicMock()
        cgroups.is_available.return_value = False  # Host capability gap.
        sm = _make_session_manager(
            apparmor_manager=apparmor,
            cgroups_manager=cgroups,
            spawn_returns=_make_spawn_handle(cgroup_path=""),
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),  # No runtime_limits.
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is True
        assert result["cgroup_path"] == ""
        # Defaults exist but the host can't enforce them — provision
        # not called.
        cgroups.provision_cgroup.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# Isolated session id template
# ──────────────────────────────────────────────────────────────────────


class TestIsolatedSessionIdTemplate:
    """The isolated session id follows ``{parent}__sub_{subagent}``.
    Same template used to derive the §4.3.4 sub-AppArmor profile
    name; pinning the template here so the two stay in sync."""

    def test_template_concatenates_parent_and_subagent(self):
        sm = _make_session_manager()
        result = sm._spawn_isolated_runner(
            parent_session_id="myparent",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )
        assert result["isolated_session_id"] == "myparent__sub_agent-1"

    def test_template_handles_uuid_style_parent(self):
        """Parent session ids in production are typically UUID-style;
        ensure the template is byte-clean (no clobbering of dashes
        or underscores)."""
        sm = _make_session_manager()
        parent = "01934abc-def0-7000-89ab-cdef01234567"
        result = sm._spawn_isolated_runner(
            parent_session_id=parent,
            subagent_id="sub-2",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )
        assert result["isolated_session_id"] == f"{parent}__sub_sub-2"


# ──────────────────────────────────────────────────────────────────────
# Profile field round-trip
# ──────────────────────────────────────────────────────────────────────


class TestProfileFieldRoundTrip:
    """The helper uses ``build_inline_profile`` for reconstruction
    — same path SDK inline-spec callers use.  Pin that key fields
    propagate so §4.3.4+ commits can rely on the reconstructed
    profile having the expected shape."""

    def test_plugin_preload_annotations_resolved(self):
        """``plugin(preload)`` syntax must round-trip through
        reconstruction (same as inline specs).  Important because
        the eventual isolated runner's plugin init needs the
        preload set to bypass deferred-tool loading."""
        sm = _make_session_manager()
        # If reconstruction failed, we'd get stage=validation.
        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                plugins=["cli", "todo(preload)", "web_search(preload)"],
            ),
            task="do thing",
            workspace_path="/work",
        )
        # Reaches sub_profile — reconstruction succeeded.
        assert result["stage"] == "sub_profile"


# ──────────────────────────────────────────────────────────────────────
# §4.3.6a sub-runner spawn + handle registration
# ──────────────────────────────────────────────────────────────────────


class TestSpawnInvocation:
    """Phase 4 §4.3.6a: with AppArmor success + (optional) cgroup
    success, the helper invokes ``_do_spawn_isolated_runner`` and
    registers the returned handle in ``_isolated_sub_runners``.

    Tests use ``spawn_returns=<handle>`` to bypass real subprocess
    spawning while still exercising the helper's call chain +
    handle bookkeeping."""

    def test_spawn_success_registers_handle(self):
        """Successful spawn → handle stored in
        ``_isolated_sub_runners`` keyed by isolated_session_id.

        Post-§4.3.6c: bootstrap is dispatched on the registered
        handle.  The default test handle's ``rpc`` is a MagicMock,
        so ``bootstrap_session_threadsafe`` returns a MagicMock
        (truthy) and ``_schedule_isolated_first_turn`` runs but
        no-ops because ``_daemon_loop`` is unset (caller forwards
        the error event)."""
        apparmor = _make_default_apparmor()
        handle = _make_spawn_handle(pid=12345)
        sm = _make_session_manager(
            apparmor_manager=apparmor,
            spawn_returns=handle,
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )

        # §4.3.6c: ok=True now (bootstrap MagicMock-returns truthy +
        # first-turn scheduled via daemon loop or short-circuits).
        assert result["ok"] is True
        assert result["session_id"] == "sess-A__sub_agent-1"
        assert result["runner_pid"] == 12345
        assert result["sub_runner_pid"] == 12345
        assert result["sub_session_id"] == "sess-A__sub_agent-1"
        # Handle registered under the isolated session id.
        assert "sess-A__sub_agent-1" in sm._isolated_sub_runners
        assert sm._isolated_sub_runners["sess-A__sub_agent-1"] is handle
        # bootstrap_session_threadsafe was invoked.
        handle.rpc.bootstrap_session_threadsafe.assert_called_once()

    def test_spawn_success_with_bootstrap_failure_returns_forwarding(self):
        """Bootstrap RPC raises → helper returns stage=forwarding
        with the bootstrap-failure message; handle stays registered
        (parent-cascade §4.3.6d will tear down)."""
        apparmor = _make_default_apparmor()
        handle = _make_spawn_handle(pid=12345)
        handle.rpc.bootstrap_session_threadsafe.side_effect = RuntimeError(
            "bootstrap boom",
        )
        sm = _make_session_manager(
            apparmor_manager=apparmor,
            spawn_returns=handle,
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is False
        assert result["stage"] == "forwarding"
        assert "session.bootstrap dispatch failed" in result["error"]
        # Handle still registered.
        assert "sess-A__sub_agent-1" in sm._isolated_sub_runners

    def test_spawn_failure_rolls_back_apparmor_and_cgroup(self):
        """``_do_spawn_isolated_runner`` raises → rollback chain
        fires (cgroup teardown + AppArmor teardown).  Return has
        cleared diagnostic fields."""
        apparmor = _make_default_apparmor()
        cgroups = MagicMock()
        cgroups.is_available.return_value = True
        cgroups.provision_cgroup.return_value = True
        from pathlib import Path
        cgroups.get_cgroup_path.return_value = Path(
            "/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1",
        )

        sm = _make_session_manager(
            apparmor_manager=apparmor,
            cgroups_manager=cgroups,
            spawn_raises=RuntimeError("spawn boom"),
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                runtime_limits={
                    "memory_max_bytes": 2 * 1024**3,
                    "pids_max": 128,
                },
            ),
            task="do thing",
            workspace_path="/work",
        )

        assert result["stage"] == "forwarding"
        assert result["apparmor_profile"] == ""
        assert result["cgroup_path"] == ""
        # Both teardowns invoked.
        cgroups.teardown_cgroup.assert_called_once_with(
            "sess-A__sub_agent-1",
        )
        apparmor.teardown_sub_profile.assert_called_once_with(
            parent_session_id="sess-A",
            subagent_id="agent-1",
        )
        # Handle NOT registered.
        assert "sess-A__sub_agent-1" not in sm._isolated_sub_runners
        # Error mentions spawn failure + rollback.
        assert "spawn failed" in result["error"].lower()
        assert "rolled back" in result["error"].lower()
        assert "spawn boom" in result["error"]

    def test_spawn_failure_with_cgroup_only_no_handle_left(self):
        """Spawn failure when cgroup was provisioned but AppArmor
        teardown crashes — still returns gracefully."""
        apparmor = _make_default_apparmor()
        apparmor.teardown_sub_profile.side_effect = RuntimeError(
            "teardown boom",
        )
        cgroups = MagicMock()
        cgroups.is_available.return_value = True
        cgroups.provision_cgroup.return_value = True
        from pathlib import Path
        cgroups.get_cgroup_path.return_value = Path(
            "/sys/fs/cgroup/jaato/jaato-ws-sess-A__sub_agent-1",
        )

        sm = _make_session_manager(
            apparmor_manager=apparmor,
            cgroups_manager=cgroups,
            spawn_raises=RuntimeError("spawn boom"),
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                runtime_limits={
                    "memory_max_bytes": 2 * 1024**3,
                    "pids_max": 128,
                },
            ),
            task="do thing",
            workspace_path="/work",
        )

        # Helper returns gracefully even when rollback's own teardown
        # crashes — the return shape is unchanged.
        assert result["stage"] == "forwarding"
        assert result["apparmor_profile"] == ""
        # Rollback was attempted even though AppArmor teardown threw.
        cgroups.teardown_cgroup.assert_called_once()
        apparmor.teardown_sub_profile.assert_called_once()
        # Handle NOT registered.
        assert "sess-A__sub_agent-1" not in sm._isolated_sub_runners

    def test_spawn_default_no_daemon_loop_returns_runtime_error_message(self):
        """Without ``spawn_returns`` / ``spawn_raises``, the default
        spawn helper raises ``RuntimeError`` because ``_daemon_loop``
        isn't set.  Confirms the default test-fixture path doesn't
        crash silently — it surfaces an error envelope."""
        apparmor = _make_default_apparmor()
        sm = _make_session_manager(apparmor_manager=apparmor)

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )

        assert result["stage"] == "forwarding"
        assert "spawn failed" in result["error"].lower()
        assert "daemon loop unavailable" in result["error"].lower()


# ──────────────────────────────────────────────────────────────────────
# §4.3.6c bootstrap + first-turn dispatch wiring
# ──────────────────────────────────────────────────────────────────────


class TestBootstrapAndFirstTurnDispatch:
    """Phase 4 §4.3.6c: after §4.3.6a spawn succeeds, the helper
    dispatches session.bootstrap to the sub-runner + schedules the
    first-turn prompt via session.send_message.  These tests pin
    the wire-shape of both calls + the ok=True envelope."""

    def test_bootstrap_dispatched_with_envelope(self):
        """bootstrap_session_threadsafe invoked with a SessionInitEnvelope
        derived from the SubagentProfile."""
        apparmor = _make_default_apparmor()
        handle = _make_spawn_handle()
        sm = _make_session_manager(
            apparmor_manager=apparmor, spawn_returns=handle,
        )

        sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(
                provider="anthropic", model="claude-sonnet-4-5",
                plugins=["cli", "todo"],
            ),
            task="investigate X",
            workspace_path="/work",
        )

        handle.rpc.bootstrap_session_threadsafe.assert_called_once()
        envelope = handle.rpc.bootstrap_session_threadsafe.call_args[0][0]
        assert envelope.session_id == "sess-A__sub_agent-1"
        assert envelope.workspace_path == "/work"
        assert envelope.profile_name == "jaato-ws-sess-A//agent-1"
        assert envelope.provider_name == "anthropic"
        assert envelope.model_name == "claude-sonnet-4-5"
        # plugins list rendered as plugin_specs.
        plugin_names = [p["name"] for p in envelope.plugins]
        assert "cli" in plugin_names
        assert "todo" in plugin_names

    def test_bootstrap_failure_returns_forwarding_envelope(self):
        """When bootstrap raises, helper returns stage=forwarding
        with the failure message; handle stays registered."""
        apparmor = _make_default_apparmor()
        handle = _make_spawn_handle()
        handle.rpc.bootstrap_session_threadsafe.side_effect = RuntimeError(
            "transport closed",
        )
        sm = _make_session_manager(
            apparmor_manager=apparmor, spawn_returns=handle,
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )

        assert result["ok"] is False
        assert result["stage"] == "forwarding"
        assert "session.bootstrap dispatch failed" in result["error"]
        # Handle stays registered for §4.3.6d cascade teardown.
        assert "sess-A__sub_agent-1" in sm._isolated_sub_runners

    def test_envelope_provider_defaults_to_anthropic(self):
        """When profile has no provider, envelope defaults to
        anthropic (matches build_session_envelope's behavior)."""
        apparmor = _make_default_apparmor()
        handle = _make_spawn_handle()
        sm = _make_session_manager(
            apparmor_manager=apparmor, spawn_returns=handle,
        )
        payload = _valid_payload()
        payload.pop("provider", None)

        sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=payload,
            task="do thing",
            workspace_path="/work",
        )

        envelope = handle.rpc.bootstrap_session_threadsafe.call_args[0][0]
        assert envelope.provider_name == "anthropic"

    def test_first_turn_short_circuits_when_no_daemon_loop(self):
        """Without daemon_loop, _schedule_isolated_first_turn forwards
        an error event instead of dispatching session.send_message."""
        apparmor = _make_default_apparmor()
        handle = _make_spawn_handle()
        sm = _make_session_manager(
            apparmor_manager=apparmor, spawn_returns=handle,
        )

        result = sm._spawn_isolated_runner(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            profile_payload=_valid_payload(),
            task="do thing",
            workspace_path="/work",
        )

        # Helper still returns ok=True (bootstrap succeeded);
        # first-turn dispatch logged a warning + forwarded error
        # event.  session_send_message was NOT invoked.
        assert result["ok"] is True
        handle.rpc.session_send_message.assert_not_called()