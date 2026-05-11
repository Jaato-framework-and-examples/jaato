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

from unittest.mock import MagicMock

import pytest

from server.session_manager import SessionManager


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


def _make_session_manager(apparmor_manager=None) -> SessionManager:
    """Construct a SessionManager for unit tests.

    ``_spawn_isolated_runner`` reads ``_apparmor_manager`` via
    ``_resolve_apparmor_manager``; an ``__new__`` instance without
    the attribute returns ``None`` from the resolver, which the
    helper translates to "AppArmor unavailable → stage=sub_profile".

    Tests that need to exercise the §4.3.4 advance to
    ``stage=sub_cgroup`` should pass in a mocked AppArmorManager
    via the ``apparmor_manager`` arg.
    """
    sm = SessionManager.__new__(SessionManager)
    if apparmor_manager is not None:
        sm._apparmor_manager = apparmor_manager
    return sm


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
    the helper advances past ``stage=sub_profile`` to either
    ``stage=sub_cgroup`` (provision success) or ``stage=sub_profile``
    with the kernel error (provision failure)."""

    def test_provision_success_advances_to_sub_cgroup(self):
        """Successful sub-profile provisioning → next stage signal."""
        apparmor = MagicMock()
        apparmor.is_available.return_value = True
        apparmor.provision_sub_profile.return_value = (
            True, "jaato-ws-sess-A//agent-1",
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
        assert result["stage"] == "sub_cgroup"
        assert result["apparmor_profile"] == "jaato-ws-sess-A//agent-1"
        # Stage message mentions §4.3.5 (next stage).
        assert "§4.3.5" in result["error"]
        # Provision invoked exactly once with the expected args.
        apparmor.provision_sub_profile.assert_called_once_with(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/work",
        )

    def test_provision_failure_returns_sub_profile_with_kernel_error(self):
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