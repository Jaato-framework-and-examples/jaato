"""Tests for ``shared.session_envelope`` — daemon → runner handshake.

Phase 3 §3.3a.

Pins the contract:
- Round-trip identity (envelope serializes + deserializes
  byte-identically).
- Versioning field present + checked on decode.
- Required fields enforced at decode.
- Optional fields default cleanly.
- Future-version envelopes refused (forward-compat is opt-in,
  not free).
- Oversize handling reuses the §2.4 framing constraint (10 MB cap)
  via :func:`shared.framing.write_frame` — verified by writing a
  giant envelope through the framing module.
"""

from __future__ import annotations

import json
import struct

import pytest

from shared.framing import (
    HEADER_SIZE,
    MAX_MESSAGE_SIZE,
    FrameTooLargeError,
    read_frame_sync,
    write_frame_sync,
)
from shared.session_envelope import (
    SESSION_ENVELOPE_VERSION,
    SessionInitEnvelope,
)


# ----------------------------------------------------------------------
# Round-trip identity
# ----------------------------------------------------------------------


def test_minimal_envelope_round_trip() -> None:
    e = SessionInitEnvelope(
        session_id="20260508_120000",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
    )
    d = e.to_dict()
    back = SessionInitEnvelope.from_dict(d)
    assert back == e


def test_suppress_base_instructions_wire_round_trip() -> None:
    # Granular frozenset serializes to a sorted list and restores intact.
    e = SessionInitEnvelope(
        session_id="s1",
        workspace_path="/tmp/ws",
        profile_name="p",
        provider_name="anthropic",
        model_name="m",
        suppress_base_instructions=frozenset({"disk", "constants"}),
    )
    d = e.to_dict()
    assert d["suppress_base_instructions"] == ["constants", "disk"]  # sorted list
    back = SessionInitEnvelope.from_dict(d)
    assert back.suppress_base_instructions == frozenset({"disk", "constants"})


def test_suppress_base_instructions_legacy_bool_wire_compat() -> None:
    # An older daemon emits a bare bool; from_dict normalizes it — true drops
    # {disk, constants}, false suppresses nothing.
    base = {"schema_version": 1, "session_id": "s1"}
    true_env = SessionInitEnvelope.from_dict(
        {**base, "suppress_base_instructions": True}
    )
    assert true_env.suppress_base_instructions == frozenset({"disk", "constants"})
    false_env = SessionInitEnvelope.from_dict(
        {**base, "suppress_base_instructions": False}
    )
    assert false_env.suppress_base_instructions == frozenset()
    # Absent field defaults to suppress-nothing.
    absent_env = SessionInitEnvelope.from_dict(base)
    assert absent_env.suppress_base_instructions == frozenset()


def test_full_envelope_round_trip() -> None:
    e = SessionInitEnvelope(
        session_id="sess-1",
        workspace_path="/tmp/ws",
        profile_name="cli_test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=[
            {"name": "signal_completion", "preload": True},
            {"name": "cli", "preload": False, "config": {"max_workers": 4}},
        ],
        system_instructions="You are a helpful agent.",
        agent_id="researcher",
        gc={"type": "budget", "threshold_percent": 80.0},
        completion_payload_schema={"type": "object", "properties": {}},
        completion_processors=[
            {"script": "scripts/processors/codegen_files_exist.py",
             "on_error": "fail_completion"},
            {"script": "scripts/processors/render_report.py",
             "output": "report.md",
             "on_error": "fail_completion"},
        ],
        agent_params={"case_id": "case-42"},
        config_root="/srv/operator/.jaato",
        env_overrides={"JAATO_PROVIDER": "anthropic"},
    )
    d = e.to_dict()
    back = SessionInitEnvelope.from_dict(d)
    assert back == e


def test_completion_processors_round_trip() -> None:
    """``completion_processors`` must survive to_dict/from_dict.

    Regression guard for the unified processor wire surface (server
    0.6.125+): replaces the prior ``completion_validators`` +
    ``completion_artifacts`` envelope fields with one
    ``completion_processors`` list.
    """
    e = SessionInitEnvelope(
        session_id="sess-processors",
        workspace_path="/tmp/ws",
        profile_name="codegen",
        provider_name="openrouter",
        model_name="claude-sonnet-4.5",
        completion_processors=[
            {"script": "scripts/processors/codegen_files_exist.py",
             "on_error": "fail_completion"},
        ],
    )
    back = SessionInitEnvelope.from_dict(e.to_dict())
    assert back.completion_processors == [
        {"script": "scripts/processors/codegen_files_exist.py",
         "on_error": "fail_completion"},
    ]


def test_completion_processors_default_empty() -> None:
    """Wire payloads without the field decode to empty list."""
    e = SessionInitEnvelope.from_dict({
        "schema_version": SESSION_ENVELOPE_VERSION,
        "session_id": "old",
        "workspace_path": None,
        "profile_name": None,
        "provider_name": "anthropic",
        "model_name": "claude-sonnet-4-6",
        # No completion_processors key
    })
    assert e.completion_processors == []


def test_envelope_carries_model_tiers() -> None:
    """v3 (2026-05-14): ``model_tiers`` must round-trip so the runner
    can resolve ``ModelTierConfig`` and register the ``enter_tier``
    lifecycle tool.  Before v3 the field didn't exist on the envelope
    and ``profile.model_tiers`` never reached the runner, suppressing
    tier mode for every pool-served session."""
    tiers = {
        "planner": "glm-5",
        "dispatcher": "glm-5-turbo",
        "executor": "glm-4.7-flash",
        "initial": "dispatcher",
        "fallback": "dispatcher",
    }
    e = SessionInitEnvelope(
        session_id="sess-tiers",
        workspace_path="/tmp/ws",
        profile_name="tier-test",
        provider_name="zhipuai",
        model_name="glm-5-turbo",
        model_tiers=tiers,
    )
    d = e.to_dict()
    assert d["model_tiers"] == tiers
    back = SessionInitEnvelope.from_dict(d)
    assert back == e
    assert back.model_tiers == tiers


def test_envelope_model_tiers_defaults_none() -> None:
    """Single-model sessions leave ``model_tiers=None``; absence in
    the wire dict round-trips as ``None`` (not ``{}``) so the runner's
    ``ModelTierConfig.resolve`` short-circuits cleanly."""
    e = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name=None,
        provider_name="anthropic",
        model_name="m",
    )
    d = e.to_dict()
    assert d["model_tiers"] is None
    back = SessionInitEnvelope.from_dict(d)
    assert back.model_tiers is None


def test_envelope_carries_cascade_driver_id() -> None:
    """v4 (2026-05-20): ``cascade_driver_id`` round-trips so the
    runner can stash it on JaatoSession for cascade-sharing slot reuse.
    See docs/design/runner-cascade-sharing.md §4.1."""
    e = SessionInitEnvelope(
        session_id="sess-cascade",
        workspace_path="/tmp/ws",
        profile_name="cascade-test",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        cascade_driver_id="cascade-X-abc123",
    )
    d = e.to_dict()
    assert d["cascade_driver_id"] == "cascade-X-abc123"
    back = SessionInitEnvelope.from_dict(d)
    assert back == e
    assert back.cascade_driver_id == "cascade-X-abc123"


def test_envelope_cascade_driver_id_defaults_none() -> None:
    """Standalone sessions leave ``cascade_driver_id=None``; absence
    in the wire dict round-trips as ``None`` (treated as 'no cascade
    affinity' by ``PoolManager.acquire_slot``)."""
    e = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name=None,
        provider_name="anthropic",
        model_name="m",
    )
    d = e.to_dict()
    assert d["cascade_driver_id"] is None
    back = SessionInitEnvelope.from_dict(d)
    assert back.cascade_driver_id is None


def test_envelope_v4_omits_cascade_field_back_compat() -> None:
    """A wire payload without the cascade_driver_id key decodes
    cleanly with the field defaulting to None — back-compat for
    older daemons rolling out v4 envelopes against runners that
    encoded v3-shaped dicts in cache / replay logs."""
    e = SessionInitEnvelope.from_dict({
        "schema_version": SESSION_ENVELOPE_VERSION,
        "session_id": "old",
        "workspace_path": None,
        "profile_name": None,
        "provider_name": "anthropic",
        "model_name": "claude-sonnet-4-6",
        # No cascade_driver_id key
    })
    assert e.cascade_driver_id is None


def test_envelope_jsonable() -> None:
    """``to_dict`` produces a JSON-serializable structure (no
    callables, datetimes, etc.).  Phase 3 §3.3a explicitly requires
    JSON friendliness — anything richer must be reduced before
    construction."""
    e = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name=None,
        provider_name="anthropic",
        model_name="m",
    )
    encoded = json.dumps(e.to_dict())
    back = SessionInitEnvelope.from_dict(json.loads(encoded))
    assert back == e


def test_envelope_handles_none_optional_fields() -> None:
    """``workspace_path``, ``profile_name``, ``system_instructions``,
    ``gc``, ``completion_payload_schema``, ``config_root`` may all
    be None; round-trip preserves None."""
    e = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name=None,
        provider_name="anthropic",
        model_name="m",
    )
    d = e.to_dict()
    assert d["workspace_path"] is None
    assert d["profile_name"] is None
    assert d["system_instructions"] is None
    assert d["gc"] is None
    back = SessionInitEnvelope.from_dict(d)
    assert back.workspace_path is None
    assert back.profile_name is None


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------


def test_envelope_default_lists_and_dicts_are_independent() -> None:
    """Default ``plugins=[]`` etc. must NOT be the shared
    class-level mutable; modifying one envelope's defaults must not
    affect another."""
    e1 = SessionInitEnvelope(
        session_id="a", workspace_path=None, profile_name=None,
        provider_name="p", model_name="m",
    )
    e2 = SessionInitEnvelope(
        session_id="b", workspace_path=None, profile_name=None,
        provider_name="p", model_name="m",
    )
    e1.plugins.append({"name": "cli", "preload": False})
    assert e2.plugins == []  # not corrupted


def test_envelope_schema_version_default() -> None:
    e = SessionInitEnvelope(
        session_id="s", workspace_path=None, profile_name=None,
        provider_name="p", model_name="m",
    )
    assert e.schema_version == SESSION_ENVELOPE_VERSION


# ----------------------------------------------------------------------
# Versioning
# ----------------------------------------------------------------------


def test_decode_missing_version_raises() -> None:
    bad = {
        "session_id": "s",
        "provider_name": "p",
        "model_name": "m",
    }
    with pytest.raises(ValueError, match="missing 'schema_version'"):
        SessionInitEnvelope.from_dict(bad)


def test_decode_future_version_refused() -> None:
    """A higher-than-known schema_version means the daemon is newer
    than the runner; mid-deploy skew should fail loudly so the
    daemon log surfaces it."""
    future = {
        "schema_version": SESSION_ENVELOPE_VERSION + 100,
        "session_id": "s",
        "provider_name": "p",
        "model_name": "m",
    }
    with pytest.raises(ValueError, match="runner-supported"):
        SessionInitEnvelope.from_dict(future)


def test_decode_equal_version_accepted() -> None:
    d = SessionInitEnvelope(
        session_id="s", workspace_path=None, profile_name=None,
        provider_name="p", model_name="m",
    ).to_dict()
    assert d["schema_version"] == SESSION_ENVELOPE_VERSION
    back = SessionInitEnvelope.from_dict(d)
    assert back.schema_version == SESSION_ENVELOPE_VERSION


# ----------------------------------------------------------------------
# Re-export from runner/envelope
# ----------------------------------------------------------------------


def test_runner_envelope_reexports_session_init() -> None:
    """The runner imports ``SessionInitEnvelope`` from a single
    canonical path under ``server.runner.envelope``."""
    from server.runner.envelope import (
        SESSION_ENVELOPE_VERSION as runner_version,
        SessionInitEnvelope as RunnerSessionInitEnvelope,
    )
    assert runner_version == SESSION_ENVELOPE_VERSION
    assert RunnerSessionInitEnvelope is SessionInitEnvelope


# ----------------------------------------------------------------------
# Oversize: framing constraint
# ----------------------------------------------------------------------


def test_oversize_envelope_rejected_at_framing() -> None:
    """An envelope serialized to > MAX_MESSAGE_SIZE bytes is rejected
    by the framing read.  Phase 3 §3.3a explicitly defers oversize
    handling to the §2.4 framing constraint; verify that constraint
    fires here.
    """
    import socket

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Forge an oversized header on the wire (we don't actually
        # need to construct a 10MB+ envelope; the framing check is
        # at the header).
        a.sendall(struct.pack(">I", MAX_MESSAGE_SIZE + 1))
        with pytest.raises(FrameTooLargeError):
            read_frame_sync(b)
    finally:
        a.close()
        b.close()


def test_envelope_at_framing_boundary_succeeds() -> None:
    """A non-trivial envelope round-trips fine through the framing.

    Uses a 16KB payload (well under default socket buffers) so the
    test exercises the wire format without a thread-synchronization
    dance.  The §2.4 framing module already covers larger-than-buffer
    cases in its own test suite.
    """
    import socket

    big_instructions = "x" * 16384  # 16KB; fits in socket buffer
    e = SessionInitEnvelope(
        session_id="s",
        workspace_path=None,
        profile_name=None,
        provider_name="p",
        model_name="m",
        system_instructions=big_instructions,
    )
    encoded = json.dumps(e.to_dict())

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        write_frame_sync(a, encoded)
        a.shutdown(socket.SHUT_WR)
        out = read_frame_sync(b)
        assert out is not None
        decoded = SessionInitEnvelope.from_dict(json.loads(out))
        assert decoded.system_instructions == big_instructions
    finally:
        a.close()
        b.close()


# ----------------------------------------------------------------------
# Required-fields enforcement at decode
# ----------------------------------------------------------------------


def test_decode_missing_session_id_raises() -> None:
    bad = {
        "schema_version": SESSION_ENVELOPE_VERSION,
        "provider_name": "p",
        "model_name": "m",
    }
    with pytest.raises(KeyError, match="session_id"):
        SessionInitEnvelope.from_dict(bad)


def test_decode_default_provider_and_model_empty_strings() -> None:
    """provider_name + model_name default to empty strings on
    decode when missing — they're not optional in spirit but the
    decode is permissive so the runner's own validation can surface
    a more useful error than KeyError.
    """
    d = {
        "schema_version": SESSION_ENVELOPE_VERSION,
        "session_id": "s",
    }
    e = SessionInitEnvelope.from_dict(d)
    assert e.provider_name == ""
    assert e.model_name == ""


# ----------------------------------------------------------------------
# Phase 3 §3.12.0 — BootstrapEnvelope (SessionManager-level)
# ----------------------------------------------------------------------


from shared.session_envelope import BootstrapEnvelope


def test_bootstrap_envelope_minimal_construction() -> None:
    """Required fields are session_id + workspace_path + name; the
    rest default."""
    env = BootstrapEnvelope(
        session_id="s-1",
        workspace_path="/tmp/ws",
        name="my-session",
    )
    assert env.session_id == "s-1"
    assert env.workspace_path == "/tmp/ws"
    assert env.name == "my-session"
    # Path discriminators all default to None.
    assert env.client_id is None
    assert env.parent_runner_handle is None
    assert env.sandbox_mode is None
    assert env.restore_state is None
    # Construction fields default sensibly.
    assert env.env_file is None
    assert env.profile is None
    assert env.agent_name == "main"
    assert env.system_instruction_override is None
    assert env.suppress_base_instructions == frozenset()
    assert env.env_overrides == {}
    assert env.config_root is None
    assert env.instruction_token_cache is None
    # Session record fields.
    assert env.provisioned is False
    assert env.created_by is None
    assert env.timestamp is None
    # Bootstrap-time event sink.
    assert env.on_event_during_init is None


def test_bootstrap_envelope_default_dicts_are_independent() -> None:
    """Default-factory dicts must not be shared across instances."""
    env_a = BootstrapEnvelope(
        session_id="a", workspace_path=None, name="A",
    )
    env_b = BootstrapEnvelope(
        session_id="b", workspace_path=None, name="B",
    )
    env_a.env_overrides["X"] = "1"
    assert env_b.env_overrides == {}


def test_bootstrap_envelope_carries_path_discriminators() -> None:
    """The four path discriminators per the §3.12.0 spec accept the
    expected types: client_id (str), parent_runner_handle (any),
    sandbox_mode (str), restore_state (dict)."""
    parent_handle = object()  # opaque handle; daemon-internal only
    env = BootstrapEnvelope(
        session_id="s-2",
        workspace_path="/tmp/ws",
        name="ephemeral-spawn",
        client_id=None,
        parent_runner_handle=parent_handle,
        sandbox_mode="apparmor",
        restore_state={"some": "saved-state"},
    )
    assert env.client_id is None
    assert env.parent_runner_handle is parent_handle
    assert env.sandbox_mode == "apparmor"
    assert env.restore_state == {"some": "saved-state"}


def test_bootstrap_envelope_holds_callable_event_sink() -> None:
    """``on_event_during_init`` is a daemon-internal Callable; the
    envelope must accept it without complaint (no JSON-shape
    constraint since BootstrapEnvelope doesn't cross the wire)."""
    received = []

    def _sink(e):
        received.append(e)

    env = BootstrapEnvelope(
        session_id="s-3",
        workspace_path=None,
        name="test",
        on_event_during_init=_sink,
    )
    env.on_event_during_init("hello")
    assert received == ["hello"]


def test_bootstrap_envelope_independent_from_session_init_envelope() -> None:
    """BootstrapEnvelope is daemon-internal; SessionInitEnvelope
    (re-exported from runner) is the wire form.  They share no
    state and serve different layers."""
    bootstrap = BootstrapEnvelope(
        session_id="s-4",
        workspace_path=None,
        name="iso",
    )
    init = SessionInitEnvelope(
        session_id="s-4",
        workspace_path=None,
        profile_name=None,
        provider_name="x",
        model_name="y",
    )
    # Both classes coexist; the bootstrap envelope's fields don't
    # overlap with the init envelope's (the init envelope is
    # JaatoSession-level, the bootstrap is SessionManager-level).
    assert not hasattr(init, "client_id")
    assert not hasattr(bootstrap, "schema_version")
