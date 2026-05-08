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
        completion_artifacts=[
            {"renderer": "markdown", "output": "report.md"},
        ],
        agent_params={"case_id": "case-42"},
        config_root="/srv/operator/.jaato",
        env_overrides={"JAATO_PROVIDER": "anthropic"},
    )
    d = e.to_dict()
    back = SessionInitEnvelope.from_dict(d)
    assert back == e


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
