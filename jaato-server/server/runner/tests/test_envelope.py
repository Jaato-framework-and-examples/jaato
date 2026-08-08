"""Tests for ``server.runner.envelope`` — RPC envelope round-trip."""

from __future__ import annotations

import pytest

from server.runner.envelope import (
    KIND_REQUEST,
    KIND_RESPONSE,
    KIND_STREAM,
    KIND_CANCEL,
    STREAM_CHANNEL_DISPLAY,
    CancelFrame,
    ErrorPayload,
    RequestEnvelope,
    ResponseEnvelope,
    StreamFrame,
)


# ----------------------------------------------------------------------
# ResponseEnvelope
# ----------------------------------------------------------------------


def test_response_envelope_minimal_round_trip() -> None:
    env = ResponseEnvelope(id=42, ok=True, result={"hello": "world"})
    d = env.to_dict()
    assert d == {
        "id": 42,
        "kind": KIND_RESPONSE,
        "ok": True,
        "result": {"hello": "world"},
        "error": None,
        "warnings": [],
        "telemetry": {},
    }
    back = ResponseEnvelope.from_dict(d)
    assert back == env


def test_response_envelope_with_warnings_and_telemetry() -> None:
    env = ResponseEnvelope(
        id=7,
        ok=True,
        result={"stdout": "ok"},
        warnings=["output-truncated", "timeout-near-cap"],
        telemetry={"jaato.cgroup.oom_kill_delta": 1},
    )
    back = ResponseEnvelope.from_dict(env.to_dict())
    assert back.warnings == ["output-truncated", "timeout-near-cap"]
    assert back.telemetry == {"jaato.cgroup.oom_kill_delta": 1}


def test_response_envelope_error_with_traceback() -> None:
    err = ErrorPayload(
        type="ValueError",
        message="bad arg",
        traceback="Traceback...\nValueError: bad arg",
    )
    env = ResponseEnvelope(id=1, ok=False, result=None, error=err)
    d = env.to_dict()
    assert d["error"]["traceback"].startswith("Traceback")
    back = ResponseEnvelope.from_dict(d)
    assert back.error == err


def test_response_envelope_error_without_traceback() -> None:
    """Plugins that return (False, dict) without raising omit traceback."""
    err = ErrorPayload(type="PermissionDenied", message="cannot write /etc")
    env = ResponseEnvelope(id=2, ok=False, result=None, error=err)
    d = env.to_dict()
    assert "traceback" not in d["error"]
    back = ResponseEnvelope.from_dict(d)
    assert back.error and back.error.traceback is None


def test_response_envelope_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError, match="kind != 'response'"):
        ResponseEnvelope.from_dict({"id": 1, "kind": "request", "ok": True})


# ----------------------------------------------------------------------
# RequestEnvelope
# ----------------------------------------------------------------------


def test_request_envelope_round_trip() -> None:
    req = RequestEnvelope(
        id=99,
        method="tool.execute",
        args={"name": "cli_based_tool", "args": {"command": "echo hi"}},
    )
    d = req.to_dict()
    assert d == {
        "id": 99,
        "kind": KIND_REQUEST,
        "method": "tool.execute",
        "args": {"name": "cli_based_tool", "args": {"command": "echo hi"}},
    }
    assert RequestEnvelope.from_dict(d) == req


def test_request_envelope_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError, match="kind != 'request'"):
        RequestEnvelope.from_dict({"id": 1, "kind": "response", "method": "x"})


# ----------------------------------------------------------------------
# StreamFrame
# ----------------------------------------------------------------------


def test_stream_frame_default_channel_is_display() -> None:
    frame = StreamFrame(id=5, source="stdout", text="line 1\n")
    d = frame.to_dict()
    assert d["channel"] == STREAM_CHANNEL_DISPLAY
    assert d["mode"] is None
    assert StreamFrame.from_dict(d) == frame


def test_stream_frame_with_mode() -> None:
    frame = StreamFrame(id=5, source="stderr", text="err", mode="error")
    back = StreamFrame.from_dict(frame.to_dict())
    assert back.mode == "error"
    assert back.source == "stderr"


# ----------------------------------------------------------------------
# CancelFrame
# ----------------------------------------------------------------------


def test_cancel_frame_round_trip() -> None:
    c = CancelFrame(id=42)
    d = c.to_dict()
    assert d == {"id": 42, "kind": KIND_CANCEL}
    assert CancelFrame.from_dict(d) == c


def test_cancel_frame_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError, match="kind != 'cancel'"):
        CancelFrame.from_dict({"id": 1, "kind": "response"})
