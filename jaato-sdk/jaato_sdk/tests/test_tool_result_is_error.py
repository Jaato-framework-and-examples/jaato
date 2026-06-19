"""Tests for the ``tool_result_is_error`` deeper-error helper and its
propagation onto ``ToolCallEndEvent`` / ``ToolCallCompletedPayload``.

``tool_result_is_error`` is the canonical "success=True but error body"
check (e.g. a tool that returns ``{"error": ...}`` or HTTP
``status_code >= 400`` while the executor reports success).  It is reused by
the reliability plugin and by the ``tool.call_completed`` event populate.
"""

import pytest

from jaato_sdk.plugins.model_provider.types import (
    ToolResult,
    tool_result_is_error,
)
from jaato_sdk.events import ToolCallEndEvent
from jaato_sdk.event_payloads import ToolCallCompletedPayload


@pytest.mark.parametrize(
    "result,expected",
    [
        # Non-dict inputs are never errors.
        ("just a string", False),
        (None, False),
        (["a", "list"], False),
        (42, False),
        # Plain success dicts.
        ({"ok": 1}, False),
        ({"result": "...", "status_code": 201}, False),
        # Error-body dicts.
        ({"error": "boom"}, True),
        ({"status_code": 500}, True),
        ({"status_code": 200}, False),
        ({"status_code": 404}, True),
        ({"status_code": 400}, True),
        ({"status_code": 399}, False),
    ],
)
def test_tool_result_is_error(result, expected):
    assert tool_result_is_error(result) is expected


def test_empty_dict_is_not_error():
    # No "error" key and default status_code 200 -> not an error.
    assert tool_result_is_error({}) is False


def test_tool_call_end_event_roundtrips_is_error_result():
    event = ToolCallEndEvent(
        agent_id="a1",
        tool_name="callService",
        call_id="c1",
        success=True,
        is_error_result=True,
        duration_seconds=0.5,
    )
    assert event.is_error_result is True
    # Round-trips through the JSON serialization used by the bus republish.
    dumped = event.to_dict()
    assert dumped["is_error_result"] is True
    assert dumped["success"] is True


def test_tool_call_end_event_defaults_is_error_result_false():
    event = ToolCallEndEvent(agent_id="a1", tool_name="t", success=True)
    assert event.is_error_result is False


def test_tool_call_completed_payload_includes_is_error_result():
    # TypedDict membership: the bus payload the reactor's merged view reads
    # must carry the key.
    assert "is_error_result" in ToolCallCompletedPayload.__annotations__


def test_tool_result_is_error_distinct_from_tool_result_is_error_flag():
    # ToolResult.is_error == not success (shallow); the helper inspects the
    # body.  A success=True call with an error body is is_error=False but
    # tool_result_is_error(...)=True.
    tr = ToolResult(call_id="c", name="t", result={"error": "x"}, is_error=False)
    assert tr.is_error is False
    assert tool_result_is_error(tr.result) is True
