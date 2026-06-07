"""Tests for the Path 1 quirk's session-side wiring (server 0.6.195+).

The session detects ``signal_completion`` returning
``{"error": "validation_failed", ...}`` and stamps
``self._pending_tool_choice_name`` so the NEXT
``provider.complete()`` call can request named-function
``tool_choice``.  The provider plugin decides whether to honor it
via its own quirk gate (vllm: ``force_tool_choice_for_lifecycle``);
providers without the quirk silently ignore the kwarg.

These tests exercise the session-side detection + consumption logic
directly without spinning a provider, since the wiring is the
load-bearing contract.  Vllm-side honoring is tested separately in
``shared/plugins/model_provider/vllm/tests/test_force_tool_choice_quirk.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.plugins.model_provider.types import ToolResult


class _FakeSession:
    """Minimal stub exposing just the Path 1 quirk methods.

    Avoids the full JaatoSession construction (which requires a
    runtime, provider, registry, etc.); the methods under test are
    pure logic that only touch ``self._pending_tool_choice_name``
    and ``self._trace``.
    """

    def __init__(self) -> None:
        self._pending_tool_choice_name: Optional[str] = None
        self._traces: List[str] = []

    def _trace(self, msg: str) -> None:
        self._traces.append(msg)

    # Bind the actual JaatoSession methods to this stub.
    from shared.jaato_session import JaatoSession
    _maybe_stamp_lifecycle_retry_tool_choice = (
        JaatoSession._maybe_stamp_lifecycle_retry_tool_choice
    )
    _consume_pending_tool_choice = JaatoSession._consume_pending_tool_choice


def _result(name: str, result: Any) -> ToolResult:
    return ToolResult(call_id=f"id_{name}", name=name, result=result)


# ----------------------------------------------------------------------
# Stamping
# ----------------------------------------------------------------------


class TestMaybeStampLifecycleRetryToolChoice:
    def test_signal_completion_validation_failed_stamps(self) -> None:
        s = _FakeSession()
        results = [_result(
            "signal_completion",
            {
                "error": "validation_failed",
                "message": "Schema mismatch",
                "validation_error": "'14' is not of type 'integer'",
            },
        )]
        s._maybe_stamp_lifecycle_retry_tool_choice(results)
        assert s._pending_tool_choice_name == "signal_completion"
        assert any("PENDING_TOOL_CHOICE" in t for t in s._traces)

    def test_signal_completion_success_does_not_stamp(self) -> None:
        s = _FakeSession()
        results = [_result(
            "signal_completion",
            {"status": "completed", "summary": "ok"},
        )]
        s._maybe_stamp_lifecycle_retry_tool_choice(results)
        assert s._pending_tool_choice_name is None

    def test_other_tool_validation_failed_does_not_stamp(self) -> None:
        """Scoped to signal_completion today — other tools that
        happen to return ``error: validation_failed`` (none today)
        wouldn't stamp.  Locking in the scope so a future broadening
        is an explicit decision."""
        s = _FakeSession()
        results = [_result(
            "some_other_tool",
            {"error": "validation_failed", "message": "..."},
        )]
        s._maybe_stamp_lifecycle_retry_tool_choice(results)
        assert s._pending_tool_choice_name is None

    def test_signal_completion_different_error_does_not_stamp(self) -> None:
        """Only the canonical ``validation_failed`` sentinel triggers
        — other error shapes (provider exceptions, processor errors
        without the sentinel, etc.) are left alone."""
        s = _FakeSession()
        results = [_result(
            "signal_completion",
            {"error": "internal_error", "message": "boom"},
        )]
        s._maybe_stamp_lifecycle_retry_tool_choice(results)
        assert s._pending_tool_choice_name is None

    def test_signal_completion_string_result_does_not_stamp(self) -> None:
        """Non-dict result shapes are ignored — the sentinel lives in
        a dict.  Defensive."""
        s = _FakeSession()
        results = [_result("signal_completion", "some string result")]
        s._maybe_stamp_lifecycle_retry_tool_choice(results)
        assert s._pending_tool_choice_name is None

    def test_empty_results_noop(self) -> None:
        s = _FakeSession()
        s._maybe_stamp_lifecycle_retry_tool_choice([])
        assert s._pending_tool_choice_name is None

    def test_mixed_results_finds_validation_failure(self) -> None:
        """When multiple tools execute in the same batch and at least
        one signal_completion returns validation_failed, stamp."""
        s = _FakeSession()
        results = [
            _result("cli_based_tool", {"stdout": "ok", "exit_code": 0}),
            _result(
                "signal_completion",
                {"error": "validation_failed", "message": "..."},
            ),
        ]
        s._maybe_stamp_lifecycle_retry_tool_choice(results)
        assert s._pending_tool_choice_name == "signal_completion"


# ----------------------------------------------------------------------
# Consumption
# ----------------------------------------------------------------------


class TestConsumePendingToolChoice:
    def test_no_pending_returns_none(self) -> None:
        s = _FakeSession()
        assert s._consume_pending_tool_choice() is None
        assert s._pending_tool_choice_name is None

    def test_pending_returns_openai_wire_shape_and_clears(self) -> None:
        s = _FakeSession()
        s._pending_tool_choice_name = "signal_completion"
        result = s._consume_pending_tool_choice()
        assert result == {
            "type": "function",
            "function": {"name": "signal_completion"},
        }
        # Cleared after one consumption — the contract is single retry.
        assert s._pending_tool_choice_name is None
        # Second call returns None (already consumed).
        assert s._consume_pending_tool_choice() is None

    def test_each_validation_failed_triggers_fresh_consumption(self) -> None:
        """Stamp → consume → stamp → consume should both fire.
        Models the realistic 'validation fails, retry; retry also
        fails, retry again' sequence."""
        s = _FakeSession()
        s._maybe_stamp_lifecycle_retry_tool_choice([_result(
            "signal_completion",
            {"error": "validation_failed", "message": "1st"},
        )])
        first = s._consume_pending_tool_choice()
        assert first is not None

        s._maybe_stamp_lifecycle_retry_tool_choice([_result(
            "signal_completion",
            {"error": "validation_failed", "message": "2nd"},
        )])
        second = s._consume_pending_tool_choice()
        assert second is not None
        assert second == first  # same wire shape
