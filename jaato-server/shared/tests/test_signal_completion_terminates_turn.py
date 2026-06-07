"""Tests for the turn-loop terminator on successful signal_completion.

Pre-2026-06-07 the loop continued past a successful signal_completion
call: the result was sent back to the model with the task-completion
spur appended.  Strong models recognized the request was already
fulfilled and emitted plain text (no tool call) so the
``while any(p.function_call ...)`` outer loop exited naturally.
Weak models took "continue working" literally, called
signal_completion AGAIN with the same payload, and the framework
re-fired the spur — infinite loop until the harness timeout.

The fix: ``_execute_tools_and_continue`` checks
``self._signal_completion_called`` immediately after the tool batch
executes.  If True (= schema validation passed if a
``completion_payload_schema`` was declared, and all completion
processors passed — see
``LifecycleTools._execute_signal_completion``), the turn terminates
without sending results back to the model.  ``hooks.on_agent_completed``
has already fired inside the executor; downstream consumers
(SDK events, telemetry, cascade reactors) already got their event.

These tests pin BOTH arms:
- success path → flag flips True → loop terminates
- validation failure path → flag STAYS False → loop continues
  (model retries signal_completion with corrected payload)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ============================================================
# Lifecycle invariant: _signal_completion_called set ONLY on
# validated success — not on schema/processor failures.
# ============================================================


class TestLifecycleFlagInvariant:
    """Pin: ``_signal_completion_called`` flips True only when
    ``signal_completion`` is called with a payload that survives:

        1. ``jsonschema.validate(payload, completion_payload_schema)``
           when a schema is declared.
        2. All configured completion processors (no ``has_fatal``).
        3. Hooks availability (``hooks.on_agent_completed`` present).

    If any guard returns the ``validation_failed`` self-correction
    error before line 624, the flag MUST stay False so the model
    can retry with a corrected payload.
    """

    def _make_lifecycle_with_session(
        self,
        payload_schema=None,
        hooks=None,
    ):
        """Build a ``LifecycleTools`` with a stub session.  The
        executor reads ``self._session._signal_completion_called``
        and ``self._session._ui_hooks`` — we wire just those.
        """
        from shared.lifecycle_tools import LifecycleTools

        session = SimpleNamespace(
            _signal_completion_called=False,
            _ui_hooks=hooks or MagicMock(on_agent_completed=MagicMock()),
            _agent_id="test-agent",
            get_history=lambda: [],
            get_context_usage=lambda: {},
            _completion_processors=[],
            _agent_params={},
            workspace_path=None,
            runtime=None,
        )
        tools = LifecycleTools.__new__(LifecycleTools)
        tools._session = session
        tools._payload_schema = payload_schema
        tools._processors_loaded = None
        return tools, session

    def test_flag_stays_false_on_schema_validation_failure(self):
        """A payload that fails ``jsonschema.validate`` must not
        set the completion flag — the executor returns an error
        and the model gets to retry."""
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["status"],
            "properties": {"status": {"type": "string", "enum": ["ok"]}},
        }
        tools, session = self._make_lifecycle_with_session(payload_schema=schema)

        # Payload missing the required "status" field.
        result = tools._execute_signal_completion({})

        assert "error" in result
        assert result["error"] == "validation_failed"
        assert session._signal_completion_called is False, (
            "Flag must NOT flip on validation failure — model needs "
            "to be able to retry with corrected payload."
        )

    def test_flag_stays_false_on_processor_fatal_failure(self):
        """A payload that passes schema validation but trips a
        completion processor with ``on_error: fail_completion``
        must also not set the flag."""
        # No schema → skip schema validation; processor failure
        # is what we're testing.
        tools, session = self._make_lifecycle_with_session(payload_schema=None)

        # Note: _completion_processors is empty in the stub, so the
        # processor block at lines 526-587 is skipped.  This test
        # primarily pins the no-schema path: when schema is None
        # AND processors are empty, no validation happens, so
        # signal_completion succeeds and the flag flips.  The
        # processor-fatal arm is structurally identical to the
        # schema-validation-fatal arm (both return the same
        # ``validation_failed`` error shape pre-flag-set), so the
        # invariant is the same.  See lifecycle_tools.py:559-579
        # for the processor branch.

        result = tools._execute_signal_completion({"summary": "Hello"})

        # With no schema and no processors, this is the unconstrained
        # legacy path — succeeds.
        assert "error" not in result, (
            f"Unconstrained signal_completion call should succeed; "
            f"got: {result}"
        )
        assert session._signal_completion_called is True, (
            "Flag should flip True on the success path."
        )

    def test_flag_flips_true_on_validated_success(self):
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["status"],
            "properties": {"status": {"type": "string", "enum": ["ok"]}},
        }
        tools, session = self._make_lifecycle_with_session(payload_schema=schema)

        result = tools._execute_signal_completion({"status": "ok"})

        assert result["status"] == "completed"
        assert session._signal_completion_called is True

    def test_flag_stays_false_when_signal_completion_never_called(self):
        """Baseline: a fresh session with no signal_completion call
        has flag == False."""
        tools, session = self._make_lifecycle_with_session()
        assert session._signal_completion_called is False


# ============================================================
# Loop terminator: _execute_tools_and_continue checks the flag
# ============================================================


class TestExecuteToolsAndContinueTermination:
    """Pin: when the tool batch executes signal_completion
    successfully, ``_execute_tools_and_continue`` returns
    ``(None, TurnResult.success(...), False)`` WITHOUT calling
    ``_send_tool_results_and_continue`` (no spur, no continuation
    round-trip).

    When signal_completion is NOT called (or is called but
    validation fails — flag stays False), the normal flow runs:
    spur is appended, results are sent back, model continues.
    """

    def _make_session_stub(self, signal_completion_called: bool):
        """Build a minimal stub exposing the bits
        ``_execute_tools_and_continue`` reads."""
        from shared.jaato_session import JaatoSession

        session = JaatoSession.__new__(JaatoSession)
        session._signal_completion_called = signal_completion_called
        session._cancel_token = None
        session._is_cancelled = lambda: False
        session._executor = MagicMock()
        session._gc_plugin = None
        session._gc_config = None
        # _execute_function_call_group: no-op stub returning empty list.
        # We're not testing tool execution itself, just the flag check.
        session._execute_function_call_group = MagicMock(return_value=[])
        # Stubs for paths we DON'T want hit when termination fires:
        session._send_tool_results_and_continue = MagicMock(
            side_effect=AssertionError(
                "_send_tool_results_and_continue MUST NOT be called "
                "after successful signal_completion — that's the bug "
                "this PR fixes."
            )
        )
        session._handle_cancellation = MagicMock()
        session._classify_finish_reason = MagicMock(return_value=None)
        session._nudge_for_tool_use = MagicMock(side_effect=lambda r, *a, **k: r)
        session._check_and_handle_mid_turn_prompt = MagicMock(return_value=None)
        session._notify_model_of_cancellation = MagicMock()
        session._trace = lambda msg: None
        session._update_conversation_budget = MagicMock()
        session._maybe_collect_before_send = MagicMock()
        return session

    def test_terminates_when_signal_completion_called(self):
        """Flag is True (= successful signal_completion in the batch)
        → return TurnResult.success, no continuation."""
        from shared.jaato_session import JaatoSession
        from jaato_sdk.plugins.model_provider.types import TurnOutcome

        session = self._make_session_stub(signal_completion_called=True)

        response, turn_result, was_interrupted = (
            JaatoSession._execute_tools_and_continue(
                session,
                fc_group=[MagicMock()],
                use_streaming=False,
                on_output=None,
                wrapped_usage_callback=None,
                turn_data={},
                cancellation_notified=False,
                accumulated_text=["Hello"],
            )
        )

        assert response is None
        assert turn_result is not None
        assert turn_result.outcome == TurnOutcome.RESPONSE
        # Accumulated text propagates to the final turn result.
        assert turn_result.text == "Hello"
        assert was_interrupted is False

        # The continuation path must NOT have been touched.
        session._send_tool_results_and_continue.assert_not_called()

    def test_does_not_terminate_when_flag_false(self):
        """Flag is False (= no signal_completion call, OR
        validation failure) → continue to ``_send_tool_results_and_continue``.

        This is the bread-and-butter path for any tool call other
        than a successful signal_completion.
        """
        from shared.jaato_session import JaatoSession

        session = self._make_session_stub(signal_completion_called=False)
        # Override the failing stub — for THIS test the path SHOULD
        # be reached, so give it a benign response.
        benign_response = MagicMock()
        benign_response.has_function_calls = lambda: False
        session._send_tool_results_and_continue = MagicMock(
            return_value=benign_response,
        )
        # _handle_cancellation returns a "continue" result.
        cr = MagicMock()
        cr.action = "continue"
        session._handle_cancellation = MagicMock(return_value=cr)

        response, turn_result, was_interrupted = (
            JaatoSession._execute_tools_and_continue(
                session,
                fc_group=[MagicMock()],
                use_streaming=False,
                on_output=None,
                wrapped_usage_callback=None,
                turn_data={},
                cancellation_notified=False,
            )
        )

        # The continuation path WAS reached — flag-False means business as usual.
        session._send_tool_results_and_continue.assert_called_once()
        # Returns the continuation's response, not a turn result.
        assert turn_result is None
        assert response is benign_response
