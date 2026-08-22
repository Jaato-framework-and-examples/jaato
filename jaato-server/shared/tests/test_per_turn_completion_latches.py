"""Completion flags are per-TURN, not per-session.

Three flags on ``JaatoSession`` looked session-lifetime but every reader asks
a per-turn question -- the turn terminator, the nudge predicate and its
budget, the quiescence hook (whose own comment says "called signal_completion
DURING THIS TURN"), the signal_completion idempotency guard ("in the same tool
batch"), the auto-finalize synthesizer, the subagent nudge loop, and the
embedded nudge gate.

Nothing reset them, which is invisible for a ONE-SHOT session: turn 0 is the
only turn. On a SUSPEND/RESUME session the agent calls ``signal_completion``
every turn (``outcome=suspended`` ends the turn; a driver wakes the same
session later), so:

  * ``_signal_completion_called`` latched on turn 0 and
    ``_execute_tools_and_continue`` TRUNCATED every later turn at its first
    tool batch -- the model was cut off before it could reach its own exit.
  * ``_completion_nudges_fired`` spent the nudge budget once per SESSION, so
    the safety net was gone exactly when the session became long-lived.

Reported as "the agent forgets its exit"; it never got there.
"""
from unittest.mock import MagicMock, patch

import pytest

from shared.jaato_session import JaatoSession


def _session(**state):
    s = JaatoSession.__new__(JaatoSession)
    s._signal_completion_called = False
    s._completion_nudges_fired = 0
    s._session_quiescent_emitted = False
    s._trace = lambda *a, **k: None
    for k, v in state.items():
        setattr(s, k, v)
    return s


def _turn_terminates(session) -> bool:
    """Drive the continuation decision only."""
    session._executor = None
    session._provider = MagicMock()
    with patch.object(JaatoSession, "_execute_function_call_group", return_value=[]):
        try:
            _, result, _ = JaatoSession._execute_tools_and_continue(
                session, fc_group=[], use_streaming=False, on_output=None,
                wrapped_usage_callback=None, turn_data={},
                cancellation_notified=False, accumulated_text=["partial"],
                context="", check_mid_turn=False,
            )
            return result is not None
        except AttributeError:
            # proceeded past the guard into state this harness doesn't build
            return False


class TestTurnStartClearsThem:
    def test_all_three_are_cleared(self):
        s = _session(_signal_completion_called=True, _completion_nudges_fired=2,
                     _session_quiescent_emitted=True)
        JaatoSession._begin_turn_completion_state(s)
        assert s._signal_completion_called is False
        assert s._completion_nudges_fired == 0
        assert s._session_quiescent_emitted is False

    def test_both_chat_loops_call_it(self):
        """Neither entry path may miss the reset.

        send_message can delegate to send_message_with_parts, so the reset
        lives in the loops -- exactly once per turn on either path.
        """
        import inspect
        for loop in (JaatoSession._run_chat_loop,
                     JaatoSession._run_chat_loop_with_parts):
            assert "_begin_turn_completion_state" in inspect.getsource(loop), (
                f"{loop.__name__} does not reset the per-turn completion state"
            )


class TestTruncation:
    def test_a_turn_that_completed_still_terminates(self):
        """The terminator itself is correct and must not change."""
        assert _turn_terminates(_session(_signal_completion_called=True)) is True

    def test_a_resumed_turn_is_not_truncated(self):
        """THE regression: a stale latch used to end the turn immediately."""
        stale = _session(_signal_completion_called=True)
        JaatoSession._begin_turn_completion_state(stale)
        assert _turn_terminates(stale) is False, (
            "a resumed turn was terminated at its first tool batch by a latch "
            "left over from a previous turn"
        )


class TestNudgeBudget:
    def test_nudge_is_reachable_again_after_a_completed_turn(self):
        s = _session(_signal_completion_called=True, _completion_nudges_fired=2)
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (False, 2)

        JaatoSession._begin_turn_completion_state(s)
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (True, 1)

    def test_budget_is_still_bounded_within_a_turn(self):
        """Per-turn must not mean unlimited."""
        s = _session()
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (True, 1)
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (True, 2)
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (False, 2)

    def test_a_turn_that_completed_is_not_nudged(self):
        s = _session(_signal_completion_called=True)
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (False, 0)
