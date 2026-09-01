"""Which completion flags are per-TURN, and which is per-SESSION.

Two latches on ``JaatoSession`` looked session-lifetime but every reader asks
a per-turn question -- the turn terminator, the quiescence hook (whose own
comment says "called signal_completion DURING THIS TURN"), the
signal_completion idempotency guard ("in the same tool batch"), and the
auto-finalize synthesizer.

Nothing reset them, which is invisible for a ONE-SHOT session: turn 0 is the
only turn. On a SUSPEND/RESUME session the agent calls ``signal_completion``
every turn (``outcome=suspended`` ends the turn; a driver wakes the same
session later), so ``_signal_completion_called`` latched on turn 0 and
``_execute_tools_and_continue`` TRUNCATED every later turn at its first tool
batch -- the model was cut off before it could reach its own exit. Reported as
"the agent forgets its exit"; it never got there.

``_completion_nudges_fired`` IS NOT ONE OF THEM, and this suite used to say it
was -- it asserted that a turn start hands the budget back, on the reasoning
that a per-session budget leaves a long-lived session with "the safety net
gone". The reasoning inverts what the net does. Spending the budget means the
framework STOPS asking; it is the ceiling, not the catch. And a nudge
RE-PROMPTS the session, so the reset ran on the very turn the nudge created
and returned the token it had just spent:

  * the top-level guard in ``core.py``'s model_thread re-armed every turn --
    observed as "nudge 1/2" logged three times in one session, and measured
    against a live daemon as 735 turns in 40 seconds for a session that never
    signals (jaato #767);
  * the subagent guard's ``while ... < MAX_COMPLETION_NUDGES`` in
    ``subagent/plugin.py``, written on the assumption that the counter only
    goes up, could not terminate at all.

``max_turns``, ``budget_control`` and the caller's own wall-clock were what
actually stopped those sessions, at whatever they had spent by then. So the
budget is per SESSION, which is what ``MAX_COMPLETION_NUDGES`` always claimed
to be, and the "long-lived session" the old reasoning worried about is one
that no longer exists: spending the budget terminates the session
(``NudgeExhausted``).
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
    def test_both_per_turn_latches_are_cleared(self):
        s = _session(_signal_completion_called=True,
                     _session_quiescent_emitted=True)
        JaatoSession._begin_turn_completion_state(s)
        assert s._signal_completion_called is False
        assert s._session_quiescent_emitted is False

    def test_the_nudge_budget_survives_a_turn_start(self):
        """The one flag a turn start must NOT touch.

        A nudge re-prompts the session, so a turn start that cleared this
        would hand back the token the nudge had just spent -- and did.
        """
        s = _session(_completion_nudges_fired=2)
        JaatoSession._begin_turn_completion_state(s)
        assert s._completion_nudges_fired == 2, (
            "the turn a nudge created refunded the nudge's own budget"
        )

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
    def test_the_nudge_loop_terminates(self):
        """The shape both guards actually run, not the predicate alone.

        A nudge re-prompts, and the re-prompt is a TURN -- so the budget must
        be read across the turn the nudge itself causes. Asserting the
        predicate in isolation is what let this ship: the counter was bounded
        within one turn and unbounded across the only sequence that exists.
        """
        s = _session()
        fired = 0
        for _ in range(50):                      # a ceiling, not an expectation
            should_nudge, _count = JaatoSession.try_completion_nudge(
                s, max_nudges=2)
            if not should_nudge:
                break
            fired += 1
            # The nudge's own turn begins.  This is the step the old reset
            # turned into a refund.
            JaatoSession._begin_turn_completion_state(s)
            s._signal_completion_called = False  # the model still doesn't signal
        else:
            pytest.fail("the completion-nudge loop never terminates")
        assert fired == 2, f"MAX_COMPLETION_NUDGES=2 allowed {fired} nudges"

    def test_budget_is_bounded_without_a_turn_in_between(self):
        s = _session()
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (True, 1)
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (True, 2)
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (False, 2)

    def test_a_turn_that_completed_is_not_nudged(self):
        s = _session(_signal_completion_called=True)
        assert JaatoSession.try_completion_nudge(s, max_nudges=2) == (False, 0)
