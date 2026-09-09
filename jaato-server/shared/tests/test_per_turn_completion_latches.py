"""Which completion flags a turn start clears, and on which turns.

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

``_completion_nudges_fired`` is the third, and it is the one whose answer is
"it depends who started this turn".

An UNCONDITIONAL reset hands the budget back, and a nudge RE-PROMPTS the
session -- so the reset ran on the very turn the nudge created and returned
the token it had just spent:

  * the top-level guard in ``core.py``'s model_thread re-armed every turn --
    observed as "nudge 1/2" logged three times in one session, and measured
    against a live daemon as 735 turns in 40 seconds for a session that never
    signals (jaato #767);
  * the subagent guard's ``while ... < MAX_COMPLETION_NUDGES`` in
    ``subagent/plugin.py``, written on the assumption that the counter only
    goes up, could not terminate at all.

NEVER resetting fixed that, and this suite said so. It rested on one clause --
"a completion-gated session is one-shot by construction", so a session-lifetime
budget costs nothing -- and #913 / #915 made that clause false: recording
``signal_completion``'s tool result is what lets a completed session be driven
again, and #845 / #914 lets the next turn arrive with an attachment. A bound
written to stop a runaway retry loop INSIDE ONE TURN had become a ceiling on
how many turns a conversation may have: measured on a voice agent that
announces ``signal_completion`` rather than invoking it, the first
``max_completion_nudges`` turns closed and every turn after died
``NudgeExhausted`` for the life of the session (jaato #934).

So the reset asks WHO STARTED THE TURN. ``try_completion_nudge`` latches
``_completion_nudge_turn_pending`` when it spends the budget; the turn that
nudge creates consumes the latch and KEEPS the counter (the loop still
terminates, exactly as #767 requires), and any other turn -- a user message, a
``session.wake``, a parent's ``send_to_subagent`` -- starts fresh. Both
properties are asserted below, and neither is safe to assert alone: the
predicate in isolation is what let #767 ship, and the loop in isolation is what
let #934 ship.
"""
from unittest.mock import MagicMock, patch

import pytest

from shared.jaato_session import JaatoSession


def _session(**state):
    s = JaatoSession.__new__(JaatoSession)
    s._signal_completion_called = False
    s._completion_nudges_fired = 0
    s._completion_nudge_turn_pending = False
    s._session_quiescent_emitted = False
    s._trace = lambda *a, **k: None
    for k, v in state.items():
        setattr(s, k, v)
    return s


def _turn_terminates(session) -> bool:
    """Drive the continuation decision only."""
    session._executor = None
    session._provider = MagicMock()
    # The terminal path writes the batch's results into history (#913);
    # this harness drives the decision only, so it is stubbed out.
    session._record_terminal_tool_results = MagicMock()
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

    def test_the_nudge_budget_survives_the_turn_a_nudge_created(self):
        """The case a turn start must NOT refund (#767).

        A nudge re-prompts the session, so a turn start that cleared the
        counter here hands back the token the nudge had just spent -- and did,
        which is why no nudge loop terminated.
        """
        s = _session(_completion_nudges_fired=2,
                     _completion_nudge_turn_pending=True)
        JaatoSession._begin_turn_completion_state(s)
        assert s._completion_nudges_fired == 2, (
            "the turn a nudge created refunded the nudge's own budget"
        )

    def test_the_nudge_that_marked_the_turn_is_consumed(self):
        """The latch is one turn's, not the session's.

        Left standing it would suppress the reset on the turn AFTER the nudge
        sequence too, which is the ceiling #934 is about.
        """
        s = _session(_completion_nudges_fired=2,
                     _completion_nudge_turn_pending=True)
        JaatoSession._begin_turn_completion_state(s)
        assert s._completion_nudge_turn_pending is False
        JaatoSession._begin_turn_completion_state(s)
        assert s._completion_nudges_fired == 0

    def test_a_caller_started_turn_gets_a_fresh_budget(self):
        """THE #934 regression, at the reset itself.

        A turn nobody nudged into existence -- a user message, a
        ``session.wake``, a parent's ``send_to_subagent`` -- is a new attempt,
        not a continuation of the last one's retries.
        """
        s = _session(_completion_nudges_fired=4)
        JaatoSession._begin_turn_completion_state(s)
        assert s._completion_nudges_fired == 0, (
            "a conversation was rationed by an earlier turn's retries"
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

    def test_spending_the_budget_marks_the_turn_the_nudge_will_start(self):
        """The latch is set by the same call that spends the token.

        Any nudge site that increments the counter itself instead of going
        through this method leaves its re-prompt looking caller-originated --
        the reset refills the budget and the loop unbounds again (#767).
        """
        s = _session()
        JaatoSession.try_completion_nudge(s, max_nudges=2)
        assert s._completion_nudge_turn_pending is True

    def test_a_refused_nudge_marks_nothing(self):
        """A session that signalled is about to run a caller's next turn, not
        a nudge's, so nothing may suppress that turn's reset."""
        s = _session(_signal_completion_called=True)
        JaatoSession.try_completion_nudge(s, max_nudges=2)
        assert s._completion_nudge_turn_pending is False


class TestAConversationIsNotRationed:
    """THE #934 shape: a completion-gated session driven for several turns.

    #913 / #915 made this session survive its own completion, so the sequence
    below is now something a caller does -- and a budget spent per turn but
    refilled never gave turn ``max_nudges + 1`` nothing to spend. Measured on a
    voice agent: turns 1-4 closed, turns 5+ died ``NudgeExhausted`` forever.
    """

    @staticmethod
    def _turn(session, max_nudges, signals_after_nudge=True):
        """One caller-started turn plus the nudges it needs. True if it closed."""
        JaatoSession._begin_turn_completion_state(session)   # the caller's turn
        for _ in range(max_nudges + 5):                      # a ceiling, not an expectation
            should_nudge, _n = JaatoSession.try_completion_nudge(
                session, max_nudges=max_nudges)
            if not should_nudge:
                return False
            # The nudge's own turn.
            JaatoSession._begin_turn_completion_state(session)
            if signals_after_nudge:
                session._signal_completion_called = True
                return True
        raise AssertionError("the completion-nudge loop never terminates")

    def test_every_turn_of_a_conversation_can_close(self):
        s = _session()
        for turn in range(1, 7):
            assert self._turn(s, max_nudges=2) is True, (
                f"turn {turn} could not be nudged into completing -- the "
                f"budget was spent by an earlier turn"
            )

    def test_a_turn_that_never_signals_still_stops_being_nudged(self):
        """#767's guarantee, unchanged: the loop inside ONE turn terminates."""
        s = _session()
        assert self._turn(s, max_nudges=2, signals_after_nudge=False) is False
        assert s._completion_nudges_fired == 2

    def test_and_the_turn_after_that_one_starts_fresh(self):
        """Exhaustion is a verdict on the turn that failed, not on the session.

        The daemon does terminate a session that exhausts its budget; what
        must not happen is a LATER turn inheriting the spent counter and
        failing before the model is asked anything.
        """
        s = _session()
        self._turn(s, max_nudges=2, signals_after_nudge=False)
        assert self._turn(s, max_nudges=2) is True
