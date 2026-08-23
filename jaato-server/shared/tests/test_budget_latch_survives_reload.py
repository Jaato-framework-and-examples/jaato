"""A ceiling that fired before a suspend still refuses after the resume.

#581 made USAGE survive an eviction. The ENFORCEMENT did not: an abort rung
latches ``_budget_exhausted_reason``, that latch was never persisted, and
``_refuse_if_budget_exhausted`` is what turns a crossed ceiling into a refused
turn.

The mechanism, reproduced before writing this:

    reloaded session, usage restored at the ceiling (turns 2.0 / 2)
      refuses at turn START?      None      <- latch gone
      -> the next turn RUNS
      after that turn ends         'budget_exhausted (... turns 150%)'
      refuses the turn AFTER that  True

So the abort DOES re-assert -- ``observe()`` re-fires it, exactly as
``restore_usage`` intends -- but in the turn's ``finally``, one turn too late.
Every reload bought a free turn, and a goal that finished inside it sailed
through a ceiling that had already aborted. Reported after a live run where the
goal completed EXIT 0 with ``refusing turn`` count 0 for the whole run.

Written BEFORE the fix, so it fails for the right reason first.
"""
from datetime import datetime, timezone

import pytest

from shared.budget_control import BudgetControlConfig, BudgetTracker
from shared.jaato_session import JaatoSession
from shared.plugins.session.base import SessionState
from shared.plugins.session.serializer import (
    deserialize_session_state,
    serialize_session_state,
)

CFG = {"limits": {"turns": 2}, "degrade": [{"at": "100%", "action": "abort"}]}
REASON = "budget_exhausted (self-enforced: turns 100%)"


def _tracker():
    return BudgetTracker(BudgetControlConfig.from_dict(CFG))


def _session(usage=None, reason=None):
    s = JaatoSession.__new__(JaatoSession)
    s._budget_tracker = _tracker()
    s._budget_exhausted_reason = None
    s._budget_terminal_action = None
    s._budget_applied_rung_pct = -1.0
    s._budget_notice_sink = []
    s._tier_config = None
    s._turn_accounting = []
    s._trace = lambda *a, **k: None
    s.request_stop = lambda reason="": None
    s._surface_budget_event = lambda m: None
    if usage:
        s._budget_tracker.restore_usage(usage)
    if reason:
        JaatoSession.restore_budget_exhausted(s, reason)
    return s


def _state(**kw):
    now = datetime.now(timezone.utc)
    return SessionState(session_id="s1", history=[], created_at=now,
                        updated_at=now, **kw)


class TestTheLatchRoundTrips:
    def test_it_reaches_the_persisted_form(self):
        payload = serialize_session_state(_state(budget_exhausted_reason=REASON))
        assert "budget_exhausted_reason" in payload, (
            "the serializer writes a fixed key list -- a field added to "
            "SessionState alone never reaches disk (see #581)"
        )
        assert deserialize_session_state(payload).budget_exhausted_reason == REASON

    def test_absent_key_still_loads(self):
        """Sessions persisted before this field existed."""
        payload = serialize_session_state(_state())
        payload.pop("budget_exhausted_reason", None)
        assert deserialize_session_state(payload).budget_exhausted_reason is None


class TestEnforcementSurvives:
    def test_a_reloaded_session_refuses_at_turn_start(self):
        """THE regression: it used to serve one more turn first."""
        reloaded = _session(usage={"turns": 2.0}, reason=REASON)
        assert JaatoSession._refuse_if_budget_exhausted(reloaded) is not None, (
            "a session reloaded at its ceiling served another turn -- the "
            "re-assert lands in the turn's finally, one turn too late"
        )

    def test_a_session_below_the_ceiling_is_not_refused(self):
        """The guard must not refuse a session that was never stopped."""
        reloaded = _session(usage={"turns": 1.0})
        assert JaatoSession._refuse_if_budget_exhausted(reloaded) is None

    def test_restore_is_a_noop_for_an_unbudgeted_session(self):
        s = JaatoSession.__new__(JaatoSession)
        s._budget_tracker = None
        s._budget_exhausted_reason = None
        JaatoSession.restore_budget_exhausted(s, REASON)   # must not raise

    def test_the_whole_chain(self):
        """Abort before the suspend -> disk -> reload -> refused at start."""
        live = _session()
        # Apply the rungs the CROSSING observe returned -- a second observe
        # returns nothing new (they are latched per tracker), so passing that
        # one applies nothing.
        fired = live._budget_tracker.observe(turns=2)      # crosses 100%
        JaatoSession._apply_budget_rungs(live, fired)
        assert live._budget_exhausted_reason, "precondition: the abort fired"

        payload = serialize_session_state(_state(
            budget_usage=JaatoSession.get_budget_usage(live),
            budget_exhausted_reason=JaatoSession.budget_exhausted_reason(live),
        ))
        state = deserialize_session_state(payload)

        reloaded = _session(usage=state.budget_usage,
                            reason=state.budget_exhausted_reason)
        assert JaatoSession._refuse_if_budget_exhausted(reloaded) is not None
