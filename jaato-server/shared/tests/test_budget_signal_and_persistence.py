"""Budget ceilings must survive a reload and announce themselves in a type.

Two defects, both found by a suspend/resume cascade whose goal actor is driven
by ``session.wake``:

1. **Usage did not survive a session unload.** ``BudgetTracker`` accumulates in
   memory only; there was a read-only ``get_budget_usage`` (for cascade-pool
   reconciliation) and no restore counterpart. Sessions unload on ORPHAN, so a
   driver that disconnects during a wait gets the session evicted -- and every
   resume rebuilt the tracker zeroed. Cross-turn dimensions (``turns``,
   ``seconds``, cumulative ``usd``) therefore never fired; only a ``usd`` limit
   small enough to cross WITHIN one turn ever tripped. Confirmed live against
   a daemon log showing an unload during every wait.

   The shape that matters: the LONGER a goal suspends, the more certainly it is
   evicted -- so the dimensions that bound long goals are exactly the ones that
   reset.

2. **An abort announced itself only in prose.** The reason was latched and
   never surfaced, so "stopped at the ceiling" was indistinguishable from a
   normal finish without substring-matching the output.
"""
import pytest

from shared.budget_control import BudgetControlConfig, BudgetTracker
from shared.jaato_session import JaatoSession


def _config(limits, rungs):
    return BudgetControlConfig.from_dict({"limits": limits, "degrade": rungs})


ABORT_AT_100 = [{"at": "100%", "action": "abort"}]


class TestUsageSurvivesReload:
    def test_a_fresh_tracker_never_reaches_a_cross_turn_ceiling(self):
        """The defect, stated as a test: this is what a reload used to do."""
        cfg = _config({"turns": 2}, ABORT_AT_100)
        first = BudgetTracker(cfg)
        first.observe(turns=1)

        reloaded = BudgetTracker(cfg)          # no restore -- the old behaviour
        assert not reloaded.observe(turns=1), (
            "a zeroed tracker cannot cross a turns ceiling, however many "
            "resumes run"
        )

    def test_restored_usage_reaches_the_ceiling(self):
        cfg = _config({"turns": 2}, ABORT_AT_100)
        first = BudgetTracker(cfg)
        first.observe(turns=1)

        reloaded = BudgetTracker(cfg)
        reloaded.restore_usage(first.usage.as_dict())
        fired = reloaded.observe(turns=1)

        assert [r.at_percent for r in fired] == [100.0]
        assert reloaded.usage.turns == 2.0

    def test_restore_is_absolute_not_a_delta(self):
        cfg = _config({"turns": 10}, ABORT_AT_100)
        t = BudgetTracker(cfg)
        t.observe(turns=3)
        t.restore_usage({"turns": 3})          # same snapshot applied again
        assert t.usage.turns == 3.0, "restore must not accumulate"

    @pytest.mark.parametrize("junk", [
        {"turns": "not-a-number"},
        {"unknown_dimension": 5},
        {},
    ])
    def test_restore_tolerates_a_snapshot_it_cannot_fully_read(self, junk):
        """A budget that refuses to restore is worse than a partial one."""
        cfg = _config({"turns": 2}, ABORT_AT_100)
        t = BudgetTracker(cfg)
        t.observe(turns=1)
        t.restore_usage(junk)                  # must not raise
        assert t.usage.turns == 1.0

    def test_a_crossed_rung_fires_again_after_a_reload(self):
        """Deliberate: firing twice is safer than skipping.

        Rungs are latched per tracker, so a restored tracker re-evaluates the
        ladder against the restored totals. A rebind is idempotent and an
        abort must re-assert, so the safe direction is to fire.
        """
        cfg = _config({"turns": 1}, ABORT_AT_100)
        first = BudgetTracker(cfg)
        assert [r.at_percent for r in first.observe(turns=1)] == [100.0]

        reloaded = BudgetTracker(cfg)
        reloaded.restore_usage(first.usage.as_dict())
        assert [r.at_percent for r in reloaded.observe(turns=0)] == [100.0]


class TestSessionLevelRestore:
    def test_restore_seeds_the_tracker(self):
        s = JaatoSession.__new__(JaatoSession)
        s._budget_tracker = BudgetTracker(_config({"turns": 2}, ABORT_AT_100))
        JaatoSession.restore_budget_usage(s, {"turns": 1})
        assert s._budget_tracker.usage.turns == 1.0

    @pytest.mark.parametrize("usage", [None, {}])
    def test_restore_is_a_noop_without_a_snapshot(self, usage):
        s = JaatoSession.__new__(JaatoSession)
        s._budget_tracker = BudgetTracker(_config({"turns": 2}, ABORT_AT_100))
        JaatoSession.restore_budget_usage(s, usage)
        assert s._budget_tracker.usage.turns == 0.0

    def test_restore_is_a_noop_for_an_unbudgeted_session(self):
        s = JaatoSession.__new__(JaatoSession)
        s._budget_tracker = None
        JaatoSession.restore_budget_usage(s, {"turns": 5})   # must not raise


class TestTypedExhaustionSignal:
    def test_the_reason_is_readable_as_a_value(self):
        s = JaatoSession.__new__(JaatoSession)
        s._budget_exhausted_reason = "budget_exhausted (self-enforced: turns 100%)"
        assert JaatoSession.budget_exhausted_reason(s) == s._budget_exhausted_reason

    def test_none_when_the_ceiling_was_never_hit(self):
        s = JaatoSession.__new__(JaatoSession)
        s._budget_exhausted_reason = None
        assert JaatoSession.budget_exhausted_reason(s) is None

    def test_the_send_rpc_carries_it(self):
        """The signal must reach the wire, not just exist on the session."""
        import inspect
        from server.runner.rpc import RunnerRPC

        src = inspect.getsource(RunnerRPC._handle_session_send_message)
        assert "budget_exhausted" in src, (
            "session.send_message does not surface the exhaustion reason -- a "
            "driver is left substring-matching prose"
        )
        assert "getattr(session, \"budget_exhausted_reason\", None)" in src, (
            "the accessor must be looked up with a default, not wrapped in a "
            "bare except that would hide a rename and emit no signal at all"
        )
