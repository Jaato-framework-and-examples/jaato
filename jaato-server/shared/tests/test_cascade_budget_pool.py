"""CascadeBudgetPool — the AGGREGATE ceiling for one cascade (design note §8/b).

The linear-chain case is the demo: a later stage is degraded by what earlier
stages already spent, with the stage's own profile never having asked for it.
"""

import threading

import pytest

from shared.budget_control import (
    BudgetControlConfig,
    CascadeBudgetPool,
)


def _cfg(**kw):
    return BudgetControlConfig.from_dict(kw)


def _pool(**limits):
    return CascadeBudgetPool("cid1", _cfg(limits=limits))


def test_remaining_starts_at_the_declared_cap():
    assert _pool(usd=10, turns=12).remaining() == {"usd": 10.0, "turns": 12.0}


def test_spend_depletes_the_pool():
    p = _pool(usd=10)
    p.spend(usd=7)
    assert p.remaining() == {"usd": 3.0}


def test_remaining_never_goes_negative():
    p = _pool(usd=10)
    p.spend(usd=25)
    assert p.remaining() == {"usd": 0.0}


def test_linear_chain_later_stage_is_clamped_by_earlier_spend():
    """THE demo: stage 3 runs on what stages 1-2 left, not on what it asked
    for, and its own profile never mentioned the cascade."""
    p = _pool(usd=10, turns=12)
    p.spend(usd=7, turns=8)                       # stages 1 + 2
    cfg, eff = p.child_config(_cfg(limits={"usd": 9, "turns": 2}))
    assert cfg.limits["usd"] == 3.0               # clamped by the cascade
    assert cfg.limits["turns"] == 2               # its own is tighter, kept
    assert eff.clamped == ("usd",)
    # both inputs observable, not just the outcome
    assert eff.profile_limits["usd"] == 9
    assert eff.cascade_remaining["usd"] == 3.0
    assert "CLAMPED" in eff.describe()


def test_child_keeps_its_own_degrade_ladder():
    """The cascade constrains CEILINGS, not a child's degradation policy."""
    p = _pool(usd=10)
    child = _cfg(limits={"usd": 4}, degrade=[{"at": 50, "action": "abort"}])
    cfg, _ = p.child_config(child)
    assert [r.action for r in cfg.degrade] == ["abort"]


def test_child_with_no_budget_still_inherits_the_cascade_ceiling():
    p = _pool(usd=10)
    p.spend(usd=6)
    cfg, eff = p.child_config(None)
    assert cfg.limits == {"usd": 4.0}
    assert eff.clamped == ("usd",)


def test_child_dimension_the_cascade_does_not_cap_is_untouched():
    p = _pool(usd=10)
    cfg, eff = p.child_config(_cfg(limits={"turns": 5}))
    assert cfg.limits["turns"] == 5
    assert "turns" not in eff.clamped


def test_exhausted_pool_refuses_the_child_rather_than_handing_it_zero():
    """Zero is not a budget a session can run under, and returning None
    ("unbudgeted") would be catastrophic — the reason there is no budget is
    that the cascade is OUT of budget. Fail loud so the caller refuses."""
    from shared.budget_control import CascadeExhaustedError
    p = _pool(usd=10)
    p.spend(usd=10)
    assert p.effective_limits_for({"usd": 5}).exhausted == ("usd",)
    with pytest.raises(CascadeExhaustedError, match="no headroom"):
        p.child_config(_cfg(limits={"usd": 5}))


def test_pool_rungs_fire_on_aggregate_spend():
    p = CascadeBudgetPool("cid1", _cfg(
        limits={"usd": 10}, degrade=[{"at": 100, "action": "abort"}]))
    assert p.spend(usd=5) == ()
    fired = p.spend(usd=5)
    assert [r.action for r in fired] == ["abort"]


def test_spend_is_atomic_under_concurrency():
    """Atomic, not snapshot: N concurrent writers must not lose spend.

    Snapshot semantics are what silently break min-wins under fan-out — each
    child passes its own check against a stale remainder and they overshoot
    collectively.
    """
    p = _pool(usd=1000)
    def burn():
        for _ in range(100):
            p.spend(usd=1)
    threads = [threading.Thread(target=burn) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert p.remaining()["usd"] == pytest.approx(1000 - 800)


def test_rung_fires_exactly_once_across_concurrent_spenders():
    p = CascadeBudgetPool("cid1", _cfg(
        limits={"usd": 100}, degrade=[{"at": 50, "action": "abort"}]))
    fired = []
    def burn():
        for _ in range(50):
            fired.extend(p.spend(usd=1))
    threads = [threading.Thread(target=burn) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(fired) == 1, "latching must hold across threads"


def test_exhaustion_error_carries_framework_generated_evidence():
    """The refusal must be demonstrable by whoever reads the trace, at the
    same fidelity as a session-level refusal — not merely proof that the
    caller's own except block ran."""
    from shared.budget_control import CascadeExhaustedError
    p = _pool(usd=10, turns=20)
    p.spend(usd=10, turns=4)
    with pytest.raises(CascadeExhaustedError) as ei:
        p.child_config(_cfg(limits={"usd": 5, "turns": 3}))
    err = ei.value
    assert err.cascade_driver_id == "cid1"
    assert err.exhausted == ("usd",)
    payload = err.as_payload()
    assert payload["reason"] == "cascade_budget_exhausted"
    assert payload["profile_limits"]["usd"] == 5      # what it asked for
    assert payload["cascade_remaining"]["usd"] == 0.0  # what was left
    assert payload["exhausted_dimensions"] == ["usd"]
    # a non-exhausted dimension still reports, so the trace is complete
    assert payload["cascade_remaining"]["turns"] == 16.0
    assert "cascade_budget_exhausted" in err.render()


# ---------- tracker-authoritative reconciliation (the event-stream fix) ----

def test_reconcile_is_idempotent_and_delta_based():
    p = _pool(tokens=12000)
    assert p.reconcile_session("s1", {"tokens": 7695})["tokens"] == 7695.0
    assert p.reconcile_session("s1", {"tokens": 7695}) == {}   # same reading
    assert p.remaining()["tokens"] == 12000 - 7695


def test_reconcile_catches_spend_the_event_stream_dropped():
    """The leak, closed at the level that matters: the tracker's absolute
    total settles it regardless of which TurnCompletedEvents arrived."""
    p = _pool(tokens=12000)
    p.reconcile_session("s1", {"tokens": 7695})     # what events carried
    p.reconcile_session("s1", {"tokens": 9314})     # what the tracker saw
    assert p.remaining()["tokens"] == pytest.approx(12000 - 9314)


def test_reconcile_ignores_a_stale_lower_reading():
    """A late or out-of-order report must never refund spend."""
    p = _pool(tokens=12000)
    p.reconcile_session("s1", {"tokens": 9314})
    assert p.reconcile_session("s1", {"tokens": 100}) == {}
    assert p.remaining()["tokens"] == pytest.approx(12000 - 9314)


def test_reconcile_tracks_sessions_independently():
    p = _pool(tokens=30000)
    p.reconcile_session("s1", {"tokens": 9300})
    p.reconcile_session("s2", {"tokens": 9300})
    assert p.remaining()["tokens"] == pytest.approx(30000 - 18600)
    assert p.session_contribution("s1")["tokens"] == 9300


def test_reconcile_ignores_unknown_dimensions():
    p = _pool(tokens=100)
    p.reconcile_session("s1", {"tokens": 10, "bogus": 999})
    assert p.remaining()["tokens"] == 90.0


def test_linear_chain_via_reconciliation_lands_on_the_predicted_number():
    """Stage 1 really spent 9314; stage 2 must be clamped from THAT, not
    from the 7695 the events carried."""
    p = _pool(tokens=12000)
    p.reconcile_session("stage1", {"tokens": 9314})
    cfg, eff = p.child_config(_cfg(limits={"tokens": 9000}))
    assert cfg.limits["tokens"] == pytest.approx(2686.0)
    assert eff.clamped == ("tokens",)
