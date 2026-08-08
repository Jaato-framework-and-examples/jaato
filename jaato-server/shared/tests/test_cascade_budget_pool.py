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


def test_a_child_with_its_own_budget_is_NOT_clamped_by_the_scope():
    """A delegation to another department with its own budget: the author
    wrote a number and the parent does not rewrite it down to whatever is
    left in the shared pot. Its spend is accounted on its own books."""
    p = _pool(usd=10, turns=12)
    p.spend(usd=7, turns=8)                       # pot down to 3
    cfg, eff = p.child_config(_cfg(limits={"usd": 9, "turns": 2}))
    assert cfg.limits["usd"] == 9                 # as declared, NOT 3
    assert cfg.limits["turns"] == 2
    # EffectiveLimits survives as observability, not as a clamp
    assert eff.profile_limits["usd"] == 9
    assert eff.cascade_remaining["usd"] == 3.0


def test_a_child_with_no_budget_of_its_own_IS_bounded_by_the_remainder():
    """Spawning without a declared budget delegates the policy to the
    parent, so the child draws on what is left."""
    p = _pool(usd=10, turns=12)
    p.spend(usd=7, turns=8)
    cfg, eff = p.child_config(None)
    assert cfg.limits["usd"] == 3.0
    # every dimension comes from the pool for a child that declared none
    assert "usd" in eff.clamped


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


# ------- whose degradation POLICY applies (literal vs inherited) ---------

def _pool_with_ladder():
    return CascadeBudgetPool("cid", _cfg(
        limits={"tokens": 15000},
        degrade=[{"at": 50, "action": "abort"}]))


def test_profileless_child_inherits_the_cascade_ladder():
    """Nothing was expressed, so the cascade's policy applies. Without this
    the child got a ceiling with NO behaviour attached — its tracker would
    cross the limit and nothing would fire, leaving a best-effort push as
    the only thing that could degrade it."""
    cfg, _ = _pool_with_ladder().child_config(None)
    assert [r.action for r in cfg.degrade] == ["abort"]


def test_limits_only_profile_is_taken_LITERALLY_and_never_degraded():
    """Declaring budget_control with limits and no degrade is a deliberate
    'cap me but do not degrade me'. The cascade constrains ceilings, never
    policy — we are not entitled to degrade a profile whose author did not
    ask for it."""
    cfg, _ = _pool_with_ladder().child_config(_cfg(limits={"tokens": 9000}))
    assert list(cfg.degrade) == []


def test_a_child_with_its_own_ladder_keeps_it():
    cfg, _ = _pool_with_ladder().child_config(_cfg(
        limits={"tokens": 9000}, degrade=[{"at": 90, "action": "finalize"}]))
    assert [r.action for r in cfg.degrade] == ["finalize"]


def test_profileless_child_of_a_ladderless_cascade_has_no_ladder():
    """Nobody expressed a policy anywhere; inheriting nothing is correct."""
    cfg, _ = _pool(tokens=15000).child_config(None)
    assert list(cfg.degrade) == []


def test_child_dimension_the_cascade_does_not_cap_is_untouched():
    p = _pool(usd=10)
    cfg, eff = p.child_config(_cfg(limits={"turns": 5}))
    assert cfg.limits["turns"] == 5
    assert "turns" not in eff.clamped


def test_exhausted_scope_refuses_a_child_that_has_no_budget_of_its_own():
    """Zero is not a budget a session can run under, and returning None
    ("unbudgeted") would be catastrophic — the reason there is none is that
    the scope is OUT. Fail loud so the caller refuses the spawn."""
    from shared.budget_control import CascadeExhaustedError
    p = _pool(usd=10)
    p.spend(usd=10)
    assert p.effective_limits_for(None).exhausted == ("usd",)
    with pytest.raises(CascadeExhaustedError, match="no headroom"):
        p.child_config(None)


def test_exhausted_scope_still_admits_a_child_with_its_own_budget():
    """Its own books. An empty shared pot is not that department's problem."""
    p = _pool(usd=10)
    p.spend(usd=10)
    cfg, _ = p.child_config(_cfg(limits={"usd": 5}))
    assert cfg.limits["usd"] == 5


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
        p.child_config(None)
    err = ei.value
    assert err.cascade_driver_id == "cid1"
    assert err.exhausted == ("usd",)
    payload = err.as_payload()
    assert payload["reason"] == "cascade_budget_exhausted"
    assert payload["cascade_remaining"]["usd"] == 0.0  # nothing left
    assert payload["cascade_remaining"]["usd"] == 0.0  # what was left
    assert payload["exhausted_dimensions"] == ["usd"]
    # a non-exhausted dimension still reports, so the trace is complete
    assert payload["cascade_remaining"]["turns"] == 16.0
    assert "cascade_budget_exhausted" in err.render()


# ---------- tracker-authoritative reconciliation (the event-stream fix) ----

def test_reconcile_is_idempotent_and_delta_based():
    p = _pool(tokens=12000)
    assert p.reconcile_session("s1", {"tokens": 7695}).deltas["tokens"] == 7695.0
    assert p.reconcile_session("s1", {"tokens": 7695}).deltas == {}  # same reading
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
    assert p.reconcile_session("s1", {"tokens": 100}).deltas == {}
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


def test_reconciliation_feeds_the_remainder_a_pool_drawing_child_gets():
    """Stage 1 really spent 9314; a child drawing on the pot must see THAT,
    not the 7695 the events carried."""
    p = _pool(tokens=12000)
    p.reconcile_session("stage1", {"tokens": 9314})
    cfg, eff = p.child_config(None)
    assert cfg.limits["tokens"] == pytest.approx(2686.0)
    assert eff.clamped == ("tokens",)
