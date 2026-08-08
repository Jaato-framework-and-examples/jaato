"""Slice 3: a cascade rung pushed to an ALREADY-RUNNING child.

Spawn-time clamping constrains only children not yet started. A pool that
let running siblings keep the ceiling they were handed would not be a
shared budget — the point of it being aggregate is that one child burning
the envelope affects everyone still running.

The pushed rung goes through the SAME _apply_budget_rungs path a session's
own ladder uses, so it produces identical machinery and evidence, tagged
origin="cascade".
"""

from types import SimpleNamespace

import pytest

from shared.budget_control import (
    BudgetControlConfig,
    BudgetTracker,
    CascadeBudgetPool,
    DegradeRung,
)
from shared.jaato_session import JaatoSession
from shared.model_tiers import (
    RESERVED_FALLBACK_KEY,
    RESERVED_INITIAL_KEY,
    ModelTierConfig,
    TierEntry,
)


def _cfg(**kw):
    return BudgetControlConfig.from_dict(kw)


_TIERS = {
    "planner": {"model": "opus", "provider": "openrouter"},
    "executor": {"model": "haiku", "provider": "openrouter"},
    RESERVED_INITIAL_KEY: "planner",
    RESERVED_FALLBACK_KEY: "executor",
}


def _session(budget=None):
    connects, notices = [], []
    s = SimpleNamespace(
        _tier_config=ModelTierConfig.from_unified_dict(dict(_TIERS)),
        _active_tier="planner", _model_name="opus",
        _provider=SimpleNamespace(
            name="openrouter",
            connect=lambda m, skip_model_test=True: connects.append(m)),
        _active_provider_name="openrouter", _provider_cache={},
        _budget_tracker=BudgetTracker(budget) if budget else None,
        _budget_terminal_action=None, _budget_exhausted_reason=None,
        _current_output_callback=lambda src, txt, mode: notices.append(txt),
        _ui_hooks=None, _connects=connects, _notices=notices,
    )
    for n in ("_is_connected_to", "_connect_tier_entry",
              "_reconnect_active_tier_if_rebound", "_apply_budget_rungs",
              "_surface_budget_event", "apply_cascade_degrade"):
        setattr(s, n, (lambda nm: (lambda *a, **k:
                getattr(JaatoSession, nm)(s, *a, **k)))(n))
    s.request_stop = lambda reason="": s.__setattr__("_stopped", reason) or True
    return s


# ------------------------------------------------- the push, session side

def test_pushed_rung_rebinds_and_reconnects_a_running_child():
    s = _session()
    out = s.apply_cascade_degrade([
        {"at": 50.0, "model_tiers": {"planner": {"model": "flash",
                                                 "provider": "openrouter"}}}])
    assert out == {"applied": 1}
    assert s._tier_config.tiers["planner"].model == "flash"
    assert s._connects == ["flash"]        # the running child re-pointed
    assert s._model_name == "flash"


def test_pushed_rung_is_tagged_cascade_in_the_client_notice():
    """A reactor must tell 'I overspent' from 'the cascade overspent' —
    they invite opposite responses, and without the tag both paths emit
    identical lines."""
    s = _session()
    s.apply_cascade_degrade([
        {"at": 50.0, "model_tiers": {"planner": "flash"}}])
    assert s._notices and "cascade budget" in s._notices[0]


def test_own_ladder_is_tagged_session_not_cascade():
    s = _session(budget=_cfg(limits={"tokens": 100},
                             degrade=[{"at": 50, "model_tiers":
                                       {"planner": "flash"}}]))
    fired = s._budget_tracker.observe(tokens=50)
    JaatoSession._apply_budget_rungs(s, fired)          # default origin
    assert s._notices and "session budget" in s._notices[0]


def test_pushed_abort_latches_a_refusal_naming_the_cascade():
    s = _session()
    s.apply_cascade_degrade([{"at": 100.0, "action": "abort"}])
    assert s._budget_terminal_action == "abort"
    assert "cascade" in s._budget_exhausted_reason
    assert "budget_exhausted" in getattr(s, "_stopped", "")


def test_push_works_on_a_child_with_no_budget_of_its_own():
    """A child need not be budgeted itself to be degraded by the pool."""
    s = _session(budget=None)
    assert s.apply_cascade_degrade([{"at": 50.0, "model_tiers":
                                     {"planner": "flash"}}]) == {"applied": 1}
    assert s._connects == ["flash"]


def test_malformed_push_is_rejected_without_raising():
    s = _session()
    out = s.apply_cascade_degrade([{"at": 999, "action": "abort"}])
    assert out["applied"] == 0 and "error" in out
    assert s._connects == []


def test_empty_push_is_a_noop():
    s = _session()
    assert s.apply_cascade_degrade([]) == {"applied": 0}


# --------------------------------------------------- the pool, fire side

def test_reconcile_returns_the_rungs_it_crossed():
    """Regression: reconcile_session used to discard what observe() fired,
    so a cascade degrade ladder could never fire at all — invisible until a
    cascade actually declared one."""
    pool = CascadeBudgetPool("cid", _cfg(
        limits={"tokens": 15000},
        degrade=[{"at": 50, "model_tiers": {"planner": "flash"}},
                 {"at": 100, "action": "abort"}]))
    assert pool.reconcile_session("a", {"tokens": 3000}).fired == ()
    fired = pool.reconcile_session("a", {"tokens": 7500}).fired
    assert [r.at_percent for r in fired] == [50.0]


def test_concurrent_children_cross_the_pool_rung_exactly_once():
    """Three siblings spending against one pool: the 50% rung fires once
    across all of them, not once per child."""
    pool = CascadeBudgetPool("cid", _cfg(
        limits={"tokens": 15000}, degrade=[{"at": 50, "action": "abort"}]))
    fired_total = []
    for child, spend in (("a", 3000), ("b", 3000), ("c", 3000)):
        fired_total.extend(pool.reconcile_session(child, {"tokens": spend}).fired)
    assert [r.at_percent for r in fired_total] == [50.0]
    assert pool.remaining()["tokens"] == 6000.0
