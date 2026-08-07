"""budget_control RUNTIME layer: tracker firing policy + the brownout.

Covers design note §5 (latching / first-dimension-wins / cumulative) and
§6.2 (the switch_tier re-resolve without which an overlay applied to the
CURRENTLY ACTIVE tier silently no-ops).

The session-level tests follow the ``test_cross_provider_tiers.py``
fixture pattern: a SimpleNamespace carrying exactly the attributes the
methods under test touch, with the real helpers bound.
"""

from types import SimpleNamespace

import pytest

from shared.budget_control import (
    BudgetControlConfig,
    BudgetTracker,
    overlay_tier_table,
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


# ------------------------------------------------------------- tracker

def test_no_limits_never_fires():
    t = BudgetTracker(BudgetControlConfig())
    assert t.observe(usd=1e9) == ()
    assert t.usage_fraction() == 0.0


def test_fires_at_threshold_not_before():
    t = BudgetTracker(_cfg(limits={"usd": 10}, degrade=[{"at": 70, "action": "abort"}]))
    assert t.observe(usd=6.9) == ()
    assert len(t.observe(usd=0.1)) == 1        # now exactly 70%


def test_latched_fires_once_even_if_usage_recovers():
    """GC LOWERS token usage; without latching a token rung would flap the
    model between an expensive and a cheap binding on every GC cycle."""
    t = BudgetTracker(_cfg(limits={"tokens": 100},
                           degrade=[{"at": 50, "action": "abort"}]))
    assert len(t.observe(tokens=60)) == 1
    assert t.observe(tokens=40) == ()          # further spend: no refire
    t._usage.tokens = 10                       # simulate a GC-style drop
    assert t.observe(tokens=90) == ()          # crossing again: still latched


def test_first_dimension_wins():
    """Blowing the dollar ceiling fires even though tokens are untouched."""
    t = BudgetTracker(_cfg(limits={"usd": 1, "tokens": 1_000_000},
                           degrade=[{"at": 100, "action": "abort"}]))
    fired = t.observe(usd=1.0)
    assert len(fired) == 1
    assert set(t.exceeded_dimensions()) == {"usd"}


def test_multiple_rungs_fire_in_order_on_one_jump():
    t = BudgetTracker(_cfg(limits={"usd": 10},
                           degrade=[{"at": 50, "action": "abort"},
                                    {"at": 90, "action": "finalize"}]))
    fired = t.observe(usd=10)
    assert [r.at_percent for r in fired] == [50.0, 90.0]


def test_unknown_cost_does_not_advance_usd():
    """A budget must never hard-stop on a number it invented."""
    t = BudgetTracker(_cfg(limits={"usd": 1}, degrade=[{"at": 100, "action": "abort"}]))
    assert t.observe(usd=None, tokens=5) == ()
    assert t.usage.usd == 0.0


def test_usage_fraction_not_clamped():
    t = BudgetTracker(_cfg(limits={"usd": 10}))
    t.observe(usd=25)
    assert t.usage_fraction() == pytest.approx(2.5)


# ------------------------------------------------------- overlay table

def test_overlay_rebinds_and_reports_changes():
    tiers = {"planner": TierEntry("opus", "openrouter"),
             "executor": TierEntry("haiku", "openrouter")}
    changes = overlay_tier_table(tiers, {"planner": TierEntry("flash", "openrouter")})
    assert tiers["planner"].model == "flash"
    assert tiers["executor"].model == "haiku"          # untouched
    assert changes == {"planner": "opus -> flash"}


def test_overlay_to_identical_binding_is_a_noop():
    tiers = {"planner": TierEntry("opus", "openrouter")}
    assert overlay_tier_table(tiers, {"planner": TierEntry("opus", "openrouter")}) == {}


# --------------------------------------------- session brownout (§6.2)

_TIERS = {
    "planner": {"model": "opus", "provider": "openrouter"},
    "executor": {"model": "haiku", "provider": "openrouter"},
    RESERVED_INITIAL_KEY: "planner",
    RESERVED_FALLBACK_KEY: "executor",
}


def _session(active="planner", model="opus", budget=None):
    connects = []
    s = SimpleNamespace(
        _tier_config=ModelTierConfig.from_unified_dict(dict(_TIERS)),
        _active_tier=active,
        _model_name=model,
        _provider=SimpleNamespace(
            name="openrouter",
            connect=lambda m, skip_model_test=True: connects.append(m)),
        _active_provider_name="openrouter",
        _provider_cache={},
        _budget_tracker=BudgetTracker(budget) if budget else None,
        _budget_terminal_action=None,
        _budget_exhausted_reason=None,
        _current_output_callback=None,
        _ui_hooks=None,
        _connects=connects,
    )
    for name in ("_is_connected_to", "_connect_tier_entry",
                 "_reconnect_active_tier_if_rebound", "_apply_budget_rungs",
                 "_surface_budget_event", "_refuse_if_budget_exhausted",
                 "switch_tier"):
        setattr(s, name, (lambda n: (lambda *a, **k:
                getattr(JaatoSession, n)(s, *a, **k)))(name))
    s.request_stop = lambda reason="": s.__setattr__("_stopped", reason) or True
    return s


def test_switch_tier_still_short_circuits_a_genuine_noop():
    s = _session()
    assert JaatoSession.switch_tier(s, "planner")["status"] == "already_at_tier"
    assert s._connects == []                    # no pointless reconnect


def test_rebinding_the_active_tier_reconnects():
    """THE §6.2 regression: name-only comparison would report
    already_at_tier and never re-connect, so the brownout would silently
    not take effect until the agent left and re-entered the tier."""
    s = _session()
    s._tier_config.tiers["planner"] = TierEntry("flash", "openrouter")
    r = JaatoSession.switch_tier(s, "planner")
    assert s._connects == ["flash"]
    assert r["model"] == "flash"


def test_degrade_rung_rebinds_active_tier_end_to_end():
    budget = _cfg(limits={"usd": 10},
                  degrade=[{"at": 70, "model_tiers": {
                      "planner": {"model": "flash", "provider": "openrouter"}}}])
    s = _session(budget=budget)
    fired = s._budget_tracker.observe(usd=7)
    JaatoSession._apply_budget_rungs(s, fired)
    assert s._tier_config.tiers["planner"].model == "flash"   # table rebound
    assert s._connects == ["flash"]                           # AND reconnected
    assert s._model_name == "flash"


def test_degrade_rung_leaves_inactive_tier_unconnected():
    """Rebinding a tier the agent is not in must not reconnect anything."""
    budget = _cfg(limits={"usd": 10},
                  degrade=[{"at": 70, "model_tiers": {
                      "executor": {"model": "nano", "provider": "openrouter"}}}])
    s = _session(active="planner", budget=budget)
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=7))
    assert s._tier_config.tiers["executor"].model == "nano"
    assert s._connects == []


def test_abort_action_requests_stop():
    budget = _cfg(limits={"usd": 1}, degrade=[{"at": 100, "action": "abort"}])
    s = _session(budget=budget)
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=1))
    assert s._budget_terminal_action == "abort"
    assert "budget_exhausted" in getattr(s, "_stopped", "")


def test_finalize_action_latches_without_stopping():
    budget = _cfg(limits={"usd": 1}, degrade=[{"at": 100, "action": "finalize"}])
    s = _session(budget=budget)
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=1))
    assert s._budget_terminal_action == "finalize"
    assert not hasattr(s, "_stopped")


def test_overlay_without_tier_config_is_ignored_not_crashed():
    budget = _cfg(limits={"usd": 1}, degrade=[{"at": 100, "model_tiers": {
        "planner": {"model": "flash", "provider": "openrouter"}}}])
    s = _session(budget=budget)
    s._tier_config = None
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(usd=1))
    assert s._connects == []


# --------------------------------------------------------- wire format

def test_envelope_round_trips_budget_control():
    from shared.session_envelope import SessionInitEnvelope
    kw = dict(session_id="s", workspace_path="/tmp", profile_name="p",
              provider_name="openrouter", model_name="m")
    payload = _cfg(limits={"usd": 3.0},
                   degrade=[{"at": 70, "action": "abort"}]).to_dict()
    env = SessionInitEnvelope(**kw, budget_control=payload)
    assert SessionInitEnvelope.from_dict(env.to_dict()).budget_control == payload
    assert SessionInitEnvelope.from_dict(
        SessionInitEnvelope(**kw).to_dict()).budget_control is None


# ------------------------------------------- findings from the live PoC run

def test_abort_latches_a_refusal_so_later_turns_cannot_run():
    """PoC finding B: abort is a COOPERATIVE cancel of the in-flight turn.
    Without a latch the client just sends again and every later turn is
    unbudgeted (rungs are latched, so 100% never re-fires) — a `turns: 4`
    budget ran to 8. A ceiling that only cancels one turn is not a ceiling."""
    budget = _cfg(limits={"turns": 4}, degrade=[{"at": 100, "action": "abort"}])
    s = _session(budget=budget)
    assert JaatoSession._refuse_if_budget_exhausted(s) is None
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(turns=4))
    reason = JaatoSession._refuse_if_budget_exhausted(s)
    assert reason is not None and "budget_exhausted" in reason


def test_finalize_does_not_latch_a_refusal():
    """finalize is graceful — it must NOT block further turns, that's abort."""
    budget = _cfg(limits={"turns": 4}, degrade=[{"at": 100, "action": "finalize"}])
    s = _session(budget=budget)
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(turns=4))
    assert JaatoSession._refuse_if_budget_exhausted(s) is None


def test_budget_event_uses_the_real_output_channel():
    """PoC finding A: this used _ui_hooks, which is never set on the runner
    path, so every budget decision was silently dropped client-side."""
    seen = []
    s = _session()
    s._current_output_callback = lambda src, txt, mode: seen.append((src, txt))
    JaatoSession._surface_budget_event(s, "degraded planner")
    assert seen and seen[0][0] == "system" and "degraded planner" in seen[0][1]


def test_surface_is_a_noop_without_a_callback():
    s = _session()
    s._current_output_callback = None
    JaatoSession._surface_budget_event(s, "x")   # must not raise


def test_pressure_names_the_driving_dimension_below_100pct():
    """PoC nit: exceeded_dimensions() is empty below 100%, so a 50% rung
    reported the useless '50% of budget' without saying WHICH dimension."""
    t = BudgetTracker(_cfg(limits={"turns": 4, "usd": 100}))
    t.observe(turns=2, usd=1)
    assert "turns 50%" in t.describe_pressure()
