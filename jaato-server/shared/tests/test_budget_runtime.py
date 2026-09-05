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
        _request_tier_output_modalities=lambda entry: None,
        _active_provider_name="openrouter",
        _provider_cache={},
        # ``__init__`` always sets these; a double that omits them is a
        # shape production never has, and the post-connect bookkeeping in
        # ``_connect_tier_entry`` reads them.
        _tier_switch_count=0,
        _cache_plugin=None,
        _cache_plugins_by_provider={},
        _instruction_budget=None,
        _runtime=None,
        _trace=lambda *a, **k: None,
        _budget_tracker=BudgetTracker(budget) if budget else None,
        _budget_terminal_action=None,
        _budget_exhausted_reason=None,
        _budget_notice_sink=None,
        _budget_applied_rung_pct=0.0,
        _current_output_callback=None,
        _ui_hooks=None,
        _connects=connects,
    )
    for name in ("_is_connected_to", "_connect_tier_entry",
                 "_reconnect_active_tier_if_rebound", "_apply_budget_rungs",
                 "_surface_budget_event", "_refuse_if_budget_exhausted",
                 # Post-connect bookkeeping, bound so the double exercises
                 # the real ones rather than silently skipping them.
                 "_wire_cache_plugin", "_retarget_reliability_model",
                 "_cache_plugin_config",
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


def test_refused_send_is_flagged_so_no_turn_completed_is_emitted():
    """PoC residual: a refused turn still emitted TURN_COMPLETED, so a client
    counting turns over-counted (8 = 4 real + 4 refusals). Worse, the runner
    sources that payload from turn_accounting[-1] and a refused turn appends
    nothing — it re-emitted the PREVIOUS turn's tokens."""
    budget = _cfg(limits={"turns": 2}, degrade=[{"at": 100, "action": "abort"}])
    s = _session(budget=budget)
    s._last_send_refused = False
    assert JaatoSession.was_last_send_refused(s) is False
    JaatoSession._apply_budget_rungs(s, s._budget_tracker.observe(turns=2))
    # the gate now reports a refusal; simulate what send_message records
    s._last_send_refused = JaatoSession._refuse_if_budget_exhausted(s) is not None
    assert JaatoSession.was_last_send_refused(s) is True


def test_turn_data_separates_context_size_from_turn_spend():
    """A turn with a tool call has >=2 responses and each is BILLED, but
    turn_data['total'] ASSIGNS the last response (= end-of-turn context size).
    Summing turn.completed therefore undercounted a real 3-turn run by 41%.
    'spend_*' accumulates, so the cascade pool and the per-session tracker
    count the same thing — min(profile, cascade_remaining) is meaningless if
    they don't."""
    turn_data = {'prompt': 0, 'output': 0, 'total': 0,
                 'spend_total': 0, 'spend_prompt': 0, 'spend_output': 0}

    def observe(prompt, output, total):
        # mirrors usage_callback_with_turn_tracking
        turn_data['prompt'] = prompt
        turn_data['output'] = output
        turn_data['total'] = total
        turn_data['spend_total'] += total
        turn_data['spend_prompt'] += prompt
        turn_data['spend_output'] += output

    observe(2000, 150, 2150)   # response 1: the tool call
    observe(2400, 104, 2504)   # response 2: after the tool result
    assert turn_data['total'] == 2504          # context size at turn end
    assert turn_data['spend_total'] == 4654    # what the turn actually cost


def test_spend_accumulates_on_every_path_not_just_streaming():
    """spend_* must live in _accumulate_turn_tokens, which runs exactly once
    per response on EVERY path. The streaming usage-callback is the wrong
    hook twice over: it never fires non-streaming (spend would stay 0), and a
    provider emitting >1 usage chunk per response would double-count."""
    from types import SimpleNamespace
    td = {'prompt': 0, 'output': 0, 'total': 0,
          'spend_total': 0, 'spend_prompt': 0, 'spend_output': 0}

    def resp(p, o, t):
        return SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=p, output_tokens=o, total_tokens=t,
            cache_read_tokens=None, cache_creation_tokens=None,
            thinking_tokens=0, cost_usd=None))

    sess = SimpleNamespace(_update_thinking_budget=lambda n: None)
    JaatoSession._accumulate_turn_tokens(sess, resp(2000, 150, 2150), td)
    JaatoSession._accumulate_turn_tokens(sess, resp(2400, 104, 2504), td)
    assert td['total'] == 2504        # context size: replaced
    assert td['spend_total'] == 4654  # spend: accumulated
    assert td['spend_prompt'] == 4400
    assert td['spend_output'] == 254
