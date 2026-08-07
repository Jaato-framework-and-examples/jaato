"""budget_control: parsing, validation, and profile inheritance.

Mirrors ``test_model_tiers.py`` — the config-layer contract for the
``budget_control`` profile block (multi-dimensional ceilings + a
degradation ladder that REBINDS model_tiers entries rather than moving
the agent between tiers).  See
``docs/design/budget-control-degradation.md``.
"""

import pytest

from shared.budget_control import (
    ACTION_FINALIZE,
    VALID_ACTIONS,
    VALID_DIMENSIONS,
    BudgetControlConfig,
    BudgetControlConfigError,
    DegradeRung,
    merge_limits,
)
from shared.plugins.subagent.config import (
    SubagentProfile,
    _merge_budget_control,
    build_inline_profile,
)


# ----------------------------------------------------------- parsing

def test_absent_and_empty_parse_to_none():
    """Absent / empty must be indistinguishable, so consumers can treat
    'unbudgeted' as a single case."""
    assert BudgetControlConfig.from_dict(None) is None
    assert BudgetControlConfig.from_dict({}) is None


def test_full_block_parses():
    cfg = BudgetControlConfig.from_dict({
        "limits": {"usd": 3.0, "tokens": 300000},
        "degrade": [
            {"at": "70%", "model_tiers": {
                "planner": {"model": "google/gemini-flash",
                            "provider": "openrouter"}}},
            {"at": 100, "action": "finalize"},
        ],
    })
    assert cfg.limits == {"usd": 3.0, "tokens": 300000}
    assert len(cfg.degrade) == 2
    assert cfg.degrade[0].at_percent == 70.0
    assert cfg.degrade[0].model_tiers["planner"].model == "google/gemini-flash"
    assert cfg.degrade[0].model_tiers["planner"].provider == "openrouter"
    assert cfg.degrade[1].action == ACTION_FINALIZE
    assert cfg.has_tier_overlays is True


def test_overlay_accepts_bare_string_shorthand():
    """An overlay entry uses the SAME grammar as a base model_tiers entry —
    the bare-string shorthand must work, or the two would drift."""
    cfg = BudgetControlConfig.from_dict({
        "limits": {"usd": 1},
        "degrade": [{"at": 50, "model_tiers": {"executor": "cheap-model"}}],
    })
    entry = cfg.degrade[0].model_tiers["executor"]
    assert entry.model == "cheap-model"
    assert entry.provider is None


@pytest.mark.parametrize("at,expected", [(70, 70.0), ("70%", 70.0),
                                         ("70", 70.0), (100, 100.0)])
def test_at_percentage_spellings(at, expected):
    cfg = BudgetControlConfig.from_dict({
        "limits": {"usd": 1}, "degrade": [{"at": at, "action": "abort"}]})
    assert cfg.degrade[0].at_percent == expected


@pytest.mark.parametrize("at", [0, -5, 101, "", "abc", True, None])
def test_at_out_of_range_or_malformed_rejected(at):
    with pytest.raises(BudgetControlConfigError):
        BudgetControlConfig.from_dict({
            "limits": {"usd": 1}, "degrade": [{"at": at, "action": "abort"}]})


def test_unknown_dimension_rejected():
    with pytest.raises(BudgetControlConfigError, match="unknown dimension"):
        BudgetControlConfig.from_dict({"limits": {"dollars": 3.0}})


@pytest.mark.parametrize("value", [0, -1, "lots", True])
def test_non_positive_limit_rejected(value):
    with pytest.raises(BudgetControlConfigError):
        BudgetControlConfig.from_dict({"limits": {"usd": value}})


def test_unknown_top_level_key_rejected():
    with pytest.raises(BudgetControlConfigError, match="unknown key"):
        BudgetControlConfig.from_dict({"limits": {"usd": 1}, "scope": "cascade"})


def test_unknown_action_rejected():
    with pytest.raises(BudgetControlConfigError, match="not a valid action"):
        BudgetControlConfig.from_dict({
            "limits": {"usd": 1}, "degrade": [{"at": 50, "action": "downgrade"}]})


def test_overlay_tier_typo_rejected():
    with pytest.raises(BudgetControlConfigError, match="not a tier name"):
        BudgetControlConfig.from_dict({
            "limits": {"usd": 1},
            "degrade": [{"at": 50, "model_tiers": {"visionn": "m"}}]})


def test_overlay_rejects_control_keys():
    """initial / fallback SELECT a tier; they bind no model, so they are
    meaningless in an overlay and must not be silently ignored."""
    with pytest.raises(BudgetControlConfigError, match="control key"):
        BudgetControlConfig.from_dict({
            "limits": {"usd": 1},
            "degrade": [{"at": 50, "model_tiers": {"initial": "planner"}}]})


def test_empty_rung_rejected():
    with pytest.raises(BudgetControlConfigError, match="does nothing"):
        BudgetControlConfig.from_dict({
            "limits": {"usd": 1}, "degrade": [{"at": 50}]})


def test_degrade_without_limits_rejected():
    """'at' is a percentage OF a limit — a ladder with no ceilings can never
    fire, so it must fail loud rather than sit inert."""
    with pytest.raises(BudgetControlConfigError, match="without limits"):
        BudgetControlConfig.from_dict({
            "degrade": [{"at": 50, "action": "abort"}]})


def test_thresholds_must_strictly_increase():
    with pytest.raises(BudgetControlConfigError, match="strictly increase"):
        BudgetControlConfig.from_dict({
            "limits": {"usd": 1},
            "degrade": [{"at": 70, "action": "abort"},
                        {"at": 70, "action": "finalize"}]})


def test_constants_are_the_documented_sets():
    assert VALID_DIMENSIONS == {"usd", "tokens", "seconds", "tool_calls", "turns"}
    assert VALID_ACTIONS == {"finalize", "abort", "escalate"}


# ------------------------------------------------------- merge_limits

def test_merge_limits_min_wins():
    assert merge_limits({"usd": 1.0}, {"usd": 99.0}) == {"usd": 1.0}
    assert merge_limits({"usd": 99.0}, {"usd": 1.0}) == {"usd": 1.0}


def test_merge_limits_unions_distinct_dimensions():
    assert merge_limits({"usd": 1.0}, {"tokens": 10}) == {"usd": 1.0, "tokens": 10}


# --------------------------------------------------- profile plumbing

def _prof(name, bc=None, **kw):
    return SubagentProfile(name=name, description="d", plugins=[],
                           budget_control=bc, **kw)


def test_inline_spec_parses_budget_control():
    p = build_inline_profile({"plugins": [], "budget_control": {
        "limits": {"usd": 2}, "degrade": [{"at": 80, "action": "finalize"}]}},
        name="t")
    assert p.budget_control.limits == {"usd": 2}


def test_inline_spec_rejects_bad_budget_control_at_load_time():
    """The silent-ignore class: a malformed block must fail at LOAD, not be
    dropped and look configured."""
    with pytest.raises(ValueError, match="budget_control"):
        build_inline_profile(
            {"plugins": [], "budget_control": {"limits": {"bogus": 1}}}, name="t")


def test_profile_without_budget_control_is_none():
    assert build_inline_profile({"plugins": []}, name="t").budget_control is None


def test_inherit_limits_min_wins_child_cannot_widen():
    """The safety invariant: a child must never grant itself a bigger ceiling
    than the profile that spawned it."""
    parent = _prof("p", BudgetControlConfig(limits={"usd": 1.0, "tokens": 50000}))
    child = _prof("c", BudgetControlConfig(limits={"usd": 99.0, "seconds": 60}))
    merged = _merge_budget_control([parent], child)
    assert merged.limits == {"usd": 1.0, "tokens": 50000, "seconds": 60}


def test_inherit_two_parents_take_the_minimum():
    merged = _merge_budget_control(
        [_prof("a", BudgetControlConfig(limits={"usd": 8})),
         _prof("b", BudgetControlConfig(limits={"usd": 3}))], _prof("c"))
    assert merged.limits == {"usd": 3}


def test_inherit_degrade_is_scalar_override():
    parent = _prof("p", BudgetControlConfig(
        limits={"usd": 5}, degrade=(DegradeRung(50.0, action="abort"),)))
    child = _prof("c", BudgetControlConfig(
        limits={"usd": 5}, degrade=(DegradeRung(90.0, action="finalize"),)))
    merged = _merge_budget_control([parent], child)
    assert [(r.at_percent, r.action) for r in merged.degrade] == [(90.0, "finalize")]


def test_inherit_parent_ladder_when_child_declares_none():
    parent = _prof("p", BudgetControlConfig(
        limits={"usd": 5}, degrade=(DegradeRung(50.0, action="abort"),)))
    merged = _merge_budget_control([parent], _prof("c"))
    assert merged.limits == {"usd": 5}
    assert [(r.at_percent, r.action) for r in merged.degrade] == [(50.0, "abort")]


def test_inherit_nothing_stays_unbudgeted():
    assert _merge_budget_control([_prof("p")], _prof("c")) is None
