"""validate_profile: model_tiers satisfies the model requirement (no false missing_model).

A provider-set profile binds a model EITHER flat (model:) OR via model_tiers (the
active model is selected per turn). The validator must not warn missing_model when
model_tiers is present — else a tiers-based profile can never reach 0 warnings
(model: present -> silently ignored; model: absent -> missing_model). Found via the
jaato-tui-driven-tests scaffold refactor (tiered_test profile).
"""
from types import SimpleNamespace

from shared.scaffold.validate import validate_profile
from shared.model_tiers import VALID_TIER_NAMES


def _profile(**kw):
    base = dict(name="t", provider="openrouter", model=None, model_tiers=None,
                plugins=[], preloaded_plugins=set(), tool_scopes={},
                plugin_configs={}, quirks={})
    base.update(kw)
    return SimpleNamespace(**base)


def _codes(diags):
    return {d.code for d in diags}


def test_model_tiers_satisfies_model_requirement():
    tier = next(iter(VALID_TIER_NAMES))
    p = _profile(model=None, model_tiers={
        tier: {"provider": "openrouter", "model": "openrouter/auto"}})
    assert "missing_model" not in _codes(
        validate_profile(p, providers={}, plugins={}, gc_names=[]))


def test_no_model_and_no_tiers_warns_missing_model():
    p = _profile(model=None, model_tiers=None)
    assert "missing_model" in _codes(
        validate_profile(p, providers={}, plugins={}, gc_names=[]))


def test_flat_model_satisfies():
    p = _profile(model="openrouter/auto", model_tiers=None)
    assert "missing_model" not in _codes(
        validate_profile(p, providers={}, plugins={}, gc_names=[]))
