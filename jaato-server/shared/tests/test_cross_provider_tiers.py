"""V2 cross-provider model tiers — the JaatoSession.switch_tier swap + cache.

Deterministic unit test of the provider-lifecycle mechanism (no live daemon /
model): a tier declaring a DIFFERENT provider swaps self._provider to a cached
per-provider instance; switching back is a cache hit (no re-create); a
same-provider switch takes no swap path at all.
"""

from types import SimpleNamespace

from shared.jaato_session import JaatoSession
from shared.model_tiers import (
    ModelTierConfig, RESERVED_INITIAL_KEY, RESERVED_FALLBACK_KEY,
)


def _session(tier_dict):
    created = []
    cfgs = []

    def fake_create(model, provider_name=None, skip_model_test=True,
                    plugin_configs=None):
        created.append((provider_name, model))
        cfgs.append(plugin_configs)
        return SimpleNamespace(
            name=provider_name,
            connect=lambda m, skip_model_test=True: None,
        )

    s = SimpleNamespace(
        _tier_config=ModelTierConfig.from_unified_dict(tier_dict),
        _active_tier="executor",
        _provider=SimpleNamespace(
            name="zhipuai", connect=lambda m, skip_model_test=True: None),
        _request_tier_output_modalities=lambda entry: None,
        _active_provider_name="zhipuai",
        _provider_cache={},
        # Realistic post-main-provider state: _provider_lazy_pending is already
        # CLEARED (_ensure_provider does this); the persistent base config is
        # what carries plugin_configs to a tier switch.
        _provider_lazy_pending=None,
        _tier_provider_base={
            "skip_model_test": True,
            "plugin_configs": {"openrouter": {"api_key": "K-openrouter"}},
        },
        _runtime=SimpleNamespace(create_provider=fake_create),
        _model_name="glm-4.6",
        _agent_type=None, _agent_name=None, _agent_id=None,
        _created=created, _cfgs=cfgs,
    )
    s._provider_cache["zhipuai"] = s._provider
    # Bind the real helpers so switch_tier's self._* calls resolve.
    s._provider_for_tier = lambda pn, m: JaatoSession._provider_for_tier(s, pn, m)
    s._is_connected_to = lambda e: JaatoSession._is_connected_to(s, e)
    s._connect_tier_entry = lambda e: JaatoSession._connect_tier_entry(s, e)
    return s


_CROSS = {
    "executor": {"model": "glm-4.6", "provider": "zhipuai"},
    "vision": {"model": "google/gemini-2.5-flash-lite", "provider": "openrouter"},
    RESERVED_INITIAL_KEY: "executor",
    RESERVED_FALLBACK_KEY: "executor",
}


def test_cross_provider_switch_swaps_and_caches():
    s = _session(_CROSS)
    r = JaatoSession.switch_tier(s, "vision")
    assert r["active_tier"] == "vision"
    assert s._active_provider_name == "openrouter"                     # swapped
    assert s._created == [("openrouter", "google/gemini-2.5-flash-lite")]
    assert "openrouter" in s._provider_cache                           # cached


def test_switch_back_is_cache_hit_no_recreate():
    s = _session(_CROSS)
    JaatoSession.switch_tier(s, "vision")
    JaatoSession.switch_tier(s, "executor")
    assert s._active_provider_name == "zhipuai"
    assert len(s._created) == 1   # reused both cached providers — no 2nd create


def test_same_provider_switch_takes_no_swap():
    s = _session({
        "executor": {"model": "glm-4.6", "provider": "zhipuai"},
        "planner": {"model": "glm-4.5", "provider": "zhipuai"},
        RESERVED_INITIAL_KEY: "executor",
        RESERVED_FALLBACK_KEY: "executor",
    })
    JaatoSession.switch_tier(s, "planner")
    assert s._active_provider_name == "zhipuai"
    assert s._created == []        # same provider → no new provider created


def test_tier_provider_gets_plugin_configs_despite_cleared_lazy_pending():
    # Regression for the #354 cross-provider tier bug: _provider_lazy_pending is
    # cleared to None once the main provider is created, but a tier switch must
    # STILL receive plugin_configs (the tier provider's api_key) — from the
    # persistent _tier_provider_base.  Pre-fix it read the cleared lazy_pending
    # and passed plugin_configs=None → the openrouter tier built with no api_key
    # → "No OpenRouter API key found".
    s = _session(_CROSS)
    assert s._provider_lazy_pending is None                  # the cleared state
    JaatoSession.switch_tier(s, "vision")
    assert s._cfgs == [{"openrouter": {"api_key": "K-openrouter"}}]   # NOT None
