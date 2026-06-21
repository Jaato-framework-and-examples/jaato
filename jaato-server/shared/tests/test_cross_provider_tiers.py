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

    def fake_create(model, provider_name=None, skip_model_test=True,
                    plugin_configs=None):
        created.append((provider_name, model))
        return SimpleNamespace(
            name=provider_name,
            connect=lambda m, skip_model_test=True: None,
        )

    s = SimpleNamespace(
        _tier_config=ModelTierConfig.from_unified_dict(tier_dict),
        _active_tier="executor",
        _provider=SimpleNamespace(
            name="zhipuai", connect=lambda m, skip_model_test=True: None),
        _active_provider_name="zhipuai",
        _provider_cache={},
        _provider_lazy_pending={"skip_model_test": True, "plugin_configs": {}},
        _runtime=SimpleNamespace(create_provider=fake_create),
        _model_name="glm-4.6",
        _agent_type=None, _agent_name=None, _agent_id=None,
        _created=created,
    )
    s._provider_cache["zhipuai"] = s._provider
    # Bind the real helper so switch_tier's self._provider_for_tier resolves.
    s._provider_for_tier = lambda pn, m: JaatoSession._provider_for_tier(s, pn, m)
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
