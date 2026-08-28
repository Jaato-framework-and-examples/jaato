"""The cache plugin follows the session's model tier.

THE GAP (closed here).  ``_wire_cache_plugin`` ran exactly once per
session, from ``_ensure_provider``.  ``enter_tier`` re-points the session
at a different (provider, model) via ``_connect_tier_entry`` and nothing
re-ran the wiring, so:

  * a CROSS-PROVIDER tier swapped ``self._provider`` to an instance that
    had never had a cache plugin attached — caching silently off for that
    tier, and for the rest of the session; and
  * a SAME-PROVIDER tier left the plugin's model name pinned to whatever
    booted the session, so model-dependent policy (Anthropic's
    minimum-cacheable-size threshold, Google's ``CachedContent`` model
    binding) was decided for the wrong model.
    ``AnthropicCachePlugin.set_model_name`` existed with no caller at all.

Both paths into ``_connect_tier_entry`` are covered: model-driven
(``switch_tier`` ← the ``enter_tier`` tool) and framework-driven
(``_reconnect_active_tier_if_rebound`` ← a budget-control degrade rung
rebinding the active tier's model in place).

See ``docs/design/model-tier-prompt-cache.md`` §5.2.
"""

from unittest.mock import MagicMock

from shared.jaato_session import JaatoSession
from shared.model_tiers import ModelTierConfig, TierEntry
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back: the tier switch stops re-wiring the cache plugin.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""        try:
            self._wire_cache_plugin()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "tier cache re-wire for %s/%s failed; continuing uncached: %s",
                self._active_provider_name, entry.model, exc,
            )""",
        replace="""        return""",
        test="test_a_cross_provider_tier_gets_a_cache_plugin",
        because="a cross-provider tier running with no cache plugin at all",
    ),
]


class _FakeCachePlugin:
    """Minimal CachePlugin stand-in that records what it was told."""

    def __init__(self, name):
        self.name = name
        self.model_names = []
        self.budgets = []

    def set_model_name(self, model_name):
        self.model_names.append(model_name)

    def set_budget(self, budget):
        self.budgets.append(budget)


def _provider(name, model):
    p = MagicMock()
    p.name = name
    p.model_name = model

    def _connect(m, **kwargs):
        p.model_name = m

    p.connect = _connect
    return p


def _session(monkeypatch, *, tiers, initial, plugin_configs=None,
             cache_providers=('anthropic', 'google_genai')):
    """A session in tier mode, with cache-plugin discovery stubbed.

    ``cache_providers`` names the providers for which a cache plugin
    exists; anything else resolves to ``None``, the way openrouter (which
    caches internally) does in production.
    """
    import shared.plugins.cache as cache_mod

    built = {}

    def _loader(provider_name, config):
        if provider_name not in cache_providers:
            return None
        plugin = _FakeCachePlugin(f'cache_{provider_name}')
        plugin.init_config = config
        built.setdefault(provider_name, []).append(plugin)
        return plugin

    monkeypatch.setattr(cache_mod, 'load_cache_plugin_for_provider', _loader)

    runtime = MagicMock()
    runtime._provider_config = MagicMock()
    runtime._provider_config.extra = {}

    initial_entry = tiers[initial]
    provider_name = initial_entry.provider or 'anthropic'

    s = JaatoSession.__new__(JaatoSession)
    s._runtime = runtime
    s._provider = _provider(provider_name, initial_entry.model)
    s._active_provider_name = provider_name
    s._model_name = initial_entry.model
    s._provider_cache = {provider_name: s._provider}
    s._cache_plugin = None
    s._cache_plugins_by_provider = {}
    s._instruction_budget = None
    s._tier_provider_base = {'skip_model_test': True,
                             'plugin_configs': plugin_configs}
    s._provider_lazy_pending = None
    s._tier_config = ModelTierConfig(
        tiers=tiers, initial_tier=initial, tier_fallback=initial)
    s._active_tier = initial
    s._agent_type = 'main'
    s._agent_name = None
    s._agent_id = 'test'
    s._trace = lambda *a, **k: None

    # Cross-provider tiers build their provider through the runtime.
    def _create_provider(model, provider_name=None, **kwargs):
        return _provider(provider_name, model)

    runtime.create_provider = _create_provider

    s._wire_cache_plugin()   # what _ensure_provider does at boot
    s._built = built
    return s


class TestCrossProviderTier:

    def test_a_cross_provider_tier_gets_a_cache_plugin(self, monkeypatch):
        """The crux: switching provider must not silently disable caching."""
        s = _session(monkeypatch, tiers={
            'dispatcher': TierEntry(model='claude-sonnet-4-5',
                                    provider='anthropic'),
            'planner': TierEntry(model='gemini-3-pro',
                                 provider='google_genai'),
        }, initial='dispatcher')
        assert s._cache_plugin.name == 'cache_anthropic'

        s.switch_tier('planner')

        assert s._cache_plugin is not None, (
            'the cross-provider tier is running with no cache plugin'
        )
        assert s._cache_plugin.name == 'cache_google_genai'
        s._provider.set_cache_plugin.assert_called_with(s._cache_plugin)

    def test_switching_back_reuses_the_same_plugin_instance(self, monkeypatch):
        """Per-provider caching: a switch back is O(1) and keeps state.

        Re-discovering would scan entry points on every hop and would
        also reset the plugin's accumulated cache metrics.
        """
        s = _session(monkeypatch, tiers={
            'dispatcher': TierEntry(model='claude-sonnet-4-5',
                                    provider='anthropic'),
            'planner': TierEntry(model='gemini-3-pro',
                                 provider='google_genai'),
        }, initial='dispatcher')
        first = s._cache_plugin

        s.switch_tier('planner')
        s.switch_tier('dispatcher')

        assert s._cache_plugin is first
        assert len(s._built['anthropic']) == 1, (
            'the anthropic cache plugin was rebuilt instead of reused'
        )

    def test_a_provider_without_one_clears_the_slot(self, monkeypatch):
        """openrouter caches internally, so no plugin — and the previous
        provider's must not stay attached.  ``_cache_plugin`` drives budget
        forwarding, usage extraction and telemetry; a stale one would book
        the new provider's cache traffic against the old provider."""
        s = _session(monkeypatch, tiers={
            'dispatcher': TierEntry(model='claude-sonnet-4-5',
                                    provider='anthropic'),
            'executor': TierEntry(model='meta-llama/llama-3.3-70b-instruct',
                                  provider='openrouter'),
        }, initial='dispatcher')
        assert s._cache_plugin is not None

        s.switch_tier('executor')
        assert s._cache_plugin is None

        # ...and switching back restores it.
        s.switch_tier('dispatcher')
        assert s._cache_plugin.name == 'cache_anthropic'

    def test_each_provider_reads_its_own_plugin_configs_section(self, monkeypatch):
        s = _session(monkeypatch, tiers={
            'dispatcher': TierEntry(model='claude-sonnet-4-5',
                                    provider='anthropic'),
            'planner': TierEntry(model='gemini-3-pro',
                                 provider='google_genai'),
        }, initial='dispatcher', plugin_configs={
            'anthropic': {'cache_ttl': '1h'},
            'google_genai': {'cache_ttl': '3600s'},
        })
        assert s._cache_plugin.init_config['cache_ttl'] == '1h'

        s.switch_tier('planner')
        assert s._cache_plugin.init_config['cache_ttl'] == '3600s'


class TestSameProviderTier:

    def test_the_plugin_is_told_the_new_model(self, monkeypatch):
        """A same-provider hop changes only the model — which is exactly
        what model-dependent cache policy keys on."""
        s = _session(monkeypatch, tiers={
            'dispatcher': TierEntry(model='claude-sonnet-4-5'),
            'executor': TierEntry(model='claude-haiku-4-5'),
        }, initial='dispatcher')
        plugin = s._cache_plugin
        assert plugin.model_names[-1] == 'claude-sonnet-4-5'

        s.switch_tier('executor')

        assert plugin.model_names[-1] == 'claude-haiku-4-5', (
            "the cache plugin is still deciding for the boot model"
        )
        assert s._cache_plugin is plugin

    def test_a_budget_degrade_rebind_also_re_wires(self, monkeypatch):
        """The framework-driven path.  A degrade rung rebinds the ACTIVE
        tier's model in place; the tier name never changes, so only the
        re-wire inside ``_connect_tier_entry`` catches it."""
        s = _session(monkeypatch, tiers={
            'planner': TierEntry(model='claude-opus-4-7'),
        }, initial='planner')
        plugin = s._cache_plugin
        assert plugin.model_names[-1] == 'claude-opus-4-7'

        # What overlay_tier_table does to a rung's model_tiers overlay.
        s._tier_config.tiers['planner'] = TierEntry(model='claude-haiku-4-5')
        s._reconnect_active_tier_if_rebound()

        assert plugin.model_names[-1] == 'claude-haiku-4-5'


class TestFailureIsNotFatal:

    def test_a_re_wire_failure_leaves_the_switch_done(self, monkeypatch):
        """Running uncached beats failing the tier switch.  The connect
        itself still raises — a session pointed at the wrong model is not
        something to continue from — but the cache is an optimisation."""
        s = _session(monkeypatch, tiers={
            'dispatcher': TierEntry(model='claude-sonnet-4-5'),
            'executor': TierEntry(model='claude-haiku-4-5'),
        }, initial='dispatcher')

        def _boom():
            raise RuntimeError('entry points unreadable')

        s._wire_cache_plugin = _boom

        result = s.switch_tier('executor')

        assert result['status'] == 'switched'
        assert result['model'] == 'claude-haiku-4-5'
        assert s._model_name == 'claude-haiku-4-5'
