"""Regression: a profile's cache knobs must reach the cache plugin.

Root cause closed here (2026-08-28): ``_wire_cache_plugin`` built the
cache plugin's config from ``runtime._provider_config.extra`` alone.
That object is assigned exactly once —
``ProviderConfig(project=..., location=...)`` in ``JaatoRuntime.connect``
— with an empty ``extra`` that nothing ever writes to.  The profile's
``plugin_configs[<provider>]`` merge happens inside
``JaatoRuntime.create_provider``, via ``dataclasses.replace`` into a
LOCAL config that is never stored back.

So every cache plugin was handed ``{}``:

  * ``plugin_configs.anthropic.enable_caching: true`` did nothing —
    Anthropic caching survived only because
    ``AnthropicCachePlugin.initialize`` falls back to the
    ``JAATO_ANTHROPIC_ENABLE_CACHING`` env var when the key is absent.
  * ``cache_ttl`` / ``cache_history`` / ``cache_min_tokens`` had no route
    at all.
  * Google's explicit ``CachedContent`` path (a hard ``False`` default,
    no env fallback) was unreachable from anywhere.

The session now reproduces the ``ProviderConfig.extra`` the ACTIVE
provider was built with, so cache knobs travel the same route as every
other provider knob.
"""

from unittest.mock import MagicMock

from shared.jaato_session import JaatoSession


def _session(
    *,
    runtime_extra=None,
    plugin_configs=None,
    active_provider_name="anthropic",
    provider_dot_name="anthropic",
    model_name="claude-sonnet-4-5",
):
    """A session with just enough wiring for ``_cache_plugin_config``."""
    runtime = MagicMock()
    runtime._provider_config = MagicMock()
    runtime._provider_config.extra = dict(runtime_extra or {})

    provider = MagicMock()
    provider.name = provider_dot_name
    provider.model_name = model_name

    s = JaatoSession.__new__(JaatoSession)
    s._runtime = runtime
    s._provider = provider
    s._active_provider_name = active_provider_name
    s._tier_provider_base = {
        'skip_model_test': True,
        'plugin_configs': plugin_configs,
    }
    s._provider_lazy_pending = None
    s._instruction_budget = None
    s._cache_plugin = None
    s._trace = lambda *a, **k: None
    return s


class TestCachePluginConfig:

    def test_profile_knobs_reach_the_plugin_config(self):
        """The crux: knobs under ``plugin_configs.<provider>`` are present."""
        s = _session(plugin_configs={
            'anthropic': {
                'enable_caching': True,
                'cache_ttl': '1h',
                'cache_history': False,
            },
        })
        cfg = s._cache_plugin_config()
        assert cfg['enable_caching'] is True
        assert cfg['cache_ttl'] == '1h'
        assert cfg['cache_history'] is False

    def test_runtime_extra_is_the_base_layer(self):
        """Runtime-level extras still apply when no profile overrides them."""
        s = _session(runtime_extra={'cache_ttl': '5m'}, plugin_configs={})
        assert s._cache_plugin_config()['cache_ttl'] == '5m'

    def test_profile_wins_over_runtime_extra(self):
        """Child-wins, matching ``create_provider``'s own merge order."""
        s = _session(
            runtime_extra={'cache_ttl': '5m'},
            plugin_configs={'anthropic': {'cache_ttl': '1h'}},
        )
        assert s._cache_plugin_config()['cache_ttl'] == '1h'

    def test_other_providers_sections_are_ignored(self):
        """Only the ACTIVE provider's section is read."""
        s = _session(
            active_provider_name='anthropic',
            plugin_configs={
                'anthropic': {'enable_caching': True},
                'google_genai': {'enable_caching': False, 'cache_ttl': '99s'},
            },
        )
        cfg = s._cache_plugin_config()
        assert cfg['enable_caching'] is True
        assert 'cache_ttl' not in cfg

    def test_lookup_uses_creation_name_not_provider_dot_name(self):
        """``provider.name`` is not the profile key.

        zhipuai subclasses the Anthropic provider and reports the parent's
        name, so only the name the provider was CREATED under selects the
        right ``plugin_configs`` section.
        """
        s = _session(
            active_provider_name='zhipuai',
            provider_dot_name='anthropic',
            plugin_configs={
                'zhipuai': {'enable_caching': True},
                'anthropic': {'enable_caching': False},
            },
        )
        assert s._cache_plugin_config()['enable_caching'] is True

    def test_api_key_is_not_handed_to_the_cache_plugin(self):
        """``create_provider`` promotes ``api_key`` out of ``extra``.

        Mirroring that keeps the config an honest view of what the
        provider sees, and keeps a credential out of a plugin that has
        no use for it.  ``oauth_token`` is NOT promoted there, so it
        stays — the rule is "match the provider", not "guess at secrets".
        """
        s = _session(plugin_configs={
            'anthropic': {
                'api_key': 'sk-ant-secret',
                'oauth_token': 'sk-ant-oat01-x',
                'enable_caching': True,
            },
        })
        cfg = s._cache_plugin_config()
        assert 'api_key' not in cfg
        assert cfg['oauth_token'] == 'sk-ant-oat01-x'

    def test_no_plugin_configs_is_not_an_error(self):
        """Single-model sessions with no profile knobs still wire up."""
        s = _session(plugin_configs=None)
        assert s._cache_plugin_config() == {}

    def test_falls_back_to_lazy_pending_before_configure_stashes_base(self):
        """``_provider_lazy_pending`` carries the same dict pre-configure."""
        s = _session(plugin_configs=None)
        s._tier_provider_base = None
        s._provider_lazy_pending = {
            'plugin_configs': {'anthropic': {'enable_caching': True}},
        }
        assert s._cache_plugin_config()['enable_caching'] is True


class TestWireCachePlugin:

    def test_wiring_initializes_the_plugin_with_profile_knobs(self, monkeypatch):
        """End-to-end through ``_wire_cache_plugin``: the knobs a profile
        declares are what ``CachePlugin.initialize`` receives, alongside
        the ``model_name`` the session adds for threshold selection."""
        seen = {}

        class _FakeCachePlugin:
            name = 'cache_fake'

            def set_budget(self, budget):
                seen['budget'] = budget

        def _loader(provider_name, config):
            seen['provider_name'] = provider_name
            seen['config'] = config
            return _FakeCachePlugin()

        import shared.plugins.cache as cache_mod
        monkeypatch.setattr(
            cache_mod, 'load_cache_plugin_for_provider', _loader)

        s = _session(plugin_configs={
            'anthropic': {'enable_caching': True, 'cache_ttl': '1h'},
        })
        s._wire_cache_plugin()

        assert seen['provider_name'] == 'anthropic'
        assert seen['config']['enable_caching'] is True
        assert seen['config']['cache_ttl'] == '1h'
        assert seen['config']['model_name'] == 'claude-sonnet-4-5'
        s._provider.set_cache_plugin.assert_called_once()
        assert s._cache_plugin.name == 'cache_fake'
