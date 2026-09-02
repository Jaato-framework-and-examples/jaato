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

import pytest

from shared.jaato_session import JaatoSession
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back: ``_cache_plugin_config`` reading only the
#: runtime-level base, which is the always-empty ``extra`` assigned once in
#: ``JaatoRuntime.connect``.  Every cache plugin is handed ``{}`` again.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""        config, _promoted_api_key = resolve_provider_extra(
            base_extra, pending.get('plugin_configs'), profile_key)
        return config""",
        replace="""        return dict(base_extra)""",
        test="test_profile_knobs_reach_the_plugin_config",
        because="a profile's cache knobs never reaching the cache plugin",
    ),
]


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
    # ``__init__`` sets this; the helper bypasses it via ``__new__``.
    s._cache_plugins_by_provider = {}
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


class TestTheTwoMergesAgree:
    """The cache plugin's config and the provider's own must not drift.

    ``_cache_plugin_config`` reproduces the merge ``create_provider``
    performs, because the two cannot share a stored result: profile
    ``plugin_configs`` are per-session while ``_provider_configs`` is
    runtime-wide, so writing the merged config back would leak one
    session's knobs into every other session on that provider.

    They share the FUNCTION instead (``resolve_provider_extra``).  This
    class is the executable check that they still do — a second inline
    copy on either side, a newly promoted field, or a changed merge
    order shows up here as a disagreement rather than as a cache plugin
    quietly configured differently from the provider it caches for.
    """

    @staticmethod
    def _provider_extra_from_create_provider(monkeypatch, plugin_configs,
                                             provider_name, base_extra):
        """The ``ProviderConfig.extra`` the REAL create_provider builds."""
        from dataclasses import dataclass, field, replace as _replace
        from typing import Any, Dict, Optional

        import shared.jaato_runtime as runtime_mod

        @dataclass(frozen=True)
        class _Config:
            project: str = ''
            location: str = ''
            api_key: Optional[str] = None
            extra: Dict[str, Any] = field(default_factory=dict)

        captured = {}

        def _fake_load_provider(name, config):
            captured['config'] = config
            prov = MagicMock()
            prov.connect = lambda *a, **k: None
            return prov

        monkeypatch.setattr(runtime_mod, 'load_provider', _fake_load_provider)

        rt = runtime_mod.JaatoRuntime.__new__(runtime_mod.JaatoRuntime)
        rt._connected = True
        rt._project = ''
        rt._location = ''
        rt._provider_name = provider_name
        rt._provider_config = _Config(extra=dict(base_extra))
        rt._provider_configs = {provider_name: _Config(extra=dict(base_extra))}
        rt._registry = None
        rt._config_root = None
        # ``__init__`` sets this; with none set, ``_inject_session_extras``
        # stamps no app identity and the extras stay what the merge made.
        rt._app_identity = None

        rt.create_provider(
            'test-model',
            provider_name=provider_name,
            skip_model_test=True,
            plugin_configs=plugin_configs,
        )
        return captured['config']

    @staticmethod
    def _every_declared_top_level_knob(provider='anthropic'):
        """Every top-level knob the provider DECLARES, each with a sentinel.

        Enumerating cases by hand does not catch the failure this class
        exists for.  ``create_provider`` already promotes ``api_key`` out
        of ``extra``, and that promotion was added late (pre-PR-149 the
        whole dict landed in ``extra``); if a SECOND field is promoted the
        same way, a hand-written case only notices when it happens to
        mention that field.  Verified: sabotaging ``create_provider`` to
        also pop ``oauth_token`` passed a hand-enumerated suite.

        So the case list is derived from ``PROVIDER_KNOBS`` instead. Any
        declared knob that one side consumes and the other does not shows
        up as a disagreement, whichever knob it turns out to be.
        """
        from shared.plugins.model_provider.anthropic import PROVIDER_KNOBS
        layer = PROVIDER_KNOBS.get_layer('top_level')
        if layer is None:
            raise AssertionError(
                f'{provider} declares no top_level layer; this guard reads '
                f'it to build its input and cannot run without it'
            )
        return {name: f'sentinel-{name}' for name in sorted(layer.keys)}

    def test_every_declared_knob_lands_on_both_sides(self, monkeypatch):
        """The drift case: one side consumes a knob, the other does not."""
        section = self._every_declared_top_level_knob()
        assert 'api_key' in section, (
            'expected api_key among the declared knobs — it is the one '
            'field known to be promoted, and the case that proves the '
            'comparison tolerates a legitimate promotion'
        )
        plugin_configs = {'anthropic': section}

        provider_config = self._provider_extra_from_create_provider(
            monkeypatch, plugin_configs, 'anthropic', {})
        s = _session(plugin_configs=plugin_configs)

        assert s._cache_plugin_config() == provider_config.extra, (
            "a declared knob reached the provider's config but not the "
            "cache plugin's (or vice versa) — both must go through "
            "resolve_provider_extra"
        )

    @pytest.mark.parametrize('plugin_configs', [
        {'anthropic': {'enable_caching': True, 'cache_ttl': '1h'}},
        {'anthropic': {'api_key': 'sk-ant-secret', 'enable_caching': True}},
        {'anthropic': {}},
        {'google_genai': {'enable_caching': True}},   # a section we don't use
        {},
        None,
    ])
    def test_the_extras_agree(self, monkeypatch, plugin_configs):
        base_extra = {'workspace_path': '/w', 'cache_ttl': '5m'}
        provider_config = self._provider_extra_from_create_provider(
            monkeypatch, plugin_configs, 'anthropic', base_extra)

        s = _session(runtime_extra=base_extra, plugin_configs=plugin_configs)
        session_view = s._cache_plugin_config()

        assert session_view == provider_config.extra, (
            "the cache plugin's config diverged from the provider's own; "
            "both must go through resolve_provider_extra"
        )

    def test_a_promoted_api_key_is_absent_on_both_sides(self, monkeypatch):
        """``api_key`` is promoted to the field, so neither view carries it."""
        plugin_configs = {'anthropic': {'api_key': 'sk-ant-secret',
                                        'enable_caching': True}}
        provider_config = self._provider_extra_from_create_provider(
            monkeypatch, plugin_configs, 'anthropic', {})
        assert provider_config.api_key == 'sk-ant-secret'
        assert 'api_key' not in provider_config.extra

        s = _session(plugin_configs=plugin_configs)
        assert 'api_key' not in s._cache_plugin_config()
