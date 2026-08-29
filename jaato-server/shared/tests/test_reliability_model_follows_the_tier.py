"""Reliability records name the model that was actually running.

THE GAP (closed here).  ``PatternDetector`` stamps ``_model_name`` into
every ``BehavioralPattern`` it emits.  That name is captured ONCE, when
the detector is constructed, from ``ReliabilityPlugin._current_model``.
``set_session_context`` is called without a model name, and
``PatternDetector.set_model_name`` had no caller anywhere.

In a ``model_tiers`` session the model changes mid-run.  So a repetition
loop or a prerequisite violation detected while the session was on the
executor tier was filed under the model that STARTED the session, and the
record could not answer the first question anyone asks of it: which model
was misbehaving.  Attribution silently wrong is worse than absent —
absent prompts a question, wrong ends one.

Same shape as §5.4 (a measurement that cannot see a tier switch), in a
different subsystem.  See ``docs/design/model-tier-prompt-cache.md`` §5.5.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from shared.jaato_session import JaatoSession
from shared.model_tiers import ModelTierConfig, TierEntry
from shared.plugins.reliability.plugin import ReliabilityPlugin
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back: the plugin stops telling its own detector.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/plugins/reliability/plugin.py",
        find="""        if self._pattern_detector:
            self._pattern_detector.set_model_name(current_model)""",
        replace="""        pass""",
        test="test_the_plugin_forwards_the_model_to_its_detector",
        because="patterns filed under the model that started the session",
    ),
]


def _plugin_with_detector(model='claude-sonnet-4-5'):
    plugin = ReliabilityPlugin()
    plugin.set_model_context(model)
    plugin.enable_pattern_detection(True)
    assert plugin._pattern_detector is not None
    assert plugin._pattern_detector._model_name == model
    return plugin


class TestThePluginForwards:

    def test_the_plugin_forwards_the_model_to_its_detector(self):
        """The crux: the detector is the thing that stamps the record."""
        plugin = _plugin_with_detector()

        plugin.set_model_context('claude-haiku-4-5')

        assert plugin._pattern_detector._model_name == 'claude-haiku-4-5'

    def test_available_models_are_not_disturbed(self):
        """A tier change does not alter the switchable-model catalogue, and
        the session re-targets without re-supplying it."""
        plugin = _plugin_with_detector()
        plugin.set_model_context('claude-sonnet-4-5', ['a', 'b'])

        plugin.set_model_context('claude-haiku-4-5')

        assert plugin._available_models == ['a', 'b']

    def test_no_detector_is_not_an_error(self):
        """Pattern detection is opt-in; the plugin must not require it."""
        plugin = ReliabilityPlugin()
        plugin.set_model_context('claude-haiku-4-5')      # must not raise
        assert plugin._current_model == 'claude-haiku-4-5'


class TestTheSessionRetargets:

    @staticmethod
    def _session(plugin, tiers, initial):
        s = JaatoSession.__new__(JaatoSession)
        s._tier_config = ModelTierConfig(
            tiers=tiers, initial_tier=initial, tier_fallback=initial)
        s._active_tier = initial
        s._tier_switch_count = 0
        s._model_name = tiers[initial].model
        s._active_provider_name = 'anthropic'
        s._provider = MagicMock()
        s._provider.name = 'anthropic'
        s._provider.model_name = tiers[initial].model

        def _connect(m, **kw):
            s._provider.model_name = m

        s._provider.connect = _connect
        s._runtime = MagicMock()
        s._runtime.reliability_plugin = plugin
        s._wire_cache_plugin = lambda: None
        s._trace = lambda *a, **k: None
        return s

    def test_enter_tier_retargets(self):
        plugin = _plugin_with_detector()
        s = self._session(plugin, {
            'dispatcher': TierEntry(model='claude-sonnet-4-5'),
            'executor': TierEntry(model='claude-haiku-4-5'),
        }, 'dispatcher')

        s.switch_tier('executor')

        assert plugin._pattern_detector._model_name == 'claude-haiku-4-5'

    def test_a_budget_rebind_retargets(self):
        """The tier NAME never changes on this path."""
        plugin = _plugin_with_detector('claude-opus-4-7')
        s = self._session(
            plugin, {'planner': TierEntry(model='claude-opus-4-7')}, 'planner')

        s._tier_config.tiers['planner'] = TierEntry(model='claude-haiku-4-5')
        s._reconnect_active_tier_if_rebound()

        assert plugin._pattern_detector._model_name == 'claude-haiku-4-5'

    def test_a_retarget_failure_does_not_fail_the_switch(self):
        """Attribution is observability; it must never break the run."""
        plugin = MagicMock()
        plugin.set_model_context.side_effect = RuntimeError('boom')
        s = self._session(plugin, {
            'dispatcher': TierEntry(model='claude-sonnet-4-5'),
            'executor': TierEntry(model='claude-haiku-4-5'),
        }, 'dispatcher')

        result = s.switch_tier('executor')

        assert result['status'] == 'switched'
        assert s._model_name == 'claude-haiku-4-5'

    def test_no_reliability_plugin_is_not_an_error(self):
        s = self._session(None, {
            'dispatcher': TierEntry(model='claude-sonnet-4-5'),
            'executor': TierEntry(model='claude-haiku-4-5'),
        }, 'dispatcher')

        assert s.switch_tier('executor')['status'] == 'switched'
