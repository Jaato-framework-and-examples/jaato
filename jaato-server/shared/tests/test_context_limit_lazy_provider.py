"""Regression: the budget-GC ``context_limit`` denominator must reflect
the provider's real context window, not a hardcoded default.

Root cause closed here (2026-06-10): the provider is lazy-created
(``_ensure_provider``, first model use) so ``self._provider`` is ``None``
when ``configure()`` populates the budget.  The old code kept a hardcoded
``128_000`` default in that window and never corrected it once the
provider materialized, so the after-turn budget-GC threshold was computed
against a model-agnostic denominator (e.g. ``128_000`` for a vLLM engine
launched with ``context_length=40944``).  GC stayed dormant and the
session walked into a context-window 400 with GC idle (kb cascade session
20260610_155254: 69633/128000 = 54% measured vs 69633/71680 = 97% real).

Two halves of the fix:
  - ``_populate_instruction_budget``: no provider yet → ``context_limit = 0``
    (an honest "unknown"), NOT a hardcoded ``128_000``.
  - ``_ensure_provider``: the moment the provider materializes, re-read
    ``get_context_limit()`` into the budget — before any conversation
    grows or any after-turn GC check runs.
"""

import threading
from unittest.mock import MagicMock

from shared.tests.test_two_phase_budget import _make_session


def _prep(session):
    session._trace = lambda *a, **k: None
    session._emit_instruction_budget_update = lambda: None


class TestContextLimitFromProvider:

    def test_no_provider_yet_means_zero_not_hardcoded_default(self):
        """Lazy provider: budget created before the provider exists must
        record an honest unknown (0), NOT the old hardcoded 128_000."""
        s = _make_session()
        _prep(s)
        s._provider = None  # lazy — provider not created at configure() time
        s._populate_instruction_budget(session_instructions="persona")
        assert s._instruction_budget.context_limit == 0

    def test_present_provider_limit_is_used(self):
        """When the provider already exists, its real limit is the
        denominator (no hardcoded substitution)."""
        s = _make_session()
        _prep(s)
        s._provider.get_context_limit.return_value = 40944
        s._populate_instruction_budget(session_instructions="persona")
        assert s._instruction_budget.context_limit == 40944

    def test_ensure_provider_corrects_unknown_limit(self):
        """The crux: a budget created with context_limit=0 (provider not
        yet lazy-created) must be corrected to the provider's real window
        the moment ``_ensure_provider`` materializes it."""
        s = _make_session()
        _prep(s)

        # Post-configure state: budget exists with an unknown limit because
        # the provider hasn't been created yet.
        s._provider = None
        s._populate_instruction_budget(session_instructions="persona")
        assert s._instruction_budget.context_limit == 0

        # Wire the lazy-create path: runtime.create_provider returns a
        # provider whose real window is 40944 (the codegen profile value).
        new_provider = MagicMock()
        new_provider.get_context_limit.return_value = 40944
        s._runtime.create_provider.return_value = new_provider
        s._provider_init_lock = threading.Lock()
        s._provider_lazy_pending = {
            'model_name': 'test-model',
            'provider_name': 'test_provider',
            'skip_model_test': True,
            'plugin_configs': {},
        }
        # _wire_cache_plugin pulls optional infra; not under test here.
        s._wire_cache_plugin = lambda: None

        provider = s._ensure_provider()

        assert provider is new_provider
        # Denominator now reflects the real model window, not 0 or 128_000.
        assert s._instruction_budget.context_limit == 40944

    def test_ensure_provider_noop_without_budget(self):
        """If no budget exists yet, _ensure_provider must not raise when
        trying to backfill the limit."""
        s = _make_session()
        _prep(s)
        s._provider = None
        s._instruction_budget = None

        new_provider = MagicMock()
        new_provider.get_context_limit.return_value = 40944
        s._runtime.create_provider.return_value = new_provider
        s._provider_init_lock = threading.Lock()
        s._provider_lazy_pending = {
            'model_name': 'test-model',
            'provider_name': 'test_provider',
            'skip_model_test': True,
            'plugin_configs': {},
        }
        s._wire_cache_plugin = lambda: None

        # Must not raise.
        assert s._ensure_provider() is new_provider
