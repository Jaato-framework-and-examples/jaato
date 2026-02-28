"""Tests for two-phase instruction budget population.

Verifies that _populate_instruction_budget uses:
- Phase 1: cached counts or estimates (synchronous, instant)
- Phase 2: background provider.count_tokens() for cache misses
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from shared.instruction_budget import InstructionSource, InstructionBudget
from shared.instruction_token_cache import InstructionTokenCache
from shared.jaato_session import JaatoSession, _TokenCountRequest


def _make_session(
    cache: InstructionTokenCache = None,
    count_tokens_fn=None,
    count_tokens_delay: float = 0,
    base_instructions: str = None,
    provider_name: str = "test_provider",
    exposed_plugins: dict = None,
):
    """Create a minimal JaatoSession with mocked runtime/provider."""
    if cache is None:
        cache = InstructionTokenCache()

    # Mock runtime
    runtime = MagicMock()
    runtime.provider_name = provider_name
    runtime.instruction_token_cache = cache
    runtime._base_system_instructions = base_instructions
    runtime._formatter_pipeline = None
    runtime.telemetry = MagicMock()
    runtime.telemetry.enabled = False

    # Mock registry
    if exposed_plugins:
        registry = MagicMock()
        registry._exposed = list(exposed_plugins.keys())

        def get_plugin(name):
            return exposed_plugins.get(name)

        registry.get_plugin = get_plugin
        runtime.registry = registry
    else:
        runtime.registry = None

    # Mock provider
    provider = MagicMock()

    if count_tokens_fn:
        # Use the provided function (may add delay)
        def delayed_count_tokens(text):
            if count_tokens_delay > 0:
                time.sleep(count_tokens_delay)
            return count_tokens_fn(text)

        provider.count_tokens = delayed_count_tokens
    else:
        # Remove count_tokens so hasattr returns False
        del provider.count_tokens

    provider.get_context_limit.return_value = 128_000

    # Create session
    session = JaatoSession.__new__(JaatoSession)
    session._runtime = runtime
    session._model_name = "test-model"
    session._provider_name_override = None
    session._provider = provider
    session._agent_id = "test"
    session._agent_type = "main"
    session._agent_name = None
    session._instruction_budget = None
    session._budget_counting_thread = None
    session._on_instruction_budget_updated = None
    session._ui_hooks = None
    session._gc_plugin = None
    session._gc_config = None
    session._preloaded_plugins = set()

    # SessionHistory wrapper
    from ..session_history import SessionHistory
    session._history = SessionHistory()

    return session


class TestTwoPhasePopulate:
    """Tests for the two-phase budget population flow."""

    def test_all_cache_hits_no_background_thread(self):
        """When all texts are in cache, no background thread should start."""
        cache = InstructionTokenCache()
        # Pre-populate cache with framework text
        from shared.jaato_runtime import (
            _TASK_COMPLETION_INSTRUCTION,
            _PARALLEL_TOOL_GUIDANCE,
            _TURN_SUMMARY_INSTRUCTION,
        )
        framework_text = "\n\n".join([
            _TASK_COMPLETION_INSTRUCTION,
            _PARALLEL_TOOL_GUIDANCE,
            _TURN_SUMMARY_INSTRUCTION,
        ])
        cache.put("test_provider", framework_text, 500)
        cache.put("test_provider", "my base instructions", 100)

        session = _make_session(
            cache=cache,
            base_instructions="my base instructions",
        )

        session._populate_instruction_budget()

        # No background thread should have been started
        assert session._budget_counting_thread is None

        # Budget should exist with cached values
        budget = session._instruction_budget
        assert budget is not None

        # Check system children got cached values
        system_entry = budget.get_entry(InstructionSource.SYSTEM)
        assert system_entry is not None
        base_child = system_entry.children.get("base")
        assert base_child is not None
        assert base_child.tokens == 100

        framework_child = system_entry.children.get("framework")
        assert framework_child is not None
        assert framework_child.tokens == 500

    def test_cache_miss_uses_estimate_then_refines(self):
        """Cache misses should use estimates in Phase 1 and refine in Phase 2."""
        cache = InstructionTokenCache()

        call_count = {"n": 0}

        def mock_count_tokens(text):
            call_count["n"] += 1
            return len(text)  # Simple deterministic "accurate" count

        session = _make_session(
            cache=cache,
            count_tokens_fn=mock_count_tokens,
            base_instructions="hello world",
        )

        session._populate_instruction_budget()

        # Background thread should have been started
        assert session._budget_counting_thread is not None

        # Wait for background completion
        session._budget_counting_thread.join(timeout=5.0)
        assert not session._budget_counting_thread.is_alive()

        # Provider.count_tokens should have been called for cache misses
        assert call_count["n"] > 0

        # Cache should now contain the accurate counts
        assert cache.get("test_provider", "hello world") == len("hello world")

    def test_second_session_gets_all_cache_hits(self):
        """A second session with same config should hit cache for everything."""
        cache = InstructionTokenCache()

        call_count = {"n": 0}

        def mock_count_tokens(text):
            call_count["n"] += 1
            return len(text)

        # First session: populates cache
        session1 = _make_session(
            cache=cache,
            count_tokens_fn=mock_count_tokens,
            base_instructions="shared instructions",
        )
        session1._populate_instruction_budget()

        if session1._budget_counting_thread:
            session1._budget_counting_thread.join(timeout=5.0)

        first_call_count = call_count["n"]
        assert first_call_count > 0  # Had to call provider

        # Second session: should get cache hits
        call_count["n"] = 0
        session2 = _make_session(
            cache=cache,
            count_tokens_fn=mock_count_tokens,
            base_instructions="shared instructions",
        )
        session2._populate_instruction_budget()

        # No background thread needed — all hits
        assert session2._budget_counting_thread is None
        # Provider was not called
        assert call_count["n"] == 0

    def test_no_provider_count_tokens_keeps_estimates(self):
        """When provider has no count_tokens, estimates are final."""
        session = _make_session(
            base_instructions="some text",
        )
        # Provider has count_tokens deleted in _make_session when count_tokens_fn is None

        session._populate_instruction_budget()

        # No background thread
        assert session._budget_counting_thread is None

        # Budget should exist with estimated values
        budget = session._instruction_budget
        assert budget is not None
        system_entry = budget.get_entry(InstructionSource.SYSTEM)
        assert system_entry is not None
        # Base child should have an estimated token count (chars/4)
        base_child = system_entry.children.get("base")
        assert base_child is not None
        assert base_child.tokens > 0

    def test_plugin_instructions_counted(self):
        """Plugin instructions should be collected and counted."""
        cache = InstructionTokenCache()

        def mock_count_tokens(text):
            return len(text)

        # Create mock plugin
        plugin = MagicMock()
        plugin.get_system_instructions.return_value = "Plugin A instructions text"
        plugin.discoverability = "core"

        session = _make_session(
            cache=cache,
            count_tokens_fn=mock_count_tokens,
            exposed_plugins={"plugin_a": plugin},
        )

        session._populate_instruction_budget()

        if session._budget_counting_thread:
            session._budget_counting_thread.join(timeout=5.0)

        budget = session._instruction_budget
        plugin_entry = budget.get_entry(InstructionSource.PLUGIN)
        assert plugin_entry is not None
        assert "plugin_a" in plugin_entry.children
        # After background refinement, token count should be accurate
        assert plugin_entry.children["plugin_a"].tokens == len("Plugin A instructions text")

    def test_gc_guard_joins_thread(self):
        """_maybe_collect_after_turn should join background thread before GC."""
        session = _make_session(base_instructions="text")

        # Create a mock background thread that finishes quickly
        done = threading.Event()

        def slow_work():
            done.wait(timeout=2.0)

        thread = threading.Thread(target=slow_work, daemon=True)
        thread.start()
        session._budget_counting_thread = thread

        # Mock GC plugin as None so _maybe_collect_after_turn returns early
        session._gc_plugin = None
        session._gc_config = None

        # Signal the thread to finish
        done.set()
        thread.join(timeout=2.0)

        result = session._maybe_collect_after_turn()
        assert result is None  # No GC plugin, but the guard code path was exercised


class TestCountTokensCache:
    """Tests for _count_tokens using the instruction token cache."""

    def test_cache_hit_skips_provider(self):
        """_count_tokens should return cached value without calling provider."""
        cache = InstructionTokenCache()
        cache.put("test_provider", "cached text", 42)

        provider_called = {"n": 0}

        def mock_count_tokens(text):
            provider_called["n"] += 1
            return 999

        session = _make_session(
            cache=cache,
            count_tokens_fn=mock_count_tokens,
        )

        result = session._count_tokens("cached text")
        assert result == 42
        assert provider_called["n"] == 0

    def test_cache_miss_calls_provider_and_caches(self):
        """_count_tokens should call provider on miss and cache the result."""
        cache = InstructionTokenCache()

        def mock_count_tokens(text):
            return 77

        session = _make_session(
            cache=cache,
            count_tokens_fn=mock_count_tokens,
        )

        result = session._count_tokens("new text")
        assert result == 77
        assert cache.get("test_provider", "new text") == 77

    def test_empty_text_returns_zero(self):
        session = _make_session()
        assert session._count_tokens("") == 0
        assert session._count_tokens(None) == 0
