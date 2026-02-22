"""Tests for GoogleGenAICachePlugin."""

import pytest
from unittest.mock import MagicMock

from shared.plugins.cache_google_genai.plugin import (
    GoogleGenAICachePlugin,
    create_plugin,
)


class TestCreatePlugin:
    def test_factory_returns_plugin(self):
        plugin = create_plugin()
        assert isinstance(plugin, GoogleGenAICachePlugin)

    def test_factory_returns_new_instance(self):
        p1 = create_plugin()
        p2 = create_plugin()
        assert p1 is not p2


class TestProperties:
    def test_name(self):
        plugin = GoogleGenAICachePlugin()
        assert plugin.name == "cache_google_genai"

    def test_provider_name(self):
        plugin = GoogleGenAICachePlugin()
        assert plugin.provider_name == "google_genai"

    def test_compatible_models(self):
        plugin = GoogleGenAICachePlugin()
        models = plugin.compatible_models
        assert any("gemini" in m for m in models)

    def test_compatible_models_cover_gemini_families(self):
        """Verify all major Gemini model families are covered."""
        plugin = GoogleGenAICachePlugin()
        models = plugin.compatible_models
        # Should cover 1.5, 2.0, 2.5, and 3.x
        assert any("1.5" in m for m in models)
        assert any("2.0" in m for m in models)
        assert any("2.5" in m for m in models)
        assert any("3" in m for m in models)


class TestInitializeShutdown:
    def test_initialize_ignores_config(self):
        """Google GenAI plugin ignores config (monitoring-only)."""
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"whatever": True})
        # Should not raise

    def test_initialize_none_config(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize(None)

    def test_initialize_no_args(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

    def test_shutdown_is_noop(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()
        plugin.shutdown()


class TestPrepareRequestPassthrough:
    """Google GenAI plugin should never modify request data."""

    def test_passthrough_system(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        system = [{"type": "text", "text": "System prompt"}]
        result = plugin.prepare_request(system, [], [])

        assert result["system"] is system  # Same reference (not copied)

    def test_passthrough_tools(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        tools = [{"name": "a", "input_schema": {}}]
        result = plugin.prepare_request([], tools, [])

        assert result["tools"] is tools

    def test_passthrough_messages(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        messages = [{"role": "user", "content": "Hi"}]
        result = plugin.prepare_request([], [], messages)

        assert result["messages"] is messages

    def test_no_cache_breakpoint(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        result = plugin.prepare_request([], [], [])
        assert result["cache_breakpoint_index"] == -1

    def test_no_cache_control_added(self):
        """Even with large content, no cache_control should be added."""
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        system = [{"type": "text", "text": "x" * 20000}]
        tools = [{"name": "t", "input_schema": {"type": "object"}}]

        result = plugin.prepare_request(system, tools, [])

        assert "cache_control" not in result["system"][-1]
        assert "cache_control" not in result["tools"][-1]

    def test_returns_all_required_keys(self):
        """Verify the returned dict has all expected keys."""
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        result = plugin.prepare_request("sys", [], [])

        assert "system" in result
        assert "tools" in result
        assert "messages" in result
        assert "cache_breakpoint_index" in result


class TestExtractCacheUsage:
    def test_tracks_cache_read_tokens(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        usage = MagicMock()
        usage.cache_read_tokens = 250

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 250

    def test_accumulates_across_calls(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        for _ in range(4):
            usage = MagicMock()
            usage.cache_read_tokens = 100
            plugin.extract_cache_usage(usage)

        assert plugin.total_cache_read_tokens == 400

    def test_ignores_none(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        usage = MagicMock()
        usage.cache_read_tokens = None

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 0

    def test_ignores_zero(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        usage = MagicMock()
        usage.cache_read_tokens = 0

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 0

    def test_ignores_negative(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        usage = MagicMock()
        usage.cache_read_tokens = -1

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 0

    def test_initial_value_is_zero(self):
        plugin = GoogleGenAICachePlugin()
        assert plugin.total_cache_read_tokens == 0


class TestOnGCResult:
    def test_gc_with_freed_tokens_increments_count(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        result = MagicMock()
        result.tokens_freed = 500

        plugin.on_gc_result(result)
        assert plugin.gc_invalidation_count == 1

    def test_gc_without_freed_tokens_no_increment(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        result = MagicMock()
        result.tokens_freed = 0

        plugin.on_gc_result(result)
        assert plugin.gc_invalidation_count == 0

    def test_count_accumulates(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        for _ in range(3):
            result = MagicMock()
            result.tokens_freed = 100
            plugin.on_gc_result(result)

        assert plugin.gc_invalidation_count == 3

    def test_initial_value_is_zero(self):
        plugin = GoogleGenAICachePlugin()
        assert plugin.gc_invalidation_count == 0


class TestSetBudget:
    def test_stores_budget_reference(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        budget = MagicMock()
        plugin.set_budget(budget)
        assert plugin._budget is budget

    def test_replaces_previous_budget(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        budget1 = MagicMock()
        budget2 = MagicMock()
        plugin.set_budget(budget1)
        plugin.set_budget(budget2)
        assert plugin._budget is budget2

    def test_initial_budget_is_none(self):
        plugin = GoogleGenAICachePlugin()
        assert plugin._budget is None


class TestProtocolCompliance:
    """Verify the plugin satisfies the CachePlugin protocol."""

    def test_has_all_protocol_methods(self):
        """Ensure all CachePlugin protocol methods exist."""
        plugin = GoogleGenAICachePlugin()
        assert hasattr(plugin, "name")
        assert hasattr(plugin, "provider_name")
        assert hasattr(plugin, "compatible_models")
        assert hasattr(plugin, "initialize")
        assert hasattr(plugin, "shutdown")
        assert hasattr(plugin, "set_budget")
        assert hasattr(plugin, "prepare_request")
        assert hasattr(plugin, "extract_cache_usage")
        assert hasattr(plugin, "on_gc_result")

    def test_isinstance_check(self):
        """Verify runtime_checkable protocol compliance."""
        from shared.plugins.cache.base import CachePlugin

        plugin = GoogleGenAICachePlugin()
        assert isinstance(plugin, CachePlugin)
