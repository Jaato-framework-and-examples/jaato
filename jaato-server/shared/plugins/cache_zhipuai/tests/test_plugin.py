"""Tests for ZhipuAICachePlugin."""

import pytest
from unittest.mock import MagicMock

from shared.plugins.cache_zhipuai.plugin import (
    ZhipuAICachePlugin,
    create_plugin,
)


class TestCreatePlugin:
    def test_factory_returns_plugin(self):
        plugin = create_plugin()
        assert isinstance(plugin, ZhipuAICachePlugin)

    def test_factory_returns_new_instance(self):
        p1 = create_plugin()
        p2 = create_plugin()
        assert p1 is not p2


class TestProperties:
    def test_name(self):
        plugin = ZhipuAICachePlugin()
        assert plugin.name == "cache_zhipuai"

    def test_provider_name(self):
        plugin = ZhipuAICachePlugin()
        assert plugin.provider_name == "zhipuai"

    def test_compatible_models(self):
        plugin = ZhipuAICachePlugin()
        models = plugin.compatible_models
        assert any("glm" in m for m in models)


class TestInitializeShutdown:
    def test_initialize_ignores_config(self):
        """ZhipuAI plugin ignores config (monitoring-only)."""
        plugin = ZhipuAICachePlugin()
        plugin.initialize({"whatever": True})
        # Should not raise

    def test_initialize_none_config(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize(None)

    def test_shutdown_is_noop(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()
        plugin.shutdown()


class TestPrepareRequestPassthrough:
    """ZhipuAI plugin should never modify request data."""

    def test_passthrough_system(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        system = [{"type": "text", "text": "System prompt"}]
        result = plugin.prepare_request(system, [], [])

        assert result["system"] is system  # Same reference (not copied)

    def test_passthrough_tools(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        tools = [{"name": "a", "input_schema": {}}]
        result = plugin.prepare_request([], tools, [])

        assert result["tools"] is tools

    def test_passthrough_messages(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        messages = [{"role": "user", "content": "Hi"}]
        result = plugin.prepare_request([], [], messages)

        assert result["messages"] is messages

    def test_no_cache_breakpoint(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        result = plugin.prepare_request([], [], [])
        assert result["cache_breakpoint_index"] == -1

    def test_no_cache_control_added(self):
        """Even with large content, no cache_control should be added."""
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        system = [{"type": "text", "text": "x" * 20000}]
        tools = [{"name": "t", "input_schema": {"type": "object"}}]

        result = plugin.prepare_request(system, tools, [])

        assert "cache_control" not in result["system"][-1]
        assert "cache_control" not in result["tools"][-1]


class TestExtractCacheUsage:
    def test_tracks_cache_read_tokens(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        usage = MagicMock()
        usage.cache_read_tokens = 250

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 250

    def test_accumulates_across_calls(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        for _ in range(4):
            usage = MagicMock()
            usage.cache_read_tokens = 100
            plugin.extract_cache_usage(usage)

        assert plugin.total_cache_read_tokens == 400

    def test_ignores_none(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        usage = MagicMock()
        usage.cache_read_tokens = None

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 0

    def test_ignores_zero(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        usage = MagicMock()
        usage.cache_read_tokens = 0

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 0


class TestOnGCResult:
    def test_gc_with_freed_tokens_increments_count(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        result = MagicMock()
        result.tokens_freed = 500

        plugin.on_gc_result(result)
        assert plugin.gc_invalidation_count == 1

    def test_gc_without_freed_tokens_no_increment(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        result = MagicMock()
        result.tokens_freed = 0

        plugin.on_gc_result(result)
        assert plugin.gc_invalidation_count == 0

    def test_count_accumulates(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        for _ in range(3):
            result = MagicMock()
            result.tokens_freed = 100
            plugin.on_gc_result(result)

        assert plugin.gc_invalidation_count == 3


class TestSetBudget:
    def test_stores_budget_reference(self):
        plugin = ZhipuAICachePlugin()
        plugin.initialize()

        budget = MagicMock()
        plugin.set_budget(budget)
        assert plugin._budget is budget
