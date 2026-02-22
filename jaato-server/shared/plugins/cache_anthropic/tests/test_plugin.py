"""Tests for AnthropicCachePlugin."""

import pytest
from unittest.mock import MagicMock, patch

from shared.plugins.cache_anthropic.plugin import (
    AnthropicCachePlugin,
    create_plugin,
    CACHE_MIN_TOKENS_SONNET,
    CACHE_MIN_TOKENS_OTHER,
    DEFAULT_CACHE_EXCLUDE_RECENT_TURNS,
)


class TestCreatePlugin:
    def test_factory_returns_plugin(self):
        plugin = create_plugin()
        assert isinstance(plugin, AnthropicCachePlugin)

    def test_factory_returns_new_instance(self):
        p1 = create_plugin()
        p2 = create_plugin()
        assert p1 is not p2


class TestInitialize:
    def test_default_config(self):
        """With no config, resolves from env (defaults to disabled)."""
        plugin = AnthropicCachePlugin()
        with patch('shared.plugins.model_provider.anthropic.env.resolve_enable_caching', return_value=False):
            plugin.initialize()
        assert not plugin._enabled
        assert plugin._cache_history is True
        assert plugin._cache_exclude_recent_turns == DEFAULT_CACHE_EXCLUDE_RECENT_TURNS

    def test_explicit_enable(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})
        assert plugin._enabled is True

    def test_config_passthrough(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({
            "enable_caching": True,
            "cache_ttl": "1h",
            "cache_history": False,
            "cache_exclude_recent_turns": 5,
            "model_name": "claude-sonnet-4-20250514",
        })
        assert plugin._enabled is True
        assert plugin._cache_ttl == "1h"
        assert plugin._cache_history is False
        assert plugin._cache_exclude_recent_turns == 5
        assert plugin._model_name == "claude-sonnet-4-20250514"


class TestProperties:
    def test_name(self):
        plugin = AnthropicCachePlugin()
        assert plugin.name == "cache_anthropic"

    def test_provider_name(self):
        plugin = AnthropicCachePlugin()
        assert plugin.provider_name == "anthropic"

    def test_compatible_models(self):
        plugin = AnthropicCachePlugin()
        models = plugin.compatible_models
        assert any("sonnet" in m for m in models)
        assert any("opus" in m for m in models)


class TestPrepareRequestDisabled:
    """When caching is disabled, prepare_request is a passthrough."""

    def test_passthrough_when_disabled(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": False})

        system = [{"type": "text", "text": "System prompt"}]
        tools = [{"name": "tool_a", "input_schema": {}}]
        messages = [{"role": "user", "content": "Hello"}]

        result = plugin.prepare_request(system, tools, messages)

        assert result["system"] == system
        assert result["tools"] == tools
        assert result["messages"] == messages
        assert result["cache_breakpoint_index"] == -1

    def test_no_cache_control_added_when_disabled(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": False})

        system = [{"type": "text", "text": "x" * 10000}]
        tools = [{"name": "t", "input_schema": {"type": "object"}}]

        result = plugin.prepare_request(system, tools, [])

        # No cache_control should be present
        assert "cache_control" not in result["system"][-1]
        assert "cache_control" not in result["tools"][-1]


class TestPrepareRequestEnabled:
    """When caching is enabled, cache_control annotations are injected."""

    def _make_plugin(self, **config):
        plugin = AnthropicCachePlugin()
        config.setdefault("enable_caching", True)
        config.setdefault("cache_min_tokens", False)  # Disable threshold for testing
        plugin.initialize(config)
        return plugin

    def test_system_breakpoint_injected(self):
        plugin = self._make_plugin()
        system = [{"type": "text", "text": "System prompt text"}]

        result = plugin.prepare_request(system, [], [])

        assert "cache_control" in result["system"][-1]
        assert result["system"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_tools_sorted_by_name(self):
        plugin = self._make_plugin()
        tools = [
            {"name": "z_tool", "input_schema": {}},
            {"name": "a_tool", "input_schema": {}},
            {"name": "m_tool", "input_schema": {}},
        ]

        result = plugin.prepare_request([], tools, [])

        names = [t["name"] for t in result["tools"]]
        assert names == ["a_tool", "m_tool", "z_tool"]

    def test_tool_breakpoint_on_last_tool(self):
        plugin = self._make_plugin()
        tools = [
            {"name": "a_tool", "input_schema": {}},
            {"name": "b_tool", "input_schema": {}},
        ]

        result = plugin.prepare_request([], tools, [])

        # cache_control on last tool only
        assert "cache_control" not in result["tools"][0]
        assert "cache_control" in result["tools"][-1]

    def test_empty_system_not_annotated(self):
        plugin = self._make_plugin()
        result = plugin.prepare_request(None, [], [])
        assert result["system"] is None

    def test_empty_tools_not_annotated(self):
        plugin = self._make_plugin()
        result = plugin.prepare_request([], [], [])
        assert result["tools"] == []

    def test_does_not_mutate_input(self):
        plugin = self._make_plugin()
        original_system = [{"type": "text", "text": "Hello"}]
        original_tools = [{"name": "a", "input_schema": {}}]

        plugin.prepare_request(original_system, original_tools, [])

        # Originals should NOT have cache_control
        assert "cache_control" not in original_system[0]
        assert "cache_control" not in original_tools[0]


class TestThresholdChecking:
    def test_below_threshold_skips_annotation(self):
        """Content below min tokens threshold should not get cache_control."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({
            "enable_caching": True,
            "cache_min_tokens": True,  # Enforce threshold
        })

        # Short content (well below 2048 token threshold)
        system = [{"type": "text", "text": "Short text"}]
        result = plugin.prepare_request(system, [], [])

        # Should NOT have cache_control (too small)
        assert "cache_control" not in result["system"][-1]

    def test_above_threshold_gets_annotation(self):
        """Content above min tokens threshold should get cache_control."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({
            "enable_caching": True,
            "cache_min_tokens": True,
        })

        # Large content (well above 2048 token threshold)
        system = [{"type": "text", "text": "x" * 20000}]
        result = plugin.prepare_request(system, [], [])

        assert "cache_control" in result["system"][-1]

    def test_sonnet_lower_threshold(self):
        """Sonnet models have a lower minimum threshold."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({
            "enable_caching": True,
            "cache_min_tokens": True,
            "model_name": "claude-3-5-sonnet-20241022",
        })

        assert plugin._get_cache_min_tokens() == CACHE_MIN_TOKENS_SONNET

    def test_non_sonnet_higher_threshold(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({
            "enable_caching": True,
            "cache_min_tokens": True,
            "model_name": "claude-sonnet-4-20250514",
        })

        assert plugin._get_cache_min_tokens() == CACHE_MIN_TOKENS_OTHER


class TestExtractCacheUsage:
    def test_tracks_read_tokens(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        usage = MagicMock()
        usage.cache_read_tokens = 500
        usage.cache_creation_tokens = None

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 500

    def test_tracks_creation_tokens(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        usage = MagicMock()
        usage.cache_read_tokens = None
        usage.cache_creation_tokens = 1000

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_creation_tokens == 1000

    def test_accumulates_across_calls(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        for i in range(3):
            usage = MagicMock()
            usage.cache_read_tokens = 100
            usage.cache_creation_tokens = 50
            plugin.extract_cache_usage(usage)

        assert plugin.total_cache_read_tokens == 300
        assert plugin.total_cache_creation_tokens == 150

    def test_ignores_zero_values(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        usage = MagicMock()
        usage.cache_read_tokens = 0
        usage.cache_creation_tokens = 0

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 0
        assert plugin.total_cache_creation_tokens == 0


class TestOnGCResult:
    def test_preservable_removal_flags_invalidation(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        result = MagicMock()
        result.details = {"preservable_removed": 3}

        plugin.on_gc_result(result)
        assert plugin.prefix_invalidated is True
        assert plugin.gc_invalidation_count == 1

    def test_no_preservable_removal_no_invalidation(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        result = MagicMock()
        result.details = {"preservable_removed": 0}

        plugin.on_gc_result(result)
        assert plugin.prefix_invalidated is False
        assert plugin.gc_invalidation_count == 0

    def test_invalidation_count_accumulates(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        for _ in range(3):
            result = MagicMock()
            result.details = {"preservable_removed": 1}
            plugin.on_gc_result(result)

        assert plugin.gc_invalidation_count == 3

    def test_prefix_invalidated_cleared_by_prepare_request(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True, "cache_min_tokens": False})

        # Trigger invalidation
        result = MagicMock()
        result.details = {"preservable_removed": 1}
        plugin.on_gc_result(result)
        assert plugin.prefix_invalidated is True

        # prepare_request should clear it
        plugin.prepare_request(
            [{"type": "text", "text": "sys"}], [], []
        )
        assert plugin.prefix_invalidated is False


class TestSetBudget:
    def test_stores_budget_reference(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        budget = MagicMock()
        plugin.set_budget(budget)
        assert plugin._budget is budget

    def test_budget_update_replaces_reference(self):
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        budget1 = MagicMock()
        budget2 = MagicMock()

        plugin.set_budget(budget1)
        plugin.set_budget(budget2)
        assert plugin._budget is budget2


class TestExtendedTTL:
    """Tests for extended 1-hour cache TTL support."""

    def test_default_ttl_is_ephemeral(self):
        """Default 5m TTL produces plain ephemeral cache_control."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True, "cache_min_tokens": False})

        result = plugin.prepare_request(
            [{"type": "text", "text": "Hello"}], [], []
        )

        assert result["system"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_extended_ttl_includes_ttl_field(self):
        """1h TTL produces cache_control with ttl field."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({
            "enable_caching": True,
            "cache_ttl": "1h",
            "cache_min_tokens": False,
        })

        result = plugin.prepare_request(
            [{"type": "text", "text": "Hello"}], [], []
        )

        assert result["system"][-1]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    def test_extended_ttl_on_tools(self):
        """1h TTL should also apply to tool breakpoints."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({
            "enable_caching": True,
            "cache_ttl": "1h",
            "cache_min_tokens": False,
        })

        tools = [{"name": "my_tool", "input_schema": {}}]
        result = plugin.prepare_request([], tools, [])

        assert result["tools"][-1]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    def test_requires_extended_cache_beta_false_by_default(self):
        """Default 5m TTL should not require the beta header."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        assert plugin.requires_extended_cache_beta is False

    def test_requires_extended_cache_beta_true_for_1h(self):
        """1h TTL requires the extended-cache-ttl beta header."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True, "cache_ttl": "1h"})

        assert plugin.requires_extended_cache_beta is True

    def test_make_cache_control_5m(self):
        """_make_cache_control for default TTL."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})

        assert plugin._make_cache_control() == {"type": "ephemeral"}

    def test_make_cache_control_1h(self):
        """_make_cache_control for extended TTL."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True, "cache_ttl": "1h"})

        assert plugin._make_cache_control() == {"type": "ephemeral", "ttl": "1h"}


class TestShutdown:
    def test_shutdown_is_noop(self):
        """Shutdown should not raise any errors."""
        plugin = AnthropicCachePlugin()
        plugin.initialize({"enable_caching": True})
        plugin.shutdown()  # Should not raise
