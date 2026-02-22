"""Tests for CachePlugin protocol compliance."""

import pytest
from shared.plugins.cache.base import CachePlugin


class TestCachePluginProtocol:
    """Verify that implementations satisfy the CachePlugin protocol."""

    def test_anthropic_implements_protocol(self):
        from shared.plugins.cache_anthropic.plugin import AnthropicCachePlugin
        plugin = AnthropicCachePlugin()
        assert isinstance(plugin, CachePlugin)

    def test_zhipuai_implements_protocol(self):
        from shared.plugins.cache_zhipuai.plugin import ZhipuAICachePlugin
        plugin = ZhipuAICachePlugin()
        assert isinstance(plugin, CachePlugin)

    def test_protocol_requires_name(self):
        """Protocol requires a name property."""
        from shared.plugins.cache_anthropic.plugin import AnthropicCachePlugin
        plugin = AnthropicCachePlugin()
        assert plugin.name == "cache_anthropic"

    def test_protocol_requires_provider_name(self):
        """Protocol requires a provider_name property."""
        from shared.plugins.cache_anthropic.plugin import AnthropicCachePlugin
        plugin = AnthropicCachePlugin()
        assert plugin.provider_name == "anthropic"

    def test_protocol_requires_compatible_models(self):
        """Protocol requires a compatible_models property."""
        from shared.plugins.cache_anthropic.plugin import AnthropicCachePlugin
        plugin = AnthropicCachePlugin()
        assert isinstance(plugin.compatible_models, list)
        assert len(plugin.compatible_models) > 0
