"""Anthropic Claude provider plugin.

This provider enables access to Claude models through the Anthropic API.

Features:
- Claude 3.5, Claude 4, and Claude Opus 4.5 model families
- Function/tool calling
- Extended thinking (reasoning traces)
- Prompt caching for cost optimization
- Real token counting via API

Usage:
    from shared.plugins.model_provider.anthropic import AnthropicProvider

    provider = AnthropicProvider()
    provider.initialize(ProviderConfig(api_key='sk-ant-...'))
    provider.connect('claude-sonnet-4-20250514')
    response = provider.send_message("Hello!")
"""

from .provider import AnthropicProvider, create_provider

__all__ = ["AnthropicProvider", "create_provider"]
