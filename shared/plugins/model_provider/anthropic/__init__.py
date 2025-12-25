"""Anthropic Claude provider plugin.

This provider enables access to Claude models through the Anthropic API.

Features:
- Claude 3.5, Claude 4, and Claude Opus 4.5 model families
- Function/tool calling
- Extended thinking (reasoning traces)
- Prompt caching for cost optimization
- Real token counting via API
- OAuth PKCE flow for Claude Pro/Max subscription

Usage:
    from shared.plugins.model_provider.anthropic import AnthropicProvider

    # Option 1: API key (uses API credits)
    provider = AnthropicProvider()
    provider.initialize(ProviderConfig(api_key='sk-ant-api03-...'))

    # Option 2: OAuth login (uses Claude Pro/Max subscription)
    from shared.plugins.model_provider.anthropic import oauth_login
    oauth_login()  # Opens browser for auth
    provider = AnthropicProvider()
    provider.initialize(ProviderConfig())  # Uses stored tokens

    provider.connect('claude-sonnet-4-20250514')
    response = provider.send_message("Hello!")
"""

from .errors import (
    AnthropicProviderError,
    APIKeyInvalidError,
    APIKeyNotFoundError,
    ContextLimitError,
    ModelNotFoundError,
    OverloadedError,
    RateLimitError,
    UsageLimitError,
)
from .oauth import (
    login as oauth_login,
    clear_tokens as oauth_clear_tokens,
    load_tokens as oauth_load_tokens,
)
from .provider import AnthropicProvider, create_provider

__all__ = [
    "AnthropicProvider",
    "create_provider",
    # OAuth
    "oauth_login",
    "oauth_clear_tokens",
    "oauth_load_tokens",
    # Errors
    "AnthropicProviderError",
    "APIKeyInvalidError",
    "APIKeyNotFoundError",
    "ContextLimitError",
    "ModelNotFoundError",
    "OverloadedError",
    "RateLimitError",
    "UsageLimitError",
]
