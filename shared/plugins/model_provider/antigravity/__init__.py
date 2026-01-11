"""Antigravity provider plugin.

This provider enables access to AI models through Google's Antigravity
backend using Google OAuth authentication.

Features:
- Gemini 3 and Claude model families via Antigravity quota
- Gemini 2.5 models via Gemini CLI quota
- Google OAuth authentication with PKCE
- Multi-account rotation for load balancing
- Streaming support with SSE
- Extended thinking support

Usage:
    from shared.plugins.model_provider.antigravity import AntigravityProvider

    # First, authenticate (one-time)
    from shared.plugins.model_provider.antigravity import oauth_login
    oauth_login()  # Opens browser for Google auth

    # Then use the provider
    provider = AntigravityProvider()
    provider.initialize()  # Uses stored OAuth tokens
    provider.connect('antigravity-gemini-3-flash')
    response = provider.send_message("Hello!")

Available Models:

    Antigravity Quota (via Google's IDE backend):
    - antigravity-gemini-3-pro (thinking levels: low, high)
    - antigravity-gemini-3-flash (thinking levels: minimal, low, medium, high)
    - antigravity-claude-sonnet-4-5
    - antigravity-claude-sonnet-4-5-thinking
    - antigravity-claude-opus-4-5-thinking

    Gemini CLI Quota:
    - gemini-2.5-flash
    - gemini-2.5-pro
    - gemini-3-flash-preview
    - gemini-3-pro-preview

Environment Variables:
    JAATO_ANTIGRAVITY_PROJECT_ID: Override project ID
    JAATO_ANTIGRAVITY_ENDPOINT: Override API endpoint
    JAATO_ANTIGRAVITY_QUOTA: Preferred quota type ('antigravity' or 'gemini-cli')
    JAATO_ANTIGRAVITY_THINKING_LEVEL: Default thinking level for Gemini 3
    JAATO_ANTIGRAVITY_THINKING_BUDGET: Thinking budget for Claude models
    JAATO_ANTIGRAVITY_AUTO_ROTATE: Enable account rotation (default: true)
"""

from .errors import (
    AntigravityProviderError,
    APIError,
    AuthenticationError,
    ContextLimitError,
    EndpointError,
    ModelNotFoundError,
    QuotaExceededError,
    RateLimitError,
    StreamingError,
    TokenExpiredError,
    TokenRefreshError,
    ToolResultMissingError,
)
from .oauth import (
    Account,
    AccountManager,
    OAuthTokens,
    clear_accounts,
    complete_interactive_login,
    get_valid_access_token,
    load_accounts,
    login as oauth_login,
    save_accounts,
    start_interactive_login,
)
from .provider import AntigravityProvider, create_provider

__all__ = [
    # Provider
    "AntigravityProvider",
    "create_provider",
    # OAuth
    "oauth_login",
    "start_interactive_login",
    "complete_interactive_login",
    "get_valid_access_token",
    "load_accounts",
    "save_accounts",
    "clear_accounts",
    "OAuthTokens",
    "Account",
    "AccountManager",
    # Errors
    "AntigravityProviderError",
    "AuthenticationError",
    "TokenExpiredError",
    "TokenRefreshError",
    "RateLimitError",
    "QuotaExceededError",
    "ModelNotFoundError",
    "EndpointError",
    "ContextLimitError",
    "APIError",
    "StreamingError",
    "ToolResultMissingError",
]
