"""Custom exceptions for Anthropic provider.

Provides descriptive error messages for common failure scenarios.
"""

from typing import List, Optional


class AnthropicProviderError(Exception):
    """Base exception for Anthropic provider errors."""

    pass


class APIKeyNotFoundError(AnthropicProviderError):
    """Raised when no API key or OAuth token is found."""

    def __init__(
        self,
        checked_locations: Optional[List[str]] = None,
    ):
        self.checked_locations = checked_locations or []

        locations = ", ".join(self.checked_locations) if self.checked_locations else "environment"
        message = (
            f"Anthropic credentials not found.\n"
            f"Checked: {locations}\n\n"
            f"To fix this, choose one option:\n\n"
            f"  Option 1 - API Key (uses API credits):\n"
            f"    1. Get an API key from https://console.anthropic.com/\n"
            f"    2. Set: export ANTHROPIC_API_KEY='sk-ant-api03-...'\n\n"
            f"  Option 2 - OAuth Token (uses Claude Pro/Max subscription):\n"
            f"    1. Install Claude Code CLI: npm install -g @anthropics/claude-code\n"
            f"    2. Run: claude setup-token\n"
            f"    3. Set: export ANTHROPIC_AUTH_TOKEN='sk-ant-oat01-...'"
        )
        super().__init__(message)


class APIKeyInvalidError(AnthropicProviderError):
    """Raised when the API key is invalid or rejected."""

    def __init__(
        self,
        reason: str = "Token rejected by API",
        key_prefix: Optional[str] = None,
        original_error: Optional[str] = None,
    ):
        self.reason = reason
        self.key_prefix = key_prefix
        self.original_error = original_error

        message = f"Anthropic API key invalid: {reason}"
        if key_prefix:
            message += f"\nKey prefix: {key_prefix}..."
        if original_error:
            message += f"\nOriginal error: {original_error}"

        super().__init__(message)


class RateLimitError(AnthropicProviderError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        retry_after: Optional[int] = None,
        original_error: Optional[str] = None,
    ):
        self.retry_after = retry_after
        self.original_error = original_error

        message = "Anthropic API rate limit exceeded."
        if retry_after:
            message += f" Retry after {retry_after} seconds."
        if original_error:
            message += f"\nOriginal error: {original_error}"

        super().__init__(message)


class ContextLimitError(AnthropicProviderError):
    """Raised when context/token limit is exceeded."""

    def __init__(
        self,
        model: str,
        max_tokens: Optional[int] = None,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.original_error = original_error

        message = f"Context limit exceeded for model {model}."
        if max_tokens:
            message += f" Maximum: {max_tokens:,} tokens."
        message += "\n\nTo fix this, try:\n  1. Reduce conversation history\n  2. Use a model with larger context"
        if original_error:
            message += f"\n\nOriginal error: {original_error}"

        super().__init__(message)


class ModelNotFoundError(AnthropicProviderError):
    """Raised when the requested model is not found."""

    def __init__(
        self,
        model: str,
        available_models: Optional[List[str]] = None,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.available_models = available_models
        self.original_error = original_error

        message = f"Model '{model}' not found."
        if available_models:
            message += f"\n\nAvailable models:\n"
            for m in available_models[:10]:  # Show first 10
                message += f"  - {m}\n"
            if len(available_models) > 10:
                message += f"  ... and {len(available_models) - 10} more"
        if original_error:
            message += f"\n\nOriginal error: {original_error}"

        super().__init__(message)


class OverloadedError(AnthropicProviderError):
    """Raised when Anthropic API is overloaded."""

    def __init__(
        self,
        original_error: Optional[str] = None,
    ):
        self.original_error = original_error

        message = (
            "Anthropic API is currently overloaded.\n"
            "Please try again in a few moments."
        )
        if original_error:
            message += f"\n\nOriginal error: {original_error}"

        super().__init__(message)


class UsageLimitError(AnthropicProviderError):
    """Raised when API usage limit is reached."""

    def __init__(
        self,
        reset_date: Optional[str] = None,
        original_error: Optional[str] = None,
    ):
        self.reset_date = reset_date
        self.original_error = original_error

        message = "Anthropic API usage limit reached."
        if reset_date:
            message += f" Access will be restored on {reset_date}."
        message += (
            "\n\nTo fix this:\n"
            "  1. Check your usage at https://console.anthropic.com/\n"
            "  2. Upgrade your plan or wait for the reset date"
        )
        if original_error:
            message += f"\n\nOriginal error: {original_error}"

        super().__init__(message)
