"""Error types for the OpenRouter provider.

These exceptions wrap the underlying SDK/API errors with actionable
guidance for users to resolve auth and configuration problems.
"""

from typing import List, Optional


class OpenRouterError(Exception):
    """Base class for OpenRouter provider errors."""
    pass


class APIKeyNotFoundError(OpenRouterError):
    """No API key could be located.

    Raised when the provider cannot find a valid OpenRouter API key
    in any of the standard locations.
    """

    def __init__(
        self,
        checked_locations: Optional[List[str]] = None,
    ):
        self.checked_locations = checked_locations or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [
            "No OpenRouter API key found.",
            "",
        ]

        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")

        lines.extend([
            "To authenticate:",
            "  1. Get an API key from https://openrouter.ai/keys",
            "  2. Set JAATO_OPENROUTER_API_KEY=sk-or-...",
            "     or run 'openrouter-auth key sk-or-...' to store it.",
        ])

        return "\n".join(lines)


class AuthenticationError(OpenRouterError):
    """API key was rejected by OpenRouter."""

    def __init__(
        self,
        original_error: Optional[str] = None,
    ):
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["OpenRouter API key is invalid or was rejected."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Verify your API key at https://openrouter.ai/keys",
            "  2. Check that the key has not been revoked",
            "  3. Generate a new key if needed",
        ])
        return "\n".join(lines)


class RateLimitError(OpenRouterError):
    """Rate limit exceeded for OpenRouter."""

    def __init__(
        self,
        retry_after: Optional[float] = None,
        original_error: Optional[str] = None,
    ):
        self.retry_after = retry_after
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["OpenRouter rate limit exceeded."]
        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Wait for the retry period to elapse",
            "  2. Top up your OpenRouter credits at https://openrouter.ai/credits",
            "  3. Switch to a model with higher per-key throughput",
        ])
        return "\n".join(lines)


class ModelNotFoundError(OpenRouterError):
    """Requested model is not available on OpenRouter."""

    def __init__(
        self,
        model: str,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [f"Model not found: {self.model}"]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Check the model ID format (e.g., 'anthropic/claude-3.5-sonnet')",
            "  2. Browse available models at https://openrouter.ai/models",
            "  3. Confirm your account has access to that model",
        ])
        return "\n".join(lines)


class ContextLimitError(OpenRouterError):
    """Request exceeds the model's context window."""

    def __init__(
        self,
        model: str,
        max_tokens: Optional[int] = None,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [f"Request too large for model: {self.model}"]
        if self.max_tokens:
            lines.append(f"Maximum tokens: {self.max_tokens}")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Clear conversation history with 'reset'",
            "  2. Reduce the size of your prompt",
            "  3. Set JAATO_OPENROUTER_CONTEXT_LENGTH to the model's actual limit",
            "  4. Switch to a model with a larger context window",
        ])
        return "\n".join(lines)


class InfrastructureError(OpenRouterError):
    """Transient infrastructure error from OpenRouter (5xx, network)."""

    def __init__(
        self,
        status_code: int = 0,
        original_error: Optional[str] = None,
    ):
        self.status_code = status_code
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.status_code == 0:
            lines = ["OpenRouter network error."]
        else:
            lines = [f"OpenRouter infrastructure error (HTTP {self.status_code})."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])
        return "\n".join(lines)
