"""Error types for the Doubleword provider.

These exceptions wrap underlying SDK/API errors with actionable guidance
for users to resolve authentication and configuration issues.
"""

from typing import List, Optional


class DoublewordError(Exception):
    """Base class for Doubleword provider errors."""
    pass


class APIKeyNotFoundError(DoublewordError):
    """No API key could be located.

    Raised when the provider cannot find an API key and the endpoint is
    not a self-hosted proxy.
    """

    def __init__(
        self,
        checked_locations: Optional[List[str]] = None,
    ):
        self.checked_locations = checked_locations or []

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = [
            "No Doubleword API key found.",
            "",
        ]

        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")

        lines.extend([
            "To authenticate:",
            "  1. Generate an API key at https://app.doubleword.ai/api-keys",
            "  2. Set JAATO_DOUBLEWORD_API_KEY=<your-key>",
        ])

        return "\n".join(lines)


class AuthenticationError(DoublewordError):
    """API key was rejected by the Doubleword API.

    Raised when a key exists but fails authentication.
    """

    def __init__(
        self,
        original_error: Optional[str] = None,
    ):
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = ["Doubleword API key is invalid or was rejected."]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Verify the key at https://app.doubleword.ai/api-keys",
            "  2. Check that the key has not been revoked",
            "  3. Regenerate the key if needed",
        ])

        return "\n".join(lines)


class RateLimitError(DoublewordError):
    """Rate limit exceeded for the Doubleword API.

    Raised when too many requests have been made in a short period.
    """

    def __init__(
        self,
        retry_after: Optional[float] = None,
        original_error: Optional[str] = None,
    ):
        self.retry_after = retry_after
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = ["Doubleword rate limit exceeded."]

        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Wait for the retry period to elapse",
            "  2. For sustained high-volume workloads, consider the flex",
            "     tier (api_params.service_tier: flex) or batch jobs —",
            "     both are designed for throughput without realtime limits",
        ])

        return "\n".join(lines)


class ModelNotFoundError(DoublewordError):
    """Requested model is not available on Doubleword.

    Raised when the specified model ID doesn't exist or isn't
    available to the API key.
    """

    def __init__(
        self,
        model: str,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = [f"Model not found: {self.model}"]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Check the model ID (Doubleword uses vendor-prefixed IDs,",
            "     e.g. 'deepseek-ai/DeepSeek-V4-Pro', 'Qwen/Qwen3-30B-A3B-FP8')",
            "  2. Browse the catalog at https://doubleword.ai/models",
            "     or list models with the provider's list_models()",
            "  3. Not every model is available in every tier — check tier",
            "     availability in the catalog when using service_tier",
        ])

        return "\n".join(lines)


class ContextLimitError(DoublewordError):
    """Request exceeds the model's context window.

    Raised when the conversation history + system instructions + prompt
    exceeds the model's maximum token limit.
    """

    def __init__(
        self,
        model: str,
        max_tokens: Optional[int] = None,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = [f"Request too large for model: {self.model}"]

        if self.max_tokens:
            lines.append(f"Maximum tokens: {self.max_tokens}")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Clear conversation history with 'clear' command",
            "  2. Reduce the size of your prompt",
            "  3. Set JAATO_DOUBLEWORD_CONTEXT_LENGTH to the model's actual limit",
        ])

        return "\n".join(lines)


class InfrastructureError(DoublewordError):
    """Transient infrastructure error from the Doubleword API.

    Raised when the API returns a 5xx error or a network error.
    These are typically retriable.
    """

    def __init__(
        self,
        status_code: int = 0,
        original_error: Optional[str] = None,
    ):
        self.status_code = status_code
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        if self.status_code == 0:
            lines = ["Doubleword network error."]
        else:
            lines = [
                f"Doubleword infrastructure error "
                f"(HTTP {self.status_code})."
            ]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])

        return "\n".join(lines)
