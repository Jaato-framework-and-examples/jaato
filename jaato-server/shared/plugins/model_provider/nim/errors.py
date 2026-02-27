"""Error types for the NVIDIA NIM provider.

These exceptions wrap underlying SDK/API errors with actionable guidance
for users to resolve authentication and configuration issues.
"""

from typing import List, Optional


class NIMError(Exception):
    """Base class for NIM provider errors."""
    pass


class APIKeyNotFoundError(NIMError):
    """No API key could be located.

    Raised when the provider cannot find an API key and the endpoint
    is not a self-hosted NIM instance.
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
            "No NVIDIA NIM API key found.",
            "",
        ]

        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")

        lines.extend([
            "To authenticate:",
            "  1. Get an API key from https://build.nvidia.com/",
            "  2. Set JAATO_NIM_API_KEY=nvapi-...",
            "",
            "For self-hosted NIM containers (no API key needed):",
            "  Set JAATO_NIM_BASE_URL=http://localhost:8000/v1",
        ])

        return "\n".join(lines)


class AuthenticationError(NIMError):
    """API key was rejected by the NIM endpoint.

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
        lines = ["NVIDIA NIM API key is invalid or was rejected."]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Verify your API key at https://build.nvidia.com/",
            "  2. Check that the key has not expired",
            "  3. Regenerate the key if needed",
        ])

        return "\n".join(lines)


class RateLimitError(NIMError):
    """Rate limit exceeded for NIM API.

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
        lines = ["NVIDIA NIM rate limit exceeded."]

        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Wait for the retry period to elapse",
            "  2. Consider using a self-hosted NIM container for unlimited throughput",
        ])

        return "\n".join(lines)


class ModelNotFoundError(NIMError):
    """Requested model is not available on the NIM endpoint.

    Raised when the specified model ID doesn't exist or isn't
    available.
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
            "  1. Check the model ID format (e.g., 'meta/llama-3.1-70b-instruct')",
            "  2. Browse available models at https://build.nvidia.com/",
            "  3. For self-hosted NIM, verify the model is loaded in the container",
        ])

        return "\n".join(lines)


class ContextLimitError(NIMError):
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
            "  3. Set JAATO_NIM_CONTEXT_LENGTH to the model's actual limit",
        ])

        return "\n".join(lines)


class InfrastructureError(NIMError):
    """Transient infrastructure error from NIM API.

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
            lines = ["NIM API network error."]
        else:
            lines = [f"NIM API infrastructure error (HTTP {self.status_code})."]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])

        return "\n".join(lines)
