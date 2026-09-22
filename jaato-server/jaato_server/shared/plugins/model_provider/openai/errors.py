"""Error types for the native OpenAI provider.

These exceptions wrap underlying SDK/API errors with actionable guidance
for users to resolve authentication and configuration issues.  Their
constructor signatures are the ones ``OpenAICompatProvider._handle_api_error``
calls with — the base's mapping is parameterized by the ``_ERR_*`` class
attributes, so the taxonomy is per-provider and the mapping is shared.
"""

from typing import List, Optional


class OpenAIError(Exception):
    """Base class for native-OpenAI provider errors."""
    pass


class APIKeyNotFoundError(OpenAIError):
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
            "No OpenAI API key found.",
            "",
        ]

        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")

        lines.extend([
            "To authenticate:",
            "  1. Create an API key at https://platform.openai.com/api-keys",
            "  2. Set JAATO_OPENAI_API_KEY=<your-key> (OPENAI_API_KEY also works)",
            "  3. Or put it in a profile as plugin_configs.openai.api_key,",
            "     where it may be a pass:// or vault:// URI rather than the",
            "     literal secret",
            "  4. Or write ~/.jaato/openai_auth.json as",
            "     {\"api_key\": \"sk-...\"} with mode 0600",
        ])

        return "\n".join(lines)


class AuthenticationError(OpenAIError):
    """API key was rejected by the OpenAI API.

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
        lines = ["OpenAI API key is invalid or was rejected."]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Verify the key at https://platform.openai.com/api-keys",
            "  2. Check that the key has not been revoked or rotated",
            "  3. If the key is project-scoped, check that the organization",
            "     and project headers match it (plugin_configs.openai.",
            "     organization / project)",
        ])

        return "\n".join(lines)


class RateLimitError(OpenAIError):
    """Rate limit or quota exceeded for the OpenAI API."""

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
        lines = ["OpenAI rate limit exceeded."]

        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Wait for the retry period to elapse (jaato retries these)",
            "  2. Check the usage limits for your organization at",
            "     https://platform.openai.com/settings/organization/limits",
            "  3. A 429 that never clears is usually 'insufficient_quota' —",
            "     a billing state, not a rate, and no amount of retrying",
            "     will clear it",
        ])

        return "\n".join(lines)


class ModelNotFoundError(OpenAIError):
    """Requested model is not available to this key."""

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
            "  1. Check the model id (e.g. 'gpt-5.1', 'gpt-4.1-mini')",
            "  2. List what this key can reach with the provider's",
            "     list_models(), or at https://platform.openai.com/docs/models",
            "  3. Model access is per-organization — a model your account",
            "     can use in the playground may not be enabled for this",
            "     project's key",
        ])

        return "\n".join(lines)


class ContextLimitError(OpenAIError):
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
            "  3. Check that plugin_configs.openai.context_length (or",
            "     JAATO_OPENAI_CONTEXT_LENGTH) matches the model's real",
            "     window — GC sizes the history against that number, so a",
            "     value set too high is exactly how this error is reached",
        ])

        return "\n".join(lines)


class InfrastructureError(OpenAIError):
    """Transient infrastructure error from the OpenAI API.

    Raised for 5xx responses and network errors.  These are retriable and
    are classified as such by the provider's ``classify_error``.
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
            lines = ["OpenAI network error."]
        else:
            lines = [
                f"OpenAI infrastructure error (HTTP {self.status_code})."
            ]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])

        return "\n".join(lines)
