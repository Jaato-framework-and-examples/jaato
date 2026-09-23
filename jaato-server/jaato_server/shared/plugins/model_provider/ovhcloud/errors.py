"""Error types for the OVHcloud AI Endpoints provider.

These exceptions wrap underlying SDK/API errors with actionable guidance
for users to resolve authentication and configuration issues.
"""

from typing import List, Optional


class OVHcloudError(Exception):
    """Base class for OVHcloud provider errors."""
    pass


class APIKeyNotFoundError(OVHcloudError):
    """No API key could be located.

    Raised when the provider cannot find an API key, the keyless free
    tier has not been opted into, and the endpoint is not a self-hosted
    proxy.
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
            "No OVHcloud AI Endpoints API key found.",
            "",
        ]

        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")

        lines.extend([
            "To authenticate:",
            "  1. Create an API key in the OVHcloud Manager:",
            "     Public Cloud → AI & Machine Learning → AI Endpoints → API keys",
            "  2. Set JAATO_OVHCLOUD_API_KEY=<your-key>",
            "     (OVH_AI_ENDPOINTS_ACCESS_TOKEN, the vendor's variable, also works)",
            "",
            "To evaluate without a key (heavily rate-limited free tier):",
            "  Set JAATO_OVHCLOUD_ALLOW_ANONYMOUS=true",
        ])

        return "\n".join(lines)


class AuthenticationError(OVHcloudError):
    """API key was rejected by the OVHcloud AI Endpoints gateway.

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
        lines = ["OVHcloud AI Endpoints API key is invalid or was rejected."]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Verify the key in the OVHcloud Manager:",
            "     Public Cloud → AI & Machine Learning → AI Endpoints → API keys",
            "  2. Check that the key has not been revoked",
            "  3. Regenerate the key if needed",
        ])

        return "\n".join(lines)


class RateLimitError(OVHcloudError):
    """Rate limit exceeded for OVHcloud AI Endpoints.

    Raised when too many requests have been made in a short period.
    Anonymous (keyless) access has a much lower limit than keyed access.
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
        lines = ["OVHcloud AI Endpoints rate limit exceeded."]

        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Wait for the retry period to elapse",
            "  2. If using the anonymous free tier"
            " (JAATO_OVHCLOUD_ALLOW_ANONYMOUS), create an API key —"
            " keyed access has far higher limits",
        ])

        return "\n".join(lines)


class ModelNotFoundError(OVHcloudError):
    """Requested model is not available on OVHcloud AI Endpoints.

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
            "  1. Check the model ID (OVHcloud IDs are case-sensitive,",
            "     e.g. 'gpt-oss-120b', 'Meta-Llama-3_3-70B-Instruct')",
            "  2. Browse the catalog at https://endpoints.ai.cloud.ovh.net/catalog",
            "     or list models with the provider's list_models()",
        ])

        return "\n".join(lines)


class ContextLimitError(OVHcloudError):
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
            "  3. Set JAATO_OVHCLOUD_CONTEXT_LENGTH to the model's actual limit",
        ])

        return "\n".join(lines)


class InfrastructureError(OVHcloudError):
    """Transient infrastructure error from OVHcloud AI Endpoints.

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
            lines = ["OVHcloud AI Endpoints network error."]
        else:
            lines = [
                f"OVHcloud AI Endpoints infrastructure error "
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
