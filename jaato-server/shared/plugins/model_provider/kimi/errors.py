"""Error types for the Moonshot AI Kimi provider.

These exceptions wrap underlying SDK/API errors with actionable guidance
for users to resolve authentication and configuration issues.  The
transient ones (``RateLimitError``, ``InfrastructureError``) are what
``classify_error`` on the shared base retries; everything else stops the
retry loop.
"""

from typing import List, Optional


class KimiError(Exception):
    """Base class for Moonshot AI Kimi provider errors."""
    pass


class APIKeyNotFoundError(KimiError):
    """No API key could be located.

    Raised when the provider cannot find an API key and the endpoint is
    not a self-hosted proxy.
    """

    def __init__(self, checked_locations: Optional[List[str]] = None):
        self.checked_locations = checked_locations or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["No Moonshot AI Kimi API key found.", ""]
        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")
        lines.extend([
            "To authenticate:",
            "  1. Generate an API key at https://platform.kimi.ai/console/api-keys",
            "  2. Set JAATO_KIMI_API_KEY=<your-key> (or MOONSHOT_API_KEY),",
            "     or run 'kimi-auth key <your-key>'",
        ])
        return "\n".join(lines)


class AuthenticationError(KimiError):
    """API key was rejected by the Moonshot AI Kimi API (HTTP 401)."""

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Moonshot AI Kimi API key is invalid or was rejected."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Verify the key at https://platform.kimi.ai/console/api-keys",
            "  2. Check that the key matches the endpoint — Keys are issued per platform (.ai vs .cn).",
            "  3. Regenerate the key if needed",
        ])
        return "\n".join(lines)


class RateLimitError(KimiError):
    """Rate limit exceeded for the Moonshot AI Kimi API (HTTP 429).

    Transient: ``with_retry`` backs off and retries, honouring
    ``retry_after`` when the wire named one.
    """

    def __init__(
        self,
        retry_after: Optional[float] = None,
        original_error: Optional[str] = None,
    ):
        self.retry_after = retry_after
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Moonshot AI Kimi rate limit exceeded."]
        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Wait for the retry period to elapse",
            "  2. Reduce request concurrency, or ask the vendor to raise the",
            "     account's limits",
        ])
        return "\n".join(lines)


class QuotaExhaustedError(KimiError):
    """The account cannot pay for the request (balance or plan window).

    NOT transient: retrying cannot help until the account is topped up
    or the plan window resets, so ``classify_error`` leaves this out of
    the retry loop rather than burning the whole backoff budget.
    """

    def __init__(self, original_error: Optional[str] = None, resets_at: Optional[str] = None):
        self.original_error = original_error
        self.resets_at = resets_at
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Moonshot AI Kimi account quota exhausted (balance or plan window)."]
        if self.resets_at:
            lines.append(f"Resets at: {self.resets_at}")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Top up the account at https://platform.kimi.ai/console/api-keys, or wait for the plan window",
            "  2. Retrying before then will not help — the request is refused, not queued",
        ])
        return "\n".join(lines)


class ModelNotFoundError(KimiError):
    """Requested model is not available on Moonshot AI Kimi (HTTP 404)."""

    def __init__(self, model: str, original_error: Optional[str] = None):
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
            "  1. Check the model ID (e.g. 'kimi-k3')",
            "  2. Browse the catalog at https://platform.kimi.ai/docs/ or list models with the",
            "     provider's list_models()",
            "  3. Retired model ids answer 404 — pick a current one",
        ])
        return "\n".join(lines)


class ContextLimitError(KimiError):
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
            "  1. Clear conversation history with 'clear' command",
            "  2. Reduce the size of your prompt",
            "  3. Set JAATO_KIMI_CONTEXT_LENGTH to the model's actual limit",
        ])
        return "\n".join(lines)


class InfrastructureError(KimiError):
    """Transient infrastructure error from the Moonshot AI Kimi API (5xx / network)."""

    def __init__(self, status_code: int = 0, original_error: Optional[str] = None):
        self.status_code = status_code
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.status_code == 0:
            lines = ["Moonshot AI Kimi network error."]
        else:
            lines = [f"Moonshot AI Kimi infrastructure error (HTTP {self.status_code})."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])
        return "\n".join(lines)
