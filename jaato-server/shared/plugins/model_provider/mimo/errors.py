"""Error types for the Xiaomi MiMo provider.

These exceptions wrap underlying SDK/API errors with actionable guidance
for users to resolve authentication and configuration issues.  The
transient ones (``RateLimitError``, ``InfrastructureError``) are what
``classify_error`` on the shared base retries; everything else stops the
retry loop.
"""

from typing import List, Optional


class MiMoError(Exception):
    """Base class for Xiaomi MiMo provider errors."""
    pass


class APIKeyNotFoundError(MiMoError):
    """No API key could be located.

    Raised when the provider cannot find an API key and the endpoint is
    not a self-hosted proxy.
    """

    def __init__(self, checked_locations: Optional[List[str]] = None):
        self.checked_locations = checked_locations or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["No Xiaomi MiMo API key found.", ""]
        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")
        lines.extend([
            "To authenticate:",
            "  1. Generate an API key at https://platform.xiaomimimo.com/#/console/api-keys",
            "  2. Set JAATO_MIMO_API_KEY=<your-key> (or MIMO_API_KEY),",
            "     or run 'mimo-auth key <your-key>'",
        ])
        return "\n".join(lines)


class AuthenticationError(MiMoError):
    """API key was rejected by the Xiaomi MiMo API (HTTP 401)."""

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Xiaomi MiMo API key is invalid or was rejected."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Verify the key at https://platform.xiaomimimo.com/#/console/api-keys",
            "  2. Check that the key matches the endpoint — One host serves every region; keys and balances are per region.",
            "  3. Regenerate the key if needed",
        ])
        return "\n".join(lines)


class RateLimitError(MiMoError):
    """Rate limit exceeded for the Xiaomi MiMo API (HTTP 429).

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
        lines = ["Xiaomi MiMo rate limit exceeded."]
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


class QuotaExhaustedError(MiMoError):
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
        lines = ["Xiaomi MiMo account quota exhausted (balance or plan window)."]
        if self.resets_at:
            lines.append(f"Resets at: {self.resets_at}")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Top up the account at https://platform.xiaomimimo.com/#/console/api-keys, or wait for the plan window",
            "  2. Retrying before then will not help — the request is refused, not queued",
        ])
        return "\n".join(lines)


class ModelNotFoundError(MiMoError):
    """Requested model is not available on Xiaomi MiMo (HTTP 404)."""

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
            "  1. Check the model ID (e.g. 'mimo-v2.5-pro')",
            "  2. Browse the catalog at https://mimo.mi.com/docs/en-US/ or list models with the",
            "     provider's list_models()",
            "  3. Retired model ids answer 404 — pick a current one",
        ])
        return "\n".join(lines)


class ContextLimitError(MiMoError):
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
            "  3. Set JAATO_MIMO_CONTEXT_LENGTH to the model's actual limit",
        ])
        return "\n".join(lines)


class InfrastructureError(MiMoError):
    """Transient infrastructure error from the Xiaomi MiMo API (5xx / network)."""

    def __init__(self, status_code: int = 0, original_error: Optional[str] = None):
        self.status_code = status_code
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.status_code == 0:
            lines = ["Xiaomi MiMo network error."]
        else:
            lines = [f"Xiaomi MiMo infrastructure error (HTTP {self.status_code})."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])
        return "\n".join(lines)


class RegionDeniedError(MiMoError):
    """The API refused the key or the caller's region (HTTP 403).

    NOT transient.  MiMo is not available in the EU, the UK or Korea, and
    a Token Plan key (``tp-...``) used against the pay-as-you-go host is
    refused the same way.
    """

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["MiMo refused the request (HTTP 403: region or key restriction)."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "MiMo is not available in the EU, the UK or Korea.  A Token Plan key",
            "(tp-...) only works against the plan hosts —",
            "https://token-plan-{cn,sgp,ams}.xiaomimimo.com/v1 — set",
            "JAATO_MIMO_BASE_URL accordingly.",
        ])
        return "\n".join(lines)


class ContentFilteredError(MiMoError):
    """The request or the response was blocked by MiMo's content filter
    (the vendor's HTTP 421).  NOT transient: the same request is blocked
    again."""

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["MiMo content filter blocked the request (HTTP 421)."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend(["", "Rephrase the prompt; retrying the same request will not help."])
        return "\n".join(lines)
