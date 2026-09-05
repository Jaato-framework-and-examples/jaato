"""Error types for the Helmcode provider.

These exceptions wrap underlying SDK/API errors with actionable guidance
for users to resolve authentication and configuration issues.

Helmcode's taxonomy carries one class the other OpenAI-compatible
providers have no analogue for: :class:`CreditsExhaustedError`, the
``402 credits_exhausted`` that only the resold frontier models can raise.
It is deliberately *not* an :class:`InfrastructureError` — see that
class's docstring for why retrying it is wrong.
"""

from typing import List, Optional


class HelmcodeError(Exception):
    """Base class for Helmcode provider errors."""
    pass


class APIKeyNotFoundError(HelmcodeError):
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
            "No Helmcode API key found.",
            "",
        ]

        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")

        lines.extend([
            "To authenticate:",
            "  1. Create an API key in the Helmcode dashboard "
            "(API Keys -> Create key).",
            "     Keys are issued per workspace, not per person.",
            "  2. Set JAATO_HELMCODE_API_KEY=<your-key> (or HELMCODE_API_KEY,",
            "     the variable Helmcode's own docs use)",
            "  3. Or run 'helmcode-auth key <your-key>' to store it",
        ])

        return "\n".join(lines)


class AuthenticationError(HelmcodeError):
    """API key was rejected by the Helmcode API.

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
        lines = ["Helmcode API key is invalid or was rejected."]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Verify the key in the Helmcode dashboard, under API Keys",
            "  2. Check that the key has not been revoked — keys belong to",
            "     the workspace and any member can revoke them",
            "  3. Create a new key if needed (plans cap how many exist at",
            "     once: 5 on Starter, 15 on Growth, 40 on Scale)",
        ])

        return "\n".join(lines)


class RateLimitError(HelmcodeError):
    """Rate limit exceeded for the Helmcode API.

    Raised when too many requests have been made in a short period.
    Helmcode's limits are per API key — requests per minute, concurrency
    and tokens per minute — and are separate from the monthly volume the
    plan covers.
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
        lines = ["Helmcode rate limit exceeded."]

        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Wait for the retry period to elapse (the Retry-After",
            "     header carries it; the framework already backs off)",
            "  2. Limits are per API key — RPM, concurrency and TPM.  Spread",
            "     a fan-out across several keys, or lower concurrency",
            "  3. Higher RPM and pooled concurrency are available on",
            "     request; Dedicated plans run on reserved hardware",
        ])

        return "\n".join(lines)


class CreditsExhaustedError(HelmcodeError):
    """Prepaid credit ran out for a resold frontier model (HTTP 402).

    Helmcode serves two families through one API and one key.  The
    open-weight models it runs itself are covered by the monthly plan at a
    flat rate; the resold frontier models (Anthropic, OpenAI and Google —
    ``claude-*``, ``gpt-5.6-*``, ``gemini-*``) are billed per token out of
    prepaid credit instead.  When that balance reaches zero, only those
    models refuse, with ``402 credits_exhausted``; everything the plan
    covers keeps answering.

    This is why the class exists rather than folding into
    :class:`InfrastructureError`: a 402 is a *permanent* refusal for the
    chosen model until a human tops the balance up.  Retrying it burns the
    backoff budget and cannot succeed, so :meth:`classify_error` leaves it
    unclassified (non-transient) and the turn fails fast with this
    message.  Switching to a plan-covered model is the in-process remedy.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        model = self.model or "unknown"
        lines = [
            f"Helmcode prepaid credit exhausted for model: {model}",
            "",
            "This model is one of the frontier models Helmcode resells "
            "(Anthropic, OpenAI, Google).",
            "Those are billed per token from prepaid credit and are not "
            "covered by any monthly plan.",
        ]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Top up the credit balance in the Helmcode console, "
            "under Credits",
            "  2. Or switch to a model your plan covers — the open-weight "
            "models Helmcode",
            "     runs itself (e.g. deepseek-v4-flash, qwen3.6, gemma4) "
            "keep answering at a",
            "     flat rate and are unaffected by the balance",
            "",
            "Retrying without one of those will fail again: the balance "
            "does not refill on its own.",
        ])

        return "\n".join(lines)


class ModelNotFoundError(HelmcodeError):
    """Requested model is not available on Helmcode.

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
            "  1. Check the model ID.  Helmcode ids are bare, not",
            "     vendor-prefixed: 'qwen3.6', 'gemma4', 'deepseek-v4-flash',",
            "     'glm-5.3'",
            "  2. Browse the catalog at https://helmcode.com/docs/models,",
            "     or list models with the provider's list_models()",
            "  3. Some ids need an entitlement the key may not carry —",
            "     GLM 5.3 is a per-key add-on, and the resold frontier",
            "     models need prepaid credit",
        ])

        return "\n".join(lines)


class ContextLimitError(HelmcodeError):
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
            "  3. Set JAATO_HELMCODE_CONTEXT_LENGTH (or",
            "     plugin_configs.helmcode.context_length) to the model's",
            "     actual limit — the per-model windows are published at",
            "     https://helmcode.com/docs/models",
        ])

        return "\n".join(lines)


class InfrastructureError(HelmcodeError):
    """Transient infrastructure error from the Helmcode API.

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
            lines = ["Helmcode network error."]
        else:
            lines = [
                f"Helmcode infrastructure error "
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
