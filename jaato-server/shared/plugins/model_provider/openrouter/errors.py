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
            "  1. Get an API key from https://openrouter.ai/settings/keys",
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
            "  1. Verify your API key at https://openrouter.ai/settings/keys",
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


class StallTimeoutError(InfrastructureError):
    """A request made no progress inside the provider's idle deadline.

    Raised when a streaming turn produces no *payload* for
    ``stream_idle_timeout`` seconds — the connection is alive (OpenRouter
    keeps sending ``: OPENROUTER PROCESSING`` SSE comments, so neither the
    socket nor ``httpx``'s read timeout ever notices) but nothing is
    coming back.  Before this existed the provider waited forever and the
    session sat until something *outside* it — a harness arm-timeout, a
    budget ceiling — tore it down (#732).

    Subclasses :class:`InfrastructureError` deliberately: a stall is a
    transient upstream condition, so
    :meth:`OpenRouterProvider.classify_error` already routes it to
    ``with_retry``'s exponential backoff without a second entry.  There
    is no ``Retry-After`` to read on this path — the upstream never
    answered — so the standard backoff applies (#720 handles the case
    where a hint *does* exist).
    """

    def __init__(
        self,
        idle_timeout: float,
        *,
        chunks_received: int = 0,
        generation_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.idle_timeout = idle_timeout
        self.chunks_received = chunks_received
        self.generation_id = generation_id
        self.model = model
        super().__init__(status_code=0, original_error=None)

    def _format_message(self) -> str:
        phase = (
            "before the first content chunk arrived"
            if self.chunks_received == 0
            else f"after {self.chunks_received} content chunk(s)"
        )
        lines = [
            f"OpenRouter stalled: no response payload for "
            f"{self.idle_timeout:g}s ({phase}).",
        ]
        if self.model:
            lines.append(f"Model: {self.model}")
        if self.generation_id:
            lines.append(f"Generation ID: {self.generation_id}")
        lines.extend([
            "",
            "The connection stayed open but the upstream stopped producing.",
            "This is a transient error.",
            "The request will be automatically retried.",
            "",
            "If the model legitimately needs longer to think, raise the",
            "deadline (0 disables it entirely):",
            "  plugin_configs.openrouter.framework_overrides.stream_idle_timeout",
            "  or JAATO_OPENROUTER_STREAM_IDLE_TIMEOUT=<seconds>",
        ])
        return "\n".join(lines)


class UpstreamFinishError(InfrastructureError):
    """A turn ended with ``finish_reason: "error"`` and no error payload.

    OpenRouter reports a mid-stream upstream failure in **two** shapes,
    and only one carries a message (#766):

    1.  A top-level ``error`` object alongside a sentinel choice with
        ``finish_reason: "error"``.  :func:`~.converters.read_chunk_error`
        reads it and the provider raises a plain
        :class:`InfrastructureError` carrying the upstream's own words.
    2.  ``finish_reason: "error"`` **alone** — no ``error`` field
        anywhere on the chunk or in the generation record.  The cause
        lives in the sibling ``native_finish_reason``, the raw word the
        upstream used before OpenRouter normalised it (e.g. Gemini's
        ``MALFORMED_FUNCTION_CALL``: the model emitted a function call
        its own serialiser rejected).

    Shape 2 used to resolve to ``FinishReason.ERROR`` and travel back as
    an ordinary response, where the session turned it into a *terminal*
    ``RuntimeError("Provider returned an error")`` — a diagnosable,
    resampling-shaped failure flattened into an opaque fatal one, while
    the identical upstream condition in shape 1 was retried.  This class
    closes both halves of that gap: it names the native reason, and it
    subclasses :class:`InfrastructureError` so
    :meth:`OpenRouterProvider.classify_error` routes it to ``with_retry``
    exactly as shape 1 already was.

    ``native_reason`` is ``None`` when the upstream reported nothing at
    all.  That is still worth raising — the turn produced no usable
    answer either way — but the message says so plainly instead of
    implying a diagnosis exists.
    """

    def __init__(
        self,
        native_reason: Optional[str] = None,
        *,
        generation_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.native_reason = native_reason
        self.generation_id = generation_id
        self.model = model
        super().__init__(status_code=0, original_error=None)

    def _format_message(self) -> str:
        lines = [
            'OpenRouter upstream ended the turn with finish_reason="error".',
        ]
        if self.native_reason:
            lines.append(f"Upstream reason (native_finish_reason): {self.native_reason}")
        else:
            lines.append(
                "The upstream reported no native_finish_reason, so no further "
                "diagnosis is available from the response itself."
            )
        if self.model:
            lines.append(f"Model: {self.model}")
        if self.generation_id:
            lines.append(f"Generation ID: {self.generation_id}")
        lines.extend([
            "",
            "No error payload accompanied the finish reason, so the native",
            "reason above is the whole of what the upstream said.",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])
        return "\n".join(lines)
