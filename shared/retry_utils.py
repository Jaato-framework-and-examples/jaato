"""Retry and rate limiting utilities for API calls.

Provides:
- Retry logic with exponential backoff for transient errors
- Request pacing to proactively avoid rate limits

Usage:
    from shared.retry_utils import with_retry, RetryConfig, RequestPacer

    # Retry on errors (reactive)
    result = with_retry(lambda: provider.send_message(msg))

    # Request pacing (proactive) - shared pacer instance
    pacer = RequestPacer()
    pacer.pace()  # waits if needed before request
    result = provider.send_message(msg)

Environment Variables:
    # Retry (reactive)
    AI_RETRY_ATTEMPTS: Max retry attempts (default: 5)
    AI_RETRY_BASE_DELAY: Initial delay in seconds (default: 1.0)
    AI_RETRY_MAX_DELAY: Maximum delay in seconds (default: 30.0)
    AI_RETRY_LOG_SILENT: Suppress retry logging when set to '1', 'true', or 'yes'

    # Pacing (proactive)
    AI_REQUEST_INTERVAL: Minimum seconds between requests (default: 0 = disabled)
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .plugins.model_provider.types import CancelToken

# Type alias for retry callback - client provides this to handle retry notifications
# Signature: (message: str, attempt: int, max_attempts: int, delay: float) -> None
RetryCallback = Callable[[str, int, int, float], None]

# Import Google exceptions for detection
try:
    from google.api_core import exceptions as google_exceptions
    GOOGLE_TRANSIENT_CLASSES: Tuple[Type[Exception], ...] = (
        google_exceptions.TooManyRequests,
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.InternalServerError,
        google_exceptions.DeadlineExceeded,
        google_exceptions.Aborted,
    )
    GOOGLE_RATE_LIMIT_CLASSES: Tuple[Type[Exception], ...] = (
        google_exceptions.TooManyRequests,
        google_exceptions.ResourceExhausted,
    )
except ImportError:
    GOOGLE_TRANSIENT_CLASSES = ()
    GOOGLE_RATE_LIMIT_CLASSES = ()

# Import GitHub Models errors for detection
try:
    from .plugins.model_provider.github_models.errors import RateLimitError as GitHubRateLimitError
    GITHUB_RATE_LIMIT_CLASSES: Tuple[Type[Exception], ...] = (GitHubRateLimitError,)
except ImportError:
    GITHUB_RATE_LIMIT_CLASSES = ()

# Import Anthropic errors for detection
try:
    from .plugins.model_provider.anthropic.errors import (
        RateLimitError as AnthropicRateLimitError,
        OverloadedError as AnthropicOverloadedError,
    )
    ANTHROPIC_RATE_LIMIT_CLASSES: Tuple[Type[Exception], ...] = (AnthropicRateLimitError,)
    ANTHROPIC_TRANSIENT_CLASSES: Tuple[Type[Exception], ...] = (
        AnthropicRateLimitError,
        AnthropicOverloadedError,
    )
except ImportError:
    ANTHROPIC_RATE_LIMIT_CLASSES = ()
    ANTHROPIC_TRANSIENT_CLASSES = ()

# Import Google GenAI SDK errors for detection (newer python-genai SDK)
# These are different from google.api_core.exceptions used by older libraries
try:
    from google.genai import errors as genai_errors
    # ClientError wraps HTTP 4xx errors including 429 rate limits
    GENAI_RATE_LIMIT_CLASSES: Tuple[Type[Exception], ...] = (genai_errors.ClientError,)
except ImportError:
    GENAI_RATE_LIMIT_CLASSES = ()


T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = field(default_factory=lambda: int(os.environ.get("AI_RETRY_ATTEMPTS", "5")))
    base_delay: float = field(default_factory=lambda: float(os.environ.get("AI_RETRY_BASE_DELAY", "1.0")))
    max_delay: float = field(default_factory=lambda: float(os.environ.get("AI_RETRY_MAX_DELAY", "30.0")))
    silent: bool = field(default_factory=lambda: os.environ.get("AI_RETRY_LOG_SILENT", "").lower() in ("1", "true", "yes"))
    jitter_factor: float = 0.5  # Random jitter range: [1-jitter, 1+jitter]


@dataclass
class RetryStats:
    """Statistics from a retry operation."""
    attempts: int = 0
    total_delay: float = 0.0
    rate_limit_errors: int = 0
    transient_errors: int = 0
    last_error: Optional[Exception] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)


def classify_error(exc: Exception) -> Dict[str, bool]:
    """Classify an exception as transient/rate-limit/infra.

    Args:
        exc: The exception to classify.

    Returns:
        Dict with keys: transient, rate_limit, infra
    """
    rate_like = False
    infra_like = False

    # Check Google api_core exceptions (older google-generativeai SDK)
    if GOOGLE_TRANSIENT_CLASSES and isinstance(exc, GOOGLE_TRANSIENT_CLASSES):
        if GOOGLE_RATE_LIMIT_CLASSES and isinstance(exc, GOOGLE_RATE_LIMIT_CLASSES):
            rate_like = True
        else:
            infra_like = True
    # Check Google GenAI ClientError (newer python-genai SDK)
    # ClientError wraps all 4xx errors, so check message for specific codes
    elif GENAI_RATE_LIMIT_CLASSES and isinstance(exc, GENAI_RATE_LIMIT_CLASSES):
        lower = str(exc).lower()
        if any(p in lower for p in ["429", "resource exhausted", "resource_exhausted", "rate limit", "quota"]):
            rate_like = True
        elif any(p in lower for p in ["503", "500", "service unavailable", "internal"]):
            infra_like = True
        # For other 4xx ClientErrors, don't retry by default (auth errors, bad requests, etc.)
    # Check GitHub rate limit
    elif GITHUB_RATE_LIMIT_CLASSES and isinstance(exc, GITHUB_RATE_LIMIT_CLASSES):
        rate_like = True
    # Check Anthropic exceptions
    elif ANTHROPIC_TRANSIENT_CLASSES and isinstance(exc, ANTHROPIC_TRANSIENT_CLASSES):
        if ANTHROPIC_RATE_LIMIT_CLASSES and isinstance(exc, ANTHROPIC_RATE_LIMIT_CLASSES):
            rate_like = True
        else:
            infra_like = True
    else:
        # Fallback: check error message for common patterns
        lower = str(exc).lower()
        if any(p in lower for p in ["429", "too many requests", "resource exhausted", "resource_exhausted", "rate limit", "rate_limit"]):
            rate_like = True
        if any(p in lower for p in ["503", "service unavailable", "temporarily unavailable", "internal error"]):
            infra_like = True

    return {
        "transient": rate_like or infra_like,
        "rate_limit": rate_like,
        "infra": infra_like,
    }


def get_retry_after(exc: Exception) -> Optional[float]:
    """Extract retry-after hint from an exception if available.

    Args:
        exc: The exception to check.

    Returns:
        Suggested retry delay in seconds, or None if not available.
    """
    # GitHub RateLimitError has retry_after attribute
    if hasattr(exc, 'retry_after') and exc.retry_after:
        return float(exc.retry_after)

    # Some HTTP errors include Retry-After header
    if hasattr(exc, 'response') and hasattr(exc.response, 'headers'):
        retry_after = exc.response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    return None


def calculate_backoff(
    attempt: int,
    config: RetryConfig,
    retry_after: Optional[float] = None
) -> float:
    """Calculate backoff delay for a retry attempt.

    Uses exponential backoff with jitter, respecting retry-after hints.

    Args:
        attempt: Current attempt number (1-indexed).
        config: Retry configuration.
        retry_after: Optional retry-after hint from server.

    Returns:
        Delay in seconds before next attempt.
    """
    # Calculate exponential backoff
    exp_delay = config.base_delay * (2 ** (attempt - 1))
    capped_delay = min(config.max_delay, exp_delay)

    # Add jitter
    jitter = random.uniform(1 - config.jitter_factor, 1 + config.jitter_factor)
    delay = capped_delay * jitter

    # Respect retry-after hint if provided and larger
    if retry_after and retry_after > delay:
        delay = retry_after

    return delay


def interruptible_sleep(
    delay: float,
    cancel_token: Optional['CancelToken'] = None,
    poll_interval: float = 0.1
) -> bool:
    """Sleep for the specified delay, but can be interrupted by cancellation.

    Args:
        delay: Seconds to sleep.
        cancel_token: Optional CancelToken to check for cancellation.
        poll_interval: How often to check for cancellation (default 0.1s).

    Returns:
        True if sleep completed normally, False if cancelled.

    Raises:
        CancelledException: If cancel_token is set and cancellation is requested.
    """
    if cancel_token is None:
        # No cancellation token, just do normal sleep
        time.sleep(delay)
        return True

    # Sleep in small increments, checking for cancellation
    end_time = time.monotonic() + delay
    while time.monotonic() < end_time:
        if cancel_token.is_cancelled:
            # Import here to avoid circular import
            from .plugins.model_provider.types import CancelledException
            raise CancelledException("Cancelled during retry backoff")
        remaining = end_time - time.monotonic()
        time.sleep(min(poll_interval, max(0, remaining)))

    return True


def with_retry(
    fn: Callable[[], T],
    config: Optional[RetryConfig] = None,
    context: str = "API call",
    on_retry: Optional[RetryCallback] = None,
    cancel_token: Optional['CancelToken'] = None,
) -> Tuple[T, RetryStats]:
    """Execute a function with automatic retry on transient errors.

    Args:
        fn: Function to execute (should take no arguments).
        config: Retry configuration (uses defaults if None).
        context: Description for logging (e.g., "send_message").
        on_retry: Optional callback for retry notifications.
            Signature: (message: str, attempt: int, max_attempts: int, delay: float) -> None
            If provided, retry messages go through this callback.
            If not provided, messages are printed to console (unless config.silent).
        cancel_token: Optional CancelToken to check for cancellation.
            If cancelled, raises CancelledException.

    Returns:
        Tuple of (result, RetryStats).

    Raises:
        The last exception if all retries are exhausted or error is non-transient.
        CancelledException: If cancel_token is set and cancellation is requested.

    Example:
        # Simple client - uses console output (default)
        response, stats = with_retry(
            lambda: provider.send_message(message),
            context="send_message"
        )

        # Rich client - routes to custom handler
        response, stats = with_retry(
            lambda: provider.send_message(message),
            context="send_message",
            on_retry=lambda msg, att, max_att, delay: queue.put(msg)
        )

        # With cancellation support
        token = CancelToken()
        response, stats = with_retry(
            lambda: provider.send_message(message),
            context="send_message",
            cancel_token=token
        )
    """
    if config is None:
        config = RetryConfig()

    stats = RetryStats()
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.max_attempts + 1):
        stats.attempts = attempt

        # Check for cancellation before each attempt
        if cancel_token and cancel_token.is_cancelled:
            from .plugins.model_provider.types import CancelledException
            raise CancelledException("Cancelled before retry attempt")

        try:
            result = fn()
            return result, stats

        except Exception as exc:
            # Check if this is a CancelledException - don't retry those
            exc_name = exc.__class__.__name__
            if exc_name == 'CancelledException':
                raise

            last_exc = exc
            stats.last_error = exc

            # Classify the error
            classification = classify_error(exc)

            # Record error details
            stats.errors.append({
                "attempt": attempt,
                "error": str(exc)[:200],
                "error_type": exc_name,
                **classification,
            })

            if classification["rate_limit"]:
                stats.rate_limit_errors += 1
            elif classification["infra"]:
                stats.transient_errors += 1

            # If not transient or last attempt, re-raise
            if not classification["transient"] or attempt == config.max_attempts:
                raise

            # Calculate and apply backoff
            retry_after = get_retry_after(exc)
            delay = calculate_backoff(attempt, config, retry_after)
            stats.total_delay += delay

            # Build retry message
            err_cls = exc_name
            tag = "rate-limit" if classification["rate_limit"] else "transient"
            exc_msg = str(exc)[:140].replace('\n', ' ')
            msg = f"[AI Retry {attempt}/{config.max_attempts}] {context} ({tag}): {err_cls}: {exc_msg} | sleep {delay:.2f}s"

            # Notify via callback or console
            if on_retry:
                # Client-provided callback handles the notification
                on_retry(msg, attempt, config.max_attempts, delay)
            elif not config.silent:
                # Default: print to console (unless silent)
                print(msg)

            # Use interruptible sleep that respects cancellation
            interruptible_sleep(delay, cancel_token)

    # Should not reach here, but just in case
    if last_exc:
        raise last_exc
    raise RuntimeError("Retry loop exited without result or exception")


class RequestPacer:
    """Enforces minimum interval between API requests to avoid rate limits.

    This provides proactive rate limiting by ensuring requests are spaced
    at least `min_interval` seconds apart.

    Usage:
        pacer = RequestPacer()  # reads AI_REQUEST_INTERVAL from env
        pacer.pace()  # call before each API request

    The pacer is thread-safe for single-writer scenarios (one session).
    For multi-session use, create one pacer per session or use a shared
    instance with external synchronization.
    """

    def __init__(self, min_interval: Optional[float] = None):
        """Initialize the pacer.

        Args:
            min_interval: Minimum seconds between requests.
                If None, reads from AI_REQUEST_INTERVAL env var (default: 0).
        """
        if min_interval is not None:
            self._min_interval = min_interval
        else:
            self._min_interval = float(os.environ.get("AI_REQUEST_INTERVAL", "0"))
        self._last_request_time: Optional[float] = None

    @property
    def min_interval(self) -> float:
        """Get the minimum interval between requests."""
        return self._min_interval

    @property
    def enabled(self) -> bool:
        """Check if pacing is enabled (interval > 0)."""
        return self._min_interval > 0

    def pace(self) -> float:
        """Wait if needed to maintain minimum request interval.

        Call this before each API request. Returns immediately if enough
        time has passed since the last request, otherwise sleeps.

        Returns:
            Seconds waited (0 if no wait was needed).
        """
        if not self.enabled:
            return 0.0

        now = time.time()
        waited = 0.0

        if self._last_request_time is not None:
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                time.sleep(wait_time)
                waited = wait_time
                now = time.time()

        self._last_request_time = now
        return waited

    def reset(self) -> None:
        """Reset the pacer (next request won't wait)."""
        self._last_request_time = None


__all__ = [
    'RequestPacer',
    'RetryCallback',
    'RetryConfig',
    'RetryStats',
    'classify_error',
    'calculate_backoff',
    'get_retry_after',
    'interruptible_sleep',
    'with_retry',
]
