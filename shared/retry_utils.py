"""Retry utilities for transient API errors.

Provides retry logic with exponential backoff for rate limit and transient
errors from various model providers.

Usage:
    from shared.retry_utils import with_retry, RetryConfig

    # Using default config (reads from environment)
    result = with_retry(lambda: provider.send_message(msg))

    # With custom config
    config = RetryConfig(max_attempts=3, base_delay=2.0)
    result = with_retry(lambda: provider.send_message(msg), config=config)

Environment Variables:
    AI_RETRY_ATTEMPTS: Max retry attempts (default: 5)
    AI_RETRY_BASE_DELAY: Initial delay in seconds (default: 1.0)
    AI_RETRY_MAX_DELAY: Maximum delay in seconds (default: 30.0)
    AI_RETRY_LOG_SILENT: Suppress retry logging when set to '1', 'true', or 'yes'
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

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

    # Check Google exceptions
    if GOOGLE_TRANSIENT_CLASSES and isinstance(exc, GOOGLE_TRANSIENT_CLASSES):
        if GOOGLE_RATE_LIMIT_CLASSES and isinstance(exc, GOOGLE_RATE_LIMIT_CLASSES):
            rate_like = True
        else:
            infra_like = True
    # Check GitHub rate limit
    elif GITHUB_RATE_LIMIT_CLASSES and isinstance(exc, GITHUB_RATE_LIMIT_CLASSES):
        rate_like = True
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


def with_retry(
    fn: Callable[[], T],
    config: Optional[RetryConfig] = None,
    context: str = "API call",
    on_retry: Optional[RetryCallback] = None,
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

    Returns:
        Tuple of (result, RetryStats).

    Raises:
        The last exception if all retries are exhausted or error is non-transient.

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
    """
    if config is None:
        config = RetryConfig()

    stats = RetryStats()
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.max_attempts + 1):
        stats.attempts = attempt

        try:
            result = fn()
            return result, stats

        except Exception as exc:
            last_exc = exc
            stats.last_error = exc

            # Classify the error
            classification = classify_error(exc)

            # Record error details
            stats.errors.append({
                "attempt": attempt,
                "error": str(exc)[:200],
                "error_type": exc.__class__.__name__,
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
            err_cls = exc.__class__.__name__
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

            time.sleep(delay)

    # Should not reach here, but just in case
    if last_exc:
        raise last_exc
    raise RuntimeError("Retry loop exited without result or exception")


__all__ = [
    'RetryCallback',
    'RetryConfig',
    'RetryStats',
    'classify_error',
    'calculate_backoff',
    'get_retry_after',
    'with_retry',
]
