"""Tests for retry_utils module."""

import unittest
from unittest.mock import patch, MagicMock
import time

from shared.retry_utils import (
    RetryConfig,
    RetryStats,
    classify_error,
    calculate_backoff,
    with_retry,
    is_context_limit_error,
)


class TestClassifyError(unittest.TestCase):
    """Tests for error classification."""

    def test_anthropic_rate_limit_class_detection(self):
        """Detects Anthropic RateLimitError class directly."""
        try:
            from shared.plugins.model_provider.anthropic.errors import RateLimitError
            exc = RateLimitError(retry_after=30, original_error="test error")
            result = classify_error(exc)
            self.assertTrue(result["transient"])
            self.assertTrue(result["rate_limit"])
            self.assertFalse(result["infra"])
        except ImportError:
            self.skipTest("Anthropic provider not available")

    def test_anthropic_overloaded_class_detection(self):
        """Detects Anthropic OverloadedError class as transient infra error."""
        try:
            from shared.plugins.model_provider.anthropic.errors import OverloadedError
            exc = OverloadedError(original_error="API overloaded")
            result = classify_error(exc)
            self.assertTrue(result["transient"])
            self.assertFalse(result["rate_limit"])
            self.assertTrue(result["infra"])
        except ImportError:
            self.skipTest("Anthropic provider not available")

    def test_rate_limit_string_detection(self):
        """Detects rate limit from error message."""
        exc = Exception("429 Too Many Requests")
        result = classify_error(exc)
        self.assertTrue(result["transient"])
        self.assertTrue(result["rate_limit"])
        self.assertFalse(result["infra"])

    def test_resource_exhausted_detection(self):
        """Detects resource exhausted from error message (both space and underscore variants)."""
        # With underscore (as it appears in actual errors)
        exc = Exception("RESOURCE_EXHAUSTED: quota exceeded")
        result = classify_error(exc)
        self.assertTrue(result["transient"])
        self.assertTrue(result["rate_limit"])

        # With space
        exc2 = Exception("resource exhausted")
        result2 = classify_error(exc2)
        self.assertTrue(result2["transient"])
        self.assertTrue(result2["rate_limit"])

    def test_service_unavailable_detection(self):
        """Detects service unavailable as transient."""
        exc = Exception("503 Service Unavailable")
        result = classify_error(exc)
        self.assertTrue(result["transient"])
        self.assertFalse(result["rate_limit"])
        self.assertTrue(result["infra"])

    def test_non_transient_error(self):
        """Non-transient errors are classified correctly."""
        exc = Exception("Invalid API key")
        result = classify_error(exc)
        self.assertFalse(result["transient"])
        self.assertFalse(result["rate_limit"])
        self.assertFalse(result["infra"])


class TestCalculateBackoff(unittest.TestCase):
    """Tests for backoff calculation."""

    def test_exponential_growth(self):
        """Backoff grows exponentially."""
        config = RetryConfig(base_delay=1.0, max_delay=100.0, jitter_factor=0.0)

        delay1 = calculate_backoff(1, config)
        delay2 = calculate_backoff(2, config)
        delay3 = calculate_backoff(3, config)

        self.assertAlmostEqual(delay1, 1.0, places=1)
        self.assertAlmostEqual(delay2, 2.0, places=1)
        self.assertAlmostEqual(delay3, 4.0, places=1)

    def test_max_delay_cap(self):
        """Backoff is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter_factor=0.0)

        delay10 = calculate_backoff(10, config)
        self.assertLessEqual(delay10, 5.0)

    def test_retry_after_hint(self):
        """Respects retry-after hint when larger than calculated delay."""
        config = RetryConfig(base_delay=1.0, max_delay=100.0, jitter_factor=0.0)

        # retry_after larger than calculated
        delay = calculate_backoff(1, config, retry_after=10.0)
        self.assertAlmostEqual(delay, 10.0, places=1)

        # retry_after smaller than calculated
        delay = calculate_backoff(5, config, retry_after=1.0)  # 2^4 = 16
        self.assertGreater(delay, 1.0)


class TestWithRetry(unittest.TestCase):
    """Tests for with_retry function."""

    def test_success_on_first_attempt(self):
        """Returns immediately on success."""
        fn = MagicMock(return_value="success")
        config = RetryConfig(max_attempts=3, silent=True)

        result, stats = with_retry(fn, config=config)

        self.assertEqual(result, "success")
        self.assertEqual(stats.attempts, 1)
        self.assertEqual(stats.rate_limit_errors, 0)
        fn.assert_called_once()

    def test_retry_on_rate_limit(self):
        """Retries on rate limit errors."""
        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("429 Too Many Requests")
            return "success"

        config = RetryConfig(max_attempts=5, base_delay=0.01, silent=True)

        result, stats = with_retry(fn, config=config)

        self.assertEqual(result, "success")
        self.assertEqual(stats.attempts, 3)
        self.assertEqual(stats.rate_limit_errors, 2)

    def test_fails_after_max_attempts(self):
        """Raises after exhausting retries."""
        fn = MagicMock(side_effect=Exception("429 rate limit"))
        config = RetryConfig(max_attempts=3, base_delay=0.01, silent=True)

        with self.assertRaises(Exception) as ctx:
            with_retry(fn, config=config)

        self.assertIn("429", str(ctx.exception))
        self.assertEqual(fn.call_count, 3)

    def test_non_transient_error_not_retried(self):
        """Non-transient errors are not retried."""
        fn = MagicMock(side_effect=ValueError("Invalid input"))
        config = RetryConfig(max_attempts=3, base_delay=0.01, silent=True)

        with self.assertRaises(ValueError):
            with_retry(fn, config=config)

        fn.assert_called_once()

    def test_stats_tracking(self):
        """Stats track retry information correctly."""
        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 rate limit")
            if call_count[0] == 2:
                raise Exception("503 service unavailable")
            return "success"

        config = RetryConfig(max_attempts=5, base_delay=0.01, silent=True)

        result, stats = with_retry(fn, config=config)

        self.assertEqual(result, "success")
        self.assertEqual(stats.attempts, 3)
        self.assertEqual(stats.rate_limit_errors, 1)
        self.assertEqual(stats.transient_errors, 1)
        self.assertEqual(len(stats.errors), 2)


class TestIsContextLimitError(unittest.TestCase):
    """Tests for is_context_limit_error detection."""

    def test_anthropic_context_limit_class(self):
        """Detects Anthropic ContextLimitError class directly."""
        try:
            from shared.plugins.model_provider.anthropic.errors import ContextLimitError
            exc = ContextLimitError(model="claude-3", original_error="context too long")
            self.assertTrue(is_context_limit_error(exc))
        except ImportError:
            self.skipTest("Anthropic provider not available")

    def test_github_context_limit_class(self):
        """Detects GitHub ContextLimitError class directly."""
        try:
            from shared.plugins.model_provider.github_models.errors import ContextLimitError
            exc = ContextLimitError(model="gpt-4", original_error="tokens exceeded")
            self.assertTrue(is_context_limit_error(exc))
        except ImportError:
            self.skipTest("GitHub Models provider not available")

    def test_github_payload_too_large_class(self):
        """Detects GitHub PayloadTooLargeError class."""
        try:
            from shared.plugins.model_provider.github_models.errors import PayloadTooLargeError
            exc = PayloadTooLargeError(original_error="HTTP 413")
            self.assertTrue(is_context_limit_error(exc))
        except ImportError:
            self.skipTest("GitHub Models provider not available")

    def test_string_token_limit_exceeded(self):
        """Detects token limit exceeded from error message."""
        exc = Exception("181145 tokens exceeds the limit of 128000")
        self.assertTrue(is_context_limit_error(exc))

    def test_string_context_too_long(self):
        """Detects context too long from error message."""
        exc = Exception("Request context is too long for this model")
        self.assertTrue(is_context_limit_error(exc))

    def test_string_prompt_too_large(self):
        """Detects prompt too large from error message."""
        exc = Exception("Prompt is too large, maximum is 100000 tokens")
        self.assertTrue(is_context_limit_error(exc))

    def test_string_http_413_payload(self):
        """Detects HTTP 413 payload too large."""
        exc = Exception("HTTP 413: payload too large")
        self.assertTrue(is_context_limit_error(exc))

    def test_rate_limit_not_context_limit(self):
        """Rate limit errors are NOT context limit errors."""
        exc = Exception("429 Too Many Requests - rate limit exceeded")
        self.assertFalse(is_context_limit_error(exc))

    def test_auth_error_not_context_limit(self):
        """Auth errors are NOT context limit errors."""
        exc = Exception("Invalid API key")
        self.assertFalse(is_context_limit_error(exc))

    def test_generic_error_not_context_limit(self):
        """Generic errors are NOT context limit errors."""
        exc = Exception("Something went wrong")
        self.assertFalse(is_context_limit_error(exc))


if __name__ == "__main__":
    unittest.main()
