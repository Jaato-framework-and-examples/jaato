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
)


class TestClassifyError(unittest.TestCase):
    """Tests for error classification."""

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


if __name__ == "__main__":
    unittest.main()
