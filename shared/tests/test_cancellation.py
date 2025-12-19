"""Tests for cancellation functionality.

Tests for CancelToken, CancelledException, interruptible_sleep, and
cancellation integration with with_retry.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock

from shared.plugins.model_provider.types import (
    CancelledException,
    CancelToken,
    FinishReason,
)
from shared.retry_utils import (
    RetryConfig,
    interruptible_sleep,
    with_retry,
)


class TestCancelToken(unittest.TestCase):
    """Tests for CancelToken class."""

    def test_initial_state_not_cancelled(self):
        """Token starts in non-cancelled state."""
        token = CancelToken()
        self.assertFalse(token.is_cancelled)

    def test_cancel_sets_state(self):
        """Calling cancel() sets the cancelled state."""
        token = CancelToken()
        token.cancel()
        self.assertTrue(token.is_cancelled)

    def test_cancel_is_idempotent(self):
        """Multiple calls to cancel() have no effect."""
        token = CancelToken()
        token.cancel()
        token.cancel()
        token.cancel()
        self.assertTrue(token.is_cancelled)

    def test_raise_if_cancelled(self):
        """raise_if_cancelled() raises when cancelled."""
        token = CancelToken()

        # Should not raise when not cancelled
        token.raise_if_cancelled()

        # Should raise when cancelled
        token.cancel()
        with self.assertRaises(CancelledException):
            token.raise_if_cancelled()

    def test_wait_returns_true_when_cancelled(self):
        """wait() returns True when cancelled."""
        token = CancelToken()

        def cancel_after_delay():
            time.sleep(0.1)
            token.cancel()

        thread = threading.Thread(target=cancel_after_delay)
        thread.start()

        result = token.wait(timeout=1.0)
        self.assertTrue(result)
        thread.join()

    def test_wait_returns_false_on_timeout(self):
        """wait() returns False when timeout expires."""
        token = CancelToken()
        result = token.wait(timeout=0.1)
        self.assertFalse(result)
        self.assertFalse(token.is_cancelled)

    def test_on_cancel_callback_called(self):
        """on_cancel() callback is invoked when cancelled."""
        token = CancelToken()
        callback = MagicMock()

        token.on_cancel(callback)
        callback.assert_not_called()

        token.cancel()
        callback.assert_called_once()

    def test_on_cancel_immediate_when_already_cancelled(self):
        """on_cancel() callback is called immediately if already cancelled."""
        token = CancelToken()
        token.cancel()

        callback = MagicMock()
        token.on_cancel(callback)
        callback.assert_called_once()

    def test_on_cancel_multiple_callbacks(self):
        """Multiple on_cancel() callbacks are all invoked."""
        token = CancelToken()
        callbacks = [MagicMock() for _ in range(3)]

        for cb in callbacks:
            token.on_cancel(cb)

        token.cancel()

        for cb in callbacks:
            cb.assert_called_once()

    def test_reset_clears_state(self):
        """reset() clears cancelled state."""
        token = CancelToken()
        token.cancel()
        self.assertTrue(token.is_cancelled)

        token.reset()
        self.assertFalse(token.is_cancelled)

    def test_thread_safety(self):
        """CancelToken is thread-safe for concurrent access."""
        token = CancelToken()
        results = []

        def check_and_cancel():
            for _ in range(100):
                _ = token.is_cancelled
                if not token.is_cancelled:
                    token.cancel()
            results.append(True)

        threads = [threading.Thread(target=check_and_cancel) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 10)
        self.assertTrue(token.is_cancelled)


class TestCancelledException(unittest.TestCase):
    """Tests for CancelledException class."""

    def test_default_message(self):
        """Default exception message is set."""
        exc = CancelledException()
        self.assertEqual(str(exc), "Operation was cancelled")

    def test_custom_message(self):
        """Custom exception message is preserved."""
        exc = CancelledException("Custom cancellation reason")
        self.assertEqual(str(exc), "Custom cancellation reason")


class TestFinishReasonCancelled(unittest.TestCase):
    """Tests for FinishReason.CANCELLED."""

    def test_cancelled_finish_reason_exists(self):
        """FinishReason.CANCELLED enum value exists."""
        self.assertEqual(FinishReason.CANCELLED.value, "cancelled")

    def test_cancelled_is_distinct(self):
        """CANCELLED is distinct from other finish reasons."""
        self.assertNotEqual(FinishReason.CANCELLED, FinishReason.STOP)
        self.assertNotEqual(FinishReason.CANCELLED, FinishReason.ERROR)
        self.assertNotEqual(FinishReason.CANCELLED, FinishReason.UNKNOWN)


class TestInterruptibleSleep(unittest.TestCase):
    """Tests for interruptible_sleep function."""

    def test_normal_sleep_without_token(self):
        """Sleeps normally when no cancel token provided."""
        start = time.monotonic()
        result = interruptible_sleep(0.1, cancel_token=None)
        elapsed = time.monotonic() - start

        self.assertTrue(result)
        self.assertGreaterEqual(elapsed, 0.1)

    def test_normal_sleep_with_token(self):
        """Sleeps normally when token is not cancelled."""
        token = CancelToken()
        start = time.monotonic()
        result = interruptible_sleep(0.1, cancel_token=token)
        elapsed = time.monotonic() - start

        self.assertTrue(result)
        self.assertGreaterEqual(elapsed, 0.1)
        self.assertFalse(token.is_cancelled)

    def test_interrupted_by_cancellation(self):
        """Sleep is interrupted when token is cancelled."""
        token = CancelToken()

        def cancel_after_delay():
            time.sleep(0.05)
            token.cancel()

        thread = threading.Thread(target=cancel_after_delay)
        thread.start()

        start = time.monotonic()
        with self.assertRaises(CancelledException):
            interruptible_sleep(1.0, cancel_token=token)
        elapsed = time.monotonic() - start

        # Should have exited early (well before 1.0s)
        self.assertLess(elapsed, 0.5)
        thread.join()

    def test_already_cancelled_token(self):
        """Raises immediately if token already cancelled."""
        token = CancelToken()
        token.cancel()

        start = time.monotonic()
        with self.assertRaises(CancelledException):
            interruptible_sleep(1.0, cancel_token=token)
        elapsed = time.monotonic() - start

        # Should exit almost immediately
        self.assertLess(elapsed, 0.2)


class TestWithRetryCancellation(unittest.TestCase):
    """Tests for with_retry with cancellation support."""

    def test_cancelled_before_first_attempt(self):
        """Raises CancelledException if cancelled before first attempt."""
        token = CancelToken()
        token.cancel()

        fn = MagicMock(return_value="success")
        config = RetryConfig(max_attempts=3, silent=True)

        with self.assertRaises(CancelledException):
            with_retry(fn, config=config, cancel_token=token)

        fn.assert_not_called()

    def test_cancelled_during_backoff(self):
        """Raises CancelledException if cancelled during backoff sleep."""
        token = CancelToken()
        call_count = [0]

        def fn():
            call_count[0] += 1
            raise Exception("429 rate limit")

        def cancel_during_backoff():
            time.sleep(0.05)
            token.cancel()

        config = RetryConfig(max_attempts=5, base_delay=1.0, silent=True)
        thread = threading.Thread(target=cancel_during_backoff)
        thread.start()

        start = time.monotonic()
        with self.assertRaises(CancelledException):
            with_retry(fn, config=config, cancel_token=token)
        elapsed = time.monotonic() - start

        # Should have exited during backoff (well before 1.0s base delay)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(call_count[0], 1)  # Only first attempt
        thread.join()

    def test_cancellation_exception_not_retried(self):
        """CancelledException from fn is not retried."""
        call_count = [0]

        def fn():
            call_count[0] += 1
            raise CancelledException("Cancelled in function")

        config = RetryConfig(max_attempts=5, base_delay=0.01, silent=True)

        with self.assertRaises(CancelledException):
            with_retry(fn, config=config)

        # Should not retry CancelledException
        self.assertEqual(call_count[0], 1)

    def test_success_with_cancel_token(self):
        """Normal success works with cancel token present."""
        token = CancelToken()
        fn = MagicMock(return_value="success")
        config = RetryConfig(max_attempts=3, silent=True)

        result, stats = with_retry(fn, config=config, cancel_token=token)

        self.assertEqual(result, "success")
        self.assertEqual(stats.attempts, 1)
        self.assertFalse(token.is_cancelled)

    def test_retry_then_success_with_cancel_token(self):
        """Retries work normally with cancel token present."""
        token = CancelToken()
        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("429 rate limit")
            return "success"

        config = RetryConfig(max_attempts=5, base_delay=0.01, silent=True)

        result, stats = with_retry(fn, config=config, cancel_token=token)

        self.assertEqual(result, "success")
        self.assertEqual(stats.attempts, 3)
        self.assertFalse(token.is_cancelled)


if __name__ == "__main__":
    unittest.main()
