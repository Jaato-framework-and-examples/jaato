"""Tests for ``jaato_sdk.helpers``.

These pin the contract that other clients (TS SDK, third-party
consumers, the TUI) rely on.  When the formula changes here, the
mirror in ``jaato-sdk-ts/src/helpers.ts`` must be updated too.
"""

import pytest

from jaato_sdk import compute_cache_hit_percent
from jaato_sdk.events import TurnCompletedEvent, TurnProgressEvent, UsageBreakdown


def _completed(
    prompt_tokens: int = 0,
    cache_read_tokens=None,
    cache_creation_tokens=None,
) -> TurnCompletedEvent:
    return TurnCompletedEvent(
        usage=UsageBreakdown(
            prompt_tokens=prompt_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        ),
    )


def _progress(
    prompt_tokens: int = 0,
    cache_read_tokens=None,
    cache_creation_tokens=None,
) -> TurnProgressEvent:
    return TurnProgressEvent(
        usage=UsageBreakdown(
            prompt_tokens=prompt_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        ),
    )


class TestComputeCacheHitPercent:

    def test_none_when_provider_does_not_report_caching(self):
        assert compute_cache_hit_percent(_completed(prompt_tokens=100)) is None

    def test_zero_when_caching_supported_but_no_hits(self):
        result = compute_cache_hit_percent(
            _completed(prompt_tokens=1000, cache_read_tokens=0)
        )
        assert result == 0.0

    def test_full_hit_returns_100(self):
        result = compute_cache_hit_percent(
            _completed(prompt_tokens=0, cache_read_tokens=5000)
        )
        assert result == 100.0

    def test_typical_hit_rate(self):
        # 4500 from cache, 500 new input → 90% hit rate.
        result = compute_cache_hit_percent(
            _completed(prompt_tokens=500, cache_read_tokens=4500)
        )
        assert result == pytest.approx(90.0)

    def test_cache_creation_excluded_from_denominator(self):
        # 1000 read + 200 prompt + 800 creation.  The TUI's prior
        # formula intentionally excluded creation from the denominator
        # — this test documents that contract.
        result = compute_cache_hit_percent(
            _completed(
                prompt_tokens=200,
                cache_read_tokens=1000,
                cache_creation_tokens=800,
            )
        )
        # 1000 / (1000 + 200) = 83.33%, NOT 1000 / (1000 + 200 + 800) = 50%.
        assert result == pytest.approx(83.333333, rel=1e-4)

    def test_zero_total_returns_zero(self):
        # Pathological: provider reports cache_read=0 and prompt=0
        # (an empty turn?).  Avoid ZeroDivisionError; surface 0.0.
        result = compute_cache_hit_percent(
            _completed(prompt_tokens=0, cache_read_tokens=0)
        )
        assert result == 0.0

    def test_works_on_progress_events_too(self):
        # The helper accepts TurnProgressEvent and TurnCompletedEvent
        # interchangeably — they share the same field names.
        result = compute_cache_hit_percent(
            _progress(prompt_tokens=100, cache_read_tokens=900)
        )
        assert result == pytest.approx(90.0)
