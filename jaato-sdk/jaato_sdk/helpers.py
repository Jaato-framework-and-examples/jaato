"""Convenience helpers over the wire-protocol event types.

These compute derived metrics that every client would otherwise
reimplement (and occasionally get wrong — see the comment trail
in ``jaato-tui/rich_client.py`` around the cache-hit calculation).

Mirrors of these helpers live in ``jaato-sdk-ts/src/helpers.ts`` so
TS and Python clients render the same numbers.
"""

from __future__ import annotations

from typing import Optional, Union

from jaato_sdk.events import TurnCompletedEvent, TurnProgressEvent

CacheReportingEvent = Union[TurnCompletedEvent, TurnProgressEvent]


def compute_cache_hit_percent(event: CacheReportingEvent) -> Optional[float]:
    """Cache hit % for a turn — share of paid input tokens served from cache.

    The denominator is ``cache_read_tokens + prompt_tokens`` (the
    "non-creation" input).  Anthropic's ``input_tokens`` /
    pydantic's ``prompt_tokens`` is the *new uncached* input — it
    excludes both ``cache_read_tokens`` and ``cache_creation_tokens``,
    so the natural hit-rate denominator is "what was the model paying
    for this turn that *could* have been a cache hit", which is
    cache_read (the actual hit) plus prompt (the new input that
    wasn't a hit).  ``cache_creation_tokens`` is excluded because it
    represents *new* content being written to cache — it's not a hit
    on prior content, just future infrastructure.

    Returns:
        ``None`` when the provider does not report cache stats
        (i.e. ``cache_read_tokens is None``).  Treating "no cache
        info" as "0% hit" would be misleading — clients should
        handle the None case explicitly (e.g. omit the cache-hit
        line from a status bar, rather than show "0%").

        ``0.0`` when ``cache_read_tokens`` is reported as 0 (the
        provider supports caching but this turn had no hits) — this
        is a real measurement, distinct from "no support".

        Otherwise the percentage in the range ``[0.0, 100.0]``.
    """
    if event.cache_read_tokens is None:
        return None
    total = event.cache_read_tokens + event.prompt_tokens
    if total == 0:
        return 0.0
    return event.cache_read_tokens / total * 100.0
