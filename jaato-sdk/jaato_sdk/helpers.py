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
    if event.usage.cache_read_tokens is None:
        return None
    total = event.usage.cache_read_tokens + event.usage.prompt_tokens
    if total == 0:
        return 0.0
    return event.usage.cache_read_tokens / total * 100.0


def truncation_reason(
    *,
    finish_reason: Optional[str] = None,
    payload: Optional[dict] = None,
    termination_reason: Optional[str] = None,
    termination_detail: Optional[str] = None,
) -> Optional[str]:
    """Did this session end where it meant to?  ``None`` if yes.

    Returns ``None`` when the session reached a terminus it DECLARED, and
    otherwise a string NAMING the mechanism — because the caller's next act
    is usually to record a blocked/incomplete verdict, and a verdict that
    does not say what was absent is a silent skip wearing a verdict's
    clothes.

    WHY THIS IS NOT ``finish_reason != "stop"``.

    That is the obvious rule, it is what ``finish_reason``'s own field
    comment used to recommend, and it is WRONG for every schema-driven
    profile.  A profile with a ``completion_payload_schema`` ends by calling
    ``signal_completion``, which terminates the session INSIDE a tool-use
    turn — so a COMPLETE run's terminal turn reports ``"tool_use"`` and no
    later turn ever says ``"stop"``.  A consumer following the old advice
    blocked every schema-driven arm as truncated with the artefact sitting
    on disk beside the verdict.

    THE ORDER IS LOAD-BEARING:

    1. ``termination_reason`` outranks everything, INCLUDING a payload.  A
       budget refusal short-circuits before any turn runs, so
       ``finish_reason`` still holds whatever the previous turn left, and a
       payload cannot mean "complete" if the session then refused.
    2. A payload settles it regardless of ``finish_reason``.  That is the
       whole schema-profile case.
    3. Then ``"stop"`` — the plain-prose terminus.
    4. Otherwise name the mechanism.  Callers quote the string.

    Args:
        finish_reason: ``TurnCompletedEvent.finish_reason`` of the last turn.
        payload: The validated ``signal_completion`` payload, or ``None``.
        termination_reason: ``SessionTerminatedEvent.reason``, or ``None``.
        termination_detail: The event's detail/error text, when it has one.

    Returns:
        ``None`` if the session reached a declared terminus; otherwise a
        human-readable string naming what stopped it.

    Note:
        Since the terminal event is emitted AFTER the turn's own events, a
        consumer reading the stream in order has ``termination_reason``
        settled by the time it evaluates this.  Before that fix the first
        branch raced the turn event.
    """
    if termination_reason == "budget_exhausted":
        return (f"stopped at its budget ceiling: {termination_detail}"
                if termination_detail else "stopped at its budget ceiling")
    if termination_reason == "error":
        return (f"ended in a session error: {termination_detail}"
                if termination_detail else "ended in a session error")
    if payload is not None:
        return None
    if finish_reason == "stop":
        return None
    if finish_reason == "tool_use":
        return ("ended mid-tool-loop (finish_reason='tool_use') having "
                "signalled no completion payload")
    return f"finish_reason={finish_reason!r}"
