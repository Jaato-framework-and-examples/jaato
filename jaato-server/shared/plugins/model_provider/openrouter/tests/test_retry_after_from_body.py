"""A server that says "wait 120" must be waited on for 120 seconds.

OpenRouter's in-flight concurrency limit returns HTTP 402 with the hint in
the JSON BODY, at ``error.metadata.headers['Retry-After']`` — not as an
HTTP header.  ``retry_utils.get_retry_after`` reads ``exc.retry_after`` and
``exc.response.headers``, so it finds nothing there, and the ladder falls
back to a generic guess.

Measured (jaato #719): a 402 carrying ``Retry-After: 120`` was retried at
1.4s, 1.5s, 3.5s and 6.3s — every attempt burned in about thirteen seconds
against a two-minute instruction, killing the turn.  The delays are
themselves the proof the hint was never read: ``calculate_backoff`` already
lets a hint override ``max_delay``, so a hint that HAD been found would
have produced a 120s wait.

Note what is NOT broken, since an earlier reading of this bug got it wrong:
``max_delay`` does not clamp a server hint.  ``retry_utils.py:311`` does
``if retry_after and retry_after > delay: delay = retry_after``.  The only
defect was finding the hint.
"""

import pytest

from shared.retry_utils import RetryConfig, calculate_backoff
from shared.plugins.model_provider.openrouter.provider import (
    OpenRouterProvider, _retry_after_from_body,
)


class _BodyError(Exception):
    """An exception carrying only a parsed JSON body, as the SDK does."""

    def __init__(self, body):
        super().__init__("api status error")
        self.body = body


def _in_flight_402(retry_after="120"):
    """The exact payload shape observed in the wild."""
    return {
        "error": {
            "message": "This request would exceed your available credits "
                       "given your current in-flight requests.",
            "code": 402,
            "metadata": {
                "reason": "in_flight_budget_exhausted",
                "limit_source": "openrouter_in_flight_budget",
                "headers": {"Retry-After": retry_after},
            },
        }
    }


def test_hint_is_read_from_the_body() -> None:
    assert _retry_after_from_body(_BodyError(_in_flight_402())) == 120.0


def test_provider_surfaces_the_hint_to_the_retry_ladder() -> None:
    """``with_retry`` asks the PROVIDER first; None falls through."""
    assert OpenRouterProvider().get_retry_after(_BodyError(_in_flight_402())) == 120.0


def test_the_hint_actually_changes_the_wait() -> None:
    """The bug was operational, so assert the operational consequence.

    Without the hint the ladder is exhausted long before the server would
    have accepted another request; with it, every attempt waits as told.
    """
    cfg = RetryConfig()
    hint = OpenRouterProvider().get_retry_after(_BodyError(_in_flight_402()))
    for attempt in (1, 2, 3, 4):
        blind = calculate_backoff(attempt, cfg, None)
        honoured = calculate_backoff(attempt, cfg, hint)
        assert blind < 30.0, "sanity: the blind ladder is short"
        assert honoured == pytest.approx(120.0), (
            f"attempt {attempt}: waited {honoured}s against a 120s "
            f"instruction — the hint is not reaching calculate_backoff"
        )


def test_hint_overrides_the_client_max_delay() -> None:
    """A server instruction is not a guess, so max_delay must not clamp it."""
    cfg = RetryConfig()
    assert cfg.max_delay < 120.0, "fixture assumes the default 30s ceiling"
    assert calculate_backoff(1, cfg, 120.0) == pytest.approx(120.0)


@pytest.mark.parametrize("body", [
    None,
    "not-a-dict",
    {},
    {"error": "not-a-dict"},
    {"error": {}},
    {"error": {"metadata": {}}},
    {"error": {"metadata": {"headers": {}}}},
    {"error": {"metadata": {"headers": {"Retry-After": "soon"}}}},
    {"error": {"metadata": {"headers": {"Retry-After": "-5"}}}},
])
def test_unparseable_bodies_fall_through_rather_than_raise(body) -> None:
    """Returning None hands off to the standard readers.

    A malformed body must cost nothing: the caller still has
    ``retry_utils.get_retry_after`` and the generic ladder behind it.  An
    exception here would convert a retryable error into a crash.
    """
    assert _retry_after_from_body(_BodyError(body)) is None
    assert OpenRouterProvider().get_retry_after(_BodyError(body)) is None


def test_a_plain_exception_is_not_mistaken_for_a_hint() -> None:
    assert OpenRouterProvider().get_retry_after(Exception("boom")) is None
