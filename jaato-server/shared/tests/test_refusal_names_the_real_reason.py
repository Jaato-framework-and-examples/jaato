"""A refusal must name the failure that happened, not the likeliest one.

``deliver_prompt_to_session`` distinguishes five outcomes.  The refusals that
quoted it collapsed all of them into **"no live runner channel"** — a
hardcoded likely-cause that was wrong for four of the five.

Observed live on a cascade: a peer that was alive, whose runner channel was
fine, whose delivery timed out because the daemon's event loop did not
schedule the coroutine within 7s.  The sending model was told its sibling had
no channel — which reads as "your peer is gone" and invites exactly the wrong
recovery.
"""

from __future__ import annotations

import threading

import pytest

from server.session_manager import SessionManager, _delivery_failure_reason


def test_each_status_renders_as_itself():
    from shared.message_delivery import BUSY, NO_SESSION, TERMINATED, UNREACHABLE

    assert "not loaded" in _delivery_failure_reason(NO_SESSION)
    assert "terminated" in _delivery_failure_reason(TERMINATED)
    assert "mid-turn" in _delivery_failure_reason(BUSY)
    assert "NOT CONFIRMED" in _delivery_failure_reason(UNREACHABLE)


def test_no_status_claims_a_cause_it_cannot_know():
    """The banned string, and the reason it was banned.

    "no live runner channel" is ONE of five failures.  Emitting it for the
    others tells a sender something false about its peer.
    """
    from shared.message_delivery import BUSY, NO_SESSION, TERMINATED, UNREACHABLE

    for status in (NO_SESSION, TERMINATED, UNREACHABLE, BUSY):
        rendered = _delivery_failure_reason(status)
        assert "no live runner channel" not in rendered, (
            f"{status!r} renders a cause it cannot know: {rendered!r}"
        )


def test_unreachable_says_not_confirmed_rather_than_not_delivered():
    """The distinction a retrying sender needs.

    On a timeout the offer may have been enqueued runner-side and only the
    acknowledgement lost.  A sender that retries may duplicate; one that gives
    up may abandon a message that arrived.  Neither is safe unless it is told.
    """
    from shared.message_delivery import UNREACHABLE

    rendered = _delivery_failure_reason(UNREACHABLE)
    assert "NOT CONFIRMED" in rendered
    assert "may still have been enqueued" in rendered


def test_an_unknown_status_is_reported_not_guessed():
    """A status nobody mapped must surface AS a status.

    Falling back to any of the four real reasons would invent a cause; that is
    the defect, one level up.
    """
    rendered = _delivery_failure_reason("some_future_status")
    assert "some_future_status" in rendered
    assert "not loaded" not in rendered
    assert "terminated" not in rendered


def test_the_sibling_refusal_quotes_the_status_it_got():
    """End to end through the real ``deliver_sibling_message``."""
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {}
    sm._lock = threading.RLock()
    sm._sibling_pending = {}
    sm._sibling_exchanges = {}

    sender = type("S", (), {})()
    sender.session_id, sender.cascade_driver_id = "s-a", "cid-1"
    sender.sibling_name = "a"
    target = type("S", (), {})()
    target.session_id, target.cascade_driver_id = "s-b", "cid-1"
    target.sibling_name = "b"
    # server=None -> deliver_prompt_to_session returns UNREACHABLE
    target.server = None
    sender.server = None
    sm._sessions = {"s-a": sender, "s-b": target}
    sm._get_persisted_sessions = lambda **kw: []

    result = sm.deliver_sibling_message("s-a", "b", "hello")

    assert result["status"] == "refused"
    assert "NOT CONFIRMED" in result["error"], (
        f"the refusal did not carry the real reason: {result['error']!r}"
    )
    assert "no live runner channel" not in result["error"]
