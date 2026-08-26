"""A refusal must name the failure that happened, not the likeliest one.

``deliver_prompt_to_session`` distinguishes five outcomes.  The refusals that
quoted it collapsed all of them into **"no live runner channel"** — a
hardcoded likely-cause that was wrong for four of the five.

Observed live on a cascade: a peer that was alive, whose runner channel was
fine, whose delivery timed out because the daemon's event loop did not
schedule the coroutine within 7s.  The sending model was told its sibling had
no channel — which reads as "your peer is gone" and invites exactly the wrong
recovery.

THE SAME COLLAPSE THEN TURNED UP INSIDE ``unreachable`` ITSELF, and these
tests pinned it.  Five producers shared that word, and the single reason
attached to it described only one of them: "on a timeout the message may
still have been enqueued and only the acknowledgement lost".  For the other
four nothing was ever offered, so the sentence was not vague, it was FALSE.
It warned a sender about a duplicate that could not exist -- and a careful
sender, told it might duplicate, declines to re-send a message that in fact
never arrived.

The test below that exercised this end to end set ``target.server = None`` --
the structural case, where nothing is sent -- and asserted the "NOT CONFIRMED"
wording.  So the guard pinned the false statement to the very case that
falsified it.  Kept as a lesson: a test can hold a defect in place as firmly
as it holds a contract, and it looks identical from the outside.
"""

from __future__ import annotations

import threading

import pytest

from server.session_manager import SessionManager, _delivery_failure_reason


def test_each_status_renders_as_itself():
    from shared.message_delivery import (
        BUSY, NO_SESSION, NOT_CONFIRMED, TERMINATED, UNREACHABLE,
    )

    assert "not loaded" in _delivery_failure_reason(NO_SESSION)
    assert "terminated" in _delivery_failure_reason(TERMINATED)
    assert "mid-turn" in _delivery_failure_reason(BUSY)
    assert "NOTHING WAS SENT" in _delivery_failure_reason(UNREACHABLE)
    assert "MAY DELIVER IT TWICE" in _delivery_failure_reason(NOT_CONFIRMED)


def test_no_status_claims_a_cause_it_cannot_know():
    """The banned string, and the reason it was banned.

    "no live runner channel" is ONE of five failures.  Emitting it for the
    others tells a sender something false about its peer.
    """
    from shared.message_delivery import (
        BUSY, NO_SESSION, NOT_CONFIRMED, TERMINATED, UNREACHABLE,
    )

    for status in (NO_SESSION, TERMINATED, UNREACHABLE, NOT_CONFIRMED, BUSY):
        rendered = _delivery_failure_reason(status)
        assert "no live runner channel" not in rendered, (
            f"{status!r} renders a cause it cannot know: {rendered!r}"
        )


def test_the_two_transport_failures_give_opposite_retry_advice():
    """The distinction a retrying sender needs -- and it is a FORK, not a note.

    ``not_confirmed``: an offer went out and its answer was lost, so the
    message may be in the target's queue.  Re-sending may duplicate.

    ``unreachable``: nothing went out at all.  Re-sending is safe and cannot
    duplicate, because there is nothing to duplicate.

    These were ONE word, and the shared reason carried the first meaning.  A
    sender following it on the second case declines to re-send a message that
    definitely never arrived -- the word meant to prevent a wrong action
    caused one.
    """
    from shared.message_delivery import NOT_CONFIRMED, UNREACHABLE

    maybe = _delivery_failure_reason(NOT_CONFIRMED)
    assert "may be in its queue" in maybe
    assert "MAY DELIVER IT TWICE" in maybe

    never = _delivery_failure_reason(UNREACHABLE)
    assert "NOTHING WAS SENT" in never
    assert "SAFE" in never

    # The regression, stated as the thing that must not come back: the
    # structural failure must never carry the duplicate warning.  Checked as
    # a PROPERTY of the rendered text rather than by comparing the two
    # strings, because they could both drift and still be wrong together.
    assert "DELIVER IT TWICE" not in never
    assert "may be in its queue" not in never


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
    # ``server = None`` means NOTHING WAS SENT.  This assertion used to read
    # "NOT CONFIRMED", pinning the duplicate warning onto the one case where
    # no duplicate is possible.
    assert "NOTHING WAS SENT" in result["error"], (
        f"the refusal did not carry the real reason: {result['error']!r}"
    )
    assert "DELIVER IT TWICE" not in result["error"], (
        "a structural failure told the sender its retry might duplicate; "
        f"nothing was ever offered: {result['error']!r}"
    )
    assert "no live runner channel" not in result["error"]
