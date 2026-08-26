"""``offer_message`` decides where the state lives, and decides atomically.

Step 2.  The daemon's ``_model_running`` is a REPLICA of the session's
``_is_running`` that clears strictly later -- only once
``session.send_message`` returns and the daemon's model thread unwinds.
Measured ~30s of staleness live.  A delivery decided on the replica can be
decided on state that has already changed, and a message queued into a turn
that has ended is drained by nothing.

Two properties are tested here, and the second is the one that makes
``"queued"`` a guarantee rather than a prediction:

1.  The answer follows the SESSION's own turn state.
2.  The check-and-enqueue cannot interleave with the turn's
    ``_is_running = False`` flip, because both hold ``_delivery_lock``.
"""

import threading

import pytest

from shared.jaato_session import JaatoSession
from shared.message_queue import MessageQueue, SourceType


def _session(running: bool) -> JaatoSession:
    """A JaatoSession with only what ``offer_message`` touches.

    Built via ``__new__`` deliberately: a real ``__init__`` drags in a
    runtime, a provider and a plugin registry, none of which this decision
    consults -- and a test that needed them would be testing something else.
    """
    s = JaatoSession.__new__(JaatoSession)
    s._delivery_lock = threading.Lock()
    s._is_running = running
    s._message_queue = MessageQueue()
    s._agent_id = "main"
    s._on_prompt_injected = None
    s._trace = lambda _msg: None
    return s


def test_a_running_turn_queues_and_says_so():
    s = _session(running=True)

    outcome = s.offer_message("mid-turn", source_id="peer",
                              source_type=SourceType.SIBLING)

    assert outcome == "queued"
    assert len(s._message_queue) == 1, (
        "queued must mean the message is actually in the queue the "
        "end-of-turn drain reads"
    )


def test_an_idle_session_asks_for_a_turn_and_queues_NOTHING():
    """The strand, inverted.

    Enqueueing here would put the message in a queue whose only drainers are
    mid-turn yield points and the end-of-turn drain -- both of which need a
    turn that is not running.  So it must NOT be enqueued: the caller is told
    to start one instead.
    """
    s = _session(running=False)

    outcome = s.offer_message("wake up", source_id="watchdog",
                              source_type=SourceType.USER)

    assert outcome == "needs_turn"
    assert len(s._message_queue) == 0, (
        "a message queued into an idle session is drained by nothing -- "
        "needs_turn must leave the queue untouched"
    )


def test_the_decision_is_taken_under_the_delivery_lock():
    """The atomicity claim, made falsifiable.

    ``_run_chat_loop`` flips ``_is_running = False`` under ``_delivery_lock``
    precisely so an offer that has already observed True finishes its enqueue
    BEFORE the flip -- and therefore before the final drain that follows it.
    Holding the lock here stands in for that flip: if ``offer_message`` did
    not take it, this would return immediately and the window would be open.
    """
    s = _session(running=True)
    done = threading.Event()

    def _offer():
        s.offer_message("racer", source_type=SourceType.SIBLING)
        done.set()

    with s._delivery_lock:
        t = threading.Thread(target=_offer, daemon=True)
        t.start()
        blocked = not done.wait(timeout=0.5)
        assert blocked, (
            "offer_message did not take _delivery_lock, so a check-and-"
            "enqueue can interleave with the turn's _is_running flip and "
            "land the message behind a drain that already ran"
        )
        assert len(s._message_queue) == 0, "it enqueued without the lock"

    assert done.wait(timeout=5.0), "offer_message never completed"
    t.join(timeout=5.0)
    assert len(s._message_queue) == 1


def test_callbacks_fire_outside_the_lock():
    """``_delivery_lock`` must have no re-entrancy surface.

    ``_on_prompt_injected`` is caller-supplied; invoking it while holding the
    lock would let any consumer deadlock the delivery path by touching the
    session from its callback.
    """
    s = _session(running=True)
    seen = {}

    def _cb(text):
        # Must be free to take the lock -- i.e. we are not inside it.
        seen["locked"] = s._delivery_lock.locked()
        seen["text"] = text

    s._on_prompt_injected = _cb
    s.offer_message("with callback", source_type=SourceType.SIBLING)

    assert seen["text"] == "with callback"
    assert seen["locked"] is False, "callback fired while holding the lock"
