"""A message queued to a busy peer that then goes idle must still be acted on.

Three symptoms, two root causes, all reported by the perpetual-monologue
cascade — a two-sibling loop where each persona ends its turn by messaging the
other, with no driver.  It ran exactly ONE round trip and stranded, twice.

1. THE DRAIN RAN ONCE.  A message arriving WHILE the end-of-turn drain was
   running found the sender still "busy" (the daemon clears
   ``_model_running`` only after the RPC unwinds), so it took the QUEUE
   branch — onto a tier the drain had already gone past.  Nothing pops it
   afterwards: ``_on_continuation_needed`` is restored to None when the RPC
   returns.  Accepted, then stranded, permanently.

   Not a rare race for a symmetric loop — the STEADY STATE.  The sender's
   post-send narration keeps it busy across exactly the window when the fast
   half replies.  A request/response cascade never sees it.

2. ``inject_prompt_to_session`` INTO AN IDLE TARGET STARTED NOTHING, for the
   same reason ``send_to_sibling``'s idle branch didn't before #612.  That
   made the documented cascade-watchdog pattern a no-op: their 180s nudges
   fired twice and produced no turn.

3. No ``session.save`` command, so their evidence stayed a model's paraphrase
   rather than an artifact.
"""

import pytest

from shared.jaato_session import ActivityPhase, JaatoSession
from shared.message_queue import MessageQueue, SourceType


class _RacingQueue(MessageQueue):
    """Injects a message the first ``n`` times it is found empty.

    Stands in for a peer replying WHILE the drain runs — the interval that
    produced the stranding.  A static fixture cannot exhibit it: the message
    has to arrive after a pass has already gone by.
    """

    def __init__(self, arrivals=1):
        super().__init__()
        self._remaining = arrivals
        self.arrived = 0

    def __len__(self):
        n = super().__len__()
        if n == 0 and self._remaining:
            self._remaining -= 1
            self.arrived += 1
            self.put(f"late-{self.arrived}", "peer", SourceType.SIBLING)
            n = super().__len__()
        return n


def _idle_session(queue, on_continuation=None):
    s = JaatoSession.__new__(JaatoSession)
    s._message_queue = queue
    s._agent_id = "conscient"
    s._trace = lambda *a, **k: None
    s._on_prompt_injected = lambda _t: None
    s._activity_phase = ActivityPhase.IDLE
    s._is_running = False
    s._on_continuation_needed = on_continuation
    return s


def test_a_message_arriving_mid_drain_is_still_drained():
    """The reported failure: one round trip, then silence."""
    q = _RacingQueue(arrivals=1)
    q.put("first", "peer", SourceType.SIBLING)
    collected = _idle_session(q)._drain_child_messages(None)

    assert "first" in collected
    assert "late-1" in collected, "the mid-drain arrival was stranded"
    assert super(_RacingQueue, q).__len__() == 0


def test_repeated_arrivals_all_drain():
    """A loop keeps replying; each pass must pick up the next."""
    q = _RacingQueue(arrivals=3)
    q.put("first", "peer", SourceType.SIBLING)
    collected = _idle_session(q)._drain_child_messages(None)
    for i in (1, 2, 3):
        assert f"late-{i}" in collected


def test_exactly_one_continuation_fires():
    """Draining N times must not start N turns.

    The continuation is what starts the next turn; firing it per pass would
    turn one stranded message into several duplicate turns — a different bug,
    and a worse one for a loop with no driver.
    """
    q = _RacingQueue(arrivals=2)
    q.put("first", "peer", SourceType.SIBLING)
    fired = []
    _idle_session(q, fired.append)._drain_child_messages(None)

    assert len(fired) == 1, f"expected 1 continuation, got {len(fired)}"
    for token in ("first", "late-1", "late-2"):
        assert token in fired[0], "the continuation lost part of the batch"


def test_an_endless_producer_does_not_hang_the_turn():
    """Bounded: turn teardown must not spin forever.

    Losing a message is bad; never finishing the turn is worse, and the
    remainder is still there for the next drain.
    """
    q = _RacingQueue(arrivals=10_000)
    q.put("first", "peer", SourceType.SIBLING)
    _idle_session(q)._drain_child_messages(None)     # must return


def test_a_quiet_queue_still_works():
    """The ordinary case must not regress."""
    q = MessageQueue()
    q.put("hello", "peer", SourceType.SIBLING)
    assert "hello" in _idle_session(q)._drain_child_messages(None)


def test_an_empty_queue_fires_no_continuation():
    """Nothing drained means nothing to react to — no spurious turn."""
    fired = []
    _idle_session(MessageQueue(), fired.append)._drain_child_messages(None)
    assert fired == []
