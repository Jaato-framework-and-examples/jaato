"""``respond_to_clarification_batch`` must reach whichever side is waiting.

A batch of answers has two possible destinations and the daemon picks
between them at reply time:

* a **runner-tier** session parked on the ``ClarificationRelayHandler``
  future — the runner-side channel is awaiting it over RPC;
* a **daemon-local** session whose ``QueueChannel`` is reading the input
  queue one answer at a time.

Both are covered here, in each case for an answer AND for a cancel,
because the cancel is what makes an unanswerable clarification
recoverable at all: a turn blocked inside a tool call does not respond to
Ctrl+C, so if the cancel does not reach the waiting side the session is
finished (#704).
"""

import queue


def _server(pending_id=None, relay=None):
    """A ``JaatoServer`` with only the state this method reads.

    Constructed via ``__new__`` deliberately: a real ``initialize()``
    would stand up plugins, a provider and a runtime to exercise a method
    whose whole job is a three-way routing decision.
    """
    from server.core import JaatoServer

    server = JaatoServer.__new__(JaatoServer)
    server._clarification_relay_handler = relay
    server._pending_clarification_request_id = pending_id
    server._channel_input_queue = queue.Queue()
    server.emitted = []
    server.emit = server.emitted.append
    return server


class _Relay:
    """Records what ``respond_to_clarification_batch`` handed the relay."""

    def __init__(self, *, resolves=True):
        self.calls = []
        self._resolves = resolves

    def resolve_response(self, request_id, answers, *, cancelled=False):
        self.calls.append((request_id, list(answers), cancelled))
        return self._resolves


def _drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def test_a_runner_tier_answer_goes_to_the_relay_not_the_queue():
    relay = _Relay()
    server = _server(relay=relay)

    server.respond_to_clarification_batch("r1", ["1", "yes"])

    assert relay.calls == [("r1", ["1", "yes"], False)]
    assert _drain(server._channel_input_queue) == []


def test_a_runner_tier_cancel_reaches_the_relay_as_a_cancel():
    relay = _Relay()
    server = _server(relay=relay)

    server.respond_to_clarification_batch("r1", [], cancelled=True)

    assert relay.calls == [("r1", [], True)]


def test_answers_fall_through_to_the_queue_when_no_relay_is_waiting():
    """A relay that has no future for this id must not swallow the reply —
    a daemon-local session's QueueChannel is the one waiting."""
    relay = _Relay(resolves=False)
    server = _server(pending_id="r1", relay=relay)

    server.respond_to_clarification_batch("r1", ["1", "yes"])

    assert relay.calls == [("r1", ["1", "yes"], False)]
    assert _drain(server._channel_input_queue) == ["1", "yes"]


def test_a_daemon_local_cancel_sends_the_sentinel_the_channel_understands():
    """One ``cancel`` ends the whole request: QueueChannel stops reading on
    it, so queueing per-question answers after it would strand them."""
    server = _server(pending_id="r1")

    server.respond_to_clarification_batch("r1", ["1", "yes"], cancelled=True)

    assert _drain(server._channel_input_queue) == ["cancel"]


def test_an_answer_for_an_unknown_request_is_reported_not_swallowed():
    server = _server(pending_id="other")

    server.respond_to_clarification_batch("r1", ["1"])

    assert len(server.emitted) == 1
    assert "r1" in server.emitted[0].error
    assert _drain(server._channel_input_queue) == []
