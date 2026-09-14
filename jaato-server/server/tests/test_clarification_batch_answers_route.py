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

Since #989 the reply may also carry ATTACHMENTS, and the routing gains a
verdict it did not have: refusing.  The daemon validates the media before
resolving anything, and on any problem it emits an ``ErrorEvent`` and
leaves the clarification OPEN — resolving it with the payload dropped
would report an answer the user never gave, and for a voice-only answer
an empty one (#838).  The refusals are covered here too, next to the
routing they interrupt.
"""

import base64
import queue

from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The refusal gate is the whole of #989's safety story on this path: with
#: it removed the daemon resolves the clarification and the media simply
#: never travels -- an answer the user did not give, and for a voice-only
#: answer an EMPTY one reported as a success (#838).  That is exactly the
#: silent-drop shape the repo keeps filing, so the guard has to be able to
#: notice its own removal rather than being taken on trust.
REVERSIONS = [
    Reversion(
        target="jaato-server/server/core.py",
        find="""        media, accepted = self._resolve_clarification_attachments(
            relay, request_id, answer_attachments, cancelled,
        )
        if not accepted:
            return""",
        replace="        media = None",
        test="test_an_attachment_on_a_question_that_does_not_exist_is_refused",
        because="a clarification resolving with its attachments silently "
                "discarded -- the answer reaches the model with the "
                "recording gone, and nothing anywhere says so",
    ),
]


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
    """Records what ``respond_to_clarification_batch`` handed the relay.

    Mirrors the real ``ClarificationRelayHandler`` surface this method
    uses: ``resolve_response`` (including #989's keyword-only
    ``answer_attachments``) and ``pending_question_count``, which the
    daemon reads to bound an attachment's question index against the
    batch that was actually asked.
    """

    def __init__(self, *, resolves=True, question_count=None):
        self.calls = []
        self.media = []
        self._resolves = resolves
        self._question_count = question_count

    def pending_question_count(self, request_id):
        return self._question_count

    def resolve_response(self, request_id, answers, *, cancelled=False,
                         answer_attachments=None):
        self.calls.append((request_id, list(answers), cancelled))
        self.media.append(answer_attachments)
        return self._resolves


def _wav(size=64):
    """One small, decodable attachment in the canonical wire shape."""
    return {
        "mime_type": "audio/wav",
        "data": base64.b64encode(b"\x00" * size).decode("ascii"),
        "display_name": "answer.wav",
    }


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


# ---------------------------------------------------------------------
# #989 — attachments on an answer
# ---------------------------------------------------------------------

def test_answer_attachments_reach_the_relay_keyed_by_question():
    relay = _Relay(question_count=2)
    server = _server(relay=relay)

    server.respond_to_clarification_batch(
        "r1", ["", "yes"], answer_attachments={"1": [_wav()]},
    )

    assert relay.calls == [("r1", ["", "yes"], False)]
    assert list(relay.media[0]) == [1]          # normalised to an int index
    assert relay.media[0][1][0]["mime_type"] == "audio/wav"
    assert server.emitted == []


def test_an_int_key_and_a_string_key_name_the_same_question():
    """JSON gives string keys; an in-process caller may pass ints."""
    relay = _Relay(question_count=1)
    server = _server(relay=relay)

    server.respond_to_clarification_batch(
        "r1", [""], answer_attachments={1: [_wav()]},
    )

    assert list(relay.media[0]) == [1]


def test_a_text_only_answer_hands_the_relay_no_media():
    relay = _Relay(question_count=1)
    server = _server(relay=relay)

    server.respond_to_clarification_batch("r1", ["Dani"])

    assert relay.media == [None]


def test_an_attachment_on_a_question_that_does_not_exist_is_refused():
    """Refused, and the clarification left OPEN: the relay is never told."""
    relay = _Relay(question_count=1)
    server = _server(relay=relay)

    server.respond_to_clarification_batch(
        "r1", [""], answer_attachments={"7": [_wav()]},
    )

    assert relay.calls == []
    assert len(server.emitted) == 1
    assert server.emitted[0].error_type == "ClarificationAttachmentError"
    assert "1..1" in server.emitted[0].error


def test_an_undecodable_payload_is_refused_by_name():
    relay = _Relay(question_count=1)
    server = _server(relay=relay)

    server.respond_to_clarification_batch(
        "r1", [""],
        answer_attachments={"1": [{"mime_type": "audio/wav",
                                   "data": "not base64!!"}]},
    )

    assert relay.calls == []
    assert "decodable" in server.emitted[0].error


def test_a_batch_over_the_byte_cap_is_refused_rather_than_hanging():
    """Over the cap the RPC response frame is never written at all, so the
    runner's call never resolves and the turn hangs behind a clarification
    nobody can answer.  Refusing at submit is what keeps it answerable."""
    from shared.plugins.clarification.attachments import (
        MAX_CLARIFICATION_ATTACHMENT_BYTES,
    )

    relay = _Relay(question_count=2)
    server = _server(relay=relay)
    half = MAX_CLARIFICATION_ATTACHMENT_BYTES // 2 + 1024

    server.respond_to_clarification_batch(
        "r1", ["", ""],
        answer_attachments={"1": [_wav(half)], "2": [_wav(half)]},
    )

    assert relay.calls == []
    assert "cap" in server.emitted[0].error


def test_attachments_with_no_relay_waiting_are_refused_not_queued():
    """The daemon-local QueueChannel carries answer STRINGS and has nowhere
    to put bytes; accepting them there would drop the payload that WAS the
    message."""
    server = _server(pending_id="r1")

    server.respond_to_clarification_batch(
        "r1", [""], answer_attachments={"1": [_wav()]},
    )

    assert _drain(server._channel_input_queue) == []
    assert len(server.emitted) == 1
    assert server.emitted[0].error_type == "ClarificationAttachmentError"
    assert "text only" in server.emitted[0].error


def test_a_cancel_ignores_attachments_rather_than_refusing_them():
    """There is no answer to attach to; the cancel must still get through."""
    relay = _Relay(question_count=1)
    server = _server(relay=relay)

    server.respond_to_clarification_batch(
        "r1", [], cancelled=True, answer_attachments={"9": [_wav()]},
    )

    assert relay.calls == [("r1", [], True)]
    assert server.emitted == []
