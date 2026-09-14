"""A message that carries an attachment is not empty (#838).

THE GAP.  ``handle_request``'s ``SendMessageRequest`` branch decided whether
to dispatch a model turn by looking at the message TEXT and nothing else::

    if not (message_text and message_text.strip()):
        self._emit_to_client(client_id, TurnCompletedEvent())
        return                                  # no model turn, ever

``event.attachments`` sat on the same object, read twenty lines later to be
handed to ``server.send_message`` -- on the path this branch had already
returned from.  So an attachment-only send never reached the wire, and the
caller was told the turn completed.

Measured, one daemon, one session profile, the same 88 KB ``audio/wav``
attachment, only the text differing:

=============================  =========================================
``"Answer what you hear."``    provider 400, surfaced as ``AgentError``
``""``                         one ``TURN_COMPLETED``, nothing else
=============================  =========================================

The first row is the diagnostic one: the same attachment WITH text got far
enough to be refused by the upstream, so the empty-text form was dropped
BEFORE the wire rather than failing at it.

WHY THIS SHAPE MATTERS.  For an image, blank text is unusual -- there is
normally a question about the picture.  For AUDIO it is the normal case: the
attachment IS the message.  A voice turn is "here is what I said", and
supplying text alongside asks a second question the persona then has to
choose between.  So the natural voice request --
``session.complete("", attachments=[utterance])`` -- was exactly the one that
silently did nothing.

WHY IT WENT UNNOTICED.  Every layer below this one already handled it.
``JaatoSession._parts_from_user_message`` says so in its own docstring ("an
empty ``message`` (image-only turn) yields parts with no text"), the runner
RPC accepts ``""`` as a valid ``str`` prompt, and the standalone-WS handler
dispatches an attachment-only send with no emptiness check at all.  One site
on the SDK path disagreed with all of them.

AND THE FAILURE MODE IS THE BAD KIND.  Not an exception, not a refusal naming
a reason -- a completed turn.  A caller reads an empty payload and cannot tell
"the model had nothing to say" from "nothing was ever asked".  So the blank
branch that remains -- a request that arrives with no text AND no attachments
-- now names itself, while the ``%name --help`` case it was built for still
closes in silence, because there the help WAS the answer.

THE CONTRACT, in the order these guards assert it:

1. blank text + an attachment dispatches a model turn, attachment intact;
2. whitespace-only text is blank for this purpose too, and still dispatches;
3. text + an attachment is unaffected (the row that already worked);
4. no text and no attachments is refused BY NAME, not closed in silence;
5. a solely-``%name --help`` message still closes quietly -- it was served.

Guards 1 and 4 declare their reversions to the meta-suite below, so each is
re-proved to notice its own defect on every commit rather than on the day it
was written.
"""
import threading
from types import SimpleNamespace

from jaato_sdk.events import (
    ErrorEvent,
    SendMessageRequest,
    TurnCompletedEvent,
)
from jaato_sdk.plugins.base import HelpLines
from server.session_manager import SessionManager
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="        if (message_text and message_text.strip()) or "
             "event.attachments:\n            return False",
        replace="        if message_text and message_text.strip():\n"
                "            return False",
        test="test_blank_text_with_an_attachment_reaches_the_model",
        because="a message whose whole payload is an attachment being "
                "dropped before the wire, so the natural voice turn "
                "``complete(\"\", attachments=[utterance])`` reports a "
                "completed turn and sends nothing",
    ),
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="""        if client_id is not None:
            if not (event.text and event.text.strip()):
                self._emit_to_client(client_id, ErrorEvent(
                    error="Empty message: no text and no attachments — "
                          "nothing was sent to the model.",
                    error_type="EmptyMessageError",
                ))
            self._emit_to_client(client_id, TurnCompletedEvent())""",
        replace="""        if client_id is not None:
            self._emit_to_client(client_id, TurnCompletedEvent())""",
        test="test_a_message_with_nothing_in_it_is_refused_by_name",
        because="a send that asked for nothing closing in the silence that "
                "is indistinguishable from a turn which ran and produced "
                "nothing",
    ),
]


# The wire shape a client-normalized attachment has by the time it reaches
# the daemon: ``{mime_type, data: base64-str, display_name}``.
UTTERANCE = {
    "mime_type": "audio/wav",
    "data": "UklGRiQAAABXQVZF",     # a base64 stub, never decoded here
    "display_name": "utterance.wav",
}


def _sm(emitted):
    """A bare SessionManager wired to reach the blank-text branch.

    ``emitted`` collects ``(client_id, event)`` pairs from
    ``_emit_to_client``.  The returned ``send_calls`` records every
    ``server.send_message(text, attachments)`` so a test can assert both
    what ran and what it was given, and ``sent`` lets a test wait on the
    daemon thread the dispatch path starts.

    A ``prompt_library`` stub is wired so ``%name --help`` resolves --
    that is the one case whose silence is correct, and it has to be
    reachable here for guard 5 to mean anything.
    """
    sm = SessionManager.__new__(SessionManager)
    sm._event_callback = lambda cid, ev: emitted.append((cid, ev))
    sm._client_to_session = {}
    sm._save_session = lambda session: None

    prompt_plugin = SimpleNamespace(
        _execute_prompt_command=lambda args: HelpLines(
            lines=[("usage: %foo", ""), ("  --help", "")]
        )
    )
    registry = SimpleNamespace(get_plugin=lambda name: prompt_plugin)
    runtime = SimpleNamespace(registry=registry)

    send_calls = []
    sent = threading.Event()

    def _send_message(text, attachments=None):
        send_calls.append((text, attachments))
        sent.set()

    server = SimpleNamespace(
        _runtime=runtime,
        get_all_session_env=lambda: {},
        send_message=_send_message,
    )
    session = SimpleNamespace(
        server=server,
        workspace_path="/ws",
        user_inputs=[],
        is_dirty=False,
        last_activity="",
    )
    sm.get_session = lambda sid: session
    return sm, send_calls, sent


def _types(emitted):
    return [type(ev).__name__ for _, ev in emitted]


def test_blank_text_with_an_attachment_reaches_the_model():
    """Guard 1 -- the voice turn.  The attachment IS the message."""
    emitted = []
    sm, send_calls, sent = _sm(emitted)

    sm.handle_request("client-1", "sess-1",
                      SendMessageRequest(text="", attachments=[UTTERANCE]))

    assert sent.wait(timeout=5), (
        "an attachment-only message was dropped before the wire: "
        f"events were {_types(emitted)}"
    )
    (text, attachments), = send_calls
    assert text == ""
    # The attachment survives intact -- dispatching a turn that lost the one
    # thing the message carried would be the same defect one layer down.
    assert attachments == [UTTERANCE]
    # No synthetic close: the real turn owns its own TurnCompletedEvent.
    assert "TurnCompletedEvent" not in _types(emitted)
    assert "ErrorEvent" not in _types(emitted)


def test_whitespace_only_text_with_an_attachment_reaches_the_model():
    """Guard 2 -- ``"  \\n"`` is blank for this purpose, not content.

    The old branch treated whitespace-only and empty identically, and that
    part was right; what it got wrong was concluding the MESSAGE was empty.
    """
    emitted = []
    sm, send_calls, sent = _sm(emitted)

    sm.handle_request("client-1", "sess-1",
                      SendMessageRequest(text="   \n",
                                         attachments=[UTTERANCE]))

    assert sent.wait(timeout=5), _types(emitted)
    (text, attachments), = send_calls
    assert text == "   \n"
    assert attachments == [UTTERANCE]


def test_text_with_an_attachment_is_unaffected():
    """Guard 3 -- the row that already worked keeps working."""
    emitted = []
    sm, send_calls, sent = _sm(emitted)

    sm.handle_request("client-1", "sess-1",
                      SendMessageRequest(text="Answer what you hear.",
                                         attachments=[UTTERANCE]))

    assert sent.wait(timeout=5), _types(emitted)
    (text, attachments), = send_calls
    assert text == "Answer what you hear."
    assert attachments == [UTTERANCE]


def test_a_message_with_nothing_in_it_is_refused_by_name():
    """Guard 4 -- no text, no attachments: say so, don't just close.

    A bare ``TurnCompletedEvent`` is what a turn that RAN and produced
    nothing also looks like.  The caller has to be able to tell the two
    apart, so this path names itself before it closes.
    """
    emitted = []
    sm, send_calls, _sent = _sm(emitted)

    sm.handle_request("client-1", "sess-1", SendMessageRequest(text=""))

    assert send_calls == [], "nothing to send, so nothing should have been sent"
    errors = [ev for _, ev in emitted if isinstance(ev, ErrorEvent)]
    assert errors, f"the empty send closed in silence: {_types(emitted)}"
    assert errors[0].error_type == "EmptyMessageError"
    assert "attachment" in errors[0].error.lower()
    # The turn lifecycle still closes -- a client waiting on a per-message
    # completion signal must not be left to its stall detector.
    assert any(isinstance(ev, TurnCompletedEvent) for _, ev in emitted)
    assert all(cid == "client-1" for cid, _ in emitted)


def test_a_help_only_message_still_closes_in_silence():
    """Guard 5 -- ``%name --help`` was answered, so there is nothing to report.

    This is the case the blank-text branch was built for.  It arrives with
    real text, which help interception consumes; the help itself already
    reached the client as a ``HelpTextEvent``, so naming an error here would
    report a failure to a caller that got exactly what it asked for.
    """
    emitted = []
    sm, send_calls, _sent = _sm(emitted)

    sm.handle_request("client-1", "sess-1",
                      SendMessageRequest(text="%foo --help"))

    assert send_calls == []
    assert "HelpTextEvent" in _types(emitted)
    assert "TurnCompletedEvent" in _types(emitted)
    assert not any(isinstance(ev, ErrorEvent) for _, ev in emitted)
