"""Both resume verbs carry an attachment, so a multimodal session can be
driven after it completes (#845).

There are two ways to drive an EXISTING session — ``session.wake`` and
``inject_prompt`` — and both were text-only, while ``attachments`` sat on
``send_message``, the LIVE-session path.  A completion-gated voice session is
designed to end (``signal_completion`` makes it quiescent and releases the
runner), and the documented way back in is ``session.wake``; for a voice agent
the next input is a spoken utterance and there was no field to put it in.  The
resume path was closed to exactly the sessions #830 made possible.

What these tests pin, in the order the payload travels:

1. the command decode — attachments are payload-only, and only mappings
   survive (a positional string would be a client-side path the daemon cannot
   read);
2. the usage check — an attachment IS content (#838), so text-or-attachments,
   not text;
3. the untrusted framing — bytes cannot be wrapped, so the manifest names them
   INSIDE the wrapper and the payload never enters the prompt;
4. the drive — the attachments reach ``SendMessageRequest``, including on a
   DEFERRED wake replayed at re-attach;
5. the inject side — an attachment-bearing delivery is IDLE-ONLY, because the
   queued path folds a message into the running turn as text and cannot carry
   bytes.  A busy target is refused with ``BUSY`` and nothing enqueued rather
   than accepted with its payload dropped.
"""

import threading

import pytest

from server.command_router import CommandRouter, _decode_wake_request
from server.session_manager import (
    SessionManager,
    _PendingWake,
    _is_contentless_wake,
    _wake_attachments,
    _wrap_wake_content,
)
from shared.message_delivery import ACCEPTED, BUSY, QUEUED
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

UTTERANCE = {"mime_type": "audio/wav", "data": "QUJD",
             "display_name": "question.wav", "attachment_id": "sha256:beef"}


#: The one-line changes that put #845 back, each naming the test that must
#: notice.  See ``test_every_guard_detects_its_own_reversion`` — a guard that
#: cannot fail on its own reversion is decorative, and this suite is the only
#: thing that checks.
REVERSIONS = [
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="""        wire_attachments = list(attachments or [])
        if wire_attachments:
            # Not a caller preference: the queue cannot carry bytes, so the
            # only delivery that keeps them is a drive.  See the docstring.
            require_idle = True
""",
        replace="""        wire_attachments = list(attachments or [])
""",
        test="test_an_attachment_bearing_inject_into_a_busy_target_enqueues_nothing",
        because="an attachment-bearing inject into a BUSY target queued a "
                "message the queue cannot carry, so the payload that WAS "
                "the message was dropped and the caller told it was delivered",
    ),
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="""        wake_attachments = _wake_attachments(attachments)
        if _is_contentless_wake(text, wake_attachments):""",
        replace="""        wake_attachments = _wake_attachments(attachments)
        if not text:""",
        test="test_wake_drives_the_utterance_with_no_text_at_all",
        because="a wake carrying only an utterance was refused as empty — "
                "which for a voice session is the NORMAL shape, since the "
                "attachment is the message",
    ),
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="""    items = _wake_attachments(attachments)
    body = text or ""
""",
        replace="""    items = []
    body = text or ""
""",
        test="test_the_manifest_sits_inside_the_untrusted_boundary",
        because="media arrived with no account of itself inside the "
                "untrusted-content boundary, so a SPOKEN instruction was "
                "weighed differently from the identical typed one",
    ),
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="""                SendMessageRequest(
                    text=text, attachments=list(attachments or []),
                ),""",
        replace="""                SendMessageRequest(text=text),""",
        test="test_the_drive_puts_the_bytes_on_the_request_itself",
        because="the bytes were dropped at the last hop: every layer "
                "carried them and the request that reaches the model did not",
    ),
    Reversion(
        target="jaato-server/server/command_router.py",
        find="""    raw = p.get("attachments")
    attachments = (
        [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []
    )""",
        replace="""    attachments = []""",
        test="test_the_handler_forwards_the_utterance",
        because="session.wake went back to being text-only on the wire, "
                "which is the issue itself",
    ),
]


# ------------------------------------------------------- 1. command decode

def test_decode_reads_attachments_from_the_payload():
    sid, text, source, event_id, atts = _decode_wake_request(
        [], {"session_id": "s-1", "text": "", "source": "phone",
             "event_id": "e1", "attachments": [UTTERANCE]})
    # Blank text decodes as ``None`` (the ``or`` fallback to the positional
    # form treats "" as absent), which is why the handler passes ``text or ""``
    # onward and the usage check asks about attachments too.
    assert (sid, text, source, event_id) == ("s-1", None, "phone", "e1")
    assert atts == [UTTERANCE]


def test_decode_drops_non_mapping_attachments():
    """A positional-looking string here is a CLIENT-side path.

    The daemon cannot read it — that is the whole reason
    ``_normalize_attachments`` expands paths on the sending side — so it is
    dropped rather than handed onward as something the multimodal path would
    have to re-check.
    """
    _, _, _, _, atts = _decode_wake_request(
        [], {"session_id": "s-1", "attachments": ["/home/me/utt.wav", 7,
                                                  UTTERANCE]})
    assert atts == [UTTERANCE]


def test_decode_of_the_positional_form_is_unchanged():
    sid, text, source, event_id, atts = _decode_wake_request(
        ["s-1", "hello", "cron", "e9"], None)
    assert (sid, text, source, event_id, atts) == (
        "s-1", "hello", "cron", "e9", [])


# ------------------------------------------------------- 2. an attachment is content

def test_an_utterance_alone_is_content():
    assert _is_contentless_wake("", [UTTERANCE]) is False
    assert _is_contentless_wake("hello", []) is False
    assert _is_contentless_wake("", []) is True


def test_wake_attachments_coerces_none_to_an_owned_list():
    assert _wake_attachments(None) == []
    got = _wake_attachments([UTTERANCE])
    got.append({})
    assert len(got) == 2   # owned copy: the caller's list is not mutated


# ------------------------------------------------------- 3. untrusted framing

def test_the_manifest_sits_inside_the_untrusted_boundary():
    wrapped = _wrap_wake_content("", [UTTERANCE], "phone")
    assert wrapped.startswith("⟦UNTRUSTED-EXTERNAL-CONTENT source=wake:phone⟧")
    assert wrapped.rstrip().endswith("⟦/UNTRUSTED-EXTERNAL-CONTENT⟧")
    body = wrapped.split("⟧", 1)[1]
    assert "question.wav" in body and "audio/wav" in body
    assert "sha256:beef" in body


def test_the_manifest_never_carries_the_payload():
    """Metadata only.  Base64 inside the prompt would double the cost of the
    bytes AND bypass the provider's own media handling — the model reads the
    audio as an ``inline_data`` part, not as text."""
    assert "QUJD" not in _wrap_wake_content("", [UTTERANCE], "user")


def test_text_and_media_share_one_boundary():
    wrapped = _wrap_wake_content("what is this?", [UTTERANCE], "github")
    assert "what is this?" in wrapped
    assert "question.wav" in wrapped
    assert wrapped.count("⟦UNTRUSTED-EXTERNAL-CONTENT") == 1


def test_a_text_only_wake_is_wrapped_exactly_as_before():
    from jaato_sdk.plugins.model_provider.types import wrap_untrusted_content
    assert _wrap_wake_content("ping", [], "cron") == wrap_untrusted_content(
        "ping", source="wake:cron")


# ------------------------------------------------------- 4. the drive

class _Manager(SessionManager):
    """A SessionManager with only what the wake path touches."""

    def __init__(self, loaded=True):
        self._lock = threading.RLock()
        warm = type("S", (), {"attached_clients": {"c1"}})()
        self._sessions = {"s-1": warm} if loaded else {}
        self._pending_wakes = {}
        self._wake_seen_event_ids = {}
        self.drove = []

    def send_message_to_session(self, sid, text, attachments=None):
        self.drove.append((sid, text, list(attachments or [])))
        return True


def test_wake_drives_the_utterance_with_no_text_at_all():
    from server.session_manager import WakeOutcome
    sm = _Manager()
    outcome, _ = sm.wake_session("s-1", "", attachments=[UTTERANCE])
    assert outcome is WakeOutcome.OK
    sid, text, atts = sm.drove[0]
    assert (sid, atts) == ("s-1", [UTTERANCE])
    assert "question.wav" in text          # the manifest, not the bytes


def test_a_wake_carrying_nothing_is_still_refused():
    from server.session_manager import WakeOutcome
    sm = _Manager()
    outcome, detail = sm.wake_session("s-1", "")
    assert outcome is WakeOutcome.INVALID
    assert sm.drove == []


def test_a_deferred_wake_replays_the_bytes_it_arrived_with():
    """A wake deferred for a client that had not attached yet is driven
    later.  Replaying its text WITHOUT its utterance would drive a turn about
    nothing — the failure this issue is about, one layer in."""
    sm = _Manager()
    sm._pending_wakes["s-1"] = _PendingWake(
        text="", source="phone", wake_ref="", cascade_driver_id="cid",
        expires_at=float("inf"), attachments=[UTTERANCE])
    assert sm.drive_pending_wake("s-1") is True
    sid, text, atts = sm.drove[0]
    assert atts == [UTTERANCE]
    assert "wake:phone" in text


def test_the_drive_puts_the_bytes_on_the_request_itself():
    """The last hop, exercised for real.

    Every test above stubs ``send_message_to_session``, so none of them sees
    the ``SendMessageRequest`` it builds — and that request is where the bytes
    either reach the multimodal path or quietly stop.
    """
    from jaato_sdk.events import SendMessageRequest

    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm._sessions = {"s-1": object()}
    sent = []
    sm.handle_request = lambda client_id, sid, event: sent.append(event)

    assert sm.send_message_to_session(
        "s-1", "listen", attachments=[UTTERANCE]) is True
    assert isinstance(sent[0], SendMessageRequest)
    assert sent[0].attachments == [UTTERANCE]


def test_pending_wake_defaults_to_no_attachments():
    """Every pre-#845 construction site is unchanged."""
    pw = _PendingWake(text="t", source="user", wake_ref="", 
                      cascade_driver_id=None, expires_at=0.0)
    assert pw.attachments == []


# ------------------------------------------------------- 5. inject is idle-only

def _delivery_manager(running, offer_outcome=None):
    """A manager whose target answers the OFFER, honouring ``require_idle``."""
    sm = SessionManager.__new__(SessionManager)
    sm._lock = threading.RLock()
    sm.offers = []
    sm.drove = []

    class _RPC:
        def session_offer_message_threadsafe(self, text, *, source_id=None,
                                             source_type=None,
                                             require_idle=False, timeout=None):
            sm.offers.append({"text": text, "require_idle": require_idle})
            if offer_outcome is not None:
                return offer_outcome
            if not running:
                return "needs_turn"
            return "busy" if require_idle else "queued"

    session = type("S", (), {})()
    session.server = type("V", (), {
        "_runner_rpc": _RPC(), "_terminal_reason": None})()
    sm._sessions = {"s-1": session}
    sm.send_message_to_session = (
        lambda sid, text, attachments=None:
        sm.drove.append((sid, text, list(attachments or []))) or True)
    return sm


def test_an_attachment_bearing_inject_into_a_busy_target_enqueues_nothing():
    """The queued path folds a message into the running turn as TEXT — it is
    appended to the last tool result's model suffix, or replayed as a user
    text message — and neither shape has anywhere to put an ``inline_data``
    part.  Accepting the message and dropping the payload would report a
    delivery that cannot happen; ``busy`` is a retry-safe refusal."""
    sm = _delivery_manager(running=True)
    status = sm.deliver_prompt_to_session("s-1", "", attachments=[UTTERANCE])
    assert status == BUSY
    assert sm.offers[0]["require_idle"] is True
    assert sm.drove == []


def test_an_attachment_bearing_inject_into_an_idle_target_drives_the_bytes():
    sm = _delivery_manager(running=False)
    status = sm.deliver_prompt_to_session("s-1", "", attachments=[UTTERANCE])
    assert status == ACCEPTED
    assert sm.drove == [("s-1", "", [UTTERANCE])]


def test_a_text_only_inject_still_queues_into_a_busy_target():
    """The idle-only rule is what attachments COST, so it must not be paid by
    a caller that sent none.  A text inject is bit-identical to before."""
    sm = _delivery_manager(running=True)
    assert sm.deliver_prompt_to_session("s-1", "steer") == QUEUED
    assert sm.offers[0]["require_idle"] is False
    assert sm.drove == []


def test_the_boolean_adapter_reports_a_busy_refusal_as_undelivered():
    sm = _delivery_manager(running=True)
    assert sm.inject_prompt_to_session(
        "s-1", "", attachments=[UTTERANCE]) is False


# ------------------------------------------------------- 6. the command handler

class _Sink:
    def __init__(self):
        self.events = []

    def send_event(self, client_id, event):
        self.events.append(event)


class _RecordingManager:
    def __init__(self):
        self.calls = []

    def wake_session(self, session_id, text, source="user", event_id=None,
                     attachments=None):
        from server.session_manager import WakeOutcome
        self.calls.append({"session_id": session_id, "text": text,
                           "source": source, "attachments": attachments})
        return (WakeOutcome.OK, "woken")


def _router():
    sm, sink = _RecordingManager(), _Sink()
    return CommandRouter(sm, sink, {}), sm, sink


def test_the_handler_forwards_the_utterance():
    router, sm, sink = _router()
    router._handle_session_wake(
        "c1", [], {"session_id": "s-1", "attachments": [UTTERANCE]})
    assert sm.calls[0]["attachments"] == [UTTERANCE]
    # Blank text decodes as None; the handler normalises it so wake_session
    # never has to ask whether "no text" is None or "".
    assert sm.calls[0]["text"] == ""
    assert sink.events == []


def test_the_handler_refuses_a_wake_with_neither():
    router, sm, sink = _router()
    router._handle_session_wake("c1", [], {"session_id": "s-1"})
    assert sm.calls == []
    assert sink.events[0].error_type == "UsageError"
    assert "attachments" in sink.events[0].error


def test_the_handler_refuses_when_every_attachment_was_unusable():
    """Dropping a malformed entry is not silent in EFFECT: a wake left with
    no usable content is refused by name rather than driving an empty turn."""
    router, sm, sink = _router()
    router._handle_session_wake(
        "c1", [], {"session_id": "s-1", "attachments": ["/local/path.wav"]})
    assert sm.calls == []
    assert sink.events[0].error_type == "UsageError"
