"""A user send that arrives mid-wind-down keeps its bytes (#877).

THE SYMPTOM.  A session that sends audio every turn loses every SECOND
utterance, and the loss surfaces as a provider refusal several layers from
its cause::

    turn 1: OK    "Hola, soy el asistente virtual de ..."
    turn 2: FAIL  400 This model requires that either input content or
                      output modality contain audio.
    turn 3: OK
    turn 4: FAIL

Identical bytes every time, so content is not a factor.  Eight distinct
chunks give the same parity (00/02/04/06 pass, 01/03/05/07 fail), so it is
not id-keyed dedup either.  The parity is structural: it asks only whether
a send landed in the daemon's turn wind-down window.

IT READS AS A #850 REGRESSION AND IS NOT ONE.  ``_evict_consumed_media``
runs at the START of a turn and strictly BEFORE the new user message is
appended, on both chat loops, so it cannot reach the turn being sent --
driving four attachment-bearing sends straight at ``JaatoSession`` puts
``[48044, 48044, 48044, 48044]`` audio bytes on the wire, at any N.  What
#850 changed was the accident that hid this: before it, the previous
utterance was still in history, so the audio-less request satisfied the
model's "must contain audio" precondition by accident and the model
re-answered the OLD clip.  #850 removed the safety net and turned a quiet
wrong-answer bug into a loud 400.

THE DEFECT, and it is above the session.  ``ask()`` returns on the first
``TurnCompletedEvent``, which the daemon emits from INSIDE the turn --
before the model thread's ``finally`` clears ``_model_running``, a race
``core.py`` documents against itself.  So the caller's next send
structurally lands in the wind-down window, where ``JaatoServer.
send_message`` took one of two branches that both read ``text`` and nothing
else:

===============================  ========================================
``session_offer_message(...)``   no ``attachments`` parameter at all, and
                                 ``require_idle`` left False -- a
                                 ``"queued"`` answer emitted
                                 ``MidTurnPromptQueuedEvent(text=...)``
                                 and the bytes were gone, unannounced
``_pending_continuations``       a ``List[str]``: text stashed,
                                 attachments dropped
===============================  ========================================

The stash was then drained as a brand-new TEXT-ONLY turn, whose
``_evict_consumed_media`` correctly purged the previous turn's audio.  The
request reached an audio-requiring model with no audio anywhere, and the
turn failed -- leaving the session genuinely idle, so the NEXT send drove a
real turn with its bytes.  OK / FAIL / OK / FAIL.

THE RULE IT VIOLATED ALREADY EXISTED.  #845 established it for
``SessionManager.deliver_prompt_to_session``: a queued message is folded
into the running turn as TEXT and has nowhere to put an ``inline_data``
part, so *"a busy target answers BUSY with NOTHING enqueued rather than
accepting the message and dropping the payload that WAS the message."*
The user-facing send path never got that treatment.

THE CONTRACT, in the order these guards assert it:

1. every arrival shape carries the bytes -- idle, session-busy, and
   this-thread-unwinding alike;
2. an attachment-bearing offer is idle-only, so it can never be traded for
   a ``MidTurnPromptQueuedEvent``;
3. a path that somehow still cannot carry bytes refuses BY NAME
   (``AttachmentDropped``) instead of proceeding as text;
4. a text-only send is completely unaffected -- it still queues;
5. the drain merges ALL of the stash, attachments concatenating;
6. the payoff: THREE consecutive attachment-bearing turns on one session
   all reach a provider that refuses an audio-less request.

WHY THREE AND NOT TWO.  This is a parity bug, so a two-turn test passes
with the defect fully intact: turn 1 lands idle and turn 2 fails, which any
"the first one worked" assertion reads as a pass.  The issue says so
explicitly, and guard 6 is written to its acceptance criterion.
"""

import asyncio
import io
import threading
import wave
from unittest.mock import MagicMock

from jaato_sdk.events import ErrorEvent, MidTurnPromptQueuedEvent
from jaato_sdk.media_identity import ATTACHMENT_ID_KEY, mint_attachment_id
from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    TurnResult,
)

from server.core import JaatoServer, merge_pending_continuations
from server.runner_rpc_client import RunnerRPCClient
from shared.jaato_session import JaatoSession
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/server/runner_rpc_client.py",
        find="""        if attachments:
            # Not a caller preference: the queue cannot carry bytes, so the
            # only delivery that keeps them is a drive.  See the docstring.
            require_idle = True
""",
        replace="",
        test="test_the_offer_rpc_raises_require_idle_for_a_payload",
        because="an attachment-bearing offer reaching the session with "
                "require_idle unset, so a busy session queues it as TEXT "
                "and the payload that WAS the message is discarded",
    ),
    Reversion(
        target="jaato-server/server/core.py",
        find="""                self._pending_continuations.append(
                    (text, list(attachments or [])),
                )""",
        replace="""                self._pending_continuations.append(
                    (text, []),
                )""",
        test="test_three_consecutive_audio_turns_all_reach_the_wire_with_audio",
        because="a send arriving while the model thread unwinds being "
                "stashed as text alone, so the drain starts an audio-less "
                "turn and an audio-requiring model refuses every second one",
    ),
]


# ============================================================ fixtures

def _wav(seconds: float = 1.0, fill: bytes = b"\x01\x02") -> bytes:
    """A real WAV file, so nothing downstream has to pretend it parsed."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(fill * int(24000 * seconds))
    return buf.getvalue()


def _utterance(fill: bytes = b"\x01\x02") -> dict:
    """One wire attachment, id minted the way the SDK client mints it.

    ``fill`` distinguishes the clips so guard 6 can assert each turn
    carried ITS OWN utterance rather than merely some audio -- the issue's
    eight-chunk run is the case where a transcript for part 02 that is
    really part 01's content would otherwise pass.
    """
    data = _wav(fill=fill)
    return {
        "mime_type": "audio/wav",
        "data": data,
        "display_name": "utterance.wav",
        ATTACHMENT_ID_KEY: mint_attachment_id(data),
    }


class _StubServer:
    """The narrowest ``self`` ``JaatoServer.send_message`` can run against.

    The two methods under test are BORROWED from the real class rather than
    reimplemented, so these guards exercise shipped code; everything else is
    the surface that code touches and nothing more.

    Attributes:
        outcome: what the runner's offer answers.  ``"needs_turn"`` (the
            session is idle), ``"queued"`` (it took the message) or
            ``"busy"`` (running, and ``require_idle`` was set).
        _model_running: whether THIS daemon's model thread is still
            unwinding -- the state that decides drive-vs-stash, and the one
            ``ask()`` structurally races.
        offers: every ``(text, require_idle, attachments)`` the offer saw.
        started: every ``(text, attachments)`` that became a turn.
        emitted: every event, so a silent drop is visible as an absence.
    """

    send_message = JaatoServer.send_message
    _emit_attachment_dropped = JaatoServer._emit_attachment_dropped

    def __init__(self, outcome="needs_turn", model_running=False,
                 honour_require_idle=True):
        self.outcome = outcome
        self._model_running = model_running
        self._honour_require_idle = honour_require_idle
        self.offers = []
        self.started = []
        self.emitted = []
        self._original_inputs = []
        self._main_agent_id = "main"
        self._pending_continuations = []
        self._pending_continuation_lock = threading.Lock()

        server = self

        class _RPC:
            def session_offer_message_threadsafe(
                self, text, *, source_id=None, source_type=None,
                require_idle=False, attachments=None, timeout=None,
            ):
                # Mirrors the real client: attachments RAISE require_idle
                # and are never put on the wire.  ``honour_require_idle``
                # False stands in for a runner predating the flag.
                if attachments:
                    require_idle = True
                server.offers.append((text, require_idle, attachments))
                if (server.outcome == "queued" and require_idle
                        and server._honour_require_idle):
                    return "busy"
                return server.outcome

        self._runner_rpc = _RPC()

    def emit(self, event):
        self.emitted.append(event)

    def _start_model_thread(self, prompt, attachments=None):
        self.started.append((prompt, list(attachments or [])))

    # --- assertions read these -------------------------------------------

    def carried_attachments(self):
        """Every attachment the daemon kept, whichever branch it took."""
        held = [a for _t, atts in self._pending_continuations for a in atts]
        driven = [a for _t, atts in self.started for a in atts]
        return held + driven

    def events_of(self, kind):
        return [e for e in self.emitted if isinstance(e, kind)]


def _audio_requiring_provider():
    """A provider that mimics ``openai/gpt-audio-*``.

    It records the message list of every request and REFUSES one that
    carries no audio, which is the upstream behaviour that made #877 loud.
    Asserting on the request rather than on ``session.get_history()`` is
    deliberate: a fix that trimmed history while the wire still carried the
    bytes, or the reverse, must not read as a pass.
    """
    requests = []

    class Refused400(RuntimeError):
        pass

    def complete(messages, **kwargs):
        snapshot = [
            Message(role=m.role, parts=list(m.parts or []))
            for m in messages
        ]
        requests.append(snapshot)
        if not _audio_bytes(snapshot):
            raise Refused400(
                "400 This model requires that either input content or "
                "output modality contain audio."
            )
        on_chunk = kwargs.get("on_chunk")
        if on_chunk is not None:
            on_chunk("Understood.")
        return TurnResult.from_provider_response(ProviderResponse(
            parts=[Part(text="Understood.")],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, output_tokens=5,
                             total_tokens=15),
        ))

    provider = MagicMock()
    provider.name = "fake"
    provider.supports_streaming.return_value = True
    provider.get_context_limit.return_value = 0
    provider.get_retry_after.return_value = None
    provider.complete.side_effect = complete
    provider.requests = requests
    provider.Refused400 = Refused400
    return provider


def _live_session(provider):
    runtime = MagicMock()
    runtime.create_provider.return_value = provider
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.ledger = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    runtime.registry.enrich_prompt.side_effect = (
        lambda prompt, **_k: MagicMock(prompt=prompt, metadata={})
    )
    session = JaatoSession(runtime, "openai/gpt-audio-mini")
    session.configure()
    return session


def _audio_bytes(messages):
    return sum(
        len(p.inline_data["data"])
        for m in messages for p in (m.parts or [])
        if p.inline_data and isinstance(p.inline_data.get("data"), bytes)
    )


def _audio_fills(messages):
    """Which clips a request carried, identified by their filler byte."""
    return {
        p.inline_data["data"][44:46]
        for m in messages for p in (m.parts or [])
        if p.inline_data and isinstance(p.inline_data.get("data"), bytes)
    }


# ============================================================ 1: every arrival

def test_an_attachment_bearing_send_is_never_queued_as_text():
    """The #845 rule, at the USER send path.

    Three arrivals, one invariant: the bytes either drive a turn or are
    held for one.  They are never traded for a
    ``MidTurnPromptQueuedEvent``, which names only the text.
    """
    arrivals = [
        ("needs_turn", False, "idle"),
        ("queued", True, "session busy"),
        ("needs_turn", True, "daemon unwinding"),
    ]
    for outcome, model_running, label in arrivals:
        server = _StubServer(outcome=outcome, model_running=model_running)
        utterance = _utterance()

        server.send_message("transcribe this", attachments=[utterance])

        assert utterance in server.carried_attachments(), (
            f"the {label} arrival dropped the attachment. For a voice turn "
            f"the attachment IS the message (#838/#877), so a send that "
            f"loses it is a turn the model has nothing to answer."
        )
        assert not server.events_of(MidTurnPromptQueuedEvent), (
            f"the {label} arrival queued an attachment-bearing send. The "
            f"mid-turn queue folds a message in as TEXT and has nowhere to "
            f"put an inline_data part (#845)."
        )


def test_the_offer_is_told_the_send_carries_bytes():
    """``require_idle`` is derived from the payload, not from a preference.

    The daemon cannot know whether the runner-side session is mid-turn --
    that is the whole reason ``offer_message`` exists -- so the refusal has
    to be requested at the offer rather than decided after the answer.
    """
    server = _StubServer(outcome="queued", model_running=True)

    server.send_message("transcribe this", attachments=[_utterance()])

    assert server.offers, "the offer was never made"
    _text, require_idle, _atts = server.offers[0]
    assert require_idle is True, (
        "an attachment-bearing offer left require_idle False, so a busy "
        "session answers 'queued' and the bytes are enqueued nowhere (#877)"
    )


def test_the_offer_rpc_raises_require_idle_for_a_payload():
    """The rule lives on the VERB, so every caller of it is covered.

    ``_StubServer``'s fake RPC mirrors this, and a mirror is not evidence
    -- so this drives the real ``RunnerRPCClient.session_offer_message``
    and reads the body it would have put on the wire.  Two properties, and
    the second is why the parameter exists at all rather than being a
    per-caller ``require_idle=bool(attachments)``:

    * ``require_idle`` is raised by the presence of a payload;
    * the payload itself does NOT travel -- the session's ``_message_queue``
      stores strings, and pretending otherwise would be a second way to
      lose the bytes.
    """

    sent = {}

    client = RunnerRPCClient.__new__(RunnerRPCClient)

    async def _call_named(method, body, timeout=None):
        sent["method"] = method
        sent["body"] = body
        return {"outcome": "busy"}

    client._call_named = _call_named

    outcome = asyncio.run(client.session_offer_message(
        "transcribe this",
        source_id="user",
        source_type="user",
        attachments=[_utterance()],
    ))

    assert outcome == "busy"
    assert sent["body"].get("require_idle") is True, (
        "an attachment-bearing offer reached the session with require_idle "
        "unset; a running session then answers 'queued' and folds the "
        "message in as text, dropping the bytes (#845's rule, #877's bug)"
    )
    assert "attachments" not in sent["body"], (
        "the payload was put on the offer wire. The mid-turn queue stores "
        "strings; a byte-carrying message there would be lost at the fold "
        "instead of at the offer, which is worse, not better."
    )


def test_a_text_only_offer_leaves_require_idle_alone():
    """Only bytes raise the flag; an explicit caller preference is kept.

    The companion to the guard above: the parameter must not turn every
    offer into a drive, or a mid-turn text send stops being deliverable.
    """

    sent = {}
    client = RunnerRPCClient.__new__(RunnerRPCClient)

    async def _call_named(method, body, timeout=None):
        sent["body"] = body
        return {"outcome": "queued"}

    client._call_named = _call_named

    asyncio.run(client.session_offer_message("just a note", source_id="user"))

    assert "require_idle" not in sent["body"]


def test_a_text_only_send_still_queues():
    """The fix must not make every mid-turn send start a second turn.

    A text message HAS somewhere to go in the running turn, and queuing it
    there is the behaviour clients rely on.  Only bytes force a drive.
    """
    server = _StubServer(outcome="queued", model_running=True)

    server.send_message("just a note, no audio")

    _text, require_idle, atts = server.offers[0]
    assert require_idle is False and not atts
    assert server.events_of(MidTurnPromptQueuedEvent), (
        "a text-only mid-turn send stopped being queued; the idle-only "
        "rule is about bytes, not about mid-turn delivery"
    )
    assert not server.started, "a text-only send started a rival turn"


# ============================================================ 3: never silent

def test_a_drop_that_cannot_be_prevented_is_refused_by_name():
    """A runner that ignores ``require_idle`` must not produce silence.

    This is unreachable against a current runner, which is exactly why it
    is reported rather than trusted: the entire cost of #877 was that the
    drop's only witness was an event that said ``text=...`` and read as a
    healthy queue.  An old daemon answering "queued" to an
    attachment-bearing offer has already enqueued the text and cannot be
    recalled -- so the bytes are named as lost instead of being passed off
    as delivered.
    """
    server = _StubServer(outcome="queued", model_running=True,
                         honour_require_idle=False)

    server.send_message("transcribe this", attachments=[_utterance()])

    dropped = [e for e in server.events_of(ErrorEvent)
               if e.error_type == "AttachmentDropped"]
    assert dropped, (
        "an attachment was discarded with no event naming it. A dropped "
        "payload must be refused by name, the way an empty message is "
        "(EmptyMessageError, #838) -- never reported as a normal queue."
    )
    assert "audio/wav" in str(dropped[0].details), (
        "the refusal does not say WHAT was dropped"
    )
    assert "UklGRg" not in str(dropped[0].details), (
        "the refusal carries the payload itself; mime, size and name are "
        "what a caller needs, and an event is not a place for the bytes"
    )


# ============================================================ 5: the drain

def test_the_drain_merges_every_stashed_message_and_its_bytes():
    """N sends in one wind-down window become ONE turn carrying all of it.

    Texts join, attachments concatenate.  Taking only the newest is #623's
    silent loss; taking only the texts is #877's.
    """
    first, second = _utterance(b"\x01\x02"), _utterance(b"\x03\x04")

    text, attachments = merge_pending_continuations(
        [("first", [first]), ("second", [second])],
    )

    assert text == "first\n\nsecond"
    assert attachments == [first, second]


def test_the_drain_keeps_a_blank_text_utterance():
    """Bytes with no text is a turn (#838).

    A voice send's natural shape is ``complete("", attachments=[clip])``,
    so a drain gated on the text alone would drop precisely the normal
    case.
    """
    utterance = _utterance()

    text, attachments = merge_pending_continuations([("", [utterance])])

    assert text == ""
    assert attachments == [utterance], (
        "a blank-text utterance was dropped by the drain, which is the "
        "voice turn's normal shape (#838)"
    )


# ============================================================ 6: the payoff

def test_three_consecutive_audio_turns_all_reach_the_wire_with_audio():
    """The issue's acceptance criterion, end to end and offline.

    Composes the two halves the defect spanned.  The DAEMON half is real:
    ``JaatoServer.send_message`` decides each arrival, with the session
    answering busy on the even ones -- the wind-down window ``ask()``
    structurally lands in -- and ``merge_pending_continuations`` folds
    whatever it stashed.  The SESSION half is real too: each resulting turn
    is driven at a live ``JaatoSession`` whose provider refuses a request
    carrying no audio, exactly as ``openai/gpt-audio-*`` does.  Only the
    thread scheduling between them is simulated, because the model thread
    is what the arrival races.

    Three distinct clips, so "carried its OWN utterance" is checked rather
    than "carried some audio" -- the issue's eight-chunk run is where a
    turn answering about the PREVIOUS clip would otherwise read as a pass.

    Before the fix: ``audio bytes per turn: [48044, 0, 0]`` and a
    ``400 ... must contain audio`` on turn 2.
    """
    provider = _audio_requiring_provider()
    session = _live_session(provider)
    clips = [_utterance(b"\x01\x02"), _utterance(b"\x03\x04"),
             _utterance(b"\x05\x06")]

    for index, clip in enumerate(clips):
        # Odd arrivals land in the wind-down window: the SESSION is idle
        # (its turn ended) while this daemon's model thread is still
        # unwinding, which is where ``ask()`` structurally puts them.
        unwinding = bool(index % 2)
        server = _StubServer(outcome="needs_turn", model_running=unwinding)

        server.send_message("", attachments=[clip])

        if unwinding:
            assert not server.started, (
                "a send during wind-down started a rival turn"
            )
            text, attachments = merge_pending_continuations(
                server._pending_continuations,
            )
        else:
            assert len(server.started) == 1
            text, attachments = server.started[0]

        session.send_message(text, lambda *a, **k: None,
                             attachments=attachments)

    assert len(provider.requests) == 3, (
        f"expected three turns to reach the provider, got "
        f"{len(provider.requests)}"
    )
    per_turn = [_audio_bytes(r) for r in provider.requests]
    assert all(per_turn), (
        f"audio bytes per turn: {per_turn}. A zero is a turn that reached "
        f"an audio-requiring model with no audio -- the 400 in #877. Note "
        f"a two-turn version of this test passes with the defect intact: "
        f"the parity means turn 1 always works."
    )
    for index, request in enumerate(provider.requests):
        own = clips[index]["data"][44:46]
        assert own in _audio_fills(request), (
            f"turn {index + 1} reached the provider without its OWN "
            f"utterance. Answering about the previous clip is the quiet "
            f"half of #877 -- the half that reports success."
        )
