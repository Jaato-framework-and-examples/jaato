"""A voice session must not re-send every utterance it has ever heard.

THE GAP (#850).  Media had a lifecycle in exactly one direction.  Outbound
was right: model media is ``CLIENT``-audience so it never enters history at
all, and ``ensure_spoken_part`` leaves the TRANSCRIPT where the audio would
have gone -- words kept, bytes discarded.  Inbound got neither treatment.
An utterance stayed in history verbatim and was replayed on every later
request, forever.

Measured on a five-question helpdesk call, ``heard N bytes`` per turn::

    603244 + 888044 + 417644 + 680044 + 243244  =  ~2.8 MB of audio

all of it on the wire of the last request (~3.8 MB base64) and growing with
every question.  The run ended with ``runner RPC closed before id=83
responded`` at 9.6/11 GiB of host memory -- causation unproven, cost and
latency not in dispute.

AND GC COULD NOT SEE ANY OF IT.  ``grep -rn inline_data shared/plugins/gc_*/``
returned nothing: every strategy reasoned in tokens and turns, and
``estimate_message_tokens`` walked text, function calls and function
responses while never looking at ``inline_data``.  A 600 KB utterance was
sized at the one-token floor, so a threshold set at 60% could not fire on
the payload it was there to bound, and eviction ordering could not prefer a
turn it believed was empty.

WHY PURGING NEEDED AN ID FIRST.  A marker that says "audio was here" and
cannot be tied to an archived recording turns a purge into a deletion, and
in a domain with call-retention duties that is the audit finding ("the
agent handled a claim from audio nobody can produce").  A user-message
attachment was normalised to ``{mime_type, data, display_name}`` -- no id,
while OUTBOUND media had carried ``stream_id``/``sequence`` since #824.  So
the identifier had to be minted at ingest before any purge could be safe,
and it is a digest of the payload precisely so the archive side can
recompute it from the recording rather than trust a mapping.

THE CONTRACT, in the order these guards assert it:

1. an attachment carries a stable id from ingest, in history;
2. the id is derivable from the bytes, so an archived recording can be
   matched to a transcript that names one;
3. audio from a completed turn is gone from later requests;
4. what stands in its place NAMES the id -- the difference between purging
   and deleting;
5. a failed turn keeps its audio, because eviction is keyed on the turn
   having completed, not on the bytes having been sent;
6. GC sizes binary payload instead of scoring it zero, and can trigger on
   media pressure while nowhere near its token threshold;
7. the payoff: request size across N voice turns does not grow with N.

WHY THE GUARDS DRIVE ``send_message(attachments=...)``.  The accumulation
is reached by ordinary voice traffic; asserting from ``evict_consumed_media``
alone would leave the session's call sites -- the actual trigger -- untested.
"""

import io
import wave
from unittest.mock import MagicMock

from jaato_sdk.media_identity import (
    ATTACHMENT_ID_KEY,
    mint_attachment_id,
)
from jaato_sdk.plugins.model_provider.types import (
    FinishReason,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    TurnResult,
)

from shared.plugins.gc import (
    GCConfig,
    GCTriggerReason,
    estimate_message_tokens,
    history_media_bytes,
    load_gc_plugin,
    media_pressure_reason,
)
from shared.plugins.gc.utils import evict_consumed_media
from shared.jaato_session import JaatoSession
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""            # Consumed media from EARLIER turns is purged before this one is
            # appended (#850).  This is the path a voice turn takes, so it is
            # the one where the accumulation was measured.
            self._evict_consumed_media()

            # Append user message to session history
            self._history.append(Message(role=Role.USER, parts=list(parts)))""",
        replace="""            # Append user message to session history
            self._history.append(Message(role=Role.USER, parts=list(parts)))""",
        test="test_request_size_does_not_grow_with_the_number_of_utterances",
        because="every utterance a session has heard riding every later "
                "request, so a five-question call puts megabytes of audio "
                "on its last wire",
    ),
    Reversion(
        target="jaato-server/shared/plugins/gc/utils.py",
        find="""            # Binary parts (audio, images, PDFs) — the payload that
            # dominates a voice request and used to be sized at zero.
            elif part.inline_data:
                media_tokens += estimate_media_tokens(part.inline_data)""",
        replace="""            elif part.inline_data:
                media_tokens += 0""",
        test="test_gc_sizes_binary_payload_instead_of_scoring_it_zero",
        because="GC sizing a 600 KB utterance at the one-token floor, so no "
                "threshold can fire on the payload it exists to bound",
    ),
    Reversion(
        target="jaato-server/shared/plugins/gc/utils.py",
        find='''    fields = [inline_data.get("display_name") or None,
              describe_attachment(mime, num_bytes, duration),
              f"ref {attachment_id}" if attachment_id else "no ref (payload "
              "did not decode at ingest)"]''',
        replace='''    fields = [inline_data.get("display_name") or None,
              describe_attachment(mime, num_bytes, duration)]''',
        test="test_what_replaces_the_bytes_names_the_attachment_id",
        because="a purge that leaves no reference being indistinguishable "
                "from a deletion -- audio nobody can produce on audit",
    ),
]


# ============================================================ fixtures

def _wav(seconds: float = 1.0, fill: bytes = b"\x01\x02") -> bytes:
    """A real WAV file, so the marker's duration is derived and not faked."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(fill * int(24000 * seconds))
    return buf.getvalue()


def _utterance(seconds: float = 1.0, fill: bytes = b"\x01\x02") -> dict:
    """One wire attachment, id minted the way the SDK client mints it."""
    data = _wav(seconds, fill)
    return {
        "mime_type": "audio/wav",
        "data": data,
        "display_name": "question.wav",
        ATTACHMENT_ID_KEY: mint_attachment_id(data),
    }


def _recording_provider(fail_turns=()):
    """A provider that records the MESSAGE LIST of every request.

    The wire is the subject: an assertion on ``session.get_history()``
    would miss a fix that trimmed history while the request still carried
    the bytes, and vice versa.

    Args:
        fail_turns: 0-based turn indices that raise instead of answering,
            standing in for a turn that dies before the model sees its audio.
    """
    requests = []

    def complete(messages, **kwargs):
        requests.append([
            Message(role=m.role, parts=list(m.parts or []))
            for m in messages
        ])
        if (len(requests) - 1) in fail_turns:
            raise RuntimeError("upstream died mid-request")
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


def _request_bytes(messages) -> int:
    """Payload of one request: text characters plus binary bytes.

    Both halves matter and only together: counting bytes alone would call
    a fix that replaced 600 KB of audio with 600 KB of prose a success.
    """
    total = 0
    for msg in messages:
        for part in msg.parts or []:
            if part.text:
                total += len(part.text)
            if part.inline_data:
                data = part.inline_data.get("data")
                total += len(data) if isinstance(data, (bytes, bytearray)) else 0
    return total


def _audio_bytes(messages) -> int:
    return sum(
        len(p.inline_data["data"])
        for m in messages for p in (m.parts or [])
        if p.inline_data and isinstance(p.inline_data.get("data"), bytes)
    )


# ============================================================ 1-2: identity

def test_an_attachment_carries_a_stable_id_from_ingest():
    """Ingest, not eviction, is where the id has to exist.

    Nothing downstream can safely purge bytes it cannot name, so the id is
    minted before the first request rather than at the moment of purging.
    """
    provider = _recording_provider()
    session = _live_session(provider)

    session.send_message("What did I say?", lambda *a, **k: None,
                         attachments=[_utterance()])

    inline = [p.inline_data for m in session.get_history()
              for p in (m.parts or []) if p.inline_data]
    assert inline, "the attachment never reached history"
    assert inline[0].get(ATTACHMENT_ID_KEY), (
        "an inbound attachment reached history with no id. Outbound media "
        "has carried stream_id/sequence since #824; without the inbound "
        "counterpart no purge can leave anything traceable behind (#850)."
    )


def test_the_id_is_recomputable_from_the_recording():
    """The property that makes the cross-reference survive.

    An id assigned at random can be resolved only through a mapping
    somebody kept.  A digest of the payload can be recomputed from the
    archived file itself, which is what an auditor holding a .wav and a
    transcript actually needs.
    """
    payload = _wav(0.5)
    attachment = {"mime_type": "audio/wav", "data": payload}
    minted = mint_attachment_id(attachment["data"])

    assert minted == mint_attachment_id(payload), (
        "the id is not a function of the bytes, so an archived recording "
        "cannot be matched to a transcript naming it"
    )
    assert minted != mint_attachment_id(_wav(0.5, fill=b"\x09\x09")), (
        "two different recordings share an id"
    )


# ============================================================ 3-4: the purge

def test_audio_from_a_completed_turn_is_gone_from_later_requests():
    provider = _recording_provider()
    session = _live_session(provider)

    session.send_message("First question", lambda *a, **k: None,
                         attachments=[_utterance()])
    session.send_message("Second question", lambda *a, **k: None,
                         attachments=[_utterance(fill=b"\x03\x04")])

    assert len(provider.requests) == 2
    carried = _audio_bytes(provider.requests[1])
    assert carried == len(_wav(1.0, fill=b"\x03\x04")), (
        f"the second request carried {carried} bytes of audio; it should "
        f"carry only THIS turn's utterance. The first question's bytes have "
        f"no further use once it has been answered -- the conversation "
        f"carries the meaning (#850)."
    )


def test_what_replaces_the_bytes_names_the_attachment_id():
    """Purging from model context is not discarding the recording.

    The marker is what makes that sentence true rather than a claim: an
    archived call can be located from a transcript that names the ref.
    """
    provider = _recording_provider()
    session = _live_session(provider)
    attachment = _utterance(seconds=2.0)

    session.send_message("First question", lambda *a, **k: None,
                         attachments=[attachment])
    session.send_message("Second question", lambda *a, **k: None)

    replayed = " ".join(
        p.text for m in provider.requests[1] for p in (m.parts or []) if p.text
    )
    assert attachment[ATTACHMENT_ID_KEY] in replayed, (
        "the evicted audio left nothing naming it. A marker that cannot be "
        "cross-referenced with an archived recording makes a purge "
        "indistinguishable from a deletion, which is exactly the audit "
        "objection #850 is written against."
    )
    assert "audio/wav" in replayed and "2.0s" in replayed, (
        "the marker names neither the mime nor the duration, so a reader "
        "cannot tell what was removed"
    )


def test_a_turn_that_failed_keeps_its_audio():
    """Eviction is keyed on the turn COMPLETING, not on the bytes being sent.

    Running it in the previous turn's ``finally`` would look equivalent and
    would purge on exactly the path where the model never consumed the
    audio, leaving the caller unable to simply send again.
    """
    provider = _recording_provider(fail_turns=(0,))
    session = _live_session(provider)

    try:
        session.send_message("First question", lambda *a, **k: None,
                             attachments=[_utterance()])
    except RuntimeError:
        pass

    assert history_media_bytes(session.get_history()) > 0, (
        "a turn that died before the model answered lost its audio; there "
        "is nothing to retry with"
    )


def test_the_id_survives_persistence_and_is_re_minted_for_old_records():
    """A revived session's history is still history.

    The on-disk part shape kept ``mime_type`` and ``data`` and dropped
    everything else, so a woken session would have marked its purged audio
    "no ref" -- traceable right up until the daemon restarted.  Re-minting
    on the way back in is free precisely because the id is a digest: a
    record written before the key existed regains the value ingest would
    have given it.
    """
    from shared.plugins.session.serializer import (
        deserialize_part, serialize_part,
    )

    payload = _wav(0.25)
    part = Part(inline_data={
        "mime_type": "audio/wav", "data": payload,
        "display_name": "question.wav",
        ATTACHMENT_ID_KEY: mint_attachment_id(payload),
    })

    revived = deserialize_part(serialize_part(part))
    assert revived.inline_data[ATTACHMENT_ID_KEY] == part.inline_data[ATTACHMENT_ID_KEY]
    assert revived.inline_data["display_name"] == "question.wav"

    import base64
    legacy = {"type": "inline_data", "mime_type": "audio/wav",
              "data": base64.b64encode(payload).decode("ascii")}
    assert deserialize_part(legacy).inline_data[ATTACHMENT_ID_KEY] == (
        mint_attachment_id(payload)
    ), "a record written before the id existed did not regain one"


# ============================================================ 5-6: GC sees it

def test_gc_sizes_binary_payload_instead_of_scoring_it_zero():
    audio = _wav(12.0)
    heard = Message(role=Role.USER, parts=[
        Part.from_text("What did I say?"),
        Part(inline_data={"mime_type": "audio/wav", "data": audio}),
    ])
    spoken = Message(role=Role.USER,
                     parts=[Part.from_text("What did I say?")])

    assert estimate_message_tokens(heard) > estimate_message_tokens(spoken) * 5, (
        f"a message carrying {len(audio)} bytes of audio is sized at "
        f"{estimate_message_tokens(heard)} tokens against "
        f"{estimate_message_tokens(spoken)} for the same words alone. GC "
        f"reasons in tokens; a payload worth zero of them is a payload no "
        f"threshold can fire on (#850)."
    )


def test_a_media_heavy_session_can_trigger_collection_below_its_threshold():
    """The second denominator, and why it is in bytes.

    Percent-used is a token quantity, and the thing that killed the voice
    session was request SIZE.  Every strategy is asked, because a fix
    landed in one of the four leaves the other three exactly as blind as
    they were.
    """
    usage = {"percent_used": 1.0, "turns": 3,
             "media_bytes": 12 * 1024 * 1024}
    config = GCConfig()

    assert media_pressure_reason(usage, config) is GCTriggerReason.MEDIA_PRESSURE
    for name in ("gc_truncate", "gc_summarize", "gc_hybrid", "gc_budget"):
        plugin = load_gc_plugin(name, {})
        should, reason = plugin.should_collect(usage, config)
        assert should and reason is GCTriggerReason.MEDIA_PRESSURE, (
            f"{name} sat at 1% of its token budget carrying 12 MB of media "
            f"and declined to collect"
        )


def test_media_pressure_is_off_when_the_operator_turns_it_off():
    usage = {"percent_used": 1.0, "media_bytes": 99 * 1024 * 1024}
    assert media_pressure_reason(usage, GCConfig(media_bytes_threshold=0)) is None
    assert media_pressure_reason({"percent_used": 1.0}, GCConfig()) is None, (
        "a context_usage with no media_bytes key (an older client, a test "
        "harness) must not be read as pressure"
    )


def test_the_session_reports_media_bytes_so_a_strategy_can_read_it():
    provider = _recording_provider()
    session = _live_session(provider)
    session.send_message("What did I say?", lambda *a, **k: None,
                         attachments=[_utterance()])

    assert session.get_context_usage().get("media_bytes") == len(_wav(1.0)), (
        "get_context_usage reports no media_bytes, so media_pressure_reason "
        "has nothing to read and every strategy stays blind"
    )


# ============================================================ 7: the payoff

def test_request_size_does_not_grow_with_the_number_of_utterances():
    """The measurement from the issue, as a bound.

    Six one-second questions.  Unfixed, request six carries all six
    recordings; fixed, it carries one.  The assertion is deliberately on
    the RATIO rather than an absolute size -- the conversation's text does
    and should grow -- so it stays true as the marker's wording changes.
    """
    provider = _recording_provider()
    session = _live_session(provider)

    for index in range(6):
        session.send_message(
            f"Question {index}", lambda *a, **k: None,
            attachments=[_utterance(fill=bytes([index + 1, index + 1]))],
        )

    sizes = [_request_bytes(r) for r in provider.requests]
    one_utterance = len(_wav(1.0))
    assert sizes[-1] < sizes[0] * 2, (
        f"request payload grew {sizes[0]} -> {sizes[-1]} bytes over six "
        f"voice turns ({[s for s in sizes]}). Every utterance the session "
        f"has heard is still on the wire; five questions measured ~2.8 MB "
        f"in #850 and it grows without bound."
    )
    assert _audio_bytes(provider.requests[-1]) == one_utterance, (
        "the last request carried audio from more than the turn it belongs to"
    )


def test_eviction_leaves_a_text_only_history_untouched():
    """The cheap path: nothing to evict must allocate nothing.

    Object identity is the signal, the same one
    ``_gate_history_for_active_modalities`` uses, and losing it would make
    every text turn in every session pay for a feature it never uses.
    """
    history = [
        Message(role=Role.USER, parts=[Part.from_text("hello")]),
        Message(role=Role.MODEL, parts=[Part.from_text("hi")]),
    ]
    new_history, reclaimed, touched = evict_consumed_media(history)

    assert new_history is history and reclaimed == 0 and touched == []


def test_images_survive_eviction_by_default():
    """Audio and images are not the same case.

    A recording re-heard says what it said the first time; an image is
    routinely re-examined across turns ("what does the third column
    say?"), so the default set is audio alone.
    """
    history = [Message(role=Role.USER, parts=[
        Part(inline_data={"mime_type": "image/png", "data": b"\x89PNG" * 500}),
        Part(inline_data={"mime_type": "audio/wav", "data": _wav(1.0)}),
    ])]
    new_history, reclaimed, _touched = evict_consumed_media(history)

    kept = [p for p in new_history[0].parts if p.inline_data]
    assert len(kept) == 1 and kept[0].inline_data["mime_type"] == "image/png"
    assert reclaimed == len(_wav(1.0))
