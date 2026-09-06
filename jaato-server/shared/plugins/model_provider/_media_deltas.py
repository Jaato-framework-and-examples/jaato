"""Model-generated media on the wire — shared machinery.

The outbound counterpart to ``_prose_tools``: a plain module, not a base
class, so a provider opts in by *importing* rather than by *inheriting*.

That distinction is the whole point.  Ten providers in this tree speak
OpenAI's chat-completions wire format, but only five inherit
``_openai_compat.OpenAICompatProvider`` -- ``openrouter``, ``lmstudio``,
``vllm``, ``tensorrt_llm`` and ``triton`` each own their streaming loop.
Machinery parked on that base class is therefore unavailable to half the
fleet that speaks its format, which is exactly how model audio ended up
unreachable through OpenRouter.  ``_prose_tools`` already solved this for
the inbound direction; this module mirrors it.

The split of responsibilities:

- **The contract** -- ``output_modalities()`` /
  ``supports_output_modality()`` / ``emit_media_delta()`` /
  ``request_output_modalities()`` -- lives on
  ``base.ModalityCapabilityMixin``, which every provider already has.  A
  provider that cannot emit media implements nothing and inherits an
  honest text-only floor.
- **This module** is one vendor's wire *format*.  ``delta.audio.data`` as
  base64 pcm16 is OpenAI's shape; Google delivers model media as
  ``inlineData`` on parts and Anthropic emits none, so this decoder must
  NOT sit in the universal contract or every provider would inherit a
  decoder for a shape it never sees.
"""

from __future__ import annotations

import logging
from base64 import b64decode as _b64decode
from binascii import Error as BinasciiError
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from jaato_sdk.plugins.model_provider.types import MediaDelta, Part

logger = logging.getLogger(__name__)


#: OpenAI streams audio as headerless pcm16 -- 24 kHz mono signed 16-bit
#: little-endian -- and ONLY pcm16: requesting wav/mp3 together with
#: ``stream=true`` is rejected upstream.  The parameters are spelled into
#: the mime type because a headerless payload carries no way to recover
#: them, and a consumer that guesses wrong plays noise.
STREAM_AUDIO_MIME = "audio/pcm;rate=24000;channels=1;encoding=s16le"

#: Chat-completions body fields that request audio OUTPUT.  ``modalities``
#: here is OpenAI's OUTPUT selector and is NOT the jaato tier key of the
#: same name, which declares INPUT roles -- see the naming-collision note
#: in ``docs/design/binary-media-chunks.md``.
MEDIA_API_PARAMS: FrozenSet[str] = frozenset({"modalities", "audio"})

#: Used when a tier asks for audio but the profile named no voice/format.
#: ``pcm16`` is not a preference: it is the only format OpenAI emits while
#: streaming, so defaulting to anything else would fail the first request.
DEFAULT_AUDIO_OPTIONS: Dict[str, Any] = {"voice": "alloy", "format": "pcm16"}

#: The modality token a tier declares to ask for spoken output.
MODALITY_AUDIO = "audio"
MODALITY_TEXT = "text"


#: Start value for a turn's media-sequence counter -- "nothing decoded yet".
#: Named rather than a bare ``-1`` because two streaming loops initialise it
#: and both ask :func:`media_arrived` about it.
NO_MEDIA_YET = -1


def media_arrived(media_sequence: int) -> bool:
    """Whether any model media was decoded during this turn.

    Used as a TERMINATION signal, which needs justifying.

    ``require_terminated_stream`` (#687) rejects a stream that stopped
    without saying why, because for a TEXT stream a missing
    ``finish_reason`` means the upstream died mid-generation and the
    fragment is not an answer.  Audio streams break that assumption:
    verified live against ``openai/gpt-audio-mini`` via OpenRouter, the
    whole stream carries **no** ``finish_reason`` and **no**
    ``native_finish_reason`` -- only audio deltas, an empty-string
    ``content``, and a closing ``usage`` block.  Applying the text
    contract there rejects every complete audio response as a fragment,
    and marks it retryable, so the turn fails and retries forever.

    Reaching the check at all already means the chunk iteration ended
    normally -- a dropped connection or stall raises earlier -- so
    "media was decoded" plus "the stream closed cleanly" is positive
    evidence the generation finished.

    Deliberately NOT keyed on having *requested* audio: what matters is
    what actually came back.  And deliberately narrow -- a text stream's
    guard is untouched, because that is the case #687 exists for.
    """
    return media_sequence > NO_MEDIA_YET


def media_chunk_count(media_sequence: int) -> int:
    """How many media chunks were emitted, from the sequence counter.

    The counter starts at :data:`NO_MEDIA_YET` and is post-incremented, so
    it is already the count minus one -- no separate tally, and no branch
    in the streaming loop to keep the two in step.

    Audio IS content: counting only ``delta.content`` reported an
    audio-only turn as "no content arrived" while 31KB of speech had in
    fact been decoded, which made the stream diagnostics lie.
    """
    return media_sequence - NO_MEDIA_YET


def stream_terminated(
    terminal_seen: bool,
    media_sequence: int,
    usage_reported: bool = False,
) -> bool:
    """Whether the stream ended for a reason, not by dying mid-generation.

    Three signals, each evidence that the upstream REACHED THE END:

    - the wire named a finish reason;
    - media was decoded (a severed stream cannot retroactively have
      delivered bytes that already played);
    - a usage frame arrived.

    The last exists because some upstreams name no finish reason at all.
    Measured against openai/gpt-audio-mini through OpenRouter, across 19
    consecutive streams -- speaking turns and tool-call-only turns alike
    -- every one reported ``terminal_seen=False``, and OpenRouter's own
    generation record confirms it at the source::

        "finish_reason": null,  "native_finish_reason": null,
        "cancelled": false,     "status": 200,
        "tokens_completion": 27

    A completed, billed generation that names no reason.  Usage is what
    distinguishes it: the request sets ``stream_options.include_usage``,
    and the usage frame is the LAST frame of a finished stream, so a
    connection cut mid-generation never delivers one.  All 19 carried
    usage (1891-2002 tokens).

    Deliberately NOT "any part at all", and specifically NOT "a complete
    tool call arrived".  Accumulated calls are exactly what a stream
    severed mid-``arguments`` also leaves behind -- the same bytes on the
    wire -- and reading them as completion is how a severed turn becomes
    an executed one.  That is what #687 protects, and every clause here
    keeps it: absent all three signals the guard still fires.
    """
    return terminal_seen or media_arrived(media_sequence) or usage_reported


def ensure_spoken_part(parts: List[Any], transcript: str) -> None:
    """Give a spoken-but-wordless turn a text Part, in place.

    Appends only when the model produced NO text of its own: a turn that
    both wrote and spoke already has its words, and appending the
    transcript too would duplicate them.

    A turn that spoke AND called a tool does get the part — the
    function-call parts say what it DID, not what it said.

    No-op when the provider sent no transcript; the caller's
    media-arrival count is the backstop for that case.
    """
    if not transcript.strip():
        return
    if any(getattr(p, "text", None) for p in parts):
        return
    parts.append(Part.from_text(transcript))


def _audio_object(delta: Any) -> Any:
    """Find the ``audio`` object on a streaming delta, or None.

    ``audio`` is absent from OpenAI's published streaming-delta schema
    and so from the generated SDK types, arriving as an attribute,
    inside ``model_extra``, or as a plain dict depending on the client.
    Probed in one place because two questions are asked of it -- what
    does it carry, and has the audio ENDED -- and a second copy of this
    walk would be a second thing to keep in step.
    """
    audio = getattr(delta, "audio", None)
    if audio is None and isinstance(delta, dict):
        audio = delta.get("audio")
    if audio is None:
        extra = getattr(delta, "model_extra", None)
        if isinstance(extra, dict):
            audio = extra.get("audio")
    return audio


def is_end_of_audio(delta: Any) -> bool:
    """Whether this delta is the upstream's END-OF-AUDIO marker.

    OpenAI closes an audio stream with a ``delta.audio`` carrying
    neither bytes nor transcript -- in practice ``{"expires_at": ...}``,
    a value it can only know once the audio object is complete.
    Observed on the wire for openai/gpt-audio-mini through OpenRouter,
    frame 16 of 20::

        15  audio keys=['data']        6400B   <- last audio bytes
        16  audio keys=['expires_at']     0B   <- this
        17  audio keys=-                        (empty delta)
        18  audio keys=-                        (carries usage)
        19  [DONE]

    The marker is what lets ``final`` be a FACT rather than a guess.
    Without it a client cannot know an utterance ended and must infer --
    and every inference tried in this tree was wrong in some case: "the
    turn ended" closed a player mid-utterance and started a second one
    over it.

    Matched by ABSENCE rather than by the ``expires_at`` key, because
    the key is one vendor's spelling of "the audio object is finished"
    and the absence is the thing that actually means it.
    """
    audio = _audio_object(delta)
    if audio is None:
        return False
    if isinstance(audio, dict):
        data, transcript = audio.get("data"), audio.get("transcript")
    else:
        data = getattr(audio, "data", None)
        transcript = getattr(audio, "transcript", None)
    return not data and not transcript


def extract_audio_delta(
    delta: Any,
) -> Optional[Tuple[Optional[bytes], str]]:
    """Pull ``(raw_bytes_or_None, transcript)`` out of a streaming ``delta.audio``.

    Returns ``None`` only when the delta carries no audio object at all,
    or one with neither bytes nor transcript -- the cheap common path for
    every text-only provider and every text chunk.

    **Bytes and transcript arrive in SEPARATE deltas.**  Measured against
    openai/gpt-audio-mini via OpenRouter: of 19 audio deltas in one turn,
    7 carried data only, 11 carried transcript only, and ZERO carried
    both.  An earlier version returned ``None`` whenever ``data`` was
    absent, which silently discarded every transcript-bearing delta --
    so a spoken turn produced no text, no Part, and no history entry,
    and the framework nudged the model to finish work it had done.

    Defensive by necessity about SHAPE, not about content: ``audio`` is
    absent from OpenAI's published streaming-delta schema and so from
    the generated SDK types, arriving as an attribute, inside
    ``model_extra``, or as a plain dict depending on the client.  A
    malformed payload yields ``None`` rather than raising -- one bad
    chunk must not abort a turn that is otherwise streaming fine.
    """
    audio = _audio_object(delta)
    if audio is None:
        return None

    if isinstance(audio, dict):
        encoded = audio.get("data")
        transcript = audio.get("transcript")
    else:
        encoded = getattr(audio, "data", None)
        transcript = getattr(audio, "transcript", None)

    # Hold the wire contract: `data` is base64 TEXT and `transcript` is a
    # STRING.  Probing by attribute name means anything answering to
    # `.audio.transcript` gets this far -- a stand-in object in a test, a
    # future SDK field of another shape -- and a non-string joined into
    # the transcript raises far from here, inside the turn's final
    # assembly.  A value of the wrong type is not the field it is named
    # after, so it is dropped where it is read.
    if not isinstance(encoded, (str, bytes)):
        encoded = None
    if not isinstance(transcript, str):
        transcript = ""

    raw: Optional[bytes] = None
    if encoded:
        try:
            raw = _b64decode(encoded) or None
        except (BinasciiError, ValueError, TypeError):
            logger.warning("Discarding an undecodable audio delta")
            raw = None

    if raw is None and not transcript:
        return None
    return raw, transcript


def emit_audio_delta(
    delta: Any,
    on_chunk: Any,
    sequence: int,
    mime_type: str = STREAM_AUDIO_MIME,
    transcript_sink: Optional[List[str]] = None,
    pending: Optional[List[Any]] = None,
) -> int:
    """Emit one model-generated audio chunk; return the new sequence.

    Two facts ride on ``delta.audio`` and they arrive SEPARATELY, so they
    are handled separately here:

    * a transcript is appended to ``transcript_sink`` whenever present,
      and never produces a chunk of its own -- emitting an empty
      ``MediaDelta`` would hand clients zero bytes to play;
    * bytes advance ``sequence`` and are emitted to ``on_chunk``.

    A transcript-only delta therefore returns ``sequence`` UNCHANGED: it
    is not an audio chunk, and counting it would put gaps in the
    sequence a client uses to detect dropped media.

    Returns ``sequence`` unchanged when the delta carries no audio at
    all, so a streaming loop can call this unconditionally.

    ``pending`` opts into marking the last chunk ``final``.  It is a
    one-slot buffer the caller owns: a chunk is held until the NEXT
    thing arrives, so the upstream's end-of-audio marker
    (:func:`is_end_of_audio`) can flag the chunk before it as the last
    one.  Measured cost on openai/gpt-audio-mini, last-bytes frame to
    marker: **144 ms**, once per utterance, on a chunk already queued
    behind seconds of buffered playback.

    Without ``pending`` the behaviour is exactly as before -- chunks go
    out immediately and ``final`` is never set -- so a caller that has
    not opted in is unaffected.
    """
    if pending is not None and is_end_of_audio(delta):
        _release(on_chunk, pending, final=True)
        return sequence
    found = extract_audio_delta(delta)
    if found is None:
        return sequence
    raw, transcript = found
    if transcript and transcript_sink is not None:
        transcript_sink.append(transcript)
    if raw is None:
        return sequence
    sequence += 1
    chunk = MediaDelta(
        mime_type=mime_type,
        data=raw,
        sequence=sequence,
        transcript=transcript,
    )
    if pending is None:
        on_chunk(chunk)
        return sequence
    _release(on_chunk, pending, final=False)   # the one before this was not last
    pending.append(chunk)
    return sequence


def _release(on_chunk: Any, pending: List[Any], final: bool) -> None:
    """Emit the held chunk, if any, marking it ``final`` or not."""
    if not pending:
        return
    chunk = pending.pop()
    chunk.final = final
    on_chunk(chunk)


def flush_audio_stream(on_chunk: Any, pending: Optional[List[Any]]) -> None:
    """Emit whatever is still held, marking it the last of its stream.

    Called when a streaming loop ends.  The upstream marker normally
    releases the final chunk, so this is the path for a provider that
    sends none -- the stream ending is then the only evidence the
    utterance is over, and it is conclusive.  Without it a held chunk
    would simply never be delivered, which is worse than an unmarked
    one.
    """
    _release(on_chunk, pending or [], final=True)


def apply_output_modalities(
    kwargs: Dict[str, Any], requested: Iterable[str],
) -> None:
    """Stamp OpenAI's ``modalities``/``audio`` onto a request body.

    Called after the profile's own ``api_params`` have been applied, and
    uses ``setdefault`` throughout, so an explicit profile value always
    wins over what a tier role implies -- the tier says *what* to emit,
    the profile says *how* (which voice, which format).

    A request for nothing (the common case, and what switching out of a
    speaking tier produces) leaves ``kwargs`` untouched rather than
    writing ``modalities: ["text"]``, so a provider that has never been
    asked for audio sends a byte-identical body to before this existed.
    """
    kinds = {k.strip().lower() for k in (requested or ()) if k}
    if MODALITY_AUDIO not in kinds:
        return
    kwargs.setdefault("modalities", ["text", MODALITY_AUDIO])
    kwargs.setdefault("audio", dict(DEFAULT_AUDIO_OPTIONS))


def drop_unrequested_audio_options(kwargs: Dict[str, Any]) -> None:
    """Remove ``audio`` when nothing in this request is asking for media.

    ``api_params`` is PROVIDER-scoped; an outbound role is TIER-scoped.
    In a mixed profile -- a text planner and an audio speaker on one
    provider, which share a single provider instance -- the profile's
    ``audio: {voice, format}`` was stamped on every request, including
    the text tier's after the session had switched away.  ``modalities``
    was correctly dropped there and ``audio`` was not, leaving a body
    that says "no audio, and here is how to render it".

    Measured tolerated on one route (OpenRouter -> Azure, gpt-4o-mini:
    HTTP 200, ignored), which is a reason not to panic and not a reason
    to keep sending it -- nothing promises the next upstream agrees.

    A profile that sets ``modalities`` ITSELF still keeps its ``audio``,
    because that profile is asking directly with no tier involved: the
    tier-less speaking profile must keep working unchanged.
    """
    requested = kwargs.get("modalities") or ()
    if any(str(kind).strip().lower() != MODALITY_TEXT for kind in requested):
        return
    kwargs.pop("audio", None)


class OpenAIMediaOutputMixin:
    """Opt a provider into OpenAI-shaped model media output.

    Adding it to a provider's bases is a single-token change that never
    disturbs the provider's own constructor -- the mixin carries no
    ``__init__`` and its one piece of state is created lazily.  It is
    inheritance used as *opt-in*, not as a place to park machinery: the
    functions above stay importable on their own for a provider that
    would rather call them directly.

    Wiring a provider takes three touches:

    1. add this mixin to its bases (before ``ModalityCapabilityMixin``, so
       these overrides win);
    2. call ``self.emit_media_delta(delta, on_chunk, seq)`` in its
       streaming loop, keeping the returned sequence;
    3. call ``self.apply_requested_output_modalities(kwargs)`` where it
       assembles the request body, after its own ``api_params``.
    """

    #: Overridable by a provider streaming a different audio format.
    STREAM_AUDIO_MIME: str = STREAM_AUDIO_MIME

    def emit_media_delta(
        self, delta: Any, on_chunk: Any, sequence: int,
        transcript_sink: Optional[List[str]] = None,
        pending: Optional[List[Any]] = None,
    ) -> int:
        """Decode and emit OpenAI-shaped model audio; see :func:`emit_audio_delta`."""
        return emit_audio_delta(
            delta, on_chunk, sequence, self.STREAM_AUDIO_MIME, transcript_sink,
            pending,
        )

    def flush_media_stream(self, on_chunk: Any, pending: Optional[List[Any]]) -> None:
        """Release a held final chunk; see :func:`flush_audio_stream`."""
        flush_audio_stream(on_chunk, pending)

    def request_output_modalities(self, kinds: Iterable[str]) -> None:
        """Record which modalities subsequent turns should ask the model to EMIT.

        Called on every tier entry with that tier's declared outbound
        roles, so an empty set is meaningful: it is how switching out of a
        speaking tier stops requesting audio.
        """
        self._requested_output_modalities = frozenset(
            k.strip().lower() for k in (kinds or ()) if k
        )

    def apply_requested_output_modalities(self, kwargs: Dict[str, Any]) -> None:
        """Stamp the recorded request onto ``kwargs``; see :func:`apply_output_modalities`."""
        apply_output_modalities(
            kwargs, getattr(self, "_requested_output_modalities", frozenset())
        )
