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


def completed_tool_call(parts: Iterable[Any]) -> bool:
    """Whether a COMPLETE function call was decoded from the stream.

    Complete means its arguments parsed.  A stream cut mid-generation
    leaves ``unreadable_args`` set -- the JSON stops short -- so a call
    whose arguments parsed is one the upstream finished sending, which
    is the distinction the termination check needs.

    Measured against openai/gpt-audio-mini via OpenRouter: a turn that
    speaks and then calls a tool carries NO finish reason at all
    (confirmed in OpenRouter's own billing export, where
    ``finish_reason_raw`` is empty for every such generation).  Without
    counting the call, that turn is discarded as a fragment, the tool
    never runs, the assistant turn never enters history, and the
    framework nudges the model to call the tool it already called.
    """
    for part in parts:
        call = getattr(part, "function_call", None)
        if call is not None and not getattr(call, "unreadable_args", None):
            return True
    return False


def stream_terminated(
    terminal_seen: bool,
    media_sequence: int,
    parts: Optional[Iterable[Any]] = None,
) -> bool:
    """Whether the stream ended for a reason, not by dying mid-generation.

    Three signals, any of which means the upstream finished rather than
    died: the wire named a finish reason; media was decoded; or a
    complete tool call was decoded.  The last two exist because an
    audio-modality request omits the finish reason entirely, so the
    first signal -- the only one a TEXT stream ever needs -- is absent
    for every such turn.

    Deliberately NOT "any part at all": a truncated TEXT stream also
    yields parts, and accepting those would retire the protection #687
    exists for.  Each clause is evidence of COMPLETION, not merely of
    output.
    """
    return (terminal_seen
            or media_arrived(media_sequence)
            or completed_tool_call(parts or ()))


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
    audio = getattr(delta, "audio", None)
    if audio is None and isinstance(delta, dict):
        audio = delta.get("audio")
    if audio is None:
        extra = getattr(delta, "model_extra", None)
        if isinstance(extra, dict):
            audio = extra.get("audio")
    if audio is None:
        return None

    if isinstance(audio, dict):
        encoded = audio.get("data")
        transcript = audio.get("transcript") or ""
    else:
        encoded = getattr(audio, "data", None)
        transcript = getattr(audio, "transcript", None) or ""

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
    """
    found = extract_audio_delta(delta)
    if found is None:
        return sequence
    raw, transcript = found
    if transcript and transcript_sink is not None:
        transcript_sink.append(transcript)
    if raw is None:
        return sequence
    sequence += 1
    on_chunk(MediaDelta(
        mime_type=mime_type,
        data=raw,
        sequence=sequence,
        transcript=transcript,
    ))
    return sequence


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
    ) -> int:
        """Decode and emit OpenAI-shaped model audio; see :func:`emit_audio_delta`."""
        return emit_audio_delta(
            delta, on_chunk, sequence, self.STREAM_AUDIO_MIME, transcript_sink
        )

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
