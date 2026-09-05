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
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

from jaato_sdk.plugins.model_provider.types import MediaDelta

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


def extract_audio_delta(delta: Any) -> Optional[Tuple[bytes, str]]:
    """Pull ``(raw_bytes, transcript)`` out of a streaming ``delta.audio``.

    Returns ``None`` when the delta carries no audio -- the case for every
    text-only provider and every text chunk, so this is the cheap common
    path.

    Defensive by necessity.  ``audio`` is absent from OpenAI's published
    OpenAPI schema for the streaming delta, and so from the generated SDK
    types too, meaning it may surface as an attribute, inside
    ``model_extra``, or as a plain dict depending on the client.  It is
    therefore probed rather than accessed, and a malformed or undecodable
    payload yields ``None`` instead of raising: one bad audio chunk must
    not abort a turn that is otherwise streaming fine.
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

    if not encoded:
        return None
    try:
        raw = _b64decode(encoded)
    except (BinasciiError, ValueError, TypeError):
        logger.warning("Discarding an undecodable audio delta")
        return None
    if not raw:
        return None
    return raw, transcript


def emit_audio_delta(
    delta: Any,
    on_chunk: Any,
    sequence: int,
    mime_type: str = STREAM_AUDIO_MIME,
) -> int:
    """Emit one model-generated audio chunk; return the new sequence.

    Returns ``sequence`` unchanged when the delta carries no audio, so a
    streaming loop can call this unconditionally for the cost of one
    function call and a ``None`` check.

    Args:
        delta: The streaming delta object from the provider's SDK.
        on_chunk: The :data:`StreamingCallback`.  Receives a
            :class:`MediaDelta`, never a ``str``.
        sequence: Last media sequence issued; incremented on emit.
        mime_type: Wire format of the payload.  Defaults to OpenAI's
            streaming pcm16; a provider streaming something else passes
            its own.
    """
    found = extract_audio_delta(delta)
    if found is None:
        return sequence
    raw, transcript = found
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

    def emit_media_delta(self, delta: Any, on_chunk: Any, sequence: int) -> int:
        """Decode and emit OpenAI-shaped model audio; see :func:`emit_audio_delta`."""
        return emit_audio_delta(
            delta, on_chunk, sequence, self.STREAM_AUDIO_MIME
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
