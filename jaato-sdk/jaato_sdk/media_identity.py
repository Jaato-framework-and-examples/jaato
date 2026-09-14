"""Stable identity — and a human description — for inbound media attachments.

WHY AN INBOUND ATTACHMENT NEEDS AN ID (#850).  Model-emitted media has
carried ``stream_id`` and ``sequence`` since #824.  An *inbound*
attachment carried nothing: the canonical wire shape is
``{mime_type, data, display_name}`` and not one of those three identifies
anything — two recordings of the same caller are both ``question.wav``.

That absence is what made purging consumed audio out of history unsafe.
Dropping the bytes is cheap; dropping them and leaving behind a marker
that cannot be tied back to an archived recording is what an audit in a
retention-regulated domain objects to ("the agent handled a claim from
audio nobody can produce").  So the identifier has to be minted at
INGEST, before anything is in a position to purge.

WHY A CONTENT DIGEST RATHER THAN A UUID.  The cross-reference has to work
from the *archive* side: someone holds a ``.wav`` and a transcript naming
an id, and must decide whether they match.  A uuid answers that only via
a mapping somebody has to have kept.  A digest of the bytes can be
RECOMPUTED from the recording itself::

    $ python -c "import hashlib,sys; \\
        print('att_'+hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])" q.wav
    att_9f2c1ab73e0d4455

so the link survives the loss of every intermediate record.  Two
byte-identical attachments collide, and that is correct — they *are* the
same recording.

Minting is idempotent and side-effect free, which is what lets the SDK
client mint it (so the caller knows the id it sent, and can name the file
it archived) while the daemon back-fills the same id for attachments that
arrive from clients which do not — a WS client, a direct
``session.complete(...)`` call — and the two agree by construction
because both hash the same bytes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import wave
from typing import Any, Dict, Optional, Tuple

#: Key carrying the id inside a wire attachment dict and inside
#: ``Part.inline_data``.  Additive: every existing reader of those dicts
#: uses ``.get()`` on the three known keys and is untouched by a fourth.
ATTACHMENT_ID_KEY = "attachment_id"

#: Prefix on every minted id.  Present so an id is recognisable as one
#: when it turns up in a transcript, a log line or an archive filename.
ATTACHMENT_ID_PREFIX = "att_"

#: Hex characters of the SHA-256 digest kept.  64 bits — short enough to
#: read aloud over a support call, wide enough that a collision between
#: two *different* recordings is not a practical concern.
ATTACHMENT_ID_DIGEST_CHARS = 16

# Raw PCM carries no header, so its duration is derivable only from the
# mime parameters.  These are the defaults the framework's own audio uses
# (``_media_deltas.STREAM_AUDIO_MIME`` — 24 kHz mono s16le), applied when
# a parameter is absent, exactly as the inbound converter treats a bare
# ``audio/pcm`` as pcm16.
_PCM_MIMES = frozenset({"audio/pcm", "audio/x-pcm"})
_PCM_DEFAULT_RATE = 24000
_PCM_DEFAULT_CHANNELS = 1
_PCM_SAMPLE_BYTES = 2

_WAV_MIMES = frozenset({
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
})


def _as_bytes(data: Any) -> Optional[bytes]:
    """The raw bytes behind ``data``, whichever form the caller holds.

    Accepts ``bytes``/``bytearray`` (the in-process form) and a base64
    ``str`` (the wire form).  Returns ``None`` for anything else — a
    ``None`` payload, or a base64 string that does not decode — so every
    caller has one place to decide what to do about an unusable payload
    instead of each guessing.
    """
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=False)
        except (binascii.Error, ValueError):
            return None
    return None


def mint_attachment_id(data: Any) -> Optional[str]:
    """The stable id for an attachment's payload, or ``None``.

    ``None`` when the payload is unusable (absent, or a base64 string that
    does not decode).  A caller must treat that as "this attachment has no
    id", never fabricate one: an id that is not a digest of the bytes
    breaks the recompute-from-the-archive property the whole scheme rests
    on.

    Args:
        data: Raw ``bytes`` or a base64 ``str``.

    Returns:
        ``att_<16 hex>``, or ``None``.
    """
    raw = _as_bytes(data)
    if raw is None:
        return None
    digest = hashlib.sha256(raw).hexdigest()[:ATTACHMENT_ID_DIGEST_CHARS]
    return f"{ATTACHMENT_ID_PREFIX}{digest}"


def ensure_attachment_id(attachment: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``attachment`` carrying an :data:`ATTACHMENT_ID_KEY`.

    Idempotent in both directions that matter: an attachment that already
    names an id is returned **unchanged** (same object — a client may have
    minted an id under a scheme of its own and the framework does not
    overwrite it), and an attachment whose payload cannot be decoded is
    also returned unchanged rather than given a fabricated id.

    Otherwise a *copy* carrying the id is returned; the input dict is
    never mutated, because the caller's list may be the caller's own
    request object.
    """
    if attachment.get(ATTACHMENT_ID_KEY):
        return attachment
    minted = mint_attachment_id(attachment.get("data"))
    if minted is None:
        return attachment
    enriched = dict(attachment)
    enriched[ATTACHMENT_ID_KEY] = minted
    return enriched


def split_mime(mime: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """``"audio/pcm;rate=24000"`` -> ``("audio/pcm", {"rate": "24000"})``.

    Lower-cases both halves and tolerates a parameter with no ``=``.  A
    deliberate small duplicate of the server-side converter's own splitter
    (``model_provider/_attachments._split_mime``): this module lives in the
    SDK, which the converters do not import, and one seven-line parser on
    each side is cheaper than a dependency edge from the wire converters to
    the SDK.
    """
    head, _, tail = (mime or "").partition(";")
    params: Dict[str, str] = {}
    for chunk in tail.split(";"):
        key, sep, value = chunk.partition("=")
        if sep:
            params[key.strip().lower()] = value.strip().strip('"').lower()
    return head.strip().lower(), params


def audio_duration_seconds(mime: Optional[str], data: Any) -> Optional[float]:
    """Best-effort duration of an audio payload, in seconds.

    Derivable without a decoder for exactly two shapes, which are the two
    this framework actually produces and consumes:

    - a **WAV** container, from its header (stdlib :mod:`wave`); and
    - **raw PCM**, from ``len(data)`` and the mime parameters, defaulting
      to the 24 kHz mono s16le the framework's own streaming audio uses.

    Every other container (mp3, ogg, m4a, flac, ...) returns ``None`` —
    their durations need a decoder, and the framework does not take a
    media dependency to put a nicety in a marker string.  ``None`` means
    *unknown*, and callers render the payload size instead; it never means
    zero.
    """
    raw = _as_bytes(data)
    if not raw:
        return None
    base, params = split_mime(mime)
    if base in _WAV_MIMES:
        try:
            with wave.open(io.BytesIO(raw), "rb") as handle:
                rate = handle.getframerate()
                return handle.getnframes() / rate if rate else None
        except (wave.Error, EOFError, ValueError):
            return None
    if base in _PCM_MIMES:
        try:
            rate = int(params.get("rate", _PCM_DEFAULT_RATE))
            channels = int(params.get("channels", _PCM_DEFAULT_CHANNELS))
        except ValueError:
            return None
        divisor = rate * channels * _PCM_SAMPLE_BYTES
        return len(raw) / divisor if divisor > 0 else None
    return None


def format_size(num_bytes: int) -> str:
    """``602144`` -> ``"588.0 KB"``.  Binary units, one decimal."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} GB"


def describe_attachment(
    mime: Optional[str],
    num_bytes: int,
    duration_seconds: Optional[float] = None,
) -> str:
    """Human-readable one-liner for a payload: mime, duration, size.

    ``"audio/wav, 12.6s, 588.0 KB"`` — or ``"audio/mpeg, 231.4 KB"`` when
    the duration is not derivable.  This is what a reader sees in place of
    bytes that were purged, so it names what was there without pretending
    to more precision than the payload supports.
    """
    fields = [mime or "unknown type"]
    if duration_seconds is not None:
        fields.append(f"{duration_seconds:.1f}s")
    fields.append(format_size(num_bytes))
    return ", ".join(fields)
