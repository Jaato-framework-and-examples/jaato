"""JSON codec for the daemon ↔ runner RPC wire (#920).

Both ends of the channel described in
``docs/design/per_session_confined_runner.md`` §4.1 speak
length-prefixed JSON frames (:mod:`shared.framing`).  JSON has no
``bytes``, and the two ends used to disagree about what that means:

- the runner wrote ``json.dumps(payload, default=str)``, so a
  ``bytes`` payload was serialised as its **Python repr** — every
  non-printable byte became ``\\xNN`` (4 characters), which JSON then
  escaped again.  Measured on a 120 s 16 kHz mono WAV: 3.84 MB of
  audio became a **16.13 MB** frame (4.20x), well over the 10 MB
  ``MAX_MESSAGE_SIZE``, so the peer refused the frame and closed the
  transport, killing every in-flight call and the session with it;
- the daemon wrote a bare ``json.dumps``, which raises ``TypeError``
  on ``bytes`` instead.

Size was the lesser half.  ``str(b'\\x00\\xff')`` does not round-trip:
the receiver got the *text* ``"b'\\\\x00\\\\xff'"``, not the payload.
The frame cap refused the big ones first, so the corruption stayed
invisible — a smaller attachment (an image, a short clip, a PDF)
serialised the same way, passed the cap, and delivered a repr string
that nothing would ever decode.

So ``bytes`` is serialised **explicitly**, as base64 under a marker
key the decoder understands, and ``str`` stays what it always was: the
fallback for genuinely diagnostic objects (a datetime, an enum, an
unexpected value) where a readable rendering beats a crash.  The same
120 s utterance now crosses at 5.12 MB (1.33x) and decodes back to the
bytes that were sent.

The marker is a one-key dict:

    {"__bytes_b64__": "<standard base64>"}

A payload dict that genuinely holds that single key and a base64
string is indistinguishable from an encoded blob and decodes to bytes.
No wire shape in ``server.runner.envelope`` uses the name, and the
double-underscore fencing is what makes an accidental collision a
non-issue rather than a hazard.

Note on frame sizing: :func:`dumps` keeps ``ensure_ascii=True`` (the
``json`` default), so every character of the result is one UTF-8 byte
and ``len(encoded)`` IS the wire size.  :func:`frame_size` exists so
callers don't re-encode a multi-megabyte string just to measure it.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Dict


#: Marker key wrapping a base64-encoded ``bytes`` payload on the wire.
BYTES_MARKER_KEY = "__bytes_b64__"


def _default(obj: Any) -> Any:
    """``json.dumps`` fallback: base64 for bytes, ``str`` for the rest.

    Args:
        obj: A value ``json`` cannot serialise on its own.

    Returns:
        ``{BYTES_MARKER_KEY: "<base64>"}`` for a binary payload (which
        :func:`loads` turns back into ``bytes``), else ``str(obj)`` —
        the historical behaviour, retained because a diagnostic field
        carrying an enum or a ``Path`` should render, not raise.
    """
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {
            BYTES_MARKER_KEY: base64.b64encode(bytes(obj)).decode("ascii"),
        }
    return str(obj)


def _object_hook(obj: Dict[str, Any]) -> Any:
    """Decode the :data:`BYTES_MARKER_KEY` wrapper back to ``bytes``.

    Applied to every decoded object, so it must be cheap and must
    leave anything it doesn't recognise exactly as it found it.  A
    marker whose value isn't valid base64 is passed through unchanged
    rather than raising: a malformed frame should surface where the
    field is read, not take down the read loop that would have logged
    it.
    """
    if len(obj) == 1:
        blob = obj.get(BYTES_MARKER_KEY)
        if isinstance(blob, str):
            try:
                return base64.b64decode(blob.encode("ascii"), validate=True)
            except (binascii.Error, ValueError, UnicodeEncodeError):
                return obj
    return obj


def dumps(payload: Any) -> str:
    """Serialise *payload* to one JSON frame body.

    ``bytes`` anywhere in the structure becomes a base64 marker dict
    (see the module docstring); everything else behaves like plain
    ``json.dumps``.
    """
    return json.dumps(payload, default=_default)


def loads(raw: str) -> Any:
    """Decode a JSON frame body written by :func:`dumps`.

    Base64 marker dicts are restored to ``bytes``.  A frame written by
    a peer that predates this codec decodes exactly as ``json.loads``
    would, so a rolling upgrade in either direction is a no-op.
    """
    return json.loads(raw, object_hook=_object_hook)


def frame_size(encoded: str) -> int:
    """Wire size in bytes of an :func:`dumps` result.

    ``ensure_ascii=True`` guarantees a pure-ASCII result, so this is
    ``len(encoded)`` — stated as a function because the equality is a
    property of :func:`dumps`, not of ``str``, and because measuring a
    multi-megabyte frame must not cost a second encode.
    """
    return len(encoded)
