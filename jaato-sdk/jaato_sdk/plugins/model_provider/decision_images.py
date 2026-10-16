"""Images in a decision request: what is checked before anything is sent.

Jev 1.14 (and 1.13) accept still images in a decision request, as a
top-level ``assets`` map the questions and the state refer to by a
reference string (docs/design/decision-models.md §12).  The vendor rejects
an image out of its limits and never resizes one, so every limit that can
be measured from the bytes is checked here, before the request leaves the
host, and a refusal names the image and the limit:

=====================  ==============================================
format                 PNG or JPEG, read from the bytes, never from the
                       declared mime; an animated PNG is refused
dimensions             each side ``MIN_SIDE``..``MAX_SIDE`` px, total
                       ``MIN_PIXELS``..``MAX_PIXELS``
bytes                  ``MAX_IMAGE_BYTES`` per image (decoded)
=====================  ==============================================

Dimensions come from the header (PNG ``IHDR``, the first JPEG ``SOFn``
marker), with the standard library only: Pillow is an optional extra.  A
header that cannot be read is refused, not sent on trust.

Not checked, because the bytes do not say: EXIF orientation (the vendor
strips it without applying it, so pixels must already be upright), and
whether the content is what a question assumes.  Nothing here converts,
rotates or resizes an image.

Pure: no I/O, so the stand-in server and every adapter share it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

#: Each side of an image, in pixels (TypeSafe, 2026-10-16).
MIN_SIDE = 280
MAX_SIDE = 1568
#: Width times height, in pixels.
MIN_PIXELS = 78_400
MAX_PIXELS = 2_458_624
#: Decoded bytes of one image.
MAX_IMAGE_BYTES = 16 * 1024 * 1024

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"

#: The mime types an image may be declared as, and the format each names.
MIME_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg"}

#: JPEG start-of-frame markers (every ``SOFn`` but DHT, JPG and DAC).
_JPEG_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                       0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
#: JPEG markers that carry no length field.
_JPEG_STANDALONE = frozenset({0x01, *range(0xD0, 0xD8)})


class ImageHeaderError(ValueError):
    """The bytes are not a readable still PNG or JPEG."""


@dataclass(frozen=True)
class ImageInfo:
    """What the header says: ``format`` (``png`` / ``jpeg``) and size."""

    format: str
    width: int
    height: int


def _png_info(data: bytes) -> ImageInfo:
    """Read ``IHDR``, then walk the chunks to ``IDAT`` looking for ``acTL``.

    APNG puts ``acTL`` before the first ``IDAT``, so a still PNG is one
    with no ``acTL`` there.
    """
    pos = len(PNG_SIGNATURE)
    if data[pos + 4:pos + 8] != b"IHDR" or len(data) < pos + 16:
        raise ImageHeaderError("PNG with no IHDR chunk first")
    width, height = struct.unpack(">II", data[pos + 8:pos + 16])
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        if kind == b"acTL":
            raise ImageHeaderError("animated PNG (an acTL chunk); send a still image")
        if kind in (b"IDAT", b"IEND"):
            break
        pos += 12 + length
    else:
        raise ImageHeaderError("PNG ends before its image data")
    return ImageInfo("png", width, height)


def _jpeg_info(data: bytes) -> ImageInfo:
    """Walk the markers to the first ``SOFn`` and read its size."""
    pos = len(JPEG_SIGNATURE)
    while pos < len(data):
        if data[pos] != 0xFF:
            raise ImageHeaderError(f"JPEG marker expected at byte {pos}")
        while pos < len(data) and data[pos] == 0xFF:   # fill bytes
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker in _JPEG_STANDALONE:
            continue
        if marker in (0xD9, 0xDA):   # EOI or SOS before any frame header
            break
        if pos + 2 > len(data):
            break
        (length,) = struct.unpack(">H", data[pos:pos + 2])
        if marker in _JPEG_SOF:
            if pos + 7 > len(data):
                break
            height, width = struct.unpack(">HH", data[pos + 3:pos + 7])
            return ImageInfo("jpeg", width, height)
        if length < 2:
            raise ImageHeaderError(f"JPEG segment with length {length}")
        pos += length
    raise ImageHeaderError("JPEG with no frame header (SOFn) before its scan data")


def read_image_info(data: bytes) -> ImageInfo:
    """The format and size of a still PNG or JPEG, from its header.

    Raises:
        ImageHeaderError: not PNG or JPEG (GIF, WebP and HEIC included),
            an animated PNG, or a header that cannot be read.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise ImageHeaderError("image data must be bytes")
    data = bytes(data)
    if data.startswith(PNG_SIGNATURE):
        try:
            return _png_info(data)
        except struct.error as exc:
            raise ImageHeaderError(f"truncated PNG header ({exc})") from exc
    if data.startswith(JPEG_SIGNATURE):
        try:
            return _jpeg_info(data)
        except struct.error as exc:
            raise ImageHeaderError(f"truncated JPEG header ({exc})") from exc
    raise ImageHeaderError("not a PNG or JPEG (convert GIF, WebP or HEIC first)")


def image_limit_problem(info: ImageInfo, size: int) -> Optional[str]:
    """The first limit ``info`` and ``size`` (decoded bytes) break, or ``None``."""
    if size > MAX_IMAGE_BYTES:
        return f"{size} bytes, over the {MAX_IMAGE_BYTES}-byte limit per image"
    for side, value in (("width", info.width), ("height", info.height)):
        if not MIN_SIDE <= value <= MAX_SIDE:
            return (f"{side} {value} px is outside {MIN_SIDE}..{MAX_SIDE} px; "
                    "the model never resizes, so scale or crop it first")
    pixels = info.width * info.height
    if not MIN_PIXELS <= pixels <= MAX_PIXELS:
        return (f"{info.width}x{info.height} is {pixels} pixels, outside "
                f"{MIN_PIXELS}..{MAX_PIXELS}")
    return None


__all__ = [
    "MIN_SIDE",
    "MAX_SIDE",
    "MIN_PIXELS",
    "MAX_PIXELS",
    "MAX_IMAGE_BYTES",
    "MIME_FORMATS",
    "ImageHeaderError",
    "ImageInfo",
    "read_image_info",
    "image_limit_problem",
]
