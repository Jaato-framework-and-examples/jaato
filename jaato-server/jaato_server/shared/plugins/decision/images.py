"""Images for the ``decide`` tool (docs/design/decision-models.md §12.5).

The model never puts image bytes in a tool call: they would enter history
and be replayed on every later request (#850).  It names each image by
one of two references, and this module turns the reference into a
:class:`DecisionImage`:

=================  =====================================================
``path``           a workspace file.  Relative paths are joined onto the
                   workspace root.  The path must pass
                   ``check_path_with_jaato_containment`` (the ``readFile``
                   rule), it is read through ``path_safety`` (no
                   check-then-open race), and ``.env`` at any depth and
                   ``.jaato/*_auth.json`` are refused: the bytes go to a
                   third party.  The format is read from the bytes, never
                   from the extension.
``attachment_id``  an image already in the session's history: one the
                   user sent, or one a tool returned.  The id is the
                   ingest digest ``media_identity`` mints (``att_<hex>``).
                   An image GC evicted (only its marker left) is refused
                   by name, so the model is told the bytes are gone
                   rather than that the id is unknown.
=================  =====================================================

Everything here raises :class:`ImageReferenceError` naming the image; the
plugin returns it as an explicit failure.  The format and size limits are
the contract's (``build_decision_request``), checked after resolution.

:func:`image_record` is what the ledger keeps: digest, size and
dimensions, never the bytes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from jaato_sdk.media_identity import ATTACHMENT_ID_KEY
from jaato_sdk.plugins.model_provider.decision_images import (
    MAX_IMAGE_BYTES,
    ImageHeaderError,
    read_image_info,
)
from jaato_sdk.plugins.model_provider.decisions import MAX_IMAGES, DecisionImage

#: The format read from the bytes, as the mime the contract expects.
FORMAT_MIMES = {"png": "image/png", "jpeg": "image/jpeg"}


class ImageReferenceError(ValueError):
    """An ``images`` entry that cannot be resolved; the message names it."""


@dataclass(frozen=True)
class ResolvedImage:
    """One image ready to send, and where it came from."""

    image: DecisionImage
    source: str   # "path" or "attachment"
    ref: str      # the path as given, or the attachment id
    width: int
    height: int
    format: str


def is_credential_path(path: str) -> bool:
    """``.env`` at any depth, or ``*_auth.json`` under a ``.jaato`` directory.

    The same rule as the daemon's workspace download
    (``server/workspace_download.is_credential_path``), restated because a
    runner-tier plugin cannot import ``jaato_server.server``.
    """
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if not parts:
        return False
    name = parts[-1]
    if name == ".env":
        return True
    return name.endswith("_auth.json") and ".jaato" in parts[:-1]


def _payload_bytes(data: Any) -> Optional[bytes]:
    """The bytes of a history payload: raw bytes, or base64 text (the wire form)."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=False)
        except (binascii.Error, ValueError):
            return None
    return None


def _from_bytes(name: str, data: bytes, source: str, ref: str,
                declared_mime: Optional[str] = None) -> ResolvedImage:
    try:
        info = read_image_info(data)
    except ImageHeaderError as exc:
        raise ImageReferenceError(f"image {name!r} ({ref}): {exc}") from exc
    mime = declared_mime or FORMAT_MIMES[info.format]
    return ResolvedImage(
        image=DecisionImage(mime_type=mime, data=data, name=name),
        source=source, ref=ref, width=info.width, height=info.height,
        format=info.format)


def resolve_path(name: str, path: str, workspace_root: Optional[str],
                 registry: Any = None) -> ResolvedImage:
    """Read a workspace image file, refusing what ``readFile`` refuses."""
    from ..path_safety import UnsafePathError, read_bytes_verified
    from ..sandbox_utils import check_path_with_jaato_containment

    if is_credential_path(path):
        raise ImageReferenceError(
            f"image {name!r}: {path!r} holds credentials and is never sent")
    full = path if os.path.isabs(path) or not workspace_root else \
        os.path.join(workspace_root, path)

    def allowed(resolved: str) -> bool:
        if is_credential_path(resolved):
            return False
        return check_path_with_jaato_containment(
            resolved, workspace_root or "", registry, mode="read")

    if not allowed(full):
        raise ImageReferenceError(
            f"image {name!r}: {path!r} is outside what this session may read")
    try:
        size = os.stat(full).st_size
    except OSError as exc:
        raise ImageReferenceError(
            f"image {name!r}: cannot read {path!r} ({exc.strerror})") from exc
    if size > MAX_IMAGE_BYTES:
        raise ImageReferenceError(
            f"image {name!r}: {path!r} is {size} bytes, over the "
            f"{MAX_IMAGE_BYTES}-byte limit per image")
    try:
        data = read_bytes_verified(full, validate=allowed,
                                   max_bytes=MAX_IMAGE_BYTES + 1)
    except UnsafePathError as exc:
        raise ImageReferenceError(
            f"image {name!r}: {path!r} is outside what this session may read"
        ) from exc
    except OSError as exc:
        raise ImageReferenceError(
            f"image {name!r}: cannot read {path!r} ({exc.strerror})") from exc
    return _from_bytes(name, data, "path", path)


def _history_views(history: Iterable[Any]) -> Iterable[Dict[str, Any]]:
    from ..gc.utils import part_media_views
    for message in history:
        for part in getattr(message, "parts", None) or []:
            yield from part_media_views(part)


def _was_evicted(history: Iterable[Any], attachment_id: str) -> bool:
    """Whether an eviction marker names ``attachment_id``."""
    needle = f"ref {attachment_id}"
    for message in history:
        for part in getattr(message, "parts", None) or []:
            text = getattr(part, "text", None) or ""
            response = getattr(part, "function_response", None)
            suffix = getattr(response, "model_suffix", None) or ""
            if "[Media evicted" in text + suffix and needle in text + suffix:
                return True
    return False


def resolve_attachment(name: str, attachment_id: str,
                       history: List[Any]) -> ResolvedImage:
    """Find an image in the session's history by its ingest id.

    The newest match wins (the same bytes, so the same id, may appear
    more than once).  The declared mime is kept, so the contract refuses
    one the bytes contradict (#829).
    """
    for view in reversed(list(_history_views(history))):
        if view.get(ATTACHMENT_ID_KEY) != attachment_id:
            continue
        data = _payload_bytes(view.get("data"))
        if data is None:
            break
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageReferenceError(
                f"image {name!r}: attachment {attachment_id} is {len(data)} "
                f"bytes, over the {MAX_IMAGE_BYTES}-byte limit per image")
        return _from_bytes(name, data, "attachment", attachment_id,
                           view.get("mime_type") or None)
    if _was_evicted(history, attachment_id):
        raise ImageReferenceError(
            f"image {name!r}: attachment {attachment_id} was evicted from "
            "history after the turn that used it; its bytes are gone")
    raise ImageReferenceError(
        f"image {name!r}: no attachment {attachment_id} in this session's "
        "history")


def resolve_images(raw: Any, workspace_root: Optional[str], history: List[Any],
                   registry: Any = None) -> List[ResolvedImage]:
    """The tool's ``images`` argument, resolved.  ``None`` or ``[]`` is no images.

    Each entry is ``{name, path}`` or ``{name, attachment_id}``, exactly
    one of the two.  Names must be unique (the contract checks too, but a
    duplicate is cheaper to refuse before reading anything).
    """
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise ImageReferenceError("`images` must be an array")
    if len(raw) > MAX_IMAGES:
        raise ImageReferenceError(
            f"{len(raw)} images in one call; at most {MAX_IMAGES}")
    out: List[ResolvedImage] = []
    seen: set = set()
    for entry in raw:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise ImageReferenceError("every image needs a non-empty `name`")
        if name in seen:
            raise ImageReferenceError(f"two images are named {name!r}")
        seen.add(name)
        path, att = entry.get("path"), entry.get("attachment_id")
        if bool(path) == bool(att):
            raise ImageReferenceError(
                f"image {name!r}: give exactly one of `path` or `attachment_id`")
        if path:
            out.append(resolve_path(name, str(path), workspace_root, registry))
        else:
            out.append(resolve_attachment(name, str(att), history))
    return out


def image_record(resolved: ResolvedImage) -> Dict[str, Any]:
    """What the ledger keeps about one image: never the bytes."""
    data = resolved.image.data
    return {
        "name": resolved.image.name,
        "source": resolved.source,
        "ref": resolved.ref,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "format": resolved.format,
        "width": resolved.width,
        "height": resolved.height,
    }


__all__ = [
    "FORMAT_MIMES",
    "ImageReferenceError",
    "ResolvedImage",
    "is_credential_path",
    "resolve_path",
    "resolve_attachment",
    "resolve_images",
    "image_record",
]
