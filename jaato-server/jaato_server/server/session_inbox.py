"""The durable per-session inbox (session group messaging, phase 2).

A message *accepted for delivery* that cannot be handed to a running turn
right now is written here BEFORE the sender is given its receipt, so "it
will be processed" is a guarantee rather than a receipt.  One inbox per
session, many messages, under the session's own record directory::

    <workspace>/.jaato/sessions/<session_id>.inbox/<message_id>.json     # envelope
    <workspace>/.jaato/sessions/<session_id>.inbox/<message_id>/         # spooled bytes
    <workspace>/.jaato/sessions/<session_id>.inbox/files/<message_id>/   # delivered files

Under ``.jaato/sessions/`` because that is the directory a revive already
reads, it is workspace STATE (the scaffold's ``.gitignore`` block already
excludes ``sessions/``), and it sits inside the target's own confinement.
The record listing globs ``*.json`` at the top of that directory, so an
``<id>.inbox/`` directory beside ``<id>.json`` is invisible to it.

What lands here, and who drains it (design §4.4):

==========================================  ======================================
the target was                              drained by
==========================================  ======================================
busy, and the text was QUEUED runner-side   the turn-end hook, which REMOVES the
(``runner_queued``)                         copy -- the running turn consumed it.
                                            The copy is what survives an unload
                                            between queue and drain.
busy, and the message carried bytes         the turn-end hook, which DRIVES it:
                                            bytes ride the drive branch only
cold, and the revive failed                 the lifetime watchdog, with backoff
revived cold with no client and a cascade   the client's attach (a deferred
observer (``defer_until_client``)           wake, the former single slot)
anything, once loaded by anything           the first drain after the load
==========================================  ======================================

The third directory is phase 3 (design §4.5): a file a peer REFERENCED from
another workspace, or a text attachment too large to inline, is COPIED
there so the target can read it inside its own confinement, and the
message names it by that path.  It is distinct from ``<message_id>/``
because the two have different lifetimes -- spooled bytes are consumed by
the drive that re-inflates them and go with the envelope, while a
delivered file must OUTLIVE the envelope: the target reads it with its own
tools on the turn the message started, and on later ones.  It goes with
the session's record (:func:`remove_all`) and with nothing sooner.

Stdlib-only, like :mod:`.session_groups`: it is read from the daemon's
listing path and must import nothing that path cannot.  Every write is a
temp file plus :func:`os.replace`, so a crash mid-write leaves no
half-envelope a drain could act on.  An envelope this build cannot parse
is skipped and named at WARNING rather than deleted: a message nobody can
read is still a message somebody sent.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .transfer_limits import STAGE_PER_FILE_LIMIT, STAGE_TOTAL_LIMIT

logger = logging.getLogger(__name__)

#: Suffix of the per-session inbox directory, beside ``<session_id>.json``.
INBOX_SUFFIX = ".inbox"

#: How long a spooled message waits before it is dropped as expired.  The
#: wake-binding TTL is the precedent: a message for a session that nothing
#: has loaded in a day is a message about a day-old situation.
DEFAULT_INBOX_TTL_SECONDS = 24 * 3600.0

#: ``InboxEntry.kind`` values.
KIND_PEER = "peer"      # a group message; driven on the idle-only SIBLING tier
KIND_WAKE = "wake"      # a deferred ``session.wake``; driven as a USER turn

#: Name of the delivered-files directory inside an inbox.  Chosen so it can
#: never collide with a ``<message_id>/`` bytes directory: message ids are
#: hex (:func:`new_message_id`), and ``files`` is not.
FILES_DIRNAME = "files"

#: How much text-attachment content one message may carry INLINE, in the
#: wrapper, in bytes (design §4.5: 32 KiB).  Beyond it a text attachment is
#: stored as a file and delivered as a reference, so the body a target's
#: model reads on one turn stays bounded whatever a peer sends.
TEXT_ATTACHMENT_INLINE_CAP = 32 * 1024

#: The copy caps -- the STAGING caps, by design: a file copied on behalf of
#: a message is bounded exactly as one staged on behalf of a client.
FILE_COPY_PER_FILE_LIMIT = STAGE_PER_FILE_LIMIT
FILE_COPY_TOTAL_LIMIT = STAGE_TOTAL_LIMIT

#: Copy chunk: the digest is computed as the bytes stream, so a 10 MB file
#: is never held whole.
_COPY_CHUNK = 1 << 20

_ENVELOPE_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class InboxEntry:
    """One spooled message.  The envelope on disk is :meth:`to_dict`.

    ``attachments`` is the MANIFEST -- one row per spooled file carrying
    ``mime_type`` / ``display_name`` / ``attachment_id`` and the relative
    ``file`` the bytes were written to -- never the bytes themselves, so
    an envelope stays small enough to read on every listing.
    :func:`load_attachments` re-inflates the rows into the canonical wire
    shape (base64 ``data``) when the message is driven.
    """
    message_id: str
    session_id: str
    #: :data:`KIND_PEER` or :data:`KIND_WAKE` -- decides which drive verb
    #: delivers it (SIBLING tier vs a USER turn).
    kind: str
    #: The label inside the untrusted wrapper (``peer:<addr>`` /
    #: ``wake:<source>``), stored so a drain wraps the body exactly as the
    #: live path would have.
    source: str
    #: The sender address for the queue's ``source_id``.
    source_id: str
    text: str
    created_at: float
    expires_at: float
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    #: Phase 3: the FILE manifest the wrapper names -- one row per file
    #: reference (``name`` / ``path`` / ``sha256`` / ``size`` /
    #: ``mime_type`` / ``disposition``), where ``path`` is already in the
    #: TARGET's terms (relative to its workspace).  Nothing here is read
    #: back from disk at drive time; the rows are rendered into the wrapper
    #: exactly as the live path would have rendered them.
    files: List[Dict[str, Any]] = field(default_factory=list)
    #: Phase 3: the text attachments small enough to travel INLINE
    #: (``display_name`` / ``mime_type`` / ``text``), stored in the envelope
    #: because they are bounded by :data:`TEXT_ATTACHMENT_INLINE_CAP`.
    text_attachments: List[Dict[str, Any]] = field(default_factory=list)
    event_id: Optional[str] = None
    #: A deferred wake: eligible only once the session has an attached
    #: client, because the woken turn may need the client's host tools.
    defer_until_client: bool = False
    #: The text was ALSO queued runner-side into the running turn.  Removed
    #: at that turn's end (consumed); re-driven if the session is loaded
    #: from disk first (the turn never completed in this daemon's life).
    runner_queued: bool = False
    wake_ref: str = ""
    cascade_driver_id: Optional[str] = None
    attempts: int = 0
    last_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["version"] = _ENVELOPE_VERSION
        return d

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "InboxEntry":
        names = {f for f in cls.__dataclass_fields__}  # noqa: SIM118
        kw = {k: v for k, v in raw.items() if k in names}
        return cls(**kw)

    def expired(self, now: Optional[float] = None) -> bool:
        return self.expires_at <= (time.time() if now is None else now)


def inbox_dir(storage_dir: Path, session_id: str) -> Path:
    """The inbox directory for *session_id* under a session storage dir."""
    return Path(storage_dir) / f"{session_id}{INBOX_SUFFIX}"


def _safe_name(index: int, display_name: Any) -> str:
    base = os.path.basename(str(display_name or "")) or "attachment"
    base = _SAFE_NAME.sub("_", base).strip("._") or "attachment"
    return f"{index:02d}-{base[:120]}"


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def spool(
    storage_dir: Path,
    entry: InboxEntry,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> InboxEntry:
    """Write *entry* and its *attachments* (canonical wire shape) to disk.

    The bytes go first, the envelope last: an envelope that exists names
    files that exist.  Returns the entry with its manifest filled in.
    """
    root = inbox_dir(storage_dir, entry.session_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    items = list(attachments or [])
    if items:
        payload_dir = root / entry.message_id
        payload_dir.mkdir(exist_ok=True)
        for i, att in enumerate(items):
            if not isinstance(att, dict):
                continue
            name = _safe_name(i, att.get("display_name"))
            raw = att.get("data")
            data = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw or b"")
            _write_atomic(payload_dir / name, data)
            manifest.append({
                "mime_type": att.get("mime_type"),
                "display_name": att.get("display_name"),
                "attachment_id": att.get("attachment_id"),
                "file": f"{entry.message_id}/{name}",
            })
    entry.attachments = manifest
    _write_atomic(root / f"{entry.message_id}.json",
                  json.dumps(entry.to_dict(), sort_keys=True).encode("utf-8"))
    return entry


def update(storage_dir: Path, entry: InboxEntry) -> None:
    """Rewrite an envelope in place (attempt counters, flags)."""
    root = inbox_dir(storage_dir, entry.session_id)
    if root.is_dir():
        _write_atomic(root / f"{entry.message_id}.json",
                      json.dumps(entry.to_dict(), sort_keys=True).encode("utf-8"))


def remove(storage_dir: Path, entry: InboxEntry) -> None:
    """Delete an envelope and its spooled bytes; the directory itself goes
    when it is empty.  The files DELIVERED with the message
    (:func:`files_dir`) are deliberately not touched: the target may still
    be reading them, and they go with the session record."""
    root = inbox_dir(storage_dir, entry.session_id)
    try:
        (root / f"{entry.message_id}.json").unlink()
    except FileNotFoundError:
        pass
    shutil.rmtree(root / entry.message_id, ignore_errors=True)
    try:
        if root.is_dir() and not any(root.iterdir()):
            root.rmdir()
    except OSError:
        pass


def remove_all(storage_dir: Path, session_id: str) -> None:
    """Delete a session's whole inbox (the session record is being deleted)."""
    shutil.rmtree(inbox_dir(storage_dir, session_id), ignore_errors=True)


def files_dir(storage_dir: Path, session_id: str, message_id: str) -> Path:
    """Where the files delivered WITH *message_id* live (phase 3): a copied
    cross-workspace reference, or a text attachment over the inline cap."""
    return inbox_dir(storage_dir, session_id) / FILES_DIRNAME / message_id


def store_file(
    storage_dir: Path, session_id: str, message_id: str, index: int,
    display_name: Any, *, source: Optional[Path] = None,
    data: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Put ONE file into the target's delivered-files directory and return
    ``{"file": <path relative to the storage dir>, "sha256", "size"}``.

    Either *source* (a file to copy, streamed) or *data* (bytes to write).
    The name is the display name's basename with an index prefix, exactly
    as spooled bytes are named, so two references to files called
    ``report.md`` in different directories do not overwrite each other.
    A temp file plus :func:`os.replace`, and the digest is taken from the
    bytes as they are written -- a caller that re-verifies against the
    digest it computed on the source is checking the COPY, not re-reading
    the source it already trusted.  Raises ``OSError`` on any failure;
    nothing half-written is left under the final name.
    """
    root = files_dir(storage_dir, session_id, message_id)
    root.mkdir(parents=True, exist_ok=True)
    name = _safe_name(index, display_name)
    dest = root / name
    tmp = dest.with_name(dest.name + f".tmp-{uuid.uuid4().hex[:8]}")
    digest = hashlib.sha256()
    size = 0
    try:
        with open(tmp, "wb") as out:
            if source is not None:
                with open(source, "rb") as src:
                    while True:
                        chunk = src.read(_COPY_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            else:
                payload = data or b""
                out.write(payload)
                digest.update(payload)
                size = len(payload)
        os.replace(tmp, dest)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    rel = dest.relative_to(Path(storage_dir)).as_posix()
    return {"file": rel, "sha256": digest.hexdigest(), "size": size}


def remove_files(storage_dir: Path, session_id: str, message_id: str) -> None:
    """Delete the files delivered with *message_id* -- the undo for a
    message that was copied for and then NOT delivered."""
    shutil.rmtree(files_dir(storage_dir, session_id, message_id),
                  ignore_errors=True)
    parent = inbox_dir(storage_dir, session_id) / FILES_DIRNAME
    try:
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


def pending(storage_dir: Path, session_id: str) -> List[InboxEntry]:
    """Every readable envelope, oldest first (``created_at``, then id).

    An unreadable envelope is skipped and named at WARNING, never deleted.
    """
    root = inbox_dir(storage_dir, session_id)
    if not root.is_dir():
        return []
    out: List[InboxEntry] = []
    for path in root.glob("*.json"):
        try:
            with open(path, "rb") as fh:
                raw = json.loads(fh.read().decode("utf-8"))
            out.append(InboxEntry.from_dict(raw))
        except Exception as exc:  # noqa: BLE001 -- one bad envelope, not the drain
            logger.warning("session inbox: cannot read %s (%s); left in place",
                           path, exc)
    out.sort(key=lambda e: (e.created_at, e.message_id))
    return out


def count_pending(storage_dir: Path, session_id: str) -> int:
    """How many envelopes the inbox holds -- one ``listdir``, no parsing,
    because this is asked on every ``session.list``."""
    root = inbox_dir(storage_dir, session_id)
    if not root.is_dir():
        return 0
    try:
        return sum(1 for n in os.listdir(root) if n.endswith(".json"))
    except OSError:
        return 0


def find_event_id(storage_dir: Path, session_id: str,
                  event_id: Optional[str]) -> Optional[InboxEntry]:
    """The spooled entry carrying *event_id*, or ``None``.  The durable half
    of the wake dedup LRU: a redelivery after a daemon restart finds its
    inbox entry and is answered ``duplicate``."""
    if not event_id:
        return None
    for entry in pending(storage_dir, session_id):
        if entry.event_id == event_id:
            return entry
    return None


def load_attachments(storage_dir: Path, entry: InboxEntry) -> List[Dict[str, Any]]:
    """Re-inflate the manifest into the canonical wire shape
    (``{mime_type, data: base64, display_name, attachment_id}``).  A file
    that vanished is skipped rather than failing the drive."""
    root = inbox_dir(storage_dir, entry.session_id)
    out: List[Dict[str, Any]] = []
    for row in entry.attachments:
        rel = row.get("file")
        if not rel:
            continue
        try:
            with open(root / rel, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            logger.warning("session inbox: spooled file %s missing (%s)", rel, exc)
            continue
        item = {"mime_type": row.get("mime_type"),
                "data": base64.b64encode(data).decode("ascii")}
        if row.get("display_name") is not None:
            item["display_name"] = row["display_name"]
        if row.get("attachment_id") is not None:
            item["attachment_id"] = row["attachment_id"]
        out.append(item)
    return out


def new_message_id() -> str:
    return uuid.uuid4().hex
