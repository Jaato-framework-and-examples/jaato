"""Which workspace file a download request may read (protocol 1.20).

``workspace.file.fetch`` is the reverse of ``StageFilesRequest``: a remote
client names a path, the daemon answers with a header and, on success, the
bytes.  This module is the one place that decides whether a named path may
leave the workspace -- no I/O beyond ``stat``, so the rules are testable
without a socket and the WS handler only moves bytes.

Three rules, each attached to a way a download could go wrong:

- **Containment is judged on the RESOLVED path.**  The workspace root and
  the target are both resolved (symlinks followed) before comparing, so a
  link planted inside the workspace -- which the agent's own file tools can
  do -- cannot hand out a file beneath it that lives elsewhere.  The root
  itself is not a file under the root, and is refused as outside it.
- **Credentials never leave by this route.**  The workspace ``.env`` is
  where ``config.update`` writes the provider key, and ``.jaato/*_auth.json``
  is where ``<provider>-auth key`` stores one.  A download link the MODEL
  can offer (``offer_download``) must not be able to carry either out,
  whatever it was asked to do -- so they are refused by name, before size.
- **A refusal names its category**, never a bare failure: the client renders
  ``not_found`` and ``credential`` differently, and a model told why can
  stop offering what it cannot deliver.
"""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

#: Largest file ``workspace.file.fetch`` will send.  The bytes are read
#: whole and sent as ONE WebSocket frame, so the bound is also a bound on
#: daemon memory per request.  50 MB matches the staging total cap
#: (``DEFAULT_STAGE_TOTAL_LIMIT``), so a file can come back out at the size
#: it was allowed in, and it sits under the default message limit of common
#: WebSocket proxies (Node's ``ws`` defaults to 100 MiB).
DEFAULT_FILE_FETCH_LIMIT = 50 * 1024 * 1024


@dataclass(frozen=True)
class DownloadTarget:
    """The answer to "may this path be downloaded, and what is it".

    ``ok`` false means ``category`` / ``error`` say why and nothing else is
    meaningful.  ``relpath`` is POSIX-style and relative to the workspace
    root, so a client keys on it the way the Files panel does.
    """

    ok: bool
    category: str = ""
    error: str = ""
    abspath: str = ""
    relpath: str = ""
    name: str = ""
    size: int = 0
    mime_type: str = ""


def _refuse(category: str, error: str) -> DownloadTarget:
    return DownloadTarget(ok=False, category=category, error=error)


def is_credential_path(relpath: str) -> bool:
    """True for a workspace file that holds credentials.

    ``relpath`` is POSIX and relative to the workspace root.  The set is
    deliberately named rather than pattern-matched: ``.env`` exactly (a
    ``.env.example`` is documentation, not a secret) at any depth, and a
    ``*_auth.json`` under a ``.jaato`` directory -- the file each
    ``<provider>-auth`` plugin stores its key in.
    """
    parts = relpath.split("/")
    name = parts[-1]
    if name == ".env":
        return True
    return name.endswith("_auth.json") and ".jaato" in parts[:-1]


def resolve_download(
    workspace_root: str,
    path: str,
    *,
    max_bytes: int = DEFAULT_FILE_FETCH_LIMIT,
) -> DownloadTarget:
    """Decide whether ``path`` may be downloaded from ``workspace_root``.

    ``path`` is relative to the root, or absolute when it lies inside it.
    Checks run in the order a caller can act on: containment, existence,
    kind, credentials, size.  Containment is checked BEFORE existence, so a
    refusal is not an oracle for what exists outside the workspace.
    """
    raw = (path or "").strip()
    if not raw:
        return _refuse("unsafe_path", "no path given")
    if "\x00" in raw:
        return _refuse("unsafe_path", "path contains a NUL byte")
    try:
        root = Path(workspace_root).resolve(strict=True)
    except OSError as exc:
        return _refuse("workspace_not_found", f"workspace root unavailable: {exc}")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    target = candidate.resolve(strict=False)
    if target == root or root not in target.parents:
        return _refuse("unsafe_path", f"{raw!r} is outside the workspace")
    relpath = target.relative_to(root).as_posix()
    if not target.exists():
        return _refuse("not_found", f"no file at {relpath!r}")
    if not target.is_file():
        return _refuse("not_a_file", f"{relpath!r} is not a regular file")
    if is_credential_path(relpath):
        return _refuse(
            "credential",
            f"{relpath!r} holds credentials and cannot be downloaded",
        )
    try:
        size = target.stat().st_size
    except OSError as exc:
        return _refuse("io_error", str(exc))
    if size > max_bytes:
        return _refuse(
            "too_large",
            f"{relpath!r} is {size} bytes; the download limit is {max_bytes}",
        )
    mime, _ = mimetypes.guess_type(target.name)
    return DownloadTarget(
        ok=True,
        abspath=os.fspath(target),
        relpath=relpath,
        name=target.name,
        size=size,
        mime_type=mime or "application/octet-stream",
    )


def read_download(target: DownloadTarget) -> bytes:
    """Read a resolved target's bytes; raises ``OSError`` if the read fails.

    The caller sizes the header from what was READ, not from the earlier
    ``stat``, so a file that changed in between is still announced with the
    length of the frame that follows it.  Kept separate from
    :func:`resolve_download` so the metadata-only path never opens the
    file, and so the WS handler can run the read off the event loop.
    """
    with open(target.abspath, "rb") as fh:
        return fh.read()
