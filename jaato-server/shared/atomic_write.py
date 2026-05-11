"""Atomic file-write helper (Phase 3 §3.14).

The contract: any framework-side persistent state file landing on
``<workspace>/.jaato/`` or ``~/.jaato/`` MUST be written atomically
so a SIGTERM-mid-write (daemon crash, OOM kill, operator kill -9)
leaves the file either pre-write or post-write — never partial,
never corrupted.

The pattern is the standard POSIX recipe: write to a sibling temp
file in the same directory, fsync, then ``os.replace`` to swap into
place.  ``os.replace`` is atomic at the filesystem level for intra-
filesystem renames (POSIX rename(2) guarantee), so a crash either
leaves the old file (rename hadn't happened) or the new file
(rename completed).  No half-written state ever appears at the
target path.

Why a helper:

- Phase 3 §3.14 elevates this from "audit pass" to enforced contract.
- Sites that hand-rolled the recipe (artifact_tracker, waypoint,
  memory, file_session-restore-path) all use slight variations:
  some use ``tempfile.mkstemp`` + ``os.fdopen``, some use
  ``tempfile.NamedTemporaryFile``, some skip the fsync.  A single
  helper kills the variations.
- The lint test (`tests/test_atomic_write_partition.py`) treats
  any non-test ``open(..., 'w')`` in ``server/`` or
  ``shared/plugins/`` on persistent state paths as a violation,
  with an explicit allow-list for tool-side file mutations
  (lsp file edits, file_edit plugin, etc. — those write user
  workspace files, not framework state).

Usage:

    from shared.atomic_write import atomic_write_text, atomic_write_json

    atomic_write_text(path, "contents")
    atomic_write_json(path, {"key": "value"}, indent=2)

Both helpers:
1. Create the parent directory if it doesn't exist.
2. Write to a sibling temp file in the same directory.
3. Flush + fsync the temp file.
4. ``os.replace`` to swap into place.
5. Clean up the temp file on any failure (including
   ``KeyboardInterrupt`` / ``SystemExit``).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union


PathLike = Union[str, Path]


def atomic_write_text(
    path: PathLike,
    content: str,
    *,
    encoding: str = "utf-8",
    fsync: bool = True,
) -> None:
    """Atomically write *content* to *path*.

    A SIGTERM (or any other process termination) at any point during
    the write leaves the target either pre-write or post-write — the
    rename to *path* is the single atomic transition point.

    Args:
        path: Destination file path.  Parent directory is created
            if missing.
        content: Text to write.
        encoding: Text encoding (default: ``utf-8``).
        fsync: When True (default), flush + fsync the temp file
            before the rename.  Set False for write-heavy paths
            where post-crash data loss is acceptable but post-crash
            corruption is not (the rename atomicity holds either
            way; fsync controls whether the WRITE is durable, not
            whether it's atomic).
    """
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    # tempfile.mkstemp gives us a fresh fd + path in the same dir,
    # so the os.replace below is an intra-filesystem rename (atomic).
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        # POSIX rename(2): atomic intra-filesystem swap.  After
        # this returns, *path* either still has the old content
        # (rename hadn't completed) or has the new content
        # (rename completed); never a partial / truncated file.
        os.replace(tmp_path, target)
    except BaseException:
        # Cleanup on any failure (incl. KeyboardInterrupt /
        # SystemExit) so we don't leak temp files into the
        # persistent state directory.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(
    path: PathLike,
    data: Any,
    *,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
    fsync: bool = True,
) -> None:
    """Atomically write *data* to *path* as JSON.

    Convenience wrapper over :func:`atomic_write_text`.

    Args:
        path: Destination file path.
        data: JSON-serializable value.
        indent: ``json.dump`` indent (default: 2).  Pass ``None`` to
            produce compact JSON for high-frequency writes.
        ensure_ascii: ``json.dump`` ensure_ascii (default: False so
            unicode survives round-trips).
        fsync: Forwarded to :func:`atomic_write_text`.
    """
    serialized = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
    atomic_write_text(path, serialized, fsync=fsync)
