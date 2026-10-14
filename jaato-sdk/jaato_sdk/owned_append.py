"""Append-only files the daemon writes into a workspace it does not own (#1674).

The application trace (``JAATO_TRACE_LOG`` / ``trace.session_log``) and the
token ledger (``LEDGER_PATH`` / ``trace.ledger``) are appended to by TWO
processes: the daemon (its own trace lines, the Art. 50(1) announcement row)
and the session's runner.  Under a runner-uid policy that drops (#1168), a
root daemon that creates either file first creates it ``root:root 0644``,
and the runner -- now the workspace's owner -- can never append to it again:
``ledger: cannot append to ... ([Errno 13] Permission denied)``.  #1528 gave
the session record and the per-client logs their owner; these two writers
live below ``jaato_server.server`` (in the SDK and in ``shared``) and cannot
ask the daemon's policy, so they were left out.

This module is the seam between the two:

* **The daemon installs a resolver** (:func:`set_owner_resolver`), a callable
  mapping a path to the ``(uid, gid)`` it should belong to, or ``None``.
  Nothing else in the process decides an owner, and nothing here knows about
  policies or workspaces.  With no resolver installed (every client, every
  runner, a daemon on the default policy) :func:`open_append` is the plain
  ``open(path, "a")`` it replaced, byte for byte.
* **With an owner**, a file the daemon CREATES is ``fchown``-ed before a byte
  is written, and an existing file the daemon owns is handed over (repaired)
  as it is opened, so a ledger an older release left root-owned is fixed by
  the daemon's next append.

Rules the owned path holds to, because the directory is model-writable and
the daemon is root:

* the final component is opened ``O_NOFOLLOW``: a planted symlink is refused,
  never written through;
* only a regular file with ONE link is written or handed over: a hard link
  planted to a root-owned file elsewhere would otherwise be ``fchown``-ed;
* a file another account already owns is never taken from it;
* directory components this creates are handed to the owner; existing ones
  are left as they are.

Stdlib only.
"""

from __future__ import annotations

import io
import logging
import os
import stat
import threading
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

Owner = Tuple[int, int]
OwnerResolver = Callable[[str], Optional[Owner]]

_resolver_lock = threading.Lock()
_resolver: Optional[OwnerResolver] = None
_repaired: set = set()


def set_owner_resolver(resolver: Optional[OwnerResolver]) -> None:
    """Install (or, with ``None``, remove) the process-wide owner resolver.

    Called once by the daemon at startup when its runner-uid policy drops;
    tests pass ``None`` to restore the default.
    """
    global _resolver
    with _resolver_lock:
        _resolver = resolver
        _repaired.clear()


def owner_for(path: str) -> Optional[Owner]:
    """The owner *path* should belong to, or ``None``.  Never raises."""
    resolver = _resolver
    if resolver is None:
        return None
    try:
        return resolver(path)
    except Exception:  # noqa: BLE001 -- a broken resolver must not break a write
        logger.debug("owned_append: resolver failed for %s", path, exc_info=True)
        return None


def _make_dirs_owned(path: str, owner: Owner) -> None:
    """``mkdir -p`` *path*, handing each directory it CREATES to *owner*."""
    missing = []
    cursor = os.path.abspath(path)
    while not os.path.lexists(cursor):
        missing.append(cursor)
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        cursor = parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory)
        except FileExistsError:
            continue
        os.lchown(directory, owner[0], owner[1])


def ensure_parent_dirs(path: str) -> None:
    """Create *path*'s parent directories, owned when an owner applies."""
    parent = os.path.dirname(path)
    if not parent:
        return
    owner = owner_for(path)
    if owner is None:
        os.makedirs(parent, exist_ok=True)
    else:
        _make_dirs_owned(parent, owner)


def _hand_over_fd(fd: int, path: str, owner: Owner, created: bool) -> None:
    """``fchown`` the open *fd* to *owner* when that is safe; else refuse.

    Raises ``PermissionError`` for a file that is not a regular file with
    one link: neither is something these writers produce, and both are what
    a planted path in a model-writable directory looks like.
    """
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise PermissionError(
            f"refusing to write {path}: not a regular file with one link")
    if st.st_uid == owner[0] or st.st_uid != os.geteuid():
        return
    os.fchown(fd, owner[0], owner[1])
    if not created and path not in _repaired:
        _repaired.add(path)
        logger.info(
            "owned_append: handed %s (left uid %d by an earlier daemon) to "
            "uid %d so the session's runner can append to it",
            path, st.st_uid, owner[0])


def open_owned_fd(path: str, flags: int, mode: int = 0o666) -> int:
    """``os.open(path, flags | O_CREAT, mode)``, owned when an owner applies.

    For writers that need a raw descriptor (the ``flock`` sibling of a
    chained ledger).  *flags* must not carry ``O_CREAT`` or ``O_EXCL``.
    """
    owner = owner_for(path)
    if owner is None:
        return os.open(path, flags | os.O_CREAT, mode)
    _make_dirs_owned(os.path.dirname(path) or ".", owner)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL | nofollow, mode)
        created = True
    except FileExistsError:
        fd = os.open(path, flags | nofollow | nonblock)
        created = False
    try:
        _hand_over_fd(fd, path, owner, created)
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_append(path: str, encoding: Optional[str] = None) -> io.TextIOWrapper:
    """``open(path, "a")``, creating or repairing the file as its owner's.

    With no owner for *path* this is exactly ``open(path, "a",
    encoding=encoding)`` (the caller creates the directories, as before).
    """
    if owner_for(path) is None:
        return open(path, "a", encoding=encoding)
    fd = open_owned_fd(path, os.O_WRONLY | os.O_APPEND)
    if hasattr(os, "set_blocking"):
        os.set_blocking(fd, True)
    return os.fdopen(fd, "a", encoding=encoding)


def hand_over_existing(path: str, owner: Optional[Owner] = None) -> bool:
    """Repair an existing daemon-owned *path* without writing to it.

    For the spawn: a ledger the daemon left root-owned and will not append
    to again this session is handed over before the runner needs it.  A
    missing file, no owner, or a file that fails the checks above is left
    alone.  *owner* names the owner explicitly (the spawn, which runs outside
    any session's workspace context); else the installed resolver decides.
    Returns whether the file was handed over.  Never raises.
    """
    owner = owner if owner is not None else owner_for(path)
    if owner is None:
        return False
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, os.O_RDONLY | nofollow | nonblock)
    except OSError:
        return False
    try:
        before = os.fstat(fd).st_uid
        _hand_over_fd(fd, path, owner, created=False)
        return os.fstat(fd).st_uid != before
    except OSError as exc:
        logger.warning("owned_append: left %s as it is (%s)", path, exc)
        return False
    finally:
        os.close(fd)
