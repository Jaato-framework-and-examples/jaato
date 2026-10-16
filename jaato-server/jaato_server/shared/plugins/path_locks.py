"""Per-path locks for tools that read a file and write it back (#1718).

Tool calls in one model response run in parallel (``JaatoSession``'s tool
pool, 8 wide by default), and in-process subagents share the same process
and plugin instances.  A tool that loads a file, computes a change from what
it loaded and writes the result is a read-modify-write: two such calls on one
file both read the original, and the second write discards the first.  Each
call reports success, so the lost edit is invisible until something re-reads
the file.

:func:`locked_paths` serialises those sequences per file.  Every caller that
reads a file in order to rewrite it (``updateFile``, ``multiFileEdit``,
``findAndReplace``, ...) holds the file's lock from the read until the write
has landed, so a second call reads what the first wrote.

Rules the implementation holds to:

- **One key per file.**  The key is the path after ``os.path.realpath`` and
  ``os.path.normcase``, so ``a/../x``, a symlink to ``x`` and ``x`` itself
  share one lock.  A path that does not exist yet (``writeNewFile``) is keyed
  on its resolved parent plus its name, which is what it will be once created.
- **Several files are taken in one sorted order.**  ``locked_paths(a, b)``
  and ``locked_paths(b, a)`` acquire in the same order, so two batches over
  overlapping files cannot deadlock each other.
- **Re-entrant.**  The locks are ``RLock`` s: a helper called with the lock
  already held (a backup, a nested call) takes it again without blocking.
- **Bounded.**  An entry exists only while some caller holds or waits for it;
  the registry is reference-counted and drops a key when its last user
  leaves, so a long session that touched thousands of files keeps none.

Scope, stated: the locks are **per process**.  They serialise every tool call
of the runner process (the session, its in-process subagents and every
parallel call of either).  They do not serialise a different process writing
the same file: another runner on the same workspace, a ``cli`` command, an
editor.  Callers that read in order to rewrite therefore also compare the
file's identity before writing (:class:`FileFingerprint`) and refuse the
write when it changed underneath them, so a lost update across processes is
an error instead of a silent success, short of the remaining window between
that check and the write.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

PathLike = Union[str, "os.PathLike[str]"]

_registry_lock = threading.Lock()
# key -> [lock, number of callers holding or waiting for it]
_registry: Dict[str, List] = {}


def lock_key(path: PathLike) -> str:
    """Return the canonical key a file's lock is registered under.

    ``os.path.realpath`` resolves every symlink it can and keeps the
    non-existent tail as written, so a file about to be created gets the key
    it will have afterwards.
    """
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _acquire_entry(key: str) -> threading.RLock:
    with _registry_lock:
        entry = _registry.get(key)
        if entry is None:
            entry = [threading.RLock(), 0]
            _registry[key] = entry
        entry[1] += 1
        return entry[0]


def _release_entry(key: str) -> None:
    with _registry_lock:
        entry = _registry.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            del _registry[key]


@contextmanager
def locked_paths(*paths: Optional[PathLike]) -> Iterator[Tuple[str, ...]]:
    """Hold the lock of every given file for the duration of the block.

    ``None`` entries are ignored and duplicates are taken once, so a caller
    may pass optional or overlapping paths (a move's source and
    destination, a batch naming one file twice) as they come.  Yields the
    sorted keys that are held.
    """
    keys = sorted({lock_key(p) for p in paths if p is not None and os.fspath(p)})
    acquired: List[Tuple[str, threading.RLock]] = []
    try:
        for key in keys:
            lock = _acquire_entry(key)
            try:
                lock.acquire()
            except BaseException:
                _release_entry(key)
                raise
            acquired.append((key, lock))
        yield tuple(keys)
    finally:
        for key, lock in reversed(acquired):
            lock.release()
            _release_entry(key)


def held_key_count() -> int:
    """Number of keys currently registered (for tests and diagnostics)."""
    with _registry_lock:
        return len(_registry)


@dataclass(frozen=True)
class FileFingerprint:
    """What a file looked like when a caller read it.

    Compared again just before the write: any difference means something
    outside this process's locks (another runner, a shell command, an
    editor) wrote the file after it was read, and writing a change computed
    from the old content would discard that write.
    """

    exists: bool
    ino: int = 0
    dev: int = 0
    size: int = 0
    mtime_ns: int = 0

    @classmethod
    def of(cls, path: PathLike) -> "FileFingerprint":
        try:
            st = os.stat(os.fspath(path))
        except FileNotFoundError:
            return cls(exists=False)
        return cls(
            exists=True,
            ino=st.st_ino,
            dev=st.st_dev,
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
        )


class FileChangedError(OSError):
    """The file changed between the read a write was computed from and the write.

    An ``OSError`` so existing ``except OSError`` handlers report it as a
    failed tool call rather than crashing the turn.
    """


def assert_unchanged(path: PathLike, before: FileFingerprint, *, what: str = "file") -> None:
    """Raise :class:`FileChangedError` when ``path`` no longer matches ``before``."""
    if FileFingerprint.of(path) != before:
        raise FileChangedError(
            f"{what} {os.fspath(path)} was modified by another process after it "
            "was read; nothing was written. Re-read the file and apply the "
            "change again."
        )
