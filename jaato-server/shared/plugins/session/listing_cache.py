"""A stat-keyed memo of the header each session record carries (#1137).

``FileSessionPlugin.list_sessions`` parses every record WHOLE to keep the
header ``deserialize_session_info`` returns -- id, description, timestamps,
turn count, profile, workspace, cascade address -- and discards the rest.
The rest is the transcript: ``history``, ``rendered_instructions``,
``profile_snapshot``, and any binary media that has not been evicted.  So
the cost of a listing tracks transcript LENGTH, and nothing bounds it.

The listing is read far more often than a record changes.  The web client's
session picker calls ``session.list`` when it opens and ``sessionListSilent``
refreshes it in the background, and ``SessionManager.list_sessions`` does
that for every known workspace -- so a poll re-parsed every transcript in
every workspace to re-derive a header that had not moved.  This memo answers
from a ``stat`` when the file is the one already parsed.

It is a cache, not a second source of truth: nothing is written, the record
on disk remains the only place a header lives, and a miss produces exactly
the parse the uncached path performed.  That is the whole reason it was
preferred to a sidecar header file -- see the PR for #1137.

WHAT MAKES AN ENTRY VALID
-------------------------
Two conditions, and the second exists because the first is not sufficient.

* **The stamp matches.**  ``(mtime_ns, size, inode)`` of the file now equals
  the stamp recorded when its bytes were parsed.  The inode is in there
  because :meth:`FileSessionPlugin.save` writes a temporary sibling and
  ``os.replace``\\ s it into position, so every save this tree performs
  lands on a NEW inode -- which makes the stamp change even where a
  filesystem's timestamp granularity would not.

* **The file was not freshly written when it was read.**  A filesystem whose
  ``st_mtime`` granularity is coarse (1 s is still common on network and
  container filesystems) can report the same mtime for a write that lands
  *after* the read that produced the entry.  Pair that with a rewrite of
  identical size onto a reused inode and the memo would serve a header the
  record no longer has -- and, on a session that is never saved again,
  serve it for the life of the daemon.  So an entry read within
  ``RACY_WINDOW_NS`` of the record's own mtime is not trusted, and the next
  listing re-reads it and re-stamps with a later observation.  The rule is
  git's "racily clean" index rule; the cost is that the one record being
  actively written -- the live session's -- is re-read rather than served,
  which is the record most likely to have changed anyway.

Both conditions are checked against a ``stat``, never against the clock
alone, so a system-clock step backwards makes entries look racy (they are
re-read: the pre-#1137 behaviour) rather than making stale ones look fresh.

WHAT IS CACHED, AND WHAT IS HANDED BACK
---------------------------------------
A parse that FAILED is cached too, as ``None``.  A corrupted record is
skipped with a warning on every listing, so a poll re-read and re-warned
about a file it could not use; remembering the verdict means the warning is
emitted once per version of the file rather than once per poll, and the
unreadable bytes are read once.  Distinguishing "no entry" from "an entry
whose value is ``None``" is what makes that work, so :meth:`lookup` returns
the :data:`MISS` sentinel rather than ``None`` for the absent case.

:class:`SessionInfo` is a mutable dataclass, so the memo copies on the way in
AND on the way out.  Either alone leaves a window: handing out the cached
instance makes one caller's edit the next listing's answer, and storing the
caller's instance hands the memo an object the caller already holds.  A cache
that mutates under its readers is worse than no cache.  Every field is an
immutable scalar, so the shallow copy is a full one.

The memo is bounded by what is on disk: :meth:`retain` drops the entries for
a directory's records that the directory no longer holds, so a deleted
session stops being remembered on the first listing after its deletion.
Entries for OTHER directories are untouched -- one plugin instance serves
every workspace the daemon knows.
"""

import dataclasses
import os
import threading
import time
from typing import Dict, Iterable, NamedTuple, Optional, Union

from .base import SessionInfo


class _Missing:
    """Type of :data:`MISS`; distinguishes "not cached" from a cached ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<listing-cache MISS>"


MISS = _Missing()
"""Returned by :meth:`SessionListingCache.lookup` when nothing is known.

A cached *failed* parse is returned as ``None``, which is a different fact:
the file was read and could not be used.
"""


class Stamp(NamedTuple):
    """What identifies the exact bytes an entry was parsed from.

    ``inode`` is not redundant with ``(mtime_ns, size)``: the atomic-rename
    save path gives every write a new inode, so it is the discriminator that
    still works on a filesystem with coarse timestamps.
    """

    mtime_ns: int
    size: int
    inode: int

    @classmethod
    def of(cls, st: os.stat_result) -> "Stamp":
        return cls(st.st_mtime_ns, st.st_size, st.st_ino)


class _Entry(NamedTuple):
    stamp: Stamp
    observed_ns: int
    """Wall clock, sampled BEFORE the read that produced ``value``."""
    value: Optional[SessionInfo]
    """The parsed header, or ``None`` when the record could not be parsed."""


class SessionListingCache:
    """Remembers a session record's header for as long as its bytes last.

    Thread-safe: ``session.list`` is served from whichever thread the
    command router is on, and one instance is shared by every workspace a
    daemon knows.
    """

    RACY_WINDOW_NS: int = 1_000_000_000
    """How recently written a record may be and still be memoised.

    One second, because that is the coarsest ``st_mtime`` granularity in
    practical use; a finer filesystem simply makes the window generous.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # directory -> filename -> entry.  Nested rather than keyed on the
        # full path so ``retain`` can prune one directory without walking
        # (or accidentally pruning) the others.
        self._by_dir: Dict[str, Dict[str, _Entry]] = {}

    def lookup(
        self,
        directory: str,
        name: str,
        st: os.stat_result,
    ) -> Union[SessionInfo, None, _Missing]:
        """Return the memoised header, ``None`` for a memoised failure, or :data:`MISS`.

        Args:
            directory: Storage directory, as the caller spells it.
            name: The record's file name within that directory.
            st: A fresh ``stat`` of the record.

        Returns:
            A copy of the cached :class:`SessionInfo`; ``None`` when the
            record's current bytes are known to be unparseable; or
            :data:`MISS` when nothing valid is remembered and the caller
            must read the file.
        """
        with self._lock:
            entry = self._by_dir.get(directory, {}).get(name)
        if entry is None or entry.stamp != Stamp.of(st):
            return MISS
        if entry.observed_ns - entry.stamp.mtime_ns <= self.RACY_WINDOW_NS:
            # Read too soon after the record's own mtime to rule out a
            # same-tick rewrite behind us.  Re-read; the next store stamps a
            # later observation and the entry settles.
            return MISS
        if entry.value is None:
            return None
        return dataclasses.replace(entry.value)

    def store(
        self,
        directory: str,
        name: str,
        st: os.stat_result,
        observed_ns: int,
        value: Optional[SessionInfo],
    ) -> None:
        """Memoise ``value`` as the header of the bytes ``st`` describes.

        Args:
            directory: Storage directory, as the caller spells it.
            name: The record's file name within that directory.
            st: ``fstat`` of the handle the bytes were read from, so the
                stamp describes exactly what was parsed rather than
                whatever the path pointed at a moment earlier.
            observed_ns: Wall clock sampled BEFORE the read.  Storing the
                later time would make an entry look older than it is,
                which is the direction that serves stale headers.
            value: The parsed header, or ``None`` when the record could
                not be parsed.

        The value is COPIED on the way in as well as on the way out.  The
        caller that fills the memo is also the caller that returns that
        parse to the listing, so storing its instance would leave the memo
        holding an object someone else already has a reference to -- and a
        mutation through that reference is indistinguishable, a listing
        later, from the record having said so.
        """
        entry = _Entry(
            Stamp.of(st),
            observed_ns,
            None if value is None else dataclasses.replace(value),
        )
        with self._lock:
            self._by_dir.setdefault(directory, {})[name] = entry

    def retain(self, directory: str, names: Iterable[str]) -> None:
        """Drop this directory's entries for records ``names`` does not hold.

        Called with the file names one listing saw, so the memo is bounded
        by what is on disk rather than by everything that ever was.
        """
        keep = set(names)
        with self._lock:
            entries = self._by_dir.get(directory)
            if entries is None:
                return
            for name in [n for n in entries if n not in keep]:
                del entries[name]
            if not entries:
                del self._by_dir[directory]

    def clear(self) -> None:
        """Forget everything.  Used at plugin shutdown."""
        with self._lock:
            self._by_dir.clear()

    def stats(self) -> Dict[str, int]:
        """Entry counts, for tests and diagnostics."""
        with self._lock:
            return {d: len(e) for d, e in self._by_dir.items()}


def now_ns() -> int:
    """Wall clock in nanoseconds, comparable with ``st_mtime_ns``.

    Named so the racy-window rule reads against one clock rather than two:
    both this and ``st_mtime_ns`` are the realtime clock, which is what makes
    their difference meaningful.
    """
    return time.time_ns()
