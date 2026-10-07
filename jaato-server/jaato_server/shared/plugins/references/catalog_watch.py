"""Has the references catalog changed on disk since this session loaded it? (#1145)

A session loads its catalog once; another session's promotion, a link edit,
a ``bundle merge`` or a ``git pull`` writes the files after that.  This
module answers the one question a running session needs before it reads the
catalog, cheaply enough to ask on every read: *did anything I loaded from
change?*

The answer is a ``stat`` per directory, never a scan.  Two facts make that
sufficient:

* **Creating, deleting or renaming a file changes its directory's mtime.**
  Every writer of catalog files in this tree writes a temp file and
  ``os.replace``-s it into place (``write_contained``, ``reconcile``,
  ``merge``), so a framework EDIT of an existing reference is a rename and
  moves the directory mtime too.
* **An in-place edit does not**, so each reference file present at the load
  is stamped as well (:func:`json_files`; #1437).  An editor saving over a
  file, or a promoted REVISION written by a writer that does not rename,
  moves the file's own mtime and size.  One ``stat`` per file is the price;
  a catalog is tens or hundreds of files, not thousands.
* **A new sub-bundle is a new directory in a tier root**, which moves the
  root's mtime; the reload that follows rediscovers the bundle list, so the
  watched set grows with the catalog.

Files are watched individually where a directory mtime would say too much:
``references.json`` sits in the workspace root, whose mtime moves with every
file created there.  The candidate list is
:func:`config_loader.config_path_candidates`, the one :func:`load_config`
walks.

A snapshot is stamped with ``(st_mtime_ns, st_ino, st_size)``; an absent or
unreadable path is ``None``, so a directory that appears later is a change.

**A write racing the load is not lost.**  Two ways it could be:

* a write landing DURING the load can be missing from the catalog and
  present in a snapshot taken afterwards.  :func:`settle` compares the
  snapshot taken before the load with the one taken after and marks every
  path that moved in between;
* a write landing just AFTER the load can leave the directory's mtime
  unchanged, because the kernel stamps mtimes from a coarse clock (a timer
  tick, several milliseconds).  :func:`settle` also marks every path whose
  mtime is within :data:`UNSETTLED_WINDOW_NS` of the snapshot, the same
  "racily clean" rule git applies to its index.

A marked path is stored as :data:`UNSETTLED`, which equals no real stamp,
so the next check reloads.  The cost is one extra reload after a load that
followed a write by less than the window.

Stdlib only; no plugin state.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

#: One path's stamp: ``(mtime_ns, inode, size)``, or ``None`` when the path
#: is absent or cannot be read.
Stamp = Optional[Tuple[int, int, int]]

#: Stored for a path whose stamp cannot vouch for the load (see the module
#: docstring).  A string, so it equals no :data:`Stamp` and the next check
#: sees a change.
UNSETTLED = "unsettled"

#: How recent an mtime must be, at snapshot time, to be distrusted: well
#: above a kernel timer tick (up to 10 ms) and the 1 s resolution of the
#: coarsest filesystems a workspace is likely to sit on.
UNSETTLED_WINDOW_NS = 2_000_000_000


def stat_path(path: str) -> Stamp:
    """The stamp of one path; ``None`` when it is absent or unreadable."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_ino, st.st_size)


def json_files(directories: Iterable[str]) -> List[str]:
    """Every ``*.json`` regular file directly in each of ``directories``.

    One ``scandir`` per directory; a link is not followed and an unreadable
    directory contributes nothing.  These are the reference files a load
    read (plus manifests, harmless to stamp), watched individually so an
    in-place edit is seen.
    """
    out: List[str] = []
    for directory in directories:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if (entry.name.endswith(".json")
                            and entry.is_file(follow_symlinks=False)):
                        out.append(entry.path)
        except OSError:
            continue
    return out


def stat_paths(paths: Iterable[str]) -> Dict[str, Stamp]:
    """Stamp every path in ``paths``."""
    return {p: stat_path(p) for p in paths}


def settle(
    pre: Mapping[str, object],
    post: Dict[str, Stamp],
    now_ns: Optional[int] = None,
) -> Dict[str, object]:
    """The snapshot to keep after a load.

    ``pre`` was taken before the load began, ``post`` after it ended, at
    ``now_ns`` (``time.time_ns()`` when omitted).  A path is stored as
    :data:`UNSETTLED` when its stamp moved between the two (written during
    the load) or its mtime is within :data:`UNSETTLED_WINDOW_NS` of
    ``now_ns`` (a later write in the same clock tick would not move it).
    Every other path keeps its ``post`` stamp.
    """
    if now_ns is None:
        now_ns = time.time_ns()
    stored: Dict[str, object] = {}
    for path, stamp in post.items():
        moved = path in pre and pre[path] != stamp
        recent = stamp is not None and now_ns - stamp[0] < UNSETTLED_WINDOW_NS
        stored[path] = UNSETTLED if (moved or recent) else stamp
    return stored


def has_changed(stored: Mapping[str, object]) -> bool:
    """Whether any path in ``stored`` no longer carries its stored stamp.

    An empty ``stored`` means nothing was loaded from disk (or the session
    has not loaded yet), and is never a change.
    """
    for path, stamp in stored.items():
        if stat_path(path) != stamp:
            return True
    return False
