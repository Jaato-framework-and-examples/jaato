"""Find files in the caller's workspace by name (protocol 1.32).

The Files panel lists what CHANGED; it has no way to reach a file nobody
touched this session, one the user hid, or one git ignores.
``workspace.files.search`` answers "where is the file called X" over the
whole workspace tree, so a client can locate any file and hand its path to
``workspace.file.fetch``.

This module is the one place that decides what a search returns -- no
socket code, so the rules are testable without a daemon:

- **The whole tree, hidden or not.**  Dotfiles, gitignored paths and the
  panel's hidden entries are all included; the panel's own filters answer a
  different question ("what should I look at"), not "what exists".  Only the
  contents of ``.git`` directories are skipped: the object store is not a
  file anybody locates by name, and it would crowd out every real match.
- **Nothing outside the workspace.**  The walk starts at the RESOLVED root
  and does not descend into a directory symlink, so a link the agent
  planted cannot extend the search to the rest of the host.  A symlinked
  FILE is listed under its own name; fetching it is still judged by
  :func:`~jaato_server.server.workspace_download.resolve_download`.
- **Bounded, and says so.**  The walk stops after
  :data:`MAX_SCANNED_ENTRIES` entries or :data:`WALK_DEADLINE_SECONDS`, and
  the answer carries ``truncated`` -- a search that stopped early must not
  read as "no such file".
- **Credential files are listed and marked.**  A path is not a secret, and
  the user may be looking for the ``.env`` precisely to know it is there;
  ``credential`` tells the client not to offer a download the fetch verb
  would refuse.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List

from .workspace_download import is_credential_path

#: Entries (files and directories) the walk may visit before stopping.
MAX_SCANNED_ENTRIES = 200_000
#: Wall-clock bound on one walk.
WALK_DEADLINE_SECONDS = 5.0
#: Default and ceiling for how many matches one answer carries.
DEFAULT_MAX_RESULTS = 100
MAX_RESULTS_CEILING = 500
#: Directory names whose CONTENTS are never searched.
SKIPPED_DIRS = frozenset({".git"})


@dataclass
class SearchResult:
    """What one search found.

    ``matches`` is ranked, best first, and capped.  ``total`` is how many
    matched before the cap.  ``truncated`` means the WALK stopped early, so
    files beyond the bound were never looked at -- distinct from
    ``total > len(matches)``, which only means the answer was capped.
    """

    matches: List[dict] = field(default_factory=list)
    total: int = 0
    truncated: bool = False
    scanned: int = 0


def _terms(query: str) -> List[str]:
    return [t for t in query.lower().split() if t]


def _rank(relpath: str, terms: List[str]) -> tuple:
    """Lower sorts first: a match in the basename beats one in a directory."""
    name = relpath.rsplit("/", 1)[-1].lower()
    in_name = sum(1 for t in terms if t in name)
    exact = 0 if name in terms else 1
    return (exact, -in_name, relpath.count("/"), len(relpath), relpath)


def search_workspace(
    workspace_root: str,
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    *,
    clock=time.monotonic,
) -> SearchResult:
    """Every file under ``workspace_root`` whose relative path holds all terms.

    ``query`` is split on whitespace; each term must appear, case-folded,
    somewhere in the POSIX relative path (so ``src test`` finds
    ``src/foo/test_bar.py``).  An empty query matches nothing.  Each match is
    ``{"path", "size", "credential"}``.
    """
    terms = _terms(query)
    result = SearchResult()
    if not terms:
        return result
    limit = max(1, min(int(max_results or DEFAULT_MAX_RESULTS), MAX_RESULTS_CEILING))
    root = os.path.realpath(workspace_root)
    deadline = clock() + WALK_DEADLINE_SECONDS
    found: List[dict] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            result.scanned += 1
            if result.scanned > MAX_SCANNED_ENTRIES or clock() > deadline:
                result.truncated = True
                stack.clear()
                break
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in SKIPPED_DIRS:
                        stack.append(entry.path)
                    continue
            except OSError:
                continue
            rel = os.path.relpath(entry.path, root).replace(os.sep, "/")
            lowered = rel.lower()
            if not all(t in lowered for t in terms):
                continue
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                size = 0
            found.append({"path": rel, "size": size, "credential": is_credential_path(rel)})
    found.sort(key=lambda m: _rank(m["path"], terms))
    result.total = len(found)
    result.matches = found[:limit]
    return result
