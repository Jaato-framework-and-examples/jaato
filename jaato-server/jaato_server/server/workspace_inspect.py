"""What ``workspace.inspect`` reports about one workspace (protocol 1.26).

Pure functions over a directory and a session listing, so the WS handler
does only the transport half (resolve the caller's workspace, run this off
the event loop, send the answer).  Everything here is bounded:

* :func:`tree_size` gives up (``None``) past a number of entries or a wall
  clock, because a workspace holding a ``node_modules`` must not stall a
  picker;
* :func:`repo_status` runs ``git`` with a timeout, with no prompt and no
  optional locks, and only verbs that never touch the network
  (``status --porcelain``, ``rev-list --count @{upstream}..HEAD`` -- which
  reads the LOCAL remote-tracking ref, so "unpushed" is relative to the
  last fetch).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .workspace_sources import workspace_sources

#: Per-git-invocation timeout, seconds.
GIT_TIMEOUT_SECONDS = 10.0
#: :func:`tree_size` bounds.
SIZE_MAX_ENTRIES = 200_000
SIZE_MAX_SECONDS = 5.0

#: git's answers when there is simply no upstream to compare against --
#: a fact about the branch, not a failure.
_NO_UPSTREAM_MARKERS = ("no upstream", "does not point to a branch",
                        "unknown revision", "no such branch")


def _git_env() -> Dict[str, str]:
    """Environment for a read-only, non-interactive, lock-free git call."""
    env = dict(os.environ)
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
    })
    return env


def _run_git(checkout: Path, *args: str) -> subprocess.CompletedProcess:
    """Run ``git -C <checkout> <args>``; raises on timeout / missing git."""
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS,
        env=_git_env(), stdin=subprocess.DEVNULL, check=False,
    )


def _unpushed(checkout: Path) -> Optional[int]:
    """Commits ahead of the upstream; ``None`` when there is no upstream.

    Raises:
        RuntimeError: git failed for another reason (its stderr).
    """
    done = _run_git(checkout, "rev-list", "--count", "@{upstream}..HEAD")
    if done.returncode == 0:
        return int(done.stdout.strip() or 0)
    err = done.stderr.strip()
    if any(marker in err.lower() for marker in _NO_UPSTREAM_MARKERS):
        return None
    raise RuntimeError(err or f"git rev-list exited {done.returncode}")


def repo_status(checkout: Path) -> Dict[str, Any]:
    """``{uncommitted, unpushed, error}`` for one checkout.

    ``uncommitted`` is the number of lines ``git status --porcelain``
    prints.  When git fails, both counts are ``None`` and ``error`` says
    why -- a count of 0 would read as "clean", which it was not shown to be.
    """
    try:
        status = _run_git(checkout, "status", "--porcelain")
        if status.returncode != 0:
            raise RuntimeError(status.stderr.strip()
                               or f"git status exited {status.returncode}")
        uncommitted = len([ln for ln in status.stdout.splitlines() if ln.strip()])
        return {"uncommitted": uncommitted, "unpushed": _unpushed(checkout),
                "error": ""}
    except subprocess.TimeoutExpired:
        return {"uncommitted": None, "unpushed": None,
                "error": f"git timed out after {GIT_TIMEOUT_SECONDS:g}s"}
    except (OSError, RuntimeError, ValueError) as exc:
        return {"uncommitted": None, "unpushed": None, "error": str(exc)}


def tree_size(
    root: Path,
    max_entries: int = SIZE_MAX_ENTRIES,
    max_seconds: float = SIZE_MAX_SECONDS,
) -> Optional[int]:
    """Recursive sum of file sizes under *root*, or ``None`` past a bound.

    Symlinks are neither followed nor counted; unreadable entries are
    skipped.  ``None`` means "too large or too slow to measure", never 0.
    """
    deadline = time.monotonic() + max_seconds
    total = 0
    seen = 0
    stack = [str(root)]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    seen += 1
                    if seen > max_entries or time.monotonic() > deadline:
                        return None
                    total += _entry_size(entry, stack)
        except OSError:
            continue
    return total


def _entry_size(entry: os.DirEntry, stack: List[str]) -> int:
    """Size of a regular file; a directory is queued on *stack* instead."""
    try:
        if entry.is_dir(follow_symlinks=False):
            stack.append(entry.path)
            return 0
        if entry.is_file(follow_symlinks=False):
            return entry.stat(follow_symlinks=False).st_size
    except OSError:
        pass
    return 0


def _same_workspace(candidate: Optional[str], root: str) -> bool:
    """Whether a session's *candidate* workspace path is the workspace *root*."""
    if not candidate:
        return False
    return os.path.realpath(candidate) == root


def session_counts(rows: Iterable[Any], workspace_path: str) -> Dict[str, int]:
    """``{total, waiting, awake, sleeping}`` for the sessions in a workspace.

    ``sleeping`` = persisted and not loaded; ``waiting`` = loaded and blocked
    on a person (``awaiting`` set, #1138); ``awake`` = loaded otherwise.
    *rows* are ``SessionManager.list_sessions()`` entries.
    """
    root = os.path.realpath(workspace_path)
    counts = {"total": 0, "waiting": 0, "awake": 0, "sleeping": 0}
    for row in rows:
        if not _same_workspace(getattr(row, "workspace_path", None), root):
            continue
        counts["total"] += 1
        if not getattr(row, "is_loaded", False):
            counts["sleeping"] += 1
        elif getattr(row, "awaiting", None):
            counts["waiting"] += 1
        else:
            counts["awake"] += 1
    return counts


def inspect_repos(root: Path) -> List[Dict[str, Any]]:
    """Every source of the workspace at *root*, with its git status merged in."""
    repos: List[Dict[str, Any]] = []
    for source in workspace_sources(root):
        checkout = root if source["path"] == "." else root / source["path"]
        repos.append({**source, **repo_status(checkout)})
    return repos


def inspect_workspace(root: Path, session_rows: Iterable[Any]) -> Dict[str, Any]:
    """The fields of a ``WorkspaceInspectEvent`` for the workspace at *root*.

    Blocking (runs git, walks the tree): call it off the event loop.
    """
    return {
        "path": str(root),
        "size_bytes": tree_size(root),
        "sessions": session_counts(session_rows, str(root)),
        "repos": inspect_repos(root),
    }
