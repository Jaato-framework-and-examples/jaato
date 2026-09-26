"""The git checkouts a workspace holds, derived from disk (protocol 1.26).

A workspace does not DECLARE its sources; they are read off the tree every
time the workspace is listed: the workspace root itself (``path == "."``)
and each IMMEDIATE child directory that holds ``.git``.  Because this runs
on every ``workspace.list``, it spawns no git process -- ``.git/HEAD`` and
``.git/config`` are read as plain files.

Each source is ``{"forge", "repo", "branch", "path"}``:

* ``forge`` -- ``"github"`` / ``"gitlab"`` for the two public forges, else
  the remote's host; ``""`` when the checkout has no ``origin`` remote.
* ``repo`` -- ``"owner/name"`` (GitLab subgroups keep their full path);
  ``""`` without an ``origin``.
* ``branch`` -- the checked-out branch, or the short sha when HEAD is
  detached; ``""`` when HEAD cannot be read.
* ``path`` -- ``"."`` for the root, else the child directory's name.

Credentials embedded in a remote URL (``https://user:token@host/...``) are
never returned: only the host and the path are kept.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: Child directories never treated as sources: the framework's own state,
#: the per-workspace subprocess HOME, and anything hidden.
_SKIPPED_CHILDREN = frozenset({".jaato", ".home"})

#: Hosts that map to a forge NAME rather than being reported as a host.
_FORGE_HOSTS = {"github.com": "github", "gitlab.com": "gitlab"}

#: ``[remote "origin"]`` section header in a git config file.
_ORIGIN_SECTION = re.compile(r'^\s*\[\s*remote\s+"origin"\s*\]\s*$')
_ANY_SECTION = re.compile(r"^\s*\[")
_URL_KEY = re.compile(r"^\s*url\s*=\s*(.*?)\s*$", re.IGNORECASE)

#: scp-like remote: ``git@github.com:owner/name.git``.
_SCP_LIKE = re.compile(r"^(?:[^@/\s]+@)?([^:/\s]+):(?!//)(.+)$")


def git_dir_of(checkout: Path) -> Optional[Path]:
    """The git directory of *checkout*, or ``None`` when it is not one.

    Handles both a ``.git`` DIRECTORY and a ``.git`` FILE (a worktree or a
    submodule: ``gitdir: <path>``, relative paths resolved against the
    checkout).
    """
    dot_git = checkout / ".git"
    if dot_git.is_dir():
        return dot_git
    if not dot_git.is_file():
        return None
    try:
        text = dot_git.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    target = Path(text[len("gitdir:"):].strip())
    return target if target.is_absolute() else (checkout / target)


def read_head_branch(git_dir: Path) -> str:
    """The branch HEAD names, the short sha when detached, ``""`` if unreadable."""
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if head.startswith("ref:"):
        ref = head[len("ref:"):].strip()
        return ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
    return head[:7]


def _config_path(git_dir: Path) -> Path:
    """The config file for *git_dir*; a worktree's lives in the common dir."""
    common = git_dir / "commondir"
    if common.is_file():
        try:
            rel = common.read_text(encoding="utf-8", errors="replace").strip()
            base = Path(rel) if Path(rel).is_absolute() else git_dir / rel
            return base / "config"
        except OSError:
            pass
    return git_dir / "config"


def read_origin_url(git_dir: Path) -> str:
    """The ``url`` of the ``origin`` remote in *git_dir*'s config, or ``""``."""
    try:
        lines = _config_path(git_dir).read_text(
            encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    in_origin = False
    for line in lines:
        if _ANY_SECTION.match(line):
            in_origin = bool(_ORIGIN_SECTION.match(line))
            continue
        match = _URL_KEY.match(line) if in_origin else None
        if match:
            return match.group(1).strip('"')
    return ""


def _host_and_path(url: str) -> Tuple[str, str]:
    """Split a remote URL into ``(host, path)``, dropping any credentials."""
    if "://" in url:
        parts = urlsplit(url)
        return (parts.hostname or "").lower(), parts.path
    match = _SCP_LIKE.match(url)
    if match:
        return match.group(1).lower(), match.group(2)
    return "", ""


def parse_remote(url: str) -> Tuple[str, str]:
    """``(forge, repo)`` for a remote URL; ``("", "")`` when unparseable.

    Accepts ``https://[user[:token]@]host/owner/name[.git]``,
    ``ssh://git@host/owner/name.git`` and ``git@host:owner/name.git``.
    Only the host and path survive, so a credential in the URL never does.
    """
    host, path = _host_and_path(url.strip())
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    if not host or "/" not in path:
        return "", ""
    return _FORGE_HOSTS.get(host, host), path


def describe_checkout(checkout: Path, rel: str) -> Optional[Dict[str, str]]:
    """The source dict for *checkout*, or ``None`` when it is not a checkout."""
    git_dir = git_dir_of(checkout)
    if git_dir is None:
        return None
    forge, repo = parse_remote(read_origin_url(git_dir))
    return {
        "forge": forge,
        "repo": repo,
        "branch": read_head_branch(git_dir),
        "path": rel,
    }


def _child_checkouts(root: Path) -> List[Dict[str, str]]:
    """Sources from the immediate children of *root*, sorted by name."""
    found: List[Dict[str, str]] = []
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        return found
    for child in children:
        if child.name.startswith(".") or child.name in _SKIPPED_CHILDREN:
            continue
        if not child.is_dir() or child.is_symlink():
            continue
        source = describe_checkout(child, child.name)
        if source is not None:
            found.append(source)
    return found


def workspace_sources(root: Path) -> List[Dict[str, str]]:
    """Every git checkout in the workspace at *root*: itself, then children.

    Never raises: an unreadable tree yields what could be read, because this
    runs inside every ``workspace.list``.
    """
    sources: List[Dict[str, str]] = []
    try:
        own = describe_checkout(root, ".")
        if own is not None:
            sources.append(own)
        sources.extend(_child_checkouts(root))
    except OSError as exc:  # pragma: no cover -- defensive
        logger.debug("workspace_sources(%s): %s", root, exc)
    return sources
