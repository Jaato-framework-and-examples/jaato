"""The user-global config tier (``~/.jaato``), as a runner reads it (#1465).

A confined runner is not granted most of ``~/.jaato``: AppArmor grants only
the subtrees plugins declare (``agents/``, ``profiles/``, ``references/``,
``memories/`` ...), and SELinux mirrors that.  Before #1465 the runner read
the rest anyway, so a present ``~/.jaato/permissions.json`` raised
``PermissionError`` at bootstrap and every confined session was refused.

The daemon is not confined, so it reads those files and ships them on the
session envelope (``SessionInitEnvelope.user_tier_files``), the way it
already ships the resolved profile and the rendered persona.  This module
is the one place both halves meet:

- :func:`collect` (daemon): read the files in :data:`SHIPPED_FILES` and
  under :data:`SHIPPED_DIRS` from one ``~/.jaato``.
- :func:`install` (runner, bootstrap): write the snapshot under the
  session's temp directory and point :func:`path` at it.
- :func:`path` (every runner-side reader of those files): the snapshot's
  copy when one is installed, else ``~/.jaato/<rel>`` on disk.
- :func:`home_path`: the same for the read-only interop directories OUTSIDE
  ``~/.jaato`` a plugin reads (:data:`HOME_SHIPPED_DIRS`, Claude Code's
  ``~/.claude/skills``).  AppArmor grants the prompt library that read;
  SELinux leaves the directory the home's type, so it is shipped instead
  (phase 2b kernel run).  Their snapshot keys carry :data:`HOME_PREFIX`,
  which no path under ``~/.jaato`` can start with.

Absent snapshot (``None``) means an older daemon or an in-process session:
readers read disk exactly as before.  An EMPTY snapshot (``{}``) means the
daemon looked and found nothing, and the runner must not read disk.

What is never shipped: credentials (``*_auth.json``, OAuth token and
account files, see :data:`CREDENTIAL_SUFFIXES`).  They stay unreadable to a
confined runner; the key belongs in ``session_env`` (#1461).  Writes are
not routed here: a writer keeps the real home path and fails there as it
did before, rather than writing into a copy nobody reads back.

Daemon-side rules, because a root daemon reads a dropped user's home
(#1168): no path component under ``~/.jaato`` may be a symlink (the file
is opened ``O_NOFOLLOW`` and every directory is ``lstat``-ed), and only
regular files owned by the owner of ``~/.jaato`` are read, so neither a
symlink nor a hard link can carry another file into a session.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

#: Files at the top of ``~/.jaato`` a runner reads and is not granted.
SHIPPED_FILES: Tuple[str, ...] = (
    "permissions.json",
    "system_instructions.md",
    "sandbox_paths.json",
    "reliability.json",
    "reliability-policies.json",
    "webhook.json",
    "thinking.json",
    "pricing.json",
)

#: Directories under ``~/.jaato`` shipped whole (regular files, recursively).
SHIPPED_DIRS: Tuple[str, ...] = (
    "instructions",
    "completion_schemas",
    "spawn_schemas",
    "scripts",
)

#: Read-only directories under the HOME (not ``~/.jaato``) shipped whole.
HOME_SHIPPED_DIRS: Tuple[str, ...] = (
    ".claude/skills",
)

#: Snapshot-key prefix of a :data:`HOME_SHIPPED_DIRS` entry.
HOME_PREFIX = "@home/"

#: Names that are credentials wherever they appear; never shipped.
CREDENTIAL_SUFFIXES: Tuple[str, ...] = (
    "_auth.json", "_oauth.json", "_accounts.json",
)

MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024

#: Directory name of an installed snapshot under the session temp dir.
SNAPSHOT_DIRNAME = "jaato-user-tier"

_installed_root: Optional[Path] = None


def is_credential(name: str) -> bool:
    """Is *name* (a file name) a stored credential?"""
    return name.endswith(CREDENTIAL_SUFFIXES)


# ----------------------------------------------------------------------
# Daemon side
# ----------------------------------------------------------------------

def collect(jaato_dir: str) -> Dict[str, str]:
    """The snapshot of *jaato_dir*: relative path -> UTF-8 text.

    ``{}`` when *jaato_dir* does not exist or is not a real directory.
    A file that is skipped (symlinked, foreign-owned, too large, not
    UTF-8, unreadable) is named in a WARNING and left out.
    """
    out: Dict[str, str] = {}
    budget = [MAX_TOTAL_BYTES]
    try:
        top = os.lstat(jaato_dir)
    except FileNotFoundError:
        top = None
    if top is not None and not stat.S_ISDIR(top.st_mode):
        logger.warning("user tier: %s is not a directory (a symlink?); "
                       "nothing shipped", jaato_dir)
        return {}
    if top is not None:
        _add(out, budget, jaato_dir, _candidates(jaato_dir), top.st_uid, "")
    _collect_home_dirs(out, budget, os.path.dirname(jaato_dir.rstrip("/")))
    return out


def _add(out: Dict[str, str], budget: list, base: str,
         rels: Iterator[str], owner: int, prefix: str) -> None:
    """Read each of *rels* under *base* into *out*, within the size budget."""
    for rel in rels:
        text = _read_owned(base, rel, owner)
        if text is None:
            continue
        size = len(text.encode("utf-8"))
        if size > budget[0]:
            logger.warning("user tier: %s/%s skipped, the snapshot would "
                           "exceed %d bytes", base, rel, MAX_TOTAL_BYTES)
            continue
        out[prefix + rel] = text
        budget[0] -= size


def _collect_home_dirs(out: Dict[str, str], budget: list, home: str) -> None:
    """:data:`HOME_SHIPPED_DIRS`, owned by the home's owner, no symlinked step."""
    try:
        owner = os.lstat(home).st_uid
    except OSError:
        return
    for rel_dir in HOME_SHIPPED_DIRS:
        if _real_dirs(home, rel_dir):
            _add(out, budget, home, _walk(home, rel_dir), owner, HOME_PREFIX)


def _real_dirs(base: str, rel_dir: str) -> bool:
    """Is every component of *rel_dir* under *base* a real directory?"""
    current = base
    for part in rel_dir.split("/"):
        current = os.path.join(current, part)
        try:
            if not stat.S_ISDIR(os.lstat(current).st_mode):
                return False
        except OSError:
            return False
    return True


def _candidates(jaato_dir: str) -> Iterator[str]:
    """Relative paths to consider, in a stable order."""
    for name in SHIPPED_FILES:
        yield name
    for top in SHIPPED_DIRS:
        yield from _walk(jaato_dir, top)


def _walk(jaato_dir: str, rel_dir: str) -> Iterator[str]:
    """Regular-file candidates under *rel_dir*; symlinked dirs not entered."""
    try:
        st = os.lstat(os.path.join(jaato_dir, rel_dir))
        if not stat.S_ISDIR(st.st_mode):
            return
        names = sorted(os.listdir(os.path.join(jaato_dir, rel_dir)))
    except OSError:
        return
    for name in names:
        rel = f"{rel_dir}/{name}"
        try:
            mode = os.lstat(os.path.join(jaato_dir, rel)).st_mode
        except OSError:
            continue
        if stat.S_ISDIR(mode):
            yield from _walk(jaato_dir, rel)
        elif stat.S_ISREG(mode):
            yield rel


def _read_owned(jaato_dir: str, rel: str, owner: int) -> Optional[str]:
    """*rel*'s text, or ``None`` when absent or skipped (with a WARNING)."""
    if is_credential(os.path.basename(rel)):
        return None
    full = os.path.join(jaato_dir, rel)
    try:
        fd = os.open(full, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("user tier: %s skipped (%s)", full, exc)
        return None
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        reason = _skip_reason(st, owner)
        if reason:
            logger.warning("user tier: %s skipped (%s)", full, reason)
            return None
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        logger.warning("user tier: %s skipped (larger than %d bytes)",
                       full, MAX_FILE_BYTES)
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("user tier: %s skipped (not UTF-8)", full)
        return None


def _skip_reason(st: os.stat_result, owner: int) -> Optional[str]:
    if not stat.S_ISREG(st.st_mode):
        return "not a regular file"
    if st.st_uid != owner:
        return f"owned by uid {st.st_uid}, not by the owner of ~/.jaato ({owner})"
    return None


# ----------------------------------------------------------------------
# Runner side
# ----------------------------------------------------------------------

def install(
    snapshot: Optional[Dict[str, str]], temp_root: str, session_id: str,
) -> Optional[Path]:
    """Install *snapshot* under *temp_root*; ``None`` reverts to disk reads.

    The directory is named for the session, because *temp_root* is
    shared on an unconfined runner (``/tmp``) and per workspace under a
    private ``/tmp`` (#1381).  Removes the previous installation (a pool
    slot serves sessions in turn).  Returns the snapshot directory, or
    ``None``.
    """
    global _installed_root
    if _installed_root is not None:
        shutil.rmtree(_installed_root, ignore_errors=True)
    _installed_root = None
    if snapshot is None:
        return None
    root = Path(temp_root) / f"{SNAPSHOT_DIRNAME}-{session_id}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(mode=0o700, parents=True)
    resolved = root.resolve()
    for rel, text in snapshot.items():
        target = (root / rel).resolve()
        if resolved not in target.parents or is_credential(target.name):
            logger.warning("user tier: snapshot entry %r refused", rel)
            continue
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _installed_root = root
    return root


def installed_root() -> Optional[Path]:
    """The installed snapshot directory, or ``None`` (disk reads)."""
    return _installed_root


def home_path(rel: str) -> Path:
    """Where a reader finds ``~/<rel>`` for a :data:`HOME_SHIPPED_DIRS` path."""
    if _installed_root is not None:
        return _installed_root / HOME_PREFIX.rstrip("/") / rel
    return Path.home() / rel


def path(rel: str) -> Path:
    """Where a reader finds ``~/.jaato/<rel>``.

    The snapshot's copy when one is installed (absent from the snapshot
    means the daemon found no such file), else the real file on disk.
    """
    if _installed_root is not None:
        return _installed_root / rel
    return Path.home() / ".jaato" / rel
