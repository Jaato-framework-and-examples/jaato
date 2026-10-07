"""Write a file under a directory without following a link out of it.

A workspace is model-writable: the agent's ``cli`` and ``file_edit`` tools
write there, so it can plant a symlink on any path, file or directory.  A
daemon that then writes into that workspace on someone else's behalf (a
staged upload, an application's managed file) must not carry the write
wherever the link points, which on a root daemon is anywhere on the host.
``Path.mkdir(parents=True)`` and ``Path.write_bytes`` both follow links.

This module is the one definition of how the daemon writes under such a
root, shared by ``workspace.app_write`` (:mod:`.workspace_app_write`) and
file staging (``websocket._write_staged_payload``, #1386):

* the root is resolved once with :func:`os.path.realpath`;
* the parent directory is walked one component at a time
  (:func:`contained_dir`).  An existing component must resolve inside the
  root and be a directory; a missing one is created with ``os.mkdir``
  inside a directory already proved contained, so no ``mkdir`` ever happens
  through a link that leaves the root.  Containment is judged before
  existence: a link pointing outside is refused whether or not its target
  exists, so the refusal is not an oracle for the host's filesystem;
* the destination itself is refused when it is a symlink, and written
  through a temp file plus :func:`os.replace` (:func:`atomic_write_bytes`),
  which replaces a link rather than following it;
* every directory created and the file written take the owner of the root
  (:mod:`jaato_server.shared.workspace_ownership`), so a root daemon writing into an account's
  workspace leaves files that account can use.

Stdlib only, no daemon state.
"""

from __future__ import annotations

import os
import secrets
from typing import Callable, Optional

from jaato_server.shared.workspace_ownership import inherit_owner


class PathLeavesRoot(ValueError):
    """The path is, or passes through, a link leaving the root (or a
    component that must be a directory is not one).  Nothing was written."""


def under(path: str, root: str) -> bool:
    """Whether the (already resolved) ``path`` is ``root`` or beneath it."""
    return path == root or path.startswith(root + os.sep)


def _default_dir_mode(_part: str) -> int:
    return 0o755


def contained_dir(
    root: str,
    rel_dir: str,
    create: bool,
    dir_mode: Callable[[str], int] = _default_dir_mode,
) -> Optional[str]:
    """The real directory ``root/rel_dir``, creating it when asked.

    ``root`` must already be resolved (:func:`os.path.realpath`).  Walks one
    component at a time and checks each existing one resolves inside
    ``root`` before descending.  ``dir_mode(part)`` gives the mode of a
    directory this call creates; each one created is handed to the owner
    of ``root``.  Returns ``None`` when the directory is
    absent and ``create`` is false; raises :class:`PathLeavesRoot` for a
    component that leaves the root or is not a directory.
    """
    current = root
    for part in [p for p in rel_dir.split("/") if p]:
        nxt = os.path.join(current, part)
        if os.path.lexists(nxt):
            real = os.path.realpath(nxt)
            if not under(real, root):
                raise PathLeavesRoot(f"{rel_dir} passes through a link leaving the workspace")
            if not os.path.isdir(real):
                raise PathLeavesRoot(f"{rel_dir}: {part} is not a directory")
            current = real
            continue
        if not create:
            return None
        os.mkdir(nxt, dir_mode(part))
        inherit_owner(nxt, root)
        current = nxt
    return current


def atomic_write_bytes(path: str, data: bytes, mode: Optional[int] = None) -> None:
    """Temp file in the same directory, then ``os.replace`` onto ``path``.

    ``os.replace`` over a symlink replaces the LINK, so a link at the
    destination is never followed.  With ``mode`` the file gets exactly
    that mode; without it the process umask decides, as it would for an
    ordinary ``open()``.
    """
    tmp = os.path.join(os.path.dirname(path), f".{secrets.token_hex(6)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if mode is None else mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_contained(root: str, rel_path: str, data: bytes, mode: Optional[int] = None) -> str:
    """Write ``data`` to ``root/rel_path``, creating parents, never through a link.

    ``rel_path`` is ``/``-separated and must already be free of ``..`` and
    absolute forms (the caller's own name check).  Raises
    :class:`PathLeavesRoot` when a parent leaves the root or the destination
    is a symlink, and ``OSError`` on an I/O failure.  Returns the real path
    written, which belongs to the owner of ``root``.
    """
    root = os.path.realpath(root)
    rel_dir, _, name = rel_path.rpartition("/")
    if not name:
        raise PathLeavesRoot(f"{rel_path!r} names no file")
    directory = contained_dir(root, rel_dir, create=True)
    dest = os.path.join(directory, name)
    if os.path.islink(dest):
        raise PathLeavesRoot(f"{rel_path} is a symlink; a staged write never goes through a link")
    atomic_write_bytes(dest, data, mode)
    inherit_owner(dest, root)
    return dest
