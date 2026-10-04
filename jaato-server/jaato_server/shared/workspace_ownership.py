"""What the daemon writes inside a workspace belongs to the workspace's owner.

A root daemon creates files in workspaces on its own account: the
workspace itself (``workspace.create``), its ``.env``, a ``git clone``,
staged uploads, an application's managed files, the reference catalog.
Under ``--runner-uid-policy workspace-owner`` the session's runner runs as
the account that owns the workspace, so a file the daemon left root-owned
is one that runner cannot rewrite, delete or, when the daemon wrote it
``0600``, even read.

The rule is one sentence: **a path the daemon creates takes the owner of
the directory tree it was created in.**  The tree is named by the caller
(the workspace, or an application's workspace root for the workspace
directory itself), never inferred from the path.

* Only a root daemon hands anything over.  A daemon running as an ordinary
  user cannot ``chown``, and everything it writes is already its own.
* A tree owned by root is left alone, so a deployment whose workspaces
  are root-owned behaves exactly as before.
* Only paths the daemon owns are changed (its own uid, which is 0 whenever
  anything is handed over).  A file another account already owns is never
  taken from it, the rule
  :mod:`jaato_server.server.runner_user` applies to the directories it
  hands over.
* Links are never followed: ``os.lchown`` changes the link itself, and the
  tree walk does not descend through one.

Stdlib only, no daemon state.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def tree_owner(directory: str) -> Optional[Tuple[int, int]]:
    """``(uid, gid)`` that paths created under *directory* are handed to.

    ``None`` when nothing is handed over: the daemon is not root, or the
    directory is root-owned.  Raises ``OSError`` when *directory* cannot be
    stat'd: a writer that could not stat the tree it is writing into has
    not written anything there either.
    """
    # No POSIX ownership on Windows (no ``geteuid``): nothing to hand over.
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return None
    st = os.stat(directory)
    if st.st_uid == os.geteuid():
        return None
    return st.st_uid, st.st_gid


def _hand(path: str, owner: Tuple[int, int]) -> None:
    """``lchown`` *path* to *owner* when the daemon owns it."""
    if os.lstat(path).st_uid == os.geteuid():
        os.lchown(path, owner[0], owner[1])


def inherit_owner(path: str, tree: str) -> None:
    """Hand *path*, just created by the daemon, to the owner of *tree*."""
    owner = tree_owner(tree)
    if owner is not None:
        _hand(path, owner)


def inherit_owner_tree(top: str, tree: str) -> None:
    """Hand *top* and everything beneath it to the owner of *tree*.

    For writers that create a whole subtree in one call (``git``, ``tar``,
    ``shutil.copytree``).  Does not descend through a symlink.
    """
    owner = tree_owner(tree)
    if owner is None:
        return
    _hand(top, owner)
    if os.path.islink(top) or not os.path.isdir(top):
        return
    for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
        for name in dirnames + filenames:
            _hand(os.path.join(dirpath, name), owner)


def inherit_owner_files(directory: str, tree: str) -> None:
    """Hand the daemon-owned regular FILES directly in *directory* to *tree*'s owner.

    For writers the daemon calls but does not own the file layout of: a
    ``<provider>-auth`` command run daemon-side stores its credential under
    ``<workspace>/.jaato/`` with a name and a mode (often ``0600``) the
    plugin chooses.  Not recursive and files only: a daemon-owned
    directory there (``sessions/``) is the daemon's own record store.
    """
    owner = tree_owner(tree)
    if owner is None or not os.path.isdir(directory):
        return
    for entry in os.scandir(directory):
        if entry.is_file(follow_symlinks=False):
            _hand(entry.path, owner)
