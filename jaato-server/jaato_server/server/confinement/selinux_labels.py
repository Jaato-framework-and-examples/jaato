"""Labelling a workspace for the SELinux backend (design §6).

The daemon labels the tree once per workspace, before the runner spawns:

1. refuse a root that must never be relabelled wholesale (``/``, ``/usr``,
   ``/etc``, ``/home``, a home directory itself);
2. create the authored directories that do not exist yet, so a runner
   creating one later cannot give it the writable workspace type;
3. walk the tree with ``lsetfilecon`` (never following a symlink, never
   crossing into another filesystem), giving each entry its type at the
   workspace's level.

Which entries are authored is not decided here: it is
``jaato_sdk.scaffold.gitignore.AUTHORED``'s ``confined=True`` set, the same
declaration the AppArmor template's write-denies are checked against, so
the two LSMs cannot disagree about what a session may not rewrite.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

from jaato_sdk.scaffold.gitignore import AUTHORED

WORKSPACE_TYPE = "jaato_workspace_t"
MANAGED_TYPE = "jaato_managed_ws_t"
AUTHORED_TYPE = "jaato_authored_t"
CLAIMS_TYPE = "jaato_claims_t"
TMP_TYPE = "jaato_tmp_t"

#: The SELinux user of files the daemon labels; restorecon uses it too.
FILE_USER = "system_u"

#: ``.jaato/references-claims/``: the runner writes claims, a child may not
#: (template v43).
CLAIMS_DIR = "references-claims"


def authored_entries() -> Tuple[Tuple[str, bool], ...]:
    """``(name, is_dir)`` for each authored entry under ``.jaato/``."""
    return tuple((e.path.rstrip("/"), e.path.endswith("/"))
                 for e in AUTHORED if e.confined)


def file_context(ftype: str, level: str) -> str:
    return f"{FILE_USER}:object_r:{ftype}:{level}"


class UnsafeWorkspace(ValueError):
    """The workspace root is one the daemon refuses to relabel."""


def check_root(workspace: str, *, home: Optional[str] = None) -> None:
    """Raise :class:`UnsafeWorkspace` for a root that must not be labelled.

    *workspace* is a realpath.  Relabelling ``/home`` or a whole home
    directory is never intended (design §6 step 1).
    """
    home = os.path.realpath(home or os.path.expanduser("~"))
    forbidden = {"/", "/usr", "/etc", "/home", "/root", "/var", "/tmp", home}
    if workspace in forbidden:
        raise UnsafeWorkspace(f"refusing to relabel {workspace!r} wholesale")
    if os.path.dirname(workspace) == "/home":
        raise UnsafeWorkspace(f"refusing to relabel the home directory {workspace!r}")


@dataclass(frozen=True)
class Plan:
    """How one workspace is labelled."""

    workspace: str
    level: str
    managed: bool
    private_tmp_dir: Optional[str] = None

    @property
    def base_type(self) -> str:
        return MANAGED_TYPE if self.managed else WORKSPACE_TYPE

    def type_for(self, path: str) -> str:
        """The type *path* (inside the workspace) is given."""
        rel = os.path.relpath(path, self.workspace)
        parts = rel.split(os.sep)
        if self.private_tmp_dir and (
                path == self.private_tmp_dir
                or path.startswith(self.private_tmp_dir + os.sep)):
            return TMP_TYPE
        if len(parts) >= 2 and parts[0] == ".jaato":
            if parts[1] == CLAIMS_DIR:
                return CLAIMS_TYPE
            for name, is_dir in authored_entries():
                if parts[1] == name and (is_dir or len(parts) == 2):
                    return AUTHORED_TYPE
        return self.base_type


def precreate_authored_dirs(workspace: str) -> None:
    """Create the authored directories that are missing (design §6 step 2)."""
    dot = os.path.join(workspace, ".jaato")
    for name, is_dir in authored_entries():
        if is_dir:
            os.makedirs(os.path.join(dot, name), exist_ok=True)
    os.makedirs(os.path.join(dot, CLAIMS_DIR), exist_ok=True)


def walk(root: str) -> Iterator[str]:
    """Every entry under *root*, *root* included, on *root*'s filesystem.

    Symlinks are yielded (their own label is set) and never followed; a
    directory on another device (a bind mount) is yielded and not entered.
    """
    root_dev = os.lstat(root).st_dev
    yield root
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as it:
            for entry in it:
                yield entry.path
                st = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(st.st_mode) and st.st_dev == root_dev:
                    stack.append(entry.path)


Setter = Callable[[str, str], None]


def apply(plan: Plan, set_context: Setter) -> int:
    """Label every entry of *plan*'s workspace; returns how many.

    *set_context* is ``lsetfilecon``: it raises ``OSError`` on failure,
    which ends the walk and propagates, so a partially labelled tree is
    never reported as labelled.
    """
    count = 0
    for path in walk(plan.workspace):
        set_context(path, file_context(plan.type_for(path), plan.level))
        count += 1
    return count
