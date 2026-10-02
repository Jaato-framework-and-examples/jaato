"""Per-workspace MCS levels for the SELinux backend (design §5.1).

One category pair per WORKSPACE, not per session: sessions of one workspace
share a boundary, as they share an AppArmor profile (#1033).  The pair is
derived from ``sha256(realpath(workspace))`` so it is stable across daemon
restarts, and recorded in a daemon-owned table so two workspaces whose
digests land on the same pair get different ones.

The table lives where the daemon keeps its own state (``~/.jaato``, mode
0600), never in the workspace: the runner can write the workspace, and a
level it could rewrite would be a boundary it could choose.

The table also carries the labelling stamp (§6 step 4), so a later session
of the workspace can skip the relabel walk when nothing changed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: Categories c0..c1023, the range the targeted policy's MCS uses.
CATEGORIES = 1024

TABLE_FILENAME = "selinux_levels.json"


def default_table_path() -> Path:
    """``~/.jaato/selinux_levels.json`` of the DAEMON's account."""
    return Path.home() / ".jaato" / TABLE_FILENAME


def pair_for(seed: int) -> Tuple[int, int]:
    """Two distinct categories from *seed*, the lower first."""
    a = seed % CATEGORIES
    b = (seed // CATEGORIES) % (CATEGORIES - 1)
    if b >= a:
        b += 1
    return (a, b) if a < b else (b, a)


def level_string(pair: Tuple[int, int]) -> str:
    """``s0:cA,cB``."""
    return f"s0:c{pair[0]},c{pair[1]}"


@dataclass(frozen=True)
class LabelStamp:
    """What the last completed relabel of a workspace applied (§6 step 4).

    Attributes:
        policy_version: The module version the types belong to.
        managed: Whether the tree was labelled executable.
        private_tmp_dir: The ``<ws>/.tmp`` labelled as tmp, if any.
        files: How many entries the walk labelled (diagnostics only).
    """

    policy_version: int
    managed: bool
    private_tmp_dir: Optional[str]
    files: int

    def to_dict(self) -> Dict[str, Any]:
        return {"policy_version": self.policy_version, "managed": self.managed,
                "private_tmp_dir": self.private_tmp_dir, "files": self.files}

    @classmethod
    def from_dict(cls, d: Any) -> Optional["LabelStamp"]:
        if not isinstance(d, dict):
            return None
        try:
            return cls(int(d["policy_version"]), bool(d["managed"]),
                       d.get("private_tmp_dir") or None, int(d.get("files", 0)))
        except (KeyError, TypeError, ValueError):
            return None


class LevelTable:
    """The daemon's record of which level each workspace has.

    Thread-safe within the daemon, and ``flock``-ed on the file so a second
    daemon process sharing the account cannot hand one pair to two
    workspaces.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path is not None else default_table_path()
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def level_for(self, workspace: str) -> str:
        """The level of *workspace* (a realpath), allocating one if new."""
        with self._locked() as table:
            entry = table.get(workspace)
            if isinstance(entry, dict) and isinstance(entry.get("level"), str):
                return entry["level"]
            taken = {e.get("level") for e in table.values() if isinstance(e, dict)}
            seed = int.from_bytes(hashlib.sha256(workspace.encode("utf-8")).digest(), "big")
            level = level_string(pair_for(seed))
            while level in taken:
                seed += 1
                level = level_string(pair_for(seed))
            table[workspace] = {"level": level}
            self._write(table)
            return level

    def stamp(self, workspace: str) -> Optional[LabelStamp]:
        """The last relabel recorded for *workspace*, or ``None``."""
        with self._locked() as table:
            entry = table.get(workspace)
            return LabelStamp.from_dict(entry.get("stamp")) if isinstance(entry, dict) else None

    def record_stamp(self, workspace: str, stamp: LabelStamp) -> None:
        with self._locked() as table:
            entry = table.setdefault(workspace, {})
            entry["stamp"] = stamp.to_dict()
            self._write(table)

    def forget_stamp(self, workspace: str) -> None:
        """Drop the stamp, so the next provision walks the tree again."""
        with self._locked() as table:
            entry = table.get(workspace)
            if isinstance(entry, dict) and entry.pop("stamp", None) is not None:
                self._write(table)

    # -- file handling --------------------------------------------------

    class _Locked:
        def __init__(self, owner: "LevelTable") -> None:
            self._owner = owner
            self._fd: Optional[int] = None
            self.table: Dict[str, Any] = {}

        def __enter__(self) -> Dict[str, Any]:
            owner = self._owner
            owner._lock.acquire()
            try:
                owner._path.parent.mkdir(parents=True, exist_ok=True)
                self._fd = os.open(str(owner._path) + ".lock",
                                   os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(self._fd, fcntl.LOCK_EX)
                self.table = owner._read()
            except BaseException:
                self._release()
                raise
            return self.table

        def __exit__(self, *exc: Any) -> None:
            self._release()

        def _release(self) -> None:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self._owner._lock.release()

    def _locked(self) -> "LevelTable._Locked":
        return LevelTable._Locked(self)

    def _read(self) -> Dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"{self._path} does not hold a JSON object")
        return data

    def _write(self, table: Dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(prefix=".selinux_levels.", dir=str(self._path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(table, fh, indent=2, sort_keys=True)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
