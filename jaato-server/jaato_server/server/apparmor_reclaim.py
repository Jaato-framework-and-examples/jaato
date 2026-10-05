"""Which ``jaato-ws-*`` profile files a daemon may reclaim at start (#1506).

A boundary profile is released when its last session ends and its last
pool slot dies (#1033, #1501), but a daemon that crashed, was killed, or
predates #1506 (whose IPC sessions never released at all) leaves its
profiles loaded in the kernel and on disk under ``/etc/apparmor.d/jaato``.
148 of them had built up on one host.  Nothing would ever remove them,
because nothing remembers them: the manager that provisioned each one is
gone.

So a starting daemon reconciles the directory.  The hard part is not the
unloading, it is deciding WHICH files are its to unload: the directory is
shared by every jaato daemon on the host, and removing a profile another
running daemon is about to confine a runner to would refuse that session.
Three facts decide it, each read here and nowhere else:

``OwnerLedger``
    ``~/.jaato/apparmor-owned.json``, written by the manager BEFORE it
    writes a profile file: ``{profile name: owner}``, where the owner is
    the writing daemon's ``pid:starttime`` (:func:`owner_key`).  A pid
    alone is reused; the start time of that pid (field 22 of
    ``/proc/<pid>/stat``) is not, so a dead owner whose pid was recycled
    still reads as dead.

the file's uid
    A file with no ledger entry was written by a daemon predating #1506
    (or one whose ``HOME`` differs).  Only one owned by this process's
    effective uid is a candidate; another account's daemon keeps its
    files.

``worn_profiles``
    Every task's ``attr/current`` under ``/proc``.  A profile any task is
    confined to is never unloaded, whoever owns it.

The verdict per file (:func:`reclaim_verdict`):

=========================================  ===========================
ledger owner is ANOTHER live daemon        keep (``owner-alive``)
ledger owner is THIS daemon                keep (``this-daemon``): live
                                           state another manager in this
                                           process tracks
not ledgered, file of another uid          keep (``foreign-uid``)
a task wears it                            keep (``worn``)
ledger owner dead, or not ledgered + our   reclaim
uid, and nobody wears it
=========================================  ===========================

Stdlib only, and no AppArmor manager import: the manager calls this.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Dict, Optional, Set

from jaato_server.server.confinement_id import PROFILE_PREFIX

#: The ledger's file name, beside ``apparmor-cache`` under ``~/.jaato``.
LEDGER_FILENAME = "apparmor-owned.json"

#: Filename separator the manager uses for an isolated sub-profile
#: (``jaato-ws-<parent>__sub_<id>``); the kernel name uses ``//``.
SUB_PROFILE_FILE_SEP = "__sub_"

KEEP_OWNER_ALIVE = "owner-alive"
KEEP_THIS_DAEMON = "this-daemon"
KEEP_FOREIGN_UID = "foreign-uid"
KEEP_WORN = "worn"
RECLAIM = "reclaim"


def _start_time(pid: int, proc_root: Path) -> Optional[str]:
    """Field 22 of ``/proc/<pid>/stat``, or ``None`` when unreadable.

    Read after the LAST ``)``: field 2 is the command name in parentheses
    and may itself contain spaces or parentheses.
    """
    try:
        raw = (proc_root / str(pid) / "stat").read_text()
    except OSError:
        return None
    tail = raw.rsplit(")", 1)[-1].split()
    # tail[0] is field 3 (state), so field 22 is tail[19].
    if len(tail) < 20:
        return None
    return tail[19]


def owner_key(pid: Optional[int] = None,
              proc_root: Path = Path("/proc")) -> str:
    """``pid:starttime`` naming one daemon PROCESS LIFETIME."""
    pid = os.getpid() if pid is None else pid
    return f"{pid}:{_start_time(pid, proc_root) or '?'}"


def owner_alive(key: str, proc_root: Path = Path("/proc")) -> bool:
    """Is the process lifetime *key* names still running?

    ``False`` for a malformed key, a pid with no ``/proc`` entry, and a
    pid whose start time differs (the pid was reused).  A key recorded
    with an unknown start time (``?``) is alive iff the pid exists: the
    weaker test, so in doubt a file is kept.
    """
    pid_s, _, start = key.partition(":")
    try:
        pid = int(pid_s)
    except ValueError:
        return False
    now = _start_time(pid, proc_root)
    if now is None:
        return False
    return start == "?" or now == start


def kernel_name_for_file(filename: str) -> str:
    """The kernel's name for profile file *filename* (sub-profiles use ``//``)."""
    return filename.replace(SUB_PROFILE_FILE_SEP, "//", 1)


def worn_profiles(proc_root: Path = Path("/proc")) -> Set[str]:
    """Every ``jaato-ws-*`` profile some task is confined to right now.

    Walks ``/proc/*/task/*/attr/current`` (per task, #1023: threads of one
    process can wear different profiles).  Each label is reduced to its
    profile name and every ``//`` prefix of it is added, so a task in
    ``jaato-ws-x//child`` counts as wearing ``jaato-ws-x`` too.
    Unreadable entries are skipped: a task we cannot see is a task we
    cannot prove wears nothing, and the caller keeps any profile whose
    owner it cannot rule out by the ledger and uid instead.
    """
    from jaato_server.shared.apparmor_label import profile_name_ignoring_mode

    worn: Set[str] = set()
    try:
        pids = [p for p in proc_root.iterdir() if p.name.isdigit()]
    except OSError:
        return worn
    for pid_dir in pids:
        try:
            tasks = list((pid_dir / "task").iterdir())
        except OSError:
            continue
        for task in tasks:
            try:
                raw = (task / "attr" / "current").read_text()
            except OSError:
                continue
            name = profile_name_ignoring_mode(raw)
            if not name.startswith(PROFILE_PREFIX):
                continue
            parts = name.split("//")
            for i in range(1, len(parts) + 1):
                worn.add("//".join(parts[:i]))
    return worn


class OwnerLedger:
    """``{profile name: owner key}``, kept under an exclusive ``flock``.

    Several daemons of one account share ``~/.jaato``, so every
    read-modify-write takes the lock on a sibling ``.lock`` file (the
    ledger itself is replaced by rename and would detach the lock).
    Best effort: an unreadable or corrupt ledger reads as empty, and a
    failed write is the caller's to log.  An empty ledger loses no file:
    an unledgered file is still judged by its uid and by whether a task
    wears it.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read(self) -> Dict[str, str]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        entries = data.get("profiles") if isinstance(data, dict) else None
        if not isinstance(entries, dict):
            return {}
        return {str(k): str(v) for k, v in entries.items()
                if isinstance(v, str)}

    def entries(self) -> Dict[str, str]:
        return self._update(None)

    def record(self, profile_name: str, owner: str) -> None:
        self._update(lambda e: e.__setitem__(profile_name, owner))

    def forget(self, profile_name: str) -> None:
        self._update(lambda e: e.pop(profile_name, None))

    def _update(self, mutate) -> Dict[str, str]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with open(lock_path, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            entries = self._read()
            if mutate is None:
                return entries
            mutate(entries)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({"version": 1, "profiles": entries},
                                      indent=1, sort_keys=True))
            os.replace(tmp, self.path)
            return entries


def reclaim_verdict(
    profile_path: Path,
    *,
    ledger_owner: Optional[str],
    this_owner: str,
    worn: Set[str],
    proc_root: Path = Path("/proc"),
    euid: Optional[int] = None,
) -> str:
    """Keep or reclaim one ``jaato-ws-*`` file; see the module table."""
    if ledger_owner is not None:
        if ledger_owner == this_owner:
            return KEEP_THIS_DAEMON
        if owner_alive(ledger_owner, proc_root):
            return KEEP_OWNER_ALIVE
    else:
        euid = os.geteuid() if euid is None else euid
        try:
            if profile_path.stat().st_uid != euid:
                return KEEP_FOREIGN_UID
        except OSError:
            return KEEP_FOREIGN_UID
    if kernel_name_for_file(profile_path.name) in worn:
        return KEEP_WORN
    return RECLAIM
