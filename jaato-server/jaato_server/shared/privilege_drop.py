"""Dropping a runner from the daemon's uid to a session's user (#1168).

A root daemon forks and execs its per-session runner under uid 0, so
every file the agent writes into a workspace is root-owned — and a umask
(#1168 step 2) cannot fix a file created with an explicit mode
(``curated.jsonl`` is ``0600``) or one written by ``tempfile.mkstemp`` +
``os.replace``.  Only running the runner AS the workspace's user fixes
ownership at the source.  This module is the one place that performs the
drop; the daemon side (``server/runner_user.py``) decides WHO.

Stdlib only, with no jaato import, because two very different callers use
it:

* the cold-spawn path, in the forked child between ``fork()`` and
  ``exec()`` of a threaded daemon — where an import could block on a lock
  another thread held at fork time, and where NSS lookups (``initgroups``,
  ``getpwnam``) are not safe for the same reason.  So a :class:`RunnerUser`
  carries its supplementary group list already resolved, and
  :func:`drop_to` calls nothing but ``setgroups`` / ``setresgid`` /
  ``setresuid``;
* the pool-slot path, ``runner/session.py``'s bootstrap step 1b3, which
  must stay importable before plugin discovery.

ORDER, and why (measured on an enforcing host, #1168 "Order A"): the drop
happens BEFORE the runner's own ``aa_change_profile``.  A transition from
``unconfined`` needs no capability and no ``change_profile`` rule, so the
confined runner keeps zero capabilities and cannot ``setuid`` afterwards —
the drop is irreversible by the time the boundary exists.  The reverse
order needs ``capability setuid, setgid,`` in every template, leaving every
confined runner able to become any uid on the box.  Nothing here adds that
rule, and nothing may.

Within the drop: supplementary groups first (``setgroups`` needs
``CAP_SETGID``, which ``setresuid`` would take away), then the gid, then the
uid.  ``setresuid`` rather than ``setuid`` so the saved-set uid is replaced
too — a saved uid of 0 is a way back.  :func:`drop_to` then PROVES the way
back is gone by attempting ``setuid(0)`` and requiring it to fail.

CPython's ``os.setresuid`` goes through glibc, which applies a credential
change to every thread of the process (the ``setxid`` broadcast), so a pool
slot's existing worker threads follow the drop — unlike an AppArmor label,
which is per task (#1023).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, MutableMapping, Optional, Tuple


class PrivilegeDropError(RuntimeError):
    """The drop could not be performed, or did not stick.

    Never caught and ignored: a runner that meant to run as a user and is
    still root is the failure this whole mechanism exists to prevent.
    """


@dataclass(frozen=True)
class RunnerUser:
    """The OS account a session's runner runs as, resolved daemon-side.

    Every field is resolved BEFORE the fork, because the child may not do
    NSS lookups (see the module docstring).

    Attributes:
        uid: Target uid.  Never 0 — :func:`drop_to` refuses it, and the
            daemon never resolves to it (a root target means "no drop").
        gid: Target primary gid.
        groups: The supplementary groups to install, primary included
            (``os.getgrouplist``).  Empty for a uid with no passwd entry:
            the daemon's own groups are cleared, never inherited.
        username: The passwd name, or ``None`` when the uid has none.
        home: The passwd home directory, or ``None``.
        source: Which policy produced this (``peer`` / ``workspace-owner``),
            for the log lines that say what happened.
    """

    uid: int
    gid: int
    groups: Tuple[int, ...] = field(default_factory=tuple)
    username: Optional[str] = None
    home: Optional[str] = None
    source: str = ""

    def describe(self) -> str:
        """``name(uid=N,gid=M)`` for log lines."""
        who = self.username or f"uid:{self.uid}"
        return f"{who}(uid={self.uid},gid={self.gid})"

    def to_dict(self) -> Dict[str, Any]:
        """The wire form carried on ``SessionInitEnvelope.runner_user``."""
        return {
            "uid": self.uid,
            "gid": self.gid,
            "groups": list(self.groups),
            "username": self.username,
            "home": self.home,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Any) -> Optional["RunnerUser"]:
        """Parse the wire form, or ``None`` for an absent/empty one.

        Raises:
            PrivilegeDropError: a dict that is present and malformed.  A
                runner that was told to drop and cannot read to whom must
                refuse, not run as the daemon's uid.
        """
        if not d:
            return None
        try:
            return cls(
                uid=int(d["uid"]),
                gid=int(d["gid"]),
                groups=tuple(int(g) for g in (d.get("groups") or ())),
                username=d.get("username") or None,
                home=d.get("home") or None,
                source=str(d.get("source") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PrivilegeDropError(
                f"malformed runner_user {d!r}: {exc}") from exc


def already_running_as(user: RunnerUser) -> bool:
    """True when real, effective and saved uid are all *user*'s already."""
    return os.getresuid() == (user.uid,) * 3 and os.getresgid()[0] == user.gid


def drop_to(user: RunnerUser) -> bool:
    """Become *user*, irreversibly.  Returns whether anything changed.

    A no-op returning ``False`` when the process already runs as *user*
    (a cold-spawned runner reaching bootstrap, a pool slot serving its
    next session of the same uid).

    Raises:
        PrivilegeDropError: *user* is uid 0; the process is not root and
            is not already *user* (a slot reused across uids — the key
            must prevent that); a syscall failed; or ``setuid(0)``
            SUCCEEDED afterwards, meaning the drop did not stick.
    """
    if user.uid == 0:
        raise PrivilegeDropError(
            "refusing to 'drop' to uid 0: a root target means the runner "
            "keeps the daemon's uid, and is decided daemon-side")
    if already_running_as(user):
        return False
    if os.geteuid() != 0:
        raise PrivilegeDropError(
            f"cannot become {user.describe()}: this process runs as uid "
            f"{os.geteuid()}, not root, and is not already that user")
    try:
        os.setgroups(list(user.groups))
        os.setresgid(user.gid, user.gid, user.gid)
        os.setresuid(user.uid, user.uid, user.uid)
    except OSError as exc:
        raise PrivilegeDropError(
            f"could not become {user.describe()}: {exc}") from exc
    _prove_irreversible(user)
    return True


def _prove_irreversible(user: RunnerUser) -> None:
    """Require the way back to root to be closed.

    ``setresuid`` replaced the saved uid, so ``setuid(0)`` must fail with
    EPERM.  If it succeeds the drop is not what it claims to be, and the
    process is root again — so it is treated as a failure, not undone.
    """
    try:
        os.setuid(0)
    except PermissionError:
        pass
    else:
        raise PrivilegeDropError(
            f"dropped to {user.describe()} but setuid(0) still succeeds; "
            "refusing to continue")
    if os.getresuid() != (user.uid,) * 3:
        raise PrivilegeDropError(
            f"after the drop getresuid() is {os.getresuid()}, "
            f"expected {user.uid} x3")


def apply_user_env(user: RunnerUser, environ: MutableMapping[str, str]) -> None:
    """Point ``HOME`` / ``USER`` / ``LOGNAME`` at *user* in *environ*.

    The runner process's own variables.  A subprocess the model drives
    still gets the workspace HOME where #1225 applies it, because the
    plugins overwrite ``HOME`` in the env they build — so this only decides
    what the runner itself, and a subprocess outside #1225, sees.
    A uid with no passwd entry gets nothing: inventing a home directory
    would be a guess.
    """
    if user.home:
        environ["HOME"] = user.home
    if user.username:
        environ["USER"] = user.username
        environ["LOGNAME"] = user.username
