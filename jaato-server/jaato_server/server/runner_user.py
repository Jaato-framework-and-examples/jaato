"""Which OS account a session's runner runs as (#1168, step 3).

A root daemon ran every runner as root, so every file an agent wrote into
a workspace was root-owned.  ``--umask`` (step 2) makes those files
group-writable and cannot help with a file created with an explicit mode
(``curated.jsonl``, ``0600``) or one written by ``mkstemp`` + ``os.replace``
(always ``0600``); only running the runner AS the workspace's user fixes
ownership at the source.  Measured on the issue: an app owning its
workspace as uid 1001 could not delete its own ``.jaato/memories`` because
the runner had created it as root.

This module decides WHO; ``shared/privilege_drop.py`` performs the drop.

THE POLICY is a property of the daemon, not of a session — the same
reason ``--umask`` is host-scoped: which account a runner may become is an
operator's decision about the host, and a profile (which a session's
author writes) must not be able to pick a uid.

==================  ===================================================
``daemon``          the default.  No drop; byte-for-byte the behaviour
                    before this existed.
``peer``            the IPC connection's ``SO_PEERCRED`` uid — the account
                    that opened the socket, vouched for by the kernel.
``workspace-owner`` the uid owning the session's workspace directory.
==================  ===================================================

Never fail open to a guessed uid.  Every case where the policy cannot be
honoured keeps the daemon's uid and says so, once per reason:

* a connection with no OS principal (WS: ``get_client_peer`` is ``None``
  by design, #1074) under ``peer``;
* a non-root daemon under any non-``daemon`` policy (it cannot drop);
* a target that IS root (a peer connecting as root, a root-owned
  workspace such as one the daemon provisioned itself) — there is nothing
  to drop to, and ``drop_to`` refuses uid 0 anyway;
* a target equal to the daemon's own uid.

One case REFUSES the session instead: a target that cannot read the
daemon's interpreter or the jaato packages.  Keeping root there would hand
that user a root runner, and dropping would produce a runner that dies on
its first lazy import with an ``ImportError`` several layers from the
cause — so :class:`RunnerUserRefused` names the path instead.

WHAT THE DAEMON STILL OWNS.  Session records, the workspace index and the
daemon's own log are written by the daemon process and stay the daemon's.
What the daemon creates FOR the runner is handed to the target uid before
the spawn (:func:`prepare_runner_owned_paths`) — the session tmpdir, the
private ``/tmp`` directory, the workspace HOME, ``.jaato/logs`` and the
session's own ``.jaato/sessions/<id>`` — but only when the daemon uid owns
it: nothing here takes a file away from another user, and nothing chowns
the user's existing tree.  A workspace already full of root-owned files
from before this existed needs ``chown -R <user> <ws>`` once.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from jaato_server.shared.peer_identity import PeerCredentials, path_reachable_by
from jaato_server.shared.privilege_drop import RunnerUser

logger = logging.getLogger(__name__)

#: The env var carrying the policy.  ``host``-scoped in
#: ``shared/env_scope.py``; the CLI twin ``--runner-uid-policy`` outranks it.
RUNNER_UID_POLICY_ENV_VAR = "JAATO_RUNNER_UID_POLICY"

POLICY_DAEMON = "daemon"
POLICY_PEER = "peer"
POLICY_WORKSPACE_OWNER = "workspace-owner"
POLICIES = (POLICY_DAEMON, POLICY_PEER, POLICY_WORKSPACE_OWNER)

#: What each policy runs a session's runner as.  Read by
#: ``jaato-scaffold explain runner-user``, so the topic cannot describe a
#: policy differently from the code that resolves it.
POLICY_TARGETS = {
    POLICY_DAEMON: "the daemon's own uid (no drop) -- the default",
    POLICY_PEER: "the uid of the OS account that opened the IPC socket "
                 "(SO_PEERCRED)",
    POLICY_WORKSPACE_OWNER: "the uid (and gid) owning the session's "
                            "workspace directory",
}

#: Why a policy may fail to name a user.  Each keeps the daemon's uid and
#: is announced once per daemon; :func:`_target_uid_gid` returns these.
REASON_NO_PEER = (
    "this connection carries no OS principal (a WebSocket connection -- "
    "#1074 -- or a platform without SO_PEERCRED)")
REASON_NO_WORKSPACE = "the session has no workspace"
REASON_WORKSPACE_UNSTATABLE = "its workspace cannot be stat'd"

_policy_lock = threading.Lock()
_policy: str = POLICY_DAEMON
_announced: set = set()


class RunnerUserRefused(RuntimeError):
    """The session's runner cannot run as the user the policy names.

    Raised from the spawn path and turned into a refused session by both
    transports — never into the in-process fallback, which would run the
    model's tools in the ROOT daemon process.
    """


# ------------------------------------------------------------------ policy

def parse_policy(raw: Optional[str]) -> Optional[str]:
    """``raw`` normalised to one of :data:`POLICIES`, or ``None``.

    ``None`` for "not configured" (``None`` / blank) AND for a malformed
    value, which is logged at ERROR.  A typo must not become a drop to an
    account nobody chose, nor silently a different policy: it keeps the
    default, visibly — the posture ``parse_umask`` takes.
    """
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower().replace("_", "-")
    if value in POLICIES:
        return value
    logger.error(
        "runner uid policy %r is not one of %s — ignoring it; runners keep "
        "the daemon's uid.", raw, ", ".join(POLICIES),
    )
    return None


def resolve_policy(explicit: Optional[str] = None) -> str:
    """The flag, else the env var, else :data:`POLICY_DAEMON`.

    A malformed flag does NOT fall through to the env var, for the reason
    ``process_posture.resolve_umask`` gives: the operator named a value.
    """
    if explicit is not None:
        return parse_policy(explicit) or POLICY_DAEMON
    return parse_policy(os.environ.get("JAATO_RUNNER_UID_POLICY")) or POLICY_DAEMON  # env: which uid a root daemon runs each session's runner as -- daemon (default), peer (the IPC SO_PEERCRED uid) or workspace-owner (#1168)


def set_policy(policy: str) -> None:
    """Install the process-wide policy.  Called once, at daemon start."""
    global _policy
    if policy not in POLICIES:
        raise ValueError(f"unknown runner uid policy {policy!r}")
    with _policy_lock:
        _policy = policy


def current_policy() -> str:
    """The policy :func:`set_policy` installed (default ``daemon``)."""
    return _policy


def _announce_once(key: str, level: int, msg: str, *args: Any) -> None:
    """Log *msg* once per *key* per process — once per REASON, not per
    session, because one line per session is noise an operator filters
    out, which is the same outcome as silence."""
    with _policy_lock:
        if key in _announced:
            return
        _announced.add(key)
    logger.log(level, msg, *args)


# ------------------------------------------------------------- resolution

def announce_if_unprivileged(policy: str) -> bool:
    """True (and WARN once) when *policy* asks for a drop a non-root daemon
    cannot perform.  Called at startup and again per session, so the line
    is there whether or not a session ever arrives."""
    if policy == POLICY_DAEMON or _running_as_root():
        return False
    _announce_once(
        "non-root", logging.WARNING,
        "runner uid policy %r is set but this daemon is not root (euid "
        "%s): it cannot drop privileges, so every runner keeps the "
        "daemon's uid.", policy, os.geteuid(),
    )
    return True


def _running_as_root() -> bool:
    geteuid = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0


def _passwd_of(uid: int) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """``(name, home, primary gid)`` for *uid*, or ``None``s with no entry."""
    try:
        import pwd
        entry = pwd.getpwuid(uid)
    except (ImportError, KeyError):
        return None, None, None
    return entry.pw_name, entry.pw_dir, entry.pw_gid


def user_for_uid(uid: int, gid: Optional[int], source: str) -> RunnerUser:
    """Build a fully-resolved :class:`RunnerUser` daemon-side.

    Everything the forked child will need is looked up HERE, because the
    child may not do NSS lookups (``shared/privilege_drop.py``).  A uid with
    no passwd entry keeps the gid it was given and gets no supplementary
    groups — the daemon's own are cleared, never inherited.
    """
    name, home, pw_gid = _passwd_of(uid)
    primary = gid if gid is not None else (pw_gid if pw_gid is not None else uid)
    groups: Tuple[int, ...] = (primary,)
    getgrouplist = getattr(os, "getgrouplist", None)
    if name and getgrouplist is not None:
        try:
            groups = tuple(sorted(set(getgrouplist(name, primary))))
        except (KeyError, OSError):
            pass
    return RunnerUser(
        uid=uid, gid=primary, groups=groups, username=name, home=home,
        source=source,
    )


def _target_uid_gid(
    policy: str, peer: Optional[PeerCredentials], workspace_path: Optional[str],
) -> Tuple[Optional[int], Optional[int], str]:
    """``(uid, gid, why_not)`` the policy names, or ``(None, None, reason)``."""
    if policy == POLICY_PEER:
        if peer is None:
            return None, None, REASON_NO_PEER
        return peer.uid, peer.gid, ""
    if not workspace_path:
        return None, None, REASON_NO_WORKSPACE
    try:
        st = os.stat(workspace_path)
    except OSError as exc:
        return None, None, f"{REASON_WORKSPACE_UNSTATABLE} ({exc})"
    return st.st_uid, st.st_gid, ""


def resolve_runner_user(
    *,
    peer: Optional[PeerCredentials],
    workspace_path: Optional[str],
    policy: Optional[str] = None,
) -> Optional[RunnerUser]:
    """The account this session's runner should become, or ``None``.

    ``None`` means "keep the daemon's uid" — the default policy, and every
    case the policy cannot honour (see the module docstring), each of which
    is announced once.

    Raises:
        RunnerUserRefused: the target cannot read what a runner imports.
    """
    policy = policy or current_policy()
    if policy == POLICY_DAEMON or announce_if_unprivileged(policy):
        return None
    uid, gid, why_not = _target_uid_gid(policy, peer, workspace_path)
    if uid is None:
        _announce_once(
            f"{policy}:{why_not}", logging.WARNING,
            "runner uid policy %r cannot name a user for a session because "
            "%s; such sessions keep the daemon's uid (root).",
            policy, why_not,
        )
        return None
    if uid == 0 or uid == os.getuid():
        _announce_once(
            f"{policy}:root-target", logging.INFO,
            "runner uid policy %r named uid %d for a session; that is the "
            "daemon's own uid, so there is nothing to drop to and the "
            "runner keeps it.", policy, uid,
        )
        return None
    user = user_for_uid(uid, gid, policy)
    refuse_if_unreachable(user, workspace_path)
    announce_daemon_credentials_hidden()
    return user


# ----------------------------------------------------- reachability check

def runner_import_paths() -> List[str]:
    """Everything a runner must be able to read to start and keep importing.

    The interpreter, the jaato packages (lazily imported for as long as the
    session runs) and the standard library.  Read once per call; cheap.
    """
    import sysconfig

    import jaato_sdk
    import jaato_server

    paths = [os.path.realpath(sys.executable)]
    for pkg in (jaato_server, jaato_sdk):
        paths.extend(os.path.realpath(p) for p in getattr(pkg, "__path__", []))
    for key in ("stdlib", "purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value and os.path.isdir(value):
            paths.append(os.path.realpath(value))
    return list(dict.fromkeys(paths))


def unreachable_for(user: RunnerUser, paths: Iterable[str]) -> List[str]:
    """The paths in *paths* the target could not demonstrably read.

    ``path_reachable_by`` answers ``None`` when it cannot tell; that counts
    as unreachable here, because the cost of a wrong "yes" is a runner that
    crashes on import and the cost of a wrong "no" is a refusal naming the
    path.
    """
    peer = PeerCredentials(
        uid=user.uid, gid=user.gid, username=user.username,
    )
    return [p for p in paths if path_reachable_by(p, peer) is not True]


def refuse_if_unreachable(user: RunnerUser, workspace_path: Optional[str]) -> None:
    """Raise :class:`RunnerUserRefused` unless *user* can run a runner.

    Checks the import paths and the workspace itself.  ACLs are not read
    (``path_reachable_by`` is stricter than the kernel, never looser), so a
    deployment that grants access only by ACL is refused — visibly.
    """
    paths = runner_import_paths()
    if workspace_path:
        paths.append(workspace_path)
    missing = unreachable_for(user, paths)
    if missing:
        raise RunnerUserRefused(
            f"the runner uid policy {user.source!r} names "
            f"{user.describe()}, who cannot read {', '.join(missing)}; a "
            "runner dropped to that user would fail on import.  Install "
            "jaato where that user can read it (a world-readable venv), "
            "or use --runner-uid-policy daemon (#1168)")


def announce_daemon_credentials_hidden() -> None:
    """Say, once, that a dropped runner does not see the daemon's ``~/.jaato``.

    A provider that finds no key in the session env falls back to a stored
    ``<provider>_auth.json`` under the config root, the workspace and
    ``~/.jaato``.  A dropped runner's ``~`` is the TARGET's home, and the
    daemon's ``~/.jaato`` is (rightly) not readable by it — so a deployment
    relying on the daemon's stored credentials sees sessions that cannot
    authenticate.  Nothing is copied to where the target could read it:
    the remedy is a key in the workspace ``.env`` / profile ``env:`` (which
    the daemon resolves and ships on the envelope), or the user's own
    ``~/.jaato``.
    """
    home = os.path.expanduser("~/.jaato")
    try:
        stored = sorted(p.name for p in Path(home).glob("*_auth.json"))
    except OSError:
        return
    if stored:
        _announce_once(
            "daemon-credentials", logging.WARNING,
            "runners dropped to another uid cannot read the daemon's "
            "stored credentials in %s (%s): a provider that falls back to "
            "them will find none.  Put the key in the workspace .env or the "
            "profile's env: map (resolved daemon-side), or store it in the "
            "user's own ~/.jaato (#1168).", home, ", ".join(stored),
        )


# ------------------------------------------------- daemon-created paths

def _chown_if_daemon_owned(path: str, user: RunnerUser) -> bool:
    """Hand *path* to *user* when the daemon uid owns it.  Never follows a
    link (``lchown``) and never takes a file from another user."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if st.st_uid != os.getuid() or st.st_uid == user.uid:
        return False
    os.lchown(path, user.uid, user.gid)
    return True


def _makedirs_owned(path: str, user: RunnerUser) -> None:
    """``makedirs`` handing each directory it CREATES to *user*.

    Existing ancestors are left as they are — they are the user's tree (or
    somebody else's) and not this function's to re-own.
    """
    missing: List[str] = []
    cursor = os.path.abspath(path)
    while not os.path.lexists(cursor):
        missing.append(cursor)
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        cursor = parent
    for directory in reversed(missing):
        os.mkdir(directory)
        os.lchown(directory, user.uid, user.gid)


def runner_owned_paths(
    *,
    session_id: str,
    workspace_path: Optional[str],
    session_tmp: Optional[str],
    private_tmp: Optional[str],
    workspace_home: Optional[str],
    log_path: Optional[str],
) -> Tuple[List[str], List[str]]:
    """``(directories, files)`` the daemon prepares for a dropped runner."""
    dirs = [p for p in (session_tmp, private_tmp, workspace_home) if p]
    files: List[str] = []
    if workspace_path:
        jaato = os.path.join(workspace_path, ".jaato")
        dirs.append(os.path.join(jaato, "logs"))
        dirs.append(os.path.join(jaato, "sessions", session_id))
    if log_path:
        files.append(log_path)
    return dirs, files


def prepare_runner_owned_paths(
    user: Optional[RunnerUser], dirs: Iterable[str], files: Iterable[str],
) -> None:
    """Create/hand over what the daemon makes FOR the runner.  Before spawn.

    Each directory is created if missing (every component it creates is
    handed to *user*) and handed over if the daemon uid owns it.  Each file
    is created empty (``0600``) if missing — the runner's log, which the
    forked child opens before the drop — and handed over likewise.

    Best-effort per path and audible: a path that cannot be prepared makes
    the runner fail where it touches it, with its own error.  ``None`` user
    is a no-op, so the call site needs no branch.
    """
    if user is None:
        return
    for path in dirs:
        try:
            _makedirs_owned(path, user)
            _chown_if_daemon_owned(path, user)
        except OSError as exc:
            logger.warning(
                "runner uid drop: could not prepare %s for %s (%s: %s)",
                path, user.describe(), type(exc).__name__, exc)
    for path in files:
        try:
            _makedirs_owned(os.path.dirname(path), user)
            if not os.path.lexists(path):
                os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600))
            _chown_if_daemon_owned(path, user)
        except OSError as exc:
            logger.warning(
                "runner uid drop: could not prepare %s for %s (%s: %s)",
                path, user.describe(), type(exc).__name__, exc)


def runner_uid_of(user: Optional[RunnerUser]) -> Optional[int]:
    """The uid for the pool's ``SlotKey``: ``None`` = the daemon's uid."""
    return user.uid if user is not None else None
