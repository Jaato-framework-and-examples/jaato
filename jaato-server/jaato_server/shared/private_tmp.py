"""A private ``/tmp`` per workspace (#1381, step 1).

Models are trained to write to ``/tmp`` and keep doing it whatever the
prompt says.  A confined session's AppArmor profile grants only its own
session tmpdir under ``/tmp`` (#1171), so every habitual ``/tmp/x`` cost a
refused call, and a kernel refusal is a bare ``EACCES`` the model tends to
misread.  This module does what systemd's ``PrivateTmp=`` does: before the
runner confines itself it unshares a mount namespace and bind-mounts the
workspace's own ``<ws>/.tmp`` over ``/tmp`` and ``/var/tmp``.  Every process
the session starts inherits the namespace, so ``/tmp`` just works for all of
them, and the host's ``/tmp`` (the daemon's socket, other sessions' temp
directories) is not visible from the session at all.

Where each piece runs
---------------------
- The **daemon** decides whether a boundary gets a private ``/tmp``
  (:func:`resolve_private_tmp`), renders the matching ``/tmp`` grants into
  the AppArmor profile, and creates ``<ws>/.tmp`` before the runner spawns
  (:func:`ensure_private_tmp_dir`).
- The **runner** enters the namespace BEFORE confinement
  (:func:`enter_private_tmp`): in the forked child before ``exec`` on a cold
  spawn, and in ``session.bootstrap`` before step 1c on a pool slot.  Every
  profile body denies ``capability sys_admin`` and ``mount``, so a confined
  process cannot do it; a reused slot that already sits in the namespace
  takes the idempotent path (:func:`private_tmp_in_effect`).

Fail closed
-----------
The ``/tmp/**`` grants are rendered ONLY for a boundary that asked for a
private ``/tmp``, and such a boundary's runner refuses to start
(:class:`PrivateTmpError`, turned into a bootstrap failure) when it cannot
set up the namespace.  It never runs under a ``/tmp/** rw`` grant against
the host's real ``/tmp``.

Stdlib-only with no jaato imports: the runner imports it before plugin
discovery, the shape ``shared/apparmor_label.py`` has.
"""

import ctypes
import logging
import os
import sys
import threading
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

#: The workspace-relative directory bound over ``/tmp``.  A dotted sibling
#: of ``.home`` (#1225): not under ``.jaato/`` (framework assets only) and
#: not a plain ``tmp/`` (Rails and many repos own one).
DEFAULT_PRIVATE_TMP_DIR = ".tmp"

#: The mount points the private directory is bound over, in order.
#: ``/tmp`` is required; ``/var/tmp`` is bound when it exists.
PRIVATE_TMP_TARGETS = ("/tmp", "/var/tmp")

#: Given a fresh tmpfs in the same namespace, when it exists.  POSIX
#: semaphores and shared memory live here (Python multiprocessing's
#: ``SemLock``), and the host's is shared by every session on the host,
#: so the profile grants it only where this mount is made.
PRIVATE_SHM_TARGET = "/dev/shm"

#: The author-facing knob: ``plugin_configs.cli.private_tmp`` (a bool),
#: beside ``workspace_home`` / ``workspace_venv`` (#1225 / #1274).
CONFIG_SURFACE = "cli"
CONFIG_KEY = "private_tmp"

# <sched.h> / <sys/mount.h>
_CLONE_NEWNS = 0x00020000
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_BIND = 4096
_MS_REC = 16384
_MS_PRIVATE = 1 << 18

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})

_warned_unavailable = False
_warn_lock = threading.Lock()

# The directory this PROCESS bound over /tmp, once it has (runner side).
_active_dir: Optional[str] = None
# ``(st_dev, st_ino)`` of the tmpfs this process mounted on /dev/shm.
_shm_identity: Optional[Tuple[int, int]] = None


class PrivateTmpError(RuntimeError):
    """The private ``/tmp`` could not be set up; the runner must not start."""


def private_tmp_dir_for(workspace_path: str) -> str:
    """``<ws>/.tmp`` for *workspace_path*, symlinks resolved."""
    return os.path.join(os.path.realpath(workspace_path), DEFAULT_PRIVATE_TMP_DIR)


def _coerce_bool(raw: Any) -> Optional[bool]:
    """A config value as a bool, or ``None`` for one that is not a bool."""
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def private_tmp_requested(
    cli_config: Optional[dict],
    workspace_path: Optional[str],
    managed_workspace_root: Optional[str],
) -> bool:
    """Whether this session asks for a private ``/tmp``.

    - An explicit ``plugin_configs.cli.private_tmp`` wins, either way.  A
      value that is not a bool is refused with a WARNING and read as OFF:
      the feature changes what ``/tmp`` IS, and a typo must not silently
      turn it on for a user's own checkout.
    - Otherwise it is on iff the workspace is one the daemon manages under
      ``managed_workspace_root`` (the #1225 / #1274 rule).  A user's own
      checkout (IPC, user-CWD) is opt-in: a private ``/tmp`` would surprise
      a user who asks the agent to "look at /tmp/log.txt".
    """
    if cli_config is not None and CONFIG_KEY in cli_config:
        value = _coerce_bool(cli_config.get(CONFIG_KEY))
        if value is None:
            logger.warning(
                "private_tmp: plugin_configs.cli.private_tmp=%r is not a "
                "boolean; treating it as off (#1381)",
                cli_config.get(CONFIG_KEY),
            )
            return False
        return value
    return _is_under(workspace_path, managed_workspace_root)


def _is_under(path: Optional[str], root: Optional[str]) -> bool:
    """True iff *path* is *root* or beneath it (symlinks resolved)."""
    if not path or not root:
        return False
    ws = os.path.realpath(path)
    base = os.path.realpath(root)
    return ws == base or ws.startswith(base + os.sep)


def under_a_target(path: str) -> bool:
    """True iff *path* lies under ``/tmp`` or ``/var/tmp``.

    Such a workspace cannot have a private ``/tmp``: binding over ``/tmp``
    would hide the workspace itself from the runner.
    """
    real = os.path.realpath(path)
    return any(real == t or real.startswith(t + os.sep)
               for t in (os.path.realpath(x) for x in PRIVATE_TMP_TARGETS))


def unavailable_reason() -> Optional[str]:
    """Why this daemon cannot give a runner a private ``/tmp``, or ``None``.

    A mount namespace needs ``CAP_SYS_ADMIN``.  A non-root daemon would need
    an unprivileged user namespace, which Ubuntu 24.04+ restricts by default
    (``kernel.apparmor_restrict_unprivileged_userns``); step 1 does not use
    one, so a non-root daemon is always "unavailable".
    """
    if not sys.platform.startswith("linux"):
        return "not a Linux host (mount namespaces are Linux-only)"
    if os.geteuid() != 0:
        return (
            "the daemon is not root, so the runner has no CAP_SYS_ADMIN to "
            "unshare a mount namespace (unprivileged user namespaces are "
            "not used)"
        )
    return None


def _workspace_reason(workspace_path: str) -> Optional[str]:
    if under_a_target(workspace_path):
        return (f"the workspace {workspace_path} lies under /tmp or /var/tmp, "
                "which a private /tmp would hide from the runner")
    return None


def _warn_unavailable_once(reason: str) -> None:
    global _warned_unavailable
    with _warn_lock:
        if _warned_unavailable:
            return
        _warned_unavailable = True
    logger.warning(
        "private_tmp: a session asked for a private /tmp but %s; sessions "
        "keep the per-session $TMPDIR under /tmp and other /tmp paths stay "
        "refused (#1381)",
        reason,
    )


def resolve_private_tmp(
    profile: Any,
    workspace_path: Optional[str],
    managed_workspace_root: Optional[str],
) -> Optional[str]:
    """The ``<ws>/.tmp`` this CONFINED session binds over ``/tmp``, or ``None``.

    Called daemon-side, only on the confined provisioning path: the value
    decides the rendered profile (so the confinement id) and what the
    runner does before confining.  Reads the explicit knob off *profile*
    (``None`` for a profile-less session).  Announces at WARNING, once per
    daemon, when a session asked and this host cannot.
    """
    if not workspace_path:
        return None
    configs = getattr(profile, "plugin_configs", None) or {}
    cli_cfg = configs.get(CONFIG_SURFACE) if isinstance(configs, dict) else None
    cli_cfg = cli_cfg if isinstance(cli_cfg, dict) else None
    if not private_tmp_requested(cli_cfg, workspace_path, managed_workspace_root):
        return None
    reason = unavailable_reason() or _workspace_reason(workspace_path)
    if reason:
        _warn_unavailable_once(reason)
        return None
    return private_tmp_dir_for(workspace_path)


def ensure_private_tmp_dir(path: Optional[str]) -> None:
    """Create ``<ws>/.tmp`` and its ``*`` gitignore, daemon-side.

    No-op for ``None``.  Created before the runner spawns, as the session
    tmpdir (#1171) and the workspace home (#1225) are.  A failure is only
    logged here: the runner's :func:`enter_private_tmp` then refuses to
    start, naming the missing directory, which is the fail-closed half.
    """
    if not path:
        return
    try:
        os.makedirs(path, exist_ok=True)
        gitignore = os.path.join(path, ".gitignore")
        if not os.path.exists(gitignore):
            with open(gitignore, "w", encoding="utf-8") as handle:
                handle.write("*\n")
    except OSError as exc:
        logger.warning(
            "private_tmp: could not prepare %s (%s: %s); the runner will "
            "refuse to start (#1381)",
            path, type(exc).__name__, exc,
        )


def _identity(path: str) -> Optional[Tuple[int, int]]:
    """``(st_dev, st_ino)`` of *path*, or ``None`` when it cannot be read."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _targets_are(identity: Optional[Tuple[int, int]]) -> bool:
    """True iff ``/tmp`` (and ``/var/tmp`` when it exists) is *identity*."""
    if identity is None or _identity("/tmp") != identity:
        return False
    return not os.path.isdir("/var/tmp") or _identity("/var/tmp") == identity


def private_tmp_in_effect(tmp_dir: str) -> bool:
    """True iff ``/tmp`` (and ``/var/tmp`` when it exists) IS *tmp_dir* here.

    Compares device and inode, so it answers for this task's mount
    namespace and cannot be fooled by a path that merely looks right.
    """
    return _targets_are(_identity(tmp_dir))


def _libc() -> Any:
    """The process's own symbol table, which carries libc's ``unshare`` /
    ``mount``.  Deliberately not ``ctypes.util.find_library``: that spawns
    helper processes, and this runs in a forked child before ``exec``."""
    return ctypes.CDLL(None, use_errno=True)


def _check(rc: int, what: str) -> None:
    if rc != 0:
        err = ctypes.get_errno()
        raise PrivateTmpError(f"{what} failed: {os.strerror(err)} (errno {err})")


def _bind_targets(libc: Any, tmp_dir: str) -> None:
    """Bind *tmp_dir* over each target, through an fd opened up front.

    The source is named as ``/proc/self/fd/<n>`` rather than by path: once
    ``/tmp`` is bound, a workspace that itself lives under ``/tmp`` no
    longer resolves by name, and the ``/var/tmp`` bind would fail (found by
    running it).  The fd pins the directory the daemon created.
    """
    fd = os.open(tmp_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        source = f"/proc/self/fd/{fd}".encode()
        for target in PRIVATE_TMP_TARGETS:
            if target != "/tmp" and not os.path.isdir(target):
                continue
            _check(
                libc.mount(source, target.encode(), None, _MS_BIND, None),
                f"bind-mounting {tmp_dir} over {target}",
            )
    finally:
        os.close(fd)


def _mount_private_shm(libc: Any) -> None:
    """Mount a fresh tmpfs on ``/dev/shm`` (inside the new namespace).

    Skipped where ``/dev/shm`` does not exist.  Verified by identity: a
    mount that left the host's ``/dev/shm`` in place is a refusal, because
    the profile grants ``/dev/shm/**`` to a boundary with a private
    ``/tmp`` and that grant must not reach the host's.
    """
    global _shm_identity
    if not os.path.isdir(PRIVATE_SHM_TARGET):
        return
    before = _identity(PRIVATE_SHM_TARGET)
    _check(
        libc.mount(b"tmpfs", PRIVATE_SHM_TARGET.encode(), b"tmpfs",
                   _MS_NOSUID | _MS_NODEV, b"mode=1777"),
        f"mounting a private tmpfs on {PRIVATE_SHM_TARGET}",
    )
    after = _identity(PRIVATE_SHM_TARGET)
    if after is None or after == before:
        raise PrivateTmpError(
            f"{PRIVATE_SHM_TARGET} is still the host's after the tmpfs mount; "
            "refusing to start")
    _shm_identity = after


def private_shm_in_effect() -> bool:
    """True iff ``/dev/shm`` is the tmpfs this process mounted."""
    return (_shm_identity is not None
            and _identity(PRIVATE_SHM_TARGET) == _shm_identity)


def enter_private_tmp(tmp_dir: Optional[str]) -> None:
    """Bind *tmp_dir* over ``/tmp`` and ``/var/tmp`` in a new mount namespace,
    and mount a fresh tmpfs on ``/dev/shm`` there.

    No-op for ``None``.  Idempotent: a process (a reused pool slot) that
    already sits in a namespace where ``/tmp`` is *tmp_dir* does nothing,
    which is the only thing a CONFINED runner can do.  Must run before
    confinement and on the thread that confines (the main thread): mount
    namespaces are per task, and the threads that later run tools are
    created by that thread (see the runner's bootstrap for why the #1023
    per-thread label check also covers the namespace).

    Raises:
        PrivateTmpError: the directory is missing, or ``unshare`` /
            ``mount`` failed, or the result could not be verified.  The
            caller must refuse to start the runner.
    """
    global _active_dir
    if not tmp_dir:
        return
    if not os.path.isdir(tmp_dir):
        raise PrivateTmpError(f"private /tmp directory {tmp_dir} does not exist")
    if under_a_target(tmp_dir):
        raise PrivateTmpError(
            f"private /tmp directory {tmp_dir} lies under /tmp or /var/tmp, "
            "which the bind would hide")
    if private_tmp_in_effect(tmp_dir):
        _active_dir = tmp_dir
        return
    identity = _identity(tmp_dir)
    libc = _libc()
    _check(libc.unshare(_CLONE_NEWNS), "unshare(CLONE_NEWNS)")
    _check(
        libc.mount(b"none", b"/", None, _MS_REC | _MS_PRIVATE, None),
        "making the mount tree private",
    )
    _bind_targets(libc, tmp_dir)
    if not _targets_are(identity):
        raise PrivateTmpError(
            f"/tmp is not {tmp_dir} after the bind mount; refusing to start")
    _mount_private_shm(libc)
    _active_dir = tmp_dir


def active_private_tmp() -> Optional[str]:
    """The directory this process bound over ``/tmp``, or ``None``."""
    return _active_dir


def describe() -> Tuple[bool, Optional[str]]:
    """``(private, backing_dir)`` for this process, for the runtime aspect.

    Re-checked against the filesystem rather than trusted from the flag,
    so a report never claims a private ``/tmp`` this task does not see.
    """
    active = _active_dir
    if active and private_tmp_in_effect(active):
        return True, active
    return False, None


def _reset_for_tests() -> None:
    """Clear the per-process state (tests only)."""
    global _active_dir, _warned_unavailable, _shm_identity
    _active_dir = None
    _shm_identity = None
    _warned_unavailable = False


def temp_files_hint() -> str:
    """One sentence on where scratch files go, for a containment refusal.

    A refusal for a ``/tmp`` path is where a model learns the temp rule, so
    it says what is true for THIS process rather than a generic rule.
    """
    if describe()[0]:
        return ("For scratch files, /tmp is this workspace's private temp "
                "directory and is writable.")
    return "For scratch files use $TMPDIR (or mktemp)."
