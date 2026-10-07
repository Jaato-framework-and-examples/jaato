"""Daemon-level health checks (server 0.6.54+).

Currently exposes one check — inotify pressure — surfaced as a
periodic ``logger.warning()`` from the daemon main loop when usage
crosses 80% of the kernel's per-user limit.  Visibility-before-
failure: lets operators see EAGAIN-class failures coming before
``inotify_init1()`` returns ``EMFILE`` and sessions fail to spawn
with ``Errno 24 inotify instance limit reached``.

Complements server 0.6.45's IPC client-disconnect fix, which closed
the leak source.  Long-running daemons can still pile up legitimate
inotify usage (many simultaneous sessions, plugins that allocate
their own watchers, other processes by the same user); the soft-warn
catches those before the hard wall.

Linux-only — Linux is where inotify lives.  On non-Linux platforms
the check returns ``None`` and is a silent no-op.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# WARN once usage crosses this fraction of the kernel limit.  80%
# gives operators ~25 instances of headroom on the default-128
# limit — enough time to act (raise the kernel limit, or schedule
# a daemon restart) before sessions start failing.
_WARN_THRESHOLD = 0.80


def _read_inotify_limit() -> Optional[int]:
    """Read ``/proc/sys/fs/inotify/max_user_instances`` or return None.

    The kernel's per-user limit on simultaneously-open inotify
    instances.  Default 128 on most Linux distros.  Reading the
    pseudo-file is cheap but can fail on non-Linux platforms or if
    procfs isn't mounted; ``None`` means "can't perform the check".
    """
    try:
        with open("/proc/sys/fs/inotify/max_user_instances") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _count_inotify_instances() -> int:
    """Count inotify instances held by THIS process via /proc/self/fd/.

    Each inotify FD shows up in ``/proc/self/fd/`` as a symlink to
    ``anon_inode:inotify`` (the kernel's pseudo-file name for
    inotify instances).  Counting is one ``readlink`` per FD; cheap
    even with hundreds of open file descriptors.

    Returns 0 on non-Linux or if procfs is unreadable.  Per-process
    rather than per-user — the soft-warn primarily catches THIS
    daemon's leakage, but the kernel's limit is per-user, so other
    processes by the same user also consume from the same pool.
    Operators investigating an alert should also check ``lsof | grep
    inotify`` system-wide.
    """
    try:
        fds = os.listdir("/proc/self/fd")
    except OSError:
        return 0
    n = 0
    for fd in fds:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target == "anon_inode:inotify":
            n += 1
    return n


def check_inotify_pressure() -> Optional[str]:
    """Return a WARNING message if inotify usage > 80% of kernel
    limit, else ``None``.

    Called periodically from the daemon main loop (see
    ``server/__main__.py``'s health-check task).  Cheap enough to
    run every 5 minutes without measurable load.

    Returns:
        A human-readable warning string with current/limit counts,
        OR ``None`` when the check can't be performed (non-Linux,
        procfs unavailable) or pressure is below the threshold.
    """
    limit = _read_inotify_limit()
    if limit is None:
        return None
    if limit <= 0:
        return None
    used = _count_inotify_instances()
    if used <= int(limit * _WARN_THRESHOLD):
        return None
    pct = 100.0 * used / limit
    return (
        f"inotify pressure: {used}/{limit} instances in use ({pct:.0f}%). "
        f"Approaching kernel limit; new sessions may fail to spawn with "
        f"Errno 24 once 100% is hit. Either raise "
        f"/proc/sys/fs/inotify/max_user_instances (sysctl), or restart "
        f"the daemon to release stuck handles. Server 0.6.45+ closed "
        f"the IPC client-disconnect leak, so legitimate growth from "
        f"long-lived sessions is the most likely cause now."
    )
