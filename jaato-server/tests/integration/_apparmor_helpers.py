"""Shared helpers for the §2.6 / §8.2 multitenant-AppArmor integration test.

Lives in its own module so the test can be imported and the helpers
re-used by sister tests / future maintainers without duplicating the
capability detection or daemon-fixture logic.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def apparmor_available() -> bool:
    """Return True iff the host has working AppArmor + we can write to it.

    Three conditions:
    1. ``apparmor_parser`` binary on PATH.
    2. ``/sys/kernel/security/apparmor`` exists (kernel module loaded).
    3. We can actually load a profile — gated by CAP_MAC_ADMIN, which
       is typically only present in privileged containers / root on a
       host with AppArmor enabled.

    Cheaper fast-path: just (1) + (2).  If a real ``apparmor_parser``
    invocation later returns EPERM the test will surface that with a
    clear message; we don't need to forge a profile load just to
    probe the cap up front.
    """
    if not shutil.which("apparmor_parser"):
        return False
    if not os.path.isdir("/sys/kernel/security/apparmor"):
        return False
    return True


def dmesg_available() -> bool:
    """``dmesg`` requires CAP_SYSLOG (or root) AND the host kernel ring
    buffer must be visible.  In an unprivileged container both fail.
    """
    if not shutil.which("dmesg"):
        return False
    try:
        subprocess.run(
            ["dmesg", "-T"],
            capture_output=True,
            check=True,
            timeout=2,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


SKIP_REASON_NO_APPARMOR = (
    "AppArmor unavailable on this host.  The §2.6 acceptance test "
    "is run on a user-hosted server with CAP_MAC_ADMIN + CAP_SYSLOG; "
    "see docs/design/per_session_confined_runner_phase2_plan.md §6.8."
)
