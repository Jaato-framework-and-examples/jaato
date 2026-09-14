"""Shared helpers for the §2.6 / §8.2 multitenant-AppArmor integration test.

Lives in its own module so the test can be imported and the helpers
re-used by sister tests / future maintainers without duplicating the
capability detection or daemon-fixture logic.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import subprocess


def apparmor_available() -> bool:
    """Return True iff the DAEMON would actually confine on this host.

    Delegates to :meth:`server.apparmor.AppArmorManager.is_available`
    rather than re-deriving the answer, because the two disagreeing is
    precisely what broke: this helper used to check only

    1. ``apparmor_parser`` on PATH, and
    2. ``/sys/kernel/security/apparmor`` present,

    naming CAP_MAC_ADMIN as a third condition and then deliberately not
    testing it — *"if a real apparmor_parser invocation later returns
    EPERM the test will surface that with a clear message"*.

    It does not.  On a GitHub runner both cheap conditions hold and the
    capability does not, so the gate did not skip; the daemon logged

        AppArmor confinement NOT available (cannot create profile dir
        /etc/apparmor.d/jaato) — running unconfined

    and the test drove its whole body against a daemon whose confinement
    was off — i.e. asserted kernel-enforced isolation on a host with
    none.  The "clear message" was a 60 s ``SessionNotConfirmed``.

    The daemon's own check additionally requires ``aa-exec``, a
    *writable* profile dir, and passwordless ``sudo apparmor_parser``.
    Asking it directly means this gate skips exactly when confinement
    would not happen, and can never drift from production again.

    Falls back to the two cheap conditions only if the daemon's manager
    cannot be imported at all (an SDK-only checkout), which is a skip in
    the safe direction: no import, no daemon, nothing to confine.
    """
    try:
        from server.apparmor import AppArmorManager
    except ImportError:             # SDK-only checkout: no daemon at all
        return (
            bool(shutil.which("apparmor_parser"))
            and os.path.isdir("/sys/kernel/security/apparmor")
        )

    # ``workspace_root`` is required but irrelevant to the probe --
    # ``is_available`` inspects the HOST (binaries, kernel module,
    # profile-dir writability, sudo), never the workspace.  Passing a
    # placeholder keeps the probe read-only.
    #
    # Deliberately NOT wrapped in a bare ``except Exception``: swallowing
    # a constructor error here would return False for a reason that has
    # nothing to do with AppArmor, and a gate that skips for the wrong
    # reason is the defect being fixed, one level up.
    return bool(AppArmorManager(workspace_root=tempfile.gettempdir())
                .is_available())


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
