"""Runner self-confinement bootstrap (§4.6 steps 1-3).

The runner subprocess starts unconfined (inheriting the daemon's
posture per §4.6 daemon-apparmor-state constraint), then transitions
to its session's per-session AppArmor profile via libapparmor's
``aa_change_profile`` symbol.

This module deliberately avoids importing any of the rest of jaato:

- It runs BEFORE plugin discovery (§4.6 step 4 — "now import plugin
  code").  Any cross-import would force code to load unconfined and
  break the "every plugin runs confined" guarantee.
- It uses ``ctypes`` rather than a Python AppArmor binding so the
  runner has zero non-stdlib runtime dependencies for its own
  bootstrap step.

API:

- :func:`confine_to_profile` — the load-bearing call.  Raises on any
  failure; the caller (``__main__``) translates raises into
  ``os._exit(2)`` per the spec's no-fallback-to-unconfined contract.
- :func:`read_current_profile` — small helper exposed mostly for
  tests.

Failure modes are distinguished so the daemon-side error message can
explain the cause:

- ``RuntimeError("apparmor not installed")`` — libapparmor missing.
- ``RuntimeError("aa_change_profile failed: ...")`` — kernel refused
  the transition (profile not loaded, transition rule missing, etc.).
- :class:`ConfinementMismatchError` — transition apparently succeeded
  but ``/proc/self/attr/current`` doesn't match the requested profile
  (silent-failure case from §6.5).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)


# Default proc-attr path; override-able for tests.  The runner uses
# the PROCESS-level path (``/proc/self/attr/current``) because
# ``aa_change_profile`` is a process-level transition — distinct from
# the daemon's old thread-level pattern at
# ``/proc/self/task/<tid>/attr/current``.
DEFAULT_PROC_ATTR_PATH = "/proc/self/attr/current"

# Default libapparmor SONAME.
DEFAULT_LIBAPPARMOR_SONAME = "libapparmor.so.1"


class ConfinementMismatchError(RuntimeError):
    """Raised when ``aa_change_profile`` reported success but
    ``/proc/self/attr/current`` does not match the requested profile.

    Mitigation for §6.5 (silent confine-failure): the bootstrap
    cross-checks the kernel's view, NOT just the libapparmor return
    code.  This catches the case where the kernel-side profile lacks
    a ``change_profile -> jaato-ws-*`` transition rule and the call
    silently no-ops.
    """

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(
            f"AppArmor confinement mismatch: requested {expected!r} but "
            f"/proc/self/attr/current reports {actual!r}"
        )
        self.expected = expected
        self.actual = actual


def _load_libapparmor(soname: str = DEFAULT_LIBAPPARMOR_SONAME) -> ctypes.CDLL:
    """Open ``libapparmor.so.1`` via ctypes.

    Tries the requested SONAME first (matches the spec's "we don't add
    pyspnego-style native bindings — the one libapparmor symbol we
    need is aa_change_profile"), then falls back to
    ``ctypes.util.find_library("apparmor")`` for distros that ship the
    library under a different name.

    Raises:
        RuntimeError: when neither lookup resolves.  Message is
            distinct from "kernel refused" so the daemon can tell
            "apparmor not installed" apart from "transition denied".
    """
    try:
        return ctypes.CDLL(soname, use_errno=True)
    except OSError:
        pass

    fallback = ctypes.util.find_library("apparmor")
    if fallback:
        try:
            return ctypes.CDLL(fallback, use_errno=True)
        except OSError:
            pass

    raise RuntimeError(
        f"libapparmor not installed: tried {soname!r} and "
        f"ctypes.util.find_library('apparmor'); install the "
        f"libapparmor1 (or distro-equivalent) package"
    )


def read_current_profile(proc_attr_path: str = DEFAULT_PROC_ATTR_PATH) -> str:
    """Return the current process's AppArmor profile string.

    Format on a confined process is ``<profile> (enforce)``; an
    unconfined process reports ``unconfined``.  The trailing newline
    written by the kernel is stripped.
    """
    with open(proc_attr_path, "r") as f:
        return f.read().rstrip("\n")


def confine_to_profile(
    profile_name: str,
    *,
    libapparmor: Optional[ctypes.CDLL] = None,
    proc_attr_path: str = DEFAULT_PROC_ATTR_PATH,
) -> None:
    """Self-confine the current process to *profile_name*.

    Implements §4.6 bootstrap steps 2-3:

    1. Call ``aa_change_profile(profile_name)`` via ctypes.
    2. Read ``/proc/self/attr/current`` and verify the kernel agrees
       the process is now confined to *profile_name*.

    Per the spec's no-fallback-to-unconfined contract, ANY failure
    raises.  The caller (``__main__``) is responsible for translating
    the raise into an explicit ``os._exit(2)`` so the daemon detects
    runner failure via socket EOF + non-zero exit, NOT a half-confined
    runner that survives.

    Args:
        profile_name: The AppArmor profile to enter.  Must already be
            loaded in the kernel by the daemon-side
            ``AppArmorManager.provision_profile`` BEFORE this runner
            was forked.
        libapparmor: Pre-opened ctypes handle (test injection point).
            When ``None``, :func:`_load_libapparmor` is called.
        proc_attr_path: Override for ``/proc/self/attr/current``
            (test injection point).

    Raises:
        RuntimeError: libapparmor lookup failure ("apparmor not
            installed") or ``aa_change_profile`` returned non-zero
            ("kernel refused the transition").
        ConfinementMismatchError: ``aa_change_profile`` returned 0
            but ``/proc/self/attr/current`` reports a different
            profile (silent-failure case — §6.5).
    """
    if not profile_name:
        raise RuntimeError("confine_to_profile: profile_name is empty")

    lib = libapparmor if libapparmor is not None else _load_libapparmor()

    # ``int aa_change_profile(const char *profile)``.  Returns 0 on
    # success, -1 on error with errno set.
    aa_change_profile = lib.aa_change_profile
    aa_change_profile.argtypes = [ctypes.c_char_p]
    aa_change_profile.restype = ctypes.c_int

    rc = aa_change_profile(profile_name.encode("utf-8"))
    if rc != 0:
        err = ctypes.get_errno()
        raise RuntimeError(
            f"aa_change_profile({profile_name!r}) failed: "
            f"errno={err} ({os.strerror(err)}); "
            f"the kernel refused the transition — verify the profile "
            f"is loaded and grants change_profile from the runner's "
            f"current profile (likely unconfined)"
        )

    # Cross-check the kernel's view.  Catches the silent-success case
    # where libapparmor returned 0 but the kernel didn't actually
    # transition (rare but observed when the parent profile lacks the
    # required change_profile -> rule).
    expected_prefix = f"{profile_name} "  # e.g. "jaato-ws-20260507_120000 (enforce)"
    actual = read_current_profile(proc_attr_path)
    if not actual.startswith(expected_prefix) and actual != profile_name:
        raise ConfinementMismatchError(expected=profile_name, actual=actual)

    logger.info(
        "runner confined to AppArmor profile %s (kernel reports: %s)",
        profile_name, actual,
    )
