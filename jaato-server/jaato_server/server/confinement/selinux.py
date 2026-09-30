"""The SELinux backend (design ``docs/design/selinux-backend.md``).

Phase 1: it answers whether this host could run SELinux confinement, and
provisions nothing.  Until provisioning exists, :meth:`is_available` is
``False`` even on a ready host, with a reason that says the host is ready
and the backend is not; :meth:`host_readiness` is the check itself, which
is what ``jaato-doctor`` and the selection log report.  Saying "available"
before a boundary can be provisioned would let a caller believe a session
was confined.

The checks follow design §10, first failing precondition wins:

1. Linux, SELinux mounted, ``libselinux`` loadable (ctypes; no new Python
   dependency).
2. SELinux is not disabled.
3. The policy defines jaato's runner domain, and the module is at least
   :data:`REQUIRED_POLICY_VERSION`.  The version is read from a marker type
   the module declares (``jaato_policy_v<N>_t``): asking the kernel whether
   a context is valid needs no privilege, where ``semodule -l`` needs root.
4. MCS is enabled (per-workspace isolation is by category).
5. This process may transition into the runner domain.
"""

from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from typing import Callable, Optional

from jaato_server.server.confinement.base import Boundary, ConfinementHandle
from jaato_server.shared.lsm_label import (
    BACKEND_SELINUX,
    AvDecision,
    load_libselinux,
    selinux_host_enforcing,
)

#: The policy module version this build needs (design §4).
REQUIRED_POLICY_VERSION = 1

#: jaato's runner domain, as a context the kernel can validate.
RUNNER_PROBE_CONTEXT = "system_u:system_r:jaato_runner_t:s0"

SELINUX_MOUNT = "/sys/fs/selinux"


def policy_marker_context(version: int) -> str:
    """The context of the marker type module version *version* declares."""
    return f"system_u:object_r:jaato_policy_v{version}_t:s0"


@dataclass(frozen=True)
class Readiness:
    """The result of :meth:`SELinuxBackend.host_readiness`.

    Attributes:
        ready: Every check passed.
        reason: The first failing check, or ``None``.
        enforcing: The host's global switch, ``None`` when unreadable.
    """

    ready: bool
    reason: Optional[str]
    enforcing: Optional[bool]


class _Kernel:
    """The libselinux calls the checks make, behind one seam for tests."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib

    def context_valid(self, context: str) -> bool:
        fn = self._lib.security_check_context
        fn.argtypes = [ctypes.c_char_p]
        fn.restype = ctypes.c_int
        return fn(context.encode("utf-8")) == 0

    def mls_enabled(self) -> bool:
        fn = self._lib.is_selinux_mls_enabled
        fn.restype = ctypes.c_int
        return fn() == 1

    def current_context(self) -> Optional[str]:
        getcon = self._lib.getcon
        getcon.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
        getcon.restype = ctypes.c_int
        out = ctypes.c_char_p()
        if getcon(ctypes.byref(out)) != 0 or not out.value:
            return None
        return out.value.decode("utf-8", "replace")

    def may_transition(self, source: str, target: str) -> Optional[bool]:
        lib = self._lib
        lib.string_to_security_class.argtypes = [ctypes.c_char_p]
        lib.string_to_security_class.restype = ctypes.c_uint16
        lib.string_to_av_perm.argtypes = [ctypes.c_uint16, ctypes.c_char_p]
        lib.string_to_av_perm.restype = ctypes.c_uint32
        lib.security_compute_av_flags.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint16,
            ctypes.c_uint32, ctypes.POINTER(AvDecision),
        ]
        lib.security_compute_av_flags.restype = ctypes.c_int
        tclass = lib.string_to_security_class(b"process")
        perm = lib.string_to_av_perm(tclass, b"transition") if tclass else 0
        if not perm:
            return None
        avd = AvDecision()
        if lib.security_compute_av_flags(
            source.encode("utf-8"), target.encode("utf-8"),
            tclass, perm, ctypes.byref(avd),
        ) != 0:
            return None
        return bool(avd.allowed & perm)


def _default_kernel() -> Optional[_Kernel]:
    lib = load_libselinux()
    return _Kernel(lib) if lib is not None else None


class SELinuxBackend:
    """SELinux confinement (phase 1: readiness only)."""

    name = BACKEND_SELINUX

    def __init__(
        self,
        *,
        kernel_factory: Callable[[], Optional[_Kernel]] = _default_kernel,
        host_enforcing: Callable[[], Optional[bool]] = selinux_host_enforcing,
        system: Callable[[], str] = platform.system,
        mount_present: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._kernel_factory = kernel_factory
        self._host_enforcing = host_enforcing
        self._system = system
        self._mount_present = mount_present or _selinux_mounted
        self._readiness: Optional[Readiness] = None

    def host_readiness(self) -> Readiness:
        """Run the §10 checks once and remember the answer."""
        if self._readiness is None:
            self._readiness = self._check()
        return self._readiness

    def _check(self) -> Readiness:
        if self._system() != "Linux":
            return Readiness(False, "not running on Linux", None)
        if not self._mount_present():
            return Readiness(
                False, f"SELinux is not mounted ({SELINUX_MOUNT} missing)", None,
            )
        kernel = self._kernel_factory()
        if kernel is None:
            return Readiness(False, "libselinux could not be loaded", None)
        enforcing = self._host_enforcing()
        if enforcing is None:
            return Readiness(False, "SELinux mode could not be read (disabled?)", None)
        reason = _policy_problem(kernel) or _transition_problem(kernel)
        return Readiness(reason is None, reason, enforcing)

    def is_available(self) -> bool:
        """``False`` until provisioning exists, whatever the host says."""
        return False

    @property
    def unavailable_reason(self) -> Optional[str]:
        readiness = self.host_readiness()
        if not readiness.ready:
            return readiness.reason
        return (
            "the host is ready for SELinux confinement, but this build "
            "cannot provision SELinux boundaries yet"
        )

    def confinement_id_for_boundary(self, boundary: Boundary) -> str:
        raise NotImplementedError("SELinux provisioning is not implemented yet")

    def provision(
        self, session_id: str, boundary: Boundary,
    ) -> Optional[ConfinementHandle]:
        return None

    def release(self, handle: ConfinementHandle) -> None:
        return None


def _policy_problem(kernel: _Kernel) -> Optional[str]:
    """Checks 3 and 4: the module is loaded, recent enough, with MCS."""
    if not kernel.context_valid(RUNNER_PROBE_CONTEXT):
        return "the jaato SELinux policy module is not loaded (jaato_runner_t is unknown)"
    if not kernel.context_valid(policy_marker_context(REQUIRED_POLICY_VERSION)):
        return (
            f"the loaded jaato policy module is older than version "
            f"{REQUIRED_POLICY_VERSION}"
        )
    if not kernel.mls_enabled():
        return "the policy has no MLS/MCS support"
    return None


def _transition_problem(kernel: _Kernel) -> Optional[str]:
    """Check 5: this process may enter the runner domain."""
    own = kernel.current_context()
    if own is None:
        return "this process's own context could not be read"
    if kernel.may_transition(own, RUNNER_PROBE_CONTEXT) is not True:
        return f"this process ({own}) may not transition into jaato_runner_t"
    return None


def _selinux_mounted() -> bool:
    import os

    return os.path.isfile(os.path.join(SELINUX_MOUNT, "enforce"))
