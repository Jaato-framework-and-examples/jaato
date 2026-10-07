"""One reading of a task's LSM label, whichever LSM wrote it.

:mod:`jaato_server.shared.apparmor_label` answers "which profile, and is
the kernel enforcing it" for AppArmor.  The SELinux backend
(``docs/design/selinux-backend.md`` §3.3) needs the same two questions
answered for an SELinux context, and the callers that ask them (the
runner's thread verification, ``interactive_shell``'s
``require_confinement``, the notebook's boundary tier, ``sandbox_mode``)
must not grow a second opinion per LSM.  This module is that one answer.

Two questions, kept apart exactly as #1014 keeps them apart:

* **which boundary a task is in** — :attr:`LsmLabel.identity`.  For
  AppArmor the profile name; for SELinux ``type:level``, because two
  runners of one domain in different workspaces differ only by level.
  Mode-tolerant, for comparing tasks with each other (#1023).
* **whether the kernel is enforcing it** — :attr:`LsmLabel.enforced`.
  The only predicate that may back a claim that a boundary exists.

SELinux has no per-label mode annotation.  Enforcement is the host's
enforcing switch AND the domain not being permissive (``semanage
permissive -a``), so both are read here.  An answer that could not be read
is "not enforced": absence of evidence is not a boundary.

"Confined" for SELinux means one of jaato's own domains
(:data:`JAATO_SELINUX_DOMAINS`).  Every task on an SELinux host has a
domain, and ``unconfined_t`` or ``unconfined_service_t`` is not a jaato
boundary; treating any context as "confined" would make every thread of an
unconfined runner look confined.

Pure stdlib, zero jaato imports beyond :mod:`apparmor_label` (itself pure
stdlib), for the reason that module gives: the runner bootstrap reads this
before plugin discovery.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from jaato_server.shared.apparmor_label import (
    SANDBOX_MODE_APPARMOR,
    SANDBOX_MODE_APPARMOR_COMPLAIN,
    parse_label as parse_apparmor_label,
)

BACKEND_APPARMOR = "apparmor"
BACKEND_SELINUX = "selinux"
BACKEND_NONE = "none"

MODE_ENFORCING = "enforcing"
MODE_PERMISSIVE = "permissive"

#: The domains jaato's policy module defines (design §4.1).  Only a task in
#: one of these is inside a jaato boundary.
JAATO_SELINUX_DOMAINS = frozenset({
    "jaato_runner_t",
    "jaato_child_t",
    "jaato_isolated_t",
    "jaato_template_t",
})

#: Where the kernel publishes SELinux's global enforcing switch.
SELINUX_ENFORCE_PATH = "/sys/fs/selinux/enforce"


@dataclass(frozen=True)
class SELinuxContext:
    """A parsed ``user:role:type[:level]`` security context."""

    user: str
    role: str
    type: str
    level: str

    @property
    def identity(self) -> str:
        """``type:level``, or ``type`` when the policy has no MLS/MCS."""
        return f"{self.type}:{self.level}" if self.level else self.type


def parse_selinux_context(raw: Optional[str]) -> Optional[SELinuxContext]:
    """Parse a context string, or ``None`` when *raw* is not one.

    The level keeps any ``:`` it contains (``s0-s0:c0.c1023``), so only the
    first three separators split.
    """
    cleaned = (raw or "").replace("\x00", "").strip()
    parts = cleaned.split(":", 3)
    if len(parts) < 3 or not all(parts[:3]):
        return None
    level = parts[3] if len(parts) == 4 else ""
    return SELinuxContext(user=parts[0], role=parts[1], type=parts[2], level=level)


@dataclass(frozen=True)
class LsmLabel:
    """A task's label, read through whichever LSM is active.

    Attributes:
        backend: ``"apparmor"``, ``"selinux"`` or ``"none"``.
        raw: The label as read (NUL and whitespace stripped); empty when
            the read failed.
        identity: Which boundary the task is in, ``""`` when none.
        mode: AppArmor's annotation (``enforce`` / ``complain``) or
            SELinux's ``enforcing`` / ``permissive``; ``None`` when
            nothing established one.
        enforced: The kernel is blocking on this boundary.
    """

    backend: str
    raw: str
    identity: str
    mode: Optional[str]
    enforced: bool

    @property
    def confined(self) -> bool:
        """Is the task inside a jaato boundary, whatever the mode?"""
        return bool(self.identity)


def selinux_host_enforcing(path: str = SELINUX_ENFORCE_PATH) -> Optional[bool]:
    """The host's global switch: ``True`` / ``False``, ``None`` if unreadable."""
    try:
        with open(path, "r") as handle:
            value = handle.read().strip()
    except OSError:
        return None
    if value == "1":
        return True
    if value == "0":
        return False
    return None


_SELINUX_AVD_FLAGS_PERMISSIVE = 0x0001

#: ``ctypes`` is imported lazily, on the first call that needs libselinux,
#: and never at module import.  The notebook kernel imports this module
#: (#1519) BEFORE it installs its audit hook, and on the audit tier a
#: ``ctypes`` already in ``sys.modules`` would hand every cell
#: ``ctypes.pythonapi`` without the ``dlopen(None)`` the hook refuses
#: (#1011).  ``AvDecision`` is still importable by name; it is built on
#: first access.
_AV_DECISION: Any = None


def _av_decision_type() -> Any:
    """libselinux's ``struct av_decision``, built on first use."""
    global _AV_DECISION
    if _AV_DECISION is None:
        import ctypes

        class AvDecision(ctypes.Structure):
            """libselinux's ``struct av_decision``."""

            _fields_ = [
                ("allowed", ctypes.c_uint32),
                ("decided", ctypes.c_uint32),
                ("auditallow", ctypes.c_uint32),
                ("auditdeny", ctypes.c_uint32),
                ("seqno", ctypes.c_uint),
                ("flags", ctypes.c_uint),
            ]

        _AV_DECISION = AvDecision
    return _AV_DECISION


def __getattr__(name: str) -> Any:
    if name == "AvDecision":
        return _av_decision_type()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_libselinux() -> Any:
    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("selinux") or "libselinux.so.1"
    try:
        return ctypes.CDLL(name, use_errno=True)
    except OSError:
        return None


def selinux_domain_permissive(context: str) -> Optional[bool]:
    """Is *context*'s domain permissive?  ``None`` when it cannot be asked.

    Asks the kernel through ``security_compute_av_flags``: the decision for
    the domain acting on itself carries ``SELINUX_AVD_FLAGS_PERMISSIVE``
    when the domain has been made permissive.  The class and the requested
    access do not affect that flag, so ``process`` / no access is enough.
    """
    lib = load_libselinux()
    if lib is None:
        return None
    import ctypes

    AvDecision = _av_decision_type()
    try:
        to_class = lib.string_to_security_class
        to_class.argtypes = [ctypes.c_char_p]
        to_class.restype = ctypes.c_uint16
        compute = lib.security_compute_av_flags
        compute.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint16,
            ctypes.c_uint32, ctypes.POINTER(AvDecision),
        ]
        compute.restype = ctypes.c_int
    except AttributeError:
        return None
    tclass = to_class(b"process")
    if not tclass:
        return None
    avd = AvDecision()
    ctx = context.encode("utf-8")
    if compute(ctx, ctx, tclass, 0, ctypes.byref(avd)) != 0:
        return None
    return bool(avd.flags & _SELINUX_AVD_FLAGS_PERMISSIVE)


def parse_selinux_label(
    raw: Optional[str],
    *,
    host_enforcing: Callable[[], Optional[bool]] = selinux_host_enforcing,
    domain_permissive: Callable[[str], Optional[bool]] = selinux_domain_permissive,
) -> LsmLabel:
    """Read an SELinux context as an :class:`LsmLabel`.

    ``enforced`` needs positive evidence on both halves: the host is
    enforcing, and the domain is known not to be permissive.  The two
    readers are parameters so a caller (or a test) with its own answer does
    not reach the kernel.
    """
    cleaned = (raw or "").replace("\x00", "").strip()
    context = parse_selinux_context(cleaned)
    if context is None or context.type not in JAATO_SELINUX_DOMAINS:
        return LsmLabel(BACKEND_SELINUX, cleaned, "", None, False)
    host = host_enforcing()
    permissive = domain_permissive(cleaned) if host else None
    if host is True and permissive is False:
        mode: Optional[str] = MODE_ENFORCING
    elif host is False or permissive is True:
        mode = MODE_PERMISSIVE
    else:
        mode = None
    return LsmLabel(
        BACKEND_SELINUX, cleaned, context.identity, mode, mode == MODE_ENFORCING,
    )


def parse_lsm_label(raw: Optional[str], backend: str, **readers) -> LsmLabel:
    """Read *raw* as a label of *backend*.

    The backend is the caller's to name, never guessed from the string: a
    daemon knows which backend provisioned a session, and a guess would let
    an AppArmor profile name containing ``:`` read as an SELinux context.
    """
    if backend == BACKEND_SELINUX:
        return parse_selinux_label(raw, **readers)
    if backend == BACKEND_APPARMOR:
        label = parse_apparmor_label(raw)
        return LsmLabel(
            BACKEND_APPARMOR, label.raw, label.profile, label.mode, label.enforced,
        )
    cleaned = (raw or "").replace("\x00", "").strip()
    return LsmLabel(BACKEND_NONE, cleaned, "", None, False)


def active_lsm_backend(
    *,
    lsm_list_path: str = "/sys/kernel/security/lsm",
    apparmor_path: str = "/sys/kernel/security/apparmor",
    selinux_path: str = "/sys/fs/selinux",
) -> str:
    """Which of the two LSMs this kernel is running, or ``"none"``.

    Read from ``/sys/kernel/security/lsm``, the kernel's own list of active
    LSMs, when it is readable; otherwise from the filesystems each LSM
    mounts.  They cannot both be the active major LSM, so the first present
    wins.  This says what the KERNEL runs; whether jaato can manage it
    (tools, policy module, privileges) is each backend's availability check.
    """
    try:
        with open(lsm_list_path, "r") as handle:
            active = {name.strip() for name in handle.read().split(",")}
    except OSError:
        active = None
    if active is not None:
        if BACKEND_APPARMOR in active:
            return BACKEND_APPARMOR
        if BACKEND_SELINUX in active:
            return BACKEND_SELINUX
        return BACKEND_NONE
    if os.path.isdir(apparmor_path):
        return BACKEND_APPARMOR
    if os.path.isfile(os.path.join(selinux_path, "enforce")):
        return BACKEND_SELINUX
    return BACKEND_NONE


# ---------------------------------------------------------------------
# The persisted vocabulary, widened (design §3.3)
# ---------------------------------------------------------------------

#: ``Session.sandbox_mode`` for an SELinux-confined session the kernel
#: enforces.
SANDBOX_MODE_SELINUX = "selinux"

#: ``Session.sandbox_mode`` for an SELinux-confined session whose host or
#: domain is permissive.  The SELinux counterpart of ``apparmor-complain``:
#: a record must not claim a boundary the kernel was not applying, and an
#: older reader comparing against ``"apparmor"`` reads it as unconfined,
#: the safe direction.
SANDBOX_MODE_SELINUX_PERMISSIVE = "selinux-permissive"


def sandbox_mode_for_selinux(*, permissive: bool) -> str:
    """Pick the ``sandbox_mode`` value for an SELinux-confined session."""
    return SANDBOX_MODE_SELINUX_PERMISSIVE if permissive else SANDBOX_MODE_SELINUX


def sandbox_mode_is_kernel(mode: Optional[str]) -> bool:
    """Did this session get a kernel confinement, in any mode, from any LSM?"""
    return mode in (
        SANDBOX_MODE_APPARMOR, SANDBOX_MODE_APPARMOR_COMPLAIN,
        SANDBOX_MODE_SELINUX, SANDBOX_MODE_SELINUX_PERMISSIVE,
    )


def sandbox_mode_is_kernel_enforced(mode: Optional[str]) -> bool:
    """Did this session run behind an enforced kernel boundary?"""
    return mode in (SANDBOX_MODE_APPARMOR, SANDBOX_MODE_SELINUX)


# ---------------------------------------------------------------------
# The notebook kernel's SELinux boundary (#1519)
# ---------------------------------------------------------------------

#: Where a task reads its own label.  The same file AppArmor's reader uses;
#: on an SELinux host it holds the task's security context.
PROC_SELF_ATTR_CURRENT = "/proc/self/attr/current"

#: The jaato domain model-driven subprocesses exec into (design §4.1).
SELINUX_CHILD_DOMAIN = "jaato_child_t"

#: The domains that bound a notebook cell: code running in them cannot
#: leave them.  The SELinux counterpart of #1323's rule that only a profile
#: a cell cannot unconfine itself from counts.  ``jaato_child_t`` has no
#: ``setexec``, ``setcurrent`` or ``dyntransition`` (the ``selinux-policy``
#: CI job checks it).  ``jaato_runner_t`` is deliberately NOT here although
#: it cannot change its own domain either: it is the runner's domain, the
#: kernel is supposed to be started out of it, and a kernel found in it means
#: the ``setexeccon`` step did not happen.
SELINUX_CELL_BOUNDARY_DOMAINS = frozenset({SELINUX_CHILD_DOMAIN})


def read_own_context(path: str = PROC_SELF_ATTR_CURRENT) -> Optional[str]:
    """This task's raw label, NUL and whitespace stripped; ``None`` if unreadable."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().replace("\x00", "").strip()
    except OSError:
        return None


@dataclass(frozen=True)
class SELinuxSessionBoundary:
    """What the daemon provisioned for this runner's session, SELinux only.

    Recorded by the runner bootstrap from ``SessionInitEnvelope.confinement``
    and read by the notebook backend when it spawns a kernel (#1519).

    Attributes:
        label: The runner's context (``…:jaato_runner_t:<level>``).
        child_label: The context subprocesses exec into
            (``…:jaato_child_t:<level>``).
        enforcing_attested: The DAEMON saw the host enforcing and neither the
            runner nor the child domain permissive when it provisioned the
            boundary.  A confined task cannot ask that itself: neither jaato
            domain may read ``/sys/fs/selinux`` or compute an access vector.
            ``False`` when the daemon did not say (an older daemon).
    """

    label: str
    child_label: str
    enforcing_attested: bool = False


_SESSION_BOUNDARY: Optional[SELinuxSessionBoundary] = None
_SESSION_BOUNDARY_LOCK = threading.Lock()


def set_selinux_session_boundary(
    boundary: Optional[SELinuxSessionBoundary],
) -> None:
    """Record (or, with ``None``, clear) this process's SELinux session boundary.

    Process-wide like ``confinement_grants``: a runner hosts one session at
    a time, and the bootstrap sets or clears it on every session so a pool
    slot never reports a previous session's boundary.
    """
    global _SESSION_BOUNDARY
    with _SESSION_BOUNDARY_LOCK:
        _SESSION_BOUNDARY = boundary


def selinux_session_boundary() -> Optional[SELinuxSessionBoundary]:
    """The boundary :func:`set_selinux_session_boundary` recorded, if any."""
    with _SESSION_BOUNDARY_LOCK:
        return _SESSION_BOUNDARY


def selinux_cell_boundary(
    *,
    expected_label: Optional[str] = None,
    enforcing_attested: bool = False,
    raw: Optional[str] = None,
    backend: Optional[str] = None,
    host_enforcing: Optional[Callable[[], Optional[bool]]] = None,
) -> Optional[str]:
    """The context bounding code this process runs, or ``None`` (#1519).

    Positive evidence only (#1014).  A context is returned when ALL hold:

    * the active LSM is SELinux, or it could not be read and the spawner
      named the label (see below);
    * this task's own context is readable and its domain is in
      :data:`SELINUX_CELL_BOUNDARY_DOMAINS`;
    * it is the ``expected_label`` the spawner set, when one is given;
    * the kernel enforces it: the host's switch, when this task can read it,
      must say enforcing; whether the DOMAIN is permissive cannot be asked
      from inside it, so that half is the daemon's attestation
      (``enforcing_attested``).  A host switch read as ``0`` refuses
      whatever was attested; an unreadable switch leaves the attestation to
      decide.

    Never asks libselinux, so it never imports ``ctypes``: the notebook
    kernel calls this before installing its audit hook (see
    :data:`_AV_DECISION`).

    Args:
        expected_label: The child context the spawner wrote to
            ``/proc/self/attr/exec``.
        enforcing_attested: The daemon's observation (see
            :class:`SELinuxSessionBoundary`).
        raw: The context to judge; read from ``/proc/self/attr/current`` when
            ``None``.  A parameter so tests need no SELinux kernel.
        backend: The active LSM; :func:`active_lsm_backend` when ``None``.
        host_enforcing: Reads the host switch;
            :func:`selinux_host_enforcing` when ``None``.

    Inside ``jaato_child_t`` the active LSM usually cannot be read:
    ``/sys/kernel/security/lsm`` and ``/sys/fs/selinux`` are ``security_t``,
    which the domain may neither read nor ``getattr``, so
    :func:`active_lsm_backend` answers ``"none"`` on every SELinux host.  A
    ``"none"`` is therefore absence of evidence, not a refusal: the tier is
    still granted when the spawner named the label (``expected_label``) and
    this task's own readable context IS that label, in a cell-boundary
    domain.  Only a positive AppArmor answer, or ``"none"`` with no
    expected label, refuses here.
    """
    detected = backend or active_lsm_backend()
    if detected == BACKEND_APPARMOR:
        return None
    if detected != BACKEND_SELINUX and not expected_label:
        return None
    context_raw = read_own_context() if raw is None else raw
    context = parse_selinux_context(context_raw)
    if context is None or context.type not in SELINUX_CELL_BOUNDARY_DOMAINS:
        return None
    cleaned = (context_raw or "").replace("\x00", "").strip()
    if expected_label and cleaned != expected_label:
        return None
    if (host_enforcing or selinux_host_enforcing)() is False:
        return None
    return cleaned if enforcing_attested else None
