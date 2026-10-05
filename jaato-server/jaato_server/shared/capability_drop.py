"""Model-driven subprocesses run with no capabilities (#1543).

A confined payload (``//child`` on AppArmor, ``jaato_child_t`` on SELinux)
used to keep the daemon's full capability bounding set: the daemon runs as
root, the runner is forked from it, and nothing reduced the set.  A root
child then started every ``exec`` with every capability permitted and
effective.  The LSM denied their USE, so the LSM policy was the one layer
between a payload and ``CAP_SYS_TIME``, ``CAP_SYSLOG``, ``CAP_SETUID`` and
the rest.  This module is a second, independent layer: it does not depend
on the policy being complete.

What the child does
-------------------
Composed into the ``//child`` preexec every ``cli`` / ``interactive_shell``
/ notebook subprocess receives (``server.runner.session._child_preexec``),
around the LSM transition and before the #1503 seccomp filter::

    PR_CAPBSET_DROP for every capability not kept      (before the LSM transition)
    <LSM transition>
    PR_CAP_AMBIENT_CLEAR_ALL
    capset(permitted = effective = kept, inheritable = 0)
    PR_SET_NO_NEW_PRIVS                                 (then the seccomp filter)

Why the order:

- ``PR_CAPBSET_DROP`` needs ``CAP_SETPCAP``, which ``//child`` does not
  grant.  It runs while the child still wears the runner's profile, which
  grants exactly that capability (AppArmor template v47, SELinux module
  1.10.0).  Under SELinux ``setexeccon`` changes nothing until ``execve``,
  so the order does not matter there.
- ``capset`` only lowers sets, which needs no capability.  AppArmor does
  not mediate it; SELinux checks ``process setcap`` on the runner domain.
- ``PR_SET_NO_NEW_PRIVS`` must come after the LSM transition (both LSMs
  refuse the transition once it is set), and is what keeps the cleared
  sets cleared across ``execve``: for a root process the kernel would
  otherwise recompute permitted as the whole bounding set.  It is set here
  too, not only by the seccomp filter, so ``seccomp: off`` does not undo
  this.

Nothing in the child allocates or logs: every structure is built in the
runner before the fork, and every call's failure is ignored there.  The
runner learns what the calls achieve by running them once, at plan time,
in a probe child (:func:`probe`), and records that as the posture.

Posture
-------
=========== =============================================================
dropped     bounding set, permitted, effective, inheritable and ambient
            are the kept set (empty by default) in every model-driven
            subprocess
partial     permitted / effective / inheritable / ambient are cleared and
            NO_NEW_PRIVS is set, but the bounding set could not be reduced
            (the runner is not root, or the loaded LSM policy predates
            #1543).  An ``exec`` still cannot gain a capability
inherit     the profile said ``capabilities: inherit`` (WARNING)
absent      a kernel boundary is active and neither half took effect
            (WARNING)
unconfined  no kernel boundary, so nothing is dropped
=========== =============================================================

None of these refuses a spawn: the LSM is still the boundary, and this is
the layer beneath it.  ``absent`` and ``partial`` are said, never silent.

Stdlib only, so the runner can load it before plugin discovery.
"""

from __future__ import annotations

import ctypes
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------- vocabulary

#: ``runtime_limits.capabilities`` scalar values.  A list of names is the
#: third form: those capabilities are kept, everything else dropped.
MODE_NONE = "none"
MODE_INHERIT = "inherit"
MODES = (MODE_NONE, MODE_INHERIT)

POSTURE_DROPPED = "dropped"
POSTURE_PARTIAL = "partial"
POSTURE_INHERIT = "inherit"
POSTURE_ABSENT = "absent"
POSTURE_UNCONFINED = "unconfined"
POSTURES = (POSTURE_DROPPED, POSTURE_PARTIAL, POSTURE_INHERIT,
            POSTURE_ABSENT, POSTURE_UNCONFINED)

#: Linux capability names (``capability(7)``, without ``CAP_``) and numbers.
CAPABILITIES: Dict[str, int] = {
    "chown": 0, "dac_override": 1, "dac_read_search": 2, "fowner": 3,
    "fsetid": 4, "kill": 5, "setgid": 6, "setuid": 7, "setpcap": 8,
    "linux_immutable": 9, "net_bind_service": 10, "net_broadcast": 11,
    "net_admin": 12, "net_raw": 13, "ipc_lock": 14, "ipc_owner": 15,
    "sys_module": 16, "sys_rawio": 17, "sys_chroot": 18, "sys_ptrace": 19,
    "sys_pacct": 20, "sys_admin": 21, "sys_boot": 22, "sys_nice": 23,
    "sys_resource": 24, "sys_time": 25, "sys_tty_config": 26, "mknod": 27,
    "lease": 28, "audit_write": 29, "audit_control": 30, "setfcap": 31,
    "mac_override": 32, "mac_admin": 33, "syslog": 34, "wake_alarm": 35,
    "block_suspend": 36, "audit_read": 37, "perfmon": 38, "bpf": 39,
    "checkpoint_restore": 40,
}

# prctl / capset constants.
PR_CAPBSET_READ = 23
PR_CAPBSET_DROP = 24
PR_SET_NO_NEW_PRIVS = 38
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
_LINUX_CAPABILITY_VERSION_3 = 0x20080522


def _cap_name(raw: str) -> str:
    """``CAP_NET_BIND_SERVICE`` / ``net_bind_service`` -> ``net_bind_service``."""
    name = raw.strip().lower()
    return name[4:] if name.startswith("cap_") else name


def unknown_capabilities(names: Optional[Iterable[str]]) -> List[str]:
    """The names in *names* that are no Linux capability."""
    return [n for n in (names or ()) if _cap_name(n) not in CAPABILITIES]


def resolve(value: Any) -> Tuple[str, Tuple[str, ...]]:
    """``(mode, kept)`` for a ``runtime_limits.capabilities`` value.

    ``None`` and ``"none"`` keep nothing; ``"inherit"`` is the opt-out; a
    list keeps the capabilities it names (unknown names are not kept).
    """
    if value is None or value == MODE_NONE:
        return MODE_NONE, ()
    if value == MODE_INHERIT:
        return MODE_INHERIT, ()
    kept = sorted({_cap_name(n) for n in value if _cap_name(n) in CAPABILITIES})
    return MODE_NONE, tuple(kept)


def _last_cap() -> int:
    try:
        with open("/proc/sys/kernel/cap_last_cap") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return max(CAPABILITIES.values())


# ------------------------------------------------------------ the syscalls


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32),
                ("permitted", ctypes.c_uint32),
                ("inheritable", ctypes.c_uint32)]


_libc: Any = None


def _load_libc() -> Any:
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
    return _libc


def _capget(libc: Any) -> Optional[Tuple[int, int, int]]:
    """``(effective, permitted, inheritable)`` of the calling thread."""
    hdr = _CapHeader(_LINUX_CAPABILITY_VERSION_3, 0)
    data = (_CapData * 2)()
    if libc.capget(ctypes.byref(hdr), data) != 0:
        return None

    def join(attr: str) -> int:
        return getattr(data[0], attr) | (getattr(data[1], attr) << 32)
    return join("effective"), join("permitted"), join("inheritable")


@dataclass
class CapabilityDrop:
    """The two halves of the drop, prepared in the runner before any fork.

    Attributes:
        keep_mask: The capabilities kept, as a bitmask.
        drop: Capability numbers removed from the bounding set.
    """

    keep_mask: int
    drop: Tuple[int, ...]
    _libc: Any = field(repr=False)
    _hdr: Any = field(repr=False)
    _data: Any = field(repr=False)

    def before_transition(self) -> None:
        """Reduce the bounding set.  Runs in the forked child, while it
        still wears the runner's profile (``CAP_SETPCAP``).  Stops at the
        first refusal: one refused drop means every drop is refused."""
        prctl = self._libc.prctl
        for cap in self.drop:
            if prctl(PR_CAPBSET_DROP, cap, 0, 0, 0) != 0:
                return

    def after_transition(self) -> None:
        """Clear ambient, set permitted / effective to the kept set and
        inheritable to empty, then NO_NEW_PRIVS.  Runs in the forked child
        after the LSM transition; failures are left to the probe to find."""
        prctl = self._libc.prctl
        prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0)
        self._libc.capset(ctypes.byref(self._hdr), self._data)
        prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)


def prepare(kept: Sequence[str] = ()) -> CapabilityDrop:
    """Build the drop for *kept* in this process.

    The kept set is intersected with this thread's permitted set, because
    ``capset`` may not raise one; the data the child hands ``capset`` is
    built here so the child allocates nothing.
    """
    libc = _load_libc()
    libc.prctl.restype = ctypes.c_int
    keep = 0
    for name in kept:
        keep |= 1 << CAPABILITIES[_cap_name(name)]
    current = _capget(libc)
    permitted = current[1] if current else 0
    mask = keep & permitted
    hdr = _CapHeader(_LINUX_CAPABILITY_VERSION_3, 0)
    data = (_CapData * 2)()
    for i in (0, 1):
        word = (mask >> (32 * i)) & 0xFFFFFFFF
        data[i].effective = word
        data[i].permitted = word
        data[i].inheritable = 0
    drop = tuple(c for c in range(_last_cap() + 1) if not keep & (1 << c))
    return CapabilityDrop(keep, drop, libc, hdr, data)


#: Probe exit-code bits.
_BOUNDING_KEPT = 1
_PROCESS_KEPT = 2


def _verify(drop: CapabilityDrop) -> int:
    """Exit-code bits for what did NOT take effect (in the probe child)."""
    bits = 0
    prctl = drop._libc.prctl
    for cap in drop.drop:
        if prctl(PR_CAPBSET_READ, cap, 0, 0, 0) != 0:
            bits |= _BOUNDING_KEPT
            break
    sets = _capget(drop._libc)
    if sets is None or any(s & ~drop.keep_mask for s in sets):
        bits |= _PROCESS_KEPT
    return bits


def probe(drop: CapabilityDrop) -> Tuple[bool, bool]:
    """Run the drop once in a forked child and report what took effect.

    Returns ``(bounding_reduced, process_cleared)``.  The child runs the
    same two halves a real spawn runs, with no LSM transition between them
    (the transition does not change what ``CAP_SETPCAP`` and ``capset`` are
    allowed: under AppArmor the bounding drop runs before it, and
    ``//child`` does not mediate ``capset``; under SELinux the domain does
    not change before ``execve``).
    """
    with warnings.catch_warnings():
        # A multi-threaded fork: the child only makes syscalls and exits.
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        code = 4
        try:
            drop.before_transition()
            drop.after_transition()
            code = _verify(drop)
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    if code < 0 or code > 3:
        return False, False
    return not code & _BOUNDING_KEPT, not code & _PROCESS_KEPT


# ------------------------------------------------------------------ plan


@dataclass
class CapabilityPlan:
    """What one session's model-driven subprocesses get.

    Attributes:
        posture: One of :data:`POSTURES`.
        drop: The prepared drop, or ``None`` when nothing is installed.
        kept: Capabilities the profile keeps.
        ignored: Names in ``runtime_limits.capabilities`` that are no
            capability, so nothing was kept for them.
        reason: Why the posture is not ``dropped``, or ``""``.
    """

    posture: str
    drop: Optional[CapabilityDrop] = None
    kept: Tuple[str, ...] = ()
    ignored: Tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """The bootstrap-answer / runtime-aspect shape."""
        out: Dict[str, Any] = {"posture": self.posture}
        if self.kept:
            out["kept"] = list(self.kept)
        if self.ignored:
            out["ignored"] = list(self.ignored)
        if self.reason:
            out["reason"] = self.reason
        return out


_current: Optional[CapabilityPlan] = None


def current_posture() -> Optional[Dict[str, Any]]:
    """The posture this process recorded last, or ``None`` before any."""
    return None if _current is None else _current.as_dict()


def record_plan(plan: CapabilityPlan) -> CapabilityPlan:
    """Record *plan* as this process's posture and return it."""
    global _current
    _current = plan
    return plan


_BOUNDING_REASON = (
    "the bounding set could not be reduced (CAP_SETPCAP refused: the runner "
    "is not root, or the loaded LSM policy predates #1543); permitted, "
    "effective, inheritable and ambient are cleared and NO_NEW_PRIVS is set, "
    "so an exec cannot gain a capability")


def plan_for_session(
    value: Any,
    *,
    boundary_active: bool,
    prober: Optional[Callable[[CapabilityDrop], Tuple[bool, bool]]] = None,
) -> CapabilityPlan:
    """Decide and record this session's capability posture.

    Args:
        value: ``runtime_limits.capabilities`` (``None`` = ``none``).
        boundary_active: A kernel boundary (LSM ``//child`` transition) is
            installed for this session's subprocesses.
        prober: Test seam; :func:`probe` by default.

    Returns:
        The recorded :class:`CapabilityPlan`.
    """
    if not boundary_active:
        return record_plan(CapabilityPlan(
            POSTURE_UNCONFINED, reason="no kernel boundary, so nothing is dropped"))
    ignored = tuple(unknown_capabilities(
        value if isinstance(value, (list, tuple)) else ()))
    if ignored:
        logger.warning(
            "capabilities: ignoring unknown runtime_limits.capabilities "
            "names %s; they are not kept", list(ignored))
    mode, kept = resolve(value)
    if mode == MODE_INHERIT:
        logger.warning(
            "capabilities: runtime_limits.capabilities is 'inherit' -- "
            "model-driven subprocesses keep the runner's capability sets, "
            "behind the LSM boundary alone")
        return record_plan(CapabilityPlan(
            POSTURE_INHERIT, reason="runtime_limits.capabilities: inherit"))
    try:
        drop = prepare(kept)
        bounding, process = (prober or probe)(drop)
    except (OSError, AttributeError) as exc:
        reason = f"the capability drop could not be prepared: {exc}"
        logger.warning("capabilities: %s", reason)
        return record_plan(CapabilityPlan(
            POSTURE_ABSENT, kept=kept, ignored=ignored, reason=reason))
    if bounding and process:
        logger.info(
            "capabilities: model-driven subprocesses run with %s",
            ", ".join(kept) if kept else "no capabilities")
        return record_plan(CapabilityPlan(
            POSTURE_DROPPED, drop=drop, kept=kept, ignored=ignored))
    if process:
        logger.warning("capabilities: %s", _BOUNDING_REASON)
        return record_plan(CapabilityPlan(
            POSTURE_PARTIAL, drop=drop, kept=kept, ignored=ignored,
            reason=_BOUNDING_REASON))
    reason = ("capset refused (the loaded SELinux policy may predate "
              "#1543's 'process setcap')")
    if bounding:
        reason += "; the bounding set was reduced"
    logger.warning("capabilities: %s -- model-driven subprocesses keep the "
                   "runner's capabilities behind the LSM boundary alone", reason)
    return record_plan(CapabilityPlan(
        POSTURE_ABSENT, drop=drop if bounding else None, kept=kept,
        ignored=ignored, reason=reason))


def compose_child_preexec(
    lsm_transition: Callable[[], None],
    plan: CapabilityPlan,
) -> Callable[[], None]:
    """Wrap *lsm_transition* in the drop: bounding set first, then the
    transition, then the process sets and NO_NEW_PRIVS.

    Returns *lsm_transition* itself when nothing is installed.  The
    seccomp filter (:func:`shared.seccomp_filter.compose_child_preexec`)
    is composed after the returned callable.
    """
    drop = plan.drop
    if drop is None:
        return lsm_transition

    def preexec() -> None:
        drop.before_transition()
        lsm_transition()
        drop.after_transition()

    preexec.capability_drop = drop  # type: ignore[attr-defined]
    return preexec
