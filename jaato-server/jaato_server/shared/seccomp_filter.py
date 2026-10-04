"""A seccomp-bpf filter for model-driven subprocesses (#1503).

The LSM a confined session wears (AppArmor ``//child``, SELinux
``jaato_child_t``) decides what a tool subprocess may TOUCH: paths,
capabilities, sockets, ``change_profile``.  It does not decide which kernel
entry points the payload may REACH.  A kernel bug reachable before the LSM
hook (``bpf``, ``io_uring_setup``, ``unshare(CLONE_NEWUSER)``, ``keyctl``,
``userfaultfd``, ...) is reachable from inside the boundary.  This module is
the second fence: a deny-list filter installed in the forked child of every
model-driven subprocess, between the LSM transition and ``exec``.

Where it runs
-------------
Never in the runner.  The runner builds the filter ONCE per session, in the
parent, through libseccomp (which resolves syscall numbers for the native
architecture and emits the architecture check), exports it as a raw BPF
program, and keeps it in a ctypes buffer.  The forked child then makes two
``prctl`` calls and nothing else::

    prctl(PR_SET_NO_NEW_PRIVS, 1)
    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog)

So nothing in the child allocates through libseccomp after a fork of a
threaded process, and the per-spawn cost is two syscalls.  No ``TSYNC``: the
forked child has one thread.  The filter is inherited by everything the
payload forks and can only be tightened.

What it denies
--------------
:data:`FAMILIES` is the deny-list, grouped so a stage can allow one family
back (``runtime_limits.seccomp_allow: [ptrace]``).  A denied call answers
``EPERM`` (a tool fails with a readable error) except:

- ``clone3`` answers ``ENOSYS``: its flags live in a struct seccomp cannot
  read, so libc is told the call does not exist and falls back to
  ``clone``, whose flags are filtered;
- ``clone`` and ``unshare`` are denied only when a ``CLONE_NEW*`` flag is
  set, so ordinary process and thread creation is untouched;
- ``personality`` is allowed only for the default persona (``0``).

A syscall of a FOREIGN architecture (an i386 ``int 0x80`` from an x86_64
process, an x32 call) kills the process: libseccomp's bad-arch action, set to
``KILL_PROCESS``.  ``EPERM`` would leave a 32-bit binary failing every call
in a loop; no toolchain issues a foreign-arch syscall on purpose.

``io_uring_*`` answers ``EPERM``, not ``ENOSYS``: since Linux 6.6 the
``kernel.io_uring_disabled`` sysctl refuses ``io_uring_setup`` with
``EPERM``, and Docker's default profile refuses it the same way, so every
runtime that probes io_uring (libuv, liburing users) already falls back on
it.  libuv falls back on any ``io_uring_setup`` failure; Python's asyncio and
tokio do not use io_uring by default.

Posture
-------
:func:`plan_for_session` decides one of four postures and records it in this
process (:func:`current_posture`), where the runtime aspect and the
diagnostics probe read it:

========== =================================================================
filter     the filter is installed in every model-driven subprocess
off        the profile said ``seccomp: off`` (announced at WARNING)
absent     a kernel boundary is active and no filter could be built
           (libseccomp missing, kernel without seccomp); WARNING, or every
           spawn refused when confinement is REQUIRED
unconfined no kernel boundary, so no filter: one is not invented
========== =================================================================

Stdlib only, with no jaato imports beyond the stdlib, so the runner can load
it before plugin discovery.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import logging
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------- vocabulary

#: ``runtime_limits.seccomp`` values.
MODE_DEFAULT = "default"
MODE_OFF = "off"
MODES = (MODE_DEFAULT, MODE_OFF)

#: Postures recorded on the session (:func:`current_posture`).
POSTURE_FILTER = "filter"
POSTURE_OFF = "off"
POSTURE_ABSENT = "absent"
POSTURE_UNCONFINED = "unconfined"
POSTURES = (POSTURE_FILTER, POSTURE_OFF, POSTURE_ABSENT, POSTURE_UNCONFINED)

#: The env vars that make a kernel boundary REQUIRED.  Read in the runner,
#: which inherits the daemon's environment; a missing filter is then a
#: refused spawn rather than a WARNING.
REQUIRE_ENV_VARS = ("JAATO_REQUIRE_CONFINEMENT", "JAATO_REQUIRE_APPARMOR")
_TRUTHY = ("1", "true", "yes", "on")

# Linux UAPI constants.
PR_SET_SECCOMP = 22
PR_GET_SECCOMP = 21
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2

# libseccomp actions (seccomp.h).
SCMP_ACT_KILL_PROCESS = 0x80000000
SCMP_ACT_KILL = 0x00000000
SCMP_ACT_ALLOW = 0x7FFF0000


def SCMP_ACT_ERRNO(code: int) -> int:  # noqa: N802 - libseccomp's own name
    """The libseccomp action "fail the call with *code*"."""
    return 0x00050000 | (code & 0x0000FFFF)


# libseccomp ``enum scmp_compare`` and ``enum scmp_filter_attr``.
SCMP_CMP_NE = 1
SCMP_CMP_MASKED_EQ = 7
SCMP_FLTATR_ACT_BADARCH = 2
SCMP_FLTATR_CTL_NNP = 3

# ``CLONE_NEW*`` flags.  ``CLONE_NEWTIME`` (0x80) shares its bit with the
# exit-signal byte of ``clone()``'s flags, so it is filtered on ``unshare``
# only; ``clone()`` cannot request it.
CLONE_NEWNS = 0x00020000
CLONE_NEWCGROUP = 0x02000000
CLONE_NEWUTS = 0x04000000
CLONE_NEWIPC = 0x08000000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
CLONE_NEWNET = 0x40000000
CLONE_NEWTIME = 0x00000080
_CLONE_NEW_FLAGS = (CLONE_NEWNS, CLONE_NEWCGROUP, CLONE_NEWUTS, CLONE_NEWIPC,
                    CLONE_NEWUSER, CLONE_NEWPID, CLONE_NEWNET)
_UNSHARE_NEW_FLAGS = _CLONE_NEW_FLAGS + (CLONE_NEWTIME,)

#: Which argument of ``clone()`` carries the flags.  s390 swaps the first
#: two; everything else jaato runs on passes them first.
_CLONE_FLAGS_ARG = {"s390": 1, "s390x": 1}


@dataclass(frozen=True)
class Family:
    """One allow-back-able group of denied syscalls.

    Attributes:
        name: The name a profile writes in ``seccomp_allow``.
        syscalls: Syscalls denied outright with ``EPERM``.
        note: One line for ``explain runtime``.
    """

    name: str
    syscalls: Tuple[str, ...]
    note: str


#: The deny-list.  Started from the families Docker's default profile also
#: blocks, minus what a build/test toolchain needs.  A syscall this
#: architecture does not have is skipped when the filter is compiled.
FAMILIES: Dict[str, Family] = {f.name: f for f in (
    Family("kernel", ("init_module", "finit_module", "delete_module",
                      "kexec_load", "kexec_file_load", "reboot", "swapon",
                      "swapoff", "acct", "syslog"),
           "module loading, kexec, reboot, swap, process accounting, syslog"),
    Family("mount", ("mount", "umount2", "pivot_root", "move_mount",
                     "open_tree", "fsopen", "fsmount", "fsconfig", "fspick",
                     "mount_setattr"),
           "the mount API, old and new"),
    Family("namespaces", ("setns",),
           "setns; unshare/clone with a CLONE_NEW* flag; clone3 answers "
           "ENOSYS so libc falls back to clone"),
    Family("bpf", ("bpf",), "the bpf(2) syscall"),
    Family("perf", ("perf_event_open",), "perf events"),
    Family("userfaultfd", ("userfaultfd",), "userfaultfd"),
    Family("io_uring", ("io_uring_setup", "io_uring_enter",
                        "io_uring_register"),
           "io_uring (runtimes fall back on EPERM)"),
    Family("keyring", ("keyctl", "add_key", "request_key"),
           "the kernel keyring"),
    Family("fanotify", ("fanotify_init",), "fanotify"),
    Family("ptrace", ("ptrace", "process_vm_readv", "process_vm_writev",
                      "kcmp", "pidfd_getfd"),
           "debugging and cross-process memory access"),
    Family("handles", ("open_by_handle_at", "name_to_handle_at",
                       "lookup_dcookie", "uselib"),
           "file-handle tricks and uselib"),
    Family("personality", (),
           "personality(2) other than the default persona"),
)}

#: The enforcer ``explain runtime`` names for the two profile keys.
ENFORCER = "kernel (seccomp)"


class SeccompUnavailable(RuntimeError):
    """No filter can be built here: libseccomp missing or kernel lacks seccomp."""


class SeccompRefused(RuntimeError):
    """Raised in a forked child when a REQUIRED filter could not be built.

    The spawn then fails, as a failed LSM transition already makes it fail.
    """


def unknown_families(names: Optional[Iterable[str]]) -> List[str]:
    """Names in *names* that are not a :data:`FAMILIES` key, in order."""
    return [n for n in (names or ()) if n not in FAMILIES]


def confinement_required(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the environment makes a kernel boundary REQUIRED."""
    env = os.environ if env is None else env
    return any(str(env.get(v, "")).strip().lower() in _TRUTHY
               for v in REQUIRE_ENV_VARS)


# ------------------------------------------------------------- libseccomp


class _ArgCmp(ctypes.Structure):
    """``struct scmp_arg_cmp``."""

    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


class _SockFilter(ctypes.Structure):
    """``struct sock_filter``: one BPF instruction, 8 bytes."""

    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    """``struct sock_fprog``."""

    _fields_ = [("len", ctypes.c_ushort),
                ("filter", ctypes.POINTER(_SockFilter))]


#: Overridable by tests to simulate a missing library.
LIBSECCOMP_NAMES: Tuple[str, ...] = ("libseccomp.so.2",)


def _load_libseccomp() -> Any:
    """Load libseccomp, or raise :class:`SeccompUnavailable`."""
    names = list(LIBSECCOMP_NAMES)
    found = ctypes.util.find_library("seccomp")
    if found and found not in names:
        names.append(found)
    for name in names:
        try:
            lib = ctypes.CDLL(name, use_errno=True)
        except OSError:
            continue
        lib.seccomp_init.restype = ctypes.c_void_p
        lib.seccomp_init.argtypes = [ctypes.c_uint32]
        lib.seccomp_release.argtypes = [ctypes.c_void_p]
        lib.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_uint32]
        lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        lib.seccomp_rule_add_array.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
            ctypes.POINTER(_ArgCmp)]
        lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.seccomp_export_pfc.argtypes = [ctypes.c_void_p, ctypes.c_int]
        return lib
    raise SeccompUnavailable(
        f"libseccomp not found (tried {', '.join(names)}); install the "
        "distribution's libseccomp2 / libseccomp package")


def _libseccomp_version(lib: Any) -> str:
    try:
        class _V(ctypes.Structure):
            _fields_ = [("major", ctypes.c_uint), ("minor", ctypes.c_uint),
                        ("micro", ctypes.c_uint)]
        lib.seccomp_version.restype = ctypes.POINTER(_V)
        v = lib.seccomp_version().contents
        return f"{v.major}.{v.minor}.{v.micro}"
    except Exception:  # noqa: BLE001 - a version string is advisory
        return "unknown"


def kernel_supports_seccomp() -> bool:
    """Whether this kernel has seccomp (``PR_GET_SECCOMP`` does not EINVAL)."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_GET_SECCOMP, 0, 0, 0, 0) < 0:
        return ctypes.get_errno() != errno.EINVAL
    return True


def _rules(allow: frozenset) -> List[Tuple[int, str, Tuple[Tuple[int, int, int, int], ...]]]:
    """``(action, syscall, comparisons)`` for every rule not allowed back."""
    eperm = SCMP_ACT_ERRNO(errno.EPERM)
    rules: List[Tuple[int, str, Tuple[Tuple[int, int, int, int], ...]]] = []
    for fam in FAMILIES.values():
        if fam.name in allow:
            continue
        rules.extend((eperm, name, ()) for name in fam.syscalls)
    if "namespaces" not in allow:
        arg = _CLONE_FLAGS_ARG.get(platform.machine(), 0)
        rules.extend((eperm, "clone", ((arg, SCMP_CMP_MASKED_EQ, f, f),))
                     for f in _CLONE_NEW_FLAGS)
        rules.extend((eperm, "unshare", ((0, SCMP_CMP_MASKED_EQ, f, f),))
                     for f in _UNSHARE_NEW_FLAGS)
        rules.append((SCMP_ACT_ERRNO(errno.ENOSYS), "clone3", ()))
    if "personality" not in allow:
        rules.append((eperm, "personality", ((0, SCMP_CMP_NE, 0, 0),)))
    return rules


def _build_ctx(lib: Any, allow: frozenset) -> Any:
    ctx = lib.seccomp_init(SCMP_ACT_ALLOW)
    if not ctx:
        raise SeccompUnavailable("seccomp_init failed")
    try:
        if lib.seccomp_attr_set(ctx, SCMP_FLTATR_ACT_BADARCH,
                                SCMP_ACT_KILL_PROCESS) < 0:
            # libseccomp < 2.4 or a kernel without KILL_PROCESS: the
            # thread-kill is the same thing for a single-threaded child.
            lib.seccomp_attr_set(ctx, SCMP_FLTATR_ACT_BADARCH, SCMP_ACT_KILL)
        # The child sets NO_NEW_PRIVS itself; the exported program does not
        # depend on this, but say so rather than rely on a default.
        lib.seccomp_attr_set(ctx, SCMP_FLTATR_CTL_NNP, 0)
        for action, name, cmps in _rules(allow):
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr < 0:
                continue  # not a syscall on this architecture
            arr = (_ArgCmp * max(1, len(cmps)))(*[_ArgCmp(*c) for c in cmps])
            rc = lib.seccomp_rule_add_array(ctx, action, nr, len(cmps), arr)
            if rc < 0:
                raise SeccompUnavailable(
                    f"seccomp_rule_add({name}) failed: {os.strerror(-rc)}")
        return ctx
    except BaseException:
        lib.seccomp_release(ctx)
        raise


def _export(lib: Any, ctx: Any, exporter: str) -> bytes:
    fd = os.memfd_create("jaato-seccomp", 0)
    try:
        rc = getattr(lib, exporter)(ctx, fd)
        if rc < 0:
            raise SeccompUnavailable(f"{exporter} failed: {os.strerror(-rc)}")
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


@dataclass
class CompiledFilter:
    """A filter built in the parent, installable in a forked child.

    Attributes:
        allowed: The families allowed back.
        program: The raw BPF program (``struct sock_filter[]``).
        libseccomp: The libseccomp version that compiled it.
    """

    allowed: Tuple[str, ...]
    program: bytes
    libseccomp: str = "unknown"
    _buf: Any = field(default=None, repr=False)
    _prog: Any = field(default=None, repr=False)
    _prctl: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        count = len(self.program) // ctypes.sizeof(_SockFilter)
        if count == 0 or count > 0xFFFF:
            raise SeccompUnavailable(f"exported filter has {count} instructions")
        self._buf = (_SockFilter * count).from_buffer_copy(self.program)
        self._prog = _SockFprog(count, ctypes.cast(
            self._buf, ctypes.POINTER(_SockFilter)))
        libc = ctypes.CDLL(None, use_errno=True)
        self._prctl = libc.prctl

    def install(self) -> None:
        """``PR_SET_NO_NEW_PRIVS`` then the filter, in the CURRENT process.

        Called as (part of) a ``preexec_fn``, between ``fork`` and ``exec``.
        Never call it in the runner: the filter cannot be removed.

        Raises:
            OSError: either ``prctl`` failed; the spawn fails with it.
        """
        prctl = self._prctl
        if prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"PR_SET_NO_NEW_PRIVS: {os.strerror(err)}")
        if prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER,
                 ctypes.byref(self._prog), 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"PR_SET_SECCOMP: {os.strerror(err)}")


def compile_filter(allow: Iterable[str] = ()) -> CompiledFilter:
    """Build the deny-list filter, minus the families in *allow*.

    Raises:
        SeccompUnavailable: libseccomp is missing, the kernel has no
            seccomp, or libseccomp refused a rule.
    """
    allowed = frozenset(a for a in allow if a in FAMILIES)
    try:
        if not kernel_supports_seccomp():
            raise SeccompUnavailable(
                "this kernel has no seccomp (PR_GET_SECCOMP: EINVAL)")
        lib = _load_libseccomp()
        ctx = _build_ctx(lib, allowed)
        try:
            program = _export(lib, ctx, "seccomp_export_bpf")
        finally:
            lib.seccomp_release(ctx)
        return CompiledFilter(
            tuple(sorted(allowed)), program, _libseccomp_version(lib))
    except (OSError, AttributeError) as exc:
        # memfd_create refused, a libseccomp missing a symbol: no filter
        # can be built here, which is what SeccompUnavailable says.
        raise SeccompUnavailable(
            f"could not build the filter: {type(exc).__name__}: {exc}") from exc


def filter_pseudocode(allow: Iterable[str] = ()) -> str:
    """libseccomp's PFC rendering of the filter, for review and tests."""
    lib = _load_libseccomp()
    ctx = _build_ctx(lib, frozenset(a for a in allow if a in FAMILIES))
    try:
        return _export(lib, ctx, "seccomp_export_pfc").decode("utf-8", "replace")
    finally:
        lib.seccomp_release(ctx)


# ------------------------------------------------------------------ plan


@dataclass
class SeccompPlan:
    """What one session's model-driven subprocesses get.

    Attributes:
        posture: One of :data:`POSTURES`.
        installer: The zero-arg callable to run in the forked child after
            the LSM transition, or ``None`` when nothing is installed.
        allowed: Families allowed back (``filter`` posture).
        reason: Why the posture is not ``filter``, or ``""``.
        required: Whether a kernel boundary was required.
        libseccomp: The library version, when one compiled the filter.
    """

    posture: str
    installer: Optional[Callable[[], None]] = None
    allowed: Tuple[str, ...] = ()
    reason: str = ""
    required: bool = False
    libseccomp: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """The record / runtime-aspect shape."""
        out: Dict[str, Any] = {"posture": self.posture}
        if self.allowed:
            out["allowed_families"] = list(self.allowed)
        if self.reason:
            out["reason"] = self.reason
        if self.required:
            out["required"] = True
        if self.libseccomp:
            out["libseccomp"] = self.libseccomp
        if self.posture == POSTURE_ABSENT and self.required:
            out["spawns_refused"] = True
        return out


_current: Optional[SeccompPlan] = None


def current_posture() -> Optional[Dict[str, Any]]:
    """The posture this process recorded last, or ``None`` before any."""
    return None if _current is None else _current.as_dict()


def record_plan(plan: SeccompPlan) -> SeccompPlan:
    """Record *plan* as this process's posture and return it."""
    global _current
    _current = plan
    return plan


_record = record_plan


def _refuser(reason: str) -> Callable[[], None]:
    def refuse() -> None:
        raise SeccompRefused(
            f"seccomp filter required but unavailable: {reason}")
    return refuse


def plan_for_session(
    mode: Optional[str],
    allow: Optional[Sequence[str]],
    *,
    boundary_active: bool,
    required: Optional[bool] = None,
    compiler: Callable[[Iterable[str]], CompiledFilter] = compile_filter,
) -> SeccompPlan:
    """Decide and record this session's seccomp posture.

    Args:
        mode: ``runtime_limits.seccomp`` (``None`` = ``default``).
        allow: ``runtime_limits.seccomp_allow``; unknown names are ignored
            with a WARNING (the family then stays closed, the safe side).
        boundary_active: A kernel boundary (LSM ``//child`` transition) is
            installed for this session's subprocesses.
        required: Whether a kernel boundary is required; ``None`` reads
            :data:`REQUIRE_ENV_VARS`.
        compiler: Test seam for :func:`compile_filter`.

    Returns:
        The recorded :class:`SeccompPlan`.
    """
    if required is None:
        required = confinement_required()
    if not boundary_active:
        return _record(SeccompPlan(
            POSTURE_UNCONFINED, reason="no kernel boundary, so no filter",
            required=required))
    if (mode or MODE_DEFAULT) == MODE_OFF:
        logger.warning(
            "seccomp: runtime_limits.seccomp is 'off' -- model-driven "
            "subprocesses reach the whole syscall table (bpf, io_uring, "
            "unshare, keyctl, ...) behind the LSM boundary alone")
        return _record(SeccompPlan(
            POSTURE_OFF, reason="runtime_limits.seccomp: off", required=required))
    unknown = unknown_families(allow)
    if unknown:
        logger.warning(
            "seccomp: ignoring unknown runtime_limits.seccomp_allow "
            "families %s (known: %s); they stay denied",
            unknown, ", ".join(FAMILIES))
    allowed = tuple(sorted({a for a in (allow or ()) if a in FAMILIES}))
    try:
        compiled = compiler(allowed)
    except SeccompUnavailable as exc:
        reason = str(exc)
        if required:
            logger.error(
                "seccomp: %s, and kernel confinement is REQUIRED -- every "
                "model-driven subprocess will be refused", reason)
            return _record(SeccompPlan(
                POSTURE_ABSENT, installer=_refuser(reason), reason=reason,
                required=True))
        logger.warning(
            "seccomp: %s -- model-driven subprocesses run behind the LSM "
            "boundary with no syscall filter", reason)
        return _record(SeccompPlan(POSTURE_ABSENT, reason=reason))
    if allowed:
        logger.warning(
            "seccomp: families allowed back for this session: %s",
            ", ".join(allowed))
    logger.info(
        "seccomp: filter armed for model-driven subprocesses "
        "(%d instructions, libseccomp %s, allowed back: %s)",
        len(compiled.program) // 8, compiled.libseccomp,
        ", ".join(allowed) or "none")
    return _record(SeccompPlan(
        POSTURE_FILTER, installer=compiled.install, allowed=allowed,
        required=required, libseccomp=compiled.libseccomp))


def compose_child_preexec(
    lsm_transition: Callable[[], None],
    installer: Optional[Callable[[], None]],
) -> Callable[[], None]:
    """The ``preexec_fn`` step for a model-driven subprocess.

    The LSM transition first (AppArmor ``changeprofile`` / SELinux
    ``setexeccon``: both are refused once NO_NEW_PRIVS would matter, so
    they must come before it), then NO_NEW_PRIVS and the filter.  The
    subprocess plugins append the cgroup attach after this callable; a
    cgroup write is an ordinary ``open``/``write`` the filter allows.

    Returns *lsm_transition* itself when there is nothing to install.
    """
    if installer is None:
        return lsm_transition

    def preexec() -> None:
        lsm_transition()
        installer()

    preexec.seccomp_installer = installer  # type: ignore[attr-defined]
    return preexec
