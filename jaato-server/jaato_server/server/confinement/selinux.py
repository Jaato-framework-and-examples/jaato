"""The SELinux backend (design ``docs/design/selinux-backend.md``).

Phase 2b: on a ready host it provisions a boundary by giving the
workspace a per-workspace MCS level (:mod:`.selinux_levels`), labelling
the tree once (:mod:`.selinux_labels`), and naming the two fixed domains
the runner and its children run in at that level.  Nothing is loaded per
session: the policy module is installed once by the operator (§9).

The checks follow design §10, first failing precondition wins:

1. Linux, SELinux mounted, ``libselinux`` loadable (ctypes; no new Python
   dependency).
2. SELinux is not disabled.
3. The policy defines jaato's runner domain, and the module is at least
   :data:`REQUIRED_POLICY_VERSION`.  The version is read from a marker type
   the module declares (``jaato_policy_v<N>_t``): asking the kernel whether
   a context is valid needs no privilege, where ``semodule -l`` needs root.
4. MCS is enabled (per-workspace isolation is by category).
5. This process may start a runner: its SELinux user and role are
   authorized for the runner domain, it may transition there, and the
   runner domain may use the interpreter as an entrypoint.

The transition permission alone proves nothing on the targeted policy:
``unconfined_t`` holds ``process transition`` to every domain, so phase 0
found it passing with the module's own rule removed.  The role and
entrypoint checks are the halves only jaato's module grants.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from jaato_server.server.confinement import selinux_labels
from jaato_server.server.confinement.base import Boundary, ConfinementHandle
from jaato_server.server.confinement.selinux_levels import LabelStamp, LevelTable
from jaato_server.server.confinement_id import confinement_id as _confinement_id
from jaato_server.server.confinement_id import session_tmpdir
from jaato_server.shared.lsm_label import (
    BACKEND_SELINUX,
    AvDecision,
    load_libselinux,
    parse_selinux_context,
    selinux_domain_permissive,
    selinux_host_enforcing,
)

logger = logging.getLogger(__name__)

#: The policy module version this build needs (design §4).  v2: the
#: isolated domains and the agent-config and prompts types (phase 3).
REQUIRED_POLICY_VERSION = 3

#: The type ``jaato.fc`` gives ``~/.jaato`` itself (search only).
USER_DIR_TYPE = "jaato_user_dir_t"

#: jaato's runner domain, as a context the kernel can validate.
RUNNER_PROBE_CONTEXT = "system_u:system_r:jaato_runner_t:s0"

RUNNER_DOMAIN = "jaato_runner_t"
CHILD_DOMAIN = "jaato_child_t"

#: An isolated subagent's sub-runner, and its read-only variant (design
#: §5.3).  Flat, as the AppArmor sub-profile is: no child domain, so a
#: subprocess would stay in the same domain (and the policy grants it
#: nothing to exec).
ISOLATED_DOMAIN = "jaato_isolated_t"
ISOLATED_RO_DOMAIN = "jaato_isolated_ro_t"

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

    def file_context(self, path: str) -> Optional[str]:
        fn = self._lib.getfilecon
        fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
        fn.restype = ctypes.c_int
        out = ctypes.c_char_p()
        if fn(path.encode("utf-8"), ctypes.byref(out)) < 0 or not out.value:
            return None
        return out.value.decode("utf-8", "replace")

    def allowed(
        self, source: str, target: str, tclass: str, perm: str,
    ) -> Optional[bool]:
        """Does the policy allow *perm*?  ``None`` when it cannot be asked."""
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
        cls = lib.string_to_security_class(tclass.encode("ascii"))
        bit = lib.string_to_av_perm(cls, perm.encode("ascii")) if cls else 0
        if not bit:
            return None
        avd = AvDecision()
        if lib.security_compute_av_flags(
            source.encode("utf-8"), target.encode("utf-8"),
            cls, bit, ctypes.byref(avd),
        ) != 0:
            return None
        return bool(avd.allowed & bit)

    def set_file_context(self, path: str, context: str) -> None:
        """``lsetfilecon``: label *path* itself, never a symlink's target."""
        fn = self._lib.lsetfilecon
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = ctypes.c_int
        if fn(path.encode("utf-8"), context.encode("utf-8")) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"lsetfilecon {context}: {os.strerror(err)}", path)

    def link_context(self, path: str) -> Optional[str]:
        """``lgetfilecon``: *path*'s own label, ``None`` when unreadable."""
        fn = self._lib.lgetfilecon
        fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
        fn.restype = ctypes.c_int
        out = ctypes.c_char_p()
        if fn(path.encode("utf-8"), ctypes.byref(out)) < 0 or not out.value:
            return None
        return out.value.decode("utf-8", "replace")


def _default_kernel() -> Optional[_Kernel]:
    lib = load_libselinux()
    return _Kernel(lib) if lib is not None else None


class SELinuxBackend:
    """SELinux confinement: per-workspace MCS level, fixed jaato domains."""

    name = BACKEND_SELINUX

    def __init__(
        self,
        *,
        kernel_factory: Callable[[], Optional[_Kernel]] = _default_kernel,
        host_enforcing: Callable[[], Optional[bool]] = selinux_host_enforcing,
        system: Callable[[], str] = platform.system,
        mount_present: Optional[Callable[[], bool]] = None,
        interpreter: Callable[[], str] = lambda: os.path.realpath(sys.executable),
        levels: Optional[LevelTable] = None,
        domain_permissive: Callable[[str], Optional[bool]] = selinux_domain_permissive,
    ) -> None:
        self._kernel_factory = kernel_factory
        self._host_enforcing = host_enforcing
        self._system = system
        self._mount_present = mount_present or _selinux_mounted
        self._interpreter = interpreter
        self._readiness: Optional[Readiness] = None
        self._levels = levels
        self._domain_permissive = domain_permissive

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
        reason = _policy_problem(kernel) or _transition_problem(
            kernel, self._interpreter(),
        )
        return Readiness(reason is None, reason, enforcing)

    def is_available(self) -> bool:
        return self.host_readiness().ready

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self.host_readiness().reason

    def host_facts(self, home: str) -> Dict[str, Optional[str]]:
        """What ``jaato-doctor`` reports about a ready host (design §9).

        Read from this process: the doctor and the daemon share a host and
        a policy, but a daemon started another way may hold another context.

        Returns:
            ``mode`` (enforcing / permissive), ``runner_domain`` (whether
            the policy has made ``jaato_runner_t`` permissive), the
            interpreter and its label, ``~/.jaato`` and its label
            (``None`` where one could not be read), and ``policy_version``:
            the module version readiness found the marker for, since a host
            reaching this method has passed that check.
        """
        kernel = self._kernel_factory()
        enforcing = self.host_readiness().enforcing
        interpreter = self._interpreter()
        user_dir = os.path.join(home, ".jaato")
        permissive = self._domain_permissive(RUNNER_PROBE_CONTEXT)
        return {
            "mode": None if enforcing is None else ("enforcing" if enforcing else "permissive"),
            "runner_domain": None if permissive is None else (
                "permissive" if permissive else "enforcing"),
            "interpreter": interpreter,
            "interpreter_label": kernel.file_context(interpreter) if kernel else None,
            "user_dir": user_dir,
            "user_dir_label": (
                kernel.link_context(user_dir)
                if kernel and os.path.lexists(user_dir) else None),
            "policy_version": str(REQUIRED_POLICY_VERSION),
        }

    def _level_table(self) -> LevelTable:
        if self._levels is None:
            self._levels = LevelTable()
        return self._levels

    def confinement_id_for_boundary(
        self, boundary: Boundary, domain: str = RUNNER_DOMAIN,
    ) -> str:
        """Slug plus a digest of what the runner may do (design §5.2).

        Nothing is rendered under SELinux, so the digest covers the inputs
        that change the boundary: policy version, domain, level, managed,
        private ``/tmp``.  The domain is part of it because an isolated
        sub-runner shares its parent's level and is still a different
        boundary.  Allocates the workspace's level on first use.
        """
        workspace = os.path.realpath(boundary.workspace_path)
        level = self._level_table().level_for(workspace)
        domain_part = "" if domain == RUNNER_DOMAIN else f" domain={domain}"
        return _confinement_id(
            workspace_root=workspace, config_root=boundary.config_root,
            rendered_body=(
                f"selinux policy=v{REQUIRED_POLICY_VERSION} level={level} "
                f"managed={boundary.managed} private_tmp={boundary.private_tmp_dir or ''}"
                f"{domain_part}"
            ),
        )

    def provision(
        self, session_id: str, boundary: Boundary,
    ) -> Optional[ConfinementHandle]:
        """Level the workspace, label it once, and name the domains.

        ``None`` on failure, logged with the cause: an unsafe root, a
        label that would not apply, an unreadable own context.  A caller
        that required confinement then refuses the session.
        """
        return self._provision(session_id, boundary, RUNNER_DOMAIN, CHILD_DOMAIN)

    def provision_isolated(
        self, session_id: str, boundary: Boundary, *, read_only: bool,
    ) -> Optional[ConfinementHandle]:
        """An isolated subagent's sub-runner, at its parent's level (§5.3).

        The sub-runner works in its parent's workspace, so it runs at that
        workspace's level; what isolates it is the domain.  Flat: the child
        label is its own label.  *read_only* is the
        ``isolated_read_only_workspace`` tightening.  The
        ``isolated_workspace_subpath`` tightening has no SELinux form and
        is refused by the caller before this is reached.
        """
        domain = ISOLATED_RO_DOMAIN if read_only else ISOLATED_DOMAIN
        return self._provision(session_id, boundary, domain, domain)

    def _provision(
        self, session_id: str, boundary: Boundary, domain: str, child_domain: str,
    ) -> Optional[ConfinementHandle]:
        """The steps both entry points share; *domain* names the boundary."""
        if not self.is_available():
            logger.error("SELinux provision for %s: backend unavailable (%s)",
                         session_id, self.unavailable_reason)
            return None
        kernel = self._kernel_factory()
        own = parse_selinux_context(kernel.current_context() if kernel else None)
        if kernel is None or own is None:
            logger.error("SELinux provision for %s: own context unreadable", session_id)
            return None
        workspace = os.path.realpath(boundary.workspace_path)
        try:
            selinux_labels.check_root(workspace)
            level = self._level_table().level_for(workspace)
            plan = selinux_labels.Plan(
                workspace=workspace, level=level, managed=boundary.managed,
                private_tmp_dir=(os.path.realpath(boundary.private_tmp_dir)
                                 if boundary.private_tmp_dir else None),
            )
            self._ensure_labelled(kernel, plan)
        except (OSError, ValueError) as exc:
            logger.error("SELinux provision for %s at %s failed: %s",
                         session_id, workspace, exc)
            return None
        label = f"{own.user}:{own.role}:{domain}:{level}"
        permissive = self._domain_permissive(label)
        handle = ConfinementHandle(
            backend=BACKEND_SELINUX,
            label=label,
            confinement_id=self.confinement_id_for_boundary(boundary, domain),
            child_label=f"{own.user}:{own.role}:{child_domain}:{level}",
            grants=_grants(plan, label, domain, child_domain),
            complain=(permissive is True or self.host_readiness().enforcing is False),
        )
        # The session tmpdir, before the spawn: the runner has no add_name
        # in /tmp and no write on user_tmp_t, so a directory it did not get
        # labelled here is one it cannot use (phase 2b kernel run: bootstrap
        # refused on a user_tmp_t tmpdir).  A failure refuses the session.
        try:
            self.prepare_session_tmpdir(
                handle, session_tmpdir(session_id, handle.confinement_id))
        except OSError as exc:
            logger.error("SELinux provision for %s: session tmpdir not "
                         "labelled: %s", session_id, exc)
            return None
        return handle

    def _ensure_labelled(self, kernel: "_Kernel", plan: "selinux_labels.Plan") -> None:
        """Walk and label unless the stamp and the root's label say done."""
        table = self._level_table()
        stamp = table.stamp(plan.workspace)
        want = selinux_labels.file_context(plan.base_type, plan.level)
        if (stamp is not None
                and stamp == LabelStamp(REQUIRED_POLICY_VERSION, plan.managed,
                                        plan.private_tmp_dir, stamp.files)
                and kernel.link_context(plan.workspace) == want):
            return
        table.forget_stamp(plan.workspace)
        selinux_labels.precreate_authored_dirs(plan.workspace)
        if plan.private_tmp_dir:
            os.makedirs(plan.private_tmp_dir, exist_ok=True)
        count = selinux_labels.apply(plan, kernel.set_file_context)
        table.record_stamp(plan.workspace, LabelStamp(
            REQUIRED_POLICY_VERSION, plan.managed, plan.private_tmp_dir, count))
        logger.info("SELinux: labelled %d entries of %s at %s (%s)",
                    count, plan.workspace, plan.level, plan.base_type)

    def prepare_session_tmpdir(self, handle: ConfinementHandle, path: str) -> None:
        """Create the session tmpdir and its boundary parent, as ``jaato_tmp_t``.

        Under ``/tmp`` the runner has no ``add_name`` (the host's ``/tmp``
        is not its own), so the daemon makes both directories and labels
        them at the boundary's level before the runner starts.
        """
        kernel = self._kernel_factory()
        ctx = parse_selinux_context(handle.label)
        if kernel is None or ctx is None:
            raise OSError(f"cannot label {path}: no libselinux or bad label {handle.label!r}")
        parent = os.path.dirname(path)
        os.makedirs(path, exist_ok=True)
        target = selinux_labels.file_context(selinux_labels.TMP_TYPE, ctx.level)
        for entry in (parent, path):
            kernel.set_file_context(entry, target)

    def prepare_runner_log(self, handle: ConfinementHandle, path: str) -> None:
        """Create an isolated sub-runner's log as ``jaato_runner_log_t``.

        The read-only isolated domain may write nothing in the workspace,
        its log included, unless the log carries a type of its own that
        it may append to (phase 3 kernel run: no log at all). Created here,
        before the spawn, with the mode the spawner would use; the
        spawner's ``O_CREAT`` then finds it.
        """
        kernel = self._kernel_factory()
        ctx = parse_selinux_context(handle.label)
        if kernel is None or ctx is None:
            raise OSError(f"cannot label {path}: no libselinux or bad label {handle.label!r}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
        kernel.set_file_context(path, selinux_labels.file_context(
            selinux_labels.RUNNER_LOG_TYPE, ctx.level))

    def release(self, handle: ConfinementHandle) -> None:
        """Nothing to unload: labels persist with the workspace (design §6)."""
        return None


def _grants(plan: "selinux_labels.Plan", label: str,
            domain: str, child_domain: str) -> Dict[str, Any]:
    """The diagnostics record (#1326), SELinux-shaped (design §8)."""
    return {
        "backend": BACKEND_SELINUX,
        "domain": domain,
        "child_domain": child_domain,
        "level": plan.level,
        "label": label,
        "labelled_roots": [plan.workspace],
        "workspace_type": plan.base_type,
        "policy_version": REQUIRED_POLICY_VERSION,
        "exec_scope": "unscoped",
    }


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


def _transition_problem(kernel: _Kernel, interpreter: str) -> Optional[str]:
    """Check 5: this process may start a runner in jaato's domain.

    The runner keeps the daemon's SELinux user and role (``setexeccon``
    changes the type and level only), so the role must be authorized for
    the runner domain.  Phase 0 showed why that matters: a daemon from a
    login shell is ``unconfined_u:unconfined_r``, one under systemd is
    ``system_u:system_r``.
    """
    own = kernel.current_context()
    if own is None:
        return "this process's own context could not be read"
    parsed = parse_selinux_context(own)
    if parsed is None:
        return f"this process's own context ({own}) could not be parsed"
    target = f"{parsed.user}:{parsed.role}:{RUNNER_DOMAIN}:s0"
    if not kernel.context_valid(target):
        return (
            f"SELinux user and role {parsed.user}:{parsed.role} are not "
            f"authorized for {RUNNER_DOMAIN}"
        )
    if kernel.allowed(own, target, "process", "transition") is not True:
        return f"this process ({own}) may not transition into {RUNNER_DOMAIN}"
    exe = kernel.file_context(interpreter)
    if exe is None:
        return f"the interpreter's label could not be read ({interpreter})"
    if kernel.allowed(target, exe, "file", "entrypoint") is not True:
        return (
            f"{RUNNER_DOMAIN} may not use the interpreter {interpreter} "
            f"({exe}) as an entrypoint"
        )
    return None


def _selinux_mounted() -> bool:
    import os

    return os.path.isfile(os.path.join(SELINUX_MOUNT, "enforce"))
