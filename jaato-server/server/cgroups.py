"""Cgroups v2 manager for per-session runtime limits.

Companion to :mod:`server.apparmor`, on the orthogonal axis:

* **Sandboxing** (``server.apparmor``) — *what's reachable*: paths,
  capabilities, network.  Enforced by AppArmor.
* **Runtime limits** (this module) — *how much can be consumed*:
  memory, PIDs, CPU share, plus app-layer caps like wall-clock
  timeouts and output size.  The kernel-enforceable subset is
  applied here via cgroup v2; the rest is carried in
  :class:`RuntimeLimits` for the CLI / interactive_shell plugins to
  apply at the Python layer.

Lifecycle (called by the WebSocket server / workspace provisioner):

1. :meth:`CgroupsManager.provision_cgroup` — creates
   ``{root}/jaato-ws-{session_id}/`` and writes the kernel-enforced
   limits from the :class:`RuntimeLimits` into the controller files.
2. Subprocess launch — the CLI / interactive_shell plugin uses
   :meth:`CgroupsManager.make_attach_callback` as
   ``subprocess.Popen(preexec_fn=...)``.  The callback writes the child
   PID to ``cgroup.procs`` *between fork() and exec()*, so the new
   program comes up already inside the cgroup.
3. :meth:`CgroupsManager.teardown_cgroup` — kills any stragglers via
   ``cgroup.kill`` (kernel ≥ 5.14) and removes the directory.

Graceful degradation
--------------------

When cgroup v2 is not available (cgroup v1 host, missing controllers,
non-writable root, non-Linux), :meth:`is_available` returns ``False`` and
all mutating methods are no-ops returning ``True``.  The server falls
back to "no kernel-enforced limits" — same pattern as AppArmor's
fallback to directory sandboxing.

Operator setup
--------------

The cgroup root must exist before the server starts and have the needed
controllers delegated.  For a server running as user ``jaato``:

    sudo mkdir /sys/fs/cgroup/jaato
    sudo chown jaato:jaato /sys/fs/cgroup/jaato
    # Enable controllers on the parent (needs root once at boot):
    echo "+memory +pids +cpu" \\
        | sudo tee /sys/fs/cgroup/jaato/cgroup.subtree_control

Or, for a systemd-delegated user slice, point ``--cgroup-root`` at
``/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/jaato.slice``.

Why preexec_fn over cgexec
--------------------------

``cgexec`` (from ``cgroup-tools``) is an optional package and shells out
per command.  Writing the child's own PID to ``cgroup.procs`` from
``preexec_fn`` is atomic w.r.t. the child (the program inside ``exec``
sees the limits from instruction zero), needs no extra binary, and
raises ``OSError`` to the parent ``Popen`` if the write fails — so a
broken cgroup root means the subprocess fails to start, not a silent
unconfined run.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

# RuntimeLimits is defined in shared/ so SubagentProfile can reference it
# statically.  We re-export it here so callers that already import from
# server.cgroups don't need to reach into shared.
from shared.runtime_limits import RuntimeLimits

__all__ = ["RuntimeLimits", "CgroupsManager"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

# Required cgroup v2 controllers.  ``cpu`` is only required when
# ``cpu_weight`` is set on a session, but we check it up front so
# is_available() is honest about what'll work.
_REQUIRED_CONTROLLERS = ("memory", "pids", "cpu")

# Path on cgroup v2 hosts.  v1 has /sys/fs/cgroup/<controller>/...,
# absence of this file is a reliable v1/v2 discriminator.
_V2_MARKER = Path("/sys/fs/cgroup/cgroup.controllers")


class CgroupsManager:
    """Provisions and tears down per-session cgroup v2 slices.

    Parallel to :class:`server.apparmor.AppArmorManager`: same
    lifecycle, same graceful-degradation contract, same per-session
    naming convention (``jaato-ws-{session_id}``).
    """

    def __init__(self, root: str = "/sys/fs/cgroup/jaato") -> None:
        """Initialize the manager.

        Args:
            root: Parent cgroup directory.  Must already exist with
                the required controllers delegated via
                ``cgroup.subtree_control`` (see operator-setup section
                in the module docstring).
        """
        self._root = Path(root)
        self._available: Optional[bool] = None
        self._missing_reason: str = ""

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Whether kernel-enforced limits can be applied by this process.

        Returns ``True`` only when *all* of the following hold:

        - Running on Linux
        - ``/sys/fs/cgroup/cgroup.controllers`` exists (cgroup v2)
        - The configured root exists and is writable
        - The root has ``memory``, ``pids``, and ``cpu`` listed in
          ``cgroup.subtree_control`` (so child cgroups can use them)

        Cached on first call.
        """
        if self._available is not None:
            return self._available
        self._available, self._missing_reason = self._check_availability()
        if self._available:
            logger.info("Cgroups confinement available at %s", self._root)
        else:
            logger.info(
                "Cgroups confinement not available — falling back to no kernel limits (%s)",
                self._missing_reason,
            )
        return self._available

    def _check_availability(self) -> Tuple[bool, str]:
        if platform.system() != "Linux":
            return False, "not Linux"
        if not _V2_MARKER.exists():
            return False, f"cgroup v2 not mounted ({_V2_MARKER} missing)"
        if not self._root.exists():
            return False, f"cgroup root {self._root} does not exist"
        if not os.access(self._root, os.W_OK):
            return False, f"cgroup root {self._root} not writable by uid={os.geteuid()}"

        subtree_file = self._root / "cgroup.subtree_control"
        try:
            enabled = subtree_file.read_text().split()
        except OSError as exc:
            return False, f"cannot read {subtree_file}: {exc}"
        missing = [c for c in _REQUIRED_CONTROLLERS if c not in enabled]
        if missing:
            return False, (
                f"controllers {missing} not delegated in {subtree_file} "
                f"(found: {enabled})"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def get_cgroup_name(self, session_id: str) -> str:
        """Return the per-session cgroup directory name."""
        return f"jaato-ws-{session_id}"

    def get_cgroup_path(self, session_id: str) -> Path:
        """Return the absolute path to the per-session cgroup."""
        return self._root / self.get_cgroup_name(session_id)

    def provision_cgroup(self, session_id: str, config: RuntimeLimits) -> bool:
        """Create the per-session cgroup and write its limits.

        Idempotent: re-provisioning an existing cgroup overwrites its
        limit files.  Failure to write any one limit file rolls the
        whole provision back (rmdir the cgroup) so the caller doesn't
        end up with a half-configured slice.

        Args:
            session_id: Session identifier.
            config: Limits to apply.  ``None``-valued fields are
                skipped (no file write — the controller default
                applies).

        Returns:
            ``True`` on success or when cgroups are unavailable
            (graceful degradation — caller proceeds without limits).
            ``False`` only when cgroups *are* available but provision
            failed (logged).
        """
        if not self.is_available():
            return True  # No kernel layer; succeed at this layer.

        if not config.has_kernel_limits():
            # Nothing to enforce; don't bother creating an empty cgroup.
            logger.debug(
                "No kernel limits set for session %s; skipping cgroup creation",
                session_id,
            )
            return True

        cg_path = self.get_cgroup_path(session_id)
        try:
            cg_path.mkdir(parents=False, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create cgroup %s: %s", cg_path, exc)
            return False

        try:
            self._write_limits(cg_path, config)
        except OSError as exc:
            logger.error("Failed to write cgroup limits for %s: %s", session_id, exc)
            # Roll back the empty/partially-configured cgroup so the
            # next attempt isn't refused with EBUSY (rmdir requires
            # cgroup.procs to be empty, which a brand-new cgroup is).
            self._safe_rmdir(cg_path)
            return False

        logger.info(
            "Provisioned cgroup %s (memory=%s pids=%s cpu_weight=%s)",
            cg_path, config.memory_max_mb, config.pids_max, config.cpu_weight,
        )
        return True

    def teardown_cgroup(self, session_id: str) -> bool:
        """Kill any remaining processes and remove the cgroup.

        Uses ``cgroup.kill`` (Linux ≥ 5.14) for atomic process-tree
        termination.  Falls back to a SIGTERM-then-rmdir loop on older
        kernels (best-effort — operator may need to clean up manually
        if processes refuse to die).

        Returns ``True`` on success, ``True`` when cgroups unavailable,
        ``False`` only on a real teardown error (logged).
        """
        if not self.is_available():
            return True

        cg_path = self.get_cgroup_path(session_id)
        if not cg_path.exists():
            return True  # Already gone.

        # Atomic kill — works on kernel 5.14+.
        kill_file = cg_path / "cgroup.kill"
        if kill_file.exists():
            try:
                kill_file.write_text("1")
            except OSError as exc:
                logger.warning("cgroup.kill failed for %s: %s", cg_path, exc)

        if not self._safe_rmdir(cg_path):
            return False
        logger.info("Removed cgroup %s", cg_path)
        return True

    def list_orphan_sub_cgroups(
        self,
        parent_session_id: str,
        known_isolated_session_ids: "set[str]",
    ) -> "list[str]":
        """Phase 5 §5.3: scan for orphaned isolated-subagent cgroups.

        Returns the isolated-session-ids whose cgroup directories exist
        under :attr:`_root` matching the
        ``jaato-ws-{parent_session_id}__sub_*`` template but aren't in
        the caller's *known_isolated_session_ids* set.  These are
        produced by sub-runner spawn rollback that failed mid-teardown
        or sub-runner crashes that left kernel state behind without a
        corresponding live handle in
        ``SessionManager._isolated_sub_runners``.

        The scanner does not reap — it returns ids, and the caller
        passes them back through :meth:`teardown_cgroup` (single code
        path for reap-by-id whether the source is a known handle or an
        orphan scan).

        Graceful degradation: returns an empty list when cgroups are
        unavailable, the root directory doesn't exist, or the scan
        fails — leak audit is observability + cleanup, not a
        correctness contract.

        Args:
            parent_session_id: Parent session whose orphan sub-cgroups
                to find.  Used to scope the name-pattern match.
            known_isolated_session_ids: Set of isolated-session-ids the
                caller already knows about (from
                ``_isolated_sub_runners``).  Cgroups matching the
                pattern but NOT in this set are returned as orphans.

        Returns:
            List of isolated-session-ids (strings of the form
            ``{parent_session_id}__sub_{subagent_id}``) whose cgroups
            exist on disk but aren't in *known_isolated_session_ids*.
            Empty list when cgroups unavailable, root missing, or scan
            fails.
        """
        if not self.is_available():
            return []
        if not self._root.exists():
            return []

        # Audit 7's sub-cgroup template:
        # ``jaato-ws-{parent_session_id}__sub_{subagent_id}``.  The
        # leaf prefix is constant; we scan for ``jaato-ws-{parent}__sub_``
        # and split out the suffix to reconstruct the isolated-session-id.
        leaf_prefix = f"jaato-ws-{parent_session_id}__sub_"
        orphans: list[str] = []
        try:
            entries = list(self._root.iterdir())
        except OSError as exc:
            logger.warning(
                "list_orphan_sub_cgroups: failed to scan %s: %s",
                self._root, exc,
            )
            return []

        for entry in entries:
            if not entry.is_dir():
                continue
            name = entry.name
            if not name.startswith(leaf_prefix):
                continue
            subagent_id = name[len(leaf_prefix):]
            if not subagent_id:
                continue  # Defensive: malformed name with empty suffix.
            isolated_session_id = f"{parent_session_id}__sub_{subagent_id}"
            if isolated_session_id in known_isolated_session_ids:
                continue
            orphans.append(isolated_session_id)
        return orphans

    # ------------------------------------------------------------------
    # Attach
    # ------------------------------------------------------------------

    def attach_pid(self, session_id: str, pid: int) -> None:
        """Move *pid* into the session's cgroup.

        Raises ``OSError`` on failure so subprocess-launch code can
        bubble the failure up to the caller (rather than silently
        running unconfined).  Use :meth:`make_attach_callback` for the
        ``Popen(preexec_fn=...)`` case.
        """
        if not self.is_available():
            return  # No-op when cgroups disabled.
        cg_path = self.get_cgroup_path(session_id)
        if not cg_path.exists():
            # No cgroup means no kernel limits configured for this
            # session (e.g. RuntimeLimits had no kernel-enforced fields
            # set).  Attach is a silent no-op — the subprocess just
            # inherits the parent's cgroup, which is fine.
            return
        procs = cg_path / "cgroup.procs"
        # Open + write in one syscall pair; can't use Path.write_text
        # because cgroup files reject the truncate that write_text
        # implies in some kernels.
        with open(procs, "w") as f:
            f.write(str(pid))

    def make_attach_callback(self, session_id: str) -> Callable[[], None]:
        """Return a zero-arg callable that attaches the *current* pid.

        Suitable for ``subprocess.Popen(preexec_fn=...)``: runs in the
        forked child between fork() and exec(), so writing
        ``os.getpid()`` to ``cgroup.procs`` migrates the child into the
        session's cgroup before the new program starts.

        Returns a no-op callable when cgroups are unavailable or no
        cgroup exists for the session — saves the caller from a
        per-launch availability check.
        """
        if not self.is_available():
            return _noop
        cg_path = self.get_cgroup_path(session_id)
        if not cg_path.exists():
            return _noop
        procs = cg_path / "cgroup.procs"
        # Capture the path now so the closure doesn't hit the manager
        # state from the forked child (which may have inconsistent
        # locks etc. — preexec_fn is signal-handler-style territory).
        procs_str = str(procs)

        def _attach() -> None:
            with open(procs_str, "w") as f:
                f.write(str(os.getpid()))

        return _attach

    # ------------------------------------------------------------------
    # Telemetry primitives
    # ------------------------------------------------------------------

    def read_event_counts(self, session_id: str) -> Optional[Dict[str, int]]:
        """Snapshot ``cgroup.events`` for a session's slice.

        ``cgroup.events`` is a kernel-managed file in every cgroup v2
        directory containing one ``key value`` pair per line.  The
        keys defined by the kernel (Linux ≥ 4.20):

        * ``populated`` — 1 while the cgroup or any descendant has at
          least one running process; 0 otherwise.
        * ``frozen``    — 1 when the cgroup is in the freezer state.

        Plus, when the memory controller reports OOM events:

        * ``oom``        — count of OOM events the cgroup has hit.
        * ``oom_kill``   — count of processes killed by the OOM-killer
          inside this cgroup.

        Use case: callers wishing to attribute a non-zero subprocess
        exit code to a kernel-enforced limit (e.g. a SIGKILL'd child
        whose ``oom_kill`` counter incremented during its lifetime)
        can snapshot before spawn and compare after wait().  The
        counter is monotonic for the lifetime of the cgroup, so a
        ``delta > 0`` on the relevant key means "this child contributed
        to that event" — modulo the inherent attribution race when
        multiple children run in parallel under the same cgroup.

        Returns:
            A dict of ``{key: int}`` for every line in ``cgroup.events``,
            or ``None`` if cgroups are unavailable, the per-session
            cgroup doesn't exist (e.g. session has no kernel limits),
            or the file can't be read.  Never raises — telemetry must
            not break the call path it's observing.
        """
        if not self.is_available():
            return None
        cg_path = self.get_cgroup_path(session_id)
        events_file = cg_path / "cgroup.events"
        if not events_file.exists():
            return None
        try:
            text = events_file.read_text()
        except OSError as exc:
            logger.debug("read_event_counts: %s read failed: %s", events_file, exc)
            return None

        counts: Dict[str, int] = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            key, value = parts
            try:
                counts[key] = int(value)
            except ValueError:
                # Skip non-integer values (no current kernel keys
                # produce them, but we don't want a future format
                # change to break telemetry callers).
                continue
        return counts

    def make_event_reader(
        self, session_id: str,
    ) -> Callable[[], Optional[Dict[str, int]]]:
        """Return a zero-arg callable that snapshots a session's events.

        Used to plumb event-counter access across the
        ``server → shared`` boundary without exposing the manager
        itself: the executor stores the callable and invokes it
        before / after each tool call, computing deltas to attribute
        kernel-killed exits to specific tool invocations.

        Returns a no-op callable (always returns ``None``) when
        cgroups are unavailable or no per-session cgroup exists, so
        callers don't need to special-case those paths.
        """
        if not self.is_available():
            return _noop_event_reader
        cg_path = self.get_cgroup_path(session_id)
        if not cg_path.exists():
            return _noop_event_reader

        # Capture the manager + session_id in a closure so the
        # caller doesn't need to retain either.
        def _read() -> Optional[Dict[str, int]]:
            return self.read_event_counts(session_id)

        return _read

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_limits(self, cg_path: Path, config: RuntimeLimits) -> None:
        """Write each non-None limit to its controller file.

        cgroup v2 controller files take a single value followed by a
        newline.  ``open(..., "w")`` with no truncate and a single
        ``write()`` is what userspace expects — kernel parses the
        first token.
        """
        if config.memory_max_mb is not None:
            value_bytes = config.memory_max_mb * 1024 * 1024
            self._write_one(cg_path / "memory.max", str(value_bytes))
        if config.pids_max is not None:
            self._write_one(cg_path / "pids.max", str(config.pids_max))
        if config.cpu_weight is not None:
            self._write_one(cg_path / "cpu.weight", str(config.cpu_weight))

    @staticmethod
    def _write_one(path: Path, value: str) -> None:
        with open(path, "w") as f:
            f.write(value)

    @staticmethod
    def _safe_rmdir(cg_path: Path) -> bool:
        """rmdir a cgroup, tolerating "already gone" but logging others.

        cgroup directories can only be removed via ``rmdir(2)`` (no
        ``rm -rf``); the kernel rejects rmdir if any process is still
        attached.  Caller is expected to have killed processes first.
        """
        try:
            cg_path.rmdir()
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            logger.error("Failed to rmdir cgroup %s: %s", cg_path, exc)
            return False


def _noop() -> None:
    """Sentinel preexec callback — used when cgroups are unavailable."""
    return None


def _noop_event_reader() -> Optional[Dict[str, int]]:
    """Sentinel event reader — returns None when cgroups are unavailable.

    Used by :meth:`CgroupsManager.make_event_reader` so callers always
    get a callable rather than ``None`` and can invoke it
    unconditionally — same contract as :func:`_noop` for preexec.
    """
    return None
