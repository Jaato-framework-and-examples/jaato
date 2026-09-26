"""Subprocess-kernel notebook backend (design option 1c).

Runs each notebook in its OWN Python subprocess launched with
``cwd=workspace_root`` (``kernel_main``), so notebook code's ``os.getcwd()`` and
relative paths resolve in-workspace — without the process-global ``os.chdir`` the
framework forbids (core.py:915).  State persists in the kernel namespace.

This is the DEFAULT backend; ``plugin_configs.notebook.backend: "local"`` opts
back to the in-process ``LocalJupyterBackend``.

**``cwd`` was never a boundary** (issue #710).  It makes relative paths resolve
in-workspace and leaves absolute ones untouched, so a cell could read
``/etc/hostname`` — or spawn ``cat`` to do it — after ``cli`` had refused the
same path.  The kernel therefore establishes a real one at startup
(``kernel_sandbox.establish_containment``): AppArmor when the runner is
confined (the kernel inherits the profile through the ``ix`` exec rule), else a
:pep:`578` audit hook applying the same containment ``cli`` applies, else a
refusal to run cells at all.  This class states the same answer one layer up in
:meth:`SubprocessKernelBackend.execution_boundary` so ``NotebookPlugin`` can
refuse before a kernel is spawned, and passes the session's ``sandbox add`` /
``sandbox deny`` paths down on every cell so an authorization granted after the
kernel spawned is honoured.
"""

import ctypes
import ctypes.util
import datetime
import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import uuid
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from jaato_server.shared.session_context import get_workspace_root
from ...workspace_venv import (
    resolve_venv_path, ensure_workspace_venv, apply_venv_to_env, venv_python,
    kernel_import_dirs,
)
from ...workspace_home import resolve_home_path, apply_home_to_env
from ...jaato_tools_path import append_path_entry, jaato_tools_dir
from .base import NotebookBackend
from .. import kernel_protocol as proto
from ..kernel_sandbox import (
    BOUNDARY_APPARMOR,
    BOUNDARY_AUDIT,
    BOUNDARY_NONE,
    BOUNDARY_OPT_OUT,
    UNCONTAINED_OPT_IN_ENV,
    apparmor_enforced_profile,
    env_truthy,
    profile_can_leave_itself,
)
from ..types import (
    BackendCapabilities, CellOutput, ExecutionResult, ExecutionStatus,
    NotebookInfo, OutputType,
)

_DEFAULT_TIMEOUT_S = 300

logger = logging.getLogger(__name__)

# Bounded capture of the kernel process's OWN stderr (fd 2) — NOT cell
# stdout/stderr, which travels as ``stream`` frames.  Process stderr is where a
# kernel death leaves its evidence: a C-level ``Fatal Python error`` /
# ``Segmentation fault`` line, an uncaught harness traceback, an OOM message.
# The reader thread drains it continuously (so a full pipe never blocks the
# kernel) into a fixed-size ring, so the tail is available when the kernel dies
# without ever buffering unbounded output (issue #1275).
_STDERR_TAIL_CHUNKS = 16
_STDERR_CHUNK_BYTES = 4096
_STDERR_TAIL_CHARS = 2000


_PR_SET_PDEATHSIG = 1

# Resolve libc + prctl at MODULE IMPORT (before any fork).  The preexec_fn below
# runs in the post-fork/pre-exec window of a MULTITHREADED process, where only
# async-signal-safe work is allowed: import / dlopen / malloc there take the
# allocator/import locks and DEADLOCK the child if a concurrent thread held one
# at the fork instant (Python's subprocess docs warn against preexec_fn in
# multithreaded processes).  That deadlock was the parallel-tools BrokenPipe —
# the kernel never reached exec, so its pipe was never healthy.  Doing the
# CDLL/dlopen here, once, leaves the preexec_fn with nothing but the resolved
# prctl syscall.  None if libc can't be loaded → PDEATHSIG is simply skipped.
try:
    _LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6",
                        use_errno=True)
    _LIBC.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                            ctypes.c_ulong, ctypes.c_ulong]
    _LIBC.prctl.restype = ctypes.c_int
except Exception:
    _LIBC = None


def _set_pdeathsig() -> None:
    """preexec_fn: SIGKILL the kernel if the runner (parent) THREAD dies, so a
    crashed runner never orphans kernels (the jdtls/LSP precedent —
    feedback_runner_child_heavyweight_proc_needs_pdeathsig).

    MUST stay async-signal-safe: ONLY the pre-resolved prctl syscall, NO import /
    dlopen / malloc (those deadlock in the multithreaded post-fork window — the
    fork-safety fix).  ``_LIBC`` is loaded at module import above."""
    if _LIBC is not None:
        _LIBC.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)


def _popen_or_close(argv, close_on_failure=(), **popen_kwargs) -> subprocess.Popen:
    """``subprocess.Popen``, closing ``close_on_failure`` if the spawn fails.

    A kernel spawn can fail on purpose: a refused ``//child`` transition
    raises in the forked child and fails ``Popen`` (#1323).  The pipes
    opened for that kernel would otherwise leak.
    """
    try:
        return subprocess.Popen(argv, **popen_kwargs)
    except BaseException:
        for fd in close_on_failure:
            try:
                os.close(fd)
            except OSError:
                pass
        raise


class _Kernel:
    """A live kernel subprocess + its pipes + a per-kernel serialization lock.

    Attributes:
        announce_boundary: Whether the NEXT cell's result should carry the
            boundary this kernel established (issue #1012).  Armed by
            ``_spawn`` and cleared by the first ``execute`` that consumes it,
            so the model is told the tier once per kernel rather than on every
            cell — and told again after a respawn, which is the case a
            standing statement in the system prompt cannot cover.
    """

    def __init__(self, info: NotebookInfo, proc: subprocess.Popen,
                 wstream, rstream):
        self.info = info
        self.proc = proc
        self.wstream = wstream          # runner → kernel
        self.rstream = rstream          # kernel → runner
        self.lock = threading.Lock()
        self.announce_boundary = False
        # Ring of the kernel process's own stderr, for post-mortem diagnosis
        # (#1275).  Bounded by ``maxlen``, drained by ``_stderr_thread`` so a
        # full stderr pipe can never wedge the kernel.
        self._stderr_chunks: Deque[bytes] = deque(maxlen=_STDERR_TAIL_CHUNKS)
        self._stderr_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None

    def start_stderr_drain(self) -> None:
        """Continuously copy the kernel's stderr into a bounded ring.

        Started once, right after spawn.  The thread is a daemon and exits on
        its own when the kernel dies (stderr hits EOF) or ``close`` kills the
        process, so nothing has to stop it.  A drain that raises (fd already
        closed under a racing teardown) simply ends — the tail captured so far
        is what ``describe_death`` reports.
        """
        stream = self.proc.stderr
        if stream is None:
            return

        def _drain() -> None:
            try:
                while True:
                    chunk = stream.read(_STDERR_CHUNK_BYTES)
                    if not chunk:
                        break
                    with self._stderr_lock:
                        self._stderr_chunks.append(chunk)
            except Exception:  # noqa: BLE001 - a drained-through-teardown read
                pass

        self._stderr_thread = threading.Thread(
            target=_drain, name=f"nbkernel-stderr-{self.info.notebook_id}",
            daemon=True)
        self._stderr_thread.start()

    def stderr_tail_text(self, limit: int = _STDERR_TAIL_CHARS) -> str:
        """The last ``limit`` characters of captured kernel stderr, decoded.

        Bounded twice over: the ring caps the bytes retained, and this caps the
        rendered string, so a death reason cannot carry an unbounded payload
        into a tool result the model reads.
        """
        with self._stderr_lock:
            raw = b"".join(self._stderr_chunks)
        text = raw.decode("utf-8", errors="replace").strip()
        if len(text) > limit:
            text = "…" + text[-limit:]
        return text

    def describe_death(self) -> str:
        """A clean, bounded reason the kernel is no longer usable (#1275).

        Names the exit signal (a negative ``returncode`` is ``-N`` for signal
        ``N``) or exit code, plus the tail of the kernel's own stderr.  Joins
        the stderr drain briefly first: a kernel that just died may have stderr
        still buffered in the pipe, and the reason is only diagnosable if that
        tail is read before it is reported.
        """
        rc = self.proc.poll()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)
        tail = self.stderr_tail_text()
        if rc is None:
            head = "notebook kernel is unresponsive (exit status unknown)"
        elif rc < 0:
            try:
                signame = signal.Signals(-rc).name
            except (ValueError, KeyError):
                signame = "unknown"
            head = f"notebook kernel crashed (signal {-rc} {signame})"
        elif rc == 0:
            head = "notebook kernel exited (code 0, unexpected mid-session exit)"
        else:
            head = f"notebook kernel exited (code {rc})"
        return f"{head}: {tail}" if tail else head

    def close(self) -> None:
        try:
            proto.write_frame(self.wstream, {"type": proto.SHUTDOWN})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
        for s in (self.wstream, self.rstream):
            try:
                s.close()
            except Exception:
                pass


_KERNEL_MODULE = "jaato_server.shared.plugins.notebook.kernel_main"


def _kernel_argv(kernel_python: str, import_dirs: List[str]) -> List[str]:
    """The command that starts the kernel, before its own arguments.

    With no ``import_dirs`` (the runner's own interpreter, which already
    imports jaato) this is ``python -m kernel_main``.

    Under a workspace venv it is a ``-c`` bootstrap that appends
    ``import_dirs`` to ``sys.path`` and then runs the same module.  A venv
    interpreter cannot import jaato by itself: that was deliberately taken
    away from every tool-venv interpreter, because it made the daemon's
    installed ``jaato_server`` shadow the checkout a session was editing
    (#1322).  The kernel is the one process that needs it.

    The dirs travel in argv, not ``PYTHONPATH``, so a process a cell starts
    (``!python -m pytest``) inherits nothing from them.  Appended, not
    prepended, so a package the model installs into the venv still wins
    inside the kernel.
    """
    if not import_dirs:
        return [kernel_python, "-m", _KERNEL_MODULE]
    bootstrap = (
        "import runpy, sys; "
        f"sys.path.extend({list(import_dirs)!r}); "
        f"runpy.run_module({_KERNEL_MODULE!r}, run_name='__main__', "
        "alter_sys=True)"
    )
    return [kernel_python, "-c", bootstrap]


class SubprocessKernelBackend(NotebookBackend):
    """Per-notebook subprocess kernels, each rooted at ``workspace_root``."""

    BACKEND_NAME = "subprocess"

    def __init__(self) -> None:
        self._workspace_root: Optional[str] = None
        # Workspace-scoped venv path for the kernel interpreter (None/empty =
        # off).  When set, the kernel runs from this venv's python so the
        # model's in-notebook pip installs persist and imports resolve.
        # See shared/plugins/workspace_venv.py.
        self._workspace_venv: Optional[str] = None
        # Workspace-scoped HOME for the kernel subprocess (None/empty = off,
        # #1225).  Points HOME + XDG at ``<ws>/<home>`` so a cell's ~ writes
        # stay per-workspace.  See shared/plugins/workspace_home.py.
        self._workspace_home: Optional[str] = None
        self._kernels: Dict[str, _Kernel] = {}
        self._lock = threading.Lock()
        # The runner's tool executor (``ToolExecutor.execute`` shape:
        # (name, args) -> (ok, result)).  Wired by the plugin; serves the
        # kernel's cross-process tools.X() calls (PR 2).  None → tools.X()
        # in the kernel reports "no executor wired".
        self._tool_executor: Optional[
            Callable[[str, Dict], Tuple[bool, object]]] = None
        # Operator opt-out from filesystem containment (#710).  The sibling of
        # LocalJupyterBackend's allow_inprocess_exec, and a different question:
        # that one is "may cells run in the host process", this one is "may
        # cells reach outside the workspace".  Announced at WARNING.
        self._allow_uncontained = False
        # Extra readable roots an operator declared
        # (plugin_configs.notebook.allow_read_paths), handed to every kernel.
        self._allow_read_paths: Tuple[str, ...] = ()
        # Reads the session's live `sandbox add` / `sandbox deny` paths.  Wired
        # by the plugin from the PluginRegistry; the answer is re-sent with
        # every cell, so an authorization granted after a kernel spawned takes
        # effect on the next cell rather than the next kernel.
        self._sandbox_paths_fn: Optional[Callable[[], Dict[str, List[str]]]] = None
        # AppArmor //child transition for the kernel (#1323).  Installed by
        # ``NotebookPlugin.set_apparmor_child_transition_callback``, the same
        # callable ``cli`` runs in its preexec_fn.  ``None`` when the runner
        # is not confined.  Without it a confined runner's kernel would stay
        # in the base profile, which a cell can leave by itself.
        self._apparmor_child_transition: Optional[Callable[[], None]] = None

    def set_tool_executor(
        self, executor_fn: Callable[[str, Dict], Tuple[bool, object]]
    ) -> None:
        """Wire the runner-side tool executor that serves kernel tool_call frames
        (mirrors LocalJupyterBackend.inject_tools_module for the in-process case)."""
        self._tool_executor = executor_fn

    def set_sandbox_paths_fn(
        self, paths_fn: Optional[Callable[[], Dict[str, List[str]]]]
    ) -> None:
        """Wire the reader for the session's authorized / denied sandbox paths.

        Args:
            paths_fn: Returns ``{"read": [...], "write": [...], "deny": [...]}``
                as the registry holds them *now* (``sandbox add`` / ``sandbox
                deny``).  Called once per cell, because those lists are
                operator-mutable mid-session while a kernel outlives many
                cells.  ``None`` unwires it, leaving kernels contained to the
                workspace alone.
        """
        self._sandbox_paths_fn = paths_fn

    def set_apparmor_child_transition(
        self, callback: Optional[Callable[[], None]],
    ) -> None:
        """Wire the AppArmor ``//child`` transition kernels are spawned with (#1323).

        Args:
            callback: The zero-arg ``preexec_fn``-style callable built by
                ``server.apparmor.make_child_transition_callback``, or
                ``None`` to unwire it.  Applies to kernels spawned after the
                call; a live kernel keeps the profile it started in.
        """
        self._apparmor_child_transition = callback

    def _kernel_preexec(self) -> Callable[[], None]:
        """The ``preexec_fn`` a kernel is spawned with.

        The ``//child`` transition first, when one is wired, then the
        parent-death signal.  A transition that fails raises in the forked
        child, so ``Popen`` fails and no kernel starts: the fail-closed
        posture ``cli`` takes, because a kernel left in the base profile can
        unconfine itself.
        """
        transition = self._apparmor_child_transition
        if transition is None:
            return _set_pdeathsig

        def _preexec() -> None:
            transition()
            _set_pdeathsig()

        return _preexec

    def _sandbox_allow_block(self) -> Optional[Dict[str, List[str]]]:
        """The ``allow`` block to attach to an ``execute`` frame, or ``None``.

        Never lets a diagnostic failure end a cell: a provider that raises is
        reported as "no operator-granted paths", which is the containment the
        kernel already has rather than a widening of it.
        """
        if self._sandbox_paths_fn is None:
            return None
        try:
            block = self._sandbox_paths_fn() or {}
        except Exception:  # noqa: BLE001 - a broken provider must not fail cells
            return None
        return {key: list(block.get(key) or ()) for key in ("read", "write", "deny")}

    # ---- protocol: identity / lifecycle -------------------------------------

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.BACKEND_NAME, supports_gpu=False,
            supports_packages=True, is_async=False, requires_auth=False)

    def initialize(self, config: Optional[Dict] = None) -> None:
        """Take the workspace, the venv and the containment posture from config.

        Every key is optional and absent keys are LEFT ALONE rather than reset:
        ``NotebookPlugin.set_workspace_path`` re-initializes this backend with
        ``{"workspace_root": ...}`` alone once the workspace arrives (the #344
        flow), and that call must not drop the containment configuration the
        first ``initialize`` established.

        Config keys (all under ``plugin_configs.notebook``):
            workspace_root: The session workspace; kernels spawn with it as cwd
                and are contained to it.
            workspace_venv: Interpreter for the kernel, and a writable root
                (``!pip install`` writes there).
            allow_uncontained_exec: Operator opt-out from containment
                (``JAATO_NOTEBOOK_ALLOW_UNCONTAINED_EXEC`` beneath it).
            allow_read_paths: Extra readable roots outside the workspace.
        """
        if config:
            self._workspace_root = config.get("workspace_root") or self._workspace_root
            if "workspace_venv" in config:
                self._workspace_venv = config.get("workspace_venv")
            if "workspace_home" in config:
                self._workspace_home = config.get("workspace_home")
            if "allow_uncontained_exec" in config:
                self._allow_uncontained = bool(config.get("allow_uncontained_exec"))
            if "allow_read_paths" in config:
                self._allow_read_paths = tuple(config.get("allow_read_paths") or ())
        self._allow_uncontained = (
            self._allow_uncontained or env_truthy(UNCONTAINED_OPT_IN_ENV))

    def boundary_kind(self) -> str:
        """The tier a kernel spawned now would establish, strongest first.

        The ladder itself, in one place.  :meth:`execution_boundary` renders
        this into prose and ``NotebookPlugin`` renders it into the model's
        instructions through ``kernel_sandbox.boundary_notice`` (#1012), so
        the two cannot disagree about which tier is in force — before #1012
        the ladder was written out inline in ``execution_boundary`` and there
        was nowhere to ask for the answer by name.

        Mirrors ``kernel_sandbox.establish_containment``, which is the
        authority: it runs *in the kernel*, and its answer travels back on the
        READY frame and is recorded on ``NotebookInfo.boundary_kind``.  One
        rung of that ladder is deliberately absent here — a kernel whose
        interpreter provides no ``addaudithook`` degrades to
        ``BOUNDARY_NONE``, and only the kernel can know that, because
        ``workspace_venv`` may point at an interpreter this process is not.
        That is why the READY answer outranks this one rather than confirming
        it.

        Returns:
            A ``BOUNDARY_*`` constant.  ``BOUNDARY_NONE`` when there is no
            workspace to contain to, which is what
            :meth:`execution_boundary` turns into a refusal.
        """
        # The kernel runs in //child when the transition is wired, and
        # inherits the runner's profile otherwise.  The base profile is not a
        # boundary for a cell (it can unconfine itself), so without the
        # transition the kernel falls through to the audit hook (#1323).
        profile = apparmor_enforced_profile()
        if profile and (self._apparmor_child_transition is not None
                        or not profile_can_leave_itself(profile)):
            return BOUNDARY_APPARMOR
        if self._allow_uncontained:
            return BOUNDARY_OPT_OUT
        if not (get_workspace_root() or self._workspace_root):
            return BOUNDARY_NONE
        return BOUNDARY_AUDIT

    def execution_boundary(self) -> Tuple[bool, str]:
        """What bounds a cell's filesystem reach, strongest available first.

        Asked by ``NotebookPlugin`` before a cell is dispatched (issue #710),
        and answered here rather than in the kernel because the refusal is
        worth making before a kernel process exists.  The kernel re-decides
        the same question for itself at startup — it is the process that runs
        the code, and a boundary it could not install must stop cells there
        too, whatever this said.

        The decision is :meth:`boundary_kind`; this method only words it.

        Returns:
            ``(allowed, description)``.  Refuses only when there is no
            workspace to contain to, since an unrooted kernel cannot be
            spawned at all.
        """
        kind = self.boundary_kind()
        if kind == BOUNDARY_APPARMOR:
            profile = apparmor_enforced_profile()
            if self._apparmor_child_transition is not None:
                return True, (
                    f"AppArmor-enforced profile {profile}//child (the kernel "
                    "is started in the //child sub-profile)")
            return True, (
                f"AppArmor-enforced profile {profile} (inherited by the kernel)")
        if kind == BOUNDARY_OPT_OUT:
            return True, "operator opt-out (uncontained)"
        if kind == BOUNDARY_NONE:
            return False, (
                "Notebook execution refused: no workspace root is resolved, so "
                "cell code cannot be contained to one. Start the session with a "
                "workspace, or set "
                f"{UNCONTAINED_OPT_IN_ENV}=1 (notebook plugin config "
                "allow_uncontained_exec=true) to accept an uncontained notebook."
            )
        workspace = get_workspace_root() or self._workspace_root
        return True, f"audit-hook workspace containment ({workspace})"

    @staticmethod
    def _consume_boundary_announcement(kernel: "_Kernel") -> Optional[str]:
        """Take this kernel's one-shot boundary announcement, or ``None``.

        Returns the tier the KERNEL reported on its READY frame — not
        :meth:`boundary_kind`'s pre-spawn expectation — so the model is told
        what actually bounded the process that ran its cell.  Where the two
        disagree (a respawn on a host whose confinement changed, an
        interpreter with no ``addaudithook``), this is the one that is true.

        Returns ``None`` on every cell after the first, and on a kernel whose
        handshake reported no tier: an announcement that repeats is noise, and
        one that guesses is the second source of truth #1012 warns against.
        """
        if not kernel.announce_boundary:
            return None
        kernel.announce_boundary = False
        return kernel.info.boundary_kind

    def is_available(self) -> bool:
        return True

    def shutdown(self) -> None:
        with self._lock:
            kernels = list(self._kernels.values())
            self._kernels.clear()
        for k in kernels:
            k.close()

    # ---- protocol: notebooks ------------------------------------------------

    def create_notebook(self, name: str, gpu_enabled: bool = False) -> NotebookInfo:
        notebook_id = uuid.uuid4().hex[:12]
        info = NotebookInfo(
            notebook_id=notebook_id, name=name, backend=self.BACKEND_NAME,
            gpu_enabled=False,
            created_at=datetime.datetime.now().isoformat(),
        )
        self._spawn(info)
        return info

    def execute(self, notebook_id: str, code: str,
                timeout_seconds: Optional[int] = None) -> ExecutionResult:
        """Run one cell, recovering transparently from a kernel that has died.

        A kernel outlives many cells and can die BETWEEN them (an OOM kill, a
        native crash from a cell that had already returned, a delayed
        ``atexit`` fault).  Left unhandled, the next write to its closed stdin
        raised a raw ``BrokenPipeError`` on EVERY later cell, forever (#1275).

        So a kernel found dead before the cell runs is reaped — capturing WHY
        it died — and respawned ONCE, so this innocent cell runs on a live
        kernel rather than hitting the dead pipe.  The re-run is bounded: it
        goes through the normal read loop, which never respawns again, so a
        cell that deterministically crashes its kernel fails cleanly rather
        than looping.  Death discovered DURING the cell (a mid-execution
        crash) is handled in :meth:`_drive_cell` — the cell that killed the
        kernel reports its own failure and the notebook is left ready to
        respawn on the next call.
        """
        timeout = timeout_seconds or _DEFAULT_TIMEOUT_S
        kernel = self._get_or_create(notebook_id)
        if kernel.proc.poll() is not None:
            reason = self._reap_dead_kernel(notebook_id, kernel)
            logger.warning(
                "notebook kernel %s died before a cell ran; respawning (%s)",
                notebook_id, reason)
            try:
                kernel = self._get_or_create(notebook_id)
            except Exception as exc:  # noqa: BLE001 - report, never re-raise raw
                return self._kernel_died_result(
                    f"{reason}; kernel respawn failed: {exc}", None)
        return self._drive_cell(notebook_id, kernel, code, timeout)

    def _drive_cell(self, notebook_id: str, kernel: "_Kernel", code: str,
                    timeout: float) -> ExecutionResult:
        """Send one cell to a (believed-live) kernel and collect its result.

        The write is guarded: even after :meth:`execute`'s liveness check a
        kernel can die in the race before the frame lands, so a
        ``BrokenPipeError`` / ``OSError`` on the write is turned into a clean
        typed failure (reaping the kernel) rather than escaping to
        ``ai_tool_runner`` as a transport traceback (#1275).  A mid-execution
        death surfaces the same way through the ``EOFError`` branch below.
        """
        cell_id = uuid.uuid4().hex[:8]
        start = datetime.datetime.now()
        # Consumed once per kernel: whatever this cell does, the model has now
        # been told which boundary ran it (#1012).  Read before the cell runs
        # so a cell that is REFUSED by containment still carries the tier that
        # refused it — that is the case the announcement exists for.
        announce = self._consume_boundary_announcement(kernel)
        with kernel.lock:
            frame_out = {"type": proto.EXECUTE, "cell_id": cell_id, "code": code}
            # The session's `sandbox add` / `sandbox deny` paths ride EVERY
            # cell rather than the kernel's argv: a kernel outlives many cells
            # and those lists are operator-mutable mid-session, so an
            # authorization granted after the spawn would otherwise be invisible
            # until the notebook was recreated (#710).
            allow_block = self._sandbox_allow_block()
            if allow_block is not None:
                frame_out["allow"] = allow_block
            try:
                proto.write_frame(kernel.wstream, frame_out)
            except (BrokenPipeError, OSError):
                # The kernel died between cells and the write is what found it.
                reason = self._reap_dead_kernel(notebook_id, kernel)
                logger.warning(
                    "notebook kernel %s dead at cell write (%s)",
                    notebook_id, reason)
                return self._kernel_died_result(reason, announce)
            outputs: List[CellOutput] = []
            while True:
                if not self._wait_readable(kernel, timeout):
                    kernel.close()
                    with self._lock:
                        self._kernels.pop(notebook_id, None)
                    return ExecutionResult(
                        status=ExecutionStatus.CANCELLED, outputs=outputs,
                        error_name="TimeoutError",
                        error_message=f"cell exceeded {timeout}s; kernel killed",
                        boundary_kind=announce)
                try:
                    frame = proto.read_frame(kernel.rstream)
                except EOFError:
                    # Kernel died mid-execution.  Reap it (capturing WHY) and
                    # drop it so the next execute spawns a fresh one — the
                    # crashing cell reports its failure, the notebook is not
                    # bricked.
                    reason = self._reap_dead_kernel(notebook_id, kernel)
                    logger.warning(
                        "notebook kernel %s died mid-execution (%s)",
                        notebook_id, reason)
                    return self._kernel_died_result(reason, announce, outputs)
                result = self._process_frame(kernel, frame, outputs, start,
                                             announce)
                if result is not None:
                    return result

    def _process_frame(self, kernel: "_Kernel", frame: Dict, outputs:
                       List[CellOutput], start: datetime.datetime,
                       announce: Optional[str]) -> Optional[ExecutionResult]:
        """Fold one kernel→runner frame into the running cell.

        Returns an :class:`ExecutionResult` when the cell is done (a ``result``
        or ``error`` frame), or ``None`` to keep reading (streamed output, or a
        tool call answered in place).
        """
        ft = frame.get("type")
        if ft == proto.STREAM:
            otype = (OutputType.STDOUT if frame.get("name") == "stdout"
                     else OutputType.STDERR)
            outputs.append(CellOutput(otype, frame.get("text", "")))
            return None
        if ft == proto.RESULT:
            if frame.get("value") is not None:
                outputs.append(CellOutput(OutputType.RESULT, frame["value"]))
            kernel.info.execution_count = frame.get(
                "execution_count", kernel.info.execution_count)
            kernel.info.last_executed_at = start.isoformat()
            dur = (datetime.datetime.now() - start).total_seconds()
            return ExecutionResult(
                status=ExecutionStatus.COMPLETED, outputs=outputs,
                execution_count=kernel.info.execution_count,
                duration_seconds=dur, boundary_kind=announce)
        if ft == proto.ERROR:
            outputs.append(CellOutput(
                OutputType.ERROR, frame.get("traceback", "")))
            dur = (datetime.datetime.now() - start).total_seconds()
            return ExecutionResult(
                status=ExecutionStatus.FAILED, outputs=outputs,
                error_name=frame.get("ename"),
                error_message=frame.get("evalue"),
                traceback=frame.get("traceback"),
                duration_seconds=dur, boundary_kind=announce)
        if ft == proto.TOOL_CALL:
            # The cell is blocked mid-exec waiting for this; run the tool
            # runner-side and reply.  The loop then continues reading the
            # cell's remaining output / further tool calls / result.
            ok, payload = self._run_tool(
                frame.get("name", ""), frame.get("args") or {})
            proto.write_frame(kernel.wstream, {
                "type": proto.TOOL_RESULT,
                "call_id": frame.get("call_id"),
                "ok": ok,
                ("result" if ok else "error"): payload,
            })
        return None

    def _reap_dead_kernel(self, notebook_id: str, kernel: "_Kernel") -> str:
        """Reap a dead kernel, capture its death reason, and drop it (#1275).

        Waits (then kills, as a backstop) to avoid a zombie, reads the exit
        signal and the tail of the kernel's stderr into a human reason, and
        removes it from the live map so the next :meth:`execute` spawns a fresh
        one.  Only drops the entry if it is still the mapped kernel, so a
        concurrent respawn is not clobbered.  Never raises: a reap failing must
        not turn a clean error into a traceback.
        """
        try:
            kernel.proc.wait(timeout=2)
        except Exception:  # noqa: BLE001 - still running / already reaped
            try:
                kernel.proc.kill()
                kernel.proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
        reason = kernel.describe_death()
        with self._lock:
            if self._kernels.get(notebook_id) is kernel:
                self._kernels.pop(notebook_id, None)
        for stream in (kernel.wstream, kernel.rstream):
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        return reason

    @staticmethod
    def _kernel_died_result(
            reason: str, announce: Optional[str],
            outputs: Optional[List[CellOutput]] = None) -> ExecutionResult:
        """A clean, typed 'the kernel crashed' result — never a raw exception.

        Shaped like a normal failed cell (``status=failed``, ``error_name`` /
        ``error_message``) so a model reads it the way it reads any cell error,
        with the death signal and stderr tail in ``error_message`` (#1275).
        """
        return ExecutionResult(
            status=ExecutionStatus.FAILED, outputs=outputs or [],
            error_name="KernelDied", error_message=reason,
            boundary_kind=announce)

    def get_execution_status(self, notebook_id: str,
                             execution_id: Optional[str] = None) -> ExecutionResult:
        # Synchronous backend — execute() blocks to completion, so there is no
        # pending async status to poll.
        return ExecutionResult(status=ExecutionStatus.COMPLETED)

    def get_variables(self, notebook_id: str) -> Dict[str, str]:
        kernel = self._kernels.get(notebook_id)
        if kernel is None:
            return {}
        with kernel.lock:
            proto.write_frame(kernel.wstream, {"type": proto.VARIABLES})
            if not self._wait_readable(kernel, 10):
                return {}
            try:
                frame = proto.read_frame(kernel.rstream)
            except EOFError:
                return {}
            return frame.get("variables", {}) if frame.get("type") == proto.VARIABLES else {}

    def reset_notebook(self, notebook_id: str) -> str:
        kernel = self._kernels.get(notebook_id)
        if kernel is not None:
            with kernel.lock:
                proto.write_frame(kernel.wstream, {"type": proto.RESET})
                if self._wait_readable(kernel, 10):
                    try:
                        proto.read_frame(kernel.rstream)
                    except EOFError:
                        pass
                kernel.info.execution_count = 0
        return notebook_id

    def delete_notebook(self, notebook_id: str) -> None:
        with self._lock:
            kernel = self._kernels.pop(notebook_id, None)
        if kernel is not None:
            kernel.close()

    def list_notebooks(self) -> List[NotebookInfo]:
        return [k.info for k in self._kernels.values()]

    # ---- internals ----------------------------------------------------------

    def _get_or_create(self, notebook_id: str) -> _Kernel:
        kernel = self._kernels.get(notebook_id)
        if kernel is None:
            info = NotebookInfo(
                notebook_id=notebook_id, name=notebook_id,
                backend=self.BACKEND_NAME,
                created_at=datetime.datetime.now().isoformat())
            kernel = self._spawn(info)
        return kernel

    def _run_tool(self, name: str, args: Dict) -> Tuple[bool, object]:
        """Run a kernel-originated tool call inside the trusted-bridge permission
        scope (so it inherits the notebook_execute approval, exactly like the
        in-process bridge) and return a JSON-safe ``(ok, result|error)``."""
        if self._tool_executor is None:
            return False, "notebook tools bridge: no tool executor wired"
        from jaato_server.shared.ai_tool_runner import trusted_bridge_context
        try:
            with trusted_bridge_context():
                ok, result = self._tool_executor(name, args)
        except Exception as exc:  # noqa: BLE001 — report to the cell
            return False, f"{type(exc).__name__}: {exc}"
        if not ok:
            return False, result if isinstance(result, str) else repr(result)
        # The result rides a JSON frame; non-JSON (e.g. binary tool output) →
        # repr for now (base64 framing of binary tool results is a follow-up).
        try:
            json.dumps(result)
            return True, result
        except (TypeError, ValueError):
            return True, repr(result)

    def _spawn(self, info: NotebookInfo) -> _Kernel:
        # The session workspace comes from the session_context ContextVar — the
        # SAME deterministic per-session source file_edit uses (get_workspace_root,
        # seeded at runner bootstrap via _set_runner_workspace_context, reliable
        # for pool-slot runners post-#344).  The plugin-config value
        # (set_workspace_path) is only a fallback for non-session contexts (unit
        # tests).  NO os.getcwd() fallback — that silently chdir'd kernels to the
        # daemon launch dir when neither was set (the bug the peer's A/B exposed).
        workspace = get_workspace_root() or self._workspace_root
        if not workspace:
            raise RuntimeError(
                "notebook subprocess kernel: no workspace_root resolved "
                "(session_context ContextVar and plugin config both unset)")
        # Kernel interpreter: the runner base python by default, or the
        # workspace venv's python when configured (so the model's in-notebook
        # pip installs persist and later imports resolve).  The tool-venv
        # cannot import jaato on its own (#1322), so a venv kernel is handed
        # jaato's import dirs at launch -- see ``_kernel_argv``.
        kernel_python = sys.executable
        kernel_env: Optional[Dict[str, str]] = None
        import_dirs: List[str] = []
        venv_path = resolve_venv_path(self._workspace_venv, workspace)
        if venv_path:
            ensure_workspace_venv(venv_path)
            kernel_python = venv_python(venv_path)
            kernel_env = os.environ.copy()
            apply_venv_to_env(kernel_env, venv_path)
            import_dirs = kernel_import_dirs()

        # Workspace HOME (#1225): point HOME + XDG at ``<ws>/<home>`` so a
        # cell's ~ writes land per-workspace.  The daemon created the
        # directory before spawn.  ``kernel_env`` may still be None (no venv),
        # so materialise it from os.environ before redirecting -- otherwise
        # Popen would inherit the daemon's HOME wholesale.
        home_path = resolve_home_path(self._workspace_home, workspace)
        if home_path:
            if kernel_env is None:
                kernel_env = os.environ.copy()
            apply_home_to_env(kernel_env, home_path)
        # jaato's own introspection tools (#1273), appended to PATH, so a
        # cell's ``!jaato-doctor`` resolves the way a cli command's does.
        # Only materialises an env when there is a directory to add, so a
        # kernel with nothing to change still inherits the environment as is.
        tools_dir = jaato_tools_dir()
        if tools_dir:
            if kernel_env is None:
                kernel_env = os.environ.copy()
            append_path_entry(kernel_env, tools_dir)

        r2k_r, r2k_w = os.pipe()   # runner → kernel
        k2r_r, k2r_w = os.pipe()   # kernel → runner
        # Containment argv (#710).  The venv is a WRITABLE root because
        # in-notebook `!pip install` writes there and it may sit outside the
        # workspace; operator-declared read paths are read-only.  `--uncontained`
        # is passed only when the operator opted out, so the kernel's default
        # is the contained one however it is launched.
        containment_args = []
        for path in self._allow_read_paths:
            containment_args += ["--allow-read", str(path)]
        if venv_path:
            containment_args += ["--allow-write", str(venv_path)]
        if self._allow_uncontained:
            containment_args.append("--uncontained")
        proc = _popen_or_close(
            [*_kernel_argv(kernel_python, import_dirs),
             "--workspace-root", workspace,
             "--read-fd", str(r2k_r), "--write-fd", str(k2r_w),
             *containment_args],
            pass_fds=(r2k_r, k2r_w),
            cwd=workspace,
            env=kernel_env,
            preexec_fn=self._kernel_preexec(),
            # Capture the kernel's OWN stderr into a bounded ring so a death is
            # diagnosable after the fact (#1275).  Cell stdout/stderr does NOT
            # come this way — it is framed over the k2r pipe as ``stream``
            # frames — so this carries only harness-level crash evidence
            # (a native fault line, an uncaught traceback), never cell output.
            stderr=subprocess.PIPE,
            close_on_failure=(r2k_r, r2k_w, k2r_r, k2r_w),
        )
        os.close(r2k_r)
        os.close(k2r_w)
        wstream = os.fdopen(r2k_w, "wb", buffering=0)
        rstream = os.fdopen(k2r_r, "rb", buffering=0)
        kernel = _Kernel(info, proc, wstream, rstream)
        kernel.start_stderr_drain()
        # Handshake: the kernel sends READY after chdir + containment + namespace
        # init, and names the boundary it actually established.  Recorded on the
        # NotebookInfo so an operator reading `notebook_list` sees what bounds
        # the kernel rather than inferring it from configuration (#710).
        if self._wait_readable(kernel, 30):
            try:
                ready = proto.read_frame(rstream)   # READY
                info.boundary = ready.get("boundary_description") or None
                # The KIND travelled on this frame all along and was thrown
                # away; it is what names the tier's consequences to the model
                # (#1012).  Taking it from here rather than re-deriving it is
                # what makes a kernel that came up under a different posture —
                # a respawn on a host whose AppArmor state changed — visible
                # instead of silently contradicting the system prompt.
                info.boundary_kind = ready.get("boundary") or None
            except EOFError:
                pass
        # Announce the boundary on this kernel's FIRST cell.  Armed per spawn,
        # so a respawn re-announces (issue #1012).
        kernel.announce_boundary = True
        with self._lock:
            self._kernels[info.notebook_id] = kernel
        return kernel

    @staticmethod
    def _wait_readable(kernel: "_Kernel", timeout: float) -> bool:
        """True if the kernel's read stream has data within ``timeout`` (and the
        process is alive)."""
        if kernel.proc.poll() is not None:
            return True  # process exited — let read_frame raise EOFError
        ready, _, _ = select.select([kernel.rstream.fileno()], [], [], timeout)
        return bool(ready)
