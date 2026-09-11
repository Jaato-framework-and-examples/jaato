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
import os
import select
import signal
import subprocess
import sys
import threading
import uuid
from typing import Callable, Dict, List, Optional, Tuple

from shared.session_context import get_workspace_root
from ...workspace_venv import (
    resolve_venv_path, ensure_workspace_venv, apply_venv_to_env, venv_python,
)
from .base import NotebookBackend
from .. import kernel_protocol as proto
from ..kernel_sandbox import (
    UNCONTAINED_OPT_IN_ENV,
    apparmor_enforced_profile,
    env_truthy,
)
from ..types import (
    BackendCapabilities, CellOutput, ExecutionResult, ExecutionStatus,
    NotebookInfo, OutputType,
)

_DEFAULT_TIMEOUT_S = 300


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


class _Kernel:
    """A live kernel subprocess + its pipes + a per-kernel serialization lock."""

    def __init__(self, info: NotebookInfo, proc: subprocess.Popen,
                 wstream, rstream):
        self.info = info
        self.proc = proc
        self.wstream = wstream          # runner → kernel
        self.rstream = rstream          # kernel → runner
        self.lock = threading.Lock()

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
            if "allow_uncontained_exec" in config:
                self._allow_uncontained = bool(config.get("allow_uncontained_exec"))
            if "allow_read_paths" in config:
                self._allow_read_paths = tuple(config.get("allow_read_paths") or ())
        self._allow_uncontained = (
            self._allow_uncontained or env_truthy(UNCONTAINED_OPT_IN_ENV))

    def execution_boundary(self) -> Tuple[bool, str]:
        """What bounds a cell's filesystem reach, strongest available first.

        Asked by ``NotebookPlugin`` before a cell is dispatched (issue #710),
        and answered here rather than in the kernel because the refusal is
        worth making before a kernel process exists.  The kernel re-decides
        the same question for itself at startup — it is the process that runs
        the code, and a boundary it could not install must stop cells there
        too, whatever this said.

        Returns:
            ``(allowed, description)``.  Refuses only when there is no
            workspace to contain to, since an unrooted kernel cannot be
            spawned at all.
        """
        profile = apparmor_enforced_profile()
        if profile:
            return True, f"AppArmor-enforced profile {profile} (inherited by the kernel)"
        if self._allow_uncontained:
            return True, "operator opt-out (uncontained)"
        workspace = get_workspace_root() or self._workspace_root
        if not workspace:
            return False, (
                "Notebook execution refused: no workspace root is resolved, so "
                "cell code cannot be contained to one. Start the session with a "
                "workspace, or set "
                f"{UNCONTAINED_OPT_IN_ENV}=1 (notebook plugin config "
                "allow_uncontained_exec=true) to accept an uncontained notebook."
            )
        return True, f"audit-hook workspace containment ({workspace})"

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
        kernel = self._get_or_create(notebook_id)
        timeout = timeout_seconds or _DEFAULT_TIMEOUT_S
        cell_id = uuid.uuid4().hex[:8]
        start = datetime.datetime.now()
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
            proto.write_frame(kernel.wstream, frame_out)
            outputs: List[CellOutput] = []
            while True:
                if not self._wait_readable(kernel, timeout):
                    kernel.close()
                    with self._lock:
                        self._kernels.pop(notebook_id, None)
                    return ExecutionResult(
                        status=ExecutionStatus.CANCELLED, outputs=outputs,
                        error_name="TimeoutError",
                        error_message=f"cell exceeded {timeout}s; kernel killed")
                try:
                    frame = proto.read_frame(kernel.rstream)
                except EOFError:
                    # Kernel died mid-execution — reap it (avoid a zombie) and
                    # drop it so the next execute spawns a fresh one.
                    try:
                        kernel.proc.wait(timeout=2)
                    except Exception:
                        kernel.proc.kill()
                    with self._lock:
                        self._kernels.pop(notebook_id, None)
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED, outputs=outputs,
                        error_name="KernelDied",
                        error_message="notebook kernel closed mid-execution")
                ft = frame.get("type")
                if ft == proto.STREAM:
                    otype = (OutputType.STDOUT if frame.get("name") == "stdout"
                             else OutputType.STDERR)
                    outputs.append(CellOutput(otype, frame.get("text", "")))
                elif ft == proto.RESULT:
                    if frame.get("value") is not None:
                        outputs.append(CellOutput(OutputType.RESULT, frame["value"]))
                    kernel.info.execution_count = frame.get(
                        "execution_count", kernel.info.execution_count)
                    kernel.info.last_executed_at = start.isoformat()
                    dur = (datetime.datetime.now() - start).total_seconds()
                    return ExecutionResult(
                        status=ExecutionStatus.COMPLETED, outputs=outputs,
                        execution_count=kernel.info.execution_count,
                        duration_seconds=dur)
                elif ft == proto.ERROR:
                    outputs.append(CellOutput(
                        OutputType.ERROR, frame.get("traceback", "")))
                    dur = (datetime.datetime.now() - start).total_seconds()
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED, outputs=outputs,
                        error_name=frame.get("ename"),
                        error_message=frame.get("evalue"),
                        traceback=frame.get("traceback"),
                        duration_seconds=dur)
                elif ft == proto.TOOL_CALL:
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
        from shared.ai_tool_runner import trusted_bridge_context
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
        # pip installs persist and later imports resolve).  ensure_workspace_venv
        # drops a runner-import bridge (.pth -> site.addsitedir) into the venv,
        # so the venv python can still import ``shared.plugins.notebook.
        # kernel_main`` from the runner env — for BOTH editable and wheel
        # installs (see shared/plugins/workspace_venv.py; --system-site-packages
        # alone can't, since a venv-from-venv resolves its base to /usr).
        kernel_python = sys.executable
        kernel_env: Optional[Dict[str, str]] = None
        venv_path = resolve_venv_path(self._workspace_venv, workspace)
        if venv_path:
            ensure_workspace_venv(venv_path)
            kernel_python = venv_python(venv_path)
            kernel_env = os.environ.copy()
            apply_venv_to_env(kernel_env, venv_path)

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
        proc = subprocess.Popen(
            [kernel_python, "-m", "shared.plugins.notebook.kernel_main",
             "--workspace-root", workspace,
             "--read-fd", str(r2k_r), "--write-fd", str(k2r_w),
             *containment_args],
            pass_fds=(r2k_r, k2r_w),
            cwd=workspace,
            env=kernel_env,
            preexec_fn=_set_pdeathsig,
        )
        os.close(r2k_r)
        os.close(k2r_w)
        wstream = os.fdopen(r2k_w, "wb", buffering=0)
        rstream = os.fdopen(k2r_r, "rb", buffering=0)
        kernel = _Kernel(info, proc, wstream, rstream)
        # Handshake: the kernel sends READY after chdir + containment + namespace
        # init, and names the boundary it actually established.  Recorded on the
        # NotebookInfo so an operator reading `notebook_list` sees what bounds
        # the kernel rather than inferring it from configuration (#710).
        if self._wait_readable(kernel, 30):
            try:
                ready = proto.read_frame(rstream)   # READY
                info.boundary = ready.get("boundary_description") or None
            except EOFError:
                pass
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
