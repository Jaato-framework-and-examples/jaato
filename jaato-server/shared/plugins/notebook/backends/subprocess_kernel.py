"""Subprocess-kernel notebook backend (design option 1c, PR 1).

Runs each notebook in its OWN Python subprocess launched with
``cwd=workspace_root`` (``kernel_main``), so notebook code's ``os.getcwd()`` and
relative paths resolve in-workspace — without the process-global ``os.chdir`` the
framework forbids (core.py:915).  State persists in the kernel namespace.

PR 1 scope: cell exec + streaming + variables/reset/lifecycle.  The ``tools.X()``
bridge is a stub in the kernel (raises) and lands in PR 2.  This backend is
OPT-IN (``plugin_configs.notebook.backend: "subprocess"``); the in-process
``LocalJupyterBackend`` stays the default until the PR 3 cutover.
"""

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

from .base import NotebookBackend
from .. import kernel_protocol as proto
from ..types import (
    BackendCapabilities, CellOutput, ExecutionResult, ExecutionStatus,
    NotebookInfo, OutputType,
)

_DEFAULT_TIMEOUT_S = 300


def _set_pdeathsig() -> None:
    """preexec: SIGKILL the kernel if the runner (parent) dies, so a crashed
    runner never orphans kernels (the jdtls/LSP precedent —
    feedback_runner_child_heavyweight_proc_needs_pdeathsig).  Best-effort."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


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
        self._kernels: Dict[str, _Kernel] = {}
        self._lock = threading.Lock()
        # The runner's tool executor (``ToolExecutor.execute`` shape:
        # (name, args) -> (ok, result)).  Wired by the plugin; serves the
        # kernel's cross-process tools.X() calls (PR 2).  None → tools.X()
        # in the kernel reports "no executor wired".
        self._tool_executor: Optional[
            Callable[[str, Dict], Tuple[bool, object]]] = None

    def set_tool_executor(
        self, executor_fn: Callable[[str, Dict], Tuple[bool, object]]
    ) -> None:
        """Wire the runner-side tool executor that serves kernel tool_call frames
        (mirrors LocalJupyterBackend.inject_tools_module for the in-process case)."""
        self._tool_executor = executor_fn

    # ---- protocol: identity / lifecycle -------------------------------------

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.BACKEND_NAME, supports_gpu=False,
            supports_packages=True, is_async=False, requires_auth=False)

    def initialize(self, config: Optional[Dict] = None) -> None:
        if config:
            self._workspace_root = config.get("workspace_root") or self._workspace_root

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
            proto.write_frame(kernel.wstream, {
                "type": proto.EXECUTE, "cell_id": cell_id, "code": code})
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
        workspace = self._workspace_root or os.getcwd()
        r2k_r, r2k_w = os.pipe()   # runner → kernel
        k2r_r, k2r_w = os.pipe()   # kernel → runner
        proc = subprocess.Popen(
            [sys.executable, "-m", "shared.plugins.notebook.kernel_main",
             "--workspace-root", workspace,
             "--read-fd", str(r2k_r), "--write-fd", str(k2r_w)],
            pass_fds=(r2k_r, k2r_w),
            cwd=workspace,
            preexec_fn=_set_pdeathsig,
        )
        os.close(r2k_r)
        os.close(k2r_w)
        wstream = os.fdopen(r2k_w, "wb", buffering=0)
        rstream = os.fdopen(k2r_r, "rb", buffering=0)
        kernel = _Kernel(info, proc, wstream, rstream)
        # Handshake: the kernel sends READY after chdir + namespace init.
        if self._wait_readable(kernel, 30):
            try:
                proto.read_frame(rstream)   # READY
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
