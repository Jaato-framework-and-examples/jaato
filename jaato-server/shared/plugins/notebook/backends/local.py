"""Local Jupyter kernel backend.

This backend uses jupyter_client to manage local IPython kernels.
It provides instant execution but no GPU support.
"""

import io
import logging
import os
import sys
import time
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .base import NotebookBackend
from ..types import (
    BackendCapabilities,
    CellOutput,
    ExecutionResult,
    ExecutionStatus,
    NotebookInfo,
    OutputType,
)

logger = logging.getLogger(__name__)

# Opt-in env var to permit in-process execution of model-authored cells
# when the process is NOT under kernel-enforced AppArmor confinement.
INPROCESS_OPT_IN_ENV = "JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC"


def _env_truthy(name: str) -> bool:
    """Return True when env var ``name`` holds a truthy value."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _apparmor_enforced_profile() -> Optional[str]:
    """Return the active AppArmor profile name iff it is *enforced*.

    Reads ``/proc/self/attr/current``. Returns the profile label (e.g.
    ``jaato-ws-<id>//child``) only when an AppArmor profile is active in
    ``enforce`` mode — a real kernel boundary on the cell's syscalls.
    Returns ``None`` when the process is unconfined, when AppArmor is
    unavailable, or when the profile is in ``complain`` mode (which logs
    but does not block, so it is not a boundary).
    """
    try:
        with open("/proc/self/attr/current", "r") as fh:
            # The kernel terminates this value with a NUL byte (and a
            # newline); str.strip() alone leaves the NUL, which would
            # false-negative an enforced profile and break the confined
            # path. Match the convention in server/runner_spawner.py:251.
            raw = fh.read().strip("\x00 \t\r\n")
    except OSError:
        return None
    if not raw or raw.startswith("unconfined"):
        return None
    # Format is typically 'name (enforce)' or 'name (complain)'; a bare
    # 'name' with no mode annotation is treated conservatively as not a
    # boundary.
    if "(" not in raw:
        return None
    name, _, mode = raw.partition(" (")
    if mode.rstrip(")").strip() != "enforce":
        return None
    name = name.strip()
    return name or None



class LocalJupyterBackend(NotebookBackend):
    """Local Python execution backend using IPython/exec.

    This provides a simple local execution environment. For MVP,
    we use direct exec() with isolated namespaces. A full implementation
    would use jupyter_client for proper kernel management.

    Attributes:
        _notebooks: Dict mapping notebook_id to namespace dict
        _notebook_info: Dict mapping notebook_id to NotebookInfo
    """

    def __init__(self):
        self._notebooks: Dict[str, Dict[str, Any]] = {}
        self._notebook_info: Dict[str, NotebookInfo] = {}
        self._execution_counts: Dict[str, int] = {}
        self._initialized = False
        # Tool bindings module to inject into new notebook namespaces
        self._tools_module = None
        # Explicit operator opt-in to in-process execution without
        # kernel-enforced confinement (env or backend config). Resolved in
        # initialize(); the confinement check is re-evaluated per execute().
        self._allow_inprocess_opt_in = False
        # One-shot guard so the "running unconfined by opt-in" warning is
        # logged once per backend rather than on every cell.
        self._opt_in_warning_logged = False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="local",
            supports_gpu=False,
            gpu_type=None,
            max_runtime_hours=None,  # Unlimited
            weekly_quota_hours=None,  # Unlimited
            supports_packages=True,
            is_async=False,
            requires_auth=False,
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the local backend.

        Resolves the in-process execution opt-in from the backend config
        (``allow_inprocess_exec``) or the ``JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC``
        env var. The opt-in only matters when the process is not under
        kernel-enforced AppArmor confinement (see ``_inprocess_exec_allowed``).
        """
        config = config or {}
        self._allow_inprocess_opt_in = (
            bool(config.get("allow_inprocess_exec", False))
            or _env_truthy(INPROCESS_OPT_IN_ENV)
        )
        self._initialized = True

    def _inprocess_exec_allowed(self) -> Tuple[bool, str]:
        """Decide whether model-authored cells may run in this process.

        Executing model-authored Python via ``exec``/``eval`` (and the
        ``!shell`` path) runs in the host interpreter: the cell can reach
        any object the process holds (the runtime, the tool executor, other
        plugins' in-memory state) — a far weaker boundary than a separate
        subprocess. We therefore permit it only when one of the following
        holds, and refuse (fail closed) otherwise:

        1. A kernel-enforced AppArmor profile is active — the cell's
           syscalls are bounded to the session workspace regardless of what
           the Python code attempts (the production confined-runner path).
        2. The operator explicitly opted in via ``allow_inprocess_exec`` /
           ``JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC`` (e.g. trusted single-user
           dev, or a deployment that accepts the risk).

        **Scope (important):** this gate only fail-closes the *unconfined*
        path. It does NOT reduce the confined-path risk: AppArmor bounds
        syscalls, not in-process Python memory, so under an enforced
        profile a cell can still reach this process's objects and state
        (including ``session_env`` secrets held by the runner). Closing
        that requires the deferred subprocess-kernel + tool-RPC redesign
        (or not exposing an in-process backend for untrusted use), not this
        gate. The gate is a stopgap that removes the silent unconfined hole.

        Returns:
            ``(allowed, reason)`` — ``reason`` is a human-readable
            explanation, used as the refusal message when not allowed.
        """
        profile = _apparmor_enforced_profile()
        if profile:
            return True, f"AppArmor-enforced confinement ({profile})"
        if self._allow_inprocess_opt_in:
            if not self._opt_in_warning_logged:
                logger.warning(
                    "Notebook local backend: executing model-authored cells "
                    "IN-PROCESS without kernel-enforced confinement (enabled "
                    "via %s / allow_inprocess_exec). Cell code can reach this "
                    "process's memory and state. Prefer an AppArmor-confined "
                    "runner for untrusted use.",
                    INPROCESS_OPT_IN_ENV,
                )
                self._opt_in_warning_logged = True
            return True, "explicit opt-in (config/env)"
        return False, (
            "Notebook execution refused: cells run model-authored Python in "
            "the host process, which is only permitted under kernel-enforced "
            "AppArmor confinement or with an explicit opt-in. Run under an "
            f"AppArmor-confined runner, or set {INPROCESS_OPT_IN_ENV}=1 (or "
            "the notebook plugin config allow_inprocess_exec=true) to accept "
            "in-process execution."
        )

    def shutdown(self) -> None:
        """Shutdown and clean up all notebooks."""
        self._notebooks.clear()
        self._notebook_info.clear()
        self._execution_counts.clear()
        self._initialized = False

    def is_available(self) -> bool:
        """Local backend is always available."""
        return True

    def create_notebook(
        self,
        name: str,
        gpu_enabled: bool = False,
    ) -> NotebookInfo:
        """Create a new notebook with an isolated namespace."""
        notebook_id = str(uuid.uuid4())[:8]

        # Create isolated namespace with common imports available
        namespace: Dict[str, Any] = {
            '__name__': '__main__',
            '__builtins__': __builtins__,
        }

        # Inject tool bindings module if available
        if self._tools_module is not None:
            namespace['tools'] = self._tools_module

        self._notebooks[notebook_id] = namespace
        self._execution_counts[notebook_id] = 0

        info = NotebookInfo(
            notebook_id=notebook_id,
            name=name,
            backend="local",
            gpu_enabled=False,  # Local doesn't support GPU
            created_at=datetime.now(timezone.utc).isoformat(),
            execution_count=0,
            variables={},
        )
        self._notebook_info[notebook_id] = info

        return info

    def execute(
        self,
        notebook_id: str,
        code: str,
        timeout_seconds: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute code in the notebook's namespace."""
        if notebook_id not in self._notebooks:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="NotebookNotFound",
                error_message=f"Notebook {notebook_id} not found",
            )

        # Fail closed: only run model-authored cells in-process when there
        # is genuine isolation (AppArmor) or an explicit operator opt-in.
        allowed, reason = self._inprocess_exec_allowed()
        if not allowed:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="InProcessExecutionRefused",
                error_message=reason,
            )

        namespace = self._notebooks[notebook_id]
        outputs: List[CellOutput] = []
        start_time = time.time()

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Handle shell commands (!pip install, etc.)
            if code.strip().startswith('!'):
                import subprocess
                cmd = code.strip()[1:]  # Remove '!'
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds or 60,
                )
                if result.stdout:
                    outputs.append(CellOutput(
                        output_type=OutputType.STDOUT,
                        content=result.stdout,
                    ))
                if result.stderr:
                    outputs.append(CellOutput(
                        output_type=OutputType.STDERR,
                        content=result.stderr,
                    ))
                if result.returncode != 0:
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        outputs=outputs,
                        error_name="ShellCommandError",
                        error_message=f"Command exited with code {result.returncode}",
                        duration_seconds=time.time() - start_time,
                    )
            else:
                # Execute Python code
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # Smart execution: try to capture last expression's value
                    result_value = self._execute_and_capture(code, namespace)
                    if result_value is not None:
                        outputs.append(CellOutput(
                            output_type=OutputType.RESULT,
                            content=repr(result_value),
                        ))

                # Capture any printed output
                stdout_content = stdout_capture.getvalue()
                stderr_content = stderr_capture.getvalue()

                if stdout_content:
                    # Insert stdout before result
                    outputs.insert(0, CellOutput(
                        output_type=OutputType.STDOUT,
                        content=stdout_content,
                    ))
                if stderr_content:
                    outputs.append(CellOutput(
                        output_type=OutputType.STDERR,
                        content=stderr_content,
                    ))

            # Update execution count
            self._execution_counts[notebook_id] += 1
            exec_count = self._execution_counts[notebook_id]

            # Update notebook info
            info = self._notebook_info[notebook_id]
            info.execution_count = exec_count
            info.last_executed_at = datetime.now(timezone.utc).isoformat()
            info.variables = self._get_variables_internal(namespace)

            return ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                outputs=outputs,
                execution_count=exec_count,
                duration_seconds=time.time() - start_time,
                variables=info.variables,
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()

            error_message = str(e)
            # Append an actionable hint for the most common cell-authoring
            # failure: apostrophes inside single-quoted string literals.
            # Python's own message ("unterminated string literal") doesn't
            # suggest a fix — surface one so the model can retry targeted.
            if isinstance(e, SyntaxError):
                from ..code_analyzer import _syntax_error_hint
                hint = _syntax_error_hint(e.msg or "")
                if hint:
                    error_message = f"{error_message}\n\nHint: {hint}"

            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                outputs=outputs,
                error_name=type(e).__name__,
                error_message=error_message,
                traceback=tb,
                duration_seconds=time.time() - start_time,
            )

    def _execute_and_capture(
        self,
        code: str,
        namespace: Dict[str, Any],
    ) -> Any:
        """Execute code and try to capture the last expression's value.

        This mimics IPython/Jupyter behavior where the last expression
        in a cell is displayed as the output.

        Args:
            code: Python code to execute
            namespace: The namespace to execute in

        Returns:
            The value of the last expression, or None if not applicable
        """
        import ast
        from ..cell_transform import transform_cell_source, inject_shell_helper

        # IPython-style `!shell` lines -> a helper call that streams output.
        code = transform_cell_source(code)
        inject_shell_helper(namespace)

        try:
            # Parse the code into an AST
            tree = ast.parse(code, mode='exec')

            if not tree.body:
                return None

            # Check if the last statement is an expression
            last_stmt = tree.body[-1]

            if isinstance(last_stmt, ast.Expr):
                # The last statement is an expression
                # Execute all but the last statement
                if len(tree.body) > 1:
                    module = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(module, '<cell>', 'exec'), namespace)

                # Evaluate the last expression and return its value
                expr = ast.Expression(body=last_stmt.value)
                return eval(compile(expr, '<cell>', 'eval'), namespace)
            else:
                # Last statement is not an expression (e.g., assignment, import)
                # Just execute everything
                exec(compile(tree, '<cell>', 'exec'), namespace)
                return None

        except SyntaxError:
            # If AST parsing fails, fall back to simple exec
            exec(code, namespace)
            return None

    def get_execution_status(
        self,
        notebook_id: str,
        execution_id: Optional[str] = None,
    ) -> ExecutionResult:
        """For local backend, execution is synchronous so this returns completed."""
        if notebook_id not in self._notebooks:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="NotebookNotFound",
                error_message=f"Notebook {notebook_id} not found",
            )

        # Local execution is synchronous, so if we're here, it completed
        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            execution_count=self._execution_counts.get(notebook_id, 0),
        )

    def get_variables(self, notebook_id: str) -> Dict[str, str]:
        """Get variables defined in the notebook."""
        if notebook_id not in self._notebooks:
            return {}
        return self._get_variables_internal(self._notebooks[notebook_id])

    def _get_variables_internal(self, namespace: Dict[str, Any]) -> Dict[str, str]:
        """Extract user-defined variables from namespace."""
        variables = {}
        skip_names = {'__name__', '__builtins__', '__doc__', '__loader__',
                      '__spec__', '__package__', '__cached__'}

        for name, value in namespace.items():
            if name.startswith('_') or name in skip_names:
                continue
            # Skip modules and functions for cleaner output
            if isinstance(value, type(sys)):  # Module
                continue
            try:
                type_name = type(value).__name__
                # Add shape info for arrays
                if hasattr(value, 'shape'):
                    type_name = f"{type_name}{value.shape}"
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    type_name = f"{type_name}[{len(value)}]"
                variables[name] = type_name
            except Exception:
                variables[name] = "unknown"

        return variables

    def reset_notebook(self, notebook_id: str) -> str:
        """Reset notebook by clearing its namespace.

        Returns the same notebook_id since local backend doesn't need
        to regenerate IDs (no remote persistence).
        """
        if notebook_id in self._notebooks:
            namespace = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
            }
            # Preserve tool bindings module across resets
            if self._tools_module is not None:
                namespace['tools'] = self._tools_module
            self._notebooks[notebook_id] = namespace
            self._execution_counts[notebook_id] = 0
            if notebook_id in self._notebook_info:
                self._notebook_info[notebook_id].variables = {}
                self._notebook_info[notebook_id].execution_count = 0
        return notebook_id

    def delete_notebook(self, notebook_id: str) -> None:
        """Delete a notebook."""
        self._notebooks.pop(notebook_id, None)
        self._notebook_info.pop(notebook_id, None)
        self._execution_counts.pop(notebook_id, None)

    def inject_tools_module(self, tools_module) -> None:
        """Inject a tool bindings module into all existing notebook namespaces.

        Also stores the module so that newly created and reset notebooks
        receive it automatically.

        Args:
            tools_module: A types.ModuleType with callable tool functions,
                generated by ``generate_tools_module()``.
        """
        self._tools_module = tools_module
        for namespace in self._notebooks.values():
            namespace['tools'] = tools_module

    def list_notebooks(self) -> List[NotebookInfo]:
        """List all active notebooks."""
        return list(self._notebook_info.values())
