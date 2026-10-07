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
from ..kernel_sandbox import (
    BOUNDARY_NONE,
    BOUNDARY_OPT_OUT,
    apparmor_enforced_profile,
    env_truthy,
)
from ..types import (
    BackendCapabilities,
    CellOutput,
    ExecutionResult,
    ExecutionStatus,
    NotebookInfo,
    OutputType,
)

logger = logging.getLogger(__name__)

# Opt-in env var to permit in-process execution of model-authored cells.
# Required under AppArmor too since #1323: the runner's profile does not
# bound an in-process cell, which can unconfine itself.
INPROCESS_OPT_IN_ENV = "JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC"

# Both helpers live in ``kernel_sandbox`` so the in-process gate here and the
# kernel's own boundary decision read the SAME answer about confinement
# (issue #710: a protection that covers one path reads as covering all of
# them).  Re-bound at module level under their historical names so the gate's
# call sites — and the tests that monkeypatch them — are unchanged.
_env_truthy = env_truthy
_apparmor_enforced_profile = apparmor_enforced_profile


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
        env var.  Without it no cell runs here (see
        ``_inprocess_exec_allowed``).
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
        plugins' in-memory state, ``session_env`` secrets).  It is permitted
        only with an explicit operator opt-in (``allow_inprocess_exec`` /
        ``JAATO_NOTEBOOK_ALLOW_INPROCESS_EXEC``), and refused otherwise.

        **AppArmor is not enough on its own (#1323).**  This used to allow
        execution whenever the process wore an enforced profile.  But the
        runner's profile is the session's BASE profile, which keeps
        ``change_profile -> unconfined`` and write access to
        ``/proc/self/attr/current`` for the framework's own use, so a cell
        could write ``changeprofile unconfined`` and leave confinement.  The
        subprocess backend (the default) starts its kernel in ``//child``,
        which drops those rules; an in-process cell cannot be moved there.

        Returns:
            ``(allowed, reason)`` — ``reason`` is a human-readable
            explanation, used as the refusal message when not allowed.
        """
        if self._allow_inprocess_opt_in:
            if not self._opt_in_warning_logged:
                profile = _apparmor_enforced_profile()
                logger.warning(
                    "Notebook local backend: executing model-authored cells "
                    "IN-PROCESS (enabled via %s / allow_inprocess_exec). Cell "
                    "code can reach this process's memory and state%s. Prefer "
                    "the default subprocess backend for untrusted use.",
                    INPROCESS_OPT_IN_ENV,
                    (f", and can leave the AppArmor profile {profile}, which "
                     "keeps change_profile -> unconfined for the framework"
                     if profile else ""),
                )
                self._opt_in_warning_logged = True
            return True, "explicit opt-in (config/env)"
        return False, (
            "Notebook execution refused: the local backend runs model-authored "
            "Python in the host process, which is permitted only with an "
            "explicit opt-in (an AppArmor profile does not bound it: the "
            "runner's profile lets code in it unconfine itself). Use the "
            "default subprocess backend (notebook plugin config "
            f"default_backend=subprocess), or set {INPROCESS_OPT_IN_ENV}=1 "
            "(or allow_inprocess_exec=true) to accept in-process execution."
        )

    def execution_boundary(self) -> Tuple[bool, str]:
        """The in-process gate, answered for ``NotebookPlugin``'s pre-dispatch check.

        Identical to what ``execute()`` enforces per cell — this is the same
        decision asked one layer up, so the plugin can refuse before it routes
        a cell here (issue #710) and the model is told which knob to reach for
        instead of seeing a bare failure.  ``execute()`` keeps its own call
        because a caller holding the backend directly (a harness, a test,
        ``install_package``) must not be able to skip the gate.

        Note what this backend does NOT bound: a cell that IS permitted here
        runs in the host interpreter with no filesystem containment of its
        own, and an AppArmor profile the process wears does not supply one
        (the cell can leave it, #1323).  Under the opt-in the operator has
        accepted that.
        """
        return self._inprocess_exec_allowed()

    def boundary_kind(self) -> str:
        """The tier a cell would run under here (issue #1012).

        Never :data:`BOUNDARY_AUDIT` (this backend installs no audit hook,
        so a cell here never meets #1011's ``import ctypes`` refusal), and
        never :data:`BOUNDARY_APPARMOR` since #1323: the profile the runner
        wears is its base profile, which a cell can leave.  So the answer is
        :data:`BOUNDARY_OPT_OUT` when the operator opted in and
        :data:`BOUNDARY_NONE` (refused) otherwise.
        """
        allowed, _ = self._inprocess_exec_allowed()
        return BOUNDARY_OPT_OUT if allowed else BOUNDARY_NONE

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
