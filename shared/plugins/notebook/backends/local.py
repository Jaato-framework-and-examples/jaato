"""Local Jupyter kernel backend.

This backend uses jupyter_client to manage local IPython kernels.
It provides instant execution but no GPU support.
"""

import io
import sys
import time
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import NotebookBackend
from ..types import (
    BackendCapabilities,
    CellOutput,
    ExecutionResult,
    ExecutionStatus,
    NotebookInfo,
    OutputType,
)


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
        """Initialize the local backend."""
        self._initialized = True

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

        self._notebooks[notebook_id] = namespace
        self._execution_counts[notebook_id] = 0

        info = NotebookInfo(
            notebook_id=notebook_id,
            name=name,
            backend="local",
            gpu_enabled=False,  # Local doesn't support GPU
            created_at=datetime.utcnow().isoformat(),
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
            info.last_executed_at = datetime.utcnow().isoformat()
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

            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                outputs=outputs,
                error_name=type(e).__name__,
                error_message=str(e),
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

    def reset_notebook(self, notebook_id: str) -> None:
        """Reset notebook by clearing its namespace."""
        if notebook_id in self._notebooks:
            self._notebooks[notebook_id] = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
            }
            self._execution_counts[notebook_id] = 0
            if notebook_id in self._notebook_info:
                self._notebook_info[notebook_id].variables = {}
                self._notebook_info[notebook_id].execution_count = 0

    def delete_notebook(self, notebook_id: str) -> None:
        """Delete a notebook."""
        self._notebooks.pop(notebook_id, None)
        self._notebook_info.pop(notebook_id, None)
        self._execution_counts.pop(notebook_id, None)

    def list_notebooks(self) -> List[NotebookInfo]:
        """List all active notebooks."""
        return list(self._notebook_info.values())
