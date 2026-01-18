"""Kaggle notebook backend.

This backend uses the official Kaggle API to execute Python code
with free GPU access (30 hours/week, Tesla P100 or T4).

Kaggle execution is asynchronous:
1. Push a kernel (notebook) to Kaggle
2. Poll for status until completion
3. Retrieve output

Environment variables:
- KAGGLE_USERNAME: Kaggle username
- KAGGLE_KEY: Kaggle API key

Or configure via ~/.kaggle/kaggle.json
"""

import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
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


# Kernel metadata template for Kaggle
KERNEL_METADATA_TEMPLATE = {
    "id": "",  # username/kernel-slug
    "title": "",
    "code_file": "script.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}


class KaggleBackend(NotebookBackend):
    """Kaggle API-based notebook backend.

    Uses the Kaggle CLI/API to push and execute kernels with GPU support.
    Execution is asynchronous - code is pushed to Kaggle and polled for results.

    Limitations:
    - State is NOT preserved between executions (each execution is a new kernel run)
    - Must wait for execution to complete (can take minutes)
    - 30 hours/week GPU quota
    """

    def __init__(self):
        self._api = None
        self._username: Optional[str] = None
        self._notebooks: Dict[str, NotebookInfo] = {}
        self._notebook_code: Dict[str, List[str]] = {}  # Accumulated code per notebook
        self._temp_dirs: Dict[str, str] = {}  # Temp directories for kernel files
        self._initialized = False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="kaggle",
            supports_gpu=True,
            gpu_type="P100/T4",
            max_runtime_hours=9.0,  # 9 hours per GPU session
            weekly_quota_hours=30.0,
            supports_packages=True,
            is_async=True,
            requires_auth=True,
        )

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Kaggle backend.

        Requires kaggle package and valid credentials.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self._api = KaggleApi()
            self._api.authenticate()

            # Get username from authenticated config
            kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_config.exists():
                with open(kaggle_config) as f:
                    creds = json.load(f)
                    self._username = creds.get("username")

            # Also check environment variable
            if not self._username:
                self._username = os.environ.get("KAGGLE_USERNAME")

            self._initialized = True

        except ImportError:
            raise RuntimeError(
                "kaggle package not installed. Install with: pip install kaggle"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to authenticate with Kaggle: {e}")

    def shutdown(self) -> None:
        """Cleanup temporary directories."""
        for temp_dir in self._temp_dirs.values():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        self._temp_dirs.clear()
        self._notebooks.clear()
        self._notebook_code.clear()
        self._initialized = False

    def is_available(self) -> bool:
        """Check if Kaggle API is configured and authenticated."""
        if not self._initialized:
            try:
                self.initialize()
            except Exception:
                return False
        return self._api is not None and self._username is not None

    def create_notebook(
        self,
        name: str,
        gpu_enabled: bool = False,
    ) -> NotebookInfo:
        """Create a new Kaggle kernel.

        Note: The actual kernel is only created on Kaggle when execute() is called.
        This method creates local tracking state.
        """
        # Generate a unique slug for this notebook
        slug = f"jaato-{uuid.uuid4().hex[:8]}"
        notebook_id = slug

        # Create temp directory for kernel files
        temp_dir = tempfile.mkdtemp(prefix="kaggle_notebook_")
        self._temp_dirs[notebook_id] = temp_dir

        info = NotebookInfo(
            notebook_id=notebook_id,
            name=name,
            backend="kaggle",
            gpu_enabled=gpu_enabled,
            created_at=datetime.utcnow().isoformat(),
            execution_count=0,
            variables={},
        )
        self._notebooks[notebook_id] = info
        self._notebook_code[notebook_id] = []

        return info

    def execute(
        self,
        notebook_id: str,
        code: str,
        timeout_seconds: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute code via Kaggle kernel.

        This pushes the code as a kernel to Kaggle and polls for results.
        Since Kaggle doesn't preserve state between runs, we accumulate
        all code and re-run the entire notebook.

        Args:
            notebook_id: Notebook ID (kernel slug)
            code: Python code to execute
            timeout_seconds: Max time to wait for completion (default: 300s)

        Returns:
            ExecutionResult with output or status
        """
        if not self._api or not self._username:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="NotInitialized",
                error_message="Kaggle backend not initialized",
            )

        if notebook_id not in self._notebooks:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="NotebookNotFound",
                error_message=f"Notebook {notebook_id} not found",
            )

        info = self._notebooks[notebook_id]
        temp_dir = self._temp_dirs[notebook_id]
        timeout = timeout_seconds or 300

        # Accumulate code (since Kaggle doesn't preserve state)
        self._notebook_code[notebook_id].append(code)

        # Build the full script with all accumulated code
        full_code = self._build_full_script(notebook_id)

        try:
            # Write the script file
            script_path = os.path.join(temp_dir, "script.py")
            with open(script_path, "w") as f:
                f.write(full_code)

            # Create kernel metadata
            kernel_id = f"{self._username}/{notebook_id}"
            metadata = KERNEL_METADATA_TEMPLATE.copy()
            metadata["id"] = kernel_id
            metadata["title"] = info.name
            metadata["enable_gpu"] = info.gpu_enabled

            metadata_path = os.path.join(temp_dir, "kernel-metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Push the kernel
            self._api.kernels_push(temp_dir)

            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < timeout:
                status = self._api.kernels_status(kernel_id)
                state = status.get("status", "unknown")

                if state == "complete":
                    # Get output
                    return self._get_kernel_output(kernel_id, notebook_id)

                if state in ("error", "cancelAcknowledged"):
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error_name="KaggleExecutionError",
                        error_message=f"Kernel execution failed: {state}",
                        duration_seconds=time.time() - start_time,
                    )

                # Still running, wait and poll again
                time.sleep(5)

            # Timeout
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="TimeoutError",
                error_message=f"Execution timed out after {timeout}s",
                duration_seconds=timeout,
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name=type(e).__name__,
                error_message=str(e),
            )

    def _build_full_script(self, notebook_id: str) -> str:
        """Build the full script from accumulated code cells."""
        cells = self._notebook_code.get(notebook_id, [])

        # Add header to capture output properly
        header = '''
# Jaato notebook execution
import sys
import json

# Store results for output
__jaato_results__ = []

'''
        # Process each cell
        processed_cells = []
        for i, cell in enumerate(cells):
            # Skip shell commands for now (handle differently)
            if cell.strip().startswith('!'):
                processed_cells.append(f"import subprocess; subprocess.run({repr(cell[1:])}, shell=True)")
            else:
                processed_cells.append(cell)

        footer = '''

# Output final variable state
import json
__vars__ = {k: str(type(v).__name__) for k, v in globals().items()
            if not k.startswith('_') and k not in ('sys', 'json', 'subprocess')}
print("__JAATO_VARS__:" + json.dumps(__vars__))
'''

        return header + "\n\n".join(processed_cells) + footer

    def _get_kernel_output(
        self,
        kernel_id: str,
        notebook_id: str,
    ) -> ExecutionResult:
        """Retrieve output from a completed kernel."""
        try:
            # Get kernel output
            output_dir = tempfile.mkdtemp(prefix="kaggle_output_")
            self._api.kernels_output(kernel_id, path=output_dir)

            outputs: List[CellOutput] = []
            variables: Dict[str, str] = {}

            # Read log file for stdout
            log_path = os.path.join(output_dir, "__results__.html")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    content = f.read()
                    # Parse output (simplified - real implementation would parse HTML)
                    outputs.append(CellOutput(
                        output_type=OutputType.STDOUT,
                        content=content,
                        mime_type="text/html",
                    ))

            # Check for output files
            for filename in os.listdir(output_dir):
                filepath = os.path.join(output_dir, filename)
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    # Image output
                    import base64
                    with open(filepath, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    mime = 'image/png' if filename.endswith('.png') else 'image/jpeg'
                    outputs.append(CellOutput(
                        output_type=OutputType.DISPLAY,
                        content=img_data,
                        mime_type=mime,
                        metadata={"filename": filename},
                    ))

            # Cleanup
            shutil.rmtree(output_dir, ignore_errors=True)

            # Update notebook info
            info = self._notebooks[notebook_id]
            info.execution_count += 1
            info.last_executed_at = datetime.utcnow().isoformat()
            info.variables = variables

            return ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                outputs=outputs,
                execution_count=info.execution_count,
                variables=variables,
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name=type(e).__name__,
                error_message=f"Failed to retrieve output: {e}",
            )

    def get_execution_status(
        self,
        notebook_id: str,
        execution_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Get status of a Kaggle kernel execution."""
        if not self._api or not self._username:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="NotInitialized",
                error_message="Kaggle backend not initialized",
            )

        kernel_id = f"{self._username}/{notebook_id}"

        try:
            status = self._api.kernels_status(kernel_id)
            state = status.get("status", "unknown")

            status_map = {
                "queued": ExecutionStatus.QUEUED,
                "running": ExecutionStatus.RUNNING,
                "complete": ExecutionStatus.COMPLETED,
                "error": ExecutionStatus.FAILED,
                "cancelAcknowledged": ExecutionStatus.CANCELLED,
            }

            return ExecutionResult(
                status=status_map.get(state, ExecutionStatus.PENDING),
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name=type(e).__name__,
                error_message=str(e),
            )

    def get_variables(self, notebook_id: str) -> Dict[str, str]:
        """Get variables from the notebook.

        Note: Since Kaggle doesn't preserve state, this returns the
        variables from the last execution.
        """
        if notebook_id in self._notebooks:
            return self._notebooks[notebook_id].variables
        return {}

    def reset_notebook(self, notebook_id: str) -> None:
        """Reset notebook by clearing accumulated code."""
        if notebook_id in self._notebook_code:
            self._notebook_code[notebook_id] = []
        if notebook_id in self._notebooks:
            self._notebooks[notebook_id].variables = {}
            self._notebooks[notebook_id].execution_count = 0

    def delete_notebook(self, notebook_id: str) -> None:
        """Delete a notebook and clean up."""
        # Clean up temp directory
        if notebook_id in self._temp_dirs:
            shutil.rmtree(self._temp_dirs[notebook_id], ignore_errors=True)
            del self._temp_dirs[notebook_id]

        self._notebooks.pop(notebook_id, None)
        self._notebook_code.pop(notebook_id, None)

        # Optionally delete from Kaggle (commented out to preserve history)
        # kernel_id = f"{self._username}/{notebook_id}"
        # self._api.kernels_delete(kernel_id)

    def list_notebooks(self) -> List[NotebookInfo]:
        """List all active notebooks."""
        return list(self._notebooks.values())
