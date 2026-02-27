"""Kaggle notebook backend.

This backend uses the official Kaggle API to execute Python code
with free GPU access (30 hours/week, Tesla P100 or T4).

Kaggle execution is asynchronous:
1. Push a kernel (notebook) to Kaggle
2. Poll for status until completion
3. Retrieve output

## Authentication Methods

1. **Access Token (KAGGLE_API_TOKEN)**: The new/preferred method.
   - Set via KAGGLE_API_TOKEN environment variable
   - Or place token in ~/.kaggle/access_token
   - Get token from: https://www.kaggle.com/settings -> API -> Create New Token

2. **Legacy Credentials (KAGGLE_USERNAME + KAGGLE_KEY)**:
   - Set via KAGGLE_USERNAME and KAGGLE_KEY environment variables
   - Or place in ~/.kaggle/kaggle.json as {"username": "...", "key": "..."}

## Polling Strategy

After pushing a kernel, there's a delay before it appears in Kaggle's API. During this
window, `kernels_status()` may return 403 Forbidden. This backend uses `kernels_list()`
for polling, which reliably shows newly pushed kernels once indexed.

Once a kernel is complete and visible in the list, `kernels_output()` works with
access tokens to retrieve the execution output.
"""

import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

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
        self._trace_fn: Optional[callable] = None  # Trace callback

    def set_trace_fn(self, trace_fn: callable) -> None:
        """Set a trace callback for logging."""
        self._trace_fn = trace_fn

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        from shared.trace import provider_trace
        provider_trace("Kaggle", msg)

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for REST API calls.

        Uses the access token from the kaggle API instance.
        """
        if not self._api:
            return {}

        # Try multiple ways to get the token from the SDK
        # The kaggle SDK stores credentials in config_values after authentication
        config = self._api.config_values

        # Method 1: Access token (KAGGLE_API_TOKEN)
        token = config.get("token")
        if token:
            return {"Authorization": f"Bearer {token}"}

        # Method 2: Check for api_token (another possible key name)
        api_token = config.get("api_token") or config.get("access_token")
        if api_token:
            return {"Authorization": f"Bearer {api_token}"}

        # Method 3: Legacy basic auth (KAGGLE_USERNAME + KAGGLE_KEY)
        username = config.get(self._api.CONFIG_NAME_USER)
        key = config.get(self._api.CONFIG_NAME_KEY)
        if username and key:
            import base64
            credentials = base64.b64encode(f"{username}:{key}".encode()).decode()
            return {"Authorization": f"Basic {credentials}"}

        # Method 4: Check environment directly as fallback
        import os
        env_token = os.environ.get("KAGGLE_API_TOKEN")
        if env_token:
            return {"Authorization": f"Bearer {env_token}"}

        env_user = os.environ.get("KAGGLE_USERNAME")
        env_key = os.environ.get("KAGGLE_KEY")
        if env_user and env_key:
            import base64
            credentials = base64.b64encode(f"{env_user}:{env_key}".encode()).decode()
            return {"Authorization": f"Basic {credentials}"}

        return {}

    def _rest_get_kernel_status(self, user: str, kernel_slug: str) -> Optional[str]:
        """Get kernel status via REST API v1.

        Uses the documented endpoint: GET /api/v1/kernels/status/{userName}/{kernelSlug}
        Note: This endpoint uses basicAuth (username/API key), not Bearer tokens.
        Returns status string or None if request fails.
        """
        url = f"https://www.kaggle.com/api/v1/kernels/status/{user}/{kernel_slug}"
        headers = self._get_auth_headers()
        auth_type = 'Bearer' if 'Bearer' in str(headers) else 'Basic' if 'Basic' in str(headers) else 'none'
        self._trace(f"REST status API: {url} (auth: {auth_type})")

        from shared.http import get_httpx_client

        try:
            with get_httpx_client(timeout=30.0) as client:
                resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status") or data.get("result", {}).get("status")
                self._trace(f"REST status API success: status={status}")
                return status
            else:
                self._trace(f"REST status API error: {resp.status_code} {resp.text[:200]}")
                return None
        except Exception as e:
            self._trace(f"REST status API request failed: {e}")
            return None

    def _rest_get_kernel(self, user: str, kernel_slug: str) -> Optional[Dict]:
        """Get kernel details via REST API v1 (deprecated - use _rest_get_kernel_status)."""
        status = self._rest_get_kernel_status(user, kernel_slug)
        if status:
            return {"status": status}
        return None

    def _rest_get_kernel_output(self, user: str, kernel_slug: str, output_dir: str) -> bool:
        """Download kernel output via REST API v1.

        Uses the documented endpoint: GET /api/v1/kernels/kernel_output
        Note: This endpoint uses basicAuth (username/API key), not Bearer tokens.
        Returns True if output was downloaded successfully.
        """
        url = f"https://www.kaggle.com/api/v1/kernels/kernel_output?user_name={user}&kernel_slug={kernel_slug}"
        headers = self._get_auth_headers()
        self._trace(f"REST output API: {url}")

        from shared.http import get_httpx_client

        try:
            with get_httpx_client(timeout=60.0) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    # Response might be JSON with file list and download URLs
                    # or it might be the actual file content
                    content_type = resp.headers.get("Content-Type", "")

                    if "application/json" in content_type:
                        data = resp.json()
                        # Handle JSON response with file list
                        files = data.get("files", [])
                        log = data.get("log", "")

                        # Save log if present
                        if log:
                            log_path = os.path.join(output_dir, "output.log")
                            with open(log_path, "w") as f:
                                f.write(log)
                            return True

                        # Download individual files
                        for file_info in files:
                            file_url = file_info.get("url")
                            file_name = file_info.get("name", "output")
                            if file_url:
                                file_resp = client.get(file_url, headers=headers, timeout=60)
                                if file_resp.status_code == 200:
                                    file_path = os.path.join(output_dir, file_name)
                                    with open(file_path, "wb") as f:
                                        f.write(file_resp.content)

                        return len(files) > 0 or bool(log)
                    else:
                        # Binary content - save directly
                        output_path = os.path.join(output_dir, "output.log")
                        with open(output_path, "wb") as f:
                            f.write(resp.content)
                        return True
                else:
                    self._trace(f"REST output API error: {resp.status_code} {resp.text[:200]}")
                    return False
        except Exception as e:
            self._trace(f"REST output API request failed: {e}")
            return False

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

        Credentials are resolved in this order:
        1. New access token: KAGGLE_API_TOKEN env var
        2. Legacy credentials: KAGGLE_USERNAME + KAGGLE_KEY env vars
        3. Config files: ~/.kaggle/access_token or ~/.kaggle/kaggle.json
        """
        try:
            # Import kaggle - its __init__.py auto-creates a global `api` instance
            # and calls authenticate(), which handles KAGGLE_API_TOKEN env var
            # (or falls back to ~/.kaggle/access_token, ~/.kaggle/kaggle.json, etc.)
            import kaggle

            # Use the global pre-authenticated API instance
            self._api = kaggle.api

            # Get username from API (set during auto-authentication)
            self._username = self._api.config_values.get(self._api.CONFIG_NAME_USER)
            if not self._username:
                raise RuntimeError(
                    "Kaggle authentication failed. Set one of:\n"
                    "  - KAGGLE_API_TOKEN env var (access token)\n"
                    "  - KAGGLE_USERNAME + KAGGLE_KEY env vars (legacy)\n"
                    "Or create ~/.kaggle/access_token or ~/.kaggle/kaggle.json"
                )

            self._initialized = True
            self._trace(f"Authenticated as {self._username}")

        except ImportError:
            raise RuntimeError(
                "kaggle package not installed. Install with: pip install kaggle"
            )
        except RuntimeError:
            raise  # Re-raise our custom errors
        except Exception as e:
            # Provide helpful message for common auth failures
            hint = ""
            if "Could not find kaggle.json" in str(e):
                hint = (
                    "\n\nSet credentials via:\n"
                    "  1. Environment variables in .env:\n"
                    "     KAGGLE_USERNAME=your_username\n"
                    "     KAGGLE_KEY=your_api_key\n"
                    "  2. Or create ~/.kaggle/kaggle.json with your credentials\n"
                    "     (download from https://www.kaggle.com/settings -> 'Create New Token')"
                )
            raise RuntimeError(f"Failed to authenticate with Kaggle: {e}{hint}")

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
        # Use full UUID to minimize collision chance across sessions
        slug = f"jaato-{uuid.uuid4().hex}"
        notebook_id = slug

        # Create temp directory for kernel files
        temp_dir = tempfile.mkdtemp(prefix="kaggle_notebook_")
        self._temp_dirs[notebook_id] = temp_dir

        info = NotebookInfo(
            notebook_id=notebook_id,
            name=name,
            backend="kaggle",
            gpu_enabled=gpu_enabled,
            created_at=datetime.now(timezone.utc).isoformat(),
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

            # Push the kernel (with retry on 409 Conflict)
            original_notebook_id = notebook_id
            max_push_attempts = 3
            push_result = None
            for push_attempt in range(max_push_attempts):
                self._trace(f"Pushing kernel {kernel_id} (gpu={info.gpu_enabled})")
                try:
                    push_result = self._api.kernels_push(temp_dir)
                    self._trace(f"Kernel pushed successfully: {push_result}")
                    break  # Success, exit retry loop
                except Exception as push_error:
                    error_msg = str(push_error)
                    if "409" in error_msg and push_attempt < max_push_attempts - 1:
                        # Kernel slug conflict - generate completely new UUID
                        notebook_id = f"jaato-{uuid.uuid4().hex}"
                        kernel_id = f"{self._username}/{notebook_id}"
                        metadata["id"] = kernel_id
                        with open(metadata_path, "w") as f:
                            json.dump(metadata, f, indent=2)
                        self._trace(f"409 Conflict, retrying with new slug: {kernel_id}")
                        continue
                    if "401" in error_msg:
                        error_msg += (
                            f"\n\nHint: Verify KAGGLE_USERNAME='{self._username}' matches "
                            f"your Kaggle profile URL exactly (case-sensitive)."
                        )
                    elif "409" in error_msg:
                        error_msg += (
                            "\n\nHint: A kernel with this ID already exists on Kaggle. "
                            "This can happen if a previous run didn't complete cleanly."
                        )
                    raise RuntimeError(error_msg)

            # Update internal tracking if notebook_id changed due to 409 retry
            if notebook_id != original_notebook_id:
                self._trace(f"Updating internal tracking: {original_notebook_id} -> {notebook_id}")
                # Move all internal state to new notebook_id
                self._notebooks[notebook_id] = self._notebooks.pop(original_notebook_id)
                self._notebooks[notebook_id].notebook_id = notebook_id
                self._notebook_code[notebook_id] = self._notebook_code.pop(original_notebook_id)
                self._temp_dirs[notebook_id] = self._temp_dirs.pop(original_notebook_id)
                info = self._notebooks[notebook_id]

            # Poll for completion using kernels_list + kernels_status
            #
            # Two-step polling:
            # 1. Use kernels_list to find the kernel (confirms it's indexed)
            # 2. Use kernels_status to get execution status
            #
            # Note: Kaggle uses the kernel title as the slug, not the id we provide.
            # So we search by title (info.name) to find our kernel.
            #
            # There can be a delay before a newly pushed kernel appears in the list,
            # so we add a brief initial wait before starting to poll.
            self._trace("Waiting 5s for kernel to be indexed...")
            time.sleep(5)

            start_time = time.time()
            poll_interval = 10  # Start with 10 seconds
            max_poll_interval = 30
            attempt = 0
            last_status = None
            kernel_ref = None  # Will be set when we find the kernel

            while time.time() - start_time < timeout:
                attempt += 1
                self._trace(f"Polling attempt {attempt} (elapsed: {int(time.time() - start_time)}s)")

                # Step 1: Find the kernel in the list by title
                # Kaggle uses title as the slug, not our generated notebook_id
                status = None
                kernel_url = None
                try:
                    if not kernel_ref:
                        # Search for our kernel by title
                        kernels = self._api.kernels_list(mine=True, page_size=50)
                        self._trace(f"kernels_list returned {len(kernels) if kernels else 0} kernels")
                        for k in kernels:
                            k_ref = getattr(k, 'ref', '') or str(k)
                            k_title = getattr(k, 'title', '')
                            # Match by title (name) since Kaggle uses that as slug
                            if k_title == info.name or info.name in k_ref:
                                kernel_ref = k_ref
                                kernel_url = f"https://www.kaggle.com/code/{kernel_ref}"
                                self._trace(f"Found kernel: ref={k_ref}, title={k_title}")
                                break

                    # Step 2: Get execution status via kernels_status
                    if kernel_ref:
                        kernel_url = f"https://www.kaggle.com/code/{kernel_ref}"
                        try:
                            status_resp = self._api.kernels_status(kernel_ref)
                            # Response has _status attribute with enum value
                            if hasattr(status_resp, '_status'):
                                status = status_resp._status.name if status_resp._status else None
                            elif hasattr(status_resp, 'status'):
                                status = status_resp.status
                            elif isinstance(status_resp, dict):
                                status = status_resp.get('status')
                            self._trace(f"kernels_status({kernel_ref}): {status}")
                        except Exception as status_err:
                            self._trace(f"kernels_status failed: {status_err}")
                            # If status fails, kernel might still be initializing
                            status = None
                except Exception as e:
                    self._trace(f"Polling error: {e}")

                if status and status != last_status:
                    last_status = status
                    self._trace(f"Kernel status: {status}")

                # Check for completion states
                status_lower = (status or "").lower()
                if status_lower in ("complete", "completed"):
                    # Kernel completed - try to get output
                    self._trace(f"Execution complete!")

                    # Try SDK output retrieval using the actual kernel ref
                    # (Kaggle uses the title as slug, not our generated notebook_id)
                    self._trace(f"Trying SDK output retrieval for {kernel_ref}...")
                    result = self._get_kernel_output(
                        kernel_ref,  # Use actual Kaggle ref, not our notebook_id
                        notebook_id,
                    )
                    result.duration_seconds = time.time() - start_time
                    if result.status == ExecutionStatus.COMPLETED:
                        self._trace(f"Output retrieved: {len(result.outputs)} outputs")
                        return result
                    else:
                        self._trace(f"SDK output retrieval failed: {result.error_message}")

                    # Fallback: report success with link to view output on Kaggle
                    result = ExecutionResult(
                        status=ExecutionStatus.COMPLETED,
                        outputs=[CellOutput(
                            output_type=OutputType.STDOUT,
                            content=f"Kernel execution completed successfully.\n\n"
                                    f"View output at: {kernel_url}\n\n"
                                    f"Note: Could not retrieve output automatically. "
                                    f"Please view results on Kaggle.com.",
                        )],
                        duration_seconds=time.time() - start_time,
                    )
                    return result
                elif status_lower in ("error", "failed", "cancelacknowledged"):
                    # Kernel failed
                    self._trace(f"Execution failed: {status}")
                    result = ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error_name="KaggleExecutionError",
                        error_message=f"Kernel failed with status: {status}\n\n"
                                      f"View error details at: {kernel_url}",
                        duration_seconds=time.time() - start_time,
                    )
                    return result
                elif status_lower in ("running", "queued", "pending"):
                    pass  # Still running, continue polling
                elif status is None:
                    self._trace("Kernel not yet visible in list, waiting...")
                else:
                    self._trace(f"Unknown status: {status}, continuing to poll...")

                # Wait before next poll, with exponential backoff
                time.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.2, max_poll_interval)

            # Timeout
            self._trace(f"Execution timed out after {timeout}s")
            if kernel_ref:
                kernel_url = f"https://www.kaggle.com/code/{kernel_ref}"
            else:
                kernel_url = f"https://www.kaggle.com/{self._username}"
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="TimeoutError",
                error_message=f"Execution timed out after {timeout}s. Last status: {last_status}\n\n"
                              f"Check status at: {kernel_url}",
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

    def _get_kernel_output_rest(self, notebook_id: str) -> ExecutionResult:
        """Retrieve output from a completed kernel using REST API.

        Uses the v1 REST API which works with access token auth,
        unlike the SDK's kernels_output() which uses gRPC endpoints.
        """
        try:
            self._trace(f"Retrieving output via REST API for {self._username}/{notebook_id}")
            output_dir = tempfile.mkdtemp(prefix="kaggle_output_")

            # Try REST API first
            if self._rest_get_kernel_output(self._username, notebook_id, output_dir):
                return self._parse_kernel_output(output_dir, notebook_id)

            # If REST API didn't return output, check if we can get log from kernel info
            kernel_info = self._rest_get_kernel(self._username, notebook_id)
            if kernel_info:
                # Some kernel info responses include log/output directly
                log = kernel_info.get("log") or kernel_info.get("output") or ""
                if log:
                    log_path = os.path.join(output_dir, "output.log")
                    with open(log_path, "w") as f:
                        f.write(log)
                    return self._parse_kernel_output(output_dir, notebook_id)

            # No output available
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="NoOutput",
                error_message="Could not retrieve kernel output via REST API",
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name=type(e).__name__,
                error_message=f"Failed to retrieve output: {e}",
            )

    def _parse_kernel_output(self, output_dir: str, notebook_id: str) -> ExecutionResult:
        """Parse kernel output files from a directory."""
        outputs: List[CellOutput] = []
        variables: Dict[str, str] = {}

        # List all files in output directory for debugging
        all_files = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), output_dir)
                all_files.append(rel_path)

        self._trace(f"Output files found: {all_files}")

        # Try multiple possible output file names
        possible_log_files = [
            "__results__.html",
            "output.log",
            "__notebook__.html",
            "__output__.html",
        ]

        log_content = None
        for log_name in possible_log_files:
            log_path = os.path.join(output_dir, log_name)
            if os.path.exists(log_path):
                with open(log_path) as f:
                    log_content = f.read()
                break

        # Also check for .log files
        if not log_content:
            for filename in all_files:
                if filename.endswith('.log') or filename.endswith('.txt'):
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath) as f:
                        log_content = f.read()
                    break

        if log_content:
            self._trace(f"Log content found: {len(log_content)} chars")
            # Strip HTML tags for cleaner output if it's HTML
            if '<html' in log_content.lower() or '<body' in log_content.lower():
                import re
                log_content = re.sub(r'<script[^>]*>.*?</script>', '', log_content, flags=re.DOTALL | re.IGNORECASE)
                log_content = re.sub(r'<style[^>]*>.*?</style>', '', log_content, flags=re.DOTALL | re.IGNORECASE)
                log_content = re.sub(r'<[^>]+>', '\n', log_content)
                log_content = re.sub(r'\n\s*\n', '\n\n', log_content).strip()

            outputs.append(CellOutput(
                output_type=OutputType.STDOUT,
                content=log_content,
            ))
        elif all_files:
            self._trace(f"No log content found, but found files: {all_files}")
            outputs.append(CellOutput(
                output_type=OutputType.STDOUT,
                content=f"Execution completed. Output files: {', '.join(all_files)}",
            ))
        else:
            self._trace("WARNING: No log content and no output files found")

        # Check for output files (images, data files)
        for filename in all_files:
            filepath = os.path.join(output_dir, filename)
            if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                import base64
                with open(filepath, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg',
                           '.jpeg': 'image/jpeg', '.gif': 'image/gif'}
                ext = os.path.splitext(filename)[1].lower()
                outputs.append(CellOutput(
                    output_type=OutputType.DISPLAY,
                    content=img_data,
                    mime_type=mime_map.get(ext, 'image/png'),
                    metadata={"filename": filename},
                ))
            elif filename.endswith('.csv'):
                with open(filepath) as f:
                    csv_content = f.read()
                if len(csv_content) > 5000:
                    csv_content = csv_content[:5000] + "\n... (truncated)"
                outputs.append(CellOutput(
                    output_type=OutputType.RESULT,
                    content=csv_content,
                    metadata={"filename": filename},
                ))

        # Cleanup
        shutil.rmtree(output_dir, ignore_errors=True)

        # Update notebook info
        if notebook_id in self._notebooks:
            info = self._notebooks[notebook_id]
            info.execution_count += 1
            info.last_executed_at = datetime.now(timezone.utc).isoformat()
            info.variables = variables

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            outputs=outputs,
            execution_count=self._notebooks.get(notebook_id, NotebookInfo(notebook_id=notebook_id, name="", backend="kaggle")).execution_count,
            variables=variables,
        )

    def _get_kernel_output(
        self,
        kernel_id: str,
        notebook_id: str,
    ) -> ExecutionResult:
        """Retrieve output from a completed kernel (uses SDK - may fail with access tokens)."""
        try:
            # Get kernel output
            self._trace(f"Retrieving output for kernel {kernel_id}")
            output_dir = tempfile.mkdtemp(prefix="kaggle_output_")
            self._api.kernels_output(kernel_id, path=output_dir)

            outputs: List[CellOutput] = []
            variables: Dict[str, str] = {}

            # List all files in output directory for debugging
            all_files = []
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    rel_path = os.path.relpath(os.path.join(root, f), output_dir)
                    all_files.append(rel_path)

            self._trace(f"Output files found: {all_files}")

            # Try multiple possible output file names
            possible_log_files = [
                "__results__.html",
                "output.log",
                "__notebook__.html",
                "__output__.html",
            ]

            log_content = None
            for log_name in possible_log_files:
                log_path = os.path.join(output_dir, log_name)
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        log_content = f.read()
                    break

            # Also check for .log files
            if not log_content:
                for filename in all_files:
                    if filename.endswith('.log') or filename.endswith('.txt'):
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath) as f:
                            log_content = f.read()
                        break

            if log_content:
                self._trace(f"Log content found: {len(log_content)} chars")
                # Strip HTML tags for cleaner output if it's HTML
                if '<html' in log_content.lower() or '<body' in log_content.lower():
                    # Basic HTML stripping - extract text between tags
                    import re
                    # Remove script and style elements
                    log_content = re.sub(r'<script[^>]*>.*?</script>', '', log_content, flags=re.DOTALL | re.IGNORECASE)
                    log_content = re.sub(r'<style[^>]*>.*?</style>', '', log_content, flags=re.DOTALL | re.IGNORECASE)
                    # Remove HTML tags
                    log_content = re.sub(r'<[^>]+>', '\n', log_content)
                    # Clean up whitespace
                    log_content = re.sub(r'\n\s*\n', '\n\n', log_content).strip()

                outputs.append(CellOutput(
                    output_type=OutputType.STDOUT,
                    content=log_content,
                ))
            elif all_files:
                # No log content found, but we have files - report what we found
                self._trace(f"No log content found, but found files: {all_files}")
                outputs.append(CellOutput(
                    output_type=OutputType.STDOUT,
                    content=f"Execution completed. Output files: {', '.join(all_files)}",
                ))
            else:
                # No log content and no files - truly empty output
                self._trace("WARNING: No log content and no output files found")

            # Check for output files (images, data files)
            for filename in all_files:
                filepath = os.path.join(output_dir, filename)
                if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    # Image output
                    import base64
                    with open(filepath, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg',
                               '.jpeg': 'image/jpeg', '.gif': 'image/gif'}
                    ext = os.path.splitext(filename)[1].lower()
                    outputs.append(CellOutput(
                        output_type=OutputType.DISPLAY,
                        content=img_data,
                        mime_type=mime_map.get(ext, 'image/png'),
                        metadata={"filename": filename},
                    ))
                elif filename.endswith('.csv'):
                    # CSV data - include preview
                    with open(filepath) as f:
                        csv_content = f.read()
                    # Truncate if too large
                    if len(csv_content) > 5000:
                        csv_content = csv_content[:5000] + "\n... (truncated)"
                    outputs.append(CellOutput(
                        output_type=OutputType.RESULT,
                        content=csv_content,
                        metadata={"filename": filename},
                    ))

            # Cleanup
            shutil.rmtree(output_dir, ignore_errors=True)

            # Update notebook info
            info = self._notebooks[notebook_id]
            info.execution_count += 1
            info.last_executed_at = datetime.now(timezone.utc).isoformat()
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
        """Get status of a Kaggle kernel execution.

        Note: With access token auth, kernels_status() returns 403 Forbidden.
        This method falls back to kernels_list() to find the kernel status.
        """
        if not self._api or not self._username:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_name="NotInitialized",
                error_message="Kaggle backend not initialized",
            )

        kernel_id = f"{self._username}/{notebook_id}"

        status_map = {
            "queued": ExecutionStatus.QUEUED,
            "running": ExecutionStatus.RUNNING,
            "complete": ExecutionStatus.COMPLETED,
            "error": ExecutionStatus.FAILED,
            "cancelAcknowledged": ExecutionStatus.CANCELLED,
        }

        # Try SDK kernels_status first (works with legacy credentials)
        try:
            status = self._api.kernels_status(kernel_id)
            state = status.get("status", "unknown")
            return ExecutionResult(
                status=status_map.get(state, ExecutionStatus.PENDING),
            )
        except Exception as e:
            error_msg = str(e)
            # If 403 Forbidden, fall back to kernels_list
            if "403" not in error_msg:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error_name=type(e).__name__,
                    error_message=error_msg,
                )

        # Fallback: use kernels_list to find kernel status
        # This works with access token auth
        try:
            kernels = self._api.kernels_list(mine=True, page_size=50)
            for k in kernels:
                k_ref = getattr(k, 'ref', '') or str(k)
                if notebook_id in k_ref:
                    state = getattr(k, 'status', 'unknown')
                    return ExecutionResult(
                        status=status_map.get(state.lower() if state else 'unknown',
                                              ExecutionStatus.PENDING),
                    )
            # Kernel not found in list
            return ExecutionResult(
                status=ExecutionStatus.PENDING,
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

    def reset_notebook(self, notebook_id: str) -> str:
        """Reset notebook by clearing accumulated code and generating a new kernel ID.

        Since Kaggle kernels persist remotely, resetting a notebook requires a new
        kernel ID to avoid 409 Conflict errors on the next execution.

        Returns:
            The new notebook_id (kernel slug)
        """
        if notebook_id not in self._notebooks:
            return notebook_id

        # Generate a new kernel ID to avoid 409 Conflict with the old one
        new_notebook_id = f"jaato-{uuid.uuid4().hex}"
        self._trace(f"Reset: generating new notebook_id: {notebook_id} -> {new_notebook_id}")

        # Move all internal state to new notebook_id
        info = self._notebooks.pop(notebook_id)
        info.notebook_id = new_notebook_id
        info.variables = {}
        info.execution_count = 0
        self._notebooks[new_notebook_id] = info

        # Clear accumulated code
        self._notebook_code.pop(notebook_id, None)
        self._notebook_code[new_notebook_id] = []

        # Move temp directory reference
        if notebook_id in self._temp_dirs:
            self._temp_dirs[new_notebook_id] = self._temp_dirs.pop(notebook_id)

        return new_notebook_id

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
