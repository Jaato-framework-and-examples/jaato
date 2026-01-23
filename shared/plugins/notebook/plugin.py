"""Notebook plugin for Python code execution with GPU support.

This plugin provides interactive Python notebook capabilities:
- Execute Python code with state preserved across calls
- Multiple backend support (local, Kaggle GPU)
- Variable inspection and notebook management
- Streaming output support for real-time execution feedback
- Security analysis for sandbox compliance
"""

import asyncio
import io
import os
import queue
import tempfile
import threading
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from ..base import UserCommand, PermissionDisplayInfo
from ..model_provider.types import ToolSchema
from ..streaming.protocol import StreamingCapable, StreamChunk, ChunkCallback
from .types import ExecutionStatus, OutputType
from .backends import NotebookBackend, LocalJupyterBackend, KaggleBackend, _KAGGLE_AVAILABLE
from .code_analyzer import CodeAnalyzer, AnalysisResult, RiskLevel
from shared.ai_tool_runner import get_current_tool_output_callback


class SandboxMode(Enum):
    """Sandbox enforcement mode for notebook execution."""
    DISABLED = "disabled"      # No analysis, execute everything
    WARN = "warn"              # Analyze and show risks, but allow execution
    BLOCK_CRITICAL = "block_critical"  # Block CRITICAL risks, warn others
    STRICT = "strict"          # Block HIGH and CRITICAL risks


# Default to local backend
DEFAULT_BACKEND = "local"

# Max output size to return to model (avoid context overflow)
MAX_OUTPUT_LENGTH = 10000


class NotebookPlugin(StreamingCapable):
    """Plugin for Python notebook execution with GPU support.

    Provides tools for:
    - Creating and managing notebooks
    - Executing Python code with persistent state
    - Switching between local (instant) and Kaggle (GPU) backends
    - Variable inspection
    - Streaming execution output in real-time

    Configuration:
        default_backend: 'local' or 'kaggle' (default: 'local')
        enable_kaggle: Whether to enable Kaggle backend (default: True)
        max_output_length: Max output chars to return (default: 10000)

    Implements StreamingCapable for real-time output streaming during execution.
    """

    def __init__(self):
        self._backends: Dict[str, NotebookBackend] = {}
        self._active_backend_name: str = DEFAULT_BACKEND
        self._current_notebook_id: Optional[str] = None
        self._max_output_length: int = MAX_OUTPUT_LENGTH
        self._initialized = False
        self._agent_name: Optional[str] = None
        self._kaggle_enabled: bool = True  # Whether to try kaggle when requested
        self._kaggle_init_attempted: bool = False  # Lazy init flag
        # Callback for streaming output during execution (tail -f style)
        self._tool_output_callback: Optional[Callable[[str], None]] = None
        # Sandbox configuration
        self._workspace_root: Optional[str] = None
        self._sandbox_mode: SandboxMode = SandboxMode.WARN
        self._code_analyzer: Optional[CodeAnalyzer] = None
        self._plugin_registry = None  # Set via set_plugin_registry() for path authorization
        # Cache last analysis for permission display
        self._last_analysis: Optional[AnalysisResult] = None

    @property
    def name(self) -> str:
        return "notebook"

    def _trace(self, msg: str) -> None:
        """Write trace message for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [NOTEBOOK{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the notebook plugin.

        Args:
            config: Optional dict with:
                - default_backend: 'local' or 'kaggle'
                - enable_kaggle: Whether to enable Kaggle backend
                - max_output_length: Max output chars to return
                - agent_name: Agent name for trace logging
                - workspace_root: Workspace root for sandbox path validation
                - sandbox_mode: 'disabled', 'warn', 'block_critical', or 'strict'
        """
        config = config or {}
        self._config = config  # Store for lazy kaggle init
        self._agent_name = config.get("agent_name")
        self._max_output_length = config.get("max_output_length", MAX_OUTPUT_LENGTH)
        self._kaggle_enabled = config.get("enable_kaggle", True)

        # Initialize sandbox configuration
        self._workspace_root = config.get("workspace_root")
        sandbox_mode_str = config.get("sandbox_mode", "warn")
        try:
            self._sandbox_mode = SandboxMode(sandbox_mode_str)
        except ValueError:
            self._sandbox_mode = SandboxMode.WARN

        # Create code analyzer (will be rebuilt when plugin_registry is set)
        self._rebuild_code_analyzer()

        # Always initialize local backend
        local_backend = LocalJupyterBackend()
        local_backend.initialize(config)
        self._backends["local"] = local_backend

        # Kaggle backend is initialized lazily when first requested (gpu=true)
        # This avoids stalling during plugin init if kaggle auth is missing/slow
        if not _KAGGLE_AVAILABLE:
            self._trace("Kaggle backend not available: kaggle package not installed")

        # Set default backend (only local is available at init time)
        self._active_backend_name = "local"

        self._initialized = True
        self._trace(f"Initialized with backend={self._active_backend_name}")

    def _ensure_kaggle_backend(self) -> Optional[str]:
        """Lazily initialize kaggle backend on first use.

        Returns:
            None if successful, error message string if failed.
        """
        self._trace(f"_ensure_kaggle_backend: called, kaggle_in_backends={('kaggle' in self._backends)}, init_attempted={self._kaggle_init_attempted}, enabled={self._kaggle_enabled}, available={_KAGGLE_AVAILABLE}")

        if "kaggle" in self._backends:
            return None  # Already initialized

        if self._kaggle_init_attempted:
            return "Kaggle backend initialization already failed"

        self._kaggle_init_attempted = True

        if not self._kaggle_enabled:
            return "Kaggle backend disabled in config"

        if not _KAGGLE_AVAILABLE:
            return "Kaggle package not installed. Install with: pip install kaggle"

        try:
            kaggle_backend = KaggleBackend()
            kaggle_backend.set_trace_fn(self._trace)  # Wire up tracing
            kaggle_backend.initialize(getattr(self, '_config', None))
            self._backends["kaggle"] = kaggle_backend
            self._trace("Kaggle backend initialized successfully (lazy)")
            return None
        except Exception as e:
            self._trace(f"Kaggle backend initialization failed: {e}")
            return str(e)

    def shutdown(self) -> None:
        """Shutdown all backends."""
        for backend in self._backends.values():
            try:
                backend.shutdown()
            except Exception:
                pass
        self._backends.clear()
        self._current_notebook_id = None
        self._initialized = False
        self._kaggle_init_attempted = False

    def set_workspace_path(self, path: str) -> None:
        """Set workspace root path (auto-wired by PluginRegistry).

        This is called during plugin registration to enable sandbox path
        validation for notebook code execution.

        Args:
            path: Absolute path to the workspace root directory.
        """
        self._workspace_root = path
        self._rebuild_code_analyzer()
        self._trace(f"Workspace path set to: {path}")

    def set_plugin_registry(self, registry) -> None:
        """Set the plugin registry for checking external path authorization.

        This is called during plugin registration to enable path authorization
        checks via the registry (e.g., for whitelisted external paths).

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry
        self._rebuild_code_analyzer()
        self._trace("set_plugin_registry: registry set")

    def _rebuild_code_analyzer(self) -> None:
        """Rebuild the code analyzer with current configuration.

        Called when workspace_root or plugin_registry changes to ensure
        the analyzer uses the latest sandbox settings.
        """
        self._code_analyzer = CodeAnalyzer(
            workspace_root=self._workspace_root,
            plugin_registry=self._plugin_registry,
            allow_tmp=True,  # Allow /tmp access like other sandboxed tools
        )

    def set_tool_output_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set the callback for streaming output during execution.

        When set, the plugin will stream formatted notebook output to the callback
        during code execution, enabling live preview in the UI tool tree.

        Args:
            callback: Function that accepts output chunks, or None to disable.
        """
        self._tool_output_callback = callback
        self._trace(f"set_tool_output_callback: callback={'SET' if callback else 'CLEARED'}")

    def _get_effective_output_callback(self) -> Optional[Callable[[str], None]]:
        """Get the effective output callback for the current execution.

        Checks thread-local storage first (for parallel execution),
        then falls back to the instance-level callback.

        Returns:
            The callback to use, or None if not set.
        """
        # Thread-local takes priority (parallel execution)
        thread_callback = get_current_tool_output_callback()
        if thread_callback is not None:
            return thread_callback
        # Fall back to instance-level (sequential execution)
        return self._tool_output_callback

    @property
    def _active_backend(self) -> NotebookBackend:
        """Get the currently active backend."""
        return self._backends[self._active_backend_name]

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for notebook operations."""
        return [
            ToolSchema(
                name="notebook_execute",
                description=(
                    "Execute Python code in a persistent notebook environment. "
                    "Variables and imports persist across executions. "
                    "Use for data analysis, calculations, and prototyping."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        },
                        "notebook_id": {
                            "type": "string",
                            "description": "Optional notebook ID. Creates new if not specified."
                        },
                    },
                    "required": ["code"]
                },
                category="code",
                discoverability="core",
            ),
            ToolSchema(
                name="notebook_create",
                description=(
                    "Create a new Python notebook. Use gpu=true for GPU-accelerated "
                    "computing (via Kaggle, async execution). Default is local (instant)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the notebook"
                        },
                        "gpu": {
                            "type": "boolean",
                            "description": "Enable GPU (uses Kaggle backend, async)",
                            "default": False
                        },
                    },
                    "required": ["name"]
                },
                category="code",
            ),
            ToolSchema(
                name="notebook_variables",
                description="Get all variables defined in the current notebook with their types.",
                parameters={
                    "type": "object",
                    "properties": {
                        "notebook_id": {
                            "type": "string",
                            "description": "Notebook ID. Uses current notebook if not specified."
                        },
                    },
                },
                category="code",
            ),
            ToolSchema(
                name="notebook_reset",
                description="Reset the notebook, clearing all variables and execution state.",
                parameters={
                    "type": "object",
                    "properties": {
                        "notebook_id": {
                            "type": "string",
                            "description": "Notebook ID. Uses current notebook if not specified."
                        },
                    },
                },
                category="code",
            ),
            ToolSchema(
                name="notebook_list",
                description="List all active notebooks with their backends and status.",
                parameters={
                    "type": "object",
                    "properties": {},
                },
                category="code",
            ),
            ToolSchema(
                name="notebook_backends",
                description="List available notebook backends and their capabilities (GPU, quotas).",
                parameters={
                    "type": "object",
                    "properties": {},
                },
                category="code",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mappings."""
        return {
            "notebook_execute": self._execute_code,
            "notebook_create": self._create_notebook,
            "notebook_variables": self._get_variables,
            "notebook_reset": self._reset_notebook,
            "notebook_list": self._list_notebooks,
            "notebook_backends": self._list_backends,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for notebook tools."""
        backends_info = []
        for name, backend in self._backends.items():
            caps = backend.capabilities
            gpu_info = f"GPU: {caps.gpu_type}" if caps.supports_gpu else "No GPU"
            backends_info.append(f"- {name}: {gpu_info}")

        # Sandbox information
        sandbox_info = ""
        if self._sandbox_mode != SandboxMode.DISABLED:
            sandbox_info = f"""

**Sandbox Restrictions (mode: {self._sandbox_mode.value}):**
- Code is analyzed for security risks before execution
- File access should use workspace-relative paths only
- Shell commands (!) and subprocess are flagged as high risk
- External path references (e.g., /home/, /etc/) will be blocked
- Use the file_edit or filesystem_query tools for file operations instead
"""
            if self._workspace_root:
                sandbox_info += f"- Workspace root: {self._workspace_root}\n"

        return f"""You have access to Python notebook tools for executing code:

**Available Backends:**
{chr(10).join(backends_info)}

**Key Tools:**
- `notebook_execute`: Run Python code with persistent state (variables preserved)
- `notebook_create`: Create a new notebook (use gpu=true for GPU computing)
- `notebook_variables`: Inspect defined variables
- `notebook_reset`: Clear notebook state

**Usage Tips:**
- For quick calculations: Just use notebook_execute (auto-creates local notebook)
- For GPU workloads: First create a notebook with gpu=true, then execute
- Variables persist across executions in the same notebook
- Use !pip install package for installing packages

**GPU Backend (Kaggle):**
- 30 hours/week free GPU (P100/T4)
- Execution is async (may take 1-5 minutes)
- Best for: ML training, large computations
{sandbox_info}"""

    def get_auto_approved_tools(self) -> List[str]:
        """Read-only tools are auto-approved."""
        return ["notebook_variables", "notebook_list", "notebook_backends"]

    def get_user_commands(self) -> List[UserCommand]:
        """No user commands for now."""
        return []

    def format_permission_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        channel_type: str,
    ) -> Optional[PermissionDisplayInfo]:
        """Format code execution for permission display."""
        if tool_name == "notebook_execute":
            code = arguments.get("code", "")
            notebook_id = arguments.get("notebook_id", self._current_notebook_id or "new")

            # Get backend info
            backend_name = self._active_backend_name
            if notebook_id and notebook_id in self._get_all_notebooks():
                # Find which backend has this notebook
                for name, backend in self._backends.items():
                    for nb in backend.list_notebooks():
                        if nb.notebook_id == notebook_id:
                            backend_name = name
                            break

            # Analyze code for security risks
            analysis_info = ""
            if self._sandbox_mode != SandboxMode.DISABLED and self._code_analyzer:
                analysis = self._code_analyzer.analyze(code)
                self._last_analysis = analysis

                if analysis.has_risks:
                    # Build risk summary for permission display
                    risk_lines = [
                        "",
                        "--- Security Analysis ---",
                        analysis.get_summary(),
                    ]
                    if analysis.external_paths:
                        risk_lines.append(f"External paths: {', '.join(analysis.external_paths[:3])}")
                    risk_lines.append("")
                    risk_lines.append(analysis.format_risks(max_items=5))
                    analysis_info = "\n".join(risk_lines)

            # Combine code and analysis
            details = code
            if analysis_info:
                details = code + "\n" + analysis_info

            summary = f"Execute IPython ({backend_name}): {notebook_id}"
            if self._last_analysis and self._last_analysis.has_risks:
                max_level = self._last_analysis.max_risk_level
                if max_level:
                    summary += f" [{max_level.value.upper()} RISK]"

            return PermissionDisplayInfo(
                summary=summary,
                details=details,
                format_hint="code",
                language="ipython",
            )
        return None

    def _get_all_notebooks(self) -> Dict[str, str]:
        """Get all notebook IDs mapped to their backend names."""
        notebooks = {}
        for name, backend in self._backends.items():
            for nb in backend.list_notebooks():
                notebooks[nb.notebook_id] = name
        return notebooks

    def _execute_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code in a notebook."""
        code = args.get("code", "")
        notebook_id = args.get("notebook_id")

        if not code.strip():
            return {"error": "No code provided"}

        # Analyze code for security risks (if sandbox is enabled)
        if self._sandbox_mode != SandboxMode.DISABLED and self._code_analyzer:
            analysis = self._code_analyzer.analyze(code)
            self._last_analysis = analysis

            if analysis.has_risks:
                max_level = analysis.max_risk_level

                # Check if we should block execution
                should_block = False
                if self._sandbox_mode == SandboxMode.STRICT:
                    should_block = max_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
                elif self._sandbox_mode == SandboxMode.BLOCK_CRITICAL:
                    should_block = max_level == RiskLevel.CRITICAL

                if should_block:
                    self._trace(f"Blocked code execution: {analysis.get_summary()}")
                    return {
                        "error": "Code blocked by sandbox",
                        "reason": analysis.get_summary(),
                        "details": analysis.format_risks(max_items=10),
                        "external_paths": analysis.external_paths,
                        "hint": "Code contains patterns that could bypass workspace sandboxing. "
                                "Use workspace-relative paths and avoid subprocess/shell commands.",
                    }

                # Log warning for non-blocking risks
                self._trace(f"Code analysis warning: {analysis.get_summary()}")

        # Auto-create notebook if needed
        if not notebook_id:
            if self._current_notebook_id:
                notebook_id = self._current_notebook_id
            else:
                # Create a default local notebook
                result = self._create_notebook({"name": "default", "gpu": False})
                if "error" in result:
                    return result
                notebook_id = result["notebook_id"]

        # Find which backend has this notebook
        backend = None
        for name, b in self._backends.items():
            for nb in b.list_notebooks():
                if nb.notebook_id == notebook_id:
                    backend = b
                    break
            if backend:
                break

        if not backend:
            return {"error": f"Notebook {notebook_id} not found"}

        self._trace(f"Executing in {notebook_id}: {code[:50]}...")

        # Execute
        result = backend.execute(notebook_id, code)

        # Format response
        response: Dict[str, Any] = {
            "notebook_id": notebook_id,
            "status": result.status.value,
            "execution_count": result.execution_count,
        }

        # Get effective callback (checks thread-local for parallel execution)
        output_callback = self._get_effective_output_callback()

        if result.status == ExecutionStatus.COMPLETED:
            # Format output with notebook cell markers for the formatter pipeline
            exec_count = result.execution_count or 1
            output_parts = []

            # Input cell with the code
            input_cell = self._format_input_cell(code, exec_count)
            output_parts.append(input_cell)
            # Stream to UI if callback is set
            if output_callback:
                output_callback(input_cell + "\n")

            # Output cells for each type
            for output in result.outputs:
                cell_output = None
                if output.output_type == OutputType.STDOUT and output.content:
                    cell_output = self._format_stdout_cell(output.content, exec_count)
                elif output.output_type == OutputType.RESULT and output.content:
                    cell_output = self._format_result_cell(output.content, exec_count)
                elif output.output_type == OutputType.STDERR and output.content:
                    cell_output = self._format_stderr_cell(output.content, exec_count)
                elif output.output_type == OutputType.DISPLAY:
                    mime = output.mime_type or "unknown"
                    if mime.startswith("image/"):
                        content = f"[Image: {mime}]"
                    else:
                        content = output.content[:500] if output.content else ""
                    cell_output = self._format_display_cell(content, exec_count)

                if cell_output:
                    output_parts.append(cell_output)
                    # Stream to UI if callback is set
                    if output_callback:
                        output_callback(cell_output + "\n")

            response["output"] = "\n".join(output_parts)
            response["variables"] = result.variables
            if result.duration_seconds:
                response["duration_seconds"] = round(result.duration_seconds, 2)

        elif result.status == ExecutionStatus.FAILED:
            exec_count = result.execution_count or 1
            # Format error with markers
            input_cell = self._format_input_cell(code, exec_count)
            error_content = result.error_message or "Unknown error"
            if result.traceback:
                error_content += f"\n{result.traceback}"
            error_cell = self._format_error_cell(error_content, exec_count)

            response["output"] = input_cell + "\n" + error_cell
            response["error"] = result.error_message

            # Stream to UI if callback is set
            if output_callback:
                output_callback(input_cell + "\n")
                output_callback(error_cell + "\n")

        elif result.status in (ExecutionStatus.QUEUED, ExecutionStatus.RUNNING):
            response["message"] = "Execution in progress (async backend). Poll with notebook_variables to check status."

        # Log response summary
        output_len = len(response.get("output", ""))
        error = response.get("error", "")
        self._trace(f"Response: status={result.status.value}, output_len={output_len}, error={error[:100] if error else 'none'}")

        return response

    def _create_notebook(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new notebook."""
        name = args.get("name", "untitled")
        gpu = args.get("gpu", False)
        self._trace(f"_create_notebook: name={name}, gpu={gpu}")

        # Choose backend based on GPU requirement
        if gpu:
            # Lazy init kaggle backend on first GPU request
            error = self._ensure_kaggle_backend()
            if error:
                return {
                    "error": f"GPU backend (Kaggle) not available: {error}",
                    "hint": "Install kaggle package and configure credentials",
                }
            backend = self._backends["kaggle"]
        else:
            backend = self._backends["local"]

        try:
            info = backend.create_notebook(name, gpu_enabled=gpu)
            self._current_notebook_id = info.notebook_id

            self._trace(f"Created notebook {info.notebook_id} on {info.backend}")

            return {
                "notebook_id": info.notebook_id,
                "name": info.name,
                "backend": info.backend,
                "gpu_enabled": info.gpu_enabled,
                "message": f"Notebook created. Use notebook_execute with code to run Python.",
            }
        except Exception as e:
            return {"error": f"Failed to create notebook: {e}"}

    def _get_variables(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get variables from a notebook."""
        notebook_id = args.get("notebook_id", self._current_notebook_id)

        if not notebook_id:
            return {"error": "No notebook specified and no current notebook"}

        # Find the backend
        for name, backend in self._backends.items():
            for nb in backend.list_notebooks():
                if nb.notebook_id == notebook_id:
                    variables = backend.get_variables(notebook_id)
                    return {
                        "notebook_id": notebook_id,
                        "backend": name,
                        "variable_count": len(variables),
                        "variables": variables,
                    }

        return {"error": f"Notebook {notebook_id} not found"}

    def _reset_notebook(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Reset a notebook."""
        notebook_id = args.get("notebook_id", self._current_notebook_id)

        if not notebook_id:
            return {"error": "No notebook specified and no current notebook"}

        # Find and reset
        for name, backend in self._backends.items():
            for nb in backend.list_notebooks():
                if nb.notebook_id == notebook_id:
                    new_notebook_id = backend.reset_notebook(notebook_id)
                    # Update current notebook reference if it changed
                    if self._current_notebook_id == notebook_id:
                        self._current_notebook_id = new_notebook_id
                    result = {
                        "notebook_id": new_notebook_id,
                        "message": "Notebook reset. All variables cleared.",
                    }
                    if new_notebook_id != notebook_id:
                        result["message"] += f" New notebook_id: {new_notebook_id}"
                        result["previous_notebook_id"] = notebook_id
                    return result

        return {"error": f"Notebook {notebook_id} not found"}

    def _list_notebooks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all notebooks."""
        notebooks = []
        for name, backend in self._backends.items():
            for nb in backend.list_notebooks():
                notebooks.append({
                    "notebook_id": nb.notebook_id,
                    "name": nb.name,
                    "backend": nb.backend,
                    "gpu_enabled": nb.gpu_enabled,
                    "execution_count": nb.execution_count,
                    "variable_count": len(nb.variables),
                })

        return {
            "notebook_count": len(notebooks),
            "current_notebook": self._current_notebook_id,
            "notebooks": notebooks,
        }

    def _list_backends(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List available backends and their capabilities."""
        backends = []
        for name, backend in self._backends.items():
            caps = backend.capabilities
            backends.append({
                "name": caps.name,
                "available": backend.is_available(),
                "supports_gpu": caps.supports_gpu,
                "gpu_type": caps.gpu_type,
                "max_runtime_hours": caps.max_runtime_hours,
                "weekly_quota_hours": caps.weekly_quota_hours,
                "is_async": caps.is_async,
                "requires_auth": caps.requires_auth,
            })

        # Show kaggle as potential backend if available but not yet initialized
        if "kaggle" not in self._backends and _KAGGLE_AVAILABLE and self._kaggle_enabled:
            backends.append({
                "name": "kaggle",
                "available": "not_initialized",  # Will be initialized on first GPU request
                "supports_gpu": True,
                "gpu_type": "P100/T4",
                "max_runtime_hours": 9.0,
                "weekly_quota_hours": 30.0,
                "is_async": True,
                "requires_auth": True,
                "note": "Will initialize on first GPU notebook request",
            })

        return {
            "active_backend": self._active_backend_name,
            "backends": backends,
        }

    # ==================== StreamingCapable Implementation ====================

    def supports_streaming(self, tool_name: str) -> bool:
        """Check if a tool supports streaming execution.

        Only notebook_execute supports streaming for now.
        """
        return tool_name == "notebook_execute"

    def get_streaming_tool_names(self) -> List[str]:
        """Get list of tools that support streaming."""
        return ["notebook_execute"]

    async def execute_streaming(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        on_chunk: Optional[ChunkCallback] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute notebook code with streaming output.

        Yields StreamChunks as stdout/stderr/results become available.
        Output is wrapped with <notebook-cell> markers for the formatter pipeline.

        Args:
            tool_name: Should be "notebook_execute".
            arguments: Tool arguments (code, notebook_id).
            on_chunk: Optional callback for each chunk.

        Yields:
            StreamChunk objects for input, stdout, stderr, result, and errors.
        """
        if tool_name != "notebook_execute":
            raise ValueError(f"Streaming not supported for tool: {tool_name}")

        code = arguments.get("code", "")
        notebook_id = arguments.get("notebook_id")

        if not code.strip():
            chunk = StreamChunk(
                content="<notebook-cell type=\"error\">No code provided</notebook-cell>",
                chunk_type="error",
            )
            if on_chunk:
                on_chunk(chunk)
            yield chunk
            return

        # Auto-create notebook if needed
        if not notebook_id:
            if self._current_notebook_id:
                notebook_id = self._current_notebook_id
            else:
                result = self._create_notebook({"name": "default", "gpu": False})
                if "error" in result:
                    chunk = StreamChunk(
                        content=f"<notebook-cell type=\"error\">{result['error']}</notebook-cell>",
                        chunk_type="error",
                    )
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk
                    return
                notebook_id = result["notebook_id"]

        # Find the backend
        backend = None
        backend_name = "local"
        for name, b in self._backends.items():
            for nb in b.list_notebooks():
                if nb.notebook_id == notebook_id:
                    backend = b
                    backend_name = name
                    break
            if backend:
                break

        if not backend:
            chunk = StreamChunk(
                content=f"<notebook-cell type=\"error\">Notebook {notebook_id} not found</notebook-cell>",
                chunk_type="error",
            )
            if on_chunk:
                on_chunk(chunk)
            yield chunk
            return

        # Get current execution count (will be incremented after execution)
        exec_count = backend._execution_counts.get(notebook_id, 0) + 1

        self._trace(f"Streaming execution in {notebook_id}: {code[:50]}...")

        # Yield the input cell first (so formatter can display In[n]:)
        input_chunk = StreamChunk(
            content=self._format_input_cell(code, exec_count),
            chunk_type="input",
            metadata={"notebook_id": notebook_id, "execution_count": exec_count},
        )
        if on_chunk:
            on_chunk(input_chunk)
        yield input_chunk

        # Execute with streaming output capture
        async for chunk in self._execute_streaming_impl(
            backend, notebook_id, code, exec_count, on_chunk
        ):
            yield chunk

    async def _execute_streaming_impl(
        self,
        backend: NotebookBackend,
        notebook_id: str,
        code: str,
        exec_count: int,
        on_chunk: Optional[ChunkCallback] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Implementation of streaming execution using a background thread.

        Runs the execution in a thread and yields output chunks as they arrive.
        """
        output_queue: queue.Queue = queue.Queue()
        result_holder: List[Any] = [None]  # To hold the final result
        error_holder: List[Optional[Exception]] = [None]

        def run_execution():
            """Run execution in a background thread."""
            try:
                result = backend.execute(notebook_id, code)
                result_holder[0] = result
            except Exception as e:
                error_holder[0] = e

        # Start execution in background thread
        exec_thread = threading.Thread(target=run_execution, daemon=True)
        exec_thread.start()

        # Wait for execution to complete (with periodic checks)
        while exec_thread.is_alive():
            await asyncio.sleep(0.1)

        # Check for errors
        if error_holder[0]:
            chunk = StreamChunk(
                content=f"<notebook-cell type=\"error\">{str(error_holder[0])}</notebook-cell>",
                chunk_type="error",
            )
            if on_chunk:
                on_chunk(chunk)
            yield chunk
            return

        result = result_holder[0]
        if result is None:
            chunk = StreamChunk(
                content="<notebook-cell type=\"error\">Execution returned no result</notebook-cell>",
                chunk_type="error",
            )
            if on_chunk:
                on_chunk(chunk)
            yield chunk
            return

        # Yield output chunks based on result
        sequence = 1

        if result.status == ExecutionStatus.COMPLETED:
            for output in result.outputs:
                if output.output_type == OutputType.STDOUT and output.content:
                    chunk = StreamChunk(
                        content=self._format_stdout_cell(output.content, exec_count),
                        chunk_type="stdout",
                        sequence=sequence,
                        metadata={"notebook_id": notebook_id, "execution_count": exec_count},
                    )
                    sequence += 1
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk

                elif output.output_type == OutputType.STDERR and output.content:
                    chunk = StreamChunk(
                        content=self._format_stderr_cell(output.content, exec_count),
                        chunk_type="stderr",
                        sequence=sequence,
                        metadata={"notebook_id": notebook_id, "execution_count": exec_count},
                    )
                    sequence += 1
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk

                elif output.output_type == OutputType.RESULT and output.content:
                    chunk = StreamChunk(
                        content=self._format_result_cell(output.content, exec_count),
                        chunk_type="result",
                        sequence=sequence,
                        metadata={"notebook_id": notebook_id, "execution_count": exec_count},
                    )
                    sequence += 1
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk

                elif output.output_type == OutputType.DISPLAY:
                    mime = output.mime_type or "unknown"
                    if mime.startswith("image/"):
                        content = f"[Image: {mime}]"
                    else:
                        content = output.content[:500] if output.content else ""
                    chunk = StreamChunk(
                        content=self._format_display_cell(content, exec_count),
                        chunk_type="display",
                        sequence=sequence,
                        metadata={"notebook_id": notebook_id, "execution_count": exec_count, "mime": mime},
                    )
                    sequence += 1
                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk

        elif result.status == ExecutionStatus.FAILED:
            error_content = result.error_message or "Unknown error"
            if result.traceback:
                error_content += f"\n{result.traceback}"
            chunk = StreamChunk(
                content=self._format_error_cell(error_content, exec_count),
                chunk_type="error",
                sequence=sequence,
                metadata={"notebook_id": notebook_id, "execution_count": exec_count},
            )
            if on_chunk:
                on_chunk(chunk)
            yield chunk

        # Final summary chunk
        duration = result.duration_seconds or 0
        summary = f"Execution completed in {duration:.2f}s"
        if result.variables:
            summary += f", {len(result.variables)} variables defined"

        yield StreamChunk(
            content=summary,
            chunk_type="summary",
            sequence=sequence + 1,
            metadata={
                "notebook_id": notebook_id,
                "execution_count": exec_count,
                "status": result.status.value,
                "duration_seconds": duration,
                "variables": result.variables,
            },
        )

    # ==================== Notebook Cell Formatting ====================
    #
    # These methods emit <nb-row> markers directly for client-side rendering.
    # The format is: <nb-row type="..." label="...">content</nb-row>
    #
    # Labels:
    #   input  → "In [n]:"
    #   stdout → "Out [n]:"
    #   stderr → "Err [n]:"
    #   result → "Out [n]:"
    #   display → "Out [n]:"
    #   error  → "Err [n]:"

    def _format_input_cell(self, code: str, exec_count: int) -> str:
        """Format code as an input cell with nb-row markers.

        Uses 'ipython' language to skip LSP validation (supports !shell, %magic).
        """
        label = f"In [{exec_count}]:"
        return f'<nb-row type="input" label="{label}">\n```ipython\n{code}\n```\n</nb-row>'

    def _format_stdout_cell(self, content: str, exec_count: int) -> str:
        """Format stdout output with nb-row markers."""
        if len(content) > self._max_output_length:
            content = content[:self._max_output_length] + "\n... (truncated)"
        label = f"Out [{exec_count}]:"
        return f'<nb-row type="stdout" label="{label}">\n{content}\n</nb-row>'

    def _format_stderr_cell(self, content: str, exec_count: int) -> str:
        """Format stderr output with nb-row markers."""
        if len(content) > self._max_output_length:
            content = content[:self._max_output_length] + "\n... (truncated)"
        label = f"Err [{exec_count}]:"
        return f'<nb-row type="stderr" label="{label}">\n{content}\n</nb-row>'

    def _format_result_cell(self, content: str, exec_count: int) -> str:
        """Format execution result with nb-row markers."""
        if len(content) > self._max_output_length:
            content = content[:self._max_output_length] + "\n... (truncated)"
        label = f"Out [{exec_count}]:"
        return f'<nb-row type="result" label="{label}">\n{content}\n</nb-row>'

    def _format_display_cell(self, content: str, exec_count: int) -> str:
        """Format display output (images, etc.) with nb-row markers."""
        label = f"Out [{exec_count}]:"
        return f'<nb-row type="display" label="{label}">\n{content}\n</nb-row>'

    def _format_error_cell(self, content: str, exec_count: int) -> str:
        """Format error output with nb-row markers."""
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        label = f"Err [{exec_count}]:"
        return f'<nb-row type="error" label="{label}">\n{content}\n</nb-row>'


def create_plugin() -> NotebookPlugin:
    """Factory function to create the notebook plugin."""
    return NotebookPlugin()
