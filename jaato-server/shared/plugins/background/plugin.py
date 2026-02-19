"""Background task orchestrator plugin.

This plugin provides tools for the model to manage background task execution
across all BackgroundCapable plugins in the registry.
"""

import logging
import os
import tempfile
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from jaato_sdk.plugins.model_provider.types import ToolSchema

from jaato_sdk.plugins.base import ToolPlugin, UserCommand
from .protocol import BackgroundCapable, TaskHandle, TaskResult, TaskStatus
from shared.trace import trace as _trace_write

if TYPE_CHECKING:
    from ..registry import PluginRegistry


PLUGIN_KIND = "tool"


class BackgroundPlugin:
    """Orchestrator for background task execution.

    This plugin provides tools for the model to:
    - Start tasks in background (when it anticipates long execution)
    - Check status of running tasks
    - Retrieve results of completed tasks
    - Cancel running tasks
    - List all active tasks

    It discovers BackgroundCapable plugins via the registry and
    delegates actual execution to them.
    """

    def __init__(self):
        self._registry: Optional['PluginRegistry'] = None
        self._capable_plugins: Dict[str, BackgroundCapable] = {}
        self._initialized = False
        self._agent_name: Optional[str] = None

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("BACKGROUND", msg)

    @property
    def name(self) -> str:
        return "background"

    def set_registry(self, registry: 'PluginRegistry') -> None:
        """Set the plugin registry for capability discovery.

        Called by JaatoClient after registry is configured.

        Args:
            registry: The plugin registry to use for discovering
                      BackgroundCapable plugins.
        """
        self._registry = registry
        self._discover_capable_plugins()

    def _discover_capable_plugins(self) -> None:
        """Scan registry for BackgroundCapable plugins."""
        if not self._registry:
            return

        self._capable_plugins.clear()

        for plugin_name in self._registry.list_exposed():
            plugin = self._registry.get_plugin(plugin_name)
            # Skip self
            if plugin is self:
                continue
            if plugin and isinstance(plugin, BackgroundCapable):
                self._capable_plugins[plugin_name] = plugin

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the background plugin.

        Args:
            config: Optional configuration dict (currently unused).
        """
        config = config or {}
        self._agent_name = config.get("agent_name")
        self._initialized = True
        self._trace("initialize")

    def shutdown(self) -> None:
        """Shutdown the background plugin."""
        self._trace("shutdown")
        # Clean up all tasks across all capable plugins
        for plugin in self._capable_plugins.values():
            try:
                plugin.cleanup_completed(max_age_seconds=0)
            except Exception as exc:
                logger.debug(f"Failed to cleanup plugin during shutdown: {exc}")
        self._initialized = False

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for background task management tools."""
        return [
            ToolSchema(
                name="startBackgroundTask",
                description="""Start a tool execution in the background.

Use this when you anticipate a tool call will take significant time
(e.g., long builds, installs, complex searches). The task runs
asynchronously and you can continue with other work.

Returns a task_id you can use to check status or get results later.""",
                parameters={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool to execute"
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments to pass to the tool"
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "description": "Optional timeout in seconds"
                        },
                    },
                    "required": ["tool_name", "arguments"]
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="getBackgroundTask",
                description="""Get status, output, and result of a background task.

This single tool handles everything - works the same whether the task
is running or completed:
- While running: returns current status + stdout/stderr so far
- When completed: returns final status + full output + returncode

Use stdout_offset to get incremental output (for monitoring progress).
Call repeatedly while has_more is true to poll for completion.

Response fields:
- status: "pending", "running", "completed", "failed", "cancelled"
- stdout/stderr: output since the specified offset
- stdout_offset/stderr_offset: use these for the next call
- has_more: true if task still running (more output expected)
- returncode: exit code when completed (null while running)
- error: error message if failed
- duration_seconds: execution time when completed""",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID returned from startBackgroundTask or auto-backgrounded command"
                        },
                        "stdout_offset": {
                            "type": "integer",
                            "description": "Byte offset to read stdout from (default: 0)"
                        },
                        "stderr_offset": {
                            "type": "integer",
                            "description": "Byte offset to read stderr from (default: 0)"
                        },
                    },
                    "required": ["task_id"]
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="cancelBackgroundTask",
                description="Cancel a running background task.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to cancel"
                        },
                    },
                    "required": ["task_id"]
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="listBackgroundTasks",
                description="List all active background tasks across all plugins.",
                parameters={
                    "type": "object",
                    "properties": {
                        "plugin_name": {
                            "type": "string",
                            "description": "Optional: filter by plugin name"
                        },
                    },
                },
                category="system",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="listBackgroundCapableTools",
                description="""List all tools that support background execution.

Use this to discover which tools can be run in background mode.""",
                parameters={
                    "type": "object",
                    "properties": {},
                },
                category="system",
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mapping for background task tools."""
        return {
            "startBackgroundTask": self._start_task,
            "getBackgroundTask": self._get_task,
            "cancelBackgroundTask": self._cancel_task,
            "listBackgroundTasks": self._list_tasks,
            "listBackgroundCapableTools": self._list_capable_tools,
        }

    def _start_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Start a background task.

        Args:
            args: Dict containing tool_name, arguments,
                  and optional timeout_seconds.

        Returns:
            Dict with success status and task handle info or error.
        """
        tool_name = args.get("tool_name")
        arguments = args.get("arguments", {})
        timeout = args.get("timeout_seconds")
        self._trace(f"startBackgroundTask: tool={tool_name}, timeout={timeout}")

        if not tool_name:
            return {
                "success": False,
                "error": "tool_name is required"
            }

        # Refresh capable plugins list
        self._discover_capable_plugins()

        # Find the plugin that provides this tool
        plugin = None
        plugin_name = None
        for pname, p in self._capable_plugins.items():
            if p.supports_background(tool_name):
                plugin = p
                plugin_name = pname
                break

        if plugin is None:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' does not support background execution. "
                         f"Use listBackgroundCapableTools to see available tools."
            }

        try:
            handle = plugin.start_background(tool_name, arguments, timeout)
            return {
                "success": True,
                "task_id": handle.task_id,
                "plugin_name": handle.plugin_name,
                "tool_name": handle.tool_name,
                "estimated_duration_seconds": handle.estimated_duration_seconds,
                "message": f"Task started in background. "
                           f"Use task_id '{handle.task_id}' to check status."
            }
        except Exception as e:
            logger.error(f"Failed to start background task: {tool_name}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _get_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get unified status, output, and result of a background task.

        Args:
            args: Dict containing task_id, and optional stdout_offset
                  and stderr_offset.

        Returns:
            Dict with task info including status, output, and result.
        """
        task_id = args.get("task_id")
        stdout_offset = args.get("stdout_offset", 0)
        stderr_offset = args.get("stderr_offset", 0)
        self._trace(
            f"getBackgroundTask: task_id={task_id}, "
            f"stdout_offset={stdout_offset}, stderr_offset={stderr_offset}"
        )

        if not task_id:
            return {"error": "task_id is required"}

        # Refresh capable plugins list
        self._discover_capable_plugins()

        for plugin_name, plugin in self._capable_plugins.items():
            try:
                # Check if plugin supports get_task
                if not hasattr(plugin, 'get_task'):
                    # Fall back to get_status for basic info
                    status = plugin.get_status(task_id)
                    return {
                        "task_id": task_id,
                        "plugin_name": plugin_name,
                        "status": status.value,
                        "stdout": "",
                        "stderr": "",
                        "stdout_offset": 0,
                        "stderr_offset": 0,
                        "has_more": status.value in ("pending", "running"),
                        "returncode": None,
                        "error": None,
                        "duration_seconds": None,
                    }

                info = plugin.get_task(
                    task_id,
                    stdout_offset=stdout_offset,
                    stderr_offset=stderr_offset,
                )
                return {
                    "task_id": info.task_id,
                    "plugin_name": plugin_name,
                    "status": info.status.value,
                    "stdout": info.stdout,
                    "stderr": info.stderr,
                    "stdout_offset": info.stdout_offset,
                    "stderr_offset": info.stderr_offset,
                    "has_more": info.has_more,
                    "returncode": info.returncode,
                    "error": info.error,
                    "duration_seconds": info.duration_seconds,
                }
            except KeyError:
                continue

        return {"error": f"Task '{task_id}' not found"}

    def _cancel_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a background task.

        Args:
            args: Dict containing task_id.

        Returns:
            Dict with cancellation result.
        """
        task_id = args.get("task_id")
        self._trace(f"cancelBackgroundTask: task_id={task_id}")
        if not task_id:
            return {"error": "task_id is required"}

        for plugin_name, plugin in self._capable_plugins.items():
            try:
                # Check if task exists
                plugin.get_status(task_id)
                success = plugin.cancel(task_id)
                return {
                    "task_id": task_id,
                    "cancelled": success,
                }
            except KeyError:
                continue

        return {"error": f"Task '{task_id}' not found"}

    def _list_tasks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all active background tasks.

        Args:
            args: Dict with optional plugin_name filter.

        Returns:
            Dict with list of active tasks.
        """
        filter_plugin = args.get("plugin_name")
        self._trace(f"listBackgroundTasks: filter={filter_plugin}")

        all_tasks = []
        for plugin_name, plugin in self._capable_plugins.items():
            if filter_plugin and plugin_name != filter_plugin:
                continue

            try:
                for handle in plugin.list_tasks():
                    all_tasks.append({
                        "task_id": handle.task_id,
                        "plugin_name": handle.plugin_name,
                        "tool_name": handle.tool_name,
                        "created_at": handle.created_at.isoformat(),
                        "estimated_duration_seconds": handle.estimated_duration_seconds,
                    })
            except Exception as exc:
                logger.debug(f"Failed to list tasks from plugin {plugin_name}: {exc}")

        return {
            "tasks": all_tasks,
            "count": len(all_tasks),
        }

    def _list_capable_tools(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all tools that support background execution.

        Args:
            args: Dict (unused, included for consistency).

        Returns:
            Dict with list of background-capable tools.
        """
        self._trace("listBackgroundCapableTools")
        # Refresh capable plugins list
        self._discover_capable_plugins()

        capable_tools = []

        for plugin_name, plugin in self._capable_plugins.items():
            # Get all tool names from the plugin's executors
            base_plugin = self._registry.get_plugin(plugin_name) if self._registry else None
            if base_plugin and hasattr(base_plugin, 'get_executors'):
                for tool_name in base_plugin.get_executors().keys():
                    try:
                        if plugin.supports_background(tool_name):
                            # Also get auto-background threshold if available
                            threshold = None
                            if hasattr(plugin, 'get_auto_background_threshold'):
                                threshold = plugin.get_auto_background_threshold(tool_name)
                            capable_tools.append({
                                "plugin_name": plugin_name,
                                "tool_name": tool_name,
                                "auto_background_threshold_seconds": threshold,
                            })
                    except Exception as exc:
                        logger.debug(f"Failed to check background support for {tool_name}: {exc}")

        return {
            "tools": capable_tools,
            "count": len(capable_tools),
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for background task tools."""
        return """## Background Task Execution

You have the ability to run long-running tool executions in the background.

**When to use background execution:**
- Commands that typically take >10 seconds (builds, installs, complex searches)
- External API calls with high latency
- Operations where you can do other useful work while waiting

**Available tools:**
- `startBackgroundTask` - Start a task in background
- `getBackgroundTask` - Get status, output, and result (single tool for everything)
- `cancelBackgroundTask` - Cancel a running task
- `listBackgroundTasks` - See all active background tasks
- `listBackgroundCapableTools` - See which tools support background mode

**Workflow:**
1. Run a command (it may auto-background if it takes >10s) or use `startBackgroundTask`
2. Use `getBackgroundTask(task_id)` to check status and get output
3. Repeat step 2 until `has_more` is false (task completed)

**Example - monitoring a build:**

```
1. cli_based_tool(command="mvn clean install")
   → {"auto_backgrounded": true, "task_id": "abc-123", ...}

2. getBackgroundTask(task_id="abc-123")
   → {"status": "running", "stdout": "Downloading dependencies...",
      "stdout_offset": 1024, "has_more": true}

3. getBackgroundTask(task_id="abc-123", stdout_offset=1024)
   → {"status": "running", "stdout": "[INFO] Building module...",
      "stdout_offset": 2048, "has_more": true}

4. getBackgroundTask(task_id="abc-123", stdout_offset=2048)
   → {"status": "completed", "stdout": "BUILD SUCCESS",
      "has_more": false, "returncode": 0, "duration_seconds": 45.2}
```

**Response fields:**
- `status`: "pending", "running", "completed", "failed", "cancelled"
- `stdout`/`stderr`: output since your offset (use for incremental reading)
- `stdout_offset`/`stderr_offset`: pass these to your next call
- `has_more`: true while task is running (more output coming)
- `returncode`: exit code when completed (null while running)
- `error`: error message if failed
- `duration_seconds`: execution time when completed

**Auto-backgrounding:**
Some tools automatically move to background if they exceed a time threshold.
When this happens, you'll receive `auto_backgrounded: true` and a `task_id`.
"""

    def get_auto_approved_tools(self) -> List[str]:
        """Return list of tools that should be auto-approved."""
        # Status checks, output reading, and listing are safe - no side effects
        return [
            "getBackgroundTask",
            "listBackgroundTasks",
            "listBackgroundCapableTools",
        ]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands."""
        return [
            UserCommand(
                name="tasks",
                description="List all active background tasks",
                share_with_model=True
            ),
        ]


def create_plugin() -> BackgroundPlugin:
    """Factory function for plugin discovery."""
    return BackgroundPlugin()


__all__ = ['BackgroundPlugin', 'create_plugin']
