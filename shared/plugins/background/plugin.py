"""Background task orchestrator plugin.

This plugin provides tools for the model to manage background task execution
across all BackgroundCapable plugins in the registry.
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..model_provider.types import ToolSchema

from ..base import ToolPlugin, UserCommand
from .protocol import BackgroundCapable, TaskHandle, TaskResult, TaskStatus

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
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the background plugin."""
        # Clean up all tasks across all capable plugins
        for plugin in self._capable_plugins.values():
            try:
                plugin.cleanup_completed(max_age_seconds=0)
            except Exception:
                pass
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
                }
            ),
            ToolSchema(
                name="getBackgroundTaskStatus",
                description="Check the current status of a background task.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID returned from startBackgroundTask"
                        },
                    },
                    "required": ["task_id"]
                }
            ),
            ToolSchema(
                name="getBackgroundTaskResult",
                description="""Get the result of a background task.

If the task is still running, returns current status without blocking.
Use this after checking status shows COMPLETED or FAILED.""",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID returned from startBackgroundTask"
                        },
                        "wait": {
                            "type": "boolean",
                            "description": "If true, block until task completes (default: false)"
                        },
                    },
                    "required": ["task_id"]
                }
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
                }
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
                }
            ),
            ToolSchema(
                name="listBackgroundCapableTools",
                description="""List all tools that support background execution.

Use this to discover which tools can be run in background mode.""",
                parameters={
                    "type": "object",
                    "properties": {},
                }
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mapping for background task tools."""
        return {
            "startBackgroundTask": self._start_task,
            "getBackgroundTaskStatus": self._get_status,
            "getBackgroundTaskResult": self._get_result,
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
            return {
                "success": False,
                "error": str(e)
            }

    def _get_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of a background task.

        Args:
            args: Dict containing task_id.

        Returns:
            Dict with task status or error.
        """
        task_id = args.get("task_id")
        if not task_id:
            return {"error": "task_id is required"}

        # Search all capable plugins for the task
        for plugin_name, plugin in self._capable_plugins.items():
            try:
                status = plugin.get_status(task_id)
                return {
                    "task_id": task_id,
                    "plugin_name": plugin_name,
                    "status": status.value,
                }
            except KeyError:
                continue

        return {"error": f"Task '{task_id}' not found"}

    def _get_result(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get result of a background task.

        Args:
            args: Dict containing task_id and optional wait flag.

        Returns:
            Dict with task result or error.
        """
        task_id = args.get("task_id")
        wait = args.get("wait", False)

        if not task_id:
            return {"error": "task_id is required"}

        for plugin_name, plugin in self._capable_plugins.items():
            try:
                result = plugin.get_result(task_id, wait=wait)
                return {
                    "task_id": task_id,
                    "plugin_name": plugin_name,
                    "status": result.status.value,
                    "result": result.result,
                    "error": result.error,
                    "duration_seconds": result.duration_seconds,
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
            except Exception:
                pass

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
                    except Exception:
                        pass

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

**Workflow:**
1. Call `startBackgroundTask` with the tool_name and arguments
2. Continue with other work or inform the user the task is running
3. Periodically check `getBackgroundTaskStatus`
4. Once complete, use `getBackgroundTaskResult` to get the output

**Available tools:**
- `listBackgroundCapableTools` - See which tools support background mode
- `startBackgroundTask` - Start a task in background
- `getBackgroundTaskStatus` - Check if a task is done
- `getBackgroundTaskResult` - Get the result
- `cancelBackgroundTask` - Cancel if no longer needed
- `listBackgroundTasks` - See all active background tasks

**Auto-backgrounding:**
Some tools may automatically move to background if they exceed a time threshold.
When this happens, you'll receive a response with `auto_backgrounded: true` and
a `task_id`. Treat this the same as if you had explicitly started a background task.
"""

    def get_auto_approved_tools(self) -> List[str]:
        """Return list of tools that should be auto-approved."""
        # Status checks and listing are safe - no side effects
        return [
            "getBackgroundTaskStatus",
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
