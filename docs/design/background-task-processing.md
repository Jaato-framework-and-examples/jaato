# Background Task Processing Design

> **Status**: Draft
> **Author**: Design discussion
> **Date**: 2025-12-07

## Overview

This document describes a design for allowing the model to delegate long-running tool executions to background processing. The design introduces:

1. A **capability protocol** (`BackgroundCapable`) that plugins can implement to declare background execution support
2. A **background orchestrator plugin** that manages lifecycle and exposes tools to the model
3. **Two triggering modes**:
   - **Explicit**: Model proactively starts tasks in background
   - **Auto-background**: Tasks exceeding a plugin-defined threshold are automatically backgrounded

## Motivation

Some tool executions can take significant time:
- MCP calls to external services with high latency
- CLI commands running complex builds or tests
- Web searches/fetches with multiple retries

Currently, all tool executions block the conversation. The model must wait for completion before responding. This limits:
- **User experience**: Long waits with no intermediate feedback
- **Parallelism**: Model can't work on other tasks while one is pending
- **Resource efficiency**: Context is held while waiting for slow operations

## Design Goals

1. **Opt-in per plugin**: Not all plugins need or want background support
2. **Clean separation**: Background orchestration logic is centralized, not scattered
3. **Model agency**: Model decides when to use background based on context
4. **Minimal coupling**: Existing plugin implementations don't need changes unless they want to support backgrounding

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         JaatoClient                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    PluginRegistry                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │  │
│  │  │  CLIPlugin  │  │  MCPPlugin  │  │ BackgroundPlugin  │  │  │
│  │  │  (capable)  │  │  (capable)  │  │  (orchestrator)   │  │  │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────┬─────────┘  │  │
│  │         │                │                   │             │  │
│  │         └────────────────┼───────────────────┘             │  │
│  │                          │                                 │  │
│  │              ┌───────────▼───────────┐                     │  │
│  │              │   BackgroundCapable   │                     │  │
│  │              │       Protocol        │                     │  │
│  │              └───────────────────────┘                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: BackgroundCapable Protocol

A mixin protocol that plugins can implement to declare background execution support.

### Protocol Definition

```python
# shared/plugins/background/protocol.py

from typing import Protocol, Dict, Any, Optional, List, runtime_checkable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class TaskStatus(Enum):
    """Status of a background task."""
    PENDING = "pending"       # Queued but not started
    RUNNING = "running"       # Currently executing
    COMPLETED = "completed"   # Finished successfully
    FAILED = "failed"         # Finished with error
    CANCELLED = "cancelled"   # Cancelled before completion
    TIMEOUT = "timeout"       # Exceeded time limit


@dataclass
class TaskHandle:
    """Handle returned when a task is started in background."""
    task_id: str
    plugin_name: str
    tool_name: str
    created_at: datetime
    estimated_duration_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TaskResult:
    """Result of a completed background task."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None      # The actual result if completed
    error: Optional[str] = None       # Error message if failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


@runtime_checkable
class BackgroundCapable(Protocol):
    """Protocol for plugins that support background task execution.

    Plugins implementing this protocol can have their tool executions
    delegated to background processing by the BackgroundPlugin orchestrator.

    Implementation notes:
    - start_background() should return immediately after spawning the task
    - The plugin is responsible for thread/process management internally
    - Results must be retrievable via get_result() after completion
    - Plugins should implement proper cleanup in cancel()
    """

    def supports_background(self, tool_name: str) -> bool:
        """Check if a specific tool supports background execution.

        Not all tools in a plugin may support backgrounding. For example,
        a quick lookup tool might not benefit from background execution.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool can be executed in background.
        """
        ...

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        """Return timeout threshold for automatic backgrounding.

        When a tool execution exceeds this threshold, the ToolExecutor
        automatically converts it to a background task and returns a handle.
        This allows reactive backgrounding without model intervention.

        The plugin controls its own thresholds because:
        - Different tools have different expected durations
        - Plugin authors know their tools' performance characteristics
        - Some tools should never auto-background (return None)

        Args:
            tool_name: Name of the tool to check.

        Returns:
            Threshold in seconds after which to auto-background,
            or None to disable auto-background for this tool.
        """
        ...

    def estimate_duration(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate execution duration in seconds.

        This helps the model and orchestrator make informed decisions
        about whether to background a task. Return None if unknown.

        Args:
            tool_name: Name of the tool.
            arguments: Arguments that would be passed to the tool.

        Returns:
            Estimated duration in seconds, or None if unknown.
        """
        ...

    def start_background(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_seconds: Optional[float] = None
    ) -> TaskHandle:
        """Start a tool execution in the background.

        This method should return immediately after spawning the task.
        The actual execution happens asynchronously.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            timeout_seconds: Optional timeout for the task.

        Returns:
            TaskHandle with the task_id for later status checks.

        Raises:
            ValueError: If tool doesn't support background execution.
            RuntimeError: If task couldn't be started.
        """
        ...

    def get_status(self, task_id: str) -> TaskStatus:
        """Get the current status of a background task.

        Args:
            task_id: ID from the TaskHandle.

        Returns:
            Current status of the task.

        Raises:
            KeyError: If task_id is not found.
        """
        ...

    def get_result(self, task_id: str, wait: bool = False) -> TaskResult:
        """Get the result of a background task.

        Args:
            task_id: ID from the TaskHandle.
            wait: If True, block until task completes. If False, return
                  immediately with current state.

        Returns:
            TaskResult with status and result/error.

        Raises:
            KeyError: If task_id is not found.
        """
        ...

    def cancel(self, task_id: str) -> bool:
        """Cancel a running background task.

        Args:
            task_id: ID from the TaskHandle.

        Returns:
            True if cancellation was successful or task was already done.
            False if cancellation failed.
        """
        ...

    def list_tasks(self) -> List[TaskHandle]:
        """List all active (pending/running) tasks for this plugin.

        Returns:
            List of TaskHandles for active tasks.
        """
        ...

    def cleanup_completed(self, max_age_seconds: float = 3600) -> int:
        """Clean up completed task records older than max_age.

        Args:
            max_age_seconds: Remove completed tasks older than this.

        Returns:
            Number of tasks cleaned up.
        """
        ...

    def register_running_task(
        self,
        future: 'Future',
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> TaskHandle:
        """Register an already-running Future as a background task.

        Called by ToolExecutor when auto-backgrounding a task that
        exceeded its threshold. The Future is already executing.

        This enables the auto-background flow where the executor starts
        a task, waits up to a threshold, then registers it as background
        if it's still running.

        Args:
            future: The concurrent.futures.Future already executing.
            tool_name: Name of the tool.
            arguments: Arguments passed to the tool.

        Returns:
            TaskHandle for tracking the task.
        """
        ...
```

### Implementation Pattern

Plugins implement this by wrapping their existing executor logic:

```python
# Example: CLI plugin with background support

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

class CLIPluginWithBackground:
    """CLI plugin with BackgroundCapable support."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._tasks: Dict[str, dict] = {}  # task_id -> task state
        self._lock = threading.Lock()

    def supports_background(self, tool_name: str) -> bool:
        # CLI commands can generally run in background
        return tool_name == "runCommand"

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        # Auto-background CLI commands after 10 seconds
        if tool_name == "runCommand":
            return 10.0
        return None  # Other tools stay synchronous

    def estimate_duration(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[float]:
        # Could analyze command to estimate (e.g., "npm install" → ~30s)
        command = arguments.get("command", "")
        if "npm install" in command or "pip install" in command:
            return 30.0
        if "make" in command or "build" in command:
            return 60.0
        return None  # Unknown

    def start_background(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_seconds: Optional[float] = None
    ) -> TaskHandle:
        task_id = str(uuid.uuid4())
        now = datetime.now()

        handle = TaskHandle(
            task_id=task_id,
            plugin_name=self.name,
            tool_name=tool_name,
            created_at=now,
            estimated_duration_seconds=self.estimate_duration(tool_name, arguments)
        )

        with self._lock:
            self._tasks[task_id] = {
                "handle": handle,
                "status": TaskStatus.PENDING,
                "result": None,
                "error": None,
                "future": None,
                "started_at": None,
                "completed_at": None,
            }

        # Submit to thread pool
        future = self._executor.submit(
            self._execute_in_background,
            task_id, tool_name, arguments, timeout_seconds
        )

        with self._lock:
            self._tasks[task_id]["future"] = future
            self._tasks[task_id]["status"] = TaskStatus.RUNNING
            self._tasks[task_id]["started_at"] = datetime.now()

        return handle

    def _execute_in_background(
        self,
        task_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float]
    ):
        """Internal method that runs in the thread pool."""
        try:
            # Call the actual executor
            result = self._run_command(arguments)  # existing method

            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.COMPLETED
                self._tasks[task_id]["result"] = result
                self._tasks[task_id]["completed_at"] = datetime.now()

        except Exception as e:
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.FAILED
                self._tasks[task_id]["error"] = str(e)
                self._tasks[task_id]["completed_at"] = datetime.now()
```

---

## Component 2: Background Orchestrator Plugin

A separate plugin that:
1. Discovers background-capable plugins via the registry
2. Exposes model-facing tools for background task management
3. Manages task lifecycle across all capable plugins

### Plugin Definition

```python
# shared/plugins/background/plugin.py

from typing import Dict, Any, List, Optional
from google.genai import types

from ..base import ToolPlugin
from ..registry import PluginRegistry
from .protocol import BackgroundCapable, TaskHandle, TaskResult, TaskStatus

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

    @property
    def name(self) -> str:
        return "background"

    def __init__(self):
        self._registry: Optional[PluginRegistry] = None
        self._capable_plugins: Dict[str, BackgroundCapable] = {}

    def set_registry(self, registry: PluginRegistry) -> None:
        """Set the plugin registry for capability discovery.

        Called by JaatoClient after registry is configured.
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
            if plugin and isinstance(plugin, BackgroundCapable):
                self._capable_plugins[plugin_name] = plugin

    def get_function_declarations(self) -> List[types.FunctionDeclaration]:
        return [
            types.FunctionDeclaration(
                name="startBackgroundTask",
                description="""Start a tool execution in the background.

Use this when you anticipate a tool call will take significant time
(e.g., long builds, complex searches, external API calls). The task
runs asynchronously and you can continue with other work.

Returns a task_id you can use to check status or get results later.""",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "plugin_name": types.Schema(
                            type=types.Type.STRING,
                            description="Name of the plugin providing the tool"
                        ),
                        "tool_name": types.Schema(
                            type=types.Type.STRING,
                            description="Name of the tool to execute"
                        ),
                        "arguments": types.Schema(
                            type=types.Type.OBJECT,
                            description="Arguments to pass to the tool"
                        ),
                        "timeout_seconds": types.Schema(
                            type=types.Type.NUMBER,
                            description="Optional timeout in seconds"
                        ),
                    },
                    required=["plugin_name", "tool_name", "arguments"]
                )
            ),
            types.FunctionDeclaration(
                name="getBackgroundTaskStatus",
                description="Check the current status of a background task.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "task_id": types.Schema(
                            type=types.Type.STRING,
                            description="Task ID returned from startBackgroundTask"
                        ),
                    },
                    required=["task_id"]
                )
            ),
            types.FunctionDeclaration(
                name="getBackgroundTaskResult",
                description="""Get the result of a background task.

If the task is still running, returns current status without blocking.
Use this after checking status shows COMPLETED or FAILED.""",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "task_id": types.Schema(
                            type=types.Type.STRING,
                            description="Task ID returned from startBackgroundTask"
                        ),
                        "wait": types.Schema(
                            type=types.Type.BOOLEAN,
                            description="If true, block until task completes (default: false)"
                        ),
                    },
                    required=["task_id"]
                )
            ),
            types.FunctionDeclaration(
                name="cancelBackgroundTask",
                description="Cancel a running background task.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "task_id": types.Schema(
                            type=types.Type.STRING,
                            description="Task ID to cancel"
                        ),
                    },
                    required=["task_id"]
                )
            ),
            types.FunctionDeclaration(
                name="listBackgroundTasks",
                description="List all active background tasks across all plugins.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "plugin_name": types.Schema(
                            type=types.Type.STRING,
                            description="Optional: filter by plugin name"
                        ),
                    },
                )
            ),
            types.FunctionDeclaration(
                name="listBackgroundCapableTools",
                description="""List all tools that support background execution.

Use this to discover which tools can be run in background mode.""",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={},
                )
            ),
        ]

    def get_executors(self) -> Dict[str, Any]:
        return {
            "startBackgroundTask": self._start_task,
            "getBackgroundTaskStatus": self._get_status,
            "getBackgroundTaskResult": self._get_result,
            "cancelBackgroundTask": self._cancel_task,
            "listBackgroundTasks": self._list_tasks,
            "listBackgroundCapableTools": self._list_capable_tools,
        }

    def _start_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Start a background task."""
        plugin_name = args["plugin_name"]
        tool_name = args["tool_name"]
        arguments = args.get("arguments", {})
        timeout = args.get("timeout_seconds")

        # Find the capable plugin
        if plugin_name not in self._capable_plugins:
            return {
                "success": False,
                "error": f"Plugin '{plugin_name}' does not support background execution. "
                         f"Capable plugins: {list(self._capable_plugins.keys())}"
            }

        plugin = self._capable_plugins[plugin_name]

        # Check if specific tool supports background
        if not plugin.supports_background(tool_name):
            return {
                "success": False,
                "error": f"Tool '{tool_name}' in plugin '{plugin_name}' does not support background execution"
            }

        try:
            handle = plugin.start_background(tool_name, arguments, timeout)
            return {
                "success": True,
                "task_id": handle.task_id,
                "plugin_name": handle.plugin_name,
                "tool_name": handle.tool_name,
                "estimated_duration_seconds": handle.estimated_duration_seconds,
                "message": f"Task started in background. Use task_id '{handle.task_id}' to check status."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _get_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of a background task."""
        task_id = args["task_id"]

        # Find which plugin owns this task
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

        return {
            "error": f"Task '{task_id}' not found"
        }

    def _get_result(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get result of a background task."""
        task_id = args["task_id"]
        wait = args.get("wait", False)

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

        return {
            "error": f"Task '{task_id}' not found"
        }

    def _cancel_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a background task."""
        task_id = args["task_id"]

        for plugin_name, plugin in self._capable_plugins.items():
            try:
                plugin.get_status(task_id)  # Check if exists
                success = plugin.cancel(task_id)
                return {
                    "task_id": task_id,
                    "cancelled": success,
                }
            except KeyError:
                continue

        return {
            "error": f"Task '{task_id}' not found"
        }

    def _list_tasks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all active background tasks."""
        filter_plugin = args.get("plugin_name")

        all_tasks = []
        for plugin_name, plugin in self._capable_plugins.items():
            if filter_plugin and plugin_name != filter_plugin:
                continue

            for handle in plugin.list_tasks():
                all_tasks.append({
                    "task_id": handle.task_id,
                    "plugin_name": handle.plugin_name,
                    "tool_name": handle.tool_name,
                    "created_at": handle.created_at.isoformat(),
                    "estimated_duration_seconds": handle.estimated_duration_seconds,
                })

        return {
            "tasks": all_tasks,
            "count": len(all_tasks),
        }

    def _list_capable_tools(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all tools that support background execution."""
        capable_tools = []

        for plugin_name, plugin in self._capable_plugins.items():
            # Get all tool names from the plugin's executors
            base_plugin = self._registry.get_plugin(plugin_name)
            if base_plugin:
                for tool_name in base_plugin.get_executors().keys():
                    if plugin.supports_background(tool_name):
                        capable_tools.append({
                            "plugin_name": plugin_name,
                            "tool_name": tool_name,
                        })

        return {
            "tools": capable_tools,
            "count": len(capable_tools),
        }

    # Standard ToolPlugin methods

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def shutdown(self) -> None:
        # Clean up all tasks across all capable plugins
        for plugin in self._capable_plugins.values():
            plugin.cleanup_completed(max_age_seconds=0)  # Force cleanup

    def get_system_instructions(self) -> Optional[str]:
        return """## Background Task Execution

You have the ability to run long-running tool executions in the background.

**When to use background execution:**
- Commands that typically take >10 seconds (builds, installs, complex searches)
- External API calls with high latency
- Operations where you can do other useful work while waiting

**Workflow:**
1. Call `startBackgroundTask` with the plugin, tool, and arguments
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
"""

    def get_auto_approved_tools(self) -> List[str]:
        # Status checks are safe
        return [
            "getBackgroundTaskStatus",
            "listBackgroundTasks",
            "listBackgroundCapableTools",
        ]

    def get_user_commands(self) -> List:
        return []


def create_plugin():
    """Factory function for plugin discovery."""
    return BackgroundPlugin()
```

---

## Component 3: Integration with JaatoClient

The `JaatoClient` needs to be aware of the background plugin to pass it the registry reference.

### Changes to JaatoClient

```python
# In configure_tools():

def configure_tools(self, registry, permission_plugin=None, ledger=None):
    # ... existing code ...

    # Configure background plugin if exposed
    self._configure_background_plugin(registry)

    # ... rest of existing code ...

def _configure_background_plugin(self, registry: PluginRegistry) -> None:
    """Pass registry to background plugin for capability discovery."""
    try:
        background_plugin = registry.get_plugin('background')
        if background_plugin and hasattr(background_plugin, 'set_registry'):
            background_plugin.set_registry(registry)
    except (KeyError, AttributeError):
        pass
```

---

## Usage Patterns

### Pattern 1: Model Proactively Uses Background

The model recognizes a command will be slow and directly starts it in background:

```
User: "Install all dependencies and then run the tests"

Model thinking: "npm install typically takes 20-30 seconds. I should run it
in background so I can prepare the test command."

Model calls: startBackgroundTask(
    plugin_name="cli",
    tool_name="runCommand",
    arguments={"command": "npm install"}
)

Model response: "I've started the dependency installation in the background
(task: abc-123). While that runs, I'll prepare to run the tests..."

[Later]
Model calls: getBackgroundTaskStatus(task_id="abc-123")
Model calls: getBackgroundTaskResult(task_id="abc-123")
Model calls: runCommand(command="npm test")
```

### Pattern 2: Model Checks Capability First

```
Model calls: listBackgroundCapableTools()

Returns: {
    "tools": [
        {"plugin_name": "cli", "tool_name": "runCommand"},
        {"plugin_name": "mcp", "tool_name": "confluence_search"}
    ]
}

Model: "I see I can run CLI commands and Confluence searches in background..."
```

### Pattern 3: Parallel Background Tasks

```
User: "Run all three test suites"

Model calls (in parallel):
  - startBackgroundTask(cli, runCommand, {command: "npm test"})
  - startBackgroundTask(cli, runCommand, {command: "pytest"})
  - startBackgroundTask(cli, runCommand, {command: "go test ./..."})

Model: "I've started all three test suites in background:
- Node tests: task-1
- Python tests: task-2
- Go tests: task-3

I'll check their status..."

Model calls: listBackgroundTasks()
```

### Pattern 4: Auto-Backgrounded Task (Reactive)

The model calls a tool normally, but it exceeds the plugin's threshold and is
automatically converted to a background task:

```
User: "Run the full test suite"

Model calls: runCommand(command="npm test -- --coverage")

[10 seconds pass - threshold exceeded]

Tool returns: {
    "auto_backgrounded": true,
    "task_id": "xyz-789",
    "threshold_seconds": 10.0,
    "message": "Task exceeded 10.0s threshold, continuing in background..."
}

Model: "The test suite is taking longer than expected and has been moved
to background processing. I'll monitor its progress..."

Model calls: getBackgroundTaskStatus(task_id="xyz-789")

[... polling until complete ...]

Model calls: getBackgroundTaskResult(task_id="xyz-789")

Model: "Tests completed! Here are the results: ..."
```

This pattern requires no proactive decision from the model - the system
automatically handles long-running tasks.

---

## Component 4: Auto-Background via ToolExecutor

The auto-background mechanism is implemented at the `ToolExecutor` level, allowing tasks
that exceed a plugin-defined threshold to be automatically converted to background tasks.

### Key Insight

We're not "pausing and resuming" a task - we're simply **deciding to stop waiting** for it.
The task continues running in its thread; we just return a handle instead of blocking.

### ToolExecutor Integration

```python
# In shared/ai_tool_runner.py

class ToolExecutor:
    """Executor with auto-background support."""

    def execute(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool, potentially auto-backgrounding if threshold exceeded."""

        # Get the plugin that owns this tool
        plugin = self._get_plugin_for_tool(name)

        # Check if plugin supports background and has auto threshold
        if isinstance(plugin, BackgroundCapable):
            threshold = plugin.get_auto_background_threshold(name)
            if threshold is not None:
                return self._execute_with_auto_background(
                    plugin, name, args, threshold
                )

        # Standard synchronous execution
        return self._execute_sync(name, args)

    def _execute_with_auto_background(
        self,
        plugin: BackgroundCapable,
        name: str,
        args: Dict[str, Any],
        threshold: float
    ) -> Any:
        """Execute with timeout-based auto-backgrounding."""

        # Start execution in thread pool
        future = self._thread_pool.submit(self._execute_sync, name, args)

        try:
            # Wait up to threshold seconds for completion
            return future.result(timeout=threshold)

        except TimeoutError:
            # Task still running - convert to background
            handle = plugin.register_running_task(future, name, args)

            return {
                "auto_backgrounded": True,
                "task_id": handle.task_id,
                "threshold_seconds": threshold,
                "message": f"Task exceeded {threshold}s threshold, continuing in background. "
                           f"Use task_id '{handle.task_id}' to check status."
            }
```

### Execution Flow

```
ToolExecutor.execute(name, args)
    │
    ├─ Find plugin for tool
    │
    ├─ Is plugin BackgroundCapable?
    │   │
    │   ├─ No → execute synchronously (blocking)
    │   │
    │   └─ Yes → get_auto_background_threshold(name)
    │             │
    │             ├─ None → execute synchronously (blocking)
    │             │
    │             └─ threshold seconds →
    │                  │
    │                  ├─ Start in thread pool
    │                  ├─ Wait up to threshold
    │                  │
    │                  ├─ Completes in time? → return result
    │                  │
    │                  └─ Exceeds threshold? →
    │                       ├─ Register as background task
    │                       └─ Return handle immediately
```

### Plugin Method for Registering Running Tasks

```python
# Addition to BackgroundCapable protocol

def register_running_task(
    self,
    future: Future,
    tool_name: str,
    arguments: Dict[str, Any]
) -> TaskHandle:
    """Register an already-running Future as a background task.

    Called by ToolExecutor when auto-backgrounding a task that
    exceeded its threshold. The Future is already running.

    Args:
        future: The concurrent.futures.Future already executing.
        tool_name: Name of the tool.
        arguments: Arguments passed to the tool.

    Returns:
        TaskHandle for tracking the task.
    """
    ...
```

### Model Handling of Auto-Backgrounded Results

The model receives a response indicating the task was auto-backgrounded:

```
Model calls: runCommand(command="npm install && npm run build")

Tool returns: {
    "auto_backgrounded": true,
    "task_id": "abc-123",
    "threshold_seconds": 10.0,
    "message": "Task exceeded 10.0s threshold, continuing in background..."
}

Model: "The build is taking longer than expected and is now running in
background. I'll check on its progress..."

Model calls: getBackgroundTaskStatus(task_id="abc-123")
```

### Why This Works

1. **No state serialization needed** - The task keeps running in the same thread
2. **Plugin controls thresholds** - Each plugin knows its tools' expected durations
3. **Transparent to plugin internals** - Existing executor logic doesn't change
4. **Model adapts naturally** - Receives a different response shape and reacts

---

## Configuration Options

```python
@dataclass
class BackgroundConfig:
    """Configuration for background task processing."""

    # Maximum concurrent background tasks across all plugins
    max_concurrent_tasks: int = 10

    # Default timeout for background tasks (seconds)
    default_timeout_seconds: float = 300.0

    # Auto-cleanup completed tasks after this duration
    completed_task_retention_seconds: float = 3600.0

    # Threshold above which model should consider backgrounding
    background_threshold_seconds: float = 10.0
```

---

## Security Considerations

1. **Resource limits**: Max concurrent tasks prevents runaway spawning
2. **Timeout enforcement**: Prevents indefinitely running tasks
3. **Cleanup**: Completed task cleanup prevents memory leaks
4. **Permission inheritance**: Background tasks should inherit the same permission checks as foreground execution
5. **Isolation**: Each plugin manages its own thread pool/execution

---

## Implementation Plan

### Phase 1: Protocol & Core Infrastructure
1. Define `BackgroundCapable` protocol in `shared/plugins/background/protocol.py`
2. Implement `BackgroundPlugin` orchestrator in `shared/plugins/background/plugin.py`
3. Add integration hooks in `JaatoClient`
4. Extend `ToolExecutor` with auto-background support:
   - Add thread pool for tool execution
   - Implement timeout-based auto-backgrounding
   - Query plugins for thresholds via `get_auto_background_threshold()`

### Phase 2: CLI Plugin Support
1. Add `BackgroundCapable` implementation to CLI plugin
2. Implement `register_running_task()` for auto-background handoff
3. Configure per-command thresholds (builds, installs, tests)
4. Add duration estimation heuristics

### Phase 3: MCP Plugin Support
1. Add `BackgroundCapable` wrapper for MCP calls
2. Handle connection management for async calls
3. Configure thresholds based on MCP server characteristics

### Phase 4: Refinements
1. Add configuration options (max concurrent, global timeout overrides)
2. Add user commands for task visibility (`/tasks`, `/cancel`)
3. Add persistence for task state (survive restarts)
4. System instructions updates for model to understand auto-backgrounded responses

---

## Open Questions

1. **Notification mechanism**: How should the model be notified when a background task completes without polling? Options:
   - Polling (simple, current design)
   - Inject completion notice into next message (complex)
   - User command that model can check

2. **Context for background tasks**: Should background tasks have access to conversation history? Probably not, but some context might be useful.

3. **Subagent interaction**: Should subagents be able to start background tasks? If so, should they be visible to the parent?

4. **Persistence**: Should task state survive client restarts? For long-running tasks, probably yes.
