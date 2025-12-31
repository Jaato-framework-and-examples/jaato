"""Protocol definitions for background task processing.

This module defines the BackgroundCapable protocol that plugins can implement
to support background task execution, along with supporting data structures.
"""

from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


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
    """Handle returned when a task is started in background.

    Attributes:
        task_id: Unique identifier for the task.
        plugin_name: Name of the plugin that owns this task.
        tool_name: Name of the tool being executed.
        created_at: Timestamp when the task was created.
        estimated_duration_seconds: Optional estimated duration.
        metadata: Optional additional metadata.
    """
    task_id: str
    plugin_name: str
    tool_name: str
    created_at: datetime
    estimated_duration_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "plugin_name": self.plugin_name,
            "tool_name": self.tool_name,
            "created_at": self.created_at.isoformat(),
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class TaskResult:
    """Result of a completed background task.

    Attributes:
        task_id: Unique identifier for the task.
        status: Final status of the task.
        result: The actual result if completed successfully.
        error: Error message if failed.
        started_at: Timestamp when execution started.
        completed_at: Timestamp when execution completed.
        duration_seconds: Total execution duration.
    """
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class TaskOutput:
    """Incremental output from a background task.

    Used for streaming stdout/stderr during task execution, allowing
    agents to monitor progress and detect errors early.

    Attributes:
        task_id: Unique identifier for the task.
        status: Current status of the task.
        stdout: New stdout content since last read (from offset).
        stderr: New stderr content since last read (from offset).
        stdout_offset: Current offset in stdout buffer after this read.
        stderr_offset: Current offset in stderr buffer after this read.
        has_more: True if task is still running (more output expected).
        returncode: Exit code if task is completed, None otherwise.
    """
    task_id: str
    status: TaskStatus
    stdout: str = ""
    stderr: str = ""
    stdout_offset: int = 0
    stderr_offset: int = 0
    has_more: bool = True
    returncode: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_offset": self.stdout_offset,
            "stderr_offset": self.stderr_offset,
            "has_more": self.has_more,
            "returncode": self.returncode,
        }


@runtime_checkable
class BackgroundCapable(Protocol):
    """Protocol for plugins that support background task execution.

    Plugins implementing this protocol can have their tool executions
    delegated to background processing by the BackgroundPlugin orchestrator.

    There are two ways tasks can be backgrounded:
    1. Explicit: Model calls startBackgroundTask via the orchestrator
    2. Auto: ToolExecutor auto-backgrounds when threshold is exceeded

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

    def get_output(
        self,
        task_id: str,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
        stream: str = "both"
    ) -> 'TaskOutput':
        """Get incremental output from a background task.

        Allows agents to monitor task progress by reading stdout/stderr
        incrementally. Returns new output since the specified offsets.

        Args:
            task_id: ID from the TaskHandle.
            stdout_offset: Byte offset to start reading stdout from.
            stderr_offset: Byte offset to start reading stderr from.
            stream: Which streams to include: "stdout", "stderr", or "both".

        Returns:
            TaskOutput with new content and updated offsets.

        Raises:
            KeyError: If task_id is not found.
        """
        ...

    def register_running_task(
        self,
        future: Future,
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


__all__ = [
    'TaskStatus',
    'TaskHandle',
    'TaskResult',
    'TaskOutput',
    'BackgroundCapable',
]
