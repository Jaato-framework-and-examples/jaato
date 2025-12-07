"""Mixin class providing default BackgroundCapable implementation.

Plugins can inherit from BackgroundCapableMixin to easily add background
task support without implementing all protocol methods from scratch.
"""

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .protocol import BackgroundCapable, TaskHandle, TaskResult, TaskStatus


class BackgroundCapableMixin:
    """Mixin providing default implementation for BackgroundCapable protocol.

    Plugins can inherit from this mixin to gain background task support.
    The mixin manages a thread pool and task state internally.

    Usage:
        class MyPlugin(BackgroundCapableMixin):
            def __init__(self):
                super().__init__()
                # ... plugin init ...

            def supports_background(self, tool_name: str) -> bool:
                return tool_name in ['slow_tool', 'another_slow_tool']

            def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
                if tool_name == 'slow_tool':
                    return 10.0  # Auto-background after 10 seconds
                return None

    Subclasses must:
    - Call super().__init__() in their __init__
    - Override supports_background() to declare which tools support backgrounding
    - Override get_auto_background_threshold() to set auto-background thresholds
    - Optionally override estimate_duration() for duration hints
    """

    def __init__(
        self,
        max_workers: int = 4,
        default_timeout: Optional[float] = 300.0
    ):
        """Initialize the background mixin.

        Args:
            max_workers: Maximum concurrent background tasks.
            default_timeout: Default timeout for background tasks in seconds.
        """
        self._bg_executor: Optional[ThreadPoolExecutor] = None
        self._bg_max_workers = max_workers
        self._bg_default_timeout = default_timeout
        self._bg_tasks: Dict[str, Dict[str, Any]] = {}
        self._bg_lock = threading.Lock()
        self._bg_initialized = False

    def _ensure_bg_executor(self) -> ThreadPoolExecutor:
        """Lazily initialize the thread pool executor."""
        if self._bg_executor is None:
            self._bg_executor = ThreadPoolExecutor(max_workers=self._bg_max_workers)
            self._bg_initialized = True
        return self._bg_executor

    def _shutdown_bg_executor(self) -> None:
        """Shutdown the thread pool executor."""
        if self._bg_executor is not None:
            self._bg_executor.shutdown(wait=False)
            self._bg_executor = None
            self._bg_initialized = False

    # --- BackgroundCapable protocol implementation ---

    def supports_background(self, tool_name: str) -> bool:
        """Check if a tool supports background execution.

        Override in subclass to declare which tools support backgrounding.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool can be executed in background.
        """
        return False  # Default: no tools support background

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        """Return timeout threshold for automatic backgrounding.

        Override in subclass to set per-tool thresholds.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            Threshold in seconds, or None to disable auto-background.
        """
        return None  # Default: no auto-backgrounding

    def estimate_duration(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate execution duration in seconds.

        Override in subclass to provide duration estimates.

        Args:
            tool_name: Name of the tool.
            arguments: Arguments that would be passed.

        Returns:
            Estimated duration in seconds, or None if unknown.
        """
        return None  # Default: unknown

    def start_background(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
        executor_fn: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> TaskHandle:
        """Start a tool execution in the background.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            timeout_seconds: Optional timeout for the task.
            executor_fn: The function to execute. If not provided,
                        subclass must override _get_executor_for_tool().

        Returns:
            TaskHandle with the task_id for later status checks.

        Raises:
            ValueError: If tool doesn't support background execution.
            RuntimeError: If task couldn't be started.
        """
        if not self.supports_background(tool_name):
            raise ValueError(f"Tool '{tool_name}' does not support background execution")

        # Get executor function
        if executor_fn is None:
            executor_fn = self._get_executor_for_tool(tool_name)
            if executor_fn is None:
                raise RuntimeError(f"No executor found for tool '{tool_name}'")

        task_id = str(uuid.uuid4())
        now = datetime.now()

        handle = TaskHandle(
            task_id=task_id,
            plugin_name=getattr(self, 'name', 'unknown'),
            tool_name=tool_name,
            created_at=now,
            estimated_duration_seconds=self.estimate_duration(tool_name, arguments),
            metadata={"arguments": arguments}
        )

        timeout = timeout_seconds or self._bg_default_timeout

        with self._bg_lock:
            self._bg_tasks[task_id] = {
                "handle": handle,
                "status": TaskStatus.PENDING,
                "result": None,
                "error": None,
                "future": None,
                "started_at": None,
                "completed_at": None,
                "timeout": timeout,
            }

        # Submit to thread pool
        pool = self._ensure_bg_executor()
        future = pool.submit(
            self._execute_background_task,
            task_id, tool_name, arguments, executor_fn, timeout
        )

        with self._bg_lock:
            self._bg_tasks[task_id]["future"] = future
            self._bg_tasks[task_id]["status"] = TaskStatus.RUNNING
            self._bg_tasks[task_id]["started_at"] = datetime.now()

        return handle

    def _get_executor_for_tool(self, tool_name: str) -> Optional[Callable[[Dict[str, Any]], Any]]:
        """Get the executor function for a tool.

        Override in subclass to provide tool executors, or pass executor_fn
        directly to start_background().

        Args:
            tool_name: Name of the tool.

        Returns:
            Callable that executes the tool, or None if not found.
        """
        # Try to get from get_executors() if available
        if hasattr(self, 'get_executors'):
            executors = self.get_executors()
            return executors.get(tool_name)
        return None

    def _execute_background_task(
        self,
        task_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        executor_fn: Callable[[Dict[str, Any]], Any],
        timeout: Optional[float]
    ) -> None:
        """Internal method that runs in the thread pool.

        Args:
            task_id: The task identifier.
            tool_name: Name of the tool being executed.
            arguments: Arguments for the tool.
            executor_fn: The function to execute.
            timeout: Timeout in seconds (for logging, actual timeout handled by Future).
        """
        try:
            result = executor_fn(arguments)

            with self._bg_lock:
                if task_id in self._bg_tasks:
                    self._bg_tasks[task_id]["status"] = TaskStatus.COMPLETED
                    self._bg_tasks[task_id]["result"] = result
                    self._bg_tasks[task_id]["completed_at"] = datetime.now()

        except Exception as e:
            with self._bg_lock:
                if task_id in self._bg_tasks:
                    self._bg_tasks[task_id]["status"] = TaskStatus.FAILED
                    self._bg_tasks[task_id]["error"] = str(e)
                    self._bg_tasks[task_id]["completed_at"] = datetime.now()

    def get_status(self, task_id: str) -> TaskStatus:
        """Get the current status of a background task.

        Args:
            task_id: ID from the TaskHandle.

        Returns:
            Current status of the task.

        Raises:
            KeyError: If task_id is not found.
        """
        with self._bg_lock:
            if task_id not in self._bg_tasks:
                raise KeyError(f"Task '{task_id}' not found")
            return self._bg_tasks[task_id]["status"]

    def get_result(self, task_id: str, wait: bool = False) -> TaskResult:
        """Get the result of a background task.

        Args:
            task_id: ID from the TaskHandle.
            wait: If True, block until task completes.

        Returns:
            TaskResult with status and result/error.

        Raises:
            KeyError: If task_id is not found.
        """
        with self._bg_lock:
            if task_id not in self._bg_tasks:
                raise KeyError(f"Task '{task_id}' not found")
            task = self._bg_tasks[task_id]
            future = task.get("future")

        # If wait requested and task is still running, wait for completion
        if wait and future is not None:
            try:
                future.result()  # Block until done
            except Exception:
                pass  # Error is already captured in task state

        with self._bg_lock:
            task = self._bg_tasks[task_id]
            started_at = task.get("started_at")
            completed_at = task.get("completed_at")

            duration = None
            if started_at and completed_at:
                duration = (completed_at - started_at).total_seconds()

            return TaskResult(
                task_id=task_id,
                status=task["status"],
                result=task.get("result"),
                error=task.get("error"),
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
            )

    def cancel(self, task_id: str) -> bool:
        """Cancel a running background task.

        Args:
            task_id: ID from the TaskHandle.

        Returns:
            True if cancellation was successful or task was already done.
        """
        with self._bg_lock:
            if task_id not in self._bg_tasks:
                return False

            task = self._bg_tasks[task_id]
            status = task["status"]

            # Already done
            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                          TaskStatus.CANCELLED, TaskStatus.TIMEOUT):
                return True

            # Try to cancel the future
            future = task.get("future")
            if future is not None:
                cancelled = future.cancel()
                if cancelled:
                    task["status"] = TaskStatus.CANCELLED
                    task["completed_at"] = datetime.now()
                    return True

            # If can't cancel (already running), mark as cancelled anyway
            # The result will be discarded
            task["status"] = TaskStatus.CANCELLED
            task["completed_at"] = datetime.now()
            return True

    def list_tasks(self) -> List[TaskHandle]:
        """List all active (pending/running) tasks for this plugin.

        Returns:
            List of TaskHandles for active tasks.
        """
        with self._bg_lock:
            active = []
            for task_id, task in self._bg_tasks.items():
                if task["status"] in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    active.append(task["handle"])
            return active

    def cleanup_completed(self, max_age_seconds: float = 3600) -> int:
        """Clean up completed task records older than max_age.

        Args:
            max_age_seconds: Remove completed tasks older than this.

        Returns:
            Number of tasks cleaned up.
        """
        now = datetime.now()
        to_remove = []

        with self._bg_lock:
            for task_id, task in self._bg_tasks.items():
                if task["status"] in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                                      TaskStatus.CANCELLED, TaskStatus.TIMEOUT):
                    completed_at = task.get("completed_at")
                    if completed_at:
                        age = (now - completed_at).total_seconds()
                        if age > max_age_seconds:
                            to_remove.append(task_id)

            for task_id in to_remove:
                del self._bg_tasks[task_id]

        return len(to_remove)

    def register_running_task(
        self,
        future: Future,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> TaskHandle:
        """Register an already-running Future as a background task.

        Called by ToolExecutor when auto-backgrounding a task that
        exceeded its threshold.

        Args:
            future: The concurrent.futures.Future already executing.
            tool_name: Name of the tool.
            arguments: Arguments passed to the tool.

        Returns:
            TaskHandle for tracking the task.
        """
        task_id = str(uuid.uuid4())
        now = datetime.now()

        handle = TaskHandle(
            task_id=task_id,
            plugin_name=getattr(self, 'name', 'unknown'),
            tool_name=tool_name,
            created_at=now,
            estimated_duration_seconds=self.estimate_duration(tool_name, arguments),
            metadata={"arguments": arguments, "auto_backgrounded": True}
        )

        with self._bg_lock:
            self._bg_tasks[task_id] = {
                "handle": handle,
                "status": TaskStatus.RUNNING,
                "result": None,
                "error": None,
                "future": future,
                "started_at": now,  # Approximate, task was already running
                "completed_at": None,
                "timeout": None,
            }

        # Add callback to update status when future completes
        def on_complete(f: Future) -> None:
            with self._bg_lock:
                if task_id not in self._bg_tasks:
                    return
                task = self._bg_tasks[task_id]
                task["completed_at"] = datetime.now()
                try:
                    result = f.result()
                    task["status"] = TaskStatus.COMPLETED
                    task["result"] = result
                except Exception as e:
                    task["status"] = TaskStatus.FAILED
                    task["error"] = str(e)

        future.add_done_callback(on_complete)

        return handle


__all__ = ['BackgroundCapableMixin']
