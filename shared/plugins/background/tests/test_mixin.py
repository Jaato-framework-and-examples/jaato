"""Tests for BackgroundCapableMixin."""

import pytest
import time
import threading
from concurrent.futures import Future
from datetime import datetime
from typing import Any, Dict, Optional

from shared.plugins.background.mixin import BackgroundCapableMixin
from shared.plugins.background.protocol import TaskStatus


class MockBackgroundPlugin(BackgroundCapableMixin):
    """Mock plugin implementing BackgroundCapable via the mixin."""

    def __init__(self, max_workers: int = 2):
        super().__init__(max_workers=max_workers)
        self._call_count = 0
        self._last_args: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return "mock_background"

    def supports_background(self, tool_name: str) -> bool:
        return tool_name in ["slow_tool", "fast_tool", "failing_tool"]

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        if tool_name == "slow_tool":
            return 1.0  # 1 second threshold for testing
        return None

    def estimate_duration(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[float]:
        if tool_name == "slow_tool":
            return arguments.get("duration", 5.0)
        return None

    def get_executors(self) -> Dict[str, Any]:
        return {
            "slow_tool": self._execute_slow,
            "fast_tool": self._execute_fast,
            "failing_tool": self._execute_failing,
        }

    def _execute_slow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a slow operation."""
        duration = args.get("duration", 2.0)
        time.sleep(duration)
        self._call_count += 1
        self._last_args = args
        return {"result": "slow_complete", "duration": duration}

    def _execute_fast(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a fast operation."""
        self._call_count += 1
        self._last_args = args
        return {"result": "fast_complete"}

    def _execute_failing(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a failing operation."""
        self._call_count += 1
        raise ValueError("Intentional failure")


class TestBackgroundCapableMixin:
    """Tests for BackgroundCapableMixin."""

    def test_supports_background(self):
        """Test supports_background returns correct values."""
        plugin = MockBackgroundPlugin()

        assert plugin.supports_background("slow_tool") is True
        assert plugin.supports_background("fast_tool") is True
        assert plugin.supports_background("unknown_tool") is False

    def test_get_auto_background_threshold(self):
        """Test get_auto_background_threshold returns correct values."""
        plugin = MockBackgroundPlugin()

        assert plugin.get_auto_background_threshold("slow_tool") == 1.0
        assert plugin.get_auto_background_threshold("fast_tool") is None
        assert plugin.get_auto_background_threshold("unknown") is None

    def test_estimate_duration(self):
        """Test estimate_duration returns correct values."""
        plugin = MockBackgroundPlugin()

        assert plugin.estimate_duration("slow_tool", {"duration": 10.0}) == 10.0
        assert plugin.estimate_duration("slow_tool", {}) == 5.0  # default
        assert plugin.estimate_duration("fast_tool", {}) is None

    def test_start_background_unsupported_tool(self):
        """Test start_background raises for unsupported tool."""
        plugin = MockBackgroundPlugin()

        with pytest.raises(ValueError, match="does not support background"):
            plugin.start_background("unknown_tool", {})

    def test_start_background_success(self):
        """Test starting a background task."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "fast_tool",
            {"key": "value"},
            executor_fn=plugin._execute_fast
        )

        assert handle.task_id is not None
        assert handle.plugin_name == "mock_background"
        assert handle.tool_name == "fast_tool"
        assert isinstance(handle.created_at, datetime)

        # Wait for completion
        time.sleep(0.1)

        status = plugin.get_status(handle.task_id)
        assert status == TaskStatus.COMPLETED

    def test_start_background_with_slow_task(self):
        """Test starting a slow background task."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 0.5},
            executor_fn=plugin._execute_slow
        )

        # Initially should be running
        status = plugin.get_status(handle.task_id)
        assert status in (TaskStatus.PENDING, TaskStatus.RUNNING)

        # Wait for completion
        time.sleep(0.7)

        status = plugin.get_status(handle.task_id)
        assert status == TaskStatus.COMPLETED

    def test_get_status_not_found(self):
        """Test get_status raises for unknown task."""
        plugin = MockBackgroundPlugin()

        with pytest.raises(KeyError, match="not found"):
            plugin.get_status("nonexistent-task-id")

    def test_get_result_completed(self):
        """Test getting result of completed task."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "fast_tool",
            {"key": "value"},
            executor_fn=plugin._execute_fast
        )

        # Wait for completion
        time.sleep(0.1)

        result = plugin.get_result(handle.task_id)

        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"result": "fast_complete"}
        assert result.error is None
        assert result.duration_seconds is not None

    def test_get_result_failed(self):
        """Test getting result of failed task."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "failing_tool",
            {},
            executor_fn=plugin._execute_failing
        )

        # Wait for failure
        time.sleep(0.1)

        result = plugin.get_result(handle.task_id)

        assert result.status == TaskStatus.FAILED
        assert result.error is not None
        assert "Intentional failure" in result.error

    def test_get_result_with_wait(self):
        """Test getting result with wait=True."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 0.3},
            executor_fn=plugin._execute_slow
        )

        # Should block until complete
        result = plugin.get_result(handle.task_id, wait=True)

        assert result.status == TaskStatus.COMPLETED
        assert result.result["result"] == "slow_complete"

    def test_cancel_pending_task(self):
        """Test cancelling a task."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 10.0},  # Long duration
            executor_fn=plugin._execute_slow
        )

        # Cancel immediately
        success = plugin.cancel(handle.task_id)
        assert success is True

        # Status should be cancelled
        status = plugin.get_status(handle.task_id)
        assert status == TaskStatus.CANCELLED

    def test_cancel_completed_task(self):
        """Test cancelling an already completed task."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "fast_tool",
            {},
            executor_fn=plugin._execute_fast
        )

        # Wait for completion
        time.sleep(0.1)

        # Cancel should return True (already done)
        success = plugin.cancel(handle.task_id)
        assert success is True

    def test_cancel_nonexistent_task(self):
        """Test cancelling a nonexistent task."""
        plugin = MockBackgroundPlugin()

        success = plugin.cancel("nonexistent-task-id")
        assert success is False

    def test_list_tasks(self):
        """Test listing active tasks."""
        plugin = MockBackgroundPlugin()

        # Start some tasks
        handle1 = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )
        handle2 = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )

        # List should include both
        tasks = plugin.list_tasks()
        task_ids = [t.task_id for t in tasks]

        assert handle1.task_id in task_ids
        assert handle2.task_id in task_ids

        # Cancel both
        plugin.cancel(handle1.task_id)
        plugin.cancel(handle2.task_id)

        # List should be empty (or not include cancelled)
        tasks = plugin.list_tasks()
        assert len(tasks) == 0

    def test_cleanup_completed(self):
        """Test cleaning up completed tasks."""
        plugin = MockBackgroundPlugin()

        # Start and complete some tasks
        for i in range(5):
            handle = plugin.start_background(
                "fast_tool",
                {"i": i},
                executor_fn=plugin._execute_fast
            )

        # Wait for completion
        time.sleep(0.2)

        # All should be completed
        count = plugin.cleanup_completed(max_age_seconds=0)
        assert count == 5

        # Internal state should be cleaned
        assert len(plugin._bg_tasks) == 0

    def test_register_running_task(self):
        """Test registering an already-running Future."""
        plugin = MockBackgroundPlugin()

        # Create a future that's already running
        pool = plugin._ensure_bg_executor()
        future = pool.submit(lambda: {"result": "external"})

        handle = plugin.register_running_task(
            future,
            "external_tool",
            {"source": "external"}
        )

        assert handle.task_id is not None
        assert handle.tool_name == "external_tool"
        assert handle.metadata.get("auto_backgrounded") is True

        # Wait for completion
        time.sleep(0.1)

        status = plugin.get_status(handle.task_id)
        assert status == TaskStatus.COMPLETED

        result = plugin.get_result(handle.task_id)
        assert result.result == {"result": "external"}

    def test_register_running_task_failure(self):
        """Test registering a Future that will fail."""
        plugin = MockBackgroundPlugin()

        def failing_fn():
            raise RuntimeError("External failure")

        pool = plugin._ensure_bg_executor()
        future = pool.submit(failing_fn)

        handle = plugin.register_running_task(
            future,
            "failing_external",
            {}
        )

        # Wait for failure
        time.sleep(0.1)

        status = plugin.get_status(handle.task_id)
        assert status == TaskStatus.FAILED

        result = plugin.get_result(handle.task_id)
        assert "External failure" in result.error

    def test_concurrent_tasks(self):
        """Test running multiple concurrent tasks."""
        plugin = MockBackgroundPlugin(max_workers=4)

        handles = []
        for i in range(4):
            handle = plugin.start_background(
                "slow_tool",
                {"duration": 0.2, "i": i},
                executor_fn=plugin._execute_slow
            )
            handles.append(handle)

        # All should be running
        time.sleep(0.05)
        running = sum(
            1 for h in handles
            if plugin.get_status(h.task_id) == TaskStatus.RUNNING
        )
        assert running >= 2  # At least some should be running concurrently

        # Wait for all to complete
        time.sleep(0.5)

        for handle in handles:
            result = plugin.get_result(handle.task_id)
            assert result.status == TaskStatus.COMPLETED

    def test_shutdown_executor(self):
        """Test shutting down the executor."""
        plugin = MockBackgroundPlugin()

        # Start a task to initialize the executor
        plugin.start_background(
            "fast_tool",
            {},
            executor_fn=plugin._execute_fast
        )

        assert plugin._bg_executor is not None

        # Shutdown
        plugin._shutdown_bg_executor()

        assert plugin._bg_executor is None
        assert plugin._bg_initialized is False
