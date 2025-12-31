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


class TestOutputStreaming:
    """Tests for output streaming functionality."""

    def test_append_output_stdout(self):
        """Test appending to stdout buffer."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )

        # Append some output
        plugin.append_output(handle.task_id, stdout=b"Hello, ")
        plugin.append_output(handle.task_id, stdout=b"World!\n")

        # Read output
        output = plugin.get_output(handle.task_id)

        assert output.stdout == "Hello, World!\n"
        assert output.stdout_offset == 14
        assert output.has_more is True  # Task still running

        # Cancel task
        plugin.cancel(handle.task_id)

    def test_append_output_stderr(self):
        """Test appending to stderr buffer."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )

        # Append to stderr
        stderr_msg = b"Error: something went wrong\n"
        plugin.append_output(handle.task_id, stderr=stderr_msg)

        # Read output
        output = plugin.get_output(handle.task_id, stream="stderr")

        assert output.stderr == "Error: something went wrong\n"
        assert output.stderr_offset == len(stderr_msg)
        assert output.stdout == ""

        plugin.cancel(handle.task_id)

    def test_append_output_both_streams(self):
        """Test appending to both stdout and stderr."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )

        plugin.append_output(handle.task_id, stdout=b"stdout line\n", stderr=b"stderr line\n")

        output = plugin.get_output(handle.task_id)

        assert output.stdout == "stdout line\n"
        assert output.stderr == "stderr line\n"
        assert output.stdout_offset == 12
        assert output.stderr_offset == 12

        plugin.cancel(handle.task_id)

    def test_get_output_with_offset(self):
        """Test reading output from specific offset."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )

        # Append output in chunks
        plugin.append_output(handle.task_id, stdout=b"Line 1\n")
        plugin.append_output(handle.task_id, stdout=b"Line 2\n")

        # Read from start
        output1 = plugin.get_output(handle.task_id, stdout_offset=0)
        assert output1.stdout == "Line 1\nLine 2\n"
        assert output1.stdout_offset == 14

        # Read from offset (should get nothing new)
        output2 = plugin.get_output(handle.task_id, stdout_offset=14)
        assert output2.stdout == ""
        assert output2.stdout_offset == 14

        # Append more
        plugin.append_output(handle.task_id, stdout=b"Line 3\n")

        # Read from offset (should get new data)
        output3 = plugin.get_output(handle.task_id, stdout_offset=14)
        assert output3.stdout == "Line 3\n"
        assert output3.stdout_offset == 21

        plugin.cancel(handle.task_id)

    def test_get_output_stream_filter(self):
        """Test filtering by stream type."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )

        plugin.append_output(handle.task_id, stdout=b"stdout\n", stderr=b"stderr\n")

        # Read stdout only
        output_stdout = plugin.get_output(handle.task_id, stream="stdout")
        assert output_stdout.stdout == "stdout\n"
        assert output_stdout.stderr == ""

        # Read stderr only
        output_stderr = plugin.get_output(handle.task_id, stream="stderr")
        assert output_stderr.stdout == ""
        assert output_stderr.stderr == "stderr\n"

        # Read both
        output_both = plugin.get_output(handle.task_id, stream="both")
        assert output_both.stdout == "stdout\n"
        assert output_both.stderr == "stderr\n"

        plugin.cancel(handle.task_id)

    def test_get_output_after_completion(self):
        """Test reading output after task completes."""
        plugin = MockBackgroundPlugin()

        # Use slow_tool with minimal duration to avoid race condition
        # between start_background setting started_at/status and
        # _execute_background_task setting completed_at/status
        handle = plugin.start_background(
            "slow_tool",
            {"duration": 0.1},
            executor_fn=plugin._execute_slow
        )

        # Append output before completion
        plugin.append_output(handle.task_id, stdout=b"Task output\n")

        # Wait for completion with explicit wait
        result = plugin.get_result(handle.task_id, wait=True)
        assert result.status == TaskStatus.COMPLETED

        # Read output after completion
        output = plugin.get_output(handle.task_id)
        assert output.stdout == "Task output\n"
        assert output.status == TaskStatus.COMPLETED
        assert output.has_more is False

    def test_set_returncode(self):
        """Test setting return code."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )

        # Set return code
        plugin.set_returncode(handle.task_id, 42)

        # Check in output
        output = plugin.get_output(handle.task_id)
        assert output.returncode == 42

        plugin.cancel(handle.task_id)

    def test_output_buffer_truncation(self):
        """Test that output buffer is truncated when exceeding max size."""
        # Create plugin with small buffer
        plugin = MockBackgroundPlugin()
        plugin._bg_max_output_buffer = 100  # 100 bytes max

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )

        # Write more than buffer size
        plugin.append_output(handle.task_id, stdout=b"A" * 50)
        plugin.append_output(handle.task_id, stdout=b"B" * 60)

        # Buffer should be truncated to last 100 bytes
        output = plugin.get_output(handle.task_id)
        assert len(output.stdout) <= 100
        # Should have more B's than A's due to truncation from start
        assert output.stdout.count("B") > output.stdout.count("A")

        plugin.cancel(handle.task_id)

    def test_get_output_not_found(self):
        """Test get_output raises KeyError for unknown task."""
        plugin = MockBackgroundPlugin()

        with pytest.raises(KeyError, match="not found"):
            plugin.get_output("nonexistent-task-id")

    def test_concurrent_output_writes(self):
        """Test concurrent writes to output buffer."""
        plugin = MockBackgroundPlugin()

        handle = plugin.start_background(
            "slow_tool",
            {"duration": 5.0},
            executor_fn=plugin._execute_slow
        )

        def write_output(prefix: str, count: int):
            for i in range(count):
                plugin.append_output(
                    handle.task_id,
                    stdout=f"{prefix}{i}\n".encode()
                )

        # Start concurrent writers
        threads = []
        for prefix in ["A", "B", "C"]:
            t = threading.Thread(target=write_output, args=(prefix, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All writes should have been recorded
        output = plugin.get_output(handle.task_id)
        # Should have 30 lines total (10 from each prefix)
        lines = [l for l in output.stdout.split('\n') if l]
        assert len(lines) == 30

        plugin.cancel(handle.task_id)
