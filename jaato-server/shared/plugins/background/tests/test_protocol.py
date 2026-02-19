"""Tests for BackgroundCapable protocol and data structures."""

import pytest
from datetime import datetime, timedelta

from shared.plugins.background.protocol import (
    BackgroundCapable,
    TaskHandle,
    TaskResult,
    TaskStatus,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.TIMEOUT.value == "timeout"

    def test_status_from_string(self):
        """Test creating status from string value."""
        assert TaskStatus("pending") == TaskStatus.PENDING
        assert TaskStatus("completed") == TaskStatus.COMPLETED


class TestTaskHandle:
    """Tests for TaskHandle dataclass."""

    def test_create_handle(self):
        """Test creating a task handle."""
        now = datetime.now()
        handle = TaskHandle(
            task_id="test-123",
            plugin_name="test_plugin",
            tool_name="test_tool",
            created_at=now,
        )

        assert handle.task_id == "test-123"
        assert handle.plugin_name == "test_plugin"
        assert handle.tool_name == "test_tool"
        assert handle.created_at == now
        assert handle.estimated_duration_seconds is None
        assert handle.metadata == {}

    def test_create_handle_with_optional_fields(self):
        """Test creating a handle with all optional fields."""
        now = datetime.now()
        handle = TaskHandle(
            task_id="test-456",
            plugin_name="cli",
            tool_name="runCommand",
            created_at=now,
            estimated_duration_seconds=30.0,
            metadata={"command": "npm install"},
        )

        assert handle.estimated_duration_seconds == 30.0
        assert handle.metadata == {"command": "npm install"}

    def test_handle_to_dict(self):
        """Test serializing handle to dictionary."""
        now = datetime(2025, 1, 15, 10, 30, 0)
        handle = TaskHandle(
            task_id="test-789",
            plugin_name="mcp",
            tool_name="search",
            created_at=now,
            estimated_duration_seconds=5.0,
        )

        d = handle.to_dict()

        assert d["task_id"] == "test-789"
        assert d["plugin_name"] == "mcp"
        assert d["tool_name"] == "search"
        assert d["created_at"] == "2025-01-15T10:30:00"
        assert d["estimated_duration_seconds"] == 5.0


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_create_completed_result(self):
        """Test creating a completed task result."""
        start = datetime.now()
        end = start + timedelta(seconds=5)

        result = TaskResult(
            task_id="test-123",
            status=TaskStatus.COMPLETED,
            result={"output": "success"},
            started_at=start,
            completed_at=end,
            duration_seconds=5.0,
        )

        assert result.task_id == "test-123"
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"output": "success"}
        assert result.error is None
        assert result.duration_seconds == 5.0

    def test_create_failed_result(self):
        """Test creating a failed task result."""
        result = TaskResult(
            task_id="test-456",
            status=TaskStatus.FAILED,
            error="Connection timeout",
        )

        assert result.status == TaskStatus.FAILED
        assert result.error == "Connection timeout"
        assert result.result is None

    def test_result_to_dict(self):
        """Test serializing result to dictionary."""
        start = datetime(2025, 1, 15, 10, 30, 0)
        end = datetime(2025, 1, 15, 10, 30, 10)

        result = TaskResult(
            task_id="test-789",
            status=TaskStatus.COMPLETED,
            result={"data": [1, 2, 3]},
            started_at=start,
            completed_at=end,
            duration_seconds=10.0,
        )

        d = result.to_dict()

        assert d["task_id"] == "test-789"
        assert d["status"] == "completed"
        assert d["result"] == {"data": [1, 2, 3]}
        assert d["started_at"] == "2025-01-15T10:30:00"
        assert d["completed_at"] == "2025-01-15T10:30:10"
        assert d["duration_seconds"] == 10.0


class TestBackgroundCapableProtocol:
    """Tests for BackgroundCapable protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that the protocol can be used with isinstance."""
        # This should not raise
        isinstance(object(), BackgroundCapable)

    def test_protocol_methods_defined(self):
        """Test that all expected methods are in the protocol."""
        # Get protocol methods
        protocol_methods = [
            name for name in dir(BackgroundCapable)
            if not name.startswith('_')
        ]

        expected_methods = [
            'supports_background',
            'get_auto_background_threshold',
            'estimate_duration',
            'start_background',
            'get_status',
            'get_result',
            'cancel',
            'list_tasks',
            'cleanup_completed',
            'register_running_task',
        ]

        for method in expected_methods:
            assert method in protocol_methods, f"Missing protocol method: {method}"
