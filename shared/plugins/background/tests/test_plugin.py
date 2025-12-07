"""Tests for BackgroundPlugin orchestrator."""

import pytest
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from shared.plugins.background.plugin import BackgroundPlugin, create_plugin
from shared.plugins.background.mixin import BackgroundCapableMixin
from shared.plugins.background.protocol import TaskHandle, TaskStatus


class MockCapablePlugin(BackgroundCapableMixin):
    """Mock plugin with background capability."""

    def __init__(self, plugin_name: str = "mock_capable"):
        super().__init__(max_workers=2)
        self._plugin_name = plugin_name

    @property
    def name(self) -> str:
        return self._plugin_name

    def supports_background(self, tool_name: str) -> bool:
        return tool_name in ["bg_tool", "slow_bg_tool"]

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        if tool_name == "slow_bg_tool":
            return 1.0
        return None

    def estimate_duration(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[float]:
        return arguments.get("duration")

    def get_executors(self) -> Dict[str, Any]:
        return {
            "bg_tool": self._execute_bg,
            "slow_bg_tool": self._execute_slow_bg,
            "normal_tool": self._execute_normal,
        }

    def _execute_bg(self, args: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(args.get("duration", 0.1))
        return {"status": "bg_done", **args}

    def _execute_slow_bg(self, args: Dict[str, Any]) -> Dict[str, Any]:
        time.sleep(args.get("duration", 2.0))
        return {"status": "slow_bg_done", **args}

    def _execute_normal(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "normal_done", **args}


class MockNonCapablePlugin:
    """Mock plugin without background capability."""

    @property
    def name(self) -> str:
        return "mock_noncapable"

    def get_executors(self) -> Dict[str, Any]:
        return {"sync_tool": lambda args: {"result": "sync"}}


class MockRegistry:
    """Mock plugin registry for testing."""

    def __init__(self, plugins: Dict[str, Any] = None):
        self._plugins = plugins or {}
        self._exposed = set(self._plugins.keys())

    def list_exposed(self) -> List[str]:
        return list(self._exposed)

    def get_plugin(self, name: str) -> Any:
        return self._plugins.get(name)

    def add_plugin(self, plugin: Any) -> None:
        self._plugins[plugin.name] = plugin
        self._exposed.add(plugin.name)


class TestBackgroundPlugin:
    """Tests for BackgroundPlugin orchestrator."""

    def test_create_plugin(self):
        """Test plugin factory function."""
        plugin = create_plugin()
        assert isinstance(plugin, BackgroundPlugin)
        assert plugin.name == "background"

    def test_initialize_and_shutdown(self):
        """Test plugin lifecycle."""
        plugin = BackgroundPlugin()
        plugin.initialize()
        assert plugin._initialized is True

        plugin.shutdown()
        assert plugin._initialized is False

    def test_set_registry_discovers_capable_plugins(self):
        """Test that set_registry discovers BackgroundCapable plugins."""
        capable = MockCapablePlugin("test_capable")
        noncapable = MockNonCapablePlugin()

        registry = MockRegistry()
        registry.add_plugin(capable)
        registry.add_plugin(noncapable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        # Should only find the capable plugin
        assert "test_capable" in plugin._capable_plugins
        assert "mock_noncapable" not in plugin._capable_plugins

    def test_get_function_declarations(self):
        """Test that all expected tools are declared."""
        plugin = BackgroundPlugin()
        declarations = plugin.get_function_declarations()

        tool_names = [d.name for d in declarations]

        expected_tools = [
            "startBackgroundTask",
            "getBackgroundTaskStatus",
            "getBackgroundTaskResult",
            "cancelBackgroundTask",
            "listBackgroundTasks",
            "listBackgroundCapableTools",
        ]

        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool declaration: {tool}"

    def test_get_executors(self):
        """Test that all expected executors are registered."""
        plugin = BackgroundPlugin()
        executors = plugin.get_executors()

        expected_executors = [
            "startBackgroundTask",
            "getBackgroundTaskStatus",
            "getBackgroundTaskResult",
            "cancelBackgroundTask",
            "listBackgroundTasks",
            "listBackgroundCapableTools",
        ]

        for name in expected_executors:
            assert name in executors, f"Missing executor: {name}"

    def test_get_system_instructions(self):
        """Test that system instructions are provided."""
        plugin = BackgroundPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "background" in instructions.lower()
        assert "startBackgroundTask" in instructions

    def test_get_auto_approved_tools(self):
        """Test that safe tools are auto-approved."""
        plugin = BackgroundPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        # Status/list tools should be auto-approved
        assert "getBackgroundTaskStatus" in auto_approved
        assert "listBackgroundTasks" in auto_approved
        assert "listBackgroundCapableTools" in auto_approved

        # Start/cancel should NOT be auto-approved
        assert "startBackgroundTask" not in auto_approved
        assert "cancelBackgroundTask" not in auto_approved

    def test_start_task_missing_params(self):
        """Test startBackgroundTask with missing parameters."""
        plugin = BackgroundPlugin()

        result = plugin._start_task({})
        assert result["success"] is False
        assert "required" in result["error"]

    def test_start_task_tool_not_found(self):
        """Test startBackgroundTask with tool that doesn't support background."""
        plugin = BackgroundPlugin()
        plugin._capable_plugins = {}

        result = plugin._start_task({
            "tool_name": "some_tool",
            "arguments": {},
        })

        assert result["success"] is False
        assert "does not support background" in result["error"]

    def test_start_task_tool_not_supported(self):
        """Test startBackgroundTask with unsupported tool."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        result = plugin._start_task({
            "tool_name": "normal_tool",  # Not in supports_background
            "arguments": {},
        })

        assert result["success"] is False
        assert "does not support background" in result["error"]

    def test_start_task_success(self):
        """Test successful startBackgroundTask."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        result = plugin._start_task({
            "tool_name": "bg_tool",
            "arguments": {"duration": 0.1},
        })

        assert result["success"] is True
        assert "task_id" in result
        assert result["plugin_name"] == "mock_capable"
        assert result["tool_name"] == "bg_tool"

    def test_get_status_not_found(self):
        """Test getBackgroundTaskStatus with unknown task."""
        plugin = BackgroundPlugin()

        result = plugin._get_status({"task_id": "nonexistent"})
        assert "error" in result
        assert "not found" in result["error"]

    def test_get_status_success(self):
        """Test getBackgroundTaskStatus with valid task."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        # Start a task
        start_result = plugin._start_task({
            "tool_name": "bg_tool",
            "arguments": {"duration": 0.5},
        })
        task_id = start_result["task_id"]

        # Check status
        status_result = plugin._get_status({"task_id": task_id})

        assert "error" not in status_result
        assert status_result["task_id"] == task_id
        assert status_result["plugin_name"] == "mock_capable"
        assert status_result["status"] in ["pending", "running", "completed"]

    def test_get_result_not_found(self):
        """Test getBackgroundTaskResult with unknown task."""
        plugin = BackgroundPlugin()

        result = plugin._get_result({"task_id": "nonexistent"})
        assert "error" in result
        assert "not found" in result["error"]

    def test_get_result_success(self):
        """Test getBackgroundTaskResult with completed task."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        # Start and wait for task
        start_result = plugin._start_task({
            "tool_name": "bg_tool",
            "arguments": {"duration": 0.1, "key": "value"},
        })
        task_id = start_result["task_id"]

        time.sleep(0.2)

        # Get result
        result = plugin._get_result({"task_id": task_id})

        assert "error" not in result
        assert result["status"] == "completed"
        assert result["result"]["status"] == "bg_done"
        assert result["result"]["key"] == "value"

    def test_get_result_with_wait(self):
        """Test getBackgroundTaskResult with wait=True."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        # Start a slightly longer task
        start_result = plugin._start_task({
            "tool_name": "bg_tool",
            "arguments": {"duration": 0.3},
        })
        task_id = start_result["task_id"]

        # Get result with wait - should block until complete
        result = plugin._get_result({"task_id": task_id, "wait": True})

        assert result["status"] == "completed"

    def test_cancel_task_not_found(self):
        """Test cancelBackgroundTask with unknown task."""
        plugin = BackgroundPlugin()

        result = plugin._cancel_task({"task_id": "nonexistent"})
        assert "error" in result
        assert "not found" in result["error"]

    def test_cancel_task_success(self):
        """Test cancelBackgroundTask with running task."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        # Start a long task
        start_result = plugin._start_task({
            "tool_name": "slow_bg_tool",
            "arguments": {"duration": 10.0},
        })
        task_id = start_result["task_id"]

        # Cancel immediately
        cancel_result = plugin._cancel_task({"task_id": task_id})

        assert "cancelled" in cancel_result
        assert cancel_result["cancelled"] is True

    def test_list_tasks_empty(self):
        """Test listBackgroundTasks with no active tasks."""
        plugin = BackgroundPlugin()

        result = plugin._list_tasks({})

        assert result["tasks"] == []
        assert result["count"] == 0

    def test_list_tasks_with_active_tasks(self):
        """Test listBackgroundTasks with active tasks."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        # Start some tasks
        task_ids = []
        for i in range(3):
            result = plugin._start_task({
                "tool_name": "slow_bg_tool",
                "arguments": {"duration": 5.0, "i": i},
            })
            task_ids.append(result["task_id"])

        # List tasks
        list_result = plugin._list_tasks({})

        assert list_result["count"] >= 1
        listed_ids = [t["task_id"] for t in list_result["tasks"]]
        for task_id in task_ids:
            assert task_id in listed_ids

        # Cleanup
        for task_id in task_ids:
            plugin._cancel_task({"task_id": task_id})

    def test_list_tasks_filter_by_plugin(self):
        """Test listBackgroundTasks with plugin filter."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        # Start some tasks
        result1 = plugin._start_task({
            "tool_name": "slow_bg_tool",
            "arguments": {"duration": 5.0},
        })
        result2 = plugin._start_task({
            "tool_name": "slow_bg_tool",
            "arguments": {"duration": 5.0},
        })

        # Filter by mock_capable plugin
        list_result = plugin._list_tasks({"plugin_name": "mock_capable"})

        assert list_result["count"] == 2
        assert all(t["plugin_name"] == "mock_capable" for t in list_result["tasks"])

        # Filter by non-existent plugin - should return empty
        list_result_empty = plugin._list_tasks({"plugin_name": "other_plugin"})
        assert list_result_empty["count"] == 0

        # Cleanup
        plugin._cancel_task({"task_id": result1["task_id"]})
        plugin._cancel_task({"task_id": result2["task_id"]})

    def test_list_capable_tools(self):
        """Test listBackgroundCapableTools."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.set_registry(registry)

        result = plugin._list_capable_tools({})

        assert result["count"] >= 1

        tool_names = [t["tool_name"] for t in result["tools"]]
        assert "bg_tool" in tool_names
        assert "slow_bg_tool" in tool_names

        # Check auto-background threshold is included
        slow_tool = next(t for t in result["tools"] if t["tool_name"] == "slow_bg_tool")
        assert slow_tool["auto_background_threshold_seconds"] == 1.0

    def test_user_commands(self):
        """Test that user commands are defined."""
        plugin = BackgroundPlugin()
        commands = plugin.get_user_commands()

        assert len(commands) >= 1
        assert any(c.name == "tasks" for c in commands)


class TestBackgroundPluginIntegration:
    """Integration tests for BackgroundPlugin."""

    def test_full_workflow(self):
        """Test complete background task workflow."""
        capable = MockCapablePlugin()
        registry = MockRegistry()
        registry.add_plugin(capable)

        plugin = BackgroundPlugin()
        plugin.initialize()
        plugin.set_registry(registry)

        # 1. List capable tools
        tools = plugin._list_capable_tools({})
        assert tools["count"] > 0

        # 2. Start a background task
        start = plugin._start_task({
            "tool_name": "bg_tool",
            "arguments": {"data": "test", "duration": 0.2},
        })
        assert start["success"] is True
        task_id = start["task_id"]

        # 3. Check status (should be running)
        status = plugin._get_status({"task_id": task_id})
        assert status["status"] in ["pending", "running"]

        # 4. List active tasks
        tasks = plugin._list_tasks({})
        assert any(t["task_id"] == task_id for t in tasks["tasks"])

        # 5. Wait and get result
        time.sleep(0.3)
        result = plugin._get_result({"task_id": task_id})
        assert result["status"] == "completed"
        assert result["result"]["data"] == "test"

        plugin.shutdown()
