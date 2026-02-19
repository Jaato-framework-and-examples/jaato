"""Tests for ToolExecutor auto-background functionality."""

import pytest
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from shared.ai_tool_runner import ToolExecutor
from shared.plugins.background.mixin import BackgroundCapableMixin
from shared.plugins.background.protocol import TaskStatus


class SlowBackgroundPlugin(BackgroundCapableMixin):
    """Plugin with auto-background support for testing."""

    def __init__(self):
        super().__init__(max_workers=2)

    @property
    def name(self) -> str:
        return "slow_plugin"

    def supports_background(self, tool_name: str) -> bool:
        return tool_name in ["auto_bg_tool", "fast_tool"]

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        if tool_name == "auto_bg_tool":
            return 0.2  # 200ms threshold for testing
        return None  # fast_tool has no auto-background

    def estimate_duration(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[float]:
        return arguments.get("duration")

    def get_executors(self) -> Dict[str, Any]:
        return {
            "auto_bg_tool": self._execute_auto_bg,
            "fast_tool": self._execute_fast,
        }

    def _execute_auto_bg(self, args: Dict[str, Any]) -> Dict[str, Any]:
        duration = args.get("duration", 1.0)
        time.sleep(duration)
        return {"status": "completed", "duration": duration}

    def _execute_fast(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "fast_completed"}


class NonBackgroundPlugin:
    """Plugin without background capability."""

    @property
    def name(self) -> str:
        return "non_bg_plugin"

    def get_executors(self) -> Dict[str, Any]:
        return {
            "sync_tool": self._execute_sync,
        }

    def _execute_sync(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "sync_completed"}


class MockRegistry:
    """Mock registry for testing."""

    def __init__(self):
        self._plugins: Dict[str, Any] = {}
        self._tool_to_plugin: Dict[str, Any] = {}

    def add_plugin(self, plugin: Any) -> None:
        self._plugins[plugin.name] = plugin
        for tool_name in plugin.get_executors().keys():
            self._tool_to_plugin[tool_name] = plugin

    def get_plugin_for_tool(self, tool_name: str) -> Optional[Any]:
        return self._tool_to_plugin.get(tool_name)

    def get_plugin(self, name: str) -> Optional[Any]:
        return self._plugins.get(name)


class TestToolExecutorAutoBackground:
    """Tests for ToolExecutor auto-background functionality."""

    def test_auto_background_disabled(self):
        """Test that auto-background can be disabled."""
        executor = ToolExecutor(auto_background_enabled=False)

        slow_plugin = SlowBackgroundPlugin()
        registry = MockRegistry()
        registry.add_plugin(slow_plugin)

        executor.set_registry(registry)
        executor.register("auto_bg_tool", slow_plugin._execute_auto_bg)

        # Should execute synchronously even with slow tool
        start = time.time()
        ok, result = executor.execute("auto_bg_tool", {"duration": 0.3})
        elapsed = time.time() - start

        # Should have waited for completion
        assert elapsed >= 0.3
        assert ok is True
        assert result["status"] == "completed"

    def test_auto_background_no_registry(self):
        """Test that without registry, auto-background is skipped."""
        executor = ToolExecutor(auto_background_enabled=True)

        def slow_fn(args):
            time.sleep(0.3)
            return {"done": True}

        executor.register("slow_tool", slow_fn)

        # No registry set - should execute synchronously
        start = time.time()
        ok, result = executor.execute("slow_tool", {})
        elapsed = time.time() - start

        assert elapsed >= 0.3
        assert ok is True
        assert result["done"] is True

    def test_auto_background_non_capable_plugin(self):
        """Test that non-capable plugins execute synchronously."""
        executor = ToolExecutor(auto_background_enabled=True)

        non_bg_plugin = NonBackgroundPlugin()
        registry = MockRegistry()
        registry.add_plugin(non_bg_plugin)

        executor.set_registry(registry)
        executor.register("sync_tool", non_bg_plugin._execute_sync)

        ok, result = executor.execute("sync_tool", {})

        assert ok is True
        assert result["status"] == "sync_completed"
        # No auto_backgrounded flag
        assert "auto_backgrounded" not in result

    def test_auto_background_tool_without_threshold(self):
        """Test that tools without threshold execute synchronously."""
        executor = ToolExecutor(auto_background_enabled=True)

        slow_plugin = SlowBackgroundPlugin()
        registry = MockRegistry()
        registry.add_plugin(slow_plugin)

        executor.set_registry(registry)
        executor.register("fast_tool", slow_plugin._execute_fast)

        # fast_tool has no auto-background threshold
        ok, result = executor.execute("fast_tool", {})

        assert ok is True
        assert result["status"] == "fast_completed"
        assert "auto_backgrounded" not in result

    def test_auto_background_fast_completion(self):
        """Test that fast-completing tools return normally."""
        executor = ToolExecutor(auto_background_enabled=True)

        slow_plugin = SlowBackgroundPlugin()
        registry = MockRegistry()
        registry.add_plugin(slow_plugin)

        executor.set_registry(registry)
        executor.register("auto_bg_tool", slow_plugin._execute_auto_bg)

        # Execute with short duration (under threshold)
        ok, result = executor.execute("auto_bg_tool", {"duration": 0.05})

        assert ok is True
        assert result["status"] == "completed"
        # Should NOT be auto-backgrounded
        assert "auto_backgrounded" not in result

    def test_auto_background_slow_execution(self):
        """Test that slow execution triggers auto-background."""
        executor = ToolExecutor(auto_background_enabled=True)

        slow_plugin = SlowBackgroundPlugin()
        registry = MockRegistry()
        registry.add_plugin(slow_plugin)

        executor.set_registry(registry)
        executor.register("auto_bg_tool", slow_plugin._execute_auto_bg)

        # Execute with long duration (over 0.2s threshold)
        start = time.time()
        ok, result = executor.execute("auto_bg_tool", {"duration": 1.0})
        elapsed = time.time() - start

        # Should return quickly (not wait for full execution)
        assert elapsed < 0.5  # Should be much less than 1.0s

        assert ok is True
        assert result.get("auto_backgrounded") is True
        assert "task_id" in result
        assert result["plugin_name"] == "slow_plugin"
        assert result["tool_name"] == "auto_bg_tool"
        assert result["threshold_seconds"] == 0.2

    def test_auto_background_result_retrieval(self):
        """Test retrieving result from auto-backgrounded task."""
        executor = ToolExecutor(auto_background_enabled=True)

        slow_plugin = SlowBackgroundPlugin()
        registry = MockRegistry()
        registry.add_plugin(slow_plugin)

        executor.set_registry(registry)
        executor.register("auto_bg_tool", slow_plugin._execute_auto_bg)

        # Trigger auto-background
        ok, result = executor.execute("auto_bg_tool", {"duration": 0.5})

        assert result.get("auto_backgrounded") is True
        task_id = result["task_id"]

        # Check status
        status = slow_plugin.get_status(task_id)
        assert status in (TaskStatus.PENDING, TaskStatus.RUNNING)

        # Wait for completion
        time.sleep(0.6)

        # Get result
        task_result = slow_plugin.get_result(task_id)
        assert task_result.status == TaskStatus.COMPLETED
        assert task_result.result["status"] == "completed"

    def test_auto_background_with_permission_metadata(self):
        """Test that permission metadata is preserved in auto-backgrounded results."""
        executor = ToolExecutor(auto_background_enabled=True)

        slow_plugin = SlowBackgroundPlugin()
        registry = MockRegistry()
        registry.add_plugin(slow_plugin)

        executor.set_registry(registry)
        executor.register("auto_bg_tool", slow_plugin._execute_auto_bg)

        # Mock permission metadata by calling internal method directly
        permission_meta = {"decision": "allowed", "method": "whitelist"}

        ok, result = executor._execute_with_auto_background(
            "auto_bg_tool",
            {"duration": 1.0},
            slow_plugin,
            0.1,  # Very short threshold
            permission_meta
        )

        assert result.get("auto_backgrounded") is True
        assert result.get("_permission") == permission_meta

    def test_executor_pool_size(self):
        """Test that executor uses configured pool size."""
        executor = ToolExecutor(
            auto_background_enabled=True,
            auto_background_pool_size=2
        )

        assert executor._auto_background_pool_size == 2
        assert executor._auto_background_pool is None  # Lazy init

        # Trigger pool creation
        pool = executor._get_auto_background_pool()
        assert pool is not None
        assert pool._max_workers == 2

    def test_execute_sync_helper(self):
        """Test _execute_sync helper method."""
        executor = ToolExecutor()

        def test_fn(args):
            return {"result": args.get("value")}

        executor.register("test_tool", test_fn)

        ok, result = executor._execute_sync("test_tool", {"value": 42})

        assert ok is True
        assert result["result"] == 42

    def test_execute_sync_not_found(self):
        """Test _execute_sync with unregistered tool."""
        executor = ToolExecutor()

        ok, result = executor._execute_sync("unknown_tool", {})

        assert ok is False
        assert "error" in result
        assert "No executor registered" in result["error"]

    def test_execute_sync_registry_fallback(self):
        """Test _execute_sync falls back to registry for unregistered tools.

        This tests the scenario where MCP tools are discovered after session
        configuration. The executor should query the registry to find the
        executor and cache it.
        """
        executor = ToolExecutor()

        # Create a mock plugin with an executor for a tool
        mock_plugin = MagicMock()
        mock_plugin.get_executors.return_value = {
            "mcp_tool": lambda args: {"result": args.get("value", 0) * 2}
        }

        # Create a mock registry that returns the plugin
        mock_registry = MagicMock()
        mock_registry.get_plugin_for_tool.return_value = mock_plugin

        executor.set_registry(mock_registry)

        # Tool not registered locally, but registry fallback should find it
        ok, result = executor._execute_sync("mcp_tool", {"value": 21})

        assert ok is True
        assert result["result"] == 42

        # Verify the registry was queried
        mock_registry.get_plugin_for_tool.assert_called_with("mcp_tool")
        mock_plugin.get_executors.assert_called_once()

        # Verify the executor was cached for future calls
        assert "mcp_tool" in executor._map

    def test_execute_registry_fallback(self):
        """Test execute() falls back to registry for unregistered tools.

        Tests the main execute() code path (via _execute_impl) for the same
        registry fallback scenario.
        """
        executor = ToolExecutor()

        # Create a mock plugin with an executor for a tool
        mock_plugin = MagicMock()
        mock_plugin.get_executors.return_value = {
            "mcp_tool": lambda args: {"result": args.get("value", 0) * 2}
        }

        # Create a mock registry that returns the plugin
        mock_registry = MagicMock()
        mock_registry.get_plugin_for_tool.return_value = mock_plugin

        executor.set_registry(mock_registry)

        # Tool not registered locally, but registry fallback should find it
        ok, result = executor.execute("mcp_tool", {"value": 21})

        assert ok is True
        assert result["result"] == 42

        # Verify the executor was cached for future calls
        assert "mcp_tool" in executor._map

        # Second call should use cached executor
        # Note: get_plugin_for_tool is still called for auto-background check,
        # but get_executors should not be called since executor is cached
        mock_plugin.reset_mock()

        ok2, result2 = executor.execute("mcp_tool", {"value": 10})
        assert ok2 is True
        assert result2["result"] == 20

        # get_executors should not be called again since executor is cached
        mock_plugin.get_executors.assert_not_called()

    def test_execute_sync_exception(self):
        """Test _execute_sync with exception."""
        executor = ToolExecutor()

        def failing_fn(args):
            raise ValueError("Test error")

        executor.register("failing_tool", failing_fn)

        ok, result = executor._execute_sync("failing_tool", {})

        assert ok is False
        assert "error" in result
        assert "Test error" in result["error"]


class TestToolExecutorWithLedger:
    """Tests for ToolExecutor auto-background with ledger recording."""

    def test_auto_background_records_to_ledger(self):
        """Test that auto-background events are recorded to ledger."""
        mock_ledger = MagicMock()

        executor = ToolExecutor(
            ledger=mock_ledger,
            auto_background_enabled=True
        )

        slow_plugin = SlowBackgroundPlugin()
        registry = MockRegistry()
        registry.add_plugin(slow_plugin)

        executor.set_registry(registry)
        executor.register("auto_bg_tool", slow_plugin._execute_auto_bg)

        # Trigger auto-background
        ok, result = executor.execute("auto_bg_tool", {"duration": 1.0})

        assert result.get("auto_backgrounded") is True

        # Check that _record was called with auto-background event
        calls = [c for c in mock_ledger._record.call_args_list
                 if c[0][0] == 'auto-background']
        assert len(calls) == 1

        event_data = calls[0][0][1]
        assert event_data['tool'] == 'auto_bg_tool'
        assert 'task_id' in event_data
        assert event_data['threshold'] == 0.2
