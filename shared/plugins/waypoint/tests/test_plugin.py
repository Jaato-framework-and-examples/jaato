"""Tests for WaypointPlugin."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from ..plugin import WaypointPlugin, create_plugin
from ..models import INITIAL_WAYPOINT_ID


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_backup_manager():
    """Create a mock BackupManager."""
    mock = MagicMock()
    mock.current_waypoint = INITIAL_WAYPOINT_ID
    mock.get_backups_by_waypoint.return_value = []
    mock.get_first_backup_per_file_by_waypoint.return_value = {}
    mock.restore_from_backup.return_value = True
    return mock


@pytest.fixture
def plugin(temp_dir, mock_backup_manager):
    """Create an initialized WaypointPlugin for testing."""
    plugin = create_plugin()
    plugin.initialize({
        "backup_manager": mock_backup_manager,
        "storage_path": temp_dir / "waypoints.json",
    })
    return plugin


class TestPluginProtocol:
    """Test plugin protocol implementation."""

    def test_name(self, plugin):
        """Test plugin name."""
        assert plugin.name == "waypoint"

    def test_no_tool_schemas(self, plugin):
        """Test that waypoint has no tool schemas (not exposed to model)."""
        assert plugin.get_tool_schemas() == []

    def test_no_system_instructions(self, plugin):
        """Test that waypoint has no system instructions."""
        assert plugin.get_system_instructions() is None

    def test_auto_approved(self, plugin):
        """Test waypoint command is auto-approved."""
        assert "waypoint" in plugin.get_auto_approved_tools()

    def test_user_commands(self, plugin):
        """Test user commands are declared."""
        commands = plugin.get_user_commands()
        assert len(commands) == 1
        assert commands[0].name == "waypoint"


class TestCommandExecution:
    """Test command execution."""

    def test_list_empty(self, plugin):
        """Test listing when only w0 exists."""
        result = plugin._execute_waypoint({"action": "list"})
        assert "waypoints" in result
        assert len(result["waypoints"]) == 1

    def test_create_with_description(self, plugin):
        """Test creating waypoint with description."""
        result = plugin._execute_waypoint({
            "action": "create",
            "target": '"test checkpoint"',
        })
        assert result["success"] is True
        assert result["id"] == "w1"

    def test_create_auto_description(self, plugin):
        """Test creating waypoint with auto-generated description."""
        result = plugin._execute_waypoint({"action": "create"})
        assert result["success"] is True
        assert result["description"]  # Should have some description

    def test_delete(self, plugin):
        """Test deleting a waypoint."""
        plugin._execute_waypoint({"action": "create", "target": '"test"'})
        result = plugin._execute_waypoint({
            "action": "delete",
            "target": "w1",
        })
        assert result["success"] is True

    def test_delete_all(self, plugin):
        """Test deleting all waypoints."""
        plugin._execute_waypoint({"action": "create", "target": '"first"'})
        plugin._execute_waypoint({"action": "create", "target": '"second"'})

        result = plugin._execute_waypoint({
            "action": "delete",
            "target": "all",
        })
        assert result["success"] is True
        assert result["deleted"] == 2

    def test_info(self, plugin):
        """Test getting waypoint info."""
        plugin._execute_waypoint({"action": "create", "target": '"test"'})
        result = plugin._execute_waypoint({
            "action": "info",
            "target": "w1",
        })
        assert result["id"] == "w1"

    def test_unknown_action(self, plugin):
        """Test unknown action returns error."""
        result = plugin._execute_waypoint({"action": "invalid"})
        assert "error" in result


class TestCommandCompletions:
    """Test command completions."""

    def test_no_args_completions(self, plugin):
        """Test completions when no args provided."""
        completions = plugin.get_command_completions("waypoint", [])
        values = [c.value for c in completions]
        assert "create" in values
        assert "restore" in values
        assert "delete" in values

    def test_restore_completions(self, plugin):
        """Test completions for restore command."""
        plugin._execute_waypoint({"action": "create", "target": '"test"'})

        completions = plugin.get_command_completions("waypoint", ["restore"])
        values = [c.value for c in completions]
        assert "w0" in values
        assert "w1" in values

    def test_delete_all_completion(self, plugin):
        """Test 'all' completion for delete command."""
        completions = plugin.get_command_completions("waypoint", ["delete"])
        values = [c.value for c in completions]
        assert "all" in values

    def test_wrong_command(self, plugin):
        """Test no completions for wrong command."""
        completions = plugin.get_command_completions("other", [])
        assert completions == []
