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
    # Use a mutable container to track current_waypoint state
    state = {"current": INITIAL_WAYPOINT_ID}
    type(mock).current_waypoint = property(lambda self: state["current"])
    mock.set_current_waypoint.side_effect = lambda wp_id: state.update({"current": wp_id})
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

    def test_list_shows_current_waypoint(self, plugin, mock_backup_manager):
        """Test that list shows which waypoint is current."""
        # Initially w0 is current
        result = plugin._execute_waypoint({"action": "list"})
        assert result["current"] == "w0"
        assert result["waypoints"][0]["is_current"] is True

        # Create a new waypoint - it becomes current
        plugin._execute_waypoint({"action": "create", "target": '"first"'})
        result = plugin._execute_waypoint({"action": "list"})
        assert result["current"] == "w1"
        # w0 should no longer be current
        w0 = next(wp for wp in result["waypoints"] if wp["id"] == "w0")
        w1 = next(wp for wp in result["waypoints"] if wp["id"] == "w1")
        assert w0["is_current"] is False
        assert w1["is_current"] is True

        # Restore to w0 - current should change back
        mock_backup_manager.get_first_backup_per_file_by_waypoint.return_value = {}
        plugin._execute_waypoint({"action": "restore", "target": "w0"})
        result = plugin._execute_waypoint({"action": "list"})
        assert result["current"] == "w0"
        w0 = next(wp for wp in result["waypoints"] if wp["id"] == "w0")
        w1 = next(wp for wp in result["waypoints"] if wp["id"] == "w1")
        assert w0["is_current"] is True
        assert w1["is_current"] is False

    def test_create_with_description(self, plugin):
        """Test creating waypoint with description."""
        result = plugin._execute_waypoint({
            "action": "create",
            "target": '"test checkpoint"',
        })
        assert result["success"] is True
        assert result["id"] == "w1"

    def test_create_without_description_returns_error(self, plugin):
        """Test that creating waypoint without description returns error."""
        result = plugin._execute_waypoint({"action": "create"})
        assert "error" in result
        assert "Description required" in result["error"]

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


class TestPromptEnrichment:
    """Test prompt enrichment for waypoint restore notifications."""

    def test_subscribes_to_prompt_enrichment(self, plugin):
        """Test that plugin subscribes to prompt enrichment."""
        assert plugin.subscribes_to_prompt_enrichment() is True

    def test_no_enrichment_without_restore(self, plugin):
        """Test that prompt is unchanged when no restore has occurred."""
        result = plugin.enrich_prompt("Hello, world!")
        assert result.prompt == "Hello, world!"
        assert "waypoint_restore" not in result.metadata

    def test_enrichment_after_restore(self, plugin, mock_backup_manager):
        """Test that prompt is enriched after a restore."""
        # Create a waypoint
        plugin._execute_waypoint({"action": "create", "target": '"test waypoint"'})

        # Mock that files were restored
        mock_backup_manager.get_first_backup_per_file_by_waypoint.return_value = {
            "/path/to/file.py": MagicMock(backup_path=Path("/backup/file.py"))
        }

        # Restore to the waypoint
        result = plugin._execute_waypoint({
            "action": "restore",
            "target": "w1",
        })
        assert result.get("success") is True

        # Now enrich a prompt - should include restore notification
        enrich_result = plugin.enrich_prompt("What's in the file?")

        # Check for hidden-wrapped waypoint-restore notification
        assert "<hidden>" in enrich_result.prompt
        assert "<waypoint-restore>" in enrich_result.prompt
        assert "w1" in enrich_result.prompt
        assert "test waypoint" in enrich_result.prompt
        assert "What's in the file?" in enrich_result.prompt
        assert enrich_result.metadata.get("waypoint_restore") is not None

    def test_enrichment_consumed_after_use(self, plugin, mock_backup_manager):
        """Test that the restore notification is only shown once."""
        # Create and restore
        plugin._execute_waypoint({"action": "create", "target": '"test"'})
        plugin._execute_waypoint({
            "action": "restore",
            "target": "w1",
        })

        # First enrichment includes notification (hidden-wrapped)
        result1 = plugin.enrich_prompt("First prompt")
        assert "<hidden>" in result1.prompt
        assert "<waypoint-restore>" in result1.prompt

        # Second enrichment should NOT include notification (consumed)
        result2 = plugin.enrich_prompt("Second prompt")
        assert "<hidden>" not in result2.prompt
        assert result2.prompt == "Second prompt"
