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

    def test_tool_schemas_for_model(self, plugin):
        """Test that waypoint exposes tool schemas for model access."""
        schemas = plugin.get_tool_schemas()
        assert len(schemas) == 5
        names = [s.name for s in schemas]
        assert "list_waypoints" in names
        assert "waypoint_info" in names
        assert "create_waypoint" in names
        assert "restore_waypoint" in names
        assert "delete_waypoint" in names

    def test_system_instructions(self, plugin):
        """Test that waypoint has system instructions for model."""
        instructions = plugin.get_system_instructions()
        assert instructions is not None
        assert "Tree Structure" in instructions
        assert "Ownership Model" in instructions
        assert "User-owned" in instructions
        assert "Model-owned" in instructions
        assert "Auto-save on restore" in instructions

    def test_auto_approved(self, plugin):
        """Test auto-approved tools include user command and safe model tools."""
        auto_approved = plugin.get_auto_approved_tools()
        # User command
        assert "waypoint" in auto_approved
        # Safe model tools (read-only and create)
        assert "list_waypoints" in auto_approved
        assert "waypoint_info" in auto_approved
        assert "create_waypoint" in auto_approved
        assert "delete_waypoint" in auto_approved  # Enforced in executor
        # Requires permission (can affect user waypoints)
        assert "restore_waypoint" not in auto_approved

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
        assert "tree" in result
        assert "current" in result
        assert result["current"] == "w0"
        assert "w0" in result["tree"]

    def test_list_shows_current_waypoint(self, plugin, mock_backup_manager):
        """Test that list shows which waypoint is current."""
        # Initially w0 is current
        result = plugin._execute_waypoint({"action": "list"})
        assert result["current"] == "w0"
        assert "w0" in result["tree"]
        assert "◀ current" in result["tree"]

        # Create a new waypoint - it becomes current
        plugin._execute_waypoint({"action": "create", "target": '"first"'})
        result = plugin._execute_waypoint({"action": "list"})
        assert result["current"] == "w1"
        # Tree should show w1 as current
        lines = result["tree"].split("\n")
        w0_line = next(l for l in lines if l.startswith("w0"))
        w1_line = next(l for l in lines if "w1" in l)
        assert "◀ current" not in w0_line
        assert "◀ current" in w1_line

        # Restore to w0 - current should change back
        mock_backup_manager.get_first_backup_per_file_by_waypoint.return_value = {}
        plugin._execute_waypoint({"action": "restore", "target": "w0"})
        result = plugin._execute_waypoint({"action": "list"})
        assert result["current"] == "w0"
        lines = result["tree"].split("\n")
        w0_line = next(l for l in lines if l.startswith("w0"))
        w1_line = next(l for l in lines if "w1" in l)
        assert "◀ current" in w0_line
        assert "◀ current" not in w1_line

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


class TestModelToolExecutors:
    """Test model tool executor methods."""

    def test_list_waypoints_includes_ownership(self, plugin):
        """Test list_waypoints returns ownership info in tree nodes."""
        # Create a user waypoint via command
        plugin._execute_waypoint({"action": "create", "target": '"user checkpoint"'})

        # Create a model waypoint via tool
        plugin._execute_create_waypoint({"description": "model checkpoint"})

        result = plugin._execute_list_waypoints({})
        assert "nodes" in result
        assert "tree" in result

        # Check ownership info is included (w1=user, w2=model - sequential IDs)
        nodes = result["nodes"]
        assert nodes["w1"]["owner"] == "user"
        assert nodes["w2"]["owner"] == "model"

        # Tree should show ownership tags
        assert "[user]" in result["tree"]
        assert "[model]" in result["tree"]

    def test_create_waypoint_model_owned(self, plugin):
        """Test create_waypoint creates model-owned waypoint with sequential ID."""
        result = plugin._execute_create_waypoint({"description": "model checkpoint"})

        assert result["success"] is True
        assert result["id"] == "w1"  # First waypoint after w0
        assert result["owner"] == "model"

    def test_create_waypoint_multiple_model_owned(self, plugin):
        """Test multiple model waypoints get sequential IDs."""
        result1 = plugin._execute_create_waypoint({"description": "first"})
        result2 = plugin._execute_create_waypoint({"description": "second"})
        result3 = plugin._execute_create_waypoint({"description": "third"})

        assert result1["id"] == "w1"
        assert result2["id"] == "w2"
        assert result3["id"] == "w3"

    def test_create_waypoint_requires_description(self, plugin):
        """Test create_waypoint requires description."""
        result = plugin._execute_create_waypoint({})
        assert "error" in result
        assert "description" in result["error"].lower()

    def test_waypoint_info_includes_ownership(self, plugin):
        """Test waypoint_info includes ownership."""
        plugin._execute_create_waypoint({"description": "model checkpoint"})

        result = plugin._execute_waypoint_info({"waypoint_id": "w1"})

        assert result["id"] == "w1"
        assert result["owner"] == "model"

    def test_delete_model_owned_waypoint(self, plugin):
        """Test model can delete its own waypoints."""
        plugin._execute_create_waypoint({"description": "to delete"})

        result = plugin._execute_delete_waypoint({"waypoint_id": "w1"})

        assert result["success"] is True
        assert result["id"] == "w1"

    def test_delete_user_owned_waypoint_rejected(self, plugin):
        """Test model cannot delete user-owned waypoints."""
        # Create user waypoint via command
        plugin._execute_waypoint({"action": "create", "target": '"user checkpoint"'})

        # Try to delete via model tool
        result = plugin._execute_delete_waypoint({"waypoint_id": "w1"})

        assert "error" in result
        assert "Cannot delete user-owned waypoint" in result["error"]

    def test_delete_initial_waypoint_rejected(self, plugin):
        """Test model cannot delete initial waypoint w0."""
        result = plugin._execute_delete_waypoint({"waypoint_id": "w0"})

        assert "error" in result
        assert "Cannot delete user-owned waypoint" in result["error"]

    def test_restore_model_owned_waypoint(self, plugin, mock_backup_manager):
        """Test model can restore its own waypoints."""
        plugin._execute_create_waypoint({"description": "checkpoint"})

        mock_backup_manager.get_first_backup_per_file_by_waypoint.return_value = {}

        result = plugin._execute_restore_waypoint({"waypoint_id": "w1"})

        assert result["success"] is True
        assert result["waypoint_id"] == "w1"

    def test_restore_nonexistent_waypoint(self, plugin):
        """Test restore returns error for nonexistent waypoint."""
        result = plugin._execute_restore_waypoint({"waypoint_id": "w999"})

        assert "error" in result
        assert "not found" in result["error"]

    def test_restore_requires_waypoint_id(self, plugin):
        """Test restore requires waypoint_id."""
        result = plugin._execute_restore_waypoint({})

        assert "error" in result
        assert "waypoint_id" in result["error"]


class TestOwnershipSeparation:
    """Test waypoint ownership and sequential IDs."""

    def test_sequential_ids_regardless_of_owner(self, plugin):
        """Test all waypoints share a single sequential counter."""
        # Create user waypoints
        plugin._execute_waypoint({"action": "create", "target": '"user 1"'})
        plugin._execute_waypoint({"action": "create", "target": '"user 2"'})

        # Create model waypoints
        plugin._execute_create_waypoint({"description": "model 1"})
        plugin._execute_create_waypoint({"description": "model 2"})

        # List all
        result = plugin._execute_list_waypoints({})
        nodes = result["nodes"]

        # Should have w0 (implicit), w1, w2 (user), w3, w4 (model) - all sequential
        assert "w0" in nodes
        assert "w1" in nodes
        assert "w2" in nodes
        assert "w3" in nodes
        assert "w4" in nodes

        # Check ownership is correct
        assert nodes["w1"]["owner"] == "user"
        assert nodes["w2"]["owner"] == "user"
        assert nodes["w3"]["owner"] == "model"
        assert nodes["w4"]["owner"] == "model"

    def test_user_can_delete_model_owned(self, plugin):
        """Test user can delete model-owned waypoints via command."""
        plugin._execute_create_waypoint({"description": "model checkpoint"})

        # Delete via user command
        result = plugin._execute_waypoint({"action": "delete", "target": "w1"})

        assert result["success"] is True

    def test_ids_monotonic_after_deletion(self, plugin):
        """Test that IDs are NOT reused after deletion (monotonic)."""
        # Create w1 (model-owned)
        plugin._execute_create_waypoint({"description": "first"})

        # Delete w1
        plugin._execute_delete_waypoint({"waypoint_id": "w1"})

        # Create again via user command - should be w2 (not w1)
        result = plugin._execute_waypoint({"action": "create", "target": '"second"'})
        assert result["id"] == "w2"  # IDs are never reused
        assert result["owner"] == "user"
