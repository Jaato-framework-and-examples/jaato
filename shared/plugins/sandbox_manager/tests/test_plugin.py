"""Tests for the SandboxManagerPlugin.

These tests verify:
- Plugin initialization and ToolPlugin protocol compliance
- Config loading from three tiers (global, workspace, session)
- User commands (sandbox list, add, remove)
- Registry integration (sync to registry deny/authorize)
- Precedence handling (session > workspace > global)
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from shared.plugins.sandbox_manager import (
    SandboxManagerPlugin,
    SandboxPath,
    SandboxConfig,
    create_plugin,
)
from shared.plugins.base import ToolPlugin


class TestPluginProtocol:
    """Tests for ToolPlugin protocol compliance."""

    def test_create_plugin_returns_instance(self):
        """Test that create_plugin() returns a SandboxManagerPlugin."""
        plugin = create_plugin()
        assert isinstance(plugin, SandboxManagerPlugin)

    def test_name_property(self):
        """Test that name property returns correct value."""
        plugin = create_plugin()
        assert plugin.name == "sandbox_manager"

    def test_get_tool_schemas_returns_empty(self):
        """Test that plugin has no model tools (user commands only)."""
        plugin = create_plugin()
        schemas = plugin.get_tool_schemas()
        assert schemas == []

    def test_get_executors_returns_sandbox_command(self):
        """Test that executors include the sandbox command."""
        plugin = create_plugin()
        executors = plugin.get_executors()
        assert "sandbox" in executors
        assert callable(executors["sandbox"])

    def test_get_system_instructions_returns_none(self):
        """Test that no system instructions are provided."""
        plugin = create_plugin()
        assert plugin.get_system_instructions() is None

    def test_get_auto_approved_tools(self):
        """Test that sandbox command is auto-approved."""
        plugin = create_plugin()
        auto_approved = plugin.get_auto_approved_tools()
        assert "sandbox" in auto_approved

    def test_get_user_commands(self):
        """Test that user commands are properly defined."""
        plugin = create_plugin()
        commands = plugin.get_user_commands()
        assert len(commands) == 1
        assert commands[0].name == "sandbox"
        assert commands[0].share_with_model is False


class TestInitialization:
    """Tests for plugin initialization."""

    def test_initialize_sets_session_id(self):
        """Test that initialize sets session_id from config."""
        plugin = create_plugin()
        plugin.initialize({"session_id": "test-session-123"})
        assert plugin._session_id == "test-session-123"
        assert plugin._initialized is True

    def test_initialize_without_config(self):
        """Test that initialize works without config."""
        plugin = create_plugin()
        plugin.initialize()
        assert plugin._session_id is None
        assert plugin._initialized is True

    def test_shutdown_clears_state(self):
        """Test that shutdown clears plugin state."""
        plugin = create_plugin()
        plugin.initialize({"session_id": "test"})
        plugin._config = SandboxConfig()

        plugin.shutdown()

        assert plugin._config is None
        assert plugin._initialized is False

    def test_set_workspace_path(self, tmp_path):
        """Test that set_workspace_path stores the path."""
        plugin = create_plugin()
        plugin.initialize()

        workspace = str(tmp_path / "workspace")
        plugin.set_workspace_path(workspace)

        assert plugin._workspace_path == workspace

    def test_set_plugin_registry(self):
        """Test that set_plugin_registry stores the registry."""
        plugin = create_plugin()
        mock_registry = Mock()

        plugin.set_plugin_registry(mock_registry)

        assert plugin._registry is mock_registry


class TestConfigLoading:
    """Tests for configuration loading from three tiers."""

    @pytest.fixture
    def setup_workspace(self, tmp_path):
        """Create a workspace with config directories."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato").mkdir()
        (workspace / ".jaato" / "sessions" / "test-session").mkdir(parents=True)
        return workspace

    def test_load_global_config(self, tmp_path, monkeypatch):
        """Test loading global config from ~/.jaato/sandbox_paths.json."""
        # Create fake home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        # Create global config
        jaato_dir = fake_home / ".jaato"
        jaato_dir.mkdir()
        global_config = jaato_dir / "sandbox_paths.json"
        global_config.write_text(json.dumps({
            "allowed_paths": ["/opt/tools", {"path": "/data/shared", "added_at": "2024-01-01T00:00:00Z"}],
            "denied_paths": ["/etc/passwd"]
        }))

        plugin = create_plugin()
        plugin.initialize()
        plugin._load_all_configs()

        assert plugin._config is not None
        assert len(plugin._config.allowed_paths) == 2
        assert len(plugin._config.denied_paths) == 1
        assert plugin._config.allowed_paths[0].path == "/opt/tools"
        assert plugin._config.allowed_paths[0].source == "global"
        assert plugin._config.denied_paths[0].path == "/etc/passwd"

    def test_load_workspace_config(self, setup_workspace):
        """Test loading workspace config from .jaato/sandbox.json."""
        workspace = setup_workspace

        # Create workspace config
        workspace_config = workspace / ".jaato" / "sandbox.json"
        workspace_config.write_text(json.dumps({
            "allowed_paths": ["/project/external-deps"],
            "denied_paths": []
        }))

        plugin = create_plugin()
        plugin.initialize()
        plugin.set_workspace_path(str(workspace))

        assert plugin._config is not None
        # Should have workspace config loaded
        workspace_paths = [p for p in plugin._config.allowed_paths if p.source == "workspace"]
        assert len(workspace_paths) == 1
        assert workspace_paths[0].path == "/project/external-deps"

    def test_load_session_config(self, setup_workspace):
        """Test loading session config from .jaato/sessions/<id>/sandbox.json."""
        workspace = setup_workspace

        # Create session config
        session_config = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        session_config.write_text(json.dumps({
            "allowed_paths": ["/tmp/session-data"],
            "denied_paths": ["/blocked/path"]
        }))

        plugin = create_plugin()
        plugin.initialize({"session_id": "test-session"})
        plugin.set_workspace_path(str(workspace))

        assert plugin._config is not None
        session_allowed = [p for p in plugin._config.allowed_paths if p.source == "session"]
        session_denied = [p for p in plugin._config.denied_paths if p.source == "session"]
        assert len(session_allowed) == 1
        assert len(session_denied) == 1
        assert session_allowed[0].path == "/tmp/session-data"
        assert session_denied[0].path == "/blocked/path"

    def test_config_merge_all_tiers(self, setup_workspace, monkeypatch):
        """Test that configs from all three tiers are merged."""
        workspace = setup_workspace

        # Create fake home
        fake_home = workspace.parent / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        # Global config
        global_jaato = fake_home / ".jaato"
        global_jaato.mkdir()
        (global_jaato / "sandbox_paths.json").write_text(json.dumps({
            "allowed_paths": ["/global/path"]
        }))

        # Workspace config
        (workspace / ".jaato" / "sandbox.json").write_text(json.dumps({
            "allowed_paths": ["/workspace/path"]
        }))

        # Session config
        (workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json").write_text(json.dumps({
            "allowed_paths": ["/session/path"]
        }))

        plugin = create_plugin()
        plugin.initialize({"session_id": "test-session"})
        plugin.set_workspace_path(str(workspace))

        # All three should be present
        assert len(plugin._config.allowed_paths) == 3
        sources = {p.source for p in plugin._config.allowed_paths}
        assert sources == {"global", "workspace", "session"}


class TestRegistryIntegration:
    """Tests for registry authorization/denial integration."""

    @pytest.fixture
    def plugin_with_registry(self, tmp_path):
        """Create plugin with mock registry."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test-session").mkdir(parents=True)

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.clear_authorized_paths = Mock()
        mock_registry.clear_denied_paths = Mock()
        mock_registry.authorize_external_path = Mock()
        mock_registry.deny_external_path = Mock()

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test-session"})
        plugin.set_workspace_path(str(workspace))

        return plugin, mock_registry, workspace

    def test_sync_authorized_paths_to_registry(self, plugin_with_registry):
        """Test that allowed paths are synced to registry."""
        plugin, mock_registry, workspace = plugin_with_registry

        # Create config with allowed path
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config_path.write_text(json.dumps({
            "allowed_paths": ["/external/allowed"]
        }))

        plugin._load_all_configs()

        mock_registry.authorize_external_path.assert_called()
        call_args = [call[0] for call in mock_registry.authorize_external_path.call_args_list]
        assert any("/external/allowed" in str(args) for args in call_args)

    def test_sync_denied_paths_to_registry(self, plugin_with_registry):
        """Test that denied paths are synced to registry."""
        plugin, mock_registry, workspace = plugin_with_registry

        # Create config with denied path
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config_path.write_text(json.dumps({
            "denied_paths": ["/blocked/path"]
        }))

        plugin._load_all_configs()

        mock_registry.deny_external_path.assert_called()
        call_args = [call[0] for call in mock_registry.deny_external_path.call_args_list]
        assert any("/blocked/path" in str(args) for args in call_args)

    def test_shutdown_clears_registry_state(self, plugin_with_registry):
        """Test that shutdown clears registry authorizations."""
        plugin, mock_registry, workspace = plugin_with_registry

        plugin.shutdown()

        mock_registry.clear_authorized_paths.assert_called_with("sandbox_manager")
        mock_registry.clear_denied_paths.assert_called_with("sandbox_manager")


class TestUserCommands:
    """Tests for user command execution."""

    @pytest.fixture
    def initialized_plugin(self, tmp_path):
        """Create fully initialized plugin."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test-session").mkdir(parents=True)

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.clear_authorized_paths = Mock()
        mock_registry.clear_denied_paths = Mock()
        mock_registry.authorize_external_path = Mock()
        mock_registry.deny_external_path = Mock()

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test-session"})
        plugin.set_workspace_path(str(workspace))

        return plugin, workspace

    def test_sandbox_list_empty(self, initialized_plugin):
        """Test sandbox list with no config."""
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({"subcommand": "list"})

        assert "effective_paths" in result
        assert "summary" in result
        assert result["summary"]["total"] == 0

    def test_sandbox_list_with_paths(self, initialized_plugin):
        """Test sandbox list shows configured paths."""
        plugin, workspace = initialized_plugin

        # Create config
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config_path.write_text(json.dumps({
            "allowed_paths": ["/allowed/path"],
            "denied_paths": ["/denied/path"]
        }))
        plugin._load_all_configs()

        result = plugin._execute_sandbox_command({"subcommand": "list"})

        assert result["summary"]["total"] == 2
        assert result["summary"]["allowed"] == 1
        assert result["summary"]["denied"] == 1

    def test_sandbox_add_creates_session_entry(self, initialized_plugin):
        """Test sandbox add creates session config entry."""
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "/new/allowed/path"
        })

        assert result["status"] == "added"
        assert result["source"] == "session"

        # Verify file was created
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        assert config_path.exists()

        config = json.loads(config_path.read_text())
        paths = [p["path"] if isinstance(p, dict) else p for p in config["allowed_paths"]]
        # Path should be absolute
        assert any("/new/allowed/path" in p for p in paths)

    def test_sandbox_add_already_allowed(self, initialized_plugin):
        """Test sandbox add returns already_allowed for existing path."""
        plugin, workspace = initialized_plugin

        # Add first time
        result1 = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "/test/path"
        })
        assert result1["status"] == "added"

        # Add second time
        result2 = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "/test/path"
        })
        assert result2["status"] == "already_allowed"

    def test_sandbox_remove_creates_deny_entry(self, initialized_plugin):
        """Test sandbox remove creates session deny entry."""
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({
            "subcommand": "remove",
            "path": "/blocked/path"
        })

        assert result["status"] == "denied"
        assert result["source"] == "session"

        # Verify file was created
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        paths = [p["path"] if isinstance(p, dict) else p for p in config["denied_paths"]]
        assert any("/blocked/path" in p for p in paths)

    def test_sandbox_remove_already_denied(self, initialized_plugin):
        """Test sandbox remove returns already_denied for existing path."""
        plugin, workspace = initialized_plugin

        # Remove first time
        result1 = plugin._execute_sandbox_command({
            "subcommand": "remove",
            "path": "/test/path"
        })
        assert result1["status"] == "denied"

        # Remove second time
        result2 = plugin._execute_sandbox_command({
            "subcommand": "remove",
            "path": "/test/path"
        })
        assert result2["status"] == "already_denied"

    def test_sandbox_add_removes_from_denied(self, initialized_plugin):
        """Test that sandbox add removes path from denied list."""
        plugin, workspace = initialized_plugin

        # First deny
        plugin._execute_sandbox_command({
            "subcommand": "remove",
            "path": "/test/path"
        })

        # Then allow (should remove from denied)
        plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "/test/path"
        })

        # Check config
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())

        # Should be in allowed, not in denied
        allowed_paths = [p["path"] if isinstance(p, dict) else p for p in config["allowed_paths"]]
        denied_paths = [p["path"] if isinstance(p, dict) else p for p in config["denied_paths"]]

        assert any("/test/path" in p for p in allowed_paths)
        assert not any("/test/path" in p for p in denied_paths)

    def test_sandbox_remove_symmetric_with_add(self, initialized_plugin):
        """Test that sandbox remove is symmetric with add for session paths.

        If a path was added to session's allowed_paths, removing it should
        just remove from allowed_paths (status: "removed"), not add to
        denied_paths (status: "denied").
        """
        plugin, workspace = initialized_plugin

        # First add a path
        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "/session/added/path"
        })
        assert result["status"] == "added"

        # Verify it's in allowed_paths
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        allowed_paths = [p["path"] if isinstance(p, dict) else p for p in config["allowed_paths"]]
        assert any("/session/added/path" in p for p in allowed_paths)

        # Now remove it - should be symmetric (just remove, not deny)
        result = plugin._execute_sandbox_command({
            "subcommand": "remove",
            "path": "/session/added/path"
        })
        assert result["status"] == "removed"  # Not "denied"
        assert result["source"] == "session"

        # Verify it's NOT in allowed_paths anymore
        config = json.loads(config_path.read_text())
        allowed_paths = [p["path"] if isinstance(p, dict) else p for p in config.get("allowed_paths", [])]
        denied_paths = [p["path"] if isinstance(p, dict) else p for p in config.get("denied_paths", [])]

        assert not any("/session/added/path" in p for p in allowed_paths)
        # And also NOT in denied_paths (symmetric undo)
        assert not any("/session/added/path" in p for p in denied_paths)

    def test_sandbox_unknown_subcommand(self, initialized_plugin):
        """Test that unknown subcommand returns error."""
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({
            "subcommand": "unknown"
        })

        assert "error" in result
        assert "Unknown subcommand" in result["error"]

    def test_sandbox_add_requires_path(self, initialized_plugin):
        """Test that sandbox add requires path argument."""
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({
            "subcommand": "add"
        })

        assert "error" in result
        assert "Path is required" in result["error"]

    def test_sandbox_remove_requires_path(self, initialized_plugin):
        """Test that sandbox remove requires path argument."""
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({
            "subcommand": "remove"
        })

        assert "error" in result
        assert "Path is required" in result["error"]


class TestPrecedence:
    """Tests for precedence handling (session > workspace > global)."""

    @pytest.fixture
    def multi_tier_setup(self, tmp_path, monkeypatch):
        """Create config at all three tiers."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test-session").mkdir(parents=True)

        # Fake home
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        (fake_home / ".jaato").mkdir()

        return workspace, fake_home

    def test_session_deny_overrides_workspace_allow(self, multi_tier_setup):
        """Test that session deny overrides workspace allow."""
        workspace, fake_home = multi_tier_setup

        # Workspace allows /some/path
        (workspace / ".jaato" / "sandbox.json").write_text(json.dumps({
            "allowed_paths": ["/some/path"]
        }))

        # Session denies /some/path
        (workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json").write_text(json.dumps({
            "denied_paths": ["/some/path"]
        }))

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.clear_authorized_paths = Mock()
        mock_registry.clear_denied_paths = Mock()
        mock_registry.authorize_external_path = Mock()
        mock_registry.deny_external_path = Mock()

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test-session"})
        plugin.set_workspace_path(str(workspace))

        # Path should be denied, not authorized
        deny_calls = [call[0][0] for call in mock_registry.deny_external_path.call_args_list]
        assert "/some/path" in deny_calls

        # The path should NOT be authorized since it's denied at higher precedence
        # (The _is_path_denied_by_higher_precedence check should prevent this)

    def test_session_allow_overrides_global_deny(self, multi_tier_setup):
        """Test that session allow overrides global deny."""
        workspace, fake_home = multi_tier_setup

        # Global denies /some/path
        (fake_home / ".jaato" / "sandbox_paths.json").write_text(json.dumps({
            "denied_paths": ["/some/path"]
        }))

        # Session allows /some/path
        (workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json").write_text(json.dumps({
            "allowed_paths": ["/some/path"]
        }))

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.clear_authorized_paths = Mock()
        mock_registry.clear_denied_paths = Mock()
        mock_registry.authorize_external_path = Mock()
        mock_registry.deny_external_path = Mock()

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test-session"})
        plugin.set_workspace_path(str(workspace))

        # Both global deny and session allow should be registered
        # The registry's is_path_denied check happens at validation time
        authorize_calls = [call[0][0] for call in mock_registry.authorize_external_path.call_args_list]
        assert "/some/path" in authorize_calls


class TestCommandCompletions:
    """Tests for command completion suggestions."""

    def test_completions_for_empty_args(self):
        """Test completions when no args provided."""
        plugin = create_plugin()
        completions = plugin.get_command_completions("sandbox", [])

        values = [c.value for c in completions]
        assert "list" in values
        assert "add" in values
        assert "remove" in values

    def test_completions_for_partial_subcommand(self):
        """Test completions filter by partial input."""
        plugin = create_plugin()

        # "li" should match "list"
        completions = plugin.get_command_completions("sandbox", ["li"])
        values = [c.value for c in completions]
        assert "list" in values
        assert "add" not in values

        # "a" should match "add"
        completions = plugin.get_command_completions("sandbox", ["a"])
        values = [c.value for c in completions]
        assert "add" in values
        assert "remove" not in values

    def test_completions_for_other_command(self):
        """Test that completions return empty for other commands."""
        plugin = create_plugin()
        completions = plugin.get_command_completions("other", [])
        assert completions == []


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_add_without_workspace(self):
        """Test sandbox add fails gracefully without workspace."""
        plugin = create_plugin()
        plugin.initialize({"session_id": "test"})
        # Don't set workspace

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "/some/path"
        })

        assert "error" in result
        assert "workspace" in result["error"].lower()

    def test_add_without_session_id(self, tmp_path):
        """Test sandbox add fails gracefully without session_id."""
        plugin = create_plugin()
        plugin.initialize()  # No session_id
        plugin.set_workspace_path(str(tmp_path))

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "/some/path"
        })

        assert "error" in result
        assert "session" in result["error"].lower()

    def test_load_invalid_json_config(self, tmp_path):
        """Test that invalid JSON config is handled gracefully."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test").mkdir(parents=True)

        # Create invalid JSON
        config_path = workspace / ".jaato" / "sessions" / "test" / "sandbox.json"
        config_path.write_text("{ invalid json }")

        plugin = create_plugin()
        plugin.initialize({"session_id": "test"})
        plugin.set_workspace_path(str(workspace))

        # Should not crash, just have empty config
        assert plugin._config is not None
        session_paths = [p for p in plugin._config.allowed_paths if p.source == "session"]
        assert len(session_paths) == 0

    def test_relative_path_normalized(self, tmp_path):
        """Test that relative paths are converted to absolute."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test").mkdir(parents=True)

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.clear_authorized_paths = Mock()
        mock_registry.clear_denied_paths = Mock()
        mock_registry.authorize_external_path = Mock()
        mock_registry.deny_external_path = Mock()

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test"})
        plugin.set_workspace_path(str(workspace))

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "relative/path"
        })

        # Path should be normalized to absolute
        assert result["status"] == "added"
        assert os.path.isabs(result["path"])
