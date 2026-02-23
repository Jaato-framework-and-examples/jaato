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
from unittest import mock
from unittest.mock import Mock, MagicMock

from shared.plugins.sandbox_manager import (
    SandboxManagerPlugin,
    SandboxPath,
    SandboxConfig,
    create_plugin,
)
from jaato_sdk.plugins.base import ToolPlugin


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
            "path": "readwrite /new/allowed/path"
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
            "path": "readwrite /test/path"
        })
        assert result1["status"] == "added"

        # Add second time
        result2 = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /test/path"
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
            "path": "readwrite /test/path"
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
            "path": "readwrite /session/added/path"
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

    def test_sandbox_add_requires_access_and_path(self, initialized_plugin):
        """Test that sandbox add requires access mode and path argument."""
        plugin, workspace = initialized_plugin

        # No arguments at all
        result = plugin._execute_sandbox_command({
            "subcommand": "add"
        })
        assert "error" in result

        # Access mode without path
        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readonly"
        })
        assert "error" in result

        # Path without access mode
        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "/some/path"
        })
        assert "error" in result

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
            "path": "readwrite /some/path"
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
            "path": "readwrite /some/path"
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
        """Test that relative paths are resolved against workspace, not server CWD."""
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
            "path": "readwrite relative/path"
        })

        # Path should be resolved relative to workspace, not process CWD
        assert result["status"] == "added"
        assert result["path"] == str(workspace / "relative" / "path")

    def test_relative_parent_path_resolved_against_workspace(self, tmp_path):
        """Test that ../../whatever resolves against workspace, not server CWD."""
        workspace = tmp_path / "deep" / "nested" / "workspace"
        workspace.mkdir(parents=True)
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
            "path": "../../sibling"
        })

        assert result["status"] == "added"
        expected = str(tmp_path / "deep" / "sibling")
        assert result["path"] == expected

    def test_remove_relative_path_resolved_against_workspace(self, tmp_path):
        """Test that sandbox remove resolves relative paths against workspace."""
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
            "subcommand": "remove",
            "path": "relative/path"
        })

        assert result["status"] == "denied"
        assert result["path"] == str(workspace / "relative" / "path")


class TestProgrammaticAPI:
    """Tests for the programmatic add/remove API used by other plugins."""

    @pytest.fixture
    def initialized_plugin(self, tmp_path):
        """Create fully initialized plugin with mock registry."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test-session").mkdir(parents=True)

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.clear_authorized_paths = Mock()
        mock_registry.clear_denied_paths = Mock()
        mock_registry.authorize_external_path = Mock()
        mock_registry.deny_external_path = Mock()
        mock_registry.deauthorize_external_path = Mock()

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test-session"})
        plugin.set_workspace_path(str(workspace))

        return plugin, mock_registry, workspace

    def test_add_path_programmatic_readonly(self, initialized_plugin):
        """Test adding a readonly path programmatically."""
        plugin, mock_registry, workspace = initialized_plugin

        result = plugin.add_path_programmatic("/docs/reference.md", access="readonly")

        assert result is True

        # Verify persisted in session config
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        allowed = config["allowed_paths"]
        assert len(allowed) == 1
        assert allowed[0]["access"] == "readonly"
        assert "/docs/reference.md" in allowed[0]["path"]

    def test_add_path_programmatic_readwrite(self, initialized_plugin):
        """Test adding a readwrite path programmatically."""
        plugin, mock_registry, workspace = initialized_plugin

        result = plugin.add_path_programmatic("/tmp/scratch", access="readwrite")

        assert result is True

        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        assert config["allowed_paths"][0]["access"] == "readwrite"

    def test_add_path_programmatic_already_exists(self, initialized_plugin):
        """Test that adding the same path twice is a noop."""
        plugin, mock_registry, workspace = initialized_plugin

        plugin.add_path_programmatic("/docs/ref.md", access="readonly")
        result = plugin.add_path_programmatic("/docs/ref.md", access="readonly")

        assert result is True

        # Should still only have one entry
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        assert len(config["allowed_paths"]) == 1

    def test_add_path_programmatic_updates_access(self, initialized_plugin):
        """Test that adding with different access mode updates the entry."""
        plugin, mock_registry, workspace = initialized_plugin

        plugin.add_path_programmatic("/docs/ref.md", access="readonly")
        result = plugin.add_path_programmatic("/docs/ref.md", access="readwrite")

        assert result is True

        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        assert config["allowed_paths"][0]["access"] == "readwrite"

    def test_add_path_programmatic_invalid_access(self, initialized_plugin):
        """Test that invalid access mode returns False."""
        plugin, mock_registry, workspace = initialized_plugin

        result = plugin.add_path_programmatic("/docs/ref.md", access="invalid")

        assert result is False

    def test_add_path_programmatic_syncs_to_registry(self, initialized_plugin):
        """Test that adding a path syncs to the registry."""
        plugin, mock_registry, workspace = initialized_plugin

        plugin.add_path_programmatic("/docs/ref.md", access="readonly")

        # The _load_all_configs -> _sync_to_registry flow should call authorize_external_path
        mock_registry.authorize_external_path.assert_called()
        # Find the call with our path
        calls = [c for c in mock_registry.authorize_external_path.call_args_list
                 if "/docs/ref.md" in str(c)]
        assert len(calls) > 0

    def test_remove_path_programmatic(self, initialized_plugin):
        """Test removing a previously added path."""
        plugin, mock_registry, workspace = initialized_plugin

        plugin.add_path_programmatic("/docs/ref.md", access="readonly")
        result = plugin.remove_path_programmatic("/docs/ref.md")

        assert result is True

        # Verify removed from session config
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        assert len(config.get("allowed_paths", [])) == 0

    def test_remove_path_programmatic_not_found(self, initialized_plugin):
        """Test removing a path that was never added returns False."""
        plugin, mock_registry, workspace = initialized_plugin

        result = plugin.remove_path_programmatic("/nonexistent/path")

        assert result is False

    def test_add_then_remove_symmetric(self, initialized_plugin):
        """Test that add then remove is symmetric - no residual state."""
        plugin, mock_registry, workspace = initialized_plugin

        plugin.add_path_programmatic("/docs/ref.md", access="readonly")
        plugin.remove_path_programmatic("/docs/ref.md")

        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())

        # Should be empty - not in allowed or denied
        assert len(config.get("allowed_paths", [])) == 0
        assert len(config.get("denied_paths", [])) == 0

    def test_add_path_programmatic_fallback_without_workspace(self):
        """Test fallback to direct registry auth when no workspace."""
        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.authorize_external_path = Mock()

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test"})
        # Don't set workspace

        result = plugin.add_path_programmatic("/docs/ref.md", access="readonly")

        assert result is True
        mock_registry.authorize_external_path.assert_called_once_with(
            "/docs/ref.md", "sandbox_manager", access="readonly"
        )

    def test_remove_path_programmatic_fallback_without_workspace(self):
        """Test fallback to direct registry deauth when no workspace."""
        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.deauthorize_external_path = Mock()

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test"})
        # Don't set workspace

        result = plugin.remove_path_programmatic("/docs/ref.md")

        assert result is True
        mock_registry.deauthorize_external_path.assert_called_once_with(
            "/docs/ref.md", "sandbox_manager"
        )

    def test_pending_paths_survive_workspace_set(self, tmp_path):
        """Test that paths added before workspace survive set_workspace_path.

        Reproduces the timing bug where:
        1. Another plugin calls add_path_programmatic() during initialization
        2. sandbox_manager has no workspace yet -> fallback to registry auth
        3. set_workspace_path() triggers _load_all_configs() -> _sync_to_registry()
        4. _sync_to_registry() clears all in-memory registry paths
        5. The originally added paths are now lost

        The fix queues these paths and replays them after workspace is set.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test-session").mkdir(parents=True)

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.clear_authorized_paths = Mock()
        mock_registry.clear_denied_paths = Mock()
        mock_registry.authorize_external_path = Mock()
        mock_registry.get_plugin = Mock(return_value=None)

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test-session"})
        # No workspace yet — simulates real initialization order

        # Another plugin (e.g. references) adds a readonly path
        result = plugin.add_path_programmatic("/docs/api-spec.md", access="readonly")
        assert result is True

        # Path should be queued
        assert len(plugin._pending_programmatic_paths) == 1

        # Now workspace becomes available (registry.set_workspace_path() is called)
        mock_registry.reset_mock()
        plugin.set_workspace_path(str(workspace))

        # Pending list should be cleared
        assert len(plugin._pending_programmatic_paths) == 0

        # Path should be persisted in session config
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        allowed = config["allowed_paths"]
        assert len(allowed) == 1
        assert allowed[0]["access"] == "readonly"
        assert "/docs/api-spec.md" in allowed[0]["path"]

        # Path should also be registered in the registry with readonly access
        auth_calls = [
            call for call in mock_registry.authorize_external_path.call_args_list
            if "/docs/api-spec.md" in str(call)
        ]
        assert len(auth_calls) >= 1
        # Last call should have access="readonly"
        last_call = auth_calls[-1]
        assert last_call[1].get("access", last_call[0][2] if len(last_call[0]) > 2 else None) == "readonly" or \
               "readonly" in str(last_call)

    def test_pending_paths_no_duplicates_on_reload(self, tmp_path):
        """Test that pending paths already in config are not duplicated."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        session_dir = workspace / ".jaato" / "sessions" / "test-session"
        session_dir.mkdir(parents=True)

        # Pre-populate session config with the same path
        config_path = session_dir / "sandbox.json"
        config_path.write_text(json.dumps({
            "allowed_paths": [{
                "path": os.path.normpath(os.path.abspath("/docs/ref.md")),
                "access": "readonly",
                "added_at": "2024-01-01T00:00:00Z",
            }],
            "denied_paths": []
        }))

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.clear_authorized_paths = Mock()
        mock_registry.clear_denied_paths = Mock()
        mock_registry.authorize_external_path = Mock()
        mock_registry.get_plugin = Mock(return_value=None)

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test-session"})
        # No workspace yet

        # Add path that already exists in session config (we don't know yet)
        plugin.add_path_programmatic("/docs/ref.md", access="readonly")

        # Set workspace — triggers replay
        plugin.set_workspace_path(str(workspace))

        # Should not duplicate the path
        config = json.loads(config_path.read_text())
        assert len(config["allowed_paths"]) == 1

    def test_multiple_pending_paths_all_persisted(self, tmp_path):
        """Test that multiple pending paths are all persisted on workspace set."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test-session").mkdir(parents=True)

        plugin = create_plugin()
        mock_registry = Mock()
        mock_registry.clear_authorized_paths = Mock()
        mock_registry.clear_denied_paths = Mock()
        mock_registry.authorize_external_path = Mock()
        mock_registry.get_plugin = Mock(return_value=None)

        plugin.set_plugin_registry(mock_registry)
        plugin.initialize({"session_id": "test-session"})

        # Add multiple paths before workspace
        plugin.add_path_programmatic("/docs/api.md", access="readonly")
        plugin.add_path_programmatic("/docs/guide.md", access="readonly")
        plugin.add_path_programmatic("/tmp/scratch", access="readwrite")

        assert len(plugin._pending_programmatic_paths) == 3

        # Set workspace
        plugin.set_workspace_path(str(workspace))

        # All should be persisted
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        allowed = config["allowed_paths"]
        assert len(allowed) == 3

        paths_with_access = {(e["path"], e["access"]) for e in allowed}
        norm = lambda p: os.path.normpath(os.path.abspath(p))
        assert (norm("/docs/api.md"), "readonly") in paths_with_access
        assert (norm("/docs/guide.md"), "readonly") in paths_with_access
        assert (norm("/tmp/scratch"), "readwrite") in paths_with_access

        # Pending list should be cleared
        assert len(plugin._pending_programmatic_paths) == 0


class TestMSYS2PathConversion:
    """Tests for MSYS2 path conversion in sandbox commands.

    Verifies that MSYS2-style paths (/c/Users/...) are correctly converted
    to Windows paths (C:/Users/...) at the input boundary, and that display
    output is converted back to MSYS2 format when running under MSYS2.

    Since tests run on Linux where C:/... is not recognized as absolute by
    os.path.isabs(), tests that involve full path processing mock isabs
    to simulate Windows behavior. Tests that only verify conversion happened
    check for the C: drive letter in the stored path.
    """

    @pytest.fixture
    def initialized_plugin(self, tmp_path):
        """Create fully initialized plugin with mock registry."""
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

    @staticmethod
    def _win_isabs(path):
        """Simulate Windows os.path.isabs: C:/ and C:\\ are absolute."""
        import re
        if re.match(r'^[a-zA-Z]:[\\/]', path):
            return True
        return os.path.isabs(path)

    def test_cmd_add_converts_msys2_drive_letter(self, initialized_plugin):
        """Test that _cmd_add converts /c/... drive prefix to C:/... for storage.

        On Linux, C:/... is not absolute so it gets joined with workspace.
        We verify the conversion happened by checking the stored path contains
        'C:/' (the converted drive letter) and not '/c/' (the raw MSYS2 prefix).
        """
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /c/Users/testuser/external"
        })

        assert result["status"] == "added"

        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        stored_path = config["allowed_paths"][0]["path"]
        # The MSYS2 conversion should have turned /c/ into C:/
        assert "C:/" in stored_path or "C:\\" in stored_path
        assert "Users/testuser/external" in stored_path.replace("\\", "/")

    @mock.patch("shared.plugins.sandbox_manager.plugin.os.path.isabs")
    def test_cmd_add_msys2_path_absolute_on_windows(self, mock_isabs, initialized_plugin):
        """Test full path processing with Windows-like isabs behavior.

        Simulates Windows where C:/Users/... is recognized as absolute.
        """
        plugin, workspace = initialized_plugin
        mock_isabs.side_effect = self._win_isabs

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /c/Users/testuser/external"
        })

        assert result["status"] == "added"

        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        stored_path = config["allowed_paths"][0]["path"]
        # Should be C:/Users/... (absolute, not joined with workspace)
        assert stored_path.startswith("C:/")
        assert "Users/testuser/external" in stored_path

    def test_cmd_add_msys2_path_not_stored_as_backslash_c(self, initialized_plugin):
        """Test that /c/Users/... is NOT stored as \\c\\Users\\... (the bug).

        The original bug was that /c/Users/... was stored as \\c\\Users\\...
        because the MSYS2 conversion was missing. After the fix, the path
        should contain C: (converted drive letter).
        """
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /c/Users/testuser/data"
        })

        assert result["status"] == "added"

        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        stored_path = config["allowed_paths"][0]["path"]
        norm = stored_path.replace("\\", "/")
        # Must NOT have the raw /c/ prefix (the unconverted MSYS2 path)
        # It should be C:/Users/... (possibly joined with workspace on Linux)
        assert "/c/Users" not in norm

    def test_cmd_add_already_allowed_with_msys2_path(self, initialized_plugin):
        """Test that duplicate detection works with MSYS2 paths."""
        plugin, workspace = initialized_plugin

        # Add first time
        result1 = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /c/Users/testuser/data"
        })
        assert result1["status"] == "added"

        # Add again with same MSYS2 path
        result2 = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /c/Users/testuser/data"
        })
        assert result2["status"] == "already_allowed"

    @mock.patch("shared.plugins.sandbox_manager.plugin.os.path.isabs")
    def test_cmd_remove_converts_msys2_path(self, mock_isabs, initialized_plugin):
        """Test that _cmd_remove converts /c/Users/... to C:/Users/... for storage."""
        plugin, workspace = initialized_plugin
        mock_isabs.side_effect = self._win_isabs

        result = plugin._execute_sandbox_command({
            "subcommand": "remove",
            "path": "/c/Users/testuser/blocked"
        })

        assert result["status"] == "denied"

        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        stored_path = config["denied_paths"][0]["path"]
        assert stored_path.startswith("C:/")
        assert "Users/testuser/blocked" in stored_path

    def test_cmd_remove_symmetric_with_msys2_add(self, initialized_plugin):
        """Test that remove works symmetrically with add when using MSYS2 paths."""
        plugin, workspace = initialized_plugin

        # Add with MSYS2 path
        plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /c/Users/testuser/project"
        })

        # Remove with same MSYS2 path - should find and remove
        result = plugin._execute_sandbox_command({
            "subcommand": "remove",
            "path": "/c/Users/testuser/project"
        })
        assert result["status"] == "removed"  # Not "denied" - symmetric undo

    @mock.patch("shared.plugins.sandbox_manager.plugin.os.path.isabs")
    @mock.patch(
        "shared.plugins.sandbox_manager.plugin.normalize_result_path",
        side_effect=lambda p: p.replace("\\", "/").replace("C:/", "/c/")
        if (p.startswith("C:/") or p.startswith("C:\\")) else p,
    )
    def test_cmd_add_returns_msys2_display_path(self, _mock_normalize, mock_isabs, initialized_plugin):
        """Test that return value uses MSYS2-friendly display path."""
        plugin, workspace = initialized_plugin
        mock_isabs.side_effect = self._win_isabs

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /c/Users/testuser/data"
        })

        assert result["status"] == "added"
        # Display path should be in MSYS2 format
        assert result["path"].startswith("/c/")

    @mock.patch("shared.plugins.sandbox_manager.plugin.os.path.isabs")
    @mock.patch(
        "shared.plugins.sandbox_manager.plugin.normalize_result_path",
        side_effect=lambda p: p.replace("\\", "/").replace("C:/", "/c/")
        if (p.startswith("C:/") or p.startswith("C:\\")) else p,
    )
    def test_cmd_remove_returns_msys2_display_path(self, _mock_normalize, mock_isabs, initialized_plugin):
        """Test that remove return value uses MSYS2-friendly display path."""
        plugin, workspace = initialized_plugin
        mock_isabs.side_effect = self._win_isabs

        result = plugin._execute_sandbox_command({
            "subcommand": "remove",
            "path": "/c/Users/testuser/blocked"
        })

        assert result["status"] == "denied"
        assert result["path"].startswith("/c/")

    def test_load_config_converts_msys2_paths(self, tmp_path):
        """Test that config files with MSYS2 paths are converted on load."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test-session").mkdir(parents=True)

        # Write config with MSYS2-format paths (simulating hand-edited config)
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config_path.write_text(json.dumps({
            "allowed_paths": [
                {"path": "/c/Users/testuser/docs", "access": "readonly"},
                "/d/shared/data",
            ],
            "denied_paths": [
                {"path": "/c/Windows/System32"},
            ]
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

        # Verify paths are converted from MSYS2 format
        allowed_paths = [p.path for p in plugin._config.allowed_paths if p.source == "session"]
        denied_paths = [p.path for p in plugin._config.denied_paths if p.source == "session"]

        # Converted paths should contain C: or D: drive letter
        assert any("C:" in p for p in allowed_paths), \
            f"Expected C: in allowed paths, got {allowed_paths}"
        assert any("D:" in p for p in allowed_paths), \
            f"Expected D: in allowed paths, got {allowed_paths}"
        assert any("C:" in p for p in denied_paths), \
            f"Expected C: in denied paths, got {denied_paths}"

    def test_cmd_add_non_msys2_path_unchanged(self, initialized_plugin):
        """Test that regular absolute paths are not modified by MSYS2 conversion."""
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /tmp/regular/path"
        })

        assert result["status"] == "added"

        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        stored_path = config["allowed_paths"][0]["path"]
        # /tmp/regular/path should NOT be treated as an MSYS2 drive path
        # (/t is not a valid single-letter drive prefix pattern)
        assert "/tmp/" in stored_path or "\\tmp\\" in stored_path

    @mock.patch("shared.plugins.sandbox_manager.plugin.os.path.isabs")
    def test_cmd_add_strips_single_quotes_from_path(self, mock_isabs, initialized_plugin):
        """Test that surrounding single quotes are stripped from paths.

        Models often wrap Windows paths in shell-style quotes like
        'C:\\Users\\...' which arrive as literal characters via capture_rest.
        Without stripping, the quote prefix prevents os.path.isabs from
        recognizing the path as absolute, causing it to be joined with workspace.
        """
        plugin, workspace = initialized_plugin
        mock_isabs.side_effect = self._win_isabs

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readonly 'C:/Users/testuser/AppData/logs'"
        })

        assert result["status"] == "added"
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        stored_path = config["allowed_paths"][0]["path"]
        # Path should be absolute C:/... not joined with workspace
        assert stored_path.startswith("C:/")
        assert "testuser/AppData/logs" in stored_path

    @mock.patch("shared.plugins.sandbox_manager.plugin.os.path.isabs")
    def test_cmd_add_strips_double_quotes_from_path(self, mock_isabs, initialized_plugin):
        """Test that surrounding double quotes are stripped from paths."""
        plugin, workspace = initialized_plugin
        mock_isabs.side_effect = self._win_isabs

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": 'readwrite "C:/Users/testuser/data"'
        })

        assert result["status"] == "added"
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        stored_path = config["allowed_paths"][0]["path"]
        assert stored_path.startswith("C:/")
        assert "testuser/data" in stored_path

    @mock.patch("shared.plugins.sandbox_manager.plugin.os.path.isabs")
    def test_cmd_remove_strips_quotes_from_path(self, mock_isabs, initialized_plugin):
        """Test that surrounding quotes are stripped from remove paths too."""
        plugin, workspace = initialized_plugin
        mock_isabs.side_effect = self._win_isabs

        # Add a path first
        plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite C:/Users/testuser/project"
        })

        # Remove with quoted path — should match the unquoted stored path
        result = plugin._execute_sandbox_command({
            "subcommand": "remove",
            "path": "'C:/Users/testuser/project'"
        })
        assert result["status"] == "removed"

    def test_cmd_add_multichar_prefix_not_converted(self, initialized_plugin):
        """Test that /config/... is not misidentified as MSYS2 drive path."""
        plugin, workspace = initialized_plugin

        result = plugin._execute_sandbox_command({
            "subcommand": "add",
            "path": "readwrite /config/settings"
        })

        assert result["status"] == "added"

        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config = json.loads(config_path.read_text())
        stored_path = config["allowed_paths"][0]["path"]
        # /config should NOT be converted - it's a multi-letter directory, not a drive
        assert "/config/" in stored_path or "\\config\\" in stored_path

    @mock.patch("shared.plugins.sandbox_manager.plugin.os.path.isabs")
    def test_cmd_list_displays_msys2_paths(self, mock_isabs, tmp_path):
        """Test that _cmd_list output uses normalize_result_path for display."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".jaato" / "sessions" / "test-session").mkdir(parents=True)

        mock_isabs.side_effect = self._win_isabs

        # Write config with a Windows-format path (as stored after conversion)
        config_path = workspace / ".jaato" / "sessions" / "test-session" / "sandbox.json"
        config_path.write_text(json.dumps({
            "allowed_paths": [
                {"path": "C:/Users/testuser/data", "access": "readwrite"},
            ]
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

        with mock.patch(
            "shared.plugins.sandbox_manager.plugin.normalize_result_path",
            side_effect=lambda p: "/c" + p[2:].replace("\\", "/")
            if (p.startswith("C:/") or p.startswith("C:\\")) else p,
        ):
            result = plugin._cmd_list()

        # The displayed path should be MSYS2-formatted
        assert len(result["effective_paths"]) == 1
        assert result["effective_paths"][0]["path"].startswith("/c/")
