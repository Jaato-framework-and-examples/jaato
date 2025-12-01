"""Tests for permission plugin integration with the plugin registry."""

import json
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest

# Import the registry and plugin
from ...registry import PluginRegistry
from ..plugin import PermissionPlugin, create_plugin


class TestRegistryPluginDiscovery:
    """Tests for discovering the permission plugin via the registry."""

    def test_permission_plugin_discovered(self):
        """Test that permission plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "permission" in discovered

    def test_permission_plugin_available(self):
        """Test that permission plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "permission" in registry.list_available()

    def test_get_permission_plugin(self):
        """Test retrieving the permission plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("permission")
        assert plugin is not None
        assert plugin.name == "permission"


class TestRegistryExposePermissionPlugin:
    """Tests for exposing the permission plugin via the registry."""

    def test_expose_permission_plugin(self):
        """Test exposing the permission plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission")

        assert registry.is_exposed("permission")
        assert "permission" in registry.list_exposed()

    def test_expose_permission_plugin_with_config(self):
        """Test exposing the permission plugin with inline policy config."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission", config={
            "policy": {
                "defaultPolicy": "allow",
                "blacklist": {"tools": ["blocked_tool"]}
            }
        })

        assert registry.is_exposed("permission")

    def test_unexpose_permission_plugin(self):
        """Test unexposing the permission plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission")
        assert registry.is_exposed("permission")

        registry.unexpose_tool("permission")
        assert not registry.is_exposed("permission")

    def test_expose_all_includes_permission(self):
        """Test that expose_all includes the permission plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("permission")

    def test_unexpose_all_includes_permission(self):
        """Test that unexpose_all unexposes the permission plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()
        assert registry.is_exposed("permission")

        registry.unexpose_all()
        assert not registry.is_exposed("permission")


class TestRegistryAskPermissionTool:
    """Tests for askPermission tool exposure via registry."""

    def test_askPermission_declaration_when_exposed(self):
        """Test that askPermission is in declarations when permission plugin is exposed."""
        registry = PluginRegistry()
        registry.discover()

        # Not exposed yet
        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]
        assert "askPermission" not in tool_names

        # Expose
        registry.expose_tool("permission")
        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]
        assert "askPermission" in tool_names

    def test_askPermission_executor_when_exposed(self):
        """Test that askPermission executor is available when permission plugin is exposed."""
        registry = PluginRegistry()
        registry.discover()

        # Not exposed yet
        executors = registry.get_exposed_executors()
        assert "askPermission" not in executors

        # Expose
        registry.expose_tool("permission")
        executors = registry.get_exposed_executors()
        assert "askPermission" in executors
        assert callable(executors["askPermission"])

    def test_askPermission_not_available_after_unexpose(self):
        """Test that askPermission is not available after unexposing."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission")
        assert "askPermission" in registry.get_exposed_executors()

        registry.unexpose_tool("permission")
        assert "askPermission" not in registry.get_exposed_executors()


class TestRegistryAskPermissionExecution:
    """Tests for executing askPermission via registry executors."""

    def test_execute_askPermission_allowed(self):
        """Test executing askPermission for an allowed tool."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission", config={
            "policy": {"defaultPolicy": "allow"}
        })

        executors = registry.get_exposed_executors()
        result = executors["askPermission"]({
            "tool_name": "some_tool",
            "arguments": {"arg": "val"}
        })

        assert result["allowed"] is True
        assert result["tool_name"] == "some_tool"

    def test_execute_askPermission_denied(self):
        """Test executing askPermission for a blocked tool."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission", config={
            "policy": {
                "defaultPolicy": "deny",
                "blacklist": {"tools": ["blocked_tool"]}
            }
        })

        executors = registry.get_exposed_executors()
        result = executors["askPermission"]({
            "tool_name": "blocked_tool",
            "arguments": {}
        })

        assert result["allowed"] is False

    def test_execute_askPermission_pattern_check(self):
        """Test executing askPermission with pattern matching."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission", config={
            "policy": {
                "defaultPolicy": "deny",
                "whitelist": {"patterns": ["git *"]}
            }
        })

        executors = registry.get_exposed_executors()

        # Should be allowed
        result = executors["askPermission"]({
            "tool_name": "cli_based_tool",
            "arguments": {"command": "git status"}
        })
        assert result["allowed"] is True

        # Should be denied
        result = executors["askPermission"]({
            "tool_name": "cli_based_tool",
            "arguments": {"command": "rm -rf /tmp"}
        })
        assert result["allowed"] is False


class TestRegistryPermissionPluginReconfiguration:
    """Tests for reconfiguring the permission plugin via registry."""

    def test_reconfigure_permission_plugin(self):
        """Test that re-exposing with new config reconfigures the plugin."""
        registry = PluginRegistry()
        registry.discover()

        # First config - allow all
        registry.expose_tool("permission", config={
            "policy": {"defaultPolicy": "allow"}
        })

        executors = registry.get_exposed_executors()
        result = executors["askPermission"]({"tool_name": "test", "arguments": {}})
        assert result["allowed"] is True

        # Reconfigure - deny all
        registry.expose_tool("permission", config={
            "policy": {"defaultPolicy": "deny"}
        })

        executors = registry.get_exposed_executors()
        result = executors["askPermission"]({"tool_name": "test", "arguments": {}})
        assert result["allowed"] is False


class TestRegistryPermissionPluginWithConfigFile:
    """Tests for permission plugin with config file via registry."""

    def test_expose_with_config_file_path(self):
        """Test exposing permission plugin with a config file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "version": "1.0",
                "defaultPolicy": "allow",
                "blacklist": {
                    "tools": ["blocked_from_file"],
                    "patterns": ["rm -rf *"]
                }
            }
            json.dump(config, f)
            f.flush()

            try:
                registry = PluginRegistry()
                registry.discover()

                registry.expose_tool("permission", config={
                    "config_path": f.name
                })

                executors = registry.get_exposed_executors()

                # Tool in file blacklist should be blocked
                result = executors["askPermission"]({
                    "tool_name": "blocked_from_file",
                    "arguments": {}
                })
                assert result["allowed"] is False

                # Pattern in file blacklist should be blocked
                result = executors["askPermission"]({
                    "tool_name": "cli_based_tool",
                    "arguments": {"command": "rm -rf /tmp/test"}
                })
                assert result["allowed"] is False

                # Other tools should be allowed (default allow)
                result = executors["askPermission"]({
                    "tool_name": "other_tool",
                    "arguments": {}
                })
                assert result["allowed"] is True

            finally:
                os.unlink(f.name)


class TestRegistryMultiplePluginsWithPermission:
    """Tests for permission plugin alongside other plugins."""

    def test_permission_plugin_declarations_combined(self):
        """Test that permission plugin declarations combine with others."""
        registry = PluginRegistry()
        registry.discover()

        # Expose permission and any other available plugins
        registry.expose_tool("permission")

        declarations = registry.get_exposed_declarations()
        tool_names = [d.name for d in declarations]

        # askPermission should be present
        assert "askPermission" in tool_names

    def test_permission_plugin_executors_combined(self):
        """Test that permission plugin executors combine with others."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission")

        executors = registry.get_exposed_executors()

        # askPermission executor should be present
        assert "askPermission" in executors


class TestRegistryPermissionPluginErrors:
    """Tests for error handling with permission plugin via registry."""

    def test_expose_nonexistent_plugin(self):
        """Test that exposing a non-existent plugin raises error."""
        registry = PluginRegistry()
        registry.discover()

        with pytest.raises(ValueError, match="not found"):
            registry.expose_tool("nonexistent_plugin")

    def test_askPermission_without_tool_name(self):
        """Test that askPermission requires tool_name."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission")
        executors = registry.get_exposed_executors()

        result = executors["askPermission"]({"arguments": {}})
        assert "error" in result


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self):
        """Test that unexposing a plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("permission")
        plugin = registry.get_plugin("permission")

        # Plugin should be initialized
        assert plugin._initialized is True

        registry.unexpose_tool("permission")

        # Plugin should be shut down
        assert plugin._initialized is False

    def test_unexpose_all_cleans_up(self):
        """Test that unexpose_all properly cleans up."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        # All should be exposed
        assert registry.is_exposed("permission")

        registry.unexpose_all()

        # All should be unexposed
        assert not registry.is_exposed("permission")
        assert len(registry.list_exposed()) == 0
