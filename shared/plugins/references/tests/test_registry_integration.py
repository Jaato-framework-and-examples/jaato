"""Tests for references plugin integration with the plugin registry."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from ...registry import PluginRegistry
from ..plugin import ReferencesPlugin, create_plugin
from ..models import ReferenceSource, InjectionMode, SourceType


class TestRegistryPluginDiscovery:
    """Tests for discovering the references plugin via the registry."""

    def test_references_plugin_discovered(self):
        """Test that references plugin is discovered by registry."""
        registry = PluginRegistry()
        discovered = registry.discover()

        assert "references" in discovered

    def test_references_plugin_available(self):
        """Test that references plugin is available after discovery."""
        registry = PluginRegistry()
        registry.discover()

        assert "references" in registry.list_available()

    def test_get_references_plugin(self):
        """Test retrieving the references plugin by name."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin("references")
        assert plugin is not None
        assert plugin.name == "references"


class TestRegistryExposeReferencesPlugin:
    """Tests for exposing the references plugin via the registry."""

    def test_expose_references_plugin(self):
        """Test exposing the references plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        assert registry.is_exposed("references")
        assert "references" in registry.list_exposed()

        registry.unexpose_tool("references")

    def test_unexpose_references_plugin(self):
        """Test unexposing the references plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        assert registry.is_exposed("references")

        registry.unexpose_tool("references")
        assert not registry.is_exposed("references")

    def test_expose_all_includes_references(self):
        """Test that expose_all includes the references plugin."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_all()

        assert registry.is_exposed("references")

        registry.unexpose_all()


class TestRegistryReferencesToolSchemas:
    """Tests for references tool schemas exposure via registry."""

    def test_references_tools_not_exposed_before_expose(self):
        """Test that references tools are not in schemas before expose."""
        registry = PluginRegistry()
        registry.discover()

        schemas = registry.get_exposed_tool_schemas()
        tool_names = [s.name for s in schemas]

        assert "selectReferences" not in tool_names
        assert "listReferences" not in tool_names

    def test_references_tools_exposed_after_expose(self):
        """Test that references tools are in schemas after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        schemas = registry.get_exposed_tool_schemas()
        tool_names = [s.name for s in schemas]

        assert "selectReferences" in tool_names
        assert "listReferences" in tool_names

        registry.unexpose_tool("references")

    def test_references_tools_not_exposed_after_unexpose(self):
        """Test that references tools are not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        registry.unexpose_tool("references")

        schemas = registry.get_exposed_tool_schemas()
        tool_names = [s.name for s in schemas]

        assert "selectReferences" not in tool_names
        assert "listReferences" not in tool_names


class TestRegistryReferencesExecutors:
    """Tests for references executors exposure via registry."""

    def test_executor_not_available_before_expose(self):
        """Test that executor is not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        executors = registry.get_exposed_executors()

        assert "selectReferences" not in executors
        assert "listReferences" not in executors
        assert "references" not in executors

    def test_executor_available_after_expose(self):
        """Test that executor is available after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        executors = registry.get_exposed_executors()

        assert "selectReferences" in executors
        assert "listReferences" in executors
        assert "references" in executors
        assert callable(executors["selectReferences"])
        assert callable(executors["listReferences"])
        assert callable(executors["references"])

        registry.unexpose_tool("references")

    def test_executor_not_available_after_unexpose(self):
        """Test that executor is not available after unexpose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        registry.unexpose_tool("references")

        executors = registry.get_exposed_executors()

        assert "selectReferences" not in executors
        assert "listReferences" not in executors
        assert "references" not in executors


class TestRegistryReferencesAutoApproval:
    """Tests for references auto-approved tools via registry."""

    def test_references_tools_are_auto_approved(self):
        """Test that references tools ARE auto-approved (user-triggered)."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        auto_approved = registry.get_auto_approved_tools()

        # Model tools and user command are all auto-approved
        assert "selectReferences" in auto_approved
        assert "listReferences" in auto_approved
        assert "references" in auto_approved

        registry.unexpose_tool("references")


class TestRegistryReferencesSystemInstructions:
    """Tests for references system instructions via registry."""

    def test_system_instructions_with_no_sources(self):
        """Test system instructions when no sources are configured."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        # Without references.json or sources, may return None
        plugin = registry.get_plugin("references")
        instructions = plugin.get_system_instructions()

        # With no sources, instructions may be None
        # This is expected behavior

        registry.unexpose_tool("references")


class TestRegistryReferencesUserCommands:
    """Tests for references user commands via registry."""

    def test_user_commands_included(self):
        """Test that references user command is included after expose."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        user_commands = registry.get_exposed_user_commands()
        command_names = [cmd.name for cmd in user_commands]

        assert "references" in command_names

        registry.unexpose_tool("references")

    def test_user_commands_not_included_before_expose(self):
        """Test that user commands are not available before expose."""
        registry = PluginRegistry()
        registry.discover()

        user_commands = registry.get_exposed_user_commands()
        command_names = [cmd.name for cmd in user_commands]

        assert "references" not in command_names

    def test_user_commands_shared_with_model(self):
        """Test that references user command has share_with_model=True."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        user_commands = registry.get_exposed_user_commands()
        ref_cmd = next((cmd for cmd in user_commands if cmd.name == "references"), None)

        assert ref_cmd is not None
        assert ref_cmd.share_with_model is True

        registry.unexpose_tool("references")


class TestRegistryPluginForTool:
    """Tests for get_plugin_for_tool with references plugin."""

    def test_get_plugin_for_select_references(self):
        """Test that get_plugin_for_tool returns references plugin for selectReferences."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        plugin = registry.get_plugin_for_tool("selectReferences")
        assert plugin is not None
        assert plugin.name == "references"

        registry.unexpose_tool("references")

    def test_get_plugin_for_list_references(self):
        """Test that get_plugin_for_tool returns references plugin for listReferences."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")

        plugin = registry.get_plugin_for_tool("listReferences")
        assert plugin is not None
        assert plugin.name == "references"

        registry.unexpose_tool("references")

    def test_get_plugin_for_tool_returns_none_when_not_exposed(self):
        """Test that get_plugin_for_tool returns None when plugin not exposed."""
        registry = PluginRegistry()
        registry.discover()

        plugin = registry.get_plugin_for_tool("selectReferences")
        assert plugin is None


class TestRegistryShutdownCleanup:
    """Tests for shutdown and cleanup behavior."""

    def test_unexpose_calls_shutdown(self):
        """Test that unexposing the plugin calls its shutdown method."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        plugin = registry.get_plugin("references")

        assert plugin._initialized is True

        registry.unexpose_tool("references")

        assert plugin._initialized is False

    def test_sources_cleared_on_shutdown(self):
        """Test that sources are cleared on shutdown."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        plugin = registry.get_plugin("references")

        registry.unexpose_tool("references")

        # After shutdown, sources should be cleared
        assert plugin._sources == []

    def test_selections_cleared_on_shutdown(self):
        """Test that selections are cleared on shutdown."""
        registry = PluginRegistry()
        registry.discover()

        registry.expose_tool("references")
        plugin = registry.get_plugin("references")

        # Simulate a selection
        plugin._selected_source_ids = ["test-source-1"]

        registry.unexpose_tool("references")

        # After shutdown, selections should be cleared
        assert plugin._selected_source_ids == []


class TestReferencesSandboxIntegration:
    """Tests for references plugin interaction with the sandbox manager.

    Verifies that selecting/unselecting references correctly adds/removes
    paths from the sandbox via the sandbox plugin's programmatic API.
    """

    @pytest.fixture
    def plugin_with_sandbox(self, tmp_path):
        """Create a references plugin with a mock sandbox_manager plugin."""
        # Create a mock registry that returns a mock sandbox plugin
        mock_sandbox = Mock()
        mock_sandbox.add_path_programmatic = Mock(return_value=True)
        mock_sandbox.remove_path_programmatic = Mock(return_value=True)

        mock_registry = Mock()
        mock_registry.get_plugin = Mock(return_value=mock_sandbox)
        mock_registry.authorize_external_path = Mock()
        mock_registry.deauthorize_external_path = Mock()
        mock_registry.clear_authorized_paths = Mock()

        # Create a local reference source with a real file
        ref_file = tmp_path / "docs" / "spec.md"
        ref_file.parent.mkdir(parents=True)
        ref_file.write_text("# API Spec\nSome content.")

        source = ReferenceSource(
            id="api-spec",
            name="API Spec",
            description="API specification",
            type=SourceType.LOCAL,
            mode=InjectionMode.SELECTABLE,
            path=str(ref_file),
            resolved_path=str(ref_file),
        )

        plugin = create_plugin()
        plugin._plugin_registry = mock_registry
        plugin._sources = [source]
        plugin._selected_source_ids = []
        plugin._initialized = True
        plugin._project_root = str(tmp_path)

        return plugin, mock_sandbox, mock_registry, source, ref_file

    def test_select_calls_sandbox_add_readonly(self, plugin_with_sandbox):
        """Test that selecting a reference calls sandbox add_path_programmatic with readonly."""
        plugin, mock_sandbox, _, source, ref_file = plugin_with_sandbox

        result = plugin._cmd_references_select("api-spec")

        assert result["status"] == "selected"
        mock_sandbox.add_path_programmatic.assert_called_once_with(
            str(ref_file), access="readonly"
        )

    def test_unselect_calls_sandbox_remove(self, plugin_with_sandbox):
        """Test that unselecting a reference calls sandbox remove_path_programmatic."""
        plugin, mock_sandbox, _, source, ref_file = plugin_with_sandbox

        # First select
        plugin._cmd_references_select("api-spec")
        mock_sandbox.reset_mock()

        # Now unselect
        result = plugin._cmd_references_unselect("api-spec")

        assert result["status"] == "unselected"
        mock_sandbox.remove_path_programmatic.assert_called_once_with(
            str(ref_file)
        )

    def test_select_unselect_roundtrip(self, plugin_with_sandbox):
        """Test that select then unselect results in clean state."""
        plugin, mock_sandbox, _, source, ref_file = plugin_with_sandbox

        # Select
        plugin._cmd_references_select("api-spec")
        assert "api-spec" in plugin._selected_source_ids
        assert mock_sandbox.add_path_programmatic.call_count == 1

        # Unselect
        plugin._cmd_references_unselect("api-spec")
        assert "api-spec" not in plugin._selected_source_ids
        assert mock_sandbox.remove_path_programmatic.call_count == 1

    def test_select_falls_back_to_registry_when_no_sandbox(self, tmp_path):
        """Test fallback to direct registry auth when sandbox plugin is not available."""
        ref_file = tmp_path / "doc.md"
        ref_file.write_text("content")

        source = ReferenceSource(
            id="doc-1",
            name="Doc",
            description="A doc",
            type=SourceType.LOCAL,
            mode=InjectionMode.SELECTABLE,
            path=str(ref_file),
            resolved_path=str(ref_file),
        )

        mock_registry = Mock()
        mock_registry.get_plugin = Mock(return_value=None)  # No sandbox plugin
        mock_registry.authorize_external_path = Mock()
        mock_registry.deauthorize_external_path = Mock()

        plugin = create_plugin()
        plugin._plugin_registry = mock_registry
        plugin._sources = [source]
        plugin._selected_source_ids = []
        plugin._initialized = True
        plugin._project_root = str(tmp_path)

        plugin._cmd_references_select("doc-1")

        # Should fall back to direct registry call with readonly
        mock_registry.authorize_external_path.assert_called_once_with(
            str(ref_file), "references", access="readonly"
        )

    def test_unselect_falls_back_to_registry_when_no_sandbox(self, tmp_path):
        """Test fallback to direct registry deauth when sandbox plugin is not available."""
        ref_file = tmp_path / "doc.md"
        ref_file.write_text("content")

        source = ReferenceSource(
            id="doc-1",
            name="Doc",
            description="A doc",
            type=SourceType.LOCAL,
            mode=InjectionMode.SELECTABLE,
            path=str(ref_file),
            resolved_path=str(ref_file),
        )

        mock_registry = Mock()
        mock_registry.get_plugin = Mock(return_value=None)  # No sandbox plugin
        mock_registry.authorize_external_path = Mock()
        mock_registry.deauthorize_external_path = Mock()

        plugin = create_plugin()
        plugin._plugin_registry = mock_registry
        plugin._sources = [source]
        plugin._selected_source_ids = ["doc-1"]
        plugin._initialized = True
        plugin._project_root = str(tmp_path)

        plugin._cmd_references_unselect("doc-1")

        # Should fall back to direct registry deauth
        mock_registry.deauthorize_external_path.assert_called_once_with(
            str(ref_file), "references"
        )

    def test_non_local_source_skips_sandbox(self, plugin_with_sandbox):
        """Test that non-LOCAL sources don't interact with sandbox."""
        plugin, mock_sandbox, _, _, _ = plugin_with_sandbox

        url_source = ReferenceSource(
            id="url-ref",
            name="URL Ref",
            description="A URL reference",
            type=SourceType.URL,
            mode=InjectionMode.SELECTABLE,
            url="https://example.com/doc",
        )
        plugin._sources.append(url_source)

        plugin._cmd_references_select("url-ref")

        # Sandbox should not be called for URL sources
        mock_sandbox.add_path_programmatic.assert_not_called()

    def test_select_via_execute_select_calls_sandbox(self, plugin_with_sandbox):
        """Test that the model tool selectReferences also calls sandbox."""
        plugin, mock_sandbox, _, source, ref_file = plugin_with_sandbox

        # Model selects directly by ID (no channel interaction)
        result = plugin._execute_select({"ids": ["api-spec"]})

        assert result["status"] == "success"
        assert result["selected_count"] == 1
        assert result["sources"][0]["id"] == "api-spec"
        assert result["sources"][0]["resolved_path"] == str(ref_file)
        mock_sandbox.add_path_programmatic.assert_called_once_with(
            str(ref_file), access="readonly"
        )

    def test_select_by_tags_calls_sandbox(self, plugin_with_sandbox):
        """Test that selecting by tags also calls sandbox."""
        plugin, mock_sandbox, _, source, ref_file = plugin_with_sandbox
        # Add tags to the source for tag-based selection
        source.tags = ["api", "spec"]

        result = plugin._execute_select({"filter_tags": ["api"]})

        assert result["status"] == "success"
        assert result["selected_count"] == 1
        assert result["sources"][0]["id"] == "api-spec"
        mock_sandbox.add_path_programmatic.assert_called_once_with(
            str(ref_file), access="readonly"
        )

    def test_select_returns_resolved_path(self, plugin_with_sandbox):
        """Test that selectReferences returns the resolved real path."""
        plugin, mock_sandbox, _, source, ref_file = plugin_with_sandbox

        result = plugin._execute_select({"ids": ["api-spec"]})

        assert result["status"] == "success"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["resolved_path"] == str(ref_file)
        assert result["sources"][0]["is_directory"] is False

    def test_select_requires_ids_or_tags(self, plugin_with_sandbox):
        """Test that selectReferences requires ids or filter_tags."""
        plugin, _, _, _, _ = plugin_with_sandbox

        result = plugin._execute_select({})

        assert result["status"] == "error"
        assert "ids" in result["message"] or "filter_tags" in result["message"]

    def test_select_already_selected_id(self, plugin_with_sandbox):
        """Test selecting an already-selected reference."""
        plugin, _, _, source, _ = plugin_with_sandbox

        # First selection
        plugin._execute_select({"ids": ["api-spec"]})

        # Second selection of the same ID — all sources already selected
        result = plugin._execute_select({"ids": ["api-spec"]})

        assert result["status"] == "all_selected"
        assert "already selected" in result["message"].lower()
