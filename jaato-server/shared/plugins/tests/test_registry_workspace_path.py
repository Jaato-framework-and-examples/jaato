# shared/plugins/tests/test_registry_workspace_path.py
"""Tests for workspace path broadcasting in PluginRegistry."""

import pytest
from unittest.mock import Mock, MagicMock

from ..registry import PluginRegistry


class MockPluginWithWorkspace:
    """Mock plugin that implements set_workspace_path."""

    def __init__(self):
        self.workspace_path = None
        self._tool_schemas = []

    def initialize(self, config=None):
        pass

    def shutdown(self):
        pass

    def get_tool_schemas(self):
        return self._tool_schemas

    def set_workspace_path(self, path):
        self.workspace_path = path


class MockPluginWithoutWorkspace:
    """Mock plugin that does NOT implement set_workspace_path."""

    def __init__(self):
        self._tool_schemas = []

    def initialize(self, config=None):
        pass

    def shutdown(self):
        pass

    def get_tool_schemas(self):
        return self._tool_schemas


class TestRegistryWorkspacePath:
    """Tests for set_workspace_path broadcasting."""

    @pytest.fixture
    def registry(self):
        """Create a registry with mock plugins."""
        reg = PluginRegistry()

        # Add mock plugins directly
        reg._plugins["with_workspace1"] = MockPluginWithWorkspace()
        reg._plugins["with_workspace2"] = MockPluginWithWorkspace()
        reg._plugins["without_workspace"] = MockPluginWithoutWorkspace()

        # Expose them
        reg._exposed.add("with_workspace1")
        reg._exposed.add("with_workspace2")
        reg._exposed.add("without_workspace")

        return reg

    def test_set_workspace_path_broadcasts_to_supporting_plugins(self, registry):
        """Should broadcast to all plugins implementing set_workspace_path."""
        registry.set_workspace_path("/test/workspace")

        # Plugins with set_workspace_path should receive it
        assert registry._plugins["with_workspace1"].workspace_path == "/test/workspace"
        assert registry._plugins["with_workspace2"].workspace_path == "/test/workspace"

    def test_set_workspace_path_skips_non_supporting_plugins(self, registry):
        """Should skip plugins without set_workspace_path (no error)."""
        # This should not raise an error
        registry.set_workspace_path("/test/workspace")

        # Plugin without method should be unchanged
        assert not hasattr(registry._plugins["without_workspace"], "workspace_path")

    def test_get_workspace_path_returns_stored_value(self, registry):
        """Should store and return the workspace path."""
        assert registry.get_workspace_path() is None

        registry.set_workspace_path("/my/workspace")
        assert registry.get_workspace_path() == "/my/workspace"

    def test_set_workspace_path_only_broadcasts_to_exposed(self, registry):
        """Should only broadcast to exposed plugins."""
        # Add unexposed plugin
        unexposed = MockPluginWithWorkspace()
        registry._plugins["unexposed"] = unexposed

        registry.set_workspace_path("/test/workspace")

        # Exposed plugins get the broadcast
        assert registry._plugins["with_workspace1"].workspace_path == "/test/workspace"

        # Unexposed plugin should NOT receive the broadcast
        assert unexposed.workspace_path is None

    def test_set_workspace_path_handles_exception_gracefully(self, registry):
        """Should continue broadcasting even if one plugin raises."""
        # Make one plugin raise
        def raise_error(path):
            raise RuntimeError("Plugin error")

        registry._plugins["with_workspace1"].set_workspace_path = raise_error

        # Should not raise, should continue to other plugins
        registry.set_workspace_path("/test/workspace")

        # Other plugin should still receive it
        assert registry._plugins["with_workspace2"].workspace_path == "/test/workspace"
