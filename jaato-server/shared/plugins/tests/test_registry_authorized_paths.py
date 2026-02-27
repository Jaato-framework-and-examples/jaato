"""Tests for the authorized external paths mechanism in PluginRegistry."""

import os
import pytest
from pathlib import Path

from ..registry import PluginRegistry


class TestAuthorizedExternalPaths:
    """Tests for external path authorization in PluginRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance."""
        return PluginRegistry()

    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Create temporary paths for testing."""
        # Create test files and directories
        file1 = tmp_path / "file1.txt"
        file1.write_text("content1")

        file2 = tmp_path / "subdir" / "file2.txt"
        file2.parent.mkdir()
        file2.write_text("content2")

        file3 = tmp_path / "subdir" / "nested" / "file3.txt"
        file3.parent.mkdir()
        file3.write_text("content3")

        return {
            "file1": str(file1),
            "file2": str(file2),
            "file3": str(file3),
            "subdir": str(tmp_path / "subdir"),
            "root": str(tmp_path),
        }

    def test_authorize_single_path(self, registry, temp_paths):
        """Test authorizing a single file path."""
        registry.authorize_external_path(temp_paths["file1"], "test_plugin")

        assert registry.is_path_authorized(temp_paths["file1"]) is True
        assert registry.is_path_authorized(temp_paths["file2"]) is False

    def test_authorize_multiple_paths(self, registry, temp_paths):
        """Test authorizing multiple paths at once."""
        paths = [temp_paths["file1"], temp_paths["file2"]]
        registry.authorize_external_paths(paths, "test_plugin")

        assert registry.is_path_authorized(temp_paths["file1"]) is True
        assert registry.is_path_authorized(temp_paths["file2"]) is True
        assert registry.is_path_authorized(temp_paths["file3"]) is False

    def test_authorize_directory_includes_children(self, registry, temp_paths):
        """Test that authorizing a directory authorizes all files within it."""
        registry.authorize_external_path(temp_paths["subdir"], "test_plugin")

        # Files within the directory should be authorized
        assert registry.is_path_authorized(temp_paths["file2"]) is True
        assert registry.is_path_authorized(temp_paths["file3"]) is True

        # File outside the directory should not be authorized
        assert registry.is_path_authorized(temp_paths["file1"]) is False

    def test_get_authorization_source(self, registry, temp_paths):
        """Test getting the source plugin that authorized a path."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file2"], "plugin_b")

        assert registry.get_path_authorization_source(temp_paths["file1"]) == "plugin_a"
        assert registry.get_path_authorization_source(temp_paths["file2"]) == "plugin_b"
        assert registry.get_path_authorization_source(temp_paths["file3"]) is None

    def test_get_authorization_source_for_directory_child(self, registry, temp_paths):
        """Test getting source for a file within an authorized directory."""
        registry.authorize_external_path(temp_paths["subdir"], "dir_plugin")

        # File within directory should return the directory's source
        assert registry.get_path_authorization_source(temp_paths["file2"]) == "dir_plugin"
        assert registry.get_path_authorization_source(temp_paths["file3"]) == "dir_plugin"

    def test_clear_all_authorized_paths(self, registry, temp_paths):
        """Test clearing all authorized paths."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file2"], "plugin_b")

        count = registry.clear_authorized_paths()

        assert count == 2
        assert registry.is_path_authorized(temp_paths["file1"]) is False
        assert registry.is_path_authorized(temp_paths["file2"]) is False

    def test_clear_paths_from_specific_plugin(self, registry, temp_paths):
        """Test clearing paths from a specific plugin only."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file2"], "plugin_b")
        registry.authorize_external_path(temp_paths["file3"], "plugin_a")

        count = registry.clear_authorized_paths("plugin_a")

        assert count == 2
        # plugin_a paths should be cleared
        assert registry.is_path_authorized(temp_paths["file1"]) is False
        assert registry.is_path_authorized(temp_paths["file3"]) is False
        # plugin_b path should remain
        assert registry.is_path_authorized(temp_paths["file2"]) is True

    def test_list_authorized_paths(self, registry, temp_paths):
        """Test listing all authorized paths."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file2"], "plugin_b")

        paths = registry.list_authorized_paths()

        # Paths should be normalized
        normalized_file1 = os.path.realpath(temp_paths["file1"])
        normalized_file2 = os.path.realpath(temp_paths["file2"])

        assert normalized_file1 in paths
        assert normalized_file2 in paths
        assert paths[normalized_file1] == "plugin_a"
        assert paths[normalized_file2] == "plugin_b"

    def test_path_normalization(self, registry, tmp_path):
        """Test that paths are normalized when authorizing and checking."""
        # Create a file
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("content")

        # Authorize using a path with ..
        rel_path = str(tmp_path / "subdir" / ".." / "test_file.txt")
        registry.authorize_external_path(rel_path, "test_plugin")

        # Check using the canonical path
        canonical = str(file_path.resolve())
        assert registry.is_path_authorized(canonical) is True

    def test_empty_registry_checks(self, registry, temp_paths):
        """Test that checks work correctly on empty registry."""
        assert registry.is_path_authorized(temp_paths["file1"]) is False
        assert registry.get_path_authorization_source(temp_paths["file1"]) is None
        assert registry.list_authorized_paths() == {}
        assert registry.clear_authorized_paths() == 0

    def test_reauthorize_same_path_different_plugin(self, registry, temp_paths):
        """Test that reauthorizing a path updates the source plugin."""
        registry.authorize_external_path(temp_paths["file1"], "plugin_a")
        registry.authorize_external_path(temp_paths["file1"], "plugin_b")

        # Should reflect the latest authorization
        assert registry.get_path_authorization_source(temp_paths["file1"]) == "plugin_b"

    def test_relative_path_authorization(self, registry, tmp_path, monkeypatch):
        """Test authorizing with a relative path."""
        # Create a file
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        # Change to tmp_path and authorize with relative path
        monkeypatch.chdir(tmp_path)
        registry.authorize_external_path("test.txt", "test_plugin")

        # Check with absolute path
        assert registry.is_path_authorized(str(file_path)) is True


class TestExposeToolPreservesAuthorizedPaths:
    """Test that expose_tool re-init preserves authorized/denied paths.

    When a plugin is re-initialized via expose_tool() with a new config,
    shutdown() clears the plugin's authorized paths from the registry.
    The subsequent initialize() may not restore all of them (e.g. a subagent
    doesn't have the parent session's sandbox config). The registry must
    preserve paths that were lost during re-initialization.
    """

    @pytest.fixture
    def sandbox_dir(self, tmp_path):
        """Create a temp directory to use as a sandbox path."""
        d = tmp_path / "sandbox_target"
        d.mkdir()
        return str(d)

    def _make_plugin(self, name, registry):
        """Create a minimal plugin that clears its paths on shutdown.

        Mimics sandbox_manager's behavior: shutdown() calls
        clear_authorized_paths/clear_denied_paths, and initialize()
        does NOT re-add them (simulating a subagent that lacks the
        parent's session config).
        """
        class StubPlugin:
            def __init__(self, plugin_name, reg):
                self._name = plugin_name
                self._registry = reg

            @property
            def name(self):
                return self._name

            def initialize(self, config=None):
                pass

            def shutdown(self):
                self._registry.clear_authorized_paths(self._name)
                self._registry.clear_denied_paths(self._name)

            def get_tool_schemas(self):
                return []

            def get_executors(self):
                return {}

            def get_auto_approved_tools(self):
                return []

        return StubPlugin(name, registry)

    def test_reinit_preserves_authorized_paths(self, sandbox_dir):
        """Authorized paths survive a plugin re-init via expose_tool."""
        registry = PluginRegistry()
        plugin = self._make_plugin("sandbox_manager", registry)
        registry.register_plugin(plugin)

        # First expose — plugin gets initialized
        registry.expose_tool("sandbox_manager", {"session_id": "main"})
        # Simulate user running 'sandbox add' which authorizes a path
        registry.authorize_external_path(sandbox_dir, "sandbox_manager", access="readwrite")
        assert registry.is_path_authorized(sandbox_dir) is True

        # Re-expose with different config (simulates subagent spawn)
        # shutdown() will clear authorized paths, initialize() won't re-add them
        registry.expose_tool("sandbox_manager", {"agent_name": "sub1"})

        # Path should still be authorized
        assert registry.is_path_authorized(sandbox_dir) is True

    def test_reinit_preserves_denied_paths(self, sandbox_dir):
        """Denied paths survive a plugin re-init via expose_tool."""
        registry = PluginRegistry()
        plugin = self._make_plugin("sandbox_manager", registry)
        registry.register_plugin(plugin)

        registry.expose_tool("sandbox_manager", {"session_id": "main"})
        registry.deny_external_path(sandbox_dir, "sandbox_manager")
        assert registry.is_path_denied(sandbox_dir) is True

        # Re-expose with different config
        registry.expose_tool("sandbox_manager", {"agent_name": "sub1"})

        assert registry.is_path_denied(sandbox_dir) is True

    def test_reinit_does_not_restore_paths_readded_by_plugin(self, tmp_path):
        """If the plugin re-adds a path during init, the snapshot doesn't duplicate."""
        new_dir = tmp_path / "new_path"
        new_dir.mkdir()

        registry = PluginRegistry()

        class ReAddPlugin:
            """Plugin that re-adds a specific path during initialize."""
            def __init__(self):
                self._registry = None
                self._readd_path = str(new_dir)

            @property
            def name(self):
                return "readd_plugin"

            def initialize(self, config=None):
                # On re-init, the plugin adds a path from its config
                if config and config.get("readd") and self._registry:
                    self._registry.authorize_external_path(
                        self._readd_path, "readd_plugin", access="readonly"
                    )

            def shutdown(self):
                if self._registry:
                    self._registry.clear_authorized_paths("readd_plugin")
                    self._registry.clear_denied_paths("readd_plugin")

            def set_plugin_registry(self, reg):
                self._registry = reg

            def get_tool_schemas(self):
                return []

            def get_executors(self):
                return {}

            def get_auto_approved_tools(self):
                return []

        plugin = ReAddPlugin()
        registry.register_plugin(plugin)

        # First expose
        registry.expose_tool("readd_plugin", {"initial": True})
        # Authorize with readwrite
        registry.authorize_external_path(str(new_dir), "readd_plugin", access="readwrite")

        # Re-expose — plugin re-adds the same path as readonly
        registry.expose_tool("readd_plugin", {"readd": True})

        # The plugin's re-added version (readonly) should take precedence
        # since it was added by the current init, not the snapshot restore
        detailed = registry.list_authorized_paths_detailed()
        norm = os.path.realpath(str(new_dir))
        assert norm in detailed
        assert detailed[norm]["access"] == "readonly"

    def test_other_plugin_paths_unaffected(self, sandbox_dir, tmp_path):
        """Re-init of one plugin doesn't touch another plugin's paths."""
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        registry = PluginRegistry()
        plugin = self._make_plugin("sandbox_manager", registry)
        registry.register_plugin(plugin)

        registry.expose_tool("sandbox_manager", {"session_id": "main"})
        # One path from sandbox_manager, one from another plugin
        registry.authorize_external_path(sandbox_dir, "sandbox_manager")
        registry.authorize_external_path(str(other_dir), "references_plugin")

        # Re-init sandbox_manager
        registry.expose_tool("sandbox_manager", {"agent_name": "sub1"})

        # Both should survive
        assert registry.is_path_authorized(sandbox_dir) is True
        assert registry.is_path_authorized(str(other_dir)) is True
        # Other plugin's source should be unchanged
        assert registry.get_path_authorization_source(str(other_dir)) == "references_plugin"
