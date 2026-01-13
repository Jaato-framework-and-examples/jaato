"""Tests for sandbox_utils module - .jaato contained symlink escape.

These tests verify the security behavior of the .jaato containment feature:
- .jaato can be a symlink to an external directory (allowed)
- Paths under .jaato must stay within the resolved .jaato boundary
- Path traversal attacks (.jaato/../secret) are blocked
- Nested symlinks inside .jaato are blocked
"""

import os
import tempfile
import pytest
from pathlib import Path

from shared.plugins.sandbox_utils import (
    is_jaato_path,
    get_jaato_boundary,
    detect_jaato_symlink,
    has_nested_symlink,
    is_path_within_jaato_boundary,
    check_path_with_jaato_containment,
    JAATO_CONFIG_DIR,
)


class TestIsJaatoPath:
    """Tests for is_jaato_path function."""

    def test_exact_jaato_dir(self, tmp_path):
        """Test that .jaato itself is detected as jaato path."""
        workspace = str(tmp_path)
        jaato_path = os.path.join(workspace, JAATO_CONFIG_DIR)
        assert is_jaato_path(jaato_path, workspace) is True

    def test_path_under_jaato(self, tmp_path):
        """Test that paths under .jaato are detected."""
        workspace = str(tmp_path)
        config_path = os.path.join(workspace, JAATO_CONFIG_DIR, "config.json")
        assert is_jaato_path(config_path, workspace) is True

    def test_nested_path_under_jaato(self, tmp_path):
        """Test that deeply nested paths under .jaato are detected."""
        workspace = str(tmp_path)
        nested_path = os.path.join(workspace, JAATO_CONFIG_DIR, "vision", "captures", "img.png")
        assert is_jaato_path(nested_path, workspace) is True

    def test_path_outside_jaato(self, tmp_path):
        """Test that paths not under .jaato return False."""
        workspace = str(tmp_path)
        src_path = os.path.join(workspace, "src", "main.py")
        assert is_jaato_path(src_path, workspace) is False

    def test_similar_name_not_matched(self, tmp_path):
        """Test that directories named .jaato-something aren't matched."""
        workspace = str(tmp_path)
        similar_path = os.path.join(workspace, ".jaato-backup", "file.txt")
        assert is_jaato_path(similar_path, workspace) is False


class TestGetJaatoBoundary:
    """Tests for get_jaato_boundary function."""

    def test_regular_jaato_dir(self, tmp_path):
        """Test boundary detection for regular .jaato directory."""
        workspace = tmp_path
        jaato_dir = workspace / JAATO_CONFIG_DIR
        jaato_dir.mkdir()

        boundary = get_jaato_boundary(str(workspace))
        assert boundary == str(jaato_dir.resolve())

    def test_symlinked_jaato_dir(self, tmp_path):
        """Test boundary detection when .jaato is a symlink."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()

        # Create symlink: workspace/.jaato -> ../external_jaato
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        boundary = get_jaato_boundary(str(workspace))
        assert boundary == str(external_jaato.resolve())

    def test_nonexistent_jaato(self, tmp_path):
        """Test that nonexistent .jaato returns None."""
        workspace = str(tmp_path)
        boundary = get_jaato_boundary(workspace)
        assert boundary is None


class TestDetectJaatoSymlink:
    """Tests for detect_jaato_symlink function."""

    def test_detect_symlink(self, tmp_path):
        """Test detection of symlinked .jaato."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "external"
        external.mkdir()

        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external)

        is_symlink, target = detect_jaato_symlink(str(workspace))
        assert is_symlink is True
        assert target == str(external.resolve())

    def test_detect_regular_dir(self, tmp_path):
        """Test that regular .jaato is not detected as symlink."""
        workspace = tmp_path
        jaato_dir = workspace / JAATO_CONFIG_DIR
        jaato_dir.mkdir()

        is_symlink, target = detect_jaato_symlink(str(workspace))
        assert is_symlink is False
        assert target is None

    def test_detect_nonexistent(self, tmp_path):
        """Test detection when .jaato doesn't exist."""
        is_symlink, target = detect_jaato_symlink(str(tmp_path))
        assert is_symlink is False
        assert target is None


class TestHasNestedSymlink:
    """Tests for has_nested_symlink function."""

    def test_no_nested_symlinks(self, tmp_path):
        """Test path with no nested symlinks."""
        jaato_boundary = tmp_path / "jaato"
        jaato_boundary.mkdir()
        subdir = jaato_boundary / "config"
        subdir.mkdir()
        config_file = subdir / "settings.json"
        config_file.touch()

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = has_nested_symlink(
            str(config_file),
            str(jaato_boundary),
            str(workspace)
        )
        assert result is False

    def test_nested_symlink_blocked(self, tmp_path):
        """Test that nested symlink inside .jaato is detected."""
        jaato_boundary = tmp_path / "jaato"
        jaato_boundary.mkdir()
        external = tmp_path / "external"
        external.mkdir()

        # Create nested symlink: jaato/plugins -> /external
        plugins_link = jaato_boundary / "plugins"
        plugins_link.symlink_to(external)

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        # Create the logical path structure
        jaato_in_workspace = workspace / JAATO_CONFIG_DIR
        jaato_in_workspace.symlink_to(jaato_boundary)

        # Path being checked: workspace/.jaato/plugins/something
        target_path = str(workspace / JAATO_CONFIG_DIR / "plugins" / "plugin.py")

        result = has_nested_symlink(
            target_path,
            str(jaato_boundary),
            str(workspace)
        )
        assert result is True

    def test_traversal_detected(self, tmp_path):
        """Test that path traversal is detected."""
        jaato_boundary = tmp_path / "jaato"
        jaato_boundary.mkdir()

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Path that tries to escape via ..
        escape_path = str(workspace / JAATO_CONFIG_DIR / ".." / "secret.txt")

        result = has_nested_symlink(
            escape_path,
            str(jaato_boundary),
            str(workspace)
        )
        assert result is True


class TestIsPathWithinJaatoBoundary:
    """Tests for is_path_within_jaato_boundary function."""

    def test_path_within_boundary(self, tmp_path):
        """Test that path within boundary is allowed."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()

        # Create config file in external jaato
        config_file = external_jaato / "config.json"
        config_file.touch()

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        # Test path through symlink
        path = str(workspace / JAATO_CONFIG_DIR / "config.json")

        result = is_path_within_jaato_boundary(
            path,
            str(workspace),
            str(external_jaato)
        )
        assert result is True

    def test_traversal_escape_blocked(self, tmp_path):
        """Test that path traversal escape is blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()

        # Create secret file outside jaato
        secret = tmp_path / "secret.txt"
        secret.touch()

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        # Try to escape via traversal
        escape_path = str(workspace / JAATO_CONFIG_DIR / ".." / ".." / "secret.txt")

        result = is_path_within_jaato_boundary(
            escape_path,
            str(workspace),
            str(external_jaato)
        )
        assert result is False

    def test_nested_symlink_escape_blocked(self, tmp_path):
        """Test that nested symlink escape is blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()
        malicious_target = tmp_path / "malicious"
        malicious_target.mkdir()
        secret = malicious_target / "secret.txt"
        secret.touch()

        # Create nested symlink that escapes
        nested_link = external_jaato / "plugins"
        nested_link.symlink_to(malicious_target)

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        # Try to access through nested symlink
        malicious_path = str(workspace / JAATO_CONFIG_DIR / "plugins" / "secret.txt")

        result = is_path_within_jaato_boundary(
            malicious_path,
            str(workspace),
            str(external_jaato)
        )
        assert result is False


class TestCheckPathWithJaatoContainment:
    """Integration tests for check_path_with_jaato_containment."""

    def test_no_workspace_root_allows_all(self):
        """Test that missing workspace_root allows all paths."""
        result = check_path_with_jaato_containment("/any/path", None)
        assert result is True

    def test_path_in_workspace_allowed(self, tmp_path):
        """Test that regular workspace paths are allowed."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        src = workspace / "src"
        src.mkdir()
        main_py = src / "main.py"
        main_py.touch()

        result = check_path_with_jaato_containment(
            str(main_py),
            str(workspace)
        )
        assert result is True

    def test_path_outside_workspace_blocked(self, tmp_path):
        """Test that paths outside workspace are blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        secret = external / "secret.txt"
        secret.touch()

        result = check_path_with_jaato_containment(
            str(secret),
            str(workspace)
        )
        assert result is False

    def test_jaato_symlink_allowed(self, tmp_path):
        """Test that .jaato symlink to external is allowed."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()
        config = external_jaato / "config.json"
        config.touch()

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        # Access through symlink
        path = str(workspace / JAATO_CONFIG_DIR / "config.json")

        result = check_path_with_jaato_containment(
            path,
            str(workspace)
        )
        assert result is True

    def test_jaato_traversal_blocked(self, tmp_path):
        """Test that .jaato/../escape is blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()
        secret = tmp_path / "secret.txt"
        secret.touch()

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        # Try to escape
        escape_path = str(workspace / JAATO_CONFIG_DIR / ".." / "secret.txt")

        result = check_path_with_jaato_containment(
            escape_path,
            str(workspace)
        )
        assert result is False

    def test_jaato_nested_symlink_blocked(self, tmp_path):
        """Test that nested symlinks inside .jaato are blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()
        malicious = tmp_path / "malicious"
        malicious.mkdir()
        secret = malicious / "passwd"
        secret.touch()

        # Create nested symlink inside jaato
        nested_link = external_jaato / "etc"
        nested_link.symlink_to(malicious)

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        # Try to access through nested symlink
        attack_path = str(workspace / JAATO_CONFIG_DIR / "etc" / "passwd")

        result = check_path_with_jaato_containment(
            attack_path,
            str(workspace)
        )
        assert result is False

    def test_nonexistent_jaato_blocks_jaato_paths(self, tmp_path):
        """Test that paths under .jaato are blocked if .jaato doesn't exist."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        # Don't create .jaato

        path = str(workspace / JAATO_CONFIG_DIR / "config.json")

        result = check_path_with_jaato_containment(
            path,
            str(workspace)
        )
        assert result is False

    def test_regular_symlink_in_workspace_blocked(self, tmp_path):
        """Test that regular symlinks in workspace (not .jaato) that escape are blocked."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        secret = external / "secret.txt"
        secret.touch()

        # Create symlink in workspace to external
        link = workspace / "external_link"
        link.symlink_to(external)

        # Try to access through symlink
        path = str(workspace / "external_link" / "secret.txt")

        result = check_path_with_jaato_containment(
            path,
            str(workspace)
        )
        assert result is False

    def test_deep_nested_path_in_jaato(self, tmp_path):
        """Test deeply nested paths in .jaato are allowed."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()

        # Create deep directory structure
        deep_dir = external_jaato / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)
        deep_file = deep_dir / "config.yaml"
        deep_file.touch()

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        path = str(workspace / JAATO_CONFIG_DIR / "level1" / "level2" / "level3" / "config.yaml")

        result = check_path_with_jaato_containment(
            path,
            str(workspace)
        )
        assert result is True


class TestPluginRegistryIntegration:
    """Tests for plugin registry authorization integration."""

    def test_registry_authorizes_external_path(self, tmp_path):
        """Test that plugin registry can authorize external paths."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_docs = tmp_path / "docs"
        external_docs.mkdir()
        doc_file = external_docs / "guide.md"
        doc_file.touch()

        # Mock registry that authorizes the external path
        class MockRegistry:
            def is_path_authorized(self, path):
                return path.startswith(str(external_docs))

        result = check_path_with_jaato_containment(
            str(doc_file),
            str(workspace),
            MockRegistry()
        )
        assert result is True

    def test_registry_does_not_override_jaato_blocking(self, tmp_path):
        """Test that registry auth doesn't bypass jaato containment checks.

        Note: The current implementation checks jaato paths first, before
        checking registry authorization. This test verifies that behavior.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()
        malicious = tmp_path / "malicious"
        malicious.mkdir()

        # Create nested symlink
        nested_link = external_jaato / "bad"
        nested_link.symlink_to(malicious)

        # Symlink .jaato
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        # Mock registry that would authorize the path
        class MockRegistry:
            def is_path_authorized(self, path):
                return True  # Authorize everything

        # Path through nested symlink should still be blocked
        attack_path = str(workspace / JAATO_CONFIG_DIR / "bad" / "secret.txt")

        result = check_path_with_jaato_containment(
            attack_path,
            str(workspace),
            MockRegistry()
        )
        # Should be blocked because of nested symlink, even though registry allows
        assert result is False
