"""Tests for sandbox_utils module - .jaato access restriction.

These tests verify the security behavior of .jaato access control:
- .jaato paths are DENIED BY DEFAULT for model tool calls
- .jaato paths can be allowed via explicit registry authorization (sandbox add)
- Even when authorized, containment rules still apply:
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
    is_pseudo_device_path,
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
        # has_nested_symlink derives the LOGICAL .jaato dir from workspace_root
        # (``<workspace>/.jaato``) and walks the checked path relative to it.
        # Handing it a path under a bare ``<tmp>/jaato`` made relpath produce
        # "../../jaato/..." -- read as an escape attempt, so it returned True.
        # Model the real shape, as the sibling symlink tests do: .jaato lives
        # in the workspace and points at the boundary.
        (workspace / JAATO_CONFIG_DIR).symlink_to(jaato_boundary)
        checked = str(workspace / JAATO_CONFIG_DIR / "config" / "settings.json")

        result = has_nested_symlink(
            checked,
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
    """Integration tests for check_path_with_jaato_containment.

    .jaato paths are denied by default and require explicit authorization
    via the plugin registry (populated by 'sandbox add').
    """

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

        # /tmp is an ALLOWED zone by default (allow_tmp=True,
        # SYSTEM_TEMP_PATHS), and pytest's tmp_path lives under it -- so this
        # "external" dir was inside the allowance and the call correctly
        # returned True.  Turn the allowance off so the test exercises the
        # workspace boundary it actually describes.
        result = check_path_with_jaato_containment(
            str(secret),
            str(workspace),
            allow_tmp=False,
        )
        assert result is False

    def test_jaato_denied_by_default(self, tmp_path):
        """Test that .jaato paths are denied without registry authorization."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        jaato_dir = workspace / JAATO_CONFIG_DIR
        jaato_dir.mkdir()
        config = jaato_dir / "config.json"
        config.touch()

        # No registry provided - .jaato is denied
        result = check_path_with_jaato_containment(
            str(config),
            str(workspace)
        )
        assert result is False

    def test_jaato_denied_by_default_no_registry(self, tmp_path):
        """Test that .jaato paths via symlink are denied without registry."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()
        config = external_jaato / "config.json"
        config.touch()

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        # No registry provided - .jaato is denied
        result = check_path_with_jaato_containment(
            str(workspace / JAATO_CONFIG_DIR / "config.json"),
            str(workspace)
        )
        assert result is False

    def test_jaato_denied_with_empty_registry(self, tmp_path):
        """Test that .jaato is denied when registry has no authorization."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        jaato_dir = workspace / JAATO_CONFIG_DIR
        jaato_dir.mkdir()
        config = jaato_dir / "config.json"
        config.touch()

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return False

        result = check_path_with_jaato_containment(
            str(config),
            str(workspace),
            MockRegistry()
        )
        assert result is False

    def test_jaato_allowed_when_authorized(self, tmp_path):
        """Test that .jaato is allowed when explicitly authorized via registry."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        jaato_dir = workspace / JAATO_CONFIG_DIR
        jaato_dir.mkdir()
        config = jaato_dir / "config.json"
        config.touch()

        jaato_real = str(jaato_dir.resolve())

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return path.startswith(jaato_real)

        result = check_path_with_jaato_containment(
            str(config),
            str(workspace),
            MockRegistry()
        )
        assert result is True

    def test_jaato_symlink_allowed_when_authorized(self, tmp_path):
        """Test that .jaato symlink to external is allowed when authorized."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()
        config = external_jaato / "config.json"
        config.touch()

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        external_real = str(external_jaato.resolve())

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return path.startswith(external_real)

        path = str(workspace / JAATO_CONFIG_DIR / "config.json")

        result = check_path_with_jaato_containment(
            path,
            str(workspace),
            MockRegistry()
        )
        assert result is True

    def test_jaato_traversal_blocked(self, tmp_path):
        """Test that .jaato/../escape is blocked even when authorized."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_jaato = tmp_path / "external_jaato"
        external_jaato.mkdir()
        secret = tmp_path / "secret.txt"
        secret.touch()

        # Symlink .jaato to external
        jaato_link = workspace / JAATO_CONFIG_DIR
        jaato_link.symlink_to(external_jaato)

        # Authorize everything
        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return True

        # Try to escape
        escape_path = str(workspace / JAATO_CONFIG_DIR / ".." / "secret.txt")

        result = check_path_with_jaato_containment(
            escape_path,
            str(workspace),
            MockRegistry()
        )
        assert result is False

    def test_jaato_nested_symlink_blocked(self, tmp_path):
        """Test that nested symlinks inside .jaato are blocked even when authorized."""
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

        # Authorize everything
        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return True

        # Try to access through nested symlink
        attack_path = str(workspace / JAATO_CONFIG_DIR / "etc" / "passwd")

        result = check_path_with_jaato_containment(
            attack_path,
            str(workspace),
            MockRegistry()
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

        # /tmp is an ALLOWED zone by default (allow_tmp=True,
        # SYSTEM_TEMP_PATHS), and pytest's tmp_path lives under it -- so this
        # "external" dir was inside the allowance and the call correctly
        # returned True.  Turn the allowance off so the test exercises the
        # workspace boundary it actually describes.
        result = check_path_with_jaato_containment(
            path,
            str(workspace),
            allow_tmp=False,
        )
        assert result is False

    def test_deep_nested_path_in_jaato_denied_by_default(self, tmp_path):
        """Test deeply nested .jaato paths are denied without authorization."""
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
        assert result is False

    def test_deep_nested_path_in_jaato_allowed_when_authorized(self, tmp_path):
        """Test deeply nested .jaato paths are allowed when authorized."""
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

        external_real = str(external_jaato.resolve())

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return path.startswith(external_real)

        path = str(workspace / JAATO_CONFIG_DIR / "level1" / "level2" / "level3" / "config.yaml")

        result = check_path_with_jaato_containment(
            path,
            str(workspace),
            MockRegistry()
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
            def is_path_authorized(self, path, mode="read"):
                return path.startswith(str(external_docs))

        result = check_path_with_jaato_containment(
            str(doc_file),
            str(workspace),
            MockRegistry()
        )
        assert result is True

    def test_registry_authorization_required_for_jaato(self, tmp_path):
        """Test that .jaato paths require explicit registry authorization.

        Even though .jaato is within the workspace, it is denied by default
        and only accessible when the registry authorizes it (via sandbox add).
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        jaato_dir = workspace / JAATO_CONFIG_DIR
        jaato_dir.mkdir()
        config = jaato_dir / "config.json"
        config.touch()

        # Registry that does NOT authorize .jaato
        class MockRegistryDeny:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return False

        result = check_path_with_jaato_containment(
            str(config),
            str(workspace),
            MockRegistryDeny()
        )
        assert result is False

        # Registry that DOES authorize .jaato
        jaato_real = str(jaato_dir.resolve())

        class MockRegistryAllow:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return path.startswith(jaato_real)

        result = check_path_with_jaato_containment(
            str(config),
            str(workspace),
            MockRegistryAllow()
        )
        assert result is True

    def test_registry_does_not_override_jaato_containment(self, tmp_path):
        """Test that registry auth doesn't bypass jaato containment checks.

        Even when the registry authorizes everything, containment rules
        (traversal escape, nested symlinks) are still enforced for .jaato paths.
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
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
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


class TestAccessMode:
    """Tests for access mode (readonly/readwrite) support."""

    def test_read_mode_allowed_on_readonly_path(self, tmp_path):
        """Test that read access is allowed on a readonly authorized path."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_docs = tmp_path / "docs"
        external_docs.mkdir()
        doc_file = external_docs / "guide.md"
        doc_file.touch()

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                if not path.startswith(str(external_docs.resolve())):
                    return False
                # Only allow reads (simulates readonly)
                return mode == "read"

        # Disable /tmp allowance to test registry-based mode checking
        result = check_path_with_jaato_containment(
            str(doc_file),
            str(workspace),
            MockRegistry(),
            allow_tmp=False,
            mode="read"
        )
        assert result is True

    def test_write_mode_blocked_on_readonly_path(self, tmp_path):
        """Test that write access is blocked on a readonly authorized path."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external_docs = tmp_path / "docs"
        external_docs.mkdir()
        doc_file = external_docs / "guide.md"
        doc_file.touch()

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                if not path.startswith(str(external_docs.resolve())):
                    return False
                # Only allow reads (simulates readonly)
                return mode == "read"

        # Disable /tmp allowance to test registry-based mode checking
        result = check_path_with_jaato_containment(
            str(doc_file),
            str(workspace),
            MockRegistry(),
            allow_tmp=False,
            mode="write"
        )
        assert result is False

    def test_readwrite_mode_allows_both(self, tmp_path):
        """Test that readwrite authorized paths allow both read and write."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        ext_file = external / "data.txt"
        ext_file.touch()

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return path.startswith(str(external.resolve()))

        # Disable /tmp allowance to test registry-based mode checking
        for mode in ("read", "write"):
            result = check_path_with_jaato_containment(
                str(ext_file),
                str(workspace),
                MockRegistry(),
                allow_tmp=False,
                mode=mode
            )
            assert result is True, f"mode={mode} should be allowed"

    def test_workspace_paths_ignore_mode(self, tmp_path):
        """Test that workspace paths allow both read and write regardless of mode."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        src = workspace / "src"
        src.mkdir()
        main_py = src / "main.py"
        main_py.touch()

        # Workspace paths don't go through registry auth, so mode doesn't matter
        for mode in ("read", "write"):
            result = check_path_with_jaato_containment(
                str(main_py),
                str(workspace),
                allow_tmp=False,
                mode=mode
            )
            assert result is True, f"mode={mode} should be allowed for workspace paths"

    def test_tmp_paths_ignore_mode(self, tmp_path):
        """Test that /tmp paths allow both read and write regardless of mode."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        for mode in ("read", "write"):
            result = check_path_with_jaato_containment(
                "/tmp/some_file.txt",
                str(workspace),
                mode=mode
            )
            assert result is True, f"mode={mode} should be allowed for /tmp paths"

    def test_mode_passed_to_registry(self, tmp_path):
        """Test that the mode parameter is correctly passed to the registry."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "external"
        external.mkdir()
        ext_file = external / "file.txt"
        ext_file.touch()

        received_modes = []

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                received_modes.append(mode)
                return True

        # Disable /tmp allowance to ensure registry check is reached
        check_path_with_jaato_containment(
            str(ext_file), str(workspace), MockRegistry(),
            allow_tmp=False, mode="read"
        )
        check_path_with_jaato_containment(
            str(ext_file), str(workspace), MockRegistry(),
            allow_tmp=False, mode="write"
        )

        assert received_modes == ["read", "write"]

    def test_jaato_readonly_blocks_write(self, tmp_path):
        """Test that .jaato authorized as readonly blocks write access."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        jaato_dir = workspace / JAATO_CONFIG_DIR
        jaato_dir.mkdir()
        config = jaato_dir / "config.json"
        config.touch()

        jaato_real = str(jaato_dir.resolve())

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                if not path.startswith(jaato_real):
                    return False
                # Only allow reads
                return mode == "read"

        # Read should be allowed
        result = check_path_with_jaato_containment(
            str(config), str(workspace), MockRegistry(), mode="read"
        )
        assert result is True

        # Write should be blocked
        result = check_path_with_jaato_containment(
            str(config), str(workspace), MockRegistry(), mode="write"
        )
        assert result is False

    def test_jaato_readwrite_allows_both(self, tmp_path):
        """Test that .jaato authorized as readwrite allows both modes."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        jaato_dir = workspace / JAATO_CONFIG_DIR
        jaato_dir.mkdir()
        config = jaato_dir / "config.json"
        config.touch()

        jaato_real = str(jaato_dir.resolve())

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                return path.startswith(jaato_real)

        for m in ("read", "write"):
            result = check_path_with_jaato_containment(
                str(config), str(workspace), MockRegistry(), mode=m
            )
            assert result is True, f"mode={m} should be allowed for readwrite .jaato"

    def test_jaato_mode_passed_to_registry(self, tmp_path):
        """Test that the mode parameter is passed to registry for .jaato paths."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        jaato_dir = workspace / JAATO_CONFIG_DIR
        jaato_dir.mkdir()
        config = jaato_dir / "config.json"
        config.touch()

        received_modes = []

        class MockRegistry:
            def is_path_denied(self, path):
                return False
            def is_path_authorized(self, path, mode="read"):
                received_modes.append(mode)
                return True

        check_path_with_jaato_containment(
            str(config), str(workspace), MockRegistry(), mode="read"
        )
        check_path_with_jaato_containment(
            str(config), str(workspace), MockRegistry(), mode="write"
        )

        assert received_modes == ["read", "write"]


class TestTempAllowanceResolvesTheTarget:
    """The /tmp allowance must judge where a symlink POINTS, not where it lives.

    ``check_path_with_jaato_containment`` allows anything under a system temp
    directory, and that branch short-circuits the workspace check beneath it.
    Deciding it on the path as written admitted a symlink for its own
    location: a link at ``/tmp/x`` aimed at ``~/.ssh/id_rsa`` read as "under
    /tmp", so the content of a file outside both /tmp and the workspace came
    back allowed.  An allow rule has to resolve for the same reason a deny
    rule does (jaato issue #669).

    Found by attacking the branch rather than reading it — credit to the
    jaato-ac review session, whose controlled repro is the second test here.

    Every test substitutes its own temp root under ``tmp_path``.  The real
    one cannot be used: ``tmp_path`` is itself under ``/tmp`` on Linux, so a
    secret placed "outside" would still resolve into the allowance and the
    test would pass for the wrong reason.  That artifact is exactly what
    makes this bug easy to mis-attribute in either direction.
    """

    @pytest.fixture
    def temp_root(self, tmp_path, monkeypatch):
        """Substitute a controlled directory for the system temp roots.

        Returns:
            The directory that ``is_under_temp_path`` will treat as /tmp.
        """
        root = tmp_path / "faketmp"
        root.mkdir()
        monkeypatch.setattr(
            "shared.plugins.sandbox_utils.SYSTEM_TEMP_PATHS", [str(root)]
        )
        return root

    @pytest.fixture
    def secret(self, tmp_path):
        """A file outside both the temp root and any workspace.

        Returns:
            Path to the off-limits file.
        """
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "secret.txt"
        target.write_text("PRIVATEKEY\n")
        return target

    def test_symlink_in_temp_pointing_outside_is_denied(
        self, tmp_path, temp_root, secret
    ):
        """The wider variant: the workspace need not be under /tmp at all."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        link = temp_root / "leak_link.txt"
        link.symlink_to(secret)

        assert check_path_with_jaato_containment(
            str(link), str(workspace), None, allow_tmp=True, mode="read"
        ) is False

    def test_workspace_under_temp_does_not_leak_through_a_leaf_symlink(
        self, temp_root, secret
    ):
        """The reported repro, with the workspace itself under a temp root."""
        workspace = temp_root / "ws"
        workspace.mkdir()
        (workspace / "home_leaf.txt").symlink_to(secret)

        assert check_path_with_jaato_containment(
            str(workspace / "home_leaf.txt"),
            str(workspace),
            None,
            allow_tmp=True,
            mode="read",
        ) is False

    def test_symlinked_directory_in_temp_pointing_outside_is_denied(
        self, tmp_path, temp_root, secret
    ):
        """The parent, not the leaf, being the link out."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (temp_root / "vendor").symlink_to(secret.parent, target_is_directory=True)

        assert check_path_with_jaato_containment(
            str(temp_root / "vendor" / "secret.txt"),
            str(workspace),
            None,
            allow_tmp=True,
            mode="read",
        ) is False

    def test_ordinary_temp_file_is_still_allowed(self, tmp_path, temp_root):
        """The allowance itself must survive: no false positives."""
        target = temp_root / "ordinary.txt"
        target.write_text("fine\n")

        assert check_path_with_jaato_containment(
            str(target), str(tmp_path / "workspace"), None, allow_tmp=True, mode="read"
        ) is True

    def test_not_yet_created_temp_path_is_still_allowed(self, tmp_path, temp_root):
        """Writing a new file under /tmp must not require it to exist first."""
        target = temp_root / "does_not_exist_yet.txt"
        assert not target.exists()

        assert check_path_with_jaato_containment(
            str(target), str(tmp_path / "workspace"), None, allow_tmp=True, mode="write"
        ) is True

    def test_symlink_staying_inside_temp_is_allowed(self, tmp_path, temp_root):
        """Resolving must not break links that stay within the allowance."""
        real = temp_root / "real.txt"
        real.write_text("fine\n")
        link = temp_root / "alias.txt"
        link.symlink_to(real)

        assert check_path_with_jaato_containment(
            str(link), str(tmp_path / "workspace"), None, allow_tmp=True, mode="read"
        ) is True

    def test_a_symlinked_temp_root_still_matches(self, tmp_path, monkeypatch):
        """macOS ships ``/tmp`` as a symlink to ``/private/tmp``.

        Resolving the candidate but not the configured roots would reject
        every real temp path on such a platform, so the roots are resolved
        too.  This models that shape with a symlinked root of our own.
        """
        real_root = tmp_path / "private_tmp"
        real_root.mkdir()
        linked_root = tmp_path / "tmp_link"
        linked_root.symlink_to(real_root, target_is_directory=True)
        (real_root / "file.txt").write_text("fine\n")

        monkeypatch.setattr(
            "shared.plugins.sandbox_utils.SYSTEM_TEMP_PATHS", [str(linked_root)]
        )
        assert check_path_with_jaato_containment(
            str(linked_root / "file.txt"),
            str(tmp_path / "workspace"),
            None,
            allow_tmp=True,
            mode="read",
        ) is True

    def test_allow_tmp_false_is_unaffected(self, tmp_path, temp_root):
        target = temp_root / "off.txt"
        target.write_text("x")

        assert check_path_with_jaato_containment(
            str(target), str(tmp_path / "workspace"), None, allow_tmp=False, mode="read"
        ) is False


class TestPseudoDevicePaths:
    """Standard POSIX pseudo-devices are outside the sandbox's remit.

    Regression cover for jaato issue #784: ``2>/dev/null`` classified
    ``/dev/null`` as an out-of-workspace *write* target, so the cli path
    sandbox refused the whole command and reported the refusal as
    ``<cmd>: /dev/null: No such file or directory`` -- from which an agent
    can only conclude the machine has no ``/dev/null``.
    """

    @pytest.mark.parametrize("path", [
        "/dev/null",
        "/dev/zero",
        "/dev/full",
        "/dev/random",
        "/dev/urandom",
        "/dev/tty",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/fd/0",
        "/dev/fd/63",
        "/dev/./null",
        "/dev/foo/../null",
    ])
    def test_recognised(self, path):
        assert is_pseudo_device_path(path) is True

    @pytest.mark.parametrize("path", [
        "/dev/sda",
        "/dev/mem",
        "/dev/kmem",
        "/dev/pts/3",
        "/dev/shm/secret",
        "/dev",
        "/dev/nullx",
        "/dev/fd/notanumber",
        "/dev/fd",
        "dev/null",            # relative: an ordinary workspace file
        "./dev/null",
        "",
    ])
    def test_not_recognised(self, path):
        assert is_pseudo_device_path(path) is False

    def test_traversal_out_of_dev_is_not_recognised(self):
        """``/dev/../etc/passwd`` normalises out of /dev entirely."""
        assert is_pseudo_device_path("/dev/../etc/passwd") is False

    @pytest.mark.parametrize("path", ["/dev/null", "/dev/stdout", "/dev/fd/2"])
    @pytest.mark.parametrize("mode", ["read", "write"])
    def test_allowed_by_the_gate(self, tmp_path, path, mode):
        """The gate allows them for both access modes, with a workspace set."""
        assert check_path_with_jaato_containment(
            path, str(tmp_path), None, mode=mode
        ) is True

    def test_block_devices_still_refused(self, tmp_path):
        """The allowance is an explicit list, not a ``/dev/`` prefix match."""
        for path in ("/dev/sda", "/dev/mem", "/dev/pts/3"):
            assert check_path_with_jaato_containment(
                path, str(tmp_path), None, mode="write"
            ) is False

    def test_explicit_deny_rule_still_wins(self, tmp_path):
        """An operator denylist entry outranks the pseudo-device allowance."""

        class _Registry:
            def is_path_denied(self, path):
                return os.path.abspath(path) == "/dev/null"

        assert check_path_with_jaato_containment(
            "/dev/null", str(tmp_path), _Registry(), mode="write"
        ) is False
