"""Tests for AppArmorManager."""

import os
import platform
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Avoid importing server.__init__ which pulls heavy deps (google, etc.)
import types
if "server" not in sys.modules:
    _stub = types.ModuleType("server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["server"] = _stub

import server.apparmor as _apparmor_mod
AppArmorManager = _apparmor_mod.AppArmorManager


@pytest.fixture
def workspace_root(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    (root / "sessions").mkdir()
    return root


@pytest.fixture
def profile_dir(tmp_path):
    d = tmp_path / "apparmor_profiles"
    d.mkdir()
    return d


@pytest.fixture
def manager(workspace_root, profile_dir):
    return AppArmorManager(
        workspace_root=str(workspace_root),
        venv_path="/usr/local/venv",
        profile_dir=str(profile_dir),
    )


class TestAvailability:
    def test_not_available_on_non_linux(self, manager):
        with patch("server.apparmor.platform.system", return_value="Darwin"):
            manager._available = None  # Reset cache
            assert manager.is_available() is False

    def test_not_available_without_apparmor_parser(self, manager):
        with patch("server.apparmor.platform.system", return_value="Linux"), \
             patch("server.apparmor.shutil.which", return_value=None):
            manager._available = None
            assert manager.is_available() is False

    def test_not_available_without_kernel_module(self, manager):
        with patch("server.apparmor.platform.system", return_value="Linux"), \
             patch("server.apparmor.shutil.which", return_value="/usr/sbin/apparmor_parser"), \
             patch("server.apparmor.Path.exists", return_value=False):
            manager._available = None
            assert manager.is_available() is False

    def test_caches_result(self, manager):
        manager._available = True
        assert manager.is_available() is True
        # Should not re-check
        manager._available = False
        assert manager.is_available() is False


class TestProfileName:
    def test_profile_name_format(self, manager):
        assert manager.get_profile_name("20250101_120000") == "jaato-ws-20250101_120000"

    def test_profile_name_with_special_chars(self, manager):
        assert manager.get_profile_name("session_1") == "jaato-ws-session_1"


class TestRenderProfile:
    def test_contains_workspace_path(self, manager):
        profile = manager._render_profile("s1", "/srv/workspaces/sessions/s1")
        assert "/srv/workspaces/sessions/s1/" in profile
        assert "/srv/workspaces/sessions/s1/**" in profile

    def test_contains_venv_path(self, manager):
        profile = manager._render_profile("s1", "/workspace")
        assert "/usr/local/venv/" in profile

    def test_denies_sessions_root(self, manager, workspace_root):
        profile = manager._render_profile("s1", "/workspace")
        sessions_root = str(workspace_root / "sessions")
        assert f"deny {sessions_root}/" in profile

    def test_profile_name_in_output(self, manager):
        profile = manager._render_profile("test_session", "/workspace")
        assert "jaato-ws-test_session" in profile


class TestWrapCommand:
    def test_wraps_when_available(self, manager):
        manager._available = True
        result = manager.wrap_command("s1", ["git", "status"])
        assert result == ["aa-exec", "-p", "jaato-ws-s1", "--", "git", "status"]

    def test_passthrough_when_unavailable(self, manager):
        manager._available = False
        cmd = ["git", "status"]
        result = manager.wrap_command("s1", cmd)
        assert result == cmd

    def test_preserves_empty_command(self, manager):
        manager._available = True
        result = manager.wrap_command("s1", [])
        assert result == ["aa-exec", "-p", "jaato-ws-s1", "--"]


class TestWrapShellCommand:
    def test_wraps_shell_command(self, manager):
        manager._available = True
        result = manager.wrap_shell_command("s1", "echo hello")
        assert "aa-exec -p jaato-ws-s1 --" in result
        assert "echo hello" in result

    def test_escapes_single_quotes(self, manager):
        manager._available = True
        result = manager.wrap_shell_command("s1", "echo 'hello world'")
        assert "aa-exec" in result
        # Should be safely escaped
        assert "hello world" in result

    def test_passthrough_when_unavailable(self, manager):
        manager._available = False
        result = manager.wrap_shell_command("s1", "echo hello")
        assert result == "echo hello"


class TestProvisionProfile:
    def test_writes_and_loads_profile(self, manager, profile_dir):
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = manager.provision_profile("s1", "/workspace/sessions/s1")

        assert result is True
        profile_path = profile_dir / "jaato-ws-s1"
        assert profile_path.exists()
        content = profile_path.read_text()
        assert "jaato-ws-s1" in content

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "apparmor_parser" in call_args[0][0][0]

    def test_returns_false_when_unavailable(self, manager):
        manager._available = False
        assert manager.provision_profile("s1", "/workspace") is False

    def test_cleans_up_on_parser_failure(self, manager, profile_dir):
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="parse error")
            result = manager.provision_profile("s1", "/workspace")

        assert result is False
        assert not (profile_dir / "jaato-ws-s1").exists()


class TestTeardownProfile:
    def test_unloads_and_removes_profile(self, manager, profile_dir):
        manager._available = True
        # Create a profile file
        profile_file = profile_dir / "jaato-ws-s1"
        profile_file.write_text("# profile")

        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = manager.teardown_profile("s1")

        assert result is True
        assert not profile_file.exists()

    def test_returns_true_if_already_gone(self, manager):
        manager._available = True
        assert manager.teardown_profile("nonexistent") is True

    def test_returns_false_when_unavailable(self, manager):
        manager._available = False
        assert manager.teardown_profile("s1") is False
