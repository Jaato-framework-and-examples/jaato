"""Tests for the CLI tool plugin."""

import os
import sys
import pytest
from typing import Optional

from ..plugin import (
    CLIToolPlugin,
    create_plugin,
    DEFAULT_AUTO_BACKGROUND_THRESHOLD,
    SLOW_COMMAND_PATTERNS,
)
from ...background.protocol import BackgroundCapable


class TestCLIPluginInitialization:
    """Tests for plugin initialization."""

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, CLIToolPlugin)

    def test_plugin_name(self):
        plugin = CLIToolPlugin()
        assert plugin.name == "cli"

    def test_initialize_without_config(self):
        plugin = CLIToolPlugin()
        plugin.initialize()
        assert plugin._initialized is True
        assert plugin._extra_paths == []

    def test_initialize_with_extra_paths_list(self):
        plugin = CLIToolPlugin()
        plugin.initialize({"extra_paths": ["/usr/local/bin", "/opt/bin"]})
        assert plugin._initialized is True
        assert plugin._extra_paths == ["/usr/local/bin", "/opt/bin"]

    def test_initialize_with_extra_paths_string(self):
        plugin = CLIToolPlugin()
        plugin.initialize({"extra_paths": "/usr/local/bin"})
        assert plugin._initialized is True
        assert plugin._extra_paths == ["/usr/local/bin"]

    def test_initialize_with_empty_extra_paths(self):
        plugin = CLIToolPlugin()
        plugin.initialize({"extra_paths": []})
        assert plugin._initialized is True
        assert plugin._extra_paths == []

    def test_shutdown(self):
        plugin = CLIToolPlugin()
        plugin.initialize({"extra_paths": ["/usr/local/bin"]})
        plugin.shutdown()

        assert plugin._initialized is False
        assert plugin._extra_paths == []


class TestCLIPluginToolSchemas:
    """Tests for tool schemas."""

    def test_get_tool_schemas(self):
        plugin = CLIToolPlugin()
        schemas = plugin.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0].name == "cli_based_tool"

    def test_cli_based_tool_schema(self):
        plugin = CLIToolPlugin()
        schemas = plugin.get_tool_schemas()
        cli_tool = schemas[0]
        schema = cli_tool.parameters

        assert schema["type"] == "object"
        assert "command" in schema["properties"]
        assert "args" in schema["properties"]
        assert "command" in schema["required"]

    def test_cli_based_tool_description(self):
        plugin = CLIToolPlugin()
        declarations = plugin.get_tool_schemas()
        cli_tool = declarations[0]

        # Description should mention shell command execution
        assert "shell command" in cli_tool.description.lower()
        assert "execute" in cli_tool.description.lower()


class TestCLIPluginExecutors:
    """Tests for executor mapping."""

    def test_get_executors(self):
        plugin = CLIToolPlugin()
        executors = plugin.get_executors()

        assert "cli_based_tool" in executors
        assert callable(executors["cli_based_tool"])


class TestCLIPluginSystemInstructions:
    """Tests for system instructions."""

    def test_get_system_instructions(self):
        plugin = CLIToolPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "cli_based_tool" in instructions
        assert "shell commands" in instructions.lower()

    def test_get_auto_approved_tools(self):
        plugin = CLIToolPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        # CLI tools require permission - should return empty list
        assert auto_approved == []


class TestCLIPluginExecution:
    """Tests for command execution."""

    def test_execute_simple_command(self):
        """Test executing a simple echo command."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        # Use echo which is available on both Unix and Windows (via cmd)
        if sys.platform == "win32":
            result = plugin._execute({"command": "cmd /c echo hello"})
        else:
            result = plugin._execute({"command": "echo hello"})

        assert "error" not in result
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    def test_execute_command_not_found(self):
        """Test handling of non-existent command."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        result = plugin._execute({"command": "nonexistent_command_xyz"})

        assert "error" in result
        assert "not found in PATH" in result["error"]
        assert "hint" in result

    def test_execute_missing_command(self):
        """Test handling of missing command parameter."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        result = plugin._execute({})

        assert "error" in result
        assert "command must be provided" in result["error"]

    def test_execute_with_args(self):
        """Test executing command with separate args."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        if sys.platform == "win32":
            # Windows: use cmd /c with args
            result = plugin._execute({"command": "cmd", "args": ["/c", "echo", "hello"]})
        else:
            result = plugin._execute({"command": "echo", "args": ["hello", "world"]})

        assert "error" not in result
        assert result["returncode"] == 0


class TestCLIPluginShellDetection:
    """Tests for shell metacharacter detection."""

    def test_requires_shell_simple_command(self):
        """Simple commands should not require shell."""
        plugin = CLIToolPlugin()
        assert plugin._requires_shell("echo hello") is False
        assert plugin._requires_shell("ls -la") is False
        assert plugin._requires_shell("git status") is False

    def test_requires_shell_pipe(self):
        """Commands with pipes require shell."""
        plugin = CLIToolPlugin()
        assert plugin._requires_shell("ls | grep foo") is True
        assert plugin._requires_shell("cat file.txt | head -5") is True

    def test_requires_shell_redirection(self):
        """Commands with redirections require shell."""
        plugin = CLIToolPlugin()
        assert plugin._requires_shell("echo hello > file.txt") is True
        assert plugin._requires_shell("echo hello >> file.txt") is True
        assert plugin._requires_shell("cat < input.txt") is True

    def test_requires_shell_command_chaining(self):
        """Commands with chaining require shell."""
        plugin = CLIToolPlugin()
        assert plugin._requires_shell("cd /tmp && ls") is True
        assert plugin._requires_shell("ls || echo 'failed'") is True
        assert plugin._requires_shell("echo a; echo b") is True

    def test_requires_shell_command_substitution(self):
        """Commands with substitution require shell."""
        plugin = CLIToolPlugin()
        assert plugin._requires_shell("echo $(date)") is True
        assert plugin._requires_shell("echo `date`") is True

    def test_requires_shell_background(self):
        """Commands with background execution require shell."""
        plugin = CLIToolPlugin()
        assert plugin._requires_shell("sleep 10 &") is True


class TestCLIPluginShellExecution:
    """Tests for shell command execution."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_pipe_command(self):
        """Test executing a command with pipe."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        result = plugin._execute({"command": "echo 'hello world' | grep hello"})

        assert "error" not in result
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_command_chaining(self):
        """Test executing chained commands with &&."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        result = plugin._execute({"command": "echo 'first' && echo 'second'"})

        assert "error" not in result
        assert result["returncode"] == 0
        assert "first" in result["stdout"]
        assert "second" in result["stdout"]

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_pipe_with_head(self):
        """Test executing pipe with head to limit output."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        result = plugin._execute({"command": "echo -e 'a\\nb\\nc\\nd\\ne' | head -2"})

        assert "error" not in result
        assert result["returncode"] == 0
        # Should only have first two lines
        lines = result["stdout"].strip().split('\n')
        assert len(lines) == 2

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_command_substitution(self):
        """Test executing command with substitution."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        result = plugin._execute({"command": "echo $(echo nested)"})

        assert "error" not in result
        assert result["returncode"] == 0
        assert "nested" in result["stdout"]


class TestCLIPluginBackgroundCapability:
    """Tests for background capability support."""

    def test_plugin_implements_background_capable_protocol(self):
        """Test that CLIToolPlugin implements BackgroundCapable protocol."""
        plugin = CLIToolPlugin()
        assert isinstance(plugin, BackgroundCapable)

    def test_supports_background_cli_tool(self):
        """Test that cli_based_tool supports background execution."""
        plugin = CLIToolPlugin()
        assert plugin.supports_background("cli_based_tool") is True

    def test_supports_background_other_tool(self):
        """Test that non-existent tools do not support background."""
        plugin = CLIToolPlugin()
        assert plugin.supports_background("other_tool") is False
        assert plugin.supports_background("") is False

    def test_get_auto_background_threshold_default(self):
        """Test default auto-background threshold."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        threshold = plugin.get_auto_background_threshold("cli_based_tool")
        assert threshold == DEFAULT_AUTO_BACKGROUND_THRESHOLD

    def test_get_auto_background_threshold_configured(self):
        """Test configured auto-background threshold."""
        plugin = CLIToolPlugin()
        plugin.initialize({"auto_background_threshold": 30.0})

        threshold = plugin.get_auto_background_threshold("cli_based_tool")
        assert threshold == 30.0

    def test_get_auto_background_threshold_other_tool(self):
        """Test threshold returns None for unsupported tools."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        threshold = plugin.get_auto_background_threshold("other_tool")
        assert threshold is None

    def test_estimate_duration_known_patterns(self):
        """Test duration estimation for known slow command patterns."""
        plugin = CLIToolPlugin()

        # Test a few known patterns
        assert plugin.estimate_duration("cli_based_tool", {"command": "npm install"}) == 30.0
        assert plugin.estimate_duration("cli_based_tool", {"command": "pip install requests"}) == 20.0
        assert plugin.estimate_duration("cli_based_tool", {"command": "cargo build --release"}) == 60.0
        assert plugin.estimate_duration("cli_based_tool", {"command": "pytest tests/"}) == 30.0
        assert plugin.estimate_duration("cli_based_tool", {"command": "docker build ."}) == 60.0

    def test_estimate_duration_unknown_command(self):
        """Test duration estimation returns None for unknown commands."""
        plugin = CLIToolPlugin()

        assert plugin.estimate_duration("cli_based_tool", {"command": "echo hello"}) is None
        assert plugin.estimate_duration("cli_based_tool", {"command": "ls -la"}) is None
        assert plugin.estimate_duration("cli_based_tool", {"command": "cat file.txt"}) is None

    def test_estimate_duration_empty_command(self):
        """Test duration estimation handles empty command."""
        plugin = CLIToolPlugin()

        assert plugin.estimate_duration("cli_based_tool", {"command": ""}) is None
        assert plugin.estimate_duration("cli_based_tool", {}) is None

    def test_estimate_duration_other_tool(self):
        """Test duration estimation returns None for other tools."""
        plugin = CLIToolPlugin()

        assert plugin.estimate_duration("other_tool", {"command": "npm install"}) is None

    def test_slow_command_patterns_exist(self):
        """Test that slow command patterns are defined."""
        assert len(SLOW_COMMAND_PATTERNS) > 0

        # Check some expected patterns exist
        assert "npm install" in SLOW_COMMAND_PATTERNS
        assert "pip install" in SLOW_COMMAND_PATTERNS
        assert "make" in SLOW_COMMAND_PATTERNS
        assert "pytest" in SLOW_COMMAND_PATTERNS

    def test_initialize_with_background_max_workers(self):
        """Test initialization with custom max workers."""
        plugin = CLIToolPlugin()
        plugin.initialize({"background_max_workers": 8})

        # Should configure the background executor
        assert plugin._bg_max_workers == 8

    def test_shutdown_cleans_up_background_executor(self):
        """Test that shutdown properly cleans up background resources."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        # Start a background execution to initialize the executor
        # Note: We don't actually need to do this since the mixin
        # handles lazy initialization

        plugin.shutdown()
        assert plugin._initialized is False


class TestCLIPluginPathSandboxing:
    """Tests for path sandboxing functionality."""

    def test_initialize_with_workspace_root(self):
        """Test initialization with workspace_root config."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": "/tmp/workspace"})

        assert plugin._workspace_root is not None
        # Should be resolved to absolute path
        assert os.path.isabs(plugin._workspace_root)

    def test_initialize_without_workspace_root(self, monkeypatch):
        """Test initialization without workspace_root (no sandboxing)."""
        # Clear env vars to ensure no auto-detection
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        plugin = CLIToolPlugin()
        plugin.initialize()

        assert plugin._workspace_root is None

    def test_shutdown_clears_workspace_root(self):
        """Test that shutdown clears workspace_root."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": "/tmp/workspace"})
        plugin.shutdown()

        assert plugin._workspace_root is None

    def test_extract_path_tokens_absolute_paths(self):
        """Test extraction of absolute paths from commands."""
        plugin = CLIToolPlugin()

        tokens = plugin._extract_path_tokens("cat /etc/passwd")
        assert "/etc/passwd" in tokens

        tokens = plugin._extract_path_tokens("ls /home/user /tmp")
        assert "/home/user" in tokens
        assert "/tmp" in tokens

    def test_extract_path_tokens_relative_traversal(self):
        """Test extraction of paths with .. traversal."""
        plugin = CLIToolPlugin()

        tokens = plugin._extract_path_tokens("cat ../../../etc/passwd")
        assert "../../../etc/passwd" in tokens

        tokens = plugin._extract_path_tokens("ls foo/../bar")
        assert "foo/../bar" in tokens

    def test_extract_path_tokens_explicit_relative(self):
        """Test extraction of ./ relative paths."""
        plugin = CLIToolPlugin()

        tokens = plugin._extract_path_tokens("cat ./config.yaml")
        assert "./config.yaml" in tokens

    def test_extract_path_tokens_home_directory(self):
        """Test extraction of ~ home directory paths."""
        plugin = CLIToolPlugin()

        tokens = plugin._extract_path_tokens("cat ~/.bashrc")
        assert "~/.bashrc" in tokens

        tokens = plugin._extract_path_tokens("ls ~/Documents")
        assert "~/Documents" in tokens

    def test_extract_path_tokens_excludes_urls(self):
        """Test that URLs are not extracted as paths."""
        plugin = CLIToolPlugin()

        tokens = plugin._extract_path_tokens("curl https://example.com/path/to/file")
        assert "https://example.com/path/to/file" not in tokens

        tokens = plugin._extract_path_tokens("wget http://example.com/download")
        assert "http://example.com/download" not in tokens

    def test_extract_path_tokens_excludes_options(self):
        """Test that option flags are not extracted as paths."""
        plugin = CLIToolPlugin()

        tokens = plugin._extract_path_tokens("ls -la --color=auto")
        assert "-la" not in tokens
        assert "--color=auto" not in tokens

    def test_extract_path_tokens_excludes_npm_packages(self):
        """Test that npm package names are not extracted as paths."""
        plugin = CLIToolPlugin()

        tokens = plugin._extract_path_tokens("npm install @scope/package")
        assert "@scope/package" not in tokens

    def test_is_path_within_workspace_no_sandboxing(self, monkeypatch):
        """Test that all paths are allowed when no workspace_root is set."""
        # Clear env vars to ensure no auto-detection
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        plugin = CLIToolPlugin()
        plugin.initialize()  # No workspace_root

        assert plugin._is_path_within_workspace("/etc/passwd") is True
        assert plugin._is_path_within_workspace("../../../anywhere") is True

    def test_is_path_within_workspace_inside(self, tmp_path):
        """Test that paths inside workspace are allowed."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        # Direct child
        assert plugin._is_path_within_workspace(str(tmp_path / "file.txt")) is True

        # Nested child
        assert plugin._is_path_within_workspace(str(tmp_path / "sub" / "file.txt")) is True

        # Workspace root itself
        assert plugin._is_path_within_workspace(str(tmp_path)) is True

    def test_is_path_within_workspace_outside(self, tmp_path):
        """Test that paths outside workspace are blocked (except /tmp which is always allowed)."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        # Absolute path outside workspace and /tmp - should be blocked
        assert plugin._is_path_within_workspace("/etc/passwd") is False

        # Parent directory under /tmp - now ALLOWED since /tmp is always accessible
        assert plugin._is_path_within_workspace(str(tmp_path.parent)) is True

        # Home directory (outside /tmp) - should be blocked
        assert plugin._is_path_within_workspace("~/.bashrc") is False

    def test_is_path_within_workspace_traversal_blocked(self, tmp_path):
        """Test that .. traversal behavior respects /tmp allowance."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        # Create cwd context - simulate being inside workspace
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Traversal that stays inside workspace is allowed
            subdir = tmp_path / "sub"
            subdir.mkdir()
            assert plugin._is_path_within_workspace("sub/../file.txt") is True

            # Traversal that escapes workspace but stays under /tmp is ALLOWED
            # (because /tmp is always accessible)
            assert plugin._is_path_within_workspace("../outside.txt") is True

            # Traversal to /etc (outside /tmp) is BLOCKED
            assert plugin._is_path_within_workspace("/etc/passwd") is False

        finally:
            os.chdir(original_cwd)

    def test_validate_command_paths_no_sandboxing(self, monkeypatch):
        """Test validation passes when no workspace_root is set."""
        # Clear env vars to ensure no auto-detection
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        plugin = CLIToolPlugin()
        plugin.initialize()

        result = plugin._validate_command_paths("cat /etc/passwd")
        assert result is None  # No blocking

    def test_validate_command_paths_allowed(self, tmp_path):
        """Test validation passes for paths inside workspace."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        # Command with path inside workspace
        result = plugin._validate_command_paths(f"cat {tmp_path}/file.txt")
        assert result is None

    def test_validate_command_paths_blocked(self, tmp_path):
        """Test validation returns blocked path for paths outside workspace."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        result = plugin._validate_command_paths("cat /etc/passwd")
        assert result == "/etc/passwd"

    def test_make_not_found_result(self):
        """Test generation of not-found error result."""
        plugin = CLIToolPlugin()

        result = plugin._make_not_found_result("/etc/passwd", "cat /etc/passwd")

        assert result["stdout"] == ""
        assert "No such file or directory" in result["stderr"]
        assert "/etc/passwd" in result["stderr"]
        assert result["returncode"] == 1

    def test_make_not_found_result_uses_command_name(self):
        """Test that error message uses the command name."""
        plugin = CLIToolPlugin()

        result = plugin._make_not_found_result("/etc/passwd", "cat /etc/passwd")
        assert result["stderr"].startswith("cat:")

        result = plugin._make_not_found_result("/etc/passwd", "ls /etc/passwd")
        assert result["stderr"].startswith("ls:")

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_blocks_path_outside_workspace(self, tmp_path):
        """Test that execute blocks commands accessing paths outside workspace."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        result = plugin._execute({"command": "cat /etc/passwd"})

        assert result["returncode"] == 1
        assert "No such file or directory" in result["stderr"]
        assert result["stdout"] == ""

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_allows_path_inside_workspace(self, tmp_path):
        """Test that execute allows commands accessing paths inside workspace."""
        # Create a file inside workspace
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        result = plugin._execute({"command": f"cat {test_file}"})

        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_blocks_traversal_to_non_tmp_path(self, tmp_path):
        """Test that traversal to paths outside /tmp is blocked."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        # Try to access /etc/passwd directly (outside /tmp) - should be blocked
        result = plugin._execute({"command": "cat /etc/passwd"})

        assert result["returncode"] == 1
        assert "No such file or directory" in result["stderr"]

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_allows_traversal_within_tmp(self, tmp_path):
        """Test that traversal within /tmp is allowed."""
        # Create a file in parent directory (still under /tmp)
        parent_file = tmp_path.parent / "parent_test.txt"
        parent_file.write_text("parent content")

        try:
            plugin = CLIToolPlugin()
            plugin.initialize({"workspace_root": str(tmp_path)})

            # Traversal to parent (still under /tmp) - should be allowed
            result = plugin._execute({"command": f"cat {parent_file}"})

            assert result["returncode"] == 0
            assert "parent content" in result["stdout"]
        finally:
            parent_file.unlink(missing_ok=True)

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_blocks_home_directory_access(self, tmp_path):
        """Test that ~ home directory access is blocked when outside workspace."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        result = plugin._execute({"command": "cat ~/.bashrc"})

        assert result["returncode"] == 1
        assert "No such file or directory" in result["stderr"]

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
    def test_execute_allows_commands_without_paths(self, tmp_path):
        """Test that commands without path arguments work normally."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        # Simple command without paths
        result = plugin._execute({"command": "echo hello"})

        assert result["returncode"] == 0
        assert "hello" in result["stdout"]

    def test_validate_arg_list_paths(self, tmp_path):
        """Test that paths in arg_list are also validated."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        # Path in arg_list should be validated
        result = plugin._validate_command_paths("cat", arg_list=["/etc/passwd"])
        assert result == "/etc/passwd"

        # Path inside workspace should pass
        result = plugin._validate_command_paths("cat", arg_list=[f"{tmp_path}/file.txt"])
        assert result is None

    def test_auto_detect_workspace_root_from_jaato_env(self, tmp_path, monkeypatch):
        """Test auto-detection of workspace_root from JAATO_WORKSPACE_ROOT."""
        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", str(tmp_path))
        # Clear workspaceRoot to ensure priority is tested
        monkeypatch.delenv("workspaceRoot", raising=False)

        plugin = CLIToolPlugin()
        plugin.initialize()  # No explicit workspace_root

        assert plugin._workspace_root == str(tmp_path.resolve())

    def test_auto_detect_workspace_root_from_dotenv(self, tmp_path, monkeypatch):
        """Test auto-detection of workspace_root from workspaceRoot (.env style)."""
        # Clear JAATO_WORKSPACE_ROOT to test fallback
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.setenv("workspaceRoot", str(tmp_path))

        plugin = CLIToolPlugin()
        plugin.initialize()

        assert plugin._workspace_root == str(tmp_path.resolve())

    def test_auto_detect_jaato_takes_precedence(self, tmp_path, monkeypatch):
        """Test that JAATO_WORKSPACE_ROOT takes precedence over workspaceRoot."""
        jaato_path = tmp_path / "jaato"
        jaato_path.mkdir()
        dotenv_path = tmp_path / "dotenv"
        dotenv_path.mkdir()

        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", str(jaato_path))
        monkeypatch.setenv("workspaceRoot", str(dotenv_path))

        plugin = CLIToolPlugin()
        plugin.initialize()

        assert plugin._workspace_root == str(jaato_path.resolve())

    def test_explicit_config_overrides_auto_detect(self, tmp_path, monkeypatch):
        """Test that explicit workspace_root config overrides auto-detection."""
        explicit_path = tmp_path / "explicit"
        explicit_path.mkdir()
        env_path = tmp_path / "env"
        env_path.mkdir()

        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", str(env_path))

        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(explicit_path)})

        assert plugin._workspace_root == str(explicit_path.resolve())

    def test_no_workspace_root_when_env_not_set(self, monkeypatch):
        """Test that sandboxing is disabled when no env vars are set."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        plugin = CLIToolPlugin()
        plugin.initialize()

        assert plugin._workspace_root is None
