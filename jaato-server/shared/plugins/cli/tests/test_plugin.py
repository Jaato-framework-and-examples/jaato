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

    def test_execute_result_includes_telemetry_dict(self):
        """Test that cli_based_tool result includes _telemetry for span enrichment."""
        plugin = CLIToolPlugin()
        plugin.initialize()

        result = plugin._execute({"command": "echo hello"})

        assert "_telemetry" in result
        telem = result["_telemetry"]

        expected_keys = [
            "jaato.cli.command",
            "jaato.cli.returncode",
            "jaato.cli.stdout_bytes",
            "jaato.cli.stderr_bytes",
            "jaato.cli.shell_mode",
            "jaato.cli.cwd",
        ]
        for key in expected_keys:
            assert key in telem, f"Expected key {key!r} missing from _telemetry"

        assert telem["jaato.cli.returncode"] == 0
        assert telem["jaato.cli.command"].startswith("echo")


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

    def test_extract_path_tokens_covers_every_segment(self):
        """Paths in later segments and substitutions are extracted too."""
        plugin = CLIToolPlugin()

        tokens = plugin._extract_path_tokens("cat ./a && rm /etc/b")
        assert tokens == ["./a", "/etc/b"]

        assert plugin._extract_path_tokens("echo $(cat /etc/passwd)") == [
            "/etc/passwd"
        ]

    def test_extract_path_tokens_includes_redirect_targets(self):
        """Redirect targets are paths; fd duplication targets are not."""
        plugin = CLIToolPlugin()

        assert plugin._extract_path_tokens("echo hi 2>/etc/x") == ["/etc/x"]
        assert plugin._extract_path_tokens("echo hi >&2") == []

    def test_extract_path_tokens_fails_closed(self):
        """There is no naive-split fallback for an unparseable command."""
        from shared.command_analysis import UnanalyzableCommand

        plugin = CLIToolPlugin()
        with pytest.raises(UnanalyzableCommand):
            plugin._extract_path_tokens('cat "/etc/passwd')

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
        """Test validation returns a not-found result for paths outside workspace."""
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        result = plugin._validate_command_paths("cat /etc/passwd")
        # New contract: a ready-to-return result dict mimicking "not found".
        assert isinstance(result, dict)
        assert result["returncode"] == 1
        assert "No such file or directory" in result["stderr"]
        assert "/etc/passwd" in result["stderr"]

    def test_validate_command_paths_unparseable_fails_closed(self, tmp_path):
        """An un-tokenisable command is refused, not degraded to str.split().

        Unbalanced quotes make ``shlex.split`` raise; the validator must
        refuse rather than fall back to a naive split that parses
        differently than the shell (which could let an out-of-workspace
        path slip through).
        """
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        result = plugin._validate_command_paths('cat "/etc/passwd')
        assert isinstance(result, dict)
        assert result["returncode"] == 2
        assert "could not be parsed" in result["stderr"]
        assert result["stdout"] == ""

    def test_validate_command_paths_unparseable_no_sandbox_allowed(self, monkeypatch):
        """Without a workspace_root the fail-closed gate does not apply."""
        monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
        monkeypatch.delenv("workspaceRoot", raising=False)

        plugin = CLIToolPlugin()
        plugin.initialize()

        # No sandboxing configured -> validator is a no-op even for
        # commands shlex can't parse.
        assert plugin._validate_command_paths('cat "/etc/passwd') is None

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

        # Path in arg_list should be validated -> not-found result dict
        result = plugin._validate_command_paths("cat", arg_list=["/etc/passwd"])
        assert isinstance(result, dict)
        assert result["returncode"] == 1
        assert "/etc/passwd" in result["stderr"]

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


class TestCLIPluginRuntimeLimits:
    """Tests for the per-session runtime-limits wiring (cgroup attach +
    app-layer caps).  See ``set_runtime_limits`` and the Popen branches
    in ``CLIToolPlugin``.
    """

    def test_default_state_has_no_attach_no_limits(self):
        # Plugins start with no per-session limits; the Popen branches
        # must treat this as "behave like before".
        plugin = CLIToolPlugin()
        assert plugin._cgroup_attach is None
        assert plugin._runtime_limits is None

    def test_set_runtime_limits_stores_values(self):
        from shared.runtime_limits import RuntimeLimits

        attach_calls = []

        def fake_attach():
            attach_calls.append(True)

        plugin = CLIToolPlugin()
        limits = RuntimeLimits(memory_max_mb=512, tool_timeout_seconds=30)
        plugin.set_runtime_limits(fake_attach, limits)

        assert plugin._cgroup_attach is fake_attach
        assert plugin._runtime_limits is limits
        # Sanity: the attach is callable with no args (the preexec_fn
        # contract) and our fake records the invocation.
        plugin._cgroup_attach()
        assert attach_calls == [True]

    def test_clear_runtime_limits_with_none(self):
        from shared.runtime_limits import RuntimeLimits

        plugin = CLIToolPlugin()
        plugin.set_runtime_limits(lambda: None, RuntimeLimits(memory_max_mb=128))
        plugin.set_runtime_limits(None, None)
        assert plugin._cgroup_attach is None
        assert plugin._runtime_limits is None


class TestCLIPluginCompoundCommandAnalysis:
    """Mode inference must hold for every segment, not just the first.

    Regression suite for the analyzer-bypass class catalogued in issue #668:
    a command string is judged by what *all* of it does, so a write hidden
    behind ``&&``, a redirect, or a command substitution is classified as a
    write rather than inheriting the leading command's read semantics.
    """

    @pytest.mark.parametrize("command,expected", [
        # --- compound-command segmentation -----------------------------
        pytest.param(
            "cat ./README.md && rm -rf ./notes",
            [("./README.md", "read"), ("./notes", "write")],
            id="and-chain",
        ),
        pytest.param(
            "cat ./a || rm ./b",
            [("./a", "read"), ("./b", "write")],
            id="or-chain",
        ),
        pytest.param(
            "cat ./a; rm ./b",
            [("./a", "read"), ("./b", "write")],
            id="semicolon",
        ),
        pytest.param(
            "cat ./a | tee ./b",
            [("./a", "read"), ("./b", "write")],
            id="pipeline",
        ),
        pytest.param(
            "cat ./a & rm ./b",
            [("./a", "read"), ("./b", "write")],
            id="background",
        ),
        pytest.param(
            "cat ./a\nrm ./b",
            [("./a", "read"), ("./b", "write")],
            id="newline",
        ),
        pytest.param(
            "( rm ./b )",
            [("./b", "write")],
            id="subshell",
        ),
        pytest.param(
            "cat ./a; cp ./b ./c",
            [("./a", "read"), ("./b", "read"), ("./c", "write")],
            id="write-last-per-segment",
        ),
        # --- redirection grammar ---------------------------------------
        pytest.param("echo hi > ./out", [("./out", "write")], id="redirect-stdout"),
        pytest.param("echo hi >> ./out", [("./out", "write")], id="redirect-append"),
        pytest.param("echo hi 2>./out", [("./out", "write")], id="redirect-fd"),
        pytest.param("echo hi 1>./out", [("./out", "write")], id="redirect-fd-1"),
        pytest.param("echo hi &>./out", [("./out", "write")], id="redirect-both"),
        pytest.param("echo hi &>>./out", [("./out", "write")], id="redirect-both-append"),
        pytest.param("echo hi >&./out", [("./out", "write")], id="redirect-dup-to-file"),
        pytest.param("echo hi >|./out", [("./out", "write")], id="redirect-clobber"),
        pytest.param("echo hi <>./out", [("./out", "write")], id="redirect-rw"),
        pytest.param("cat < ./in", [("./in", "read")], id="redirect-stdin"),
        pytest.param("cat ./a >&2", [("./a", "read")], id="redirect-fd-dup-not-a-path"),
        pytest.param(
            "cp ./a ./b > ./log",
            [("./a", "read"), ("./b", "write"), ("./log", "write")],
            id="write-last-plus-redirect",
        ),
        # --- command substitution --------------------------------------
        pytest.param(
            "echo `rm -rf ./notes`",
            [("./notes", "write")],
            id="backtick-substitution",
        ),
        pytest.param(
            'echo "$(tee ./out)"',
            [("./out", "write")],
            id="dollar-paren-substitution",
        ),
        pytest.param(
            "diff <(cat ./a) <(rm ./b)",
            [("./a", "read"), ("./b", "write")],
            id="process-substitution",
        ),
        # --- CC checklist rows -----------------------------------------
        pytest.param(
            "FOO=bar rm ./notes",
            [("./notes", "write")],
            id="inline-env-assignment",
        ),
        pytest.param(
            "OPTIND=1 RANDOM=2 rm ./notes",
            [("./notes", "write")],
            id="multiple-assignments",
        ),
        pytest.param(
            "\\rm -rf ./notes",
            [("./notes", "write")],
            id="backslash-escaped-command",
        ),
        pytest.param(
            "cat ./a && \\\nrm ./b",
            [("./a", "read"), ("./b", "write")],
            id="line-continuation",
        ),
        pytest.param(
            "sudo rm ./notes",
            [("./notes", "write")],
            id="wrapper-command",
        ),
        pytest.param(
            "env FOO=bar rm ./notes",
            [("./notes", "write")],
            id="env-wrapper",
        ),
        pytest.param(
            "man -P ./pager ls",
            [("./pager", "read")],
            id="pager-option-value-is-a-path",
        ),
        pytest.param(
            "cat <<EOF\n./not-a-real-path\nEOF",
            [],
            id="heredoc-body-is-data",
        ),
    ])
    def test_classify_path_modes(self, command, expected):
        plugin = CLIToolPlugin()
        assert plugin._classify_path_modes(command) == expected

    def test_write_wins_when_a_path_is_read_and_written(self):
        """A path used both ways unions to the stricter mode."""
        plugin = CLIToolPlugin()
        assert plugin._classify_path_modes("cat ./a && rm ./a") == [("./a", "write")]

    def test_long_command_still_sees_the_hidden_segment(self):
        """A 10k-character command is analysed, not waved through."""
        plugin = CLIToolPlugin()
        filler = " ".join(["x"] * 4000)
        command = f"echo {filler} && rm -rf ./notes"
        assert ("./notes", "write") in plugin._classify_path_modes(command)


class TestCLIPluginAnalyzerFailsClosed:
    """Commands the analyzer cannot model are refused, not degraded."""

    @pytest.mark.parametrize("command", [
        pytest.param('cat "/etc/passwd', id="unbalanced-double-quote"),
        pytest.param("cat '/etc/passwd", id="unbalanced-single-quote"),
        pytest.param("echo $(rm -rf /", id="unbalanced-substitution"),
        pytest.param("echo `rm -rf /", id="unbalanced-backtick"),
        pytest.param("cat /etc/passwd \\", id="dangling-backslash"),
        pytest.param("cat >", id="redirect-without-target"),
        pytest.param("cat /etc/passwd &&", id="dangling-and"),
        pytest.param("cat /etc/passwd ||", id="dangling-or"),
    ])
    def test_refused_with_syntax_error(self, tmp_path, command):
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        result = plugin._validate_command_paths(command)
        assert isinstance(result, dict)
        assert result["returncode"] == 2
        assert "could not be parsed" in result["stderr"]

    @pytest.mark.parametrize("command", [
        pytest.param("ls && cat /etc/passwd", id="hidden-behind-and"),
        pytest.param("ls; cat /etc/passwd", id="hidden-behind-semicolon"),
        pytest.param("echo hi > /etc/jaato-probe", id="redirect-outside"),
        pytest.param("echo hi 2> /etc/jaato-probe", id="fd-redirect-outside"),
        pytest.param("echo hi &> /etc/jaato-probe", id="both-redirect-outside"),
        pytest.param("echo $(cat /etc/passwd)", id="substitution-outside"),
        pytest.param("echo `cat /etc/passwd`", id="backtick-outside"),
        pytest.param("FOO=bar cat /etc/passwd", id="assignment-prefix-outside"),
    ])
    def test_out_of_workspace_path_blocked_in_any_position(self, tmp_path, command):
        plugin = CLIToolPlugin()
        plugin.initialize({"workspace_root": str(tmp_path)})

        result = plugin._validate_command_paths(command)
        assert isinstance(result, dict), f"not blocked: {command}"
        assert result["returncode"] == 1
        assert "No such file or directory" in result["stderr"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pseudo-devices")
class TestRedirectToPseudoDevices:
    """``2>/dev/null`` must survive the path sandbox (jaato issue #784).

    The plugin is configured here exactly as a runner session configures
    it -- ``set_workspace_path`` on the session workspace -- because that
    is the only tier where the defect showed.  A test driving
    ``shared.subprocess_runner.run_command`` directly passes on the broken
    build and proves nothing: ``run_command`` has no path sandbox, so it
    honoured the redirect all along.

    What actually failed: ``_classify_path_modes`` reads ``2>/dev/null``
    as a *write* redirection whose target is outside the workspace, so
    ``_validate_command_paths`` refused the command before it ever
    reached a shell and synthesised
    ``<cmd>: /dev/null: No such file or directory``.  The program-name
    prefix comes from ``_make_not_found_result``, not from the program --
    which is why the observed inference was "this sandbox has no
    /dev/null" rather than "my command was refused".
    """

    def _plugin(self, workspace):
        plugin = CLIToolPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(workspace))
        return plugin

    def test_stderr_redirect_is_not_refused(self, tmp_path):
        """The whole defect, at the tier where it happened."""
        plugin = self._plugin(tmp_path)

        result = plugin._execute({"command": "ls does-not-exist 2>/dev/null"})

        # The redirect was applied by a real shell: ls's complaint went to
        # /dev/null, so nothing reaches us.
        assert result["stderr"] == ""
        assert "/dev/null" not in result.get("stderr", "")
        assert "error" not in result

    @pytest.mark.parametrize("command", [
        "find . -type d -name scaffold 2>/dev/null",
        "ls tests/ 2>/dev/null || ls .",
        "cat missing.txt 2>/dev/null; echo ok",
        "ls -la . 2>/dev/null | head -20",
        "echo hi >/dev/null",
        "cat /dev/null",
        "echo hi 2>&1 >/dev/null",
    ])
    def test_shapes_from_the_report_are_allowed(self, tmp_path, command):
        """Every failing shape recorded on the issue, plus its relatives."""
        assert self._plugin(tmp_path)._validate_command_paths(command) is None

    def test_out_of_workspace_targets_are_still_refused(self, tmp_path):
        """The allowance is for pseudo-devices, not for redirects at large."""
        plugin = self._plugin(tmp_path)

        refusal = plugin._validate_command_paths("echo x > /etc/cron.d/pwn")

        assert refusal is not None
        assert refusal["returncode"] == 1
        assert "/etc/cron.d/pwn" in refusal["stderr"]

    def test_block_devices_are_still_refused(self, tmp_path):
        """``/dev/`` is not blanket-allowed -- only the pseudo-devices are."""
        plugin = self._plugin(tmp_path)

        assert plugin._validate_command_paths("echo x > /dev/sda") is not None
        assert plugin._validate_command_paths("cat /dev/mem") is not None
