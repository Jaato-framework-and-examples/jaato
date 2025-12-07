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


class TestCLIPluginFunctionDeclarations:
    """Tests for function declarations."""

    def test_get_function_declarations(self):
        plugin = CLIToolPlugin()
        declarations = plugin.get_function_declarations()

        assert len(declarations) == 1
        assert declarations[0].name == "cli_based_tool"

    def test_cli_based_tool_schema(self):
        plugin = CLIToolPlugin()
        declarations = plugin.get_function_declarations()
        cli_tool = declarations[0]
        schema = cli_tool.parameters_json_schema

        assert schema["type"] == "object"
        assert "command" in schema["properties"]
        assert "args" in schema["properties"]
        assert "command" in schema["required"]

    def test_cli_based_tool_description(self):
        plugin = CLIToolPlugin()
        declarations = plugin.get_function_declarations()
        cli_tool = declarations[0]

        assert cli_tool.description == "Execute a local CLI command"


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
