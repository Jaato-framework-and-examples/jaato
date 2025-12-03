"""Tests for the CLI tool plugin."""

import os
import sys
import pytest

from ..plugin import CLIToolPlugin, create_plugin


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
