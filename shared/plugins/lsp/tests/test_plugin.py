"""Tests for LSP plugin."""

import pytest
from ..plugin import LSPToolPlugin, create_plugin
from ..lsp_client import Position, Range, Location, Diagnostic, ServerConfig


class TestLSPTypes:
    """Test LSP data types."""

    def test_position_to_dict(self):
        pos = Position(line=5, character=10)
        assert pos.to_dict() == {"line": 5, "character": 10}

    def test_position_from_dict(self):
        pos = Position.from_dict({"line": 5, "character": 10})
        assert pos.line == 5
        assert pos.character == 10

    def test_range_to_dict(self):
        r = Range(
            start=Position(line=1, character=0),
            end=Position(line=1, character=10)
        )
        assert r.to_dict() == {
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 10}
        }

    def test_location_from_dict(self):
        loc = Location.from_dict({
            "uri": "file:///test.py",
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 5}
            }
        })
        assert loc.uri == "file:///test.py"
        assert loc.range.start.line == 0

    def test_diagnostic_severity_names(self):
        d1 = Diagnostic(
            range=Range(Position(0, 0), Position(0, 5)),
            message="Error",
            severity=1
        )
        assert d1.severity_name == "Error"

        d2 = Diagnostic(
            range=Range(Position(0, 0), Position(0, 5)),
            message="Warning",
            severity=2
        )
        assert d2.severity_name == "Warning"


class TestServerConfig:
    """Test server configuration."""

    def test_basic_config(self):
        config = ServerConfig(
            name="python",
            command="pyright-langserver",
            args=["--stdio"]
        )
        assert config.name == "python"
        assert config.command == "pyright-langserver"
        assert config.args == ["--stdio"]

    def test_config_with_env(self):
        config = ServerConfig(
            name="typescript",
            command="typescript-language-server",
            args=["--stdio"],
            env={"NODE_OPTIONS": "--max-old-space-size=4096"}
        )
        assert config.env["NODE_OPTIONS"] == "--max-old-space-size=4096"


class TestLSPToolPlugin:
    """Test LSP plugin interface."""

    def test_create_plugin(self):
        plugin = create_plugin()
        assert plugin is not None
        assert plugin.name == "lsp"

    def test_plugin_name(self):
        plugin = LSPToolPlugin()
        assert plugin.name == "lsp"

    def test_get_tool_schemas(self):
        plugin = LSPToolPlugin()
        # Don't initialize - just check schema definitions
        plugin._initialized = True  # Skip actual initialization
        schemas = plugin.get_tool_schemas()

        assert len(schemas) >= 7
        names = {s.name for s in schemas}
        assert "lsp_goto_definition" in names
        assert "lsp_find_references" in names
        assert "lsp_hover" in names
        assert "lsp_get_diagnostics" in names
        assert "lsp_document_symbols" in names
        assert "lsp_workspace_symbols" in names
        assert "lsp_rename_symbol" in names

    def test_get_executors(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True
        executors = plugin.get_executors()

        assert "lsp_goto_definition" in executors
        assert "lsp_find_references" in executors
        assert "lsp" in executors  # User command

    def test_get_user_commands(self):
        plugin = LSPToolPlugin()
        commands = plugin.get_user_commands()

        assert len(commands) == 1
        assert commands[0].name == "lsp"

    def test_auto_approved_tools(self):
        plugin = LSPToolPlugin()
        approved = plugin.get_auto_approved_tools()

        # Read-only tools should be auto-approved
        assert "lsp_goto_definition" in approved
        assert "lsp_find_references" in approved
        assert "lsp_hover" in approved
        # rename is not auto-approved (modifies files)
        assert "lsp_rename_symbol" not in approved

    def test_system_instructions(self):
        plugin = LSPToolPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "lsp_goto_definition" in instructions
        assert "lsp_find_references" in instructions

    def test_command_completions(self):
        plugin = LSPToolPlugin()
        completions = plugin.get_command_completions("lsp", [])

        assert len(completions) > 0
        values = {c.value for c in completions}
        assert "list" in values
        assert "status" in values
        assert "connect" in values

    def test_command_completions_partial(self):
        plugin = LSPToolPlugin()
        completions = plugin.get_command_completions("lsp", ["st"])

        values = {c.value for c in completions}
        assert "status" in values
        assert "list" not in values  # doesn't start with "st"

    def test_execute_help_command(self):
        plugin = LSPToolPlugin()
        result = plugin.execute_user_command("lsp", {"subcommand": "help"})

        assert "lsp list" in result
        assert "lsp status" in result
        assert "lsp connect" in result

    def test_execute_list_no_config(self):
        plugin = LSPToolPlugin()
        plugin._config_cache = {}
        result = plugin.execute_user_command("lsp", {"subcommand": "list"})

        assert "No LSP servers configured" in result


class TestLSPClient:
    """Test LSP client utilities."""

    def test_uri_from_path_unix(self):
        from ..lsp_client import LSPClient, ServerConfig
        import sys

        config = ServerConfig(name="test", command="test")
        client = LSPClient(config)

        if sys.platform != 'win32':
            uri = client.uri_from_path("/home/user/test.py")
            assert uri == "file:///home/user/test.py"

    def test_guess_language_id(self):
        from ..lsp_client import LSPClient, ServerConfig

        config = ServerConfig(name="test", command="test")
        client = LSPClient(config)

        assert client._guess_language_id("test.py") == "python"
        assert client._guess_language_id("test.js") == "javascript"
        assert client._guess_language_id("test.ts") == "typescript"
        assert client._guess_language_id("test.go") == "go"
        assert client._guess_language_id("test.rs") == "rust"
        assert client._guess_language_id("test.unknown") == "plaintext"
