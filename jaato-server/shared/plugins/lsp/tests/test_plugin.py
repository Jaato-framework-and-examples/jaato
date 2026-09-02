"""Tests for LSP plugin."""

import asyncio
import json
import os
import pytest
import signal
import sys
import tempfile
import threading
import queue
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from ..plugin import LSPToolPlugin, create_plugin, LogEntry, LOG_INFO, LOG_ERROR
from jaato_sdk.plugins.base import HelpLines
from ..lsp_client import (
    Position, Range, Location, Diagnostic, CompletionItem, Hover,
    SymbolInformation, ServerCapabilities, ServerConfig, LSPClient
)


# =============================================================================
# LSP Type Tests
# =============================================================================

class TestPosition:
    """Test Position dataclass."""

    def test_to_dict(self):
        pos = Position(line=5, character=10)
        assert pos.to_dict() == {"line": 5, "character": 10}

    def test_from_dict(self):
        pos = Position.from_dict({"line": 5, "character": 10})
        assert pos.line == 5
        assert pos.character == 10

    def test_zero_position(self):
        pos = Position(line=0, character=0)
        assert pos.to_dict() == {"line": 0, "character": 0}


class TestRange:
    """Test Range dataclass."""

    def test_to_dict(self):
        r = Range(
            start=Position(line=1, character=0),
            end=Position(line=1, character=10)
        )
        assert r.to_dict() == {
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 10}
        }

    def test_from_dict(self):
        r = Range.from_dict({
            "start": {"line": 5, "character": 2},
            "end": {"line": 10, "character": 15}
        })
        assert r.start.line == 5
        assert r.start.character == 2
        assert r.end.line == 10
        assert r.end.character == 15

    def test_single_line_range(self):
        r = Range(Position(0, 5), Position(0, 10))
        d = r.to_dict()
        assert d["start"]["line"] == d["end"]["line"]


class TestLocation:
    """Test Location dataclass."""

    def test_from_dict(self):
        loc = Location.from_dict({
            "uri": "file:///home/user/test.py",
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 5}
            }
        })
        assert loc.uri == "file:///home/user/test.py"
        assert loc.range.start.line == 0
        assert loc.range.end.character == 5

    def test_windows_uri(self):
        loc = Location.from_dict({
            "uri": "file:///C:/Users/test/file.py",
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}}
        })
        assert "C:" in loc.uri


class TestDiagnostic:
    """Test Diagnostic dataclass."""

    def test_from_dict_minimal(self):
        d = Diagnostic.from_dict({
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
            "message": "Undefined variable 'x'"
        })
        assert d.message == "Undefined variable 'x'"
        assert d.severity == 1  # Default

    def test_from_dict_full(self):
        d = Diagnostic.from_dict({
            "range": {"start": {"line": 10, "character": 4}, "end": {"line": 10, "character": 8}},
            "message": "Unused import",
            "severity": 2,
            "source": "pyright",
            "code": "reportUnusedImport"
        })
        assert d.message == "Unused import"
        assert d.severity == 2
        assert d.source == "pyright"
        assert d.code == "reportUnusedImport"

    def test_severity_names(self):
        assert Diagnostic(Range(Position(0,0), Position(0,0)), "e", severity=1).severity_name == "Error"
        assert Diagnostic(Range(Position(0,0), Position(0,0)), "w", severity=2).severity_name == "Warning"
        assert Diagnostic(Range(Position(0,0), Position(0,0)), "i", severity=3).severity_name == "Info"
        assert Diagnostic(Range(Position(0,0), Position(0,0)), "h", severity=4).severity_name == "Hint"
        assert Diagnostic(Range(Position(0,0), Position(0,0)), "?", severity=99).severity_name == "Unknown"


class TestCompletionItem:
    """Test CompletionItem dataclass."""

    def test_from_dict_minimal(self):
        item = CompletionItem.from_dict({"label": "print"})
        assert item.label == "print"
        assert item.kind is None

    def test_from_dict_full(self):
        item = CompletionItem.from_dict({
            "label": "print",
            "kind": 3,  # Function
            "detail": "def print(*args, **kwargs)",
            "documentation": "Print objects to the text stream file."
        })
        assert item.label == "print"
        assert item.kind == 3
        assert item.detail == "def print(*args, **kwargs)"
        assert "Print objects" in item.documentation

    def test_from_dict_markdown_doc(self):
        item = CompletionItem.from_dict({
            "label": "func",
            "documentation": {"kind": "markdown", "value": "**Bold** docs"}
        })
        assert item.documentation == "**Bold** docs"


class TestHover:
    """Test Hover dataclass."""

    def test_from_dict_string_contents(self):
        hover = Hover.from_dict({"contents": "def foo() -> int"})
        assert hover.contents == "def foo() -> int"
        assert hover.range is None

    def test_from_dict_markup_contents(self):
        hover = Hover.from_dict({
            "contents": {"kind": "markdown", "value": "```python\ndef foo():\n    pass\n```"}
        })
        assert "def foo()" in hover.contents

    def test_from_dict_list_contents(self):
        hover = Hover.from_dict({
            "contents": [
                {"language": "python", "value": "def foo()"},
                "A function that does foo."
            ]
        })
        assert "def foo()" in hover.contents
        assert "A function" in hover.contents

    def test_from_dict_with_range(self):
        hover = Hover.from_dict({
            "contents": "info",
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}}
        })
        assert hover.range is not None
        assert hover.range.end.character == 3


class TestSymbolInformation:
    """Test SymbolInformation dataclass."""

    def test_from_dict(self):
        sym = SymbolInformation.from_dict({
            "name": "MyClass",
            "kind": 5,  # Class
            "location": {
                "uri": "file:///test.py",
                "range": {"start": {"line": 10, "character": 0}, "end": {"line": 50, "character": 0}}
            },
            "containerName": "mymodule"
        })
        assert sym.name == "MyClass"
        assert sym.kind == 5
        assert sym.kind_name == "Class"
        assert sym.container_name == "mymodule"

    def test_kind_names(self):
        base = {
            "name": "x",
            "location": {"uri": "file:///x", "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}}}
        }
        assert SymbolInformation.from_dict({**base, "kind": 1}).kind_name == "File"
        assert SymbolInformation.from_dict({**base, "kind": 5}).kind_name == "Class"
        assert SymbolInformation.from_dict({**base, "kind": 6}).kind_name == "Method"
        assert SymbolInformation.from_dict({**base, "kind": 12}).kind_name == "Function"
        assert SymbolInformation.from_dict({**base, "kind": 13}).kind_name == "Variable"
        assert SymbolInformation.from_dict({**base, "kind": 999}).kind_name == "Unknown(999)"


class TestServerCapabilities:
    """Test ServerCapabilities dataclass."""

    def test_from_dict_empty(self):
        caps = ServerCapabilities.from_dict({})
        assert caps.hover is False
        assert caps.definition is False
        assert caps.diagnostics is True  # Always assumed

    def test_from_dict_full(self):
        caps = ServerCapabilities.from_dict({
            "hoverProvider": True,
            "completionProvider": {"triggerCharacters": ["."]},
            "definitionProvider": True,
            "referencesProvider": True,
            "documentSymbolProvider": True,
            "workspaceSymbolProvider": True,
            "renameProvider": True,
            "codeActionProvider": True
        })
        assert caps.hover is True
        assert caps.completion is True
        assert caps.definition is True
        assert caps.references is True
        assert caps.document_symbol is True
        assert caps.workspace_symbol is True
        assert caps.rename is True
        assert caps.code_action is True


class TestServerConfig:
    """Test ServerConfig dataclass."""

    def test_basic_config(self):
        config = ServerConfig(
            name="python",
            command="pyright-langserver",
            args=["--stdio"]
        )
        assert config.name == "python"
        assert config.command == "pyright-langserver"
        assert config.args == ["--stdio"]
        assert config.env is None
        assert config.root_uri is None

    def test_full_config(self):
        config = ServerConfig(
            name="typescript",
            command="typescript-language-server",
            args=["--stdio"],
            env={"NODE_OPTIONS": "--max-old-space-size=4096"},
            root_uri="file:///home/user/project",
            language_id="typescript"
        )
        assert config.env["NODE_OPTIONS"] == "--max-old-space-size=4096"
        assert config.root_uri == "file:///home/user/project"
        assert config.language_id == "typescript"


# =============================================================================
# LSP Client Tests
# =============================================================================

class TestLSPClientUtilities:
    """Test LSP client utility methods."""

    def test_uri_from_path_unix(self):
        config = ServerConfig(name="test", command="test")
        client = LSPClient(config)

        # Test with absolute path
        uri = client.uri_from_path("/home/user/test.py")
        assert uri.startswith("file://")
        assert "test.py" in uri

    def test_guess_language_id(self):
        config = ServerConfig(name="test", command="test")
        client = LSPClient(config)

        # Python
        assert client._guess_language_id("test.py") == "python"
        assert client._guess_language_id("/path/to/module.py") == "python"

        # JavaScript/TypeScript
        assert client._guess_language_id("app.js") == "javascript"
        assert client._guess_language_id("app.ts") == "typescript"
        assert client._guess_language_id("component.tsx") == "typescriptreact"
        assert client._guess_language_id("component.jsx") == "javascriptreact"

        # Systems languages
        assert client._guess_language_id("main.go") == "go"
        assert client._guess_language_id("lib.rs") == "rust"
        assert client._guess_language_id("Main.java") == "java"
        assert client._guess_language_id("main.c") == "c"
        assert client._guess_language_id("main.cpp") == "cpp"

        # Other
        assert client._guess_language_id("config.json") == "json"
        assert client._guess_language_id("config.yaml") == "yaml"
        assert client._guess_language_id("index.html") == "html"
        assert client._guess_language_id("styles.css") == "css"
        assert client._guess_language_id("README.md") == "markdown"

        # Unknown
        assert client._guess_language_id("file.xyz") == "plaintext"
        assert client._guess_language_id("noextension") == "plaintext"


# =============================================================================
# Plugin Tests
# =============================================================================

class TestLSPToolPluginBasics:
    """Test LSP plugin basic interface."""

    def test_create_plugin(self):
        plugin = create_plugin()
        assert plugin is not None
        assert isinstance(plugin, LSPToolPlugin)

    def test_plugin_name(self):
        plugin = LSPToolPlugin()
        assert plugin.name == "lsp"

    def test_get_tool_schemas_without_init(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True  # Skip actual initialization
        schemas = plugin.get_tool_schemas()

        assert len(schemas) >= 7
        names = {s.name for s in schemas}

        # Check all expected tools
        expected_tools = {
            "lsp_goto_definition",
            "lsp_find_references",
            "lsp_hover",
            "lsp_get_diagnostics",
            "lsp_document_symbols",
            "lsp_workspace_symbols",
            "lsp_rename_symbol"
        }
        assert expected_tools.issubset(names)

    def test_tool_schema_structure(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True
        schemas = plugin.get_tool_schemas()

        # Find goto_definition schema - now uses symbol-based API
        goto_def = next(s for s in schemas if s.name == "lsp_goto_definition")

        assert goto_def.description
        assert "definition" in goto_def.description.lower()
        assert goto_def.parameters["type"] == "object"
        assert "symbol" in goto_def.parameters["properties"]
        assert "file_path" in goto_def.parameters["properties"]  # Optional for disambiguation
        assert set(goto_def.parameters["required"]) == {"symbol"}

    def test_symbol_based_tools(self):
        """Test that symbol-based tools have correct schema structure."""
        plugin = LSPToolPlugin()
        plugin._initialized = True
        schemas = plugin.get_tool_schemas()

        # Symbol-based tools should require 'symbol' not 'line'/'character'
        symbol_tools = ["lsp_goto_definition", "lsp_find_references", "lsp_hover", "lsp_rename_symbol"]

        for tool_name in symbol_tools:
            schema = next(s for s in schemas if s.name == tool_name)
            assert "symbol" in schema.parameters["properties"], f"{tool_name} should have 'symbol' parameter"
            assert "symbol" in schema.parameters.get("required", []), f"{tool_name} should require 'symbol'"
            assert "line" not in schema.parameters.get("required", []), f"{tool_name} should not require 'line'"
            assert "character" not in schema.parameters.get("required", []), f"{tool_name} should not require 'character'"

    def test_get_executors(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True
        executors = plugin.get_executors()

        assert "lsp_goto_definition" in executors
        assert "lsp_find_references" in executors
        assert "lsp_hover" in executors
        assert "lsp_get_diagnostics" in executors
        assert "lsp_document_symbols" in executors
        assert "lsp_workspace_symbols" in executors
        assert "lsp_rename_symbol" in executors
        assert "lsp" in executors  # User command

        # Check executors are callable
        for name, executor in executors.items():
            assert callable(executor)

    def test_get_user_commands(self):
        plugin = LSPToolPlugin()
        commands = plugin.get_user_commands()

        assert len(commands) == 1
        cmd = commands[0]
        assert cmd.name == "lsp"
        # The `lsp` command is OPERATOR-facing (list/status/connect/reload)
        # and is explicitly declared share_with_model=False in the plugin --
        # its output is server management chatter, not model context.
        assert cmd.share_with_model is False
        assert cmd.parameters is not None
        assert len(cmd.parameters) == 2

    def test_auto_approved_tools(self):
        plugin = LSPToolPlugin()
        approved = plugin.get_auto_approved_tools()

        # Read-only tools should be auto-approved
        assert "lsp_goto_definition" in approved
        assert "lsp_find_references" in approved
        assert "lsp_hover" in approved
        assert "lsp_get_diagnostics" in approved
        assert "lsp_document_symbols" in approved
        assert "lsp_workspace_symbols" in approved
        assert "lsp" in approved  # User command

        # Rename modifies files - should NOT be auto-approved
        assert "lsp_rename_symbol" not in approved

    def test_system_instructions(self):
        plugin = LSPToolPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "lsp_goto_definition" in instructions
        assert "lsp_find_references" in instructions
        # Symbol-based API - should mention symbol parameter
        assert "symbol" in instructions.lower()
        # Diagnostics recommendation
        assert "lsp_get_diagnostics" in instructions


class TestLSPToolPluginCommands:
    """Test LSP plugin user commands."""

    def test_command_completions_empty(self):
        plugin = LSPToolPlugin()
        completions = plugin.get_command_completions("lsp", [])

        assert len(completions) > 0
        values = {c.value for c in completions}
        assert "list" in values
        assert "status" in values
        assert "connect" in values
        assert "disconnect" in values
        assert "reload" in values
        assert "logs" in values
        assert "help" in values

    def test_command_completions_partial(self):
        plugin = LSPToolPlugin()

        # "st" should match "status"
        completions = plugin.get_command_completions("lsp", ["st"])
        values = {c.value for c in completions}
        assert "status" in values
        assert "list" not in values

        # "co" should match "connect"
        completions = plugin.get_command_completions("lsp", ["co"])
        values = {c.value for c in completions}
        assert "connect" in values

    def test_command_completions_wrong_command(self):
        plugin = LSPToolPlugin()
        completions = plugin.get_command_completions("other", [])
        assert completions == []

    def test_execute_help_command(self):
        plugin = LSPToolPlugin()
        result = plugin.execute_user_command("lsp", {"subcommand": "help"})

        # Help now returns HelpLines for pager display
        assert isinstance(result, HelpLines)
        help_text = "\n".join(text for text, _ in result.lines)
        assert "lsp" in help_text.lower()
        assert "list" in help_text.lower()
        assert "status" in help_text.lower()
        assert "connect" in help_text.lower()
        assert "disconnect" in help_text.lower()
        assert "reload" in help_text.lower()
        assert ".lsp.json" in help_text

    def test_execute_empty_subcommand(self):
        plugin = LSPToolPlugin()
        result = plugin.execute_user_command("lsp", {"subcommand": ""})
        # Should show help (HelpLines)
        assert isinstance(result, HelpLines)
        help_text = "\n".join(text for text, _ in result.lines)
        assert "list" in help_text.lower()

    def test_execute_unknown_subcommand(self):
        plugin = LSPToolPlugin()
        result = plugin.execute_user_command("lsp", {"subcommand": "unknown"})
        assert isinstance(result, str)
        assert "Unknown subcommand" in result
        assert "lsp help" in result  # Suggests using help command

    def test_execute_list_no_config(self):
        plugin = LSPToolPlugin()
        plugin._config_cache = {}
        result = plugin.execute_user_command("lsp", {"subcommand": "list"})
        assert "No LSP servers configured" in result

    def test_execute_list_with_servers(self):
        plugin = LSPToolPlugin()
        plugin._config_cache = {
            "languageServers": {
                "python": {"command": "pyright-langserver", "args": ["--stdio"]},
                "typescript": {"command": "typescript-language-server"}
            }
        }
        result = plugin.execute_user_command("lsp", {"subcommand": "list"})
        assert "python" in result
        assert "typescript" in result
        assert "pyright-langserver" in result

    def test_execute_status_no_config(self):
        plugin = LSPToolPlugin()
        plugin._config_cache = {}
        result = plugin.execute_user_command("lsp", {"subcommand": "status"})
        assert "No LSP servers configured" in result

    def test_execute_connect_no_name(self):
        plugin = LSPToolPlugin()
        result = plugin.execute_user_command("lsp", {"subcommand": "connect", "rest": ""})
        assert "Usage:" in result

    def test_execute_connect_unknown_server(self):
        plugin = LSPToolPlugin()
        plugin._config_cache = {"languageServers": {}}
        plugin._initialized = True
        plugin._request_queue = queue.Queue()
        plugin._response_queue = queue.Queue()

        result = plugin.execute_user_command("lsp", {"subcommand": "connect", "rest": "unknown"})
        assert "not found" in result

    def test_execute_disconnect_no_name(self):
        plugin = LSPToolPlugin()
        result = plugin.execute_user_command("lsp", {"subcommand": "disconnect", "rest": ""})
        assert "Usage:" in result

    def test_execute_disconnect_not_connected(self):
        plugin = LSPToolPlugin()
        plugin._connected_servers = set()
        result = plugin.execute_user_command("lsp", {"subcommand": "disconnect", "rest": "python"})
        assert "not connected" in result

    def test_execute_logs_empty(self):
        plugin = LSPToolPlugin()
        result = plugin.execute_user_command("lsp", {"subcommand": "logs", "rest": ""})
        assert "No log entries" in result

    def test_execute_logs_clear(self):
        plugin = LSPToolPlugin()
        plugin._log_event(LOG_INFO, "Test event")
        assert len(plugin._log) > 0

        result = plugin.execute_user_command("lsp", {"subcommand": "logs", "rest": "clear"})
        assert "cleared" in result.lower()
        assert len(plugin._log) == 0

    def test_execute_unknown_command(self):
        plugin = LSPToolPlugin()
        result = plugin.execute_user_command("other", {})
        assert "Unknown command" in result


class TestLogEntry:
    """Test LogEntry formatting."""

    def test_format_with_timestamp(self):
        from datetime import datetime
        entry = LogEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, 45, 123000),
            level=LOG_INFO,
            server="python",
            event="Connected",
            details="pyright-langserver"
        )
        formatted = entry.format(include_timestamp=True)
        assert "10:30:45" in formatted
        assert "[INFO]" in formatted
        assert "[python]" in formatted
        assert "Connected" in formatted
        assert "pyright-langserver" in formatted

    def test_format_without_timestamp(self):
        from datetime import datetime
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LOG_ERROR,
            server=None,
            event="Failed to connect"
        )
        formatted = entry.format(include_timestamp=False)
        assert "[ERROR]" in formatted
        assert "Failed to connect" in formatted
        assert ":" not in formatted.split("[")[0]  # No timestamp


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_config_from_file(self):
        plugin = LSPToolPlugin()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, ".lsp.json")
            config = {
                "languageServers": {
                    "python": {"command": "pyright-langserver", "args": ["--stdio"]}
                }
            }
            with open(config_path, "w") as f:
                json.dump(config, f)

            # Set workspace path instead of patching getcwd
            plugin.set_workspace_path(tmpdir)
            plugin._load_config_cache(force=True)

            assert "languageServers" in plugin._config_cache
            assert "python" in plugin._config_cache["languageServers"]

    def test_load_config_missing_file(self):
        plugin = LSPToolPlugin()
        plugin._config_cache = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.getcwd", return_value=tmpdir):
                with patch("os.path.expanduser", return_value=os.path.join(tmpdir, "nope")):
                    plugin._load_config_cache(force=True)

        assert plugin._config_cache == {}


class TestPluginLifecycle:
    """Test plugin initialization and shutdown."""

    def test_shutdown_without_init(self):
        plugin = LSPToolPlugin()
        # Should not raise
        plugin.shutdown()
        assert not plugin._initialized

    def test_double_shutdown(self):
        plugin = LSPToolPlugin()
        plugin.shutdown()
        plugin.shutdown()  # Should not raise


# =============================================================================
# Integration Tests (with mocked subprocess)
# =============================================================================

class TestLSPClientIntegration:
    """Integration tests for LSP client with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_client_lifecycle(self):
        """Test client start/stop with mocked process."""
        config = ServerConfig(name="test", command="echo", args=["test"])
        client = LSPClient(config)

        # Mock the subprocess
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = Mock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(return_value=b"")
        mock_process.stdout.read = AsyncMock(return_value=b"")
        mock_process.terminate = Mock()
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Client should handle empty response gracefully
            # (In real usage, server would respond with initialize result)
            try:
                await asyncio.wait_for(client.start(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Expected - no real server

            await client.stop()
            mock_process.terminate.assert_called()


# =============================================================================
# New Refactoring Types Tests
# =============================================================================

from ..lsp_client import TextEdit, WorkspaceEdit, CodeAction


class TestTextEdit:
    """Test TextEdit dataclass."""

    def test_from_dict(self):
        edit = TextEdit.from_dict({
            "range": {
                "start": {"line": 10, "character": 5},
                "end": {"line": 10, "character": 15}
            },
            "newText": "newValue"
        })
        assert edit.range.start.line == 10
        assert edit.range.start.character == 5
        assert edit.range.end.line == 10
        assert edit.range.end.character == 15
        assert edit.new_text == "newValue"

    def test_to_dict(self):
        edit = TextEdit(
            range=Range(Position(5, 0), Position(5, 10)),
            new_text="replacement"
        )
        d = edit.to_dict()
        assert d["range"]["start"]["line"] == 5
        assert d["range"]["end"]["character"] == 10
        assert d["newText"] == "replacement"

    def test_empty_new_text(self):
        edit = TextEdit.from_dict({
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
            "newText": ""
        })
        assert edit.new_text == ""


class TestWorkspaceEdit:
    """Test WorkspaceEdit dataclass."""

    def test_from_dict_changes_format(self):
        edit = WorkspaceEdit.from_dict({
            "changes": {
                "file:///test.py": [
                    {"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}}, "newText": "foo"},
                    {"range": {"start": {"line": 5, "character": 0}, "end": {"line": 5, "character": 3}}, "newText": "foo"}
                ],
                "file:///other.py": [
                    {"range": {"start": {"line": 10, "character": 4}, "end": {"line": 10, "character": 7}}, "newText": "foo"}
                ]
            }
        })
        assert len(edit.changes) == 2
        assert len(edit.changes["file:///test.py"]) == 2
        assert len(edit.changes["file:///other.py"]) == 1

    def test_from_dict_document_changes_format(self):
        edit = WorkspaceEdit.from_dict({
            "documentChanges": [
                {
                    "textDocument": {"uri": "file:///test.py", "version": 1},
                    "edits": [
                        {"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}}, "newText": "bar"}
                    ]
                }
            ]
        })
        assert len(edit.changes) == 1
        assert "file:///test.py" in edit.changes
        assert edit.changes["file:///test.py"][0].new_text == "bar"

    def test_get_affected_files(self):
        edit = WorkspaceEdit.from_dict({
            "changes": {
                "file:///a.py": [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}, "newText": "x"}],
                "file:///b.py": [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}, "newText": "y"}],
                "file:///c.py": [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}, "newText": "z"}]
            }
        })
        files = edit.get_affected_files()
        assert len(files) == 3
        assert "file:///a.py" in files
        assert "file:///b.py" in files
        assert "file:///c.py" in files

    def test_empty_workspace_edit(self):
        edit = WorkspaceEdit.from_dict({})
        assert len(edit.changes) == 0
        assert edit.get_affected_files() == []


class TestCodeAction:
    """Test CodeAction dataclass."""

    def test_from_dict_minimal(self):
        action = CodeAction.from_dict({
            "title": "Extract method"
        })
        assert action.title == "Extract method"
        assert action.kind is None
        assert action.edit is None
        assert action.command is None

    def test_from_dict_full(self):
        action = CodeAction.from_dict({
            "title": "Extract to function 'newFunc'",
            "kind": "refactor.extract",
            "isPreferred": True,
            "edit": {
                "changes": {
                    "file:///test.py": [
                        {"range": {"start": {"line": 10, "character": 0}, "end": {"line": 15, "character": 0}}, "newText": "def newFunc():\n    pass\n"}
                    ]
                }
            }
        })
        assert action.title == "Extract to function 'newFunc'"
        assert action.kind == "refactor.extract"
        assert action.is_preferred is True
        assert action.edit is not None
        assert len(action.edit.changes) == 1

    def test_from_dict_with_command(self):
        action = CodeAction.from_dict({
            "title": "Organize imports",
            "kind": "source.organizeImports",
            "command": {
                "command": "python.sortImports",
                "arguments": ["/path/to/file.py"]
            }
        })
        assert action.command is not None
        assert action.command["command"] == "python.sortImports"

    def test_from_dict_disabled(self):
        action = CodeAction.from_dict({
            "title": "Extract variable",
            "kind": "refactor.extract",
            "disabled": {"reason": "Selection is not an expression"}
        })
        assert action.disabled == "Selection is not an expression"

    def test_is_refactoring(self):
        refactor = CodeAction.from_dict({"title": "Extract", "kind": "refactor.extract"})
        quickfix = CodeAction.from_dict({"title": "Fix", "kind": "quickfix"})
        source = CodeAction.from_dict({"title": "Organize", "kind": "source.organizeImports"})

        assert refactor.is_refactoring() is True
        assert quickfix.is_refactoring() is False
        assert source.is_refactoring() is False

    def test_is_quickfix(self):
        quickfix = CodeAction.from_dict({"title": "Fix", "kind": "quickfix"})
        refactor = CodeAction.from_dict({"title": "Extract", "kind": "refactor.extract"})

        assert quickfix.is_quickfix() is True
        assert refactor.is_quickfix() is False

    def test_to_summary(self):
        action = CodeAction.from_dict({
            "title": "Extract method",
            "kind": "refactor.extract",
            "isPreferred": True,
            "edit": {
                "changes": {
                    "file:///a.py": [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}, "newText": "x"}],
                    "file:///b.py": [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}, "newText": "y"}]
                }
            }
        })
        summary = action.to_summary()
        assert summary["title"] == "Extract method"
        assert summary["kind"] == "refactor.extract"
        assert summary["preferred"] is True
        assert summary["has_edit"] is True
        assert summary["affected_files"] == 2


# =============================================================================
# Workspace Edit Application Tests
# =============================================================================

from ..plugin import _apply_text_edits_to_content, apply_workspace_edit


class TestApplyTextEditsToContent:
    """Test text edit application logic."""

    def test_single_line_replacement(self):
        content = "hello world"
        edits = [TextEdit(Range(Position(0, 6), Position(0, 11)), "universe")]
        result = _apply_text_edits_to_content(content, edits)
        assert result == "hello universe"

    def test_insertion(self):
        content = "hello world"
        edits = [TextEdit(Range(Position(0, 5), Position(0, 5)), " beautiful")]
        result = _apply_text_edits_to_content(content, edits)
        assert result == "hello beautiful world"

    def test_deletion(self):
        content = "hello beautiful world"
        edits = [TextEdit(Range(Position(0, 5), Position(0, 15)), "")]
        result = _apply_text_edits_to_content(content, edits)
        assert result == "hello world"

    def test_multiple_edits_same_line(self):
        content = "foo bar baz"
        edits = [
            TextEdit(Range(Position(0, 0), Position(0, 3)), "FOO"),
            TextEdit(Range(Position(0, 8), Position(0, 11)), "BAZ")
        ]
        result = _apply_text_edits_to_content(content, edits)
        assert result == "FOO bar BAZ"

    def test_multi_line_content(self):
        content = "line1\nline2\nline3"
        edits = [TextEdit(Range(Position(1, 0), Position(1, 5)), "REPLACED")]
        result = _apply_text_edits_to_content(content, edits)
        assert result == "line1\nREPLACED\nline3"

    def test_cross_line_replacement(self):
        content = "line1\nline2\nline3"
        edits = [TextEdit(Range(Position(0, 3), Position(2, 2)), "X")]
        result = _apply_text_edits_to_content(content, edits)
        assert result == "linXne3"

    def test_insert_new_lines(self):
        content = "line1\nline2"
        edits = [TextEdit(Range(Position(1, 0), Position(1, 0)), "inserted\n")]
        result = _apply_text_edits_to_content(content, edits)
        assert result == "line1\ninserted\nline2"

    def test_empty_content(self):
        content = ""
        edits = [TextEdit(Range(Position(0, 0), Position(0, 0)), "new content")]
        result = _apply_text_edits_to_content(content, edits)
        assert result == "new content"

    def test_edits_applied_in_reverse_order(self):
        """Edits should be applied bottom-to-top to preserve positions."""
        content = "aaa\nbbb\nccc"
        edits = [
            TextEdit(Range(Position(0, 0), Position(0, 3)), "AAA"),
            TextEdit(Range(Position(2, 0), Position(2, 3)), "CCC")
        ]
        result = _apply_text_edits_to_content(content, edits)
        assert result == "AAA\nbbb\nCCC"


class TestApplyWorkspaceEdit:
    """Test workspace edit file application."""

    def test_apply_to_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, 'w') as f:
                f.write("old_name = 1\nprint(old_name)")

            uri = f"file://{test_file}"
            edit = WorkspaceEdit(changes={
                uri: [
                    TextEdit(Range(Position(0, 0), Position(0, 8)), "new_name"),
                    TextEdit(Range(Position(1, 6), Position(1, 14)), "new_name")
                ]
            })

            result = apply_workspace_edit(edit)

            assert result["success"] is True
            assert len(result["files_modified"]) == 1
            assert test_file in result["files_modified"]

            with open(test_file, 'r') as f:
                content = f.read()
            assert content == "new_name = 1\nprint(new_name)"

    def test_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            original = "original content"
            with open(test_file, 'w') as f:
                f.write(original)

            uri = f"file://{test_file}"
            edit = WorkspaceEdit(changes={
                uri: [TextEdit(Range(Position(0, 0), Position(0, 8)), "modified")]
            })

            result = apply_workspace_edit(edit, dry_run=True)

            assert result["success"] is True
            assert len(result["files_modified"]) == 0  # No files modified in dry run
            assert len(result["changes"]) == 1  # But changes are reported

            with open(test_file, 'r') as f:
                content = f.read()
            assert content == original  # File unchanged

    def test_file_not_found(self):
        edit = WorkspaceEdit(changes={
            "file:///nonexistent/path/file.py": [
                TextEdit(Range(Position(0, 0), Position(0, 5)), "test")
            ]
        })

        result = apply_workspace_edit(edit)

        assert result["success"] is False
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0]["error"].lower()


# =============================================================================
# Refactoring Tool Tests
# =============================================================================

class TestRefactoringToolSchemas:
    """Test that refactoring tool schemas are properly defined."""

    def test_rename_symbol_has_apply_parameter(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True
        schemas = plugin.get_tool_schemas()

        rename_schema = next(s for s in schemas if s.name == "lsp_rename_symbol")
        assert "apply" in rename_schema.parameters["properties"]
        assert rename_schema.parameters["properties"]["apply"]["type"] == "boolean"

    def test_get_code_actions_schema(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True
        schemas = plugin.get_tool_schemas()

        schema = next(s for s in schemas if s.name == "lsp_get_code_actions")
        props = schema.parameters["properties"]

        assert "file_path" in props
        assert "start_line" in props
        assert "start_column" in props
        assert "end_line" in props
        assert "end_column" in props
        assert "only_refactorings" in props

        required = set(schema.parameters["required"])
        assert required == {"file_path", "start_line", "start_column", "end_line", "end_column"}

    def test_apply_code_action_schema(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True
        schemas = plugin.get_tool_schemas()

        schema = next(s for s in schemas if s.name == "lsp_apply_code_action")
        props = schema.parameters["properties"]

        assert "file_path" in props
        assert "action_title" in props

        required = set(schema.parameters["required"])
        assert "action_title" in required

    def test_get_code_actions_is_auto_approved(self):
        plugin = LSPToolPlugin()
        approved = plugin.get_auto_approved_tools()
        assert "lsp_get_code_actions" in approved

    def test_apply_code_action_not_auto_approved(self):
        plugin = LSPToolPlugin()
        approved = plugin.get_auto_approved_tools()
        assert "lsp_apply_code_action" not in approved

    def test_rename_symbol_not_auto_approved(self):
        plugin = LSPToolPlugin()
        approved = plugin.get_auto_approved_tools()
        assert "lsp_rename_symbol" not in approved


class TestRefactoringExecutors:
    """Test refactoring executor methods."""

    def test_get_code_actions_validates_parameters(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True
        plugin._connected_servers = set()  # No servers connected

        # Missing file_path
        result = plugin._exec_get_code_actions({
            "start_line": 1, "start_column": 1,
            "end_line": 1, "end_column": 10
        })
        assert "error" in result
        assert "file_path" in result["error"]

        # Missing start parameters
        result = plugin._exec_get_code_actions({
            "file_path": "/test.py",
            "end_line": 1, "end_column": 10
        })
        assert "error" in result

    def test_apply_code_action_validates_parameters(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True
        plugin._connected_servers = set()

        # Missing action_title
        result = plugin._exec_apply_code_action({
            "file_path": "/test.py",
            "start_line": 1, "start_column": 1,
            "end_line": 1, "end_column": 10
        })
        assert "error" in result
        assert "action_title" in result["error"]

    def test_rename_symbol_validates_parameters(self):
        plugin = LSPToolPlugin()
        plugin._initialized = True
        plugin._connected_servers = set()

        # Missing symbol
        result = plugin._exec_rename_symbol({"new_name": "foo"})
        assert "error" in result
        assert "symbol" in result["error"]

        # Missing new_name
        result = plugin._exec_rename_symbol({"symbol": "bar"})
        assert "error" in result
        assert "new_name" in result["error"]


class TestSystemInstructions:
    """Test updated system instructions."""

    def test_system_instructions_mention_refactoring(self):
        plugin = LSPToolPlugin()
        instructions = plugin.get_system_instructions()

        assert "Refactoring tools" in instructions
        assert "lsp_rename_symbol" in instructions
        assert "lsp_get_code_actions" in instructions
        assert "lsp_apply_code_action" in instructions
        assert "apply=True" in instructions or "apply=true" in instructions.lower()


class TestConnectTimeoutKnob:
    """Pin behavior of plugin_configs.lsp.connect_timeout_seconds.

    Pre-server-0.6.133 the per-server LSP `initialize` timeout was a
    hard-coded 15.0 at lsp/plugin.py:1699 — enough for pyright /
    typescript-language-server (5s cold-start typical) but starves
    Eclipse JDT LS (jdtls) on Maven workspaces where the full
    initialize + workspace import routinely runs 60-120s+.  The default
    was later raised 30->180 (#284) and the knob lets profiles tune
    per workspace.
    """

    def test_default_is_180_seconds(self):
        """Pin: unconfigured plugin uses the documented default.

        Raised 30->180 (#284): a cold jdtls Maven import routinely runs
        60-120s+, so a 30s default timed out WHILE jdtls was legitimately
        starting — orphaning the spawned subprocess and triggering a
        retry-autoconnect duplicate.  180s covers a cold import while still
        bounding a genuinely hung server (< MAX 300s)."""
        from ..plugin import DEFAULT_CONNECT_TIMEOUT_SECONDS
        plugin = LSPToolPlugin()
        # We must not touch _ensure_thread; just verify initial state +
        # configure() does the right thing without starting LSP servers.
        assert plugin._connect_timeout_seconds == DEFAULT_CONNECT_TIMEOUT_SECONDS
        assert DEFAULT_CONNECT_TIMEOUT_SECONDS == 180.0

    def test_initialize_applies_config_value(self):
        """Pin: an explicit value flows from config to the instance attr."""
        plugin = LSPToolPlugin()
        # Skip the background thread spawn by pre-marking _initialized
        # AFTER we run initialize() — initialize() itself does not start
        # threads; _ensure_thread() does, and we don't call that here.
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"connect_timeout_seconds": 60.0})
        assert plugin._connect_timeout_seconds == 60.0

    def test_initialize_clamps_below_min(self):
        """Pin: values below MIN_CONNECT_TIMEOUT_SECONDS clamp up."""
        from ..plugin import MIN_CONNECT_TIMEOUT_SECONDS
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"connect_timeout_seconds": 0.0})
        assert plugin._connect_timeout_seconds == MIN_CONNECT_TIMEOUT_SECONDS

    def test_initialize_clamps_above_max(self):
        """Pin: values above MAX_CONNECT_TIMEOUT_SECONDS clamp down."""
        from ..plugin import MAX_CONNECT_TIMEOUT_SECONDS
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"connect_timeout_seconds": 9999.0})
        assert plugin._connect_timeout_seconds == MAX_CONNECT_TIMEOUT_SECONDS

    def test_initialize_falls_back_on_non_numeric(self):
        """Pin: non-numeric input does not break startup; default applies."""
        from ..plugin import DEFAULT_CONNECT_TIMEOUT_SECONDS
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"connect_timeout_seconds": "not-a-number"})
        assert plugin._connect_timeout_seconds == DEFAULT_CONNECT_TIMEOUT_SECONDS

    def test_initialize_no_key_uses_default(self):
        """Pin: omitting the key leaves the default in place (a profile
        that only sets other lsp config must not reset the timeout)."""
        from ..plugin import DEFAULT_CONNECT_TIMEOUT_SECONDS
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"workspace_path": "/tmp/foo"})
        assert plugin._connect_timeout_seconds == DEFAULT_CONNECT_TIMEOUT_SECONDS

    def test_config_schema_exposes_knob(self):
        """Pin: the knob is discoverable via get_config_schema() so
        profile managers / settings forms can surface it."""
        plugin = LSPToolPlugin()
        schema = plugin.get_config_schema()
        names = [s.name for s in schema]
        assert "connect_timeout_seconds" in names
        entry = next(s for s in schema if s.name == "connect_timeout_seconds")
        assert entry.type == "float"
        assert entry.default == 180.0


class TestReapFailedClient:
    """Pin behavior of LSPToolPlugin._reap_failed_client (#284).

    When ``connect_server`` times out or raises, the spawned language-server
    subprocess (created in ``LSPClient.start()`` BEFORE the slow
    ``_initialize()`` handshake) is left running but UNTRACKED —
    ``self._clients[name]`` is only set on the success path.  The reaper
    calls ``client.stop()`` (the same teardown ``disconnect_server`` uses) so
    the orphan doesn't leak and the retry-autoconnect doesn't spawn a
    duplicate.  Before this fix every premature 30s timeout cost one jdtls
    cold-start of leaked RAM until the daemon OOMed.
    """

    @pytest.mark.asyncio
    async def test_reaps_spawned_client(self):
        """A spawned-but-failed client is stopped (subprocess killed)."""
        plugin = LSPToolPlugin()
        plugin._trace = Mock()
        client = AsyncMock()
        await plugin._reap_failed_client("java", client)
        client.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_when_no_client_spawned(self):
        """Failure before LSPClient(...) leaves client=None — must not raise
        and must not attempt any teardown."""
        plugin = LSPToolPlugin()
        plugin._trace = Mock()
        # Must not raise.
        await plugin._reap_failed_client("java", None)

    @pytest.mark.asyncio
    async def test_swallows_stop_exception(self):
        """A reap is best-effort: an exception from client.stop() must not
        propagate out of the except handler (it would mask the original
        connect failure and abort the auto-connect loop)."""
        plugin = LSPToolPlugin()
        plugin._trace = Mock()
        client = AsyncMock()
        client.stop.side_effect = RuntimeError("process already dead")
        # Must not raise.
        await plugin._reap_failed_client("java", client)
        client.stop.assert_awaited_once()


class TestAtexitReapJdtls:
    """Pin the process-exit jdtls reaper (#284 per-slot leak OOM-stopper).

    The pre-warm pool-slot teardown path (daemon closes the slot socket →
    runner RPC ``serve()`` returns on EOF → slot ``sys.exit(0)``) does NOT
    call ``plugin.shutdown()``.  Without an atexit backstop the connected
    jdtls (0.5-1.5 GB) was abandoned, re-parented to the daemon subreaper,
    and accumulated one-per-slot across cascade stages until the daemon
    OOMed.  The reaper SIGKILLs each tracked subprocess by pid at
    interpreter exit so jdtls dies WITH its slot.
    """

    def _client_with_pid(self, pid):
        client = Mock()
        client._process = Mock()
        client._process.pid = pid
        return client

    def test_sigkills_each_tracked_pid(self):
        plugin = LSPToolPlugin()
        plugin._clients = {
            "java": self._client_with_pid(4242),
            "pyright": self._client_with_pid(4343),
        }
        with patch("shared.plugins.lsp.plugin.os.kill") as mock_kill:
            plugin._atexit_reap_jdtls()
        sent = {c.args for c in mock_kill.call_args_list}
        assert (4242, signal.SIGKILL) in sent
        assert (4343, signal.SIGKILL) in sent

    def test_noop_with_no_clients(self):
        plugin = LSPToolPlugin()
        plugin._clients = {}
        with patch("shared.plugins.lsp.plugin.os.kill") as mock_kill:
            plugin._atexit_reap_jdtls()
        mock_kill.assert_not_called()

    def test_skips_client_without_live_process(self):
        """A client whose subprocess was never spawned (``_process`` None) or
        has no pid is skipped — no kill, no raise."""
        plugin = LSPToolPlugin()
        no_proc = Mock()
        no_proc._process = None
        plugin._clients = {"java": no_proc}
        with patch("shared.plugins.lsp.plugin.os.kill") as mock_kill:
            plugin._atexit_reap_jdtls()
        mock_kill.assert_not_called()

    def test_swallows_kill_errors(self):
        """A dead/already-reaped pid (ProcessLookupError) must not propagate
        out of the atexit handler."""
        plugin = LSPToolPlugin()
        plugin._clients = {"java": self._client_with_pid(999999)}
        with patch(
            "shared.plugins.lsp.plugin.os.kill",
            side_effect=ProcessLookupError(),
        ):
            plugin._atexit_reap_jdtls()  # must not raise

    def test_register_is_idempotent(self):
        """_register_atexit_reaper registers exactly once even if called
        repeatedly (each _ensure_thread call invokes it)."""
        plugin = LSPToolPlugin()
        with patch("shared.plugins.lsp.plugin.atexit.register") as mock_reg:
            plugin._register_atexit_reaper()
            plugin._register_atexit_reaper()
            plugin._register_atexit_reaper()
        mock_reg.assert_called_once_with(plugin._atexit_reap_jdtls)
        assert plugin._atexit_registered is True


class TestDiagnosticsWaitKnobs:
    """Pin behavior of plugin_configs.lsp.diagnostics_{max,min}_wait_seconds.

    Pre-server-0.6.134 the post-didOpen wait was a hard-coded
    `asyncio.sleep(0.8)` at lsp/plugin.py:1962.  Sufficient for
    pyright / typescript-language-server but starved jdtls on Maven
    workspaces (3-8s first publishDiagnostics).  The 0.6.134 knobs
    let profiles raise the upper bound — without paying the cost in
    the fast-server case, because the framework now awaits a per-URI
    `asyncio.Event` signalled by the JSON-RPC reader.
    """

    def test_defaults_are_five_and_half(self):
        from ..plugin import (
            DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS,
            DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS,
        )
        plugin = LSPToolPlugin()
        assert plugin._diagnostics_max_wait_seconds == DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS
        assert plugin._diagnostics_min_wait_seconds == DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS
        assert DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS == 5.0
        assert DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS == 0.5

    def test_initialize_applies_max_wait_config(self):
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"diagnostics_max_wait_seconds": 10.0})
        assert plugin._diagnostics_max_wait_seconds == 10.0

    def test_initialize_applies_min_wait_config(self):
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"diagnostics_min_wait_seconds": 1.5})
        assert plugin._diagnostics_min_wait_seconds == 1.5

    def test_min_wait_clamped_to_max_wait(self):
        """Pin: min_wait cannot exceed max_wait. If operator sets
        min=10 but max=5, min is silently clamped to 5 (and traced)."""
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({
                "diagnostics_max_wait_seconds": 5.0,
                "diagnostics_min_wait_seconds": 10.0,
            })
        assert plugin._diagnostics_max_wait_seconds == 5.0
        assert plugin._diagnostics_min_wait_seconds == 5.0

    def test_max_wait_clamps_above_ceiling(self):
        from ..plugin import MAX_DIAGNOSTICS_MAX_WAIT_SECONDS
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"diagnostics_max_wait_seconds": 9999.0})
        assert plugin._diagnostics_max_wait_seconds == MAX_DIAGNOSTICS_MAX_WAIT_SECONDS

    def test_max_wait_zero_disables_await(self):
        """Pin: 0 is valid and means 'legacy read-cache-as-is'."""
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({
                "diagnostics_max_wait_seconds": 0.0,
                "diagnostics_min_wait_seconds": 0.0,
            })
        assert plugin._diagnostics_max_wait_seconds == 0.0
        assert plugin._diagnostics_min_wait_seconds == 0.0

    def test_non_numeric_falls_back_to_default(self):
        from ..plugin import (
            DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS,
            DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS,
        )
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({
                "diagnostics_max_wait_seconds": "not-a-number",
                "diagnostics_min_wait_seconds": "also-not",
            })
        assert plugin._diagnostics_max_wait_seconds == DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS
        assert plugin._diagnostics_min_wait_seconds == DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS

    def test_config_schema_exposes_both_knobs(self):
        plugin = LSPToolPlugin()
        names = [s.name for s in plugin.get_config_schema()]
        assert "diagnostics_max_wait_seconds" in names
        assert "diagnostics_min_wait_seconds" in names
        assert "diagnostics_convergence_window_seconds" in names


class TestAwaitDiagnostics:
    """Pin LSPClient.await_diagnostics semantics — the bounded-poll
    primitive that replaced the 0.8s sleep at _call_lsp_method:1962."""

    @pytest.mark.asyncio
    async def test_returns_true_when_publishdiagnostics_arrives(self):
        """Pin: simulate the JSON-RPC reader firing the event mid-await;
        the coroutine returns True and is unblocked immediately."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        path = "/tmp/foo.py"
        uri = client.uri_from_path(path)

        async def fire_event_soon():
            await asyncio.sleep(0.05)
            # Simulate what the reader does on publishDiagnostics:
            ev = client._diagnostics_events.get(uri)
            if ev is None:
                ev = asyncio.Event()
                client._diagnostics_events[uri] = ev
            ev.set()

        firer = asyncio.create_task(fire_event_soon())
        result = await client.await_diagnostics(
            path, max_wait=2.0, min_wait=0.0
        )
        await firer
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        """Pin: no notification within max_wait → returns False; caller
        must still get_diagnostics afterwards in case earlier batches
        landed."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        result = await client.await_diagnostics(
            "/tmp/foo.py", max_wait=0.1, min_wait=0.0
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_min_wait_enforced(self):
        """Pin: even if event fires immediately, await blocks at least
        min_wait. Multi-stage analysis pipelines (parser → compiler →
        linter) get room to deliver later batches."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        path = "/tmp/foo.py"
        uri = client.uri_from_path(path)
        # Pre-set the event so it fires "before" the await call
        ev = asyncio.Event()
        ev.set()
        client._diagnostics_events[uri] = ev

        start = asyncio.get_event_loop().time()
        await client.await_diagnostics(path, max_wait=1.0, min_wait=0.2)
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed >= 0.2, f"min_wait floor not enforced: elapsed={elapsed}"

    @pytest.mark.asyncio
    async def test_max_wait_zero_short_circuits(self):
        """Pin: max_wait=0 returns immediately reporting only whether
        the event was already set (legacy read-cache-as-is)."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        path = "/tmp/foo.py"

        # Event not pre-set → returns False without any waiting
        start = asyncio.get_event_loop().time()
        result = await client.await_diagnostics(path, max_wait=0.0, min_wait=0.0)
        elapsed = asyncio.get_event_loop().time() - start
        assert result is False
        assert elapsed < 0.05  # truly no wait

    @pytest.mark.asyncio
    async def test_event_cleared_for_next_cycle(self):
        """Pin: after one successful await, the per-URI Event is
        cleared so a subsequent didChange + await cycle works
        correctly for in-session re-renders."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        path = "/tmp/foo.py"
        uri = client.uri_from_path(path)
        ev = asyncio.Event()
        ev.set()
        client._diagnostics_events[uri] = ev

        await client.await_diagnostics(path, max_wait=1.0, min_wait=0.0)
        # Event was cleared by await_diagnostics; second call without
        # another publishDiagnostics will time out.
        result = await client.await_diagnostics(path, max_wait=0.1, min_wait=0.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_diagnostics_cache_not_cleared(self):
        """Pin: await_diagnostics clears the Event but NOT the
        diagnostic cache — readers must still see the batch."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        path = "/tmp/foo.py"
        uri = client.uri_from_path(path)
        # Simulate a publishDiagnostics arrival populating both:
        from ..lsp_client import Diagnostic, Range, Position
        ev = asyncio.Event()
        ev.set()
        client._diagnostics_events[uri] = ev
        client._diagnostics[uri] = [
            Diagnostic(
                range=Range(start=Position(0, 0), end=Position(0, 5)),
                message="test diagnostic",
            ),
        ]
        await client.await_diagnostics(path, max_wait=1.0, min_wait=0.0)
        # Cache survives:
        assert len(client._diagnostics[uri]) == 1
        assert client._diagnostics[uri][0].message == "test diagnostic"


class TestAwaitDiagnosticsConvergenceLoop:
    """Pin PR-224 (server 0.6.193): bounded-poll convergence wait.

    Empirical motivation (2026-06-05 instrumented cascade):
    jdtls publishes Customer.java with 12 errors at T+0, then
    re-publishes with 0 errors at T+2.2s once intra-project imports
    settle.  Pre-PR-224 `await_diagnostics` returned on the first
    publish; the cache reader saw 12 phantom errors, the agent
    rejected the (clean) file, NudgeExhausted aborted.

    The convergence loop keeps listening for follow-up publishes
    for ``convergence_window`` seconds after the first one lands.
    Each follow-up resets the timer (jdtls's cascade may span
    several re-publishes).  ``max_wait`` is a hard ceiling.

    91 adjacent-publish races mapped in the cascade log; p50 = 1.46s,
    p90 = 17.9s (edit-cycle tail, not jdtls indexing — a longer
    window can't fix those).  Default 3.0s catches the fresh-render
    cluster.
    """

    @pytest.mark.asyncio
    async def test_zero_window_preserves_legacy_first_publish_return(self):
        """Pin: ``convergence_window=0`` (default for tests not opting
        in) preserves the pre-PR-224 first-publish semantics — returns
        as soon as the Event fires, no extra waiting."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        path = "/tmp/foo.py"
        uri = client.uri_from_path(path)

        ev = asyncio.Event()
        ev.set()  # Event pre-set, simulating publish already arrived
        client._diagnostics_events[uri] = ev

        start = asyncio.get_event_loop().time()
        result = await client.await_diagnostics(
            path, max_wait=5.0, min_wait=0.0, convergence_window=0.0,
        )
        elapsed = asyncio.get_event_loop().time() - start
        assert result is True
        assert elapsed < 0.05, (
            f"convergence_window=0 should not wait; elapsed={elapsed}"
        )

    @pytest.mark.asyncio
    async def test_convergence_window_waits_for_followup_publish(self):
        """Pin: after first publish, the loop keeps listening; a
        follow-up publish within the window is observed and the loop
        continues until the window elapses without a new publish."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        path = "/tmp/foo.py"
        uri = client.uri_from_path(path)

        # First publish arrives immediately, then a follow-up after 100ms.
        # The convergence window is 300ms — long enough that the
        # follow-up arrives in time but no further publish follows.
        async def publish_then_followup() -> None:
            ev = client._diagnostics_events.setdefault(uri, asyncio.Event())
            ev.set()  # T+0 — first publish
            await asyncio.sleep(0.1)
            ev.set()  # T+0.1s — follow-up publish

        firer = asyncio.create_task(publish_then_followup())
        start = asyncio.get_event_loop().time()
        result = await client.await_diagnostics(
            path, max_wait=2.0, min_wait=0.0, convergence_window=0.3,
        )
        elapsed = asyncio.get_event_loop().time() - start
        await firer

        assert result is True
        # First publish at T+0, follow-up at T+0.1s, then 0.3s window
        # elapses without further publish → exits around T+0.4s.
        assert 0.35 <= elapsed < 0.6, (
            f"expected ~0.4s (0.1 + 0.3 window), got {elapsed}"
        )

    @pytest.mark.asyncio
    async def test_convergence_window_exits_without_followup(self):
        """Pin: first publish lands, no follow-up arrives within the
        window — loop exits after exactly one window elapses."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        path = "/tmp/foo.py"
        uri = client.uri_from_path(path)

        ev = asyncio.Event()
        ev.set()
        client._diagnostics_events[uri] = ev

        start = asyncio.get_event_loop().time()
        result = await client.await_diagnostics(
            path, max_wait=5.0, min_wait=0.0, convergence_window=0.2,
        )
        elapsed = asyncio.get_event_loop().time() - start

        assert result is True
        # Single window with no follow-up → ~0.2s elapsed.
        assert 0.18 <= elapsed < 0.4, (
            f"expected ~0.2s window timeout, got {elapsed}"
        )

    @pytest.mark.asyncio
    async def test_max_wait_is_hard_ceiling_even_with_runaway_publishes(self):
        """Pin: max_wait caps total elapsed time even when follow-up
        publishes keep arriving — a runaway-republishing server can't
        hang the call site."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        path = "/tmp/foo.py"
        uri = client.uri_from_path(path)

        # A pathological publisher that re-publishes every 50ms
        # forever.  Without max_wait the convergence loop would
        # never exit.
        stop = asyncio.Event()

        async def runaway() -> None:
            ev = client._diagnostics_events.setdefault(uri, asyncio.Event())
            while not stop.is_set():
                ev.set()
                await asyncio.sleep(0.05)

        firer = asyncio.create_task(runaway())
        start = asyncio.get_event_loop().time()
        result = await client.await_diagnostics(
            path, max_wait=0.5, min_wait=0.0, convergence_window=1.0,
        )
        elapsed = asyncio.get_event_loop().time() - start
        stop.set()
        await firer

        assert result is True
        # max_wait = 0.5s caps the total despite convergence_window=1.0
        # and runaway publishes.
        assert elapsed < 0.7, (
            f"max_wait must cap total wait; got {elapsed}"
        )

    @pytest.mark.asyncio
    async def test_no_first_publish_within_max_wait_returns_false(self):
        """Pin: max_wait expires before any first publish → False;
        convergence loop is never entered."""
        config = ServerConfig(name="test", command="dummy", args=[])
        client = LSPClient(config)
        result = await client.await_diagnostics(
            "/tmp/foo.py",
            max_wait=0.1,
            min_wait=0.0,
            convergence_window=2.0,
        )
        assert result is False


class TestValidateSnippetUsesAwaitDiagnostics:
    """Pin server-0.6.135 fix: `lsp_validate_snippet` was missed by
    PR-3 and continued to use `await asyncio.sleep(0.5)` even after
    the dispatch path was converted to bounded poll. PR-4 closes
    that wait-window gap for the validate_snippet branch too,
    inheriting the same `diagnostics_{max,min}_wait_seconds` knobs.
    """

    def test_validate_snippet_source_no_longer_hardcodes_sleep(self):
        """Pin: the `validate_snippet` branch in `_call_lsp_method`
        no longer contains `asyncio.sleep(0.5)`. Catches a future
        refactor that accidentally reintroduces the hardcoded sleep.
        """
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        # Locate the validate_snippet branch
        start = src.index("elif method == 'validate_snippet':")
        # End at the next `elif method == ` or `else:` at the same
        # indent level — bounded scan is fine since the branch is
        # ~50 lines.
        end_marker_a = src.find("elif method ==", start + 1)
        end_marker_b = src.find("\n        else:", start + 1)
        end_candidates = [m for m in (end_marker_a, end_marker_b) if m != -1]
        end = min(end_candidates) if end_candidates else start + 4000
        branch_src = src[start:end]
        # Match actual code (`await asyncio.sleep(...)`), not the
        # explanatory comment which references the historical literal.
        assert "await asyncio.sleep(0.5)" not in branch_src, (
            "validate_snippet branch still contains hardcoded "
            "`await asyncio.sleep(0.5)`; should use "
            "`await client.await_diagnostics(...)` with the "
            "configured knobs (PR-4 / server 0.6.135 fix)."
        )

    def test_validate_snippet_source_uses_await_diagnostics(self):
        """Pin: the validate_snippet branch calls
        `client.await_diagnostics(...)` with the plugin's knobs."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        start = src.index("elif method == 'validate_snippet':")
        end_marker_a = src.find("elif method ==", start + 1)
        end_marker_b = src.find("\n        else:", start + 1)
        end_candidates = [m for m in (end_marker_a, end_marker_b) if m != -1]
        end = min(end_candidates) if end_candidates else start + 4000
        branch_src = src[start:end]
        assert "await client.await_diagnostics(" in branch_src, (
            "validate_snippet branch should call "
            "`await client.await_diagnostics(...)` to wait for the "
            "first publishDiagnostics batch (PR-4)."
        )
        # Both knobs must be passed in — not hard-coded values.
        assert "self._diagnostics_max_wait_seconds" in branch_src
        assert "self._diagnostics_min_wait_seconds" in branch_src


class TestDebugLogPathKnob:
    """Pin server-0.6.136 fix: pre-fix, the diagnostic log was hard-
    coded to `tempfile.gettempdir()/lsp_debug.log` (e.g.
    /tmp/lsp_debug.log) which apparmor-confined runners couldn't
    write.  The outer try/except in `_load_config_cache` misclassified
    the apparmor PermissionError as a JSON-load failure, reset
    `_config_cache = {}`, and the entire LSP enrichment chain
    silently broke for v141-v144.

    Fix: operator-configurable path knob defaulting to
    `.jaato/logs/lsp_debug.log` (workspace-relative).
    `get_apparmor_rules` emits the matching rw grant.  Symmetric
    path resolution at write site + apparmor composer.
    """

    def test_default_path_is_workspace_relative(self):
        from ..plugin import DEFAULT_DEBUG_LOG_PATH
        assert DEFAULT_DEBUG_LOG_PATH == ".jaato/logs/lsp_debug.log"
        plugin = LSPToolPlugin()
        assert plugin._debug_log_path_raw == DEFAULT_DEBUG_LOG_PATH

    def test_initialize_applies_config_value(self):
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"debug_log_path": "/var/log/jaato/lsp.log"})
        assert plugin._debug_log_path_raw == "/var/log/jaato/lsp.log"

    def test_initialize_empty_string_disables_log(self):
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"debug_log_path": ""})
        assert plugin._debug_log_path_raw == ""

    def test_initialize_none_disables_log(self):
        plugin = LSPToolPlugin()
        with patch.object(plugin, "_ensure_thread"):
            plugin.initialize({"debug_log_path": None})
        assert plugin._debug_log_path_raw == ""

    def test_resolve_absolute_path_passthrough(self):
        result = LSPToolPlugin._resolve_debug_log_path(
            "/var/log/lsp.log", workspace_path="/workspace"
        )
        assert result == "/var/log/lsp.log"

    def test_resolve_relative_joins_workspace(self):
        result = LSPToolPlugin._resolve_debug_log_path(
            ".jaato/logs/lsp_debug.log", workspace_path="/workspace"
        )
        assert result == "/workspace/.jaato/logs/lsp_debug.log"

    def test_resolve_empty_returns_none(self):
        result = LSPToolPlugin._resolve_debug_log_path(
            "", workspace_path="/workspace"
        )
        assert result is None

    def test_resolve_relative_without_workspace_returns_none(self):
        """Pin: writing a workspace-relative path without a workspace
        is meaningless — return None to suppress the diagnostic
        rather than fall back to daemon cwd (no-hardcoded-fallback
        rule)."""
        result = LSPToolPlugin._resolve_debug_log_path(
            ".jaato/logs/lsp_debug.log", workspace_path=None
        )
        assert result is None

    def test_apparmor_fragment_grants_resolved_parent_dir(self):
        """Pin: get_apparmor_rules emits the rules for the parent dir
        of the resolved log path.  Two-rule subtree (PR-147 file_edit
        pattern) covers the mkdir chain + descendants."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path="/workspace",
            session_id="test-session",
            config_root=None,
            plugin_config={"debug_log_path": ".jaato/logs/lsp_debug.log"},
        )
        assert "/workspace/.jaato/logs/    rw," in rules
        assert "/workspace/.jaato/logs/**  rw," in rules

    def test_apparmor_fragment_uses_default_when_not_in_config(self):
        """Pin: composer uses DEFAULT_DEBUG_LOG_PATH when operator
        does not set the knob.  Same path as the write site falls
        through to."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path="/workspace",
            session_id="test-session",
            config_root=None,
            plugin_config={},
        )
        # Default is .jaato/logs/lsp_debug.log
        assert "/workspace/.jaato/logs/    rw," in rules

    def test_apparmor_fragment_handles_absolute_path(self):
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path="/workspace",
            session_id="test-session",
            config_root=None,
            plugin_config={"debug_log_path": "/var/log/jaato/lsp.log"},
        )
        assert "/var/log/jaato/    rw," in rules
        assert "/var/log/jaato/**  rw," in rules

    def test_apparmor_fragment_empty_path_emits_no_rules(self):
        """Pin: operator disables the log → no rules emitted (their
        responsibility per no-fallback rule)."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path="/workspace",
            session_id="test-session",
            config_root=None,
            plugin_config={"debug_log_path": ""},
        )
        assert rules == []

    def test_load_config_cache_does_not_abort_on_debug_write_failure(self):
        """Pin: the actual bug.  Pre-0.6.136, a debug-log write
        failure (PermissionError under apparmor, etc.) was caught by
        the outer try/except, misclassified as a config-load failure,
        and `_config_cache = {}` was set → "No LSP servers
        configured" → enrichment dead.  Post-fix, the write failure
        is scoped to its own try/except so config_cache survives."""
        plugin = LSPToolPlugin()
        plugin._workspace_path = None  # avoid workspace_path resolution
        plugin._debug_log_path_raw = "/dev/full"  # simulate write-fails-everywhere
        # Build an in-memory .lsp.json via a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lsp.json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"languageServers": {"java": {"command": "jdtls"}}}, f)
            config_file = f.name
        try:
            plugin._custom_config_path = config_file
            plugin._load_config_cache()
            # The CRITICAL assertion: config_cache survives even though
            # the debug-write path is unwritable.  Pre-fix, this would
            # be empty.
            assert plugin._config_cache.get("languageServers", {}).get("java"), (
                "config_cache was cleared by a debug-write failure — "
                "regression of the v141-v144 bug class."
            )
        finally:
            os.unlink(config_file)

    def test_config_schema_exposes_debug_log_path(self):
        plugin = LSPToolPlugin()
        names = [s.name for s in plugin.get_config_schema()]
        assert "debug_log_path" in names


class TestServerBinaryApparmorGrants:
    """Pin server-0.6.137 fix (PR-154): `get_apparmor_rules` also
    emits `ix` grants for each LSP server's canonical binary path,
    plus a `<install-dir>/** r,` glob, plus a Python interpreter
    `ix,` grant when the binary is a shebang script.

    Motivating evidence (v145, server 0.6.136 with PR-153 applied):
    `lsp logs` showed `Connection failed - [Errno 13] Permission
    denied: jdtls` despite the daemon-side (unconfined) instance
    happily spawning the same binary.  PR-148 confinement requires
    explicit `ix` grants on the canonical binary path for the
    runner's `asyncio.create_subprocess_exec` to succeed.
    """

    def test_no_lsp_json_yields_only_debug_log_rules(self, tmp_path):
        """Pin: composer tolerates missing `.lsp.json` (no servers
        to grant).  The debug_log_path rules still come through."""
        workspace = str(tmp_path)
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=workspace,
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        # Debug log rules present
        assert any(f"{workspace}/.jaato/logs/" in r for r in rules)
        # No `ix,` rules emitted (no servers)
        assert not any(" ix," in r for r in rules)

    def test_resolves_absolute_command_path(self, tmp_path):
        """Pin: an absolute command path is emitted directly."""
        # Write a stub server binary
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "languageId": "stub",
                }
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        # Canonical path appears with ix,
        assert any(
            r.startswith(str(stub)) and r.endswith(" ix,")
            for r in rules
        )

    def test_resolves_bare_command_name_via_path(self, tmp_path, monkeypatch):
        """Pin: a bare command name (e.g. `jdtls`) is resolved via
        `shutil.which` against the composer's PATH."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "fake-server"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))

        workspace = tmp_path / "ws"
        workspace.mkdir()
        lsp_json = workspace / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "fake": {"command": "fake-server", "languageId": "fake"}
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(workspace),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        assert any(
            r.startswith(str(stub)) and r.endswith(" ix,")
            for r in rules
        )

    def test_unresolved_command_silently_skipped(self, tmp_path, monkeypatch):
        """Pin: a bare-name command that doesn't exist on PATH is
        skipped (no rule emitted; runtime spawn will fail loudly
        with the same EACCES it currently does — more diagnosable
        than a silent missing grant)."""
        # PATH doesn't contain the target
        monkeypatch.setenv("PATH", "/nonexistent")

        workspace = tmp_path
        lsp_json = workspace / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "absent": {"command": "i-do-not-exist-on-path"}
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(workspace),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        # No `ix,` rule for the absent command
        assert not any("i-do-not-exist-on-path" in r for r in rules)

    def test_python_shebang_emits_interpreter_grant(self, tmp_path, monkeypatch):
        """Pin: jdtls case.  A `#!/usr/bin/env python3` wrapper
        script gets BOTH the wrapper AND the Python interpreter as
        `ix,` grants."""
        # Create a Python-wrapper-shaped stub
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        wrapper = bin_dir / "jdtls-stub"
        wrapper.write_text("#!/usr/bin/env python3\nprint('jdtls')\n")
        wrapper.chmod(0o755)

        # Ensure python3 is resolvable on PATH for the test
        # (composer uses shutil.which to find the interpreter).
        # Use whatever the test environment's PATH has.

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {"command": str(wrapper)}
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        # Wrapper itself granted ix
        assert any(
            r.startswith(str(wrapper)) and r.endswith(" ix,")
            for r in rules
        )
        # Python interpreter ALSO granted ix
        python_grants = [r for r in rules if "python" in r and r.endswith(" ix,")]
        assert python_grants, f"expected python ix grant, got: {rules}"

    def test_install_dir_glob_emitted(self, tmp_path):
        """Pin: install-dir glob (`<grandparent>/** r,`) emitted so
        the binary can read its bundled plugins / jars."""
        bin_dir = tmp_path / "myapp" / "bin"
        bin_dir.mkdir(parents=True)
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {"command": str(stub)}
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        # Install dir is binary's grandparent (myapp/), read glob
        install_dir = str(tmp_path / "myapp")
        assert any(
            r == f"{install_dir}/** r,"
            for r in rules
        )

    def test_load_lsp_config_static_search_order(self, tmp_path):
        """Pin: composer's path-search matches runtime's exactly.
        plugin_config.config_path wins; workspace .lsp.json next;
        ~/.lsp.json last."""
        # Just verify config_path wins when set
        ws = tmp_path / "ws"
        ws.mkdir()
        ws_config = ws / ".lsp.json"
        ws_config.write_text(json.dumps({"languageServers": {"a": {}}}))

        custom_config = tmp_path / "custom.lsp.json"
        custom_config.write_text(json.dumps({"languageServers": {"b": {}}}))

        result = LSPToolPlugin._load_lsp_config_static(
            workspace_path=str(ws),
            plugin_config={"config_path": str(custom_config)},
        )
        assert result is not None
        assert "b" in result["languageServers"]
        assert "a" not in result["languageServers"]

    def test_load_lsp_config_static_tolerates_missing(self, tmp_path):
        """Pin: composer doesn't crash if no .lsp.json exists
        anywhere.  Returns None; caller emits no server grants."""
        result = LSPToolPlugin._load_lsp_config_static(
            workspace_path=str(tmp_path),
            plugin_config={},
        )
        # tmp_path has no .lsp.json; ~/.lsp.json may or may not exist
        # on the test host.  If it doesn't, result is None; if it
        # does, result is its content.  Either way, no crash.
        assert result is None or isinstance(result, dict)

    def test_resolve_command_canonical_absolute(self, tmp_path):
        """Pin: absolute path passes through realpath."""
        stub = tmp_path / "bin-stub"
        stub.write_text("")
        result = LSPToolPlugin._resolve_command_canonical(str(stub))
        assert result == str(stub)

    def test_resolve_command_canonical_bare_not_found(self, monkeypatch):
        """Pin: bare name not on PATH returns None."""
        monkeypatch.setenv("PATH", "/nonexistent")
        result = LSPToolPlugin._resolve_command_canonical("doesnotexist")
        assert result is None

    def test_detect_shebang_python_env_style(self, tmp_path):
        """Pin: `#!/usr/bin/env python3` recognised, env-resolved."""
        script = tmp_path / "wrapper"
        script.write_text("#!/usr/bin/env python3\n")
        result = LSPToolPlugin._detect_shebang_interpreter(str(script))
        # Should resolve to wherever shutil.which finds python3
        assert result is not None
        assert "python" in result

    def test_detect_shebang_python_direct_style(self, tmp_path):
        """Pin: `#!/usr/bin/python3` recognised, returned as-is."""
        script = tmp_path / "wrapper"
        script.write_text("#!/usr/bin/python3\n")
        result = LSPToolPlugin._detect_shebang_interpreter(str(script))
        assert result is not None
        assert "python" in result

    def test_detect_shebang_non_python_ignored(self, tmp_path):
        """Pin: non-Python shebangs return None (deferred until
        evidence supports them)."""
        script = tmp_path / "wrapper"
        script.write_text("#!/bin/bash\n")
        result = LSPToolPlugin._detect_shebang_interpreter(str(script))
        assert result is None

    def test_detect_shebang_no_shebang(self, tmp_path):
        """Pin: binary without shebang returns None."""
        binary = tmp_path / "compiled-binary"
        binary.write_bytes(b"\x7fELF\x02\x01\x01")  # ELF magic, not a script
        result = LSPToolPlugin._detect_shebang_interpreter(str(binary))
        assert result is None

    def test_duplicate_servers_dedupe_canonical(self, tmp_path):
        """Pin: two server entries pointing at the same canonical
        binary emit ONE `ix,` grant, not two."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "shared-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "first": {"command": str(stub)},
                "second": {"command": str(stub)},  # same binary
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        ix_rules = [r for r in rules if r.endswith(" ix,") and str(stub) in r]
        assert len(ix_rules) == 1, f"expected 1 ix rule for shared binary, got: {ix_rules}"


class TestServerDataDirApparmorGrants:
    """Pin server-0.6.138 fix (PR-155): the apparmor composer also
    walks each server's `args` and emits rw grants for any
    operator-passed ``-data <path>`` / ``--data-dir <path>`` flags.

    Motivating case (v146-redo on server 0.6.137 with PR-154 applied):
    jdtls Python wrapper crashed at `tempfile.gettempdir()` under
    apparmor confinement before invoking java — wrapper's default
    data dir computation reached for /tmp / /var/tmp / /usr/tmp, all
    denied.  Operator unblock: pass `-data <workspace-relative-path>`
    in `.lsp.json args` so the wrapper uses an explicit dir.  This
    composer auto-emits the matching rw grant.
    """

    def test_data_flag_with_space_separator(self, tmp_path):
        """Pin: `args: ["-data", "<path>"]` emits rw grants for the
        resolved path."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        data_dir = ".jaato/jdtls-data"  # workspace-relative
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "args": ["-data", data_dir],
                }
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        expected_resolved = str(tmp_path / data_dir)
        assert f"{expected_resolved}/    rw," in rules
        assert f"{expected_resolved}/**  rw," in rules

    def test_data_dir_flag_with_space_separator(self, tmp_path):
        """Pin: `--data-dir <path>` recognised (pyright convention)."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "args": ["--data-dir", ".jaato/pyright-data"],
                }
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        expected = str(tmp_path / ".jaato/pyright-data")
        assert f"{expected}/    rw," in rules
        assert f"{expected}/**  rw," in rules

    def test_data_flag_with_equals_separator(self, tmp_path):
        """Pin: `args: ["-data=<path>"]` (combined form) recognised."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "args": ["-data=.jaato/jdtls-data"],
                }
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        expected = str(tmp_path / ".jaato/jdtls-data")
        assert f"{expected}/    rw," in rules
        assert f"{expected}/**  rw," in rules

    def test_workspace_root_variable_expansion(self, tmp_path):
        """Pin: `${workspaceRoot}/.jaato/jdtls-data` expands at
        composer time to match what runtime expand_variables
        produces."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "args": ["-data", "${workspaceRoot}/.jaato/jdtls-data"],
                }
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        expected = str(tmp_path / ".jaato/jdtls-data")
        assert any(expected in r for r in rules), (
            f"workspaceRoot expansion failed; rules: {rules}"
        )

    def test_absolute_data_path_passthrough(self, tmp_path):
        """Pin: absolute `-data /abs/path` granted as-is."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        abs_data = "/var/lib/jdtls-data"

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "args": ["-data", abs_data],
                }
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        assert f"{abs_data}/    rw," in rules
        assert f"{abs_data}/**  rw," in rules

    def test_no_data_flag_emits_no_data_rules(self, tmp_path):
        """Pin: servers without -data flags emit no data-dir rules
        (still get the binary ix grant from PR-154 — orthogonal)."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "args": ["--stdio"],  # no data flag
                }
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        # binary ix grant still present
        assert any(r.endswith(" ix,") for r in rules)
        # but NO rw rules from a -data flag (the only rw rules
        # should be the debug_log path grants from PR-153)
        data_rw = [r for r in rules if "jdtls-data" in r or "pyright-data" in r]
        assert data_rw == []

    def test_data_flag_without_value_skipped(self, tmp_path):
        """Pin: a trailing `-data` with no following arg is silently
        ignored — composer doesn't crash, just doesn't emit a rule
        for the malformed flag."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "args": ["--stdio", "-data"],  # -data with no value
                }
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        # No data-dir rules emitted; binary ix still present
        assert any(r.endswith(" ix,") for r in rules)
        # No rule should reference the literal "-data" string
        assert not any("-data" in r for r in rules)

    def test_multiple_data_flags_all_granted(self, tmp_path):
        """Pin: a server with multiple data-style flags emits grants
        for each unique path."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "args": [
                        "-data", ".jaato/d1",
                        "--data-dir", ".jaato/d2",
                    ],
                }
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        d1_resolved = str(tmp_path / ".jaato/d1")
        d2_resolved = str(tmp_path / ".jaato/d2")
        assert f"{d1_resolved}/    rw," in rules
        assert f"{d2_resolved}/    rw," in rules

    def test_duplicate_data_paths_dedupe(self, tmp_path):
        """Pin: same path passed twice (e.g. via two server entries)
        emits ONE pair of rules, not two."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        shared_data = "${workspaceRoot}/.jaato/shared-data"

        lsp_json = tmp_path / ".lsp.json"
        lsp_json.write_text(json.dumps({
            "languageServers": {
                "first": {
                    "command": str(stub),
                    "args": ["-data", shared_data],
                },
                "second": {
                    "command": str(stub),  # same binary too
                    "args": ["-data", shared_data],
                },
            }
        }))

        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        # Only ONE pair of rules for the shared path
        resolved = str(tmp_path / ".jaato/shared-data")
        matching = [r for r in rules if resolved in r]
        assert len(matching) == 2, (
            f"expected exactly 2 rules (rw + rw**) for shared path, "
            f"got: {matching}"
        )

    def test_no_workspace_path_no_data_rules(self, tmp_path):
        """Pin: data-dir rules require workspace_path (relative paths
        can't resolve without it).  No grants emitted."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "stub-lsp"
        stub.write_text("#!/bin/sh\n")
        stub.chmod(0o755)

        # write .lsp.json to home so _load_lsp_config_static finds it
        # without workspace_path
        home_lsp = tmp_path / "home.lsp.json"
        home_lsp.write_text(json.dumps({
            "languageServers": {
                "stub": {
                    "command": str(stub),
                    "args": ["-data", ".jaato/jdtls-data"],
                }
            }
        }))

        # Compose with workspace_path=None
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=None,  # type: ignore[arg-type]
            session_id="test",
            config_root=None,
            plugin_config={"config_path": str(home_lsp)},
        )
        # No data-dir rules; relative path can't resolve
        assert not any(".jaato/jdtls-data" in r and r.endswith(" rw,")
                       for r in rules)


class TestExtractFirstDataDirFromArgs:
    """Pin server-0.6.139 fix (PR-156): runtime helper that returns
    the FIRST -data path from args, used by `connect_server` to
    inject TMPDIR into the LSP subprocess env.

    Motivating case (v147 on server 0.6.138): PR-155 apparmor grants
    landed correctly but jdtls Python wrapper still crashed at
    `bin/jdtls.py:74` because `tempfile.gettempdir()` is computed
    EAGERLY as the default-value for the `-data` argparse arg —
    BEFORE argparse parses CLI input.  Even passing `-data <path>`
    on the command line doesn't help; the gettempdir() call fires
    first.  Python's tempfile honors TMPDIR before /tmp, so
    injecting it makes line 74 succeed.

    Upstream jdtls fixed this in commit `d871e83` (Oct 2025) by
    replacing gettempdir() with `$HOME/.cache` on Linux.  Our
    injection is forward-compatible (no-op on post-d871e83 builds).
    """

    def test_returns_data_path_space_separated(self):
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            ["-data", "${workspaceRoot}/.jaato/jdtls-data"],
            workspace_path="/ws",
        )
        assert result == "/ws/.jaato/jdtls-data"

    def test_returns_data_dir_path(self):
        """Pin: `--data-dir` flag also recognised."""
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            ["--data-dir", ".cache/pyright"],
            workspace_path="/ws",
        )
        assert result == "/ws/.cache/pyright"

    def test_returns_equals_form(self):
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            ["-data=.jaato/jdtls"],
            workspace_path="/ws",
        )
        assert result == "/ws/.jaato/jdtls"

    def test_absolute_path_passthrough(self):
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            ["-data", "/abs/data"],
            workspace_path="/ws",
        )
        assert result == "/abs/data"

    def test_no_data_flag_returns_none(self):
        """Pin: servers without -data flag (e.g. just `--stdio`)
        return None — no TMPDIR to inject."""
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            ["--stdio"],
            workspace_path="/ws",
        )
        assert result is None

    def test_no_workspace_returns_none(self):
        """Pin: relative paths cannot resolve without workspace_path.
        Return None rather than emit a wrong path."""
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            ["-data", ".jaato/jdtls"],
            workspace_path=None,
        )
        assert result is None

    def test_returns_first_when_multiple_data_flags(self):
        """Pin: multiple -data flags → return the FIRST one (TMPDIR
        is a single value; multiple flags would only confuse
        the wrapper anyway)."""
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            ["-data", ".jaato/first", "--data-dir", ".jaato/second"],
            workspace_path="/ws",
        )
        assert result == "/ws/.jaato/first"

    def test_trailing_data_flag_no_value_returns_none(self):
        """Pin: `args` ending in `-data` without a value → None
        (malformed; runtime spawn will fail loudly with the same
        EACCES it currently does, which is more diagnosable than
        a silent wrong TMPDIR)."""
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            ["--stdio", "-data"],
            workspace_path="/ws",
        )
        assert result is None

    def test_non_list_args_returns_none(self):
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            None, workspace_path="/ws",
        )
        assert result is None

    def test_workspace_root_variable_expanded(self):
        """Pin: `${workspaceRoot}` expanded same way runtime does."""
        result = LSPToolPlugin._extract_first_data_dir_from_args(
            ["-data", "${workspaceRoot}/.cache/jdtls"],
            workspace_path="/ws/proj",
        )
        # Result has workspace prefix resolved
        assert "/ws/proj" in result
        assert ".cache/jdtls" in result


class TestConnectServerTmpdirInjection:
    """Pin server-0.6.139: connect_server auto-injects TMPDIR into
    the LSP subprocess env when a -data path is detected, respecting
    operator-explicit TMPDIR (never overrides).

    These tests exercise the env-merge logic via source-pinning
    rather than running an actual subprocess (which would require
    mocking asyncio.create_subprocess_exec, the LSP handshake, etc.).
    """

    def test_connect_server_source_calls_extract_helper(self):
        """Pin: connect_server invokes
        `_extract_first_data_dir_from_args` to compute TMPDIR.
        Catches future refactors that accidentally remove the
        TMPDIR injection."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        # Locate the connect_server async def
        start = src.index("async def connect_server(name: str, spec: dict)")
        end = src.find("async def disconnect_server", start)
        if end == -1:
            end = start + 4000
        body = src[start:end]
        assert "_extract_first_data_dir_from_args" in body, (
            "connect_server should call "
            "`LSPToolPlugin._extract_first_data_dir_from_args(...)` "
            "to compute TMPDIR injection (PR-156 / server 0.6.139)."
        )

    def test_connect_server_source_respects_operator_tmpdir(self):
        """Pin: operator-explicit TMPDIR is NEVER overridden by the
        auto-injection (`'TMPDIR' not in augmented_env` check)."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        start = src.index("async def connect_server(name: str, spec: dict)")
        end = src.find("async def disconnect_server", start)
        if end == -1:
            end = start + 4000
        body = src[start:end]
        # The membership check ensures operator-explicit TMPDIR wins
        assert "'TMPDIR' not in augmented_env" in body or \
               '"TMPDIR" not in augmented_env' in body, (
            "connect_server should check `'TMPDIR' not in augmented_env` "
            "before auto-injecting, so operator-explicit TMPDIR is "
            "honored (PR-156)."
        )


class TestWorkspaceRootSymmetricResolution:
    """Pin server-0.6.140 fix (PR-157): symmetric workspace_path
    resolution at connect_server time matches the composer (PR-155).

    Motivating evidence (v148 on server 0.6.139): PR-155 composer
    correctly resolved `${workspaceRoot}` to cascade workspace, but
    PR-156 connect_server resolved to daemon cwd because
    `self._workspace_path` was None at connect time (initial
    auto-connect runs inside `initialize()` BEFORE the framework's
    `set_workspace_path()` broadcast fires after `expose_all()`).

    Fix: (1) defer initial auto-connect when `_workspace_path` is
    None; (2) `set_workspace_path()` dispatches MSG_RETRY_AUTOCONNECT;
    (3) request loop handles it by re-attempting connect for any
    not-yet-connected server with the now-correct workspace_path;
    (4) `expand_variables(raw_args)` call gains
    `workspace_root_override=self._workspace_path` for symmetric
    `${workspaceRoot}` resolution.
    """

    def test_connect_server_passes_workspace_root_override(self):
        """Pin: `expand_variables(raw_args, ...)` call at line 2373
        (post-PR-157) passes `workspace_root_override=self._workspace_path`
        so `${workspaceRoot}` in args resolves to session workspace,
        not daemon cwd auto-detect."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        start = src.index("async def connect_server(name: str, spec: dict)")
        end = src.find("async def disconnect_server", start)
        if end == -1:
            end = start + 6000
        body = src[start:end]
        # The expand_variables call MUST include the override kwarg
        assert "workspace_root_override=self._workspace_path" in body, (
            "connect_server's `expand_variables(raw_args, ...)` call "
            "must pass `workspace_root_override=self._workspace_path` "
            "(PR-157) so `${workspaceRoot}` in args resolves to the "
            "session workspace symmetric with PR-155 composer."
        )

    def test_set_workspace_path_dispatches_retry(self):
        """Pin: set_workspace_path() posts MSG_RETRY_AUTOCONNECT to
        the background thread's request_queue when workspace_path
        changes (and the plugin is initialized)."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        start = src.index("def set_workspace_path(self, path: str)")
        # Find the next def to bound the method
        end = src.find("\n    def ", start + 1)
        if end == -1:
            end = start + 3000
        body = src[start:end]
        assert "MSG_RETRY_AUTOCONNECT" in body, (
            "set_workspace_path should dispatch MSG_RETRY_AUTOCONNECT "
            "via request_queue (PR-157) so the background thread "
            "retries connect for any not-yet-connected server with "
            "the now-correct workspace_path."
        )

    def test_msg_retry_autoconnect_constant_defined(self):
        """Pin: the new message type is exported as a module-level
        constant so source-pin tests can grep for it + the request
        loop handler can dispatch on it."""
        from ..plugin import MSG_RETRY_AUTOCONNECT
        assert MSG_RETRY_AUTOCONNECT == 'retry_autoconnect'

    def test_msg_retry_autoconnect_handler_present(self):
        """Pin: the request loop in _thread_main handles
        MSG_RETRY_AUTOCONNECT. Catches a future refactor that
        accidentally removes the handler."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        # The handler appears in the request loop in _thread_main
        assert "elif msg_type == MSG_RETRY_AUTOCONNECT:" in src, (
            "_thread_main's request loop should have a handler for "
            "MSG_RETRY_AUTOCONNECT (PR-157)."
        )

    def test_initial_autoconnect_deferred_when_no_workspace(self):
        """Pin: the initial auto-connect loop in run_lsp skips when
        self._workspace_path is None. set_workspace_path triggers
        the deferred retry via MSG_RETRY_AUTOCONNECT."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        # Find the auto-connect block in run_lsp
        anchor = "# Auto-connect to configured servers"
        start = src.index(anchor)
        # ~30-40 lines of context
        body = src[start:start + 2000]
        assert "self._workspace_path is None" in body, (
            "run_lsp's auto-connect block should check "
            "`self._workspace_path is None` and defer if so (PR-157)."
        )
        assert "Auto-connect deferred" in body, (
            "run_lsp should log a clear 'Auto-connect deferred' "
            "message when workspace_path isn't yet set."
        )

    def test_retry_handler_clears_failed_servers(self):
        """Pin: the MSG_RETRY_AUTOCONNECT handler clears
        `_failed_servers[name]` BEFORE retrying connect, so a
        previous failure (from the initial run with workspace=None)
        doesn't short-circuit the retry."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        handler_start = src.index("elif msg_type == MSG_RETRY_AUTOCONNECT:")
        handler_end = src.find("except queue.Empty", handler_start)
        body = src[handler_start:handler_end]
        assert "_failed_servers.pop(name, None)" in body, (
            "MSG_RETRY_AUTOCONNECT handler should clear "
            "_failed_servers[name] before retrying (PR-157)."
        )

    def test_retry_handler_does_not_push_response_queue(self):
        """Pin: MSG_RETRY_AUTOCONNECT is fire-and-forget. The handler
        must NOT put on response_queue (would desync the next
        sync request's response read)."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        handler_start = src.index("elif msg_type == MSG_RETRY_AUTOCONNECT:")
        handler_end = src.find("except queue.Empty", handler_start)
        body = src[handler_start:handler_end]
        assert "self._response_queue.put" not in body, (
            "MSG_RETRY_AUTOCONNECT handler must be fire-and-forget — "
            "no response_queue push (PR-157)."
        )


class TestApparmorExtraRulesKnob:
    """Pin server-0.6.141 fix (PR-158): operator-supplied apparmor
    rules via `plugin_configs.lsp.apparmor_extra_rules` in the
    profile YAML.

    **Trust boundary**: this knob is read from `plugin_config` (=
    profile YAML, `.jaato/profiles/`), NOT from the workspace's
    `.lsp.json`.  Probe-confirmed 2026-05-20: `.lsp.json` lives at
    workspace root and is writable from BOTH runner main AND
    //child sub-profile (via `cascade_smoke/** rwkl,`).  An
    LLM-driven tool could inject arbitrary rules into `.lsp.json`
    → cross-session privilege escalation.

    `.jaato/profiles/**` has `audit deny ... wlk,` on both layers
    — operator-only territory.  PR-158 scopes the knob to
    profile YAML deliberately.

    Motivating case (v149, server 0.6.140): PR-157 closed the
    asymmetric resolution layer; jdtls wrapper proceeded past
    argparse and tried `subprocess.check_output(["java",
    "-version"])` — apparmor denied the `java` exec.  Operator
    fix: add `/usr/bin/java ix,` + `/usr/lib/jvm/** r,` to
    `apparmor_extra_rules` in the codegen profile.
    """

    def test_single_rule_emitted_verbatim(self, tmp_path):
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={
                "apparmor_extra_rules": ["/usr/bin/java ix,"],
            },
        )
        assert "/usr/bin/java ix," in rules

    def test_multiple_rules_emitted_in_order(self, tmp_path):
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={
                "apparmor_extra_rules": [
                    "/usr/bin/java ix,",
                    "/usr/lib/jvm/** r,",
                    "/etc/java-*/** r,",
                ],
            },
        )
        assert "/usr/bin/java ix," in rules
        assert "/usr/lib/jvm/** r," in rules
        assert "/etc/java-*/** r," in rules

    def test_workspace_root_expansion(self, tmp_path):
        """Pin: ${workspaceRoot} expands at composer time."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={
                "apparmor_extra_rules": [
                    "${workspaceRoot}/.cache/extra/** rw,",
                ],
            },
        )
        expected = f"{tmp_path}/.cache/extra/** rw,"
        assert expected in rules

    def test_empty_list_yields_no_rules(self, tmp_path):
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={"apparmor_extra_rules": []},
        )
        # Only the debug_log_path rules remain (no extra rules)
        assert not any(r == "/usr/bin/java ix," for r in rules)

    def test_non_list_silently_skipped(self, tmp_path):
        """Pin: non-list value silently skipped (no crash, no
        rules)."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={
                "apparmor_extra_rules": "not a list",
            },
        )
        # No "not a list" appearing as a rule
        assert not any("not a list" in r for r in rules)

    def test_non_string_entries_skipped(self, tmp_path):
        """Pin: list with mixed types — strings emitted, non-string
        entries skipped silently."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={
                "apparmor_extra_rules": [
                    "/usr/bin/java ix,",
                    123,           # not a string
                    None,          # not a string
                    {"key": "value"},  # not a string
                    "/etc/java-*/** r,",
                ],
            },
        )
        assert "/usr/bin/java ix," in rules
        assert "/etc/java-*/** r," in rules

    def test_whitespace_only_entries_skipped(self, tmp_path):
        """Pin: empty / whitespace-only strings skipped."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={
                "apparmor_extra_rules": [
                    "",
                    "   ",
                    "\t\n",
                    "/usr/bin/java ix,",
                ],
            },
        )
        assert "/usr/bin/java ix," in rules
        # No blank rule entries
        assert "" not in rules
        assert "   " not in rules

    def test_duplicate_rules_deduped(self, tmp_path):
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={
                "apparmor_extra_rules": [
                    "/usr/bin/java ix,",
                    "/usr/bin/java ix,",  # duplicate
                    "  /usr/bin/java ix,  ",  # duplicate (whitespace)
                ],
            },
        )
        # Only ONE instance of the rule, even though specified 3x
        count = sum(1 for r in rules if r == "/usr/bin/java ix,")
        assert count == 1, f"expected 1 dedup, got {count}"

    def test_absent_knob_no_rules(self, tmp_path):
        """Pin: plugin_config without apparmor_extra_rules key
        emits zero extra rules (other parts of get_apparmor_rules
        still emit their grants)."""
        rules = LSPToolPlugin.get_apparmor_rules(
            workspace_path=str(tmp_path),
            session_id="test",
            config_root=None,
            plugin_config={},
        )
        # Debug log rules still present (PR-153)
        assert any(".jaato/logs" in r for r in rules)
        # No java grants (no apparmor_extra_rules)
        assert not any("/usr/bin/java" in r for r in rules)

    def test_source_pin_reads_from_plugin_config_not_lsp_json(self):
        """Pin (trust boundary): the `apparmor_extra_rules` source
        is `plugin_config` (profile YAML), NEVER `.lsp.json`.
        Catches a future refactor that accidentally moves the knob
        to `.lsp.json` — which is writable from inside confinement
        and would be a cross-session privilege escalation."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        # Find the helper method
        helper_start = src.index("def _compose_lsp_apparmor_extra_rules(")
        helper_end = src.find("\n    @classmethod", helper_start + 1)
        if helper_end == -1:
            helper_end = src.find("\n    @staticmethod", helper_start + 1)
        if helper_end == -1:
            helper_end = helper_start + 4000
        body = src[helper_start:helper_end]
        assert "plugin_config.get('apparmor_extra_rules')" in body or \
               'plugin_config.get("apparmor_extra_rules")' in body, (
            "_compose_lsp_apparmor_extra_rules MUST read from "
            "plugin_config (profile YAML), NOT from .lsp.json. "
            "Trust boundary: .jaato/profiles/ is write-protected "
            "by audit deny rules; .lsp.json is writable from "
            "//child sub-profile (LLM-driven tools)."
        )
        # Anti-pattern check: helper should NOT call _load_lsp_config_static
        assert "_load_lsp_config_static" not in body, (
            "_compose_lsp_apparmor_extra_rules must NOT read from "
            "`.lsp.json` — that file is writable from inside "
            "confinement.  Cross-session privilege escalation."
        )

    def test_get_apparmor_rules_calls_extra_rules_composer(self):
        """Pin: get_apparmor_rules dispatches to the extra-rules
        composer.  Catches a future refactor that drops the call."""
        from pathlib import Path
        plugin_path = Path(__file__).resolve().parent.parent / "plugin.py"
        src = plugin_path.read_text(encoding="utf-8")
        outer_start = src.index("def get_apparmor_rules(")
        outer_end = src.find("\n    @classmethod", outer_start + 1)
        if outer_end == -1:
            outer_end = outer_start + 4000
        body = src[outer_start:outer_end]
        assert "_compose_lsp_apparmor_extra_rules" in body, (
            "get_apparmor_rules must call "
            "_compose_lsp_apparmor_extra_rules to emit operator-"
            "supplied rules (PR-158)."
        )


class TestDaemonProcessSuppressesLSP:
    """#284/#285: the daemon PROCESS must NOT host a language server.

    The leaking jdtls came from the LSP background thread auto-connecting in the
    long-lived daemon process (no owning slot → never reaped → OOM).  Detection
    is by PROCESS IDENTITY (``_running_in_daemon_process``): the #285 diagnostic
    proved the daemon-side LSP instance has ``_plugin_registry is None`` at
    connect time, so the prior ``registry.runner_rpc`` gate was structurally
    unreachable and never fired.  The per-session runner hosts LSP instead.
    """

    def test_detects_daemon_via_main_package(self, monkeypatch):
        import shared.plugins.lsp.plugin as lspmod
        monkeypatch.setattr(sys.modules["__main__"], "__package__", "server", raising=False)
        assert lspmod._running_in_daemon_process() is True

    def test_detects_runner_via_main_package(self, monkeypatch):
        import shared.plugins.lsp.plugin as lspmod
        monkeypatch.setattr(sys.modules["__main__"], "__package__", "server.runner", raising=False)
        assert lspmod._running_in_daemon_process() is False

    def test_unknown_context_defaults_to_not_daemon(self, monkeypatch):
        """Tests / odd launchers → False (don't suppress unless sure)."""
        import shared.plugins.lsp.plugin as lspmod
        monkeypatch.setattr(sys.modules["__main__"], "__package__", "pytest", raising=False)
        monkeypatch.setattr(sys, "argv", ["/usr/bin/pytest"])
        assert lspmod._running_in_daemon_process() is False

    def test_argv_fallback_detects_daemon(self, monkeypatch):
        import shared.plugins.lsp.plugin as lspmod
        monkeypatch.setattr(sys.modules["__main__"], "__package__", "", raising=False)
        monkeypatch.setattr(sys, "argv", ["/x/jaato-server/server/__main__.py", "--daemon"])
        assert lspmod._running_in_daemon_process() is True

    def test_daemon_does_not_subscribe_to_enrichment(self, monkeypatch):
        import shared.plugins.lsp.plugin as lspmod
        monkeypatch.setattr(lspmod, "_running_in_daemon_process", lambda: True)
        p = LSPToolPlugin()
        assert p._daemon_must_not_host_lsp() is True
        assert p.subscribes_to_tool_result_enrichment() is False

    def test_runner_subscribes_to_enrichment(self, monkeypatch):
        import shared.plugins.lsp.plugin as lspmod
        monkeypatch.setattr(lspmod, "_running_in_daemon_process", lambda: False)
        p = LSPToolPlugin()
        assert p._daemon_must_not_host_lsp() is False
        assert p.subscribes_to_tool_result_enrichment() is True

    def test_daemon_does_not_start_lsp_thread(self, monkeypatch):
        """The fix: ``_ensure_thread`` is a no-op in the daemon process — no
        thread means no auto-connect means no daemon-side jdtls (#284)."""
        import shared.plugins.lsp.plugin as lspmod
        monkeypatch.setattr(lspmod, "_running_in_daemon_process", lambda: True)
        p = LSPToolPlugin()
        p._ensure_thread()
        assert p._thread is None
