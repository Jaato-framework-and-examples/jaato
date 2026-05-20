"""Tests for LSP plugin."""

import asyncio
import json
import os
import pytest
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
        assert cmd.share_with_model is True
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
    initialize + workspace import easily exceeds 30s.  The knob lets
    profiles tune per workspace.
    """

    def test_default_is_thirty_seconds(self):
        """Pin: unconfigured plugin uses the documented default."""
        from ..plugin import DEFAULT_CONNECT_TIMEOUT_SECONDS
        plugin = LSPToolPlugin()
        # We must not touch _ensure_thread; just verify initial state +
        # configure() does the right thing without starting LSP servers.
        assert plugin._connect_timeout_seconds == DEFAULT_CONNECT_TIMEOUT_SECONDS
        assert DEFAULT_CONNECT_TIMEOUT_SECONDS == 30.0

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
        assert entry.default == 30.0


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
