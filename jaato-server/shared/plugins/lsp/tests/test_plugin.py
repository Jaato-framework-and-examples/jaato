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
