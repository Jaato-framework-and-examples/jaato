"""Tests for the MCP tool plugin."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from ..plugin import MCPToolPlugin, create_plugin


class TestMCPPluginInitialization:
    """Tests for plugin initialization."""

    def test_create_plugin_factory(self):
        plugin = create_plugin()
        assert isinstance(plugin, MCPToolPlugin)

    def test_plugin_name(self):
        plugin = MCPToolPlugin()
        assert plugin.name == "mcp"

    def test_initial_state(self):
        plugin = MCPToolPlugin()
        assert plugin._initialized is False
        assert plugin._tool_cache == {}
        assert plugin._thread is None

    def test_shutdown_before_initialize(self):
        """Shutdown should work even if never initialized."""
        plugin = MCPToolPlugin()
        plugin.shutdown()  # Should not raise

        assert plugin._initialized is False
        assert plugin._tool_cache == {}


class TestMCPPluginFunctionDeclarations:
    """Tests for function declarations."""

    def test_get_tool_schemas_empty(self):
        """Without MCP servers, should return empty list."""
        plugin = MCPToolPlugin()
        # Don't initialize - just check the cache-based behavior
        plugin._initialized = True  # Skip actual initialization
        schemas = plugin.get_tool_schemas()

        assert schemas == []

    def test_get_executors_empty(self):
        """Without MCP servers, should only return the mcp user command executor."""
        plugin = MCPToolPlugin()
        plugin._initialized = True  # Skip actual initialization
        executors = plugin.get_executors()

        # Only the 'mcp' user command executor should be present
        assert 'mcp' in executors
        assert len(executors) == 1


class TestMCPPluginSystemInstructions:
    """Tests for system instructions."""

    def test_get_system_instructions_empty(self):
        """Without MCP tools, should return None."""
        plugin = MCPToolPlugin()
        plugin._initialized = True  # Skip actual initialization
        instructions = plugin.get_system_instructions()

        assert instructions is None

    def test_get_auto_approved_tools(self):
        """User commands should be auto-approved."""
        plugin = MCPToolPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        # User commands are auto-approved since they're invoked by the user
        assert 'mcp' in auto_approved


class TestMCPPluginUserCommands:
    """Tests for user-facing commands."""

    def test_get_user_commands(self):
        """Should return the mcp command."""
        plugin = MCPToolPlugin()
        commands = plugin.get_user_commands()

        assert len(commands) == 1
        assert commands[0].name == 'mcp'
        assert commands[0].share_with_model is False
        assert len(commands[0].parameters) == 2

    def test_execute_user_command_help(self):
        """Help subcommand should return help text."""
        plugin = MCPToolPlugin()
        result = plugin.execute_user_command('mcp', {'subcommand': 'help'})

        assert 'MCP Server Configuration Commands' in result
        assert 'mcp list' in result
        assert 'mcp add' in result

    def test_execute_user_command_empty(self):
        """Empty subcommand should return help text."""
        plugin = MCPToolPlugin()
        result = plugin.execute_user_command('mcp', {'subcommand': ''})

        assert 'MCP Server Configuration Commands' in result

    def test_execute_user_command_unknown_subcommand(self):
        """Unknown subcommand should return error and help."""
        plugin = MCPToolPlugin()
        result = plugin.execute_user_command('mcp', {'subcommand': 'unknown'})

        assert 'Unknown subcommand: unknown' in result
        assert 'MCP Server Configuration Commands' in result

    def test_execute_user_command_unknown_command(self):
        """Unknown command should return error."""
        plugin = MCPToolPlugin()
        result = plugin.execute_user_command('notmcp', {'subcommand': 'list'})

        assert 'Unknown command: notmcp' in result

    def test_cmd_list_empty(self):
        """List should show message when no servers configured."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {'mcpServers': {}}
        result = plugin._cmd_list()

        assert 'No MCP servers configured' in result

    def test_cmd_list_with_servers(self):
        """List should show configured servers."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {
            'mcpServers': {
                'TestServer': {'command': '/usr/bin/test-mcp'}
            }
        }
        plugin._connected_servers = set()
        plugin._failed_servers = {}
        result = plugin._cmd_list()

        assert 'TestServer' in result
        assert '/usr/bin/test-mcp' in result
        assert 'disconnected' in result

    def test_cmd_list_connected_server(self):
        """List should show connected status."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {
            'mcpServers': {
                'TestServer': {'command': '/usr/bin/test-mcp'}
            }
        }
        plugin._connected_servers = {'TestServer'}
        plugin._failed_servers = {}
        result = plugin._cmd_list()

        assert 'connected' in result

    def test_cmd_list_failed_server(self):
        """List should show failed status with error."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {
            'mcpServers': {
                'TestServer': {'command': '/usr/bin/test-mcp'}
            }
        }
        plugin._connected_servers = set()
        plugin._failed_servers = {'TestServer': 'Connection refused'}
        result = plugin._cmd_list()

        assert 'failed' in result
        assert 'Connection refused' in result

    def test_cmd_show_no_server_name(self):
        """Show without server name should return usage."""
        plugin = MCPToolPlugin()
        result = plugin._cmd_show('')

        assert 'Usage: mcp show <server_name>' in result

    def test_cmd_show_not_found(self):
        """Show for non-existent server should return error."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {'mcpServers': {}}
        result = plugin._cmd_show('NonExistent')

        assert "not found in configuration" in result

    def test_cmd_show_existing(self):
        """Show should display server configuration."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {
            'mcpServers': {
                'TestServer': {
                    'command': '/usr/bin/test-mcp',
                    'args': ['--verbose'],
                    'env': {'API_KEY': '${SECRET_KEY}'}
                }
            }
        }
        plugin._connected_servers = set()
        plugin._failed_servers = {}
        result = plugin._cmd_show('TestServer')

        assert 'TestServer' in result
        assert '/usr/bin/test-mcp' in result
        assert '--verbose' in result
        assert 'API_KEY' in result
        assert 'DISCONNECTED' in result

    def test_cmd_add_no_args(self):
        """Add without arguments should return usage."""
        plugin = MCPToolPlugin()
        result = plugin._cmd_add('')

        assert 'Usage: mcp add' in result

    def test_cmd_add_missing_command(self):
        """Add with only name should return usage."""
        plugin = MCPToolPlugin()
        result = plugin._cmd_add('servername')

        assert 'Usage: mcp add' in result

    def test_cmd_add_existing_server(self):
        """Add should fail if server already exists."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {
            'mcpServers': {
                'TestServer': {'command': 'test'}
            }
        }
        result = plugin._cmd_add('TestServer /usr/bin/new-mcp')

        assert 'already exists' in result

    def test_cmd_add_success(self):
        """Add should create new server entry."""
        plugin = MCPToolPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, '.mcp.json')
            plugin._config_path = config_path
            plugin._config_cache = {'mcpServers': {}}

            result = plugin._cmd_add('NewServer /usr/bin/new-mcp --verbose')

            assert 'Added MCP server' in result
            assert 'NewServer' in result
            assert plugin._config_cache['mcpServers']['NewServer']['command'] == '/usr/bin/new-mcp'
            assert plugin._config_cache['mcpServers']['NewServer']['args'] == ['--verbose']

    def test_cmd_remove_no_server_name(self):
        """Remove without server name should return usage."""
        plugin = MCPToolPlugin()
        result = plugin._cmd_remove('')

        assert 'Usage: mcp remove' in result

    def test_cmd_remove_not_found(self):
        """Remove for non-existent server should return error."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {'mcpServers': {}}
        result = plugin._cmd_remove('NonExistent')

        assert 'not found' in result

    def test_cmd_remove_success(self):
        """Remove should delete server from config."""
        plugin = MCPToolPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, '.mcp.json')
            plugin._config_path = config_path
            plugin._config_cache = {
                'mcpServers': {
                    'ToRemove': {'command': 'test'}
                }
            }
            plugin._connected_servers = set()
            plugin._failed_servers = {}

            result = plugin._cmd_remove('ToRemove')

            assert 'Removed MCP server' in result
            assert 'ToRemove' not in plugin._config_cache['mcpServers']

    def test_cmd_status_empty(self):
        """Status with no servers should show message."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {'mcpServers': {}}
        result = plugin._cmd_status()

        assert 'No MCP servers configured' in result

    def test_cmd_status_with_connected(self):
        """Status should show connected servers with tool count."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {
            'mcpServers': {
                'TestServer': {'command': 'test'}
            }
        }
        plugin._connected_servers = {'TestServer'}
        plugin._failed_servers = {}
        # Create mock tools
        mock_tool = MagicMock()
        mock_tool.name = 'test_tool'
        plugin._tool_cache = {'TestServer': [mock_tool]}

        result = plugin._cmd_status()

        assert 'CONNECTED' in result
        assert '1 tools' in result
        assert 'test_tool' in result

    def test_cmd_connect_no_server_name(self):
        """Connect without server name should return usage."""
        plugin = MCPToolPlugin()
        result = plugin._cmd_connect('')

        assert 'Usage: mcp connect' in result

    def test_cmd_connect_not_found(self):
        """Connect to non-existent server should return error."""
        plugin = MCPToolPlugin()
        plugin._initialized = True
        plugin._config_cache = {'mcpServers': {}}
        result = plugin._cmd_connect('NonExistent')

        assert 'not found in configuration' in result

    def test_cmd_connect_already_connected(self):
        """Connect to already connected server should return message."""
        plugin = MCPToolPlugin()
        plugin._initialized = True
        plugin._config_cache = {
            'mcpServers': {
                'TestServer': {'command': 'test'}
            }
        }
        plugin._connected_servers = {'TestServer'}
        result = plugin._cmd_connect('TestServer')

        assert 'already connected' in result

    def test_cmd_disconnect_no_server_name(self):
        """Disconnect without server name should return usage."""
        plugin = MCPToolPlugin()
        result = plugin._cmd_disconnect('')

        assert 'Usage: mcp disconnect' in result

    def test_cmd_disconnect_not_connected(self):
        """Disconnect from non-connected server should return message."""
        plugin = MCPToolPlugin()
        plugin._connected_servers = set()
        result = plugin._cmd_disconnect('TestServer')

        assert 'not connected' in result


class TestMCPPluginCommandCompletions:
    """Tests for command completions."""

    def test_completions_wrong_command(self):
        """Completions for wrong command should be empty."""
        plugin = MCPToolPlugin()
        completions = plugin.get_command_completions('notmcp', [])

        assert completions == []

    def test_completions_no_args(self):
        """Completions with no args should show all subcommands."""
        plugin = MCPToolPlugin()
        completions = plugin.get_command_completions('mcp', [])

        values = [c.value for c in completions]
        assert 'list' in values
        assert 'show' in values
        assert 'add' in values
        assert 'remove' in values
        assert 'connect' in values
        assert 'disconnect' in values
        assert 'reload' in values
        assert 'status' in values

    def test_completions_partial_subcommand(self):
        """Completions with partial subcommand should filter."""
        plugin = MCPToolPlugin()
        completions = plugin.get_command_completions('mcp', ['li'])

        values = [c.value for c in completions]
        assert 'list' in values
        assert 'show' not in values

    def test_completions_show_servers(self):
        """Completions for show should list servers."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {
            'mcpServers': {
                'Server1': {'command': 'test1'},
                'Server2': {'command': 'test2'}
            }
        }
        plugin._connected_servers = set()
        completions = plugin.get_command_completions('mcp', ['show', ''])

        values = [c.value for c in completions]
        assert 'Server1' in values
        assert 'Server2' in values

    def test_completions_connect_filters_connected(self):
        """Connect completions should not include already connected servers."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {
            'mcpServers': {
                'Connected': {'command': 'test1'},
                'Disconnected': {'command': 'test2'}
            }
        }
        plugin._connected_servers = {'Connected'}
        completions = plugin.get_command_completions('mcp', ['connect', ''])

        values = [c.value for c in completions]
        assert 'Connected' not in values
        assert 'Disconnected' in values

    def test_completions_disconnect_filters_disconnected(self):
        """Disconnect completions should only include connected servers."""
        plugin = MCPToolPlugin()
        plugin._config_cache = {
            'mcpServers': {
                'Connected': {'command': 'test1'},
                'Disconnected': {'command': 'test2'}
            }
        }
        plugin._connected_servers = {'Connected'}
        completions = plugin.get_command_completions('mcp', ['disconnect', ''])

        values = [c.value for c in completions]
        assert 'Connected' in values
        assert 'Disconnected' not in values


class TestMCPPluginConfigSave:
    """Tests for configuration saving."""

    def test_save_config_success(self):
        """Config should be saved to file."""
        plugin = MCPToolPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, '.mcp.json')
            plugin._config_path = config_path
            plugin._config_cache = {
                'mcpServers': {
                    'TestServer': {
                        'type': 'stdio',
                        'command': '/usr/bin/test-mcp',
                        'args': ['--verbose']
                    }
                }
            }

            result = plugin._save_config()

            assert result is None  # Success returns None
            assert os.path.exists(config_path)

            with open(config_path, 'r') as f:
                saved = json.load(f)
            assert 'mcpServers' in saved
            assert 'TestServer' in saved['mcpServers']

    def test_save_config_default_path(self):
        """Config should be saved to .mcp.json in current directory if no path set."""
        plugin = MCPToolPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                plugin._config_path = None
                plugin._config_cache = {'mcpServers': {}}

                result = plugin._save_config()

                assert result is None
                assert os.path.exists('.mcp.json')
            finally:
                os.chdir(old_cwd)


class TestMCPPluginSchemaClean:
    """Tests for schema cleaning utility."""

    def test_clean_schema_removes_unsupported_fields(self):
        plugin = MCPToolPlugin()
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "test",
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        cleaned = plugin._clean_schema_for_vertex(schema)

        assert "$schema" not in cleaned
        assert "$id" not in cleaned
        assert cleaned["type"] == "object"
        assert "name" in cleaned["properties"]

    def test_clean_schema_nested_properties(self):
        plugin = MCPToolPlugin()
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "$ref": "#/definitions/Something",
                    "type": "object",
                    "properties": {
                        "deep": {"type": "string"}
                    }
                }
            }
        }
        cleaned = plugin._clean_schema_for_vertex(schema)

        assert "$ref" not in cleaned["properties"]["nested"]
        assert cleaned["properties"]["nested"]["type"] == "object"

    def test_clean_schema_items(self):
        plugin = MCPToolPlugin()
        schema = {
            "type": "array",
            "items": {
                "$ref": "#/definitions/Item",
                "type": "string"
            }
        }
        cleaned = plugin._clean_schema_for_vertex(schema)

        assert "$ref" not in cleaned["items"]
        assert cleaned["items"]["type"] == "string"

    def test_clean_schema_non_dict(self):
        """Non-dict values should be returned as-is."""
        plugin = MCPToolPlugin()
        assert plugin._clean_schema_for_vertex("string") == "string"
        assert plugin._clean_schema_for_vertex(123) == 123
        assert plugin._clean_schema_for_vertex(None) is None


class TestMCPPluginRegistryLoading:
    """Tests for registry loading."""

    def test_load_registry_no_file(self):
        """Should return empty dict when no config file exists."""
        plugin = MCPToolPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                registry = plugin._load_mcp_registry()
                assert registry == {}
            finally:
                os.chdir(old_cwd)

    def test_load_registry_from_file(self):
        """Should load registry from .mcp.json."""
        plugin = MCPToolPlugin()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "mcpServers": {
                    "Test": {
                        "type": "stdio",
                        "command": "test-mcp"
                    }
                }
            }
            config_path = os.path.join(tmpdir, '.mcp.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                registry = plugin._load_mcp_registry()
                assert "mcpServers" in registry
                assert "Test" in registry["mcpServers"]
            finally:
                os.chdir(old_cwd)
