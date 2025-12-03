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

    def test_get_function_declarations_empty(self):
        """Without MCP servers, should return empty list."""
        plugin = MCPToolPlugin()
        # Don't initialize - just check the cache-based behavior
        plugin._initialized = True  # Skip actual initialization
        declarations = plugin.get_function_declarations()

        assert declarations == []

    def test_get_executors_empty(self):
        """Without MCP servers, should return empty dict."""
        plugin = MCPToolPlugin()
        plugin._initialized = True  # Skip actual initialization
        executors = plugin.get_executors()

        assert executors == {}


class TestMCPPluginSystemInstructions:
    """Tests for system instructions."""

    def test_get_system_instructions_empty(self):
        """Without MCP tools, should return None."""
        plugin = MCPToolPlugin()
        plugin._initialized = True  # Skip actual initialization
        instructions = plugin.get_system_instructions()

        assert instructions is None

    def test_get_auto_approved_tools(self):
        """MCP tools require permission - should return empty list."""
        plugin = MCPToolPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert auto_approved == []


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
