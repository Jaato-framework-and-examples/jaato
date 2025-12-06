"""Tests for clarification plugin integration with PluginRegistry."""

import pytest

from ...registry import PluginRegistry
from ..plugin import ClarificationPlugin


class TestClarificationPluginDiscovery:
    """Tests for plugin discovery via registry."""

    def test_discover_clarification_plugin(self):
        registry = PluginRegistry()
        available = registry.discover()

        assert "clarification" in available

    def test_plugin_kind_is_tool(self):
        from .. import PLUGIN_KIND

        assert PLUGIN_KIND == "tool"


class TestClarificationPluginExpose:
    """Tests for exposing the clarification plugin."""

    def test_expose_clarification_plugin(self):
        registry = PluginRegistry()
        registry.discover()
        registry.expose_tool("clarification", {"actor_type": "auto"})

        exposed = registry.list_exposed()
        assert "clarification" in exposed

    def test_unexpose_clarification_plugin(self):
        registry = PluginRegistry()
        registry.discover()
        registry.expose_tool("clarification", {"actor_type": "auto"})
        registry.unexpose_tool("clarification")

        exposed = registry.list_exposed()
        assert "clarification" not in exposed


class TestClarificationPluginDeclarations:
    """Tests for getting declarations from registry."""

    def test_get_declarations(self):
        registry = PluginRegistry()
        registry.discover()
        registry.expose_tool("clarification", {"actor_type": "auto"})

        declarations = registry.get_exposed_declarations()
        names = [d.name for d in declarations]

        assert "request_clarification" in names


class TestClarificationPluginExecutors:
    """Tests for getting executors from registry."""

    def test_get_executors(self):
        registry = PluginRegistry()
        registry.discover()
        registry.expose_tool("clarification", {"actor_type": "auto"})

        executors = registry.get_exposed_executors()

        assert "request_clarification" in executors
        assert callable(executors["request_clarification"])

    def test_execute_via_registry(self):
        registry = PluginRegistry()
        registry.discover()
        registry.expose_tool("clarification", {"actor_type": "auto"})

        executors = registry.get_exposed_executors()
        result = executors["request_clarification"]({
            "context": "Test",
            "questions": [
                {
                    "text": "Question",
                    "choices": ["A"],
                },
            ],
        })

        assert "error" not in result
        assert "responses" in result


class TestClarificationPluginSystemInstructions:
    """Tests for system instructions from registry."""

    def test_get_system_instructions(self):
        registry = PluginRegistry()
        registry.discover()
        registry.expose_tool("clarification", {"actor_type": "auto"})

        instructions = registry.get_system_instructions()

        assert instructions is not None
        assert "clarification" in instructions.lower() or "request_clarification" in instructions


class TestClarificationPluginForTool:
    """Tests for getting plugin by tool name."""

    def test_get_plugin_for_tool(self):
        registry = PluginRegistry()
        registry.discover()
        registry.expose_tool("clarification", {"actor_type": "auto"})

        plugin = registry.get_plugin_for_tool("request_clarification")

        assert plugin is not None
        assert isinstance(plugin, ClarificationPlugin)

    def test_get_plugin_for_unknown_tool(self):
        registry = PluginRegistry()
        registry.discover()
        registry.expose_tool("clarification", {"actor_type": "auto"})

        plugin = registry.get_plugin_for_tool("unknown_tool")

        assert plugin is None
