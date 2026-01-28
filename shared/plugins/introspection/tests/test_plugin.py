"""Unit tests for the Tool Discovery & Introspection Plugin."""

import sys
import pytest
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field


# Import ToolSchema and TOOL_CATEGORIES directly to avoid shared/__init__.py imports
sys.path.insert(0, '/home/user/jaato')
from shared.plugins.model_provider.types import ToolSchema, TOOL_CATEGORIES
from shared.plugins.introspection.plugin import IntrospectionPlugin, create_plugin


class MockPlugin:
    """A mock plugin for testing."""

    def __init__(self, plugin_name: str, tools: List[ToolSchema]):
        self._name = plugin_name
        self._tools = tools

    @property
    def name(self) -> str:
        return self._name

    def get_tool_schemas(self) -> List[ToolSchema]:
        return self._tools

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        return {tool.name: lambda args: {"result": "ok"} for tool in self._tools}

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_system_instructions(self) -> Optional[str]:
        return None

    def get_auto_approved_tools(self) -> List[str]:
        return []

    def get_user_commands(self):
        return []


class MockRegistry:
    """A mock PluginRegistry for testing."""

    def __init__(self):
        self._plugins: Dict[str, MockPlugin] = {}
        self._exposed: set = set()
        self._disabled_tools: set = set()

    def register_plugin(self, plugin: MockPlugin) -> None:
        self._plugins[plugin.name] = plugin
        self._exposed.add(plugin.name)

    def get_exposed_tool_schemas(self) -> List[ToolSchema]:
        schemas = []
        for name in self._exposed:
            schemas.extend(self._plugins[name].get_tool_schemas())
        return schemas

    def get_plugin_for_tool(self, tool_name: str):
        for plugin in self._plugins.values():
            for schema in plugin.get_tool_schemas():
                if schema.name == tool_name:
                    return plugin
        return None

    def is_tool_enabled(self, tool_name: str) -> bool:
        return tool_name not in self._disabled_tools

    def disable_tool(self, tool_name: str) -> None:
        self._disabled_tools.add(tool_name)


class TestIntrospectionPlugin:
    """Tests for IntrospectionPlugin."""

    def test_create_plugin(self):
        """Test factory function creates plugin correctly."""
        plugin = create_plugin()
        assert plugin is not None
        assert isinstance(plugin, IntrospectionPlugin)
        assert plugin.name == "introspection"

    def test_initialize_and_shutdown(self):
        """Test plugin initialization and shutdown."""
        plugin = create_plugin()
        plugin.initialize({})
        assert plugin._initialized is True

        plugin.shutdown()
        assert plugin._initialized is False

    def test_get_tool_schemas(self):
        """Test that plugin exposes list_tools and get_tool_schemas."""
        plugin = create_plugin()
        schemas = plugin.get_tool_schemas()

        assert len(schemas) == 2
        schema_names = [s.name for s in schemas]
        assert "list_tools" in schema_names
        assert "get_tool_schemas" in schema_names

        # Verify schemas have correct categories and discoverability
        for schema in schemas:
            assert schema.category == "system"
            assert schema.discoverability == "core"

    def test_get_executors(self):
        """Test that plugin provides executors for both tools."""
        plugin = create_plugin()
        executors = plugin.get_executors()

        assert "list_tools" in executors
        assert "get_tool_schemas" in executors
        assert callable(executors["list_tools"])
        assert callable(executors["get_tool_schemas"])

    def test_auto_approved_tools(self):
        """Test that introspection tools are auto-approved."""
        plugin = create_plugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "list_tools" in auto_approved
        assert "get_tool_schemas" in auto_approved


class TestListTools:
    """Tests for the list_tools tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = create_plugin()
        self.plugin.initialize({})

        # Create mock registry with test tools
        self.registry = MockRegistry()

        # Register a filesystem plugin
        fs_plugin = MockPlugin("file_edit", [
            ToolSchema(
                name="readFile",
                description="Read a file from disk. Returns content as text.",
                parameters={"type": "object", "properties": {}},
                category="filesystem",
            ),
            ToolSchema(
                name="writeFile",
                description="Write content to a file. Creates or overwrites.",
                parameters={"type": "object", "properties": {}},
                category="filesystem",
            ),
        ])
        self.registry.register_plugin(fs_plugin)

        # Register a search plugin
        search_plugin = MockPlugin("web_search", [
            ToolSchema(
                name="web_search",
                description="Search the web for information on any topic.",
                parameters={"type": "object", "properties": {}},
                category="search",
            ),
        ])
        self.registry.register_plugin(search_plugin)

        # Register a planning plugin
        planning_plugin = MockPlugin("todo", [
            ToolSchema(
                name="createPlan",
                description="Create a new execution plan with steps.",
                parameters={"type": "object", "properties": {}},
                category="coordination",
            ),
        ])
        self.registry.register_plugin(planning_plugin)

        # Wire up plugin with registry
        self.plugin.set_plugin_registry(self.registry)

    def test_list_tools_no_category_returns_categories(self):
        """Test list_tools without category returns category summary."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({})

        assert "categories" in result
        assert "total_tools" in result
        assert result["total_tools"] == 4

        # Check categories are listed with counts
        category_names = [c["name"] for c in result["categories"]]
        assert "filesystem" in category_names
        assert "search" in category_names
        assert "planning" in category_names

        # Check counts
        fs_cat = next(c for c in result["categories"] if c["name"] == "filesystem")
        assert fs_cat["tool_count"] == 2

    def test_list_tools_with_category_returns_tools(self):
        """Test list_tools with category returns tools in that category."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"category": "filesystem"})

        assert result["category"] == "filesystem"
        assert result["tool_count"] == 2
        assert "tools" in result

        tool_names = [t["name"] for t in result["tools"]]
        assert "readFile" in tool_names
        assert "writeFile" in tool_names

    def test_list_tools_includes_plugin_source(self):
        """Test list_tools includes plugin source for each tool."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"category": "filesystem"})

        for tool in result["tools"]:
            assert "plugin_source" in tool
            assert tool["plugin_source"] == "file_edit"

    def test_list_tools_concise_descriptions(self):
        """Test list_tools truncates long descriptions by default."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"category": "filesystem", "verbose": False})

        for tool in result["tools"]:
            assert "description" in tool
            # Descriptions should be truncated to first sentence or 100 chars
            assert len(tool["description"]) <= 150

    def test_list_tools_verbose_mode(self):
        """Test list_tools returns full descriptions in verbose mode."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"category": "search", "verbose": True})

        # Find the search tool and check its description is complete
        search_tool = result["tools"][0]
        assert search_tool["description"] == "Search the web for information on any topic."

    def test_list_tools_includes_enabled_status(self):
        """Test list_tools includes enabled/disabled status."""
        # Disable a tool
        self.registry.disable_tool("readFile")

        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"category": "filesystem"})

        for tool in result["tools"]:
            assert "enabled" in tool
            if tool["name"] == "readFile":
                assert tool["enabled"] is False
            else:
                assert tool["enabled"] is True

    def test_list_tools_sorted_by_name(self):
        """Test list_tools returns tools sorted by name."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"category": "filesystem"})

        tools = result["tools"]
        names = [t["name"] for t in tools]
        assert names == sorted(names)

    def test_list_tools_no_registry_error(self):
        """Test list_tools returns error when registry not set."""
        plugin = create_plugin()
        plugin.initialize({})
        # Don't set registry

        executors = plugin.get_executors()
        result = executors["list_tools"]({})

        assert "error" in result


class TestGetToolSchemas:
    """Tests for the get_tool_schemas tool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = create_plugin()
        self.plugin.initialize({})

        self.registry = MockRegistry()

        # Register a plugin with detailed schema
        test_plugin = MockPlugin("test_plugin", [
            ToolSchema(
                name="testTool",
                description="A test tool for unit testing.",
                parameters={
                    "type": "object",
                    "properties": {
                        "required_arg": {
                            "type": "string",
                            "description": "A required string argument."
                        },
                        "optional_arg": {
                            "type": "integer",
                            "description": "An optional integer argument.",
                            "default": 10
                        },
                        "enum_arg": {
                            "type": "string",
                            "enum": ["option1", "option2", "option3"],
                            "description": "An argument with allowed values."
                        },
                        "array_arg": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array of strings."
                        }
                    },
                    "required": ["required_arg"]
                },
                category="code",
            ),
            ToolSchema(
                name="anotherTool",
                description="Another test tool.",
                parameters={"type": "object", "properties": {}},
                category="code",
            ),
        ])
        self.registry.register_plugin(test_plugin)
        self.plugin.set_plugin_registry(self.registry)

    def test_get_tool_schemas_returns_full_details(self):
        """Test get_tool_schemas returns complete tool information."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schemas"]({"names": ["testTool"]})

        assert "schemas" in result
        assert result["count"] == 1
        schema = result["schemas"][0]
        assert schema["name"] == "testTool"
        assert schema["description"] == "A test tool for unit testing."
        assert schema["plugin_source"] == "test_plugin"
        assert schema["category"] == "code"
        assert schema["enabled"] is True
        assert "parameters" in schema

    def test_get_tool_schemas_multiple_tools(self):
        """Test get_tool_schemas handles multiple tool requests."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schemas"]({"names": ["testTool", "anotherTool"]})

        assert result["count"] == 2
        names = [s["name"] for s in result["schemas"]]
        assert "testTool" in names
        assert "anotherTool" in names

    def test_get_tool_schemas_formats_parameters(self):
        """Test get_tool_schemas formats parameters in readable structure."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schemas"]({"names": ["testTool"]})

        params = result["schemas"][0]["parameters"]

        # Check required arg
        assert "required_arg" in params
        assert params["required_arg"]["type"] == "string"
        assert params["required_arg"]["required"] is True
        assert "description" in params["required_arg"]

        # Check optional arg with default
        assert "optional_arg" in params
        assert params["optional_arg"]["required"] is False
        assert params["optional_arg"]["default"] == 10

        # Check enum arg
        assert "enum_arg" in params
        assert params["enum_arg"]["allowed_values"] == ["option1", "option2", "option3"]

        # Check array arg
        assert "array_arg" in params
        assert params["array_arg"]["items_type"] == "string"

    def test_get_tool_schemas_partial_not_found(self):
        """Test get_tool_schemas handles mix of found and not found tools."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schemas"]({"names": ["testTool", "nonExistent"]})

        assert result["count"] == 1
        assert "not_found" in result
        assert "nonExistent" in result["not_found"]

    def test_get_tool_schemas_suggests_similar_tools(self):
        """Test get_tool_schemas suggests similar tools when not found."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schemas"]({"names": ["test"]})

        assert "not_found" in result
        assert "suggestions" in result
        # Should suggest testTool as it contains "test"
        assert "testTool" in result["suggestions"]["test"]

    def test_get_tool_schemas_requires_names(self):
        """Test get_tool_schemas returns error without names."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schemas"]({})

        assert "error" in result
        assert "required" in result["error"].lower()

    def test_get_tool_schemas_no_registry_error(self):
        """Test get_tool_schemas returns error when registry not set."""
        plugin = create_plugin()
        plugin.initialize({})
        # Don't set registry

        executors = plugin.get_executors()
        result = executors["get_tool_schemas"]({"names": ["testTool"]})

        assert "error" in result

    def test_get_tool_schemas_tracks_accessed_tools(self):
        """Test that get_tool_schemas tracks which tools were accessed."""
        executors = self.plugin.get_executors()

        # Initially no tools accessed
        assert len(self.plugin.get_accessed_tools()) == 0

        # Access some tools
        executors["get_tool_schemas"]({"names": ["testTool"]})
        assert "testTool" in self.plugin.get_accessed_tools()

        # Access more tools
        executors["get_tool_schemas"]({"names": ["anotherTool"]})
        accessed = self.plugin.get_accessed_tools()
        assert "testTool" in accessed
        assert "anotherTool" in accessed

    def test_clear_accessed_tools(self):
        """Test that accessed tools can be cleared."""
        executors = self.plugin.get_executors()
        executors["get_tool_schemas"]({"names": ["testTool"]})

        assert len(self.plugin.get_accessed_tools()) == 1
        self.plugin.clear_accessed_tools()
        assert len(self.plugin.get_accessed_tools()) == 0


class TestToolCategories:
    """Tests for tool category constants."""

    def test_standard_categories_defined(self):
        """Test that standard categories are defined."""
        assert "filesystem" in TOOL_CATEGORIES
        assert "code" in TOOL_CATEGORIES
        assert "search" in TOOL_CATEGORIES
        assert "memory" in TOOL_CATEGORIES
        assert "coordination" in TOOL_CATEGORIES
        assert "system" in TOOL_CATEGORIES
        assert "web" in TOOL_CATEGORIES
        assert "communication" in TOOL_CATEGORIES

    def test_tool_schema_accepts_category(self):
        """Test that ToolSchema accepts category field."""
        schema = ToolSchema(
            name="test",
            description="Test tool",
            parameters={},
            category="filesystem",
        )
        assert schema.category == "filesystem"

    def test_tool_schema_category_optional(self):
        """Test that ToolSchema category is optional."""
        schema = ToolSchema(
            name="test",
            description="Test tool",
            parameters={},
        )
        assert schema.category is None


class TestToolDiscoverability:
    """Tests for tool discoverability feature."""

    def test_tool_schema_discoverability_default(self):
        """Test that ToolSchema discoverability defaults to 'discoverable'."""
        schema = ToolSchema(
            name="test",
            description="Test tool",
            parameters={},
        )
        assert schema.discoverability == "discoverable"

    def test_tool_schema_discoverability_core(self):
        """Test that ToolSchema accepts 'core' discoverability."""
        schema = ToolSchema(
            name="test",
            description="Test tool",
            parameters={},
            discoverability="core",
        )
        assert schema.discoverability == "core"

    def test_introspection_tools_are_core(self):
        """Test that introspection plugin tools are marked as core."""
        plugin = create_plugin()
        schemas = plugin.get_tool_schemas()

        for schema in schemas:
            assert schema.discoverability == "core", \
                f"Tool {schema.name} should be core but is {schema.discoverability}"
