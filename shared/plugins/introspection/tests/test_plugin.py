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
        """Test that plugin exposes list_tools and get_tool_schema."""
        plugin = create_plugin()
        schemas = plugin.get_tool_schemas()

        assert len(schemas) == 2
        schema_names = [s.name for s in schemas]
        assert "list_tools" in schema_names
        assert "get_tool_schema" in schema_names

        # Verify schemas have correct categories
        for schema in schemas:
            assert schema.category == "system"

    def test_get_executors(self):
        """Test that plugin provides executors for both tools."""
        plugin = create_plugin()
        executors = plugin.get_executors()

        assert "list_tools" in executors
        assert "get_tool_schema" in executors
        assert callable(executors["list_tools"])
        assert callable(executors["get_tool_schema"])

    def test_auto_approved_tools(self):
        """Test that introspection tools are auto-approved."""
        plugin = create_plugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "list_tools" in auto_approved
        assert "get_tool_schema" in auto_approved


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
                category="planning",
            ),
        ])
        self.registry.register_plugin(planning_plugin)

        # Wire up plugin with registry
        self.plugin.set_plugin_registry(self.registry)

    def test_list_tools_returns_all_tools(self):
        """Test list_tools returns all registered tools."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({})

        assert "tools" in result
        assert result["total_count"] == 4
        tool_names = [t["name"] for t in result["tools"]]
        assert "readFile" in tool_names
        assert "writeFile" in tool_names
        assert "web_search" in tool_names
        assert "createPlan" in tool_names

    def test_list_tools_filter_by_category(self):
        """Test list_tools filters by category correctly."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"category": "filesystem"})

        assert result["filtered_by"] == "filesystem"
        assert result["total_count"] == 2
        for tool in result["tools"]:
            assert tool["category"] == "filesystem"

    def test_list_tools_category_all(self):
        """Test list_tools with category='all' returns all tools."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"category": "all"})

        assert result["total_count"] == 4
        assert "filtered_by" not in result

    def test_list_tools_includes_plugin_source(self):
        """Test list_tools includes plugin source for each tool."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({})

        for tool in result["tools"]:
            assert "plugin_source" in tool
            if tool["name"] == "readFile":
                assert tool["plugin_source"] == "file_edit"
            elif tool["name"] == "web_search":
                assert tool["plugin_source"] == "web_search"

    def test_list_tools_concise_descriptions(self):
        """Test list_tools truncates long descriptions by default."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"verbose": False})

        for tool in result["tools"]:
            assert "description" in tool
            # Descriptions should be truncated to first sentence or 100 chars
            assert len(tool["description"]) <= 150

    def test_list_tools_verbose_mode(self):
        """Test list_tools returns full descriptions in verbose mode."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({"verbose": True})

        # Find the search tool and check its description is complete
        search_tool = next(t for t in result["tools"] if t["name"] == "web_search")
        assert search_tool["description"] == "Search the web for information on any topic."

    def test_list_tools_includes_enabled_status(self):
        """Test list_tools includes enabled/disabled status."""
        # Disable a tool
        self.registry.disable_tool("readFile")

        executors = self.plugin.get_executors()
        result = executors["list_tools"]({})

        for tool in result["tools"]:
            assert "enabled" in tool
            if tool["name"] == "readFile":
                assert tool["enabled"] is False
            else:
                assert tool["enabled"] is True

    def test_list_tools_sorted_by_category_and_name(self):
        """Test list_tools returns tools sorted by category then name."""
        executors = self.plugin.get_executors()
        result = executors["list_tools"]({})

        tools = result["tools"]
        # Should be sorted: filesystem (readFile, writeFile), planning (createPlan), search (web_search)
        categories = [t.get("category", "zzz") for t in tools]
        assert categories == sorted(categories)

    def test_list_tools_no_registry_error(self):
        """Test list_tools returns error when registry not set."""
        plugin = create_plugin()
        plugin.initialize({})
        # Don't set registry

        executors = plugin.get_executors()
        result = executors["list_tools"]({})

        assert "error" in result


class TestGetToolSchema:
    """Tests for the get_tool_schema tool."""

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
        ])
        self.registry.register_plugin(test_plugin)
        self.plugin.set_plugin_registry(self.registry)

    def test_get_tool_schema_returns_full_details(self):
        """Test get_tool_schema returns complete tool information."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schema"]({"tool_name": "testTool"})

        assert result["name"] == "testTool"
        assert result["description"] == "A test tool for unit testing."
        assert result["plugin_source"] == "test_plugin"
        assert result["category"] == "code"
        assert result["enabled"] is True
        assert "parameters" in result

    def test_get_tool_schema_formats_parameters(self):
        """Test get_tool_schema formats parameters in readable structure."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schema"]({"tool_name": "testTool"})

        params = result["parameters"]

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

    def test_get_tool_schema_not_found(self):
        """Test get_tool_schema returns error for unknown tool."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schema"]({"tool_name": "nonExistentTool"})

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_get_tool_schema_suggests_similar_tools(self):
        """Test get_tool_schema suggests similar tools when not found."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schema"]({"tool_name": "test"})

        assert "error" in result
        # Should suggest testTool as it contains "test"
        assert "testTool" in result["error"]

    def test_get_tool_schema_requires_tool_name(self):
        """Test get_tool_schema returns error without tool_name."""
        executors = self.plugin.get_executors()
        result = executors["get_tool_schema"]({})

        assert "error" in result
        assert "required" in result["error"].lower()

    def test_get_tool_schema_no_registry_error(self):
        """Test get_tool_schema returns error when registry not set."""
        plugin = create_plugin()
        plugin.initialize({})
        # Don't set registry

        executors = plugin.get_executors()
        result = executors["get_tool_schema"]({"tool_name": "testTool"})

        assert "error" in result


class TestToolCategories:
    """Tests for tool category constants."""

    def test_standard_categories_defined(self):
        """Test that standard categories are defined."""
        assert "filesystem" in TOOL_CATEGORIES
        assert "code" in TOOL_CATEGORIES
        assert "search" in TOOL_CATEGORIES
        assert "memory" in TOOL_CATEGORIES
        assert "planning" in TOOL_CATEGORIES
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
