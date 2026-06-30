"""Unit tests for the Tool Discovery & Introspection Plugin."""

import sys
import pytest
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

from jaato_sdk.plugins.model_provider.types import ToolSchema, TOOL_CATEGORIES
from shared.plugins.introspection.plugin import IntrospectionPlugin, create_plugin
from shared.tool_id_map import name_to_id


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

    def get_category_descriptions(self) -> Dict[str, str]:
        return {}

    def register_category(self, name: str, description: str) -> None:
        pass


class TestIntrospectionPlugin:

    def test_create_plugin(self):
        plugin = create_plugin()
        assert plugin is not None
        assert isinstance(plugin, IntrospectionPlugin)
        assert plugin.name == "introspection"

    def test_initialize_and_shutdown(self):
        plugin = create_plugin()
        plugin.initialize({})
        assert plugin._initialized is True
        plugin.shutdown()
        assert plugin._initialized is False

    def test_get_tool_schemas(self):
        plugin = create_plugin()
        schemas = plugin.get_tool_schemas()
        assert len(schemas) == 2
        schema_names = [s.name for s in schemas]
        assert "list_tools" in schema_names
        assert "get_tool_schemas" in schema_names
        for schema in schemas:
            assert schema.category == "system"
            assert schema.discoverability == "core"

    def test_get_executors(self):
        plugin = create_plugin()
        executors = plugin.get_executors()
        assert "list_tools" in executors
        assert "get_tool_schemas" in executors

    def test_auto_approved_tools(self):
        plugin = create_plugin()
        auto_approved = plugin.get_auto_approved_tools()
        assert "list_tools" in auto_approved
        assert "get_tool_schemas" in auto_approved


def _make_test_env():
    """Create plugin + registry with test tools."""
    plugin = create_plugin()
    plugin.initialize({})
    registry = MockRegistry()

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
    registry.register_plugin(fs_plugin)

    search_plugin = MockPlugin("web_search", [
        ToolSchema(
            name="web_search",
            description="Search the web for information on any topic.",
            parameters={"type": "object", "properties": {}},
            category="search",
        ),
    ])
    registry.register_plugin(search_plugin)

    planning_plugin = MockPlugin("todo", [
        ToolSchema(
            name="createPlan",
            description="Create a new execution plan with steps.",
            parameters={"type": "object", "properties": {}},
            category="coordination",
        ),
    ])
    registry.register_plugin(planning_plugin)

    plugin.set_plugin_registry(registry)
    return plugin, registry


class TestListTools:

    def setup_method(self):
        self.plugin, self.registry = _make_test_env()

    def test_list_tools_no_category_returns_categories(self):
        result = self.plugin.get_executors()["list_tools"]({})
        assert "categories" in result
        assert "total_tools" in result
        assert result["total_tools"] == 4

        cat_names = [c["name"] for c in result["categories"]]
        assert "filesystem" in cat_names
        assert "search" in cat_names
        assert "coordination" in cat_names

        fs_cat = next(c for c in result["categories"] if c["name"] == "filesystem")
        assert fs_cat["tool_count"] == 2
        assert fs_cat["id"] == name_to_id("filesystem", prefix="c")

    def test_list_tools_with_category_returns_tools(self):
        cat_id = name_to_id("filesystem", prefix="c")
        result = self.plugin.get_executors()["list_tools"]({"category_id": cat_id})

        assert result["category"] == "filesystem"
        assert result["tool_count"] == 2
        assert "tools" in result

        tool_names = [t["name"] for t in result["tools"]]
        assert "readFile" in tool_names
        assert "writeFile" in tool_names

    def test_list_tools_includes_plugin_source(self):
        cat_id = name_to_id("filesystem", prefix="c")
        result = self.plugin.get_executors()["list_tools"]({"category_id": cat_id})
        # Plugin source is no longer exposed to model — tools identified by ID only
        for tool in result["tools"]:
            assert "id" in tool

    def test_list_tools_concise_descriptions(self):
        cat_id = name_to_id("filesystem", prefix="c")
        result = self.plugin.get_executors()["list_tools"]({"category_id": cat_id, "verbose": False})
        for tool in result["tools"]:
            assert "description" in tool
            assert len(tool["description"]) <= 150

    def test_list_tools_verbose_mode(self):
        cat_id = name_to_id("search", prefix="c")
        result = self.plugin.get_executors()["list_tools"]({"category_id": cat_id, "verbose": True})
        search_tool = result["tools"][0]
        assert search_tool["description"] == "Search the web for information on any topic."

    def test_list_tools_includes_enabled_status(self):
        self.registry.disable_tool("readFile")
        cat_id = name_to_id("filesystem", prefix="c")
        result = self.plugin.get_executors()["list_tools"]({"category_id": cat_id})

        for tool in result["tools"]:
            assert "enabled" in tool

        read_tool = next(t for t in result["tools"] if t["name"] == "readFile")
        assert read_tool["enabled"] is False

    def test_list_tools_sorted_by_id(self):
        cat_id = name_to_id("filesystem", prefix="c")
        result = self.plugin.get_executors()["list_tools"]({"category_id": cat_id})
        ids = [t["id"] for t in result["tools"]]
        assert ids == sorted(ids)

    def test_list_tools_no_registry_error(self):
        plugin = create_plugin()
        plugin.initialize({})
        result = plugin.get_executors()["list_tools"]({})
        assert "error" in result


class TestGetToolSchemas:

    def setup_method(self):
        self.plugin = create_plugin()
        self.plugin.initialize({})
        self.registry = MockRegistry()

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
        tool_id = name_to_id("testTool")
        result = self.plugin.get_executors()["get_tool_schemas"]({"tool_ids": [tool_id]})

        assert "schemas" in result
        assert result["count"] == 1
        schema = result["schemas"][0]
        assert schema["id"] == tool_id
        assert schema["name"] == "testTool"
        assert schema["description"] == "A test tool for unit testing."
        assert schema["category_id"] == name_to_id("code", prefix="c")
        assert schema["category"] == "code"
        assert schema["enabled"] is True
        assert "parameters" in schema

    def test_get_tool_schemas_multiple_tools(self):
        ids = [name_to_id("testTool"), name_to_id("anotherTool")]
        result = self.plugin.get_executors()["get_tool_schemas"]({"tool_ids": ids})

        assert result["count"] == 2
        schema_names = [s["name"] for s in result["schemas"]]
        assert "testTool" in schema_names
        assert "anotherTool" in schema_names

    def test_get_tool_schemas_formats_parameters(self):
        tool_id = name_to_id("testTool")
        result = self.plugin.get_executors()["get_tool_schemas"]({"tool_ids": [tool_id]})
        params = result["schemas"][0]["parameters"]

        assert "required_arg" in params
        assert params["required_arg"]["type"] == "string"
        assert params["required_arg"]["required"] is True
        assert "optional_arg" in params
        assert params["optional_arg"]["required"] is False
        assert params["optional_arg"]["default"] == 10
        assert "enum_arg" in params
        assert params["enum_arg"]["allowed_values"] == ["option1", "option2", "option3"]
        assert "array_arg" in params
        assert params["array_arg"]["items_type"] == "string"

    def test_get_tool_schemas_partial_not_found(self):
        tool_id = name_to_id("testTool")
        result = self.plugin.get_executors()["get_tool_schemas"]({"tool_ids": [tool_id, "t_nonexist"]})
        assert result["count"] == 1
        assert "not_found" in result
        assert "t_nonexist" in result["not_found"]

    def test_get_tool_schemas_requires_tool_ids(self):
        result = self.plugin.get_executors()["get_tool_schemas"]({})
        assert "error" in result
        assert "required" in result["error"].lower()

    def test_get_tool_schemas_no_registry_error(self):
        plugin = create_plugin()
        plugin.initialize({})
        result = plugin.get_executors()["get_tool_schemas"]({"tool_ids": ["t_abc"]})
        assert "error" in result

    def test_get_tool_schemas_tracks_accessed_tools(self):
        assert len(self.plugin.get_accessed_tools()) == 0
        tool_id = name_to_id("testTool")
        self.plugin.get_executors()["get_tool_schemas"]({"tool_ids": [tool_id]})
        assert "testTool" in self.plugin.get_accessed_tools()

        tool_id2 = name_to_id("anotherTool")
        self.plugin.get_executors()["get_tool_schemas"]({"tool_ids": [tool_id2]})
        accessed = self.plugin.get_accessed_tools()
        assert "testTool" in accessed
        assert "anotherTool" in accessed

    def test_clear_accessed_tools(self):
        tool_id = name_to_id("testTool")
        self.plugin.get_executors()["get_tool_schemas"]({"tool_ids": [tool_id]})
        assert len(self.plugin.get_accessed_tools()) == 1
        self.plugin.clear_accessed_tools()
        assert len(self.plugin.get_accessed_tools()) == 0


class TestToolCategories:

    def test_standard_categories_defined(self):
        assert "filesystem" in TOOL_CATEGORIES
        assert "code" in TOOL_CATEGORIES
        assert "search" in TOOL_CATEGORIES
        assert "memory" in TOOL_CATEGORIES
        assert "coordination" in TOOL_CATEGORIES
        assert "system" in TOOL_CATEGORIES
        assert "web" in TOOL_CATEGORIES
        assert "communication" in TOOL_CATEGORIES
        assert "prompt" in TOOL_CATEGORIES
        assert "MCP" in TOOL_CATEGORIES

    def test_tool_schema_accepts_category(self):
        schema = ToolSchema(name="test", description="Test", parameters={}, category="filesystem")
        assert schema.category == "filesystem"

    def test_tool_schema_category_optional(self):
        schema = ToolSchema(name="test", description="Test", parameters={})
        assert schema.category is None


class TestTelemetry:

    def setup_method(self):
        self.plugin, self.registry = _make_test_env()

    def test_list_tools_result_includes_telemetry_dict(self):
        result = self.plugin.get_executors()["list_tools"]({})
        assert "_telemetry" in result
        telem = result["_telemetry"]
        assert telem["jaato.introspection.operation"] == "list_tools"
        assert "jaato.introspection.total_tools" in telem
        assert telem["jaato.introspection.total_tools"] >= 1


class TestToolDiscoverability:

    def test_tool_schema_discoverability_default(self):
        schema = ToolSchema(name="test", description="Test", parameters={})
        assert schema.discoverability == "discoverable"

    def test_tool_schema_discoverability_core(self):
        schema = ToolSchema(name="test", description="Test", parameters={}, discoverability="core")
        assert schema.discoverability == "core"

    def test_introspection_tools_are_core(self):
        plugin = create_plugin()
        for schema in plugin.get_tool_schemas():
            assert schema.discoverability == "core"


class TestExploreToolsGuidanceWording:
    """2026-05-15 regression pin: the explore-tools guidance is phrased
    CONDITIONALLY ("WHEN REQUIRED" / "when your current information is
    insufficient"), not as an unconditional imperative ("ALWAYS explore").

    **Why the wording matters.**  Personas that scope the agent to a
    narrow tool set (e.g. discovery profile saying "only call
    signal_completion; do not call list_tools / get_tool_schemas")
    must coexist with the framework's general guidance.  An
    unconditional ``"ALWAYS explore tools first"`` directly
    contradicts the persona — same model, same prompt, two
    competing imperatives, and the framework's strong word often
    won (peer's v86 cascade aborted at discovery for exactly this
    reason).

    A conditional phrasing makes the framework instruction a
    FALLBACK for unclear cases ("explore WHEN you need to") while
    leaving room for the persona to say "exploration isn't
    required, the tool is X".  Both coexist without contradiction.

    See ``feedback_kernel_scoping_beats_persona_prose`` for the
    governing rule: framework instructions must not contradict
    session-level contracts.
    """

    def test_guidance_phrased_conditionally_not_unconditionally(self):
        """The exact word ``ALWAYS`` (as an unqualified imperative
        directing the model to explore) must not appear in the
        guidance.  Pin the absence of that phrase so a future
        restorer can't accidentally re-introduce it."""
        plugin = create_plugin()
        si = plugin.get_system_instructions()
        assert si is not None
        # Lowercase compare to catch any casing variation.
        si_lower = si.lower()
        # The specific imperative shape that caused v86: "ALWAYS
        # explore available tools first".  Forbid this exact pattern.
        assert "always explore" not in si_lower, (
            "Introspection guidance must not contain the unconditional "
            "imperative 'ALWAYS explore'.  Personas that need to scope "
            "the agent to a narrow tool set (e.g. typed-completion "
            "stages) need the framework instruction to be conditional "
            "so it doesn't contradict their explicit tool-list. "
            f"Got: {si!r}"
        )

    def test_guidance_contains_conditional_qualifier(self):
        """The guidance must carry an explicit qualifier indicating
        WHEN exploration applies (the conditional that lets a
        narrow-tool-list persona override default exploration)."""
        plugin = create_plugin()
        si = plugin.get_system_instructions()
        assert si is not None
        # One of the qualifier phrases must be present so the
        # instruction reads as conditional rather than absolute.
        qualifiers = [
            "when required",
            "when your current information is insufficient",
            "skip discovery when the persona",
        ]
        si_lower = si.lower()
        assert any(q in si_lower for q in qualifiers), (
            "Introspection guidance must contain a conditional "
            "qualifier (one of: 'when required', 'when your "
            "current information is insufficient', 'skip discovery "
            "when the persona') so personas can override the "
            "default exploration behaviour.  Got: "
            f"{si!r}"
        )

    def test_guidance_still_describes_workflow_for_unclear_cases(self):
        """When exploration IS warranted, the guidance must still tell
        the agent HOW (the list_tools → get_tool_schemas → call
        chain).  We're softening the trigger condition, not removing
        the workflow."""
        plugin = create_plugin()
        si = plugin.get_system_instructions()
        assert si is not None
        # The workflow steps should remain.
        assert "list_tools" in si
        assert "get_tool_schemas" in si
        # The category quick-reference should remain.
        assert "CATEGORY QUICK REFERENCE" in si

    def test_guidance_returned_for_typed_completion_sessions(self):
        """Sessions with typed completion contracts STILL receive the
        guidance — we don't gate by session state.  The wording is
        what changed.  The persona handles the "don't explore"
        decision; the framework just provides the general
        capability description.

        Pin: a typed-completion session (the v86 repro case) still
        gets the guidance, just no longer in an unconditional form
        that contradicts its persona."""
        plugin = create_plugin()

        class TypedSession:
            _completion_payload_schema = {"type": "object", "properties": {}}

        plugin.set_session(TypedSession())
        si = plugin.get_system_instructions()
        assert si is not None, (
            "Guidance must be returned even for typed-completion "
            "sessions — the fix is in the wording, not in suppression. "
            "Suppression-based gating was the wrong approach: it "
            "made framework guidance conditional on session state "
            "rather than letting personas authoritatively scope "
            "behaviour."
        )
        # And the conditional qualifier must still be present.
        assert "always explore" not in si.lower()


def test_discovery_guidance_suppressed_when_session_flags_it():
    """The introspection discovery guidance (list_tools/get_tool_schemas
    workflow) must be suppressed when the session dropped introspection's TOOLS
    (nothing to discover) — else the prompt nudges the model to call tools that
    aren't on its wire, it invents them, hits no-executor, and loops (the ex08
    0-tool subagent hang). Instruction-gate aligned with tool-gate."""
    from shared.plugins.introspection.plugin import create_plugin

    p = create_plugin()
    # No session / flag unset -> guidance present.
    instr = p.get_system_instructions()
    assert instr is not None and "CAPABILITY DISCOVERY" in instr

    class _FlaggedSession:
        _introspection_guidance_suppressed = True

    p.set_session(_FlaggedSession())
    assert p.get_system_instructions() is None  # suppressed

    class _NormalSession:
        _introspection_guidance_suppressed = False

    p.set_session(_NormalSession())
    assert "CAPABILITY DISCOVERY" in (p.get_system_instructions() or "")
