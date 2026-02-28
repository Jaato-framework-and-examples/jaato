"""Tests for deferred plugin system instructions.

When deferred tool loading is enabled (JAATO_DEFERRED_TOOLS=true, the default),
plugins that have no core tools should NOT have their system instructions
included in the initial model context or the instruction budget.  Those
instructions should only be injected when the model discovers and activates
one of the plugin's tools via activate_discovered_tools().
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from shared.instruction_budget import (
    InstructionBudget,
    InstructionSource,
    PluginToolType,
    DEFAULT_TOOL_POLICIES,
)
from shared.instruction_token_cache import InstructionTokenCache
from shared.jaato_session import JaatoSession
from jaato_sdk.plugins.model_provider.types import ToolSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plugin(name: str, tools: list, system_instructions: str = None):
    """Create a mock plugin with the given tools and system instructions.

    Args:
        name: Plugin name.
        tools: List of ToolSchema objects.
        system_instructions: Text returned by get_system_instructions().
    """
    plugin = MagicMock()
    type(plugin).name = PropertyMock(return_value=name)
    plugin.get_tool_schemas.return_value = tools
    plugin.get_system_instructions.return_value = system_instructions
    plugin.get_executors.return_value = {t.name: (lambda args: {"ok": True}) for t in tools}
    plugin.get_auto_approved_tools.return_value = []
    return plugin


def _make_session(
    exposed_plugins: dict = None,
    deferred_enabled: bool = True,
    base_instructions: str = None,
):
    """Create a minimal JaatoSession wired to mock runtime/provider.

    Args:
        exposed_plugins: Dict of {plugin_name: mock_plugin}.
        deferred_enabled: Whether deferred tool loading is enabled.
        base_instructions: Optional base system instructions.
    """
    cache = InstructionTokenCache()

    # Mock runtime
    runtime = MagicMock()
    runtime.provider_name = "test_provider"
    runtime.instruction_token_cache = cache
    runtime._base_system_instructions = base_instructions
    runtime._formatter_pipeline = None
    runtime.telemetry = MagicMock()
    runtime.telemetry.enabled = False

    # Mock registry
    if exposed_plugins:
        registry = MagicMock()
        registry._exposed = list(exposed_plugins.keys())
        registry._plugins = dict(exposed_plugins)

        def get_plugin(name):
            return exposed_plugins.get(name)

        registry.get_plugin = get_plugin

        # Implement plugin_has_core_tools using real logic
        def plugin_has_core_tools(pname):
            p = exposed_plugins.get(pname)
            if not p or not hasattr(p, 'get_tool_schemas'):
                return False
            for schema in p.get_tool_schemas():
                if getattr(schema, 'discoverability', 'discoverable') == 'core':
                    return True
            return False

        registry.plugin_has_core_tools = plugin_has_core_tools

        # get_exposed_tool_schemas returns all tool schemas from exposed plugins
        def get_exposed_tool_schemas():
            schemas = []
            for pname in registry._exposed:
                p = exposed_plugins.get(pname)
                if p and hasattr(p, 'get_tool_schemas'):
                    schemas.extend(p.get_tool_schemas())
            return schemas

        registry.get_exposed_tool_schemas = get_exposed_tool_schemas

        # get_plugin_for_tool looks up which plugin owns a tool
        def get_plugin_for_tool(tool_name):
            for pname, p in exposed_plugins.items():
                if hasattr(p, 'get_tool_schemas'):
                    for s in p.get_tool_schemas():
                        if s.name == tool_name:
                            return p
            return None

        registry.get_plugin_for_tool = get_plugin_for_tool

        runtime.registry = registry
    else:
        runtime.registry = None

    # Mock provider
    provider = MagicMock()
    del provider.count_tokens  # No background counting
    provider.get_context_limit.return_value = 128_000

    # Build session without calling __init__ (like the existing tests)
    session = JaatoSession.__new__(JaatoSession)
    session._runtime = runtime
    session._model_name = "test-model"
    session._provider_name_override = None
    session._provider = provider
    session._agent_id = "test"
    session._agent_type = "main"
    session._agent_name = None
    session._instruction_budget = None
    session._budget_counting_thread = None
    session._on_instruction_budget_updated = None
    session._ui_hooks = None
    session._gc_plugin = None
    session._gc_config = None
    session._system_instruction = None
    session._tools = []
    session._deferred_plugin_instructions = set()
    session._preloaded_plugins = set()

    # Phase 1: SessionHistory wrapper (canonical history owned by session)
    from ..session_history import SessionHistory
    session._history = SessionHistory()

    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeferredPluginInstructions:
    """Verify that discoverable-only plugin instructions are deferred."""

    def _make_core_plugin(self):
        """Plugin with one core tool and system instructions."""
        return _make_plugin(
            "introspection",
            tools=[
                ToolSchema(
                    name="list_tools",
                    description="List available tools",
                    parameters={},
                    discoverability="core",
                ),
            ],
            system_instructions="Introspection: use list_tools to discover tools.",
        )

    def _make_discoverable_plugin(self):
        """Plugin with only discoverable tools and system instructions."""
        return _make_plugin(
            "web_search",
            tools=[
                ToolSchema(
                    name="web_search",
                    description="Search the web",
                    parameters={},
                    discoverability="discoverable",
                ),
            ],
            system_instructions="Web search: use web_search to find information.",
        )

    @patch("shared.jaato_runtime._is_deferred_tools_enabled", return_value=True)
    def test_discoverable_only_plugin_excluded_from_initial_budget(self, _mock):
        """Discoverable-only plugin instructions are NOT in the initial budget."""
        core = self._make_core_plugin()
        disc = self._make_discoverable_plugin()

        session = _make_session(
            exposed_plugins={"introspection": core, "web_search": disc},
            deferred_enabled=True,
        )

        session._populate_instruction_budget()

        budget = session._instruction_budget
        plugin_entry = budget.get_entry(InstructionSource.PLUGIN)
        assert plugin_entry is not None

        # Core plugin instructions should be present
        assert "introspection" in plugin_entry.children

        # Discoverable-only plugin instructions should NOT be present
        assert "web_search" not in plugin_entry.children

    @patch("shared.jaato_runtime._is_deferred_tools_enabled", return_value=True)
    def test_deferred_set_tracks_skipped_plugins(self, _mock):
        """The session should remember which plugins had instructions deferred."""
        core = self._make_core_plugin()
        disc = self._make_discoverable_plugin()

        session = _make_session(
            exposed_plugins={"introspection": core, "web_search": disc},
        )

        session._populate_instruction_budget()

        assert "web_search" in session._deferred_plugin_instructions
        assert "introspection" not in session._deferred_plugin_instructions

    @patch("shared.jaato_runtime._is_deferred_tools_enabled", return_value=False)
    def test_no_deferral_when_deferred_tools_disabled(self, _mock):
        """When deferred tools are disabled, all plugin instructions are included."""
        core = self._make_core_plugin()
        disc = self._make_discoverable_plugin()

        session = _make_session(
            exposed_plugins={"introspection": core, "web_search": disc},
            deferred_enabled=False,
        )

        session._populate_instruction_budget()

        budget = session._instruction_budget
        plugin_entry = budget.get_entry(InstructionSource.PLUGIN)

        # Both should be present
        assert "introspection" in plugin_entry.children
        assert "web_search" in plugin_entry.children

        # Nothing should be deferred
        assert len(session._deferred_plugin_instructions) == 0

    @patch("shared.jaato_runtime._is_deferred_tools_enabled", return_value=True)
    def test_activate_injects_deferred_instructions(self, _mock):
        """Activating a tool injects its plugin's deferred system instructions."""
        core = self._make_core_plugin()
        disc = self._make_discoverable_plugin()

        session = _make_session(
            exposed_plugins={"introspection": core, "web_search": disc},
        )

        # Populate budget (defers web_search instructions)
        session._populate_instruction_budget()
        assert "web_search" in session._deferred_plugin_instructions

        # Set initial system instruction to simulate configure()
        session._system_instruction = "Initial instructions."

        # Activate the discoverable tool
        activated = session.activate_discovered_tools(["web_search"])

        assert "web_search" in activated

        # System instruction should now include the deferred plugin instructions
        assert "Web search: use web_search to find information." in session._system_instruction

        # Plugin should be removed from the deferred set
        assert "web_search" not in session._deferred_plugin_instructions

        # Budget should now include the plugin's instructions
        budget = session._instruction_budget
        plugin_entry = budget.get_entry(InstructionSource.PLUGIN)
        assert "web_search" in plugin_entry.children
        assert plugin_entry.children["web_search"].tokens > 0

    @patch("shared.jaato_runtime._is_deferred_tools_enabled", return_value=True)
    def test_activate_only_injects_once(self, _mock):
        """Activating a second tool from the same plugin doesn't re-inject."""
        plugin = _make_plugin(
            "multi_tool",
            tools=[
                ToolSchema(name="tool_a", description="Tool A", parameters={},
                           discoverability="discoverable"),
                ToolSchema(name="tool_b", description="Tool B", parameters={},
                           discoverability="discoverable"),
            ],
            system_instructions="Multi-tool plugin instructions.",
        )

        session = _make_session(
            exposed_plugins={"multi_tool": plugin},
        )

        session._populate_instruction_budget()
        session._system_instruction = "Base."

        # Activate first tool — injects instructions
        session.activate_discovered_tools(["tool_a"])
        instr_after_first = session._system_instruction

        # Activate second tool — should NOT re-inject (already done)
        session.activate_discovered_tools(["tool_b"])
        instr_after_second = session._system_instruction

        # Instructions should appear exactly once
        assert instr_after_first == instr_after_second
        assert instr_after_first.count("Multi-tool plugin instructions.") == 1

    @patch("shared.jaato_runtime._is_deferred_tools_enabled", return_value=True)
    def test_second_tool_accumulates_tokens_in_same_entry(self, _mock):
        """Activating a second tool from the same plugin accumulates tokens."""
        plugin = _make_plugin(
            "multi_tool",
            tools=[
                ToolSchema(name="tool_a", description="Tool A", parameters={},
                           discoverability="discoverable"),
                ToolSchema(name="tool_b", description="Tool B", parameters={},
                           discoverability="discoverable"),
            ],
            system_instructions="Multi-tool plugin instructions.",
        )

        session = _make_session(
            exposed_plugins={"multi_tool": plugin},
        )

        session._populate_instruction_budget()
        session._system_instruction = "Base."

        # Activate first tool
        session.activate_discovered_tools(["tool_a"])
        budget = session._instruction_budget
        plugin_entry = budget.get_entry(InstructionSource.PLUGIN)
        tokens_after_first = plugin_entry.children["multi_tool"].tokens

        # Activate second tool — tokens should increase, still one entry
        session.activate_discovered_tools(["tool_b"])
        tokens_after_second = plugin_entry.children["multi_tool"].tokens

        assert tokens_after_second > tokens_after_first
        # Still only one entry for the plugin, no per-tool entries
        plugin_children_keys = list(plugin_entry.children.keys())
        assert plugin_children_keys == ["multi_tool"]

    @patch("shared.jaato_runtime._is_deferred_tools_enabled", return_value=True)
    def test_plugin_with_no_instructions_not_tracked(self, _mock):
        """Plugins with no system instructions are not added to the deferred set."""
        plugin = _make_plugin(
            "quiet_plugin",
            tools=[
                ToolSchema(name="quiet_tool", description="Quiet", parameters={},
                           discoverability="discoverable"),
            ],
            system_instructions=None,  # No instructions
        )

        session = _make_session(
            exposed_plugins={"quiet_plugin": plugin},
        )

        session._populate_instruction_budget()

        # Should NOT be in deferred set since there's nothing to inject
        assert "quiet_plugin" not in session._deferred_plugin_instructions


class TestRegistryPluginHasCoreTools:
    """Verify PluginRegistry.plugin_has_core_tools works correctly."""

    def test_plugin_with_core_tool(self):
        """Plugin with a core tool returns True."""
        from shared.plugins.registry import PluginRegistry
        registry = PluginRegistry()

        plugin = _make_plugin(
            "core_plugin",
            tools=[
                ToolSchema(name="core_tool", description="Core", parameters={},
                           discoverability="core"),
            ],
        )
        registry._plugins["core_plugin"] = plugin

        assert registry.plugin_has_core_tools("core_plugin") is True

    def test_plugin_with_only_discoverable_tools(self):
        """Plugin with only discoverable tools returns False."""
        from shared.plugins.registry import PluginRegistry
        registry = PluginRegistry()

        plugin = _make_plugin(
            "disc_plugin",
            tools=[
                ToolSchema(name="disc_tool", description="Disc", parameters={},
                           discoverability="discoverable"),
            ],
        )
        registry._plugins["disc_plugin"] = plugin

        assert registry.plugin_has_core_tools("disc_plugin") is False

    def test_mixed_plugin(self):
        """Plugin with both core and discoverable tools returns True."""
        from shared.plugins.registry import PluginRegistry
        registry = PluginRegistry()

        plugin = _make_plugin(
            "mixed_plugin",
            tools=[
                ToolSchema(name="core_one", description="Core", parameters={},
                           discoverability="core"),
                ToolSchema(name="disc_one", description="Disc", parameters={},
                           discoverability="discoverable"),
            ],
        )
        registry._plugins["mixed_plugin"] = plugin

        assert registry.plugin_has_core_tools("mixed_plugin") is True

    def test_unknown_plugin(self):
        """Unknown plugin name returns False."""
        from shared.plugins.registry import PluginRegistry
        registry = PluginRegistry()

        assert registry.plugin_has_core_tools("nonexistent") is False
