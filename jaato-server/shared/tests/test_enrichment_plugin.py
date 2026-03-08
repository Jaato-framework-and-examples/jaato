"""Tests for EnrichmentPlugin protocol and registry integration."""

import pytest

from jaato_sdk.plugins.base import (
    EnrichmentPlugin,
    ToolPlugin,
    PromptEnrichmentResult,
    SystemInstructionEnrichmentResult,
    ToolResultEnrichmentResult,
)
from shared.plugins.registry import PluginRegistry


# ---------- Test fixtures: minimal enrichment plugin implementations ----------


class PromptEnrichmentOnly:
    """Minimal enrichment plugin that only enriches prompts."""

    @property
    def name(self) -> str:
        return "prompt_enricher"

    def initialize(self, config=None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def subscribes_to_prompt_enrichment(self) -> bool:
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        return PromptEnrichmentResult(
            prompt=prompt + " [enriched]",
            metadata={"enriched": True},
        )


class SystemInstructionEnrichmentOnly:
    """Minimal enrichment plugin that only enriches system instructions."""

    @property
    def name(self) -> str:
        return "sysinstruction_enricher"

    def initialize(self, config=None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def subscribes_to_system_instruction_enrichment(self) -> bool:
        return True

    def enrich_system_instructions(
        self, instructions: str
    ) -> SystemInstructionEnrichmentResult:
        return SystemInstructionEnrichmentResult(
            instructions=instructions + " [sys-enriched]",
            metadata={"enriched": True},
        )


class ToolResultEnrichmentOnly:
    """Minimal enrichment plugin that only enriches tool results."""

    @property
    def name(self) -> str:
        return "result_enricher"

    def initialize(self, config=None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def subscribes_to_tool_result_enrichment(self) -> bool:
        return True

    def enrich_tool_result(
        self, tool_name: str, result: str, **kwargs
    ) -> ToolResultEnrichmentResult:
        return ToolResultEnrichmentResult(
            result=result + " [result-enriched]",
            metadata={"tool_name": tool_name},
        )


class MultiEnrichmentPlugin:
    """Enrichment plugin subscribing to all three enrichment types."""

    @property
    def name(self) -> str:
        return "multi_enricher"

    def initialize(self, config=None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def subscribes_to_prompt_enrichment(self) -> bool:
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        return PromptEnrichmentResult(prompt=prompt + " [multi-prompt]")

    def subscribes_to_system_instruction_enrichment(self) -> bool:
        return True

    def enrich_system_instructions(
        self, instructions: str
    ) -> SystemInstructionEnrichmentResult:
        return SystemInstructionEnrichmentResult(
            instructions=instructions + " [multi-sys]"
        )

    def subscribes_to_tool_result_enrichment(self) -> bool:
        return True

    def enrich_tool_result(
        self, tool_name: str, result: str, **kwargs
    ) -> ToolResultEnrichmentResult:
        return ToolResultEnrichmentResult(result=result + " [multi-result]")

    def get_enrichment_priority(self) -> int:
        return 30


class PriorityEnrichmentPlugin:
    """Enrichment plugin with custom priority for ordering tests."""

    def __init__(self, plugin_name: str, priority: int, suffix: str):
        self._name = plugin_name
        self._priority = priority
        self._suffix = suffix

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, config=None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def subscribes_to_prompt_enrichment(self) -> bool:
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        return PromptEnrichmentResult(prompt=prompt + self._suffix)

    def get_enrichment_priority(self) -> int:
        return self._priority


# ---------- Protocol conformance tests ----------


class TestEnrichmentPluginProtocol:
    """Test that EnrichmentPlugin protocol works with isinstance checks."""

    def test_prompt_enrichment_only_is_enrichment_plugin(self):
        plugin = PromptEnrichmentOnly()
        assert isinstance(plugin, EnrichmentPlugin)

    def test_system_instruction_enrichment_only_is_enrichment_plugin(self):
        plugin = SystemInstructionEnrichmentOnly()
        assert isinstance(plugin, EnrichmentPlugin)

    def test_tool_result_enrichment_only_is_enrichment_plugin(self):
        plugin = ToolResultEnrichmentOnly()
        assert isinstance(plugin, EnrichmentPlugin)

    def test_multi_enrichment_is_enrichment_plugin(self):
        plugin = MultiEnrichmentPlugin()
        assert isinstance(plugin, EnrichmentPlugin)

    def test_enrichment_plugin_is_not_tool_plugin(self):
        """EnrichmentPlugin should NOT satisfy ToolPlugin protocol."""
        plugin = PromptEnrichmentOnly()
        assert not isinstance(plugin, ToolPlugin)


# ---------- Registry integration tests ----------


class TestRegistryEnrichmentPlugin:
    """Test EnrichmentPlugin integration with PluginRegistry."""

    def test_register_enrichment_plugin_auto_enrichment_only(self):
        """EnrichmentPlugin is automatically registered as enrichment-only."""
        registry = PluginRegistry()
        plugin = PromptEnrichmentOnly()

        registry.register_plugin(plugin)

        assert plugin.name in registry._enrichment_only
        assert plugin.name not in registry._exposed

    def test_register_enrichment_plugin_explicit_enrichment_only(self):
        """Explicit enrichment_only=True also works."""
        registry = PluginRegistry()
        plugin = PromptEnrichmentOnly()

        registry.register_plugin(plugin, enrichment_only=True)

        assert plugin.name in registry._enrichment_only

    def test_register_enrichment_plugin_expose_ignored(self):
        """expose=True is ignored for EnrichmentPlugin instances."""
        registry = PluginRegistry()
        plugin = PromptEnrichmentOnly()

        # expose=True should be ignored since it's an EnrichmentPlugin
        registry.register_plugin(plugin, expose=True)

        assert plugin.name in registry._enrichment_only
        assert plugin.name not in registry._exposed

    def test_enrichment_plugin_in_prompt_subscribers(self):
        """EnrichmentPlugin shows up in prompt enrichment subscribers."""
        registry = PluginRegistry()
        plugin = PromptEnrichmentOnly()

        registry.register_plugin(plugin)

        subscribers = registry.get_prompt_enrichment_subscribers()
        assert len(subscribers) == 1
        assert subscribers[0].name == "prompt_enricher"

    def test_enrichment_plugin_prompt_enrichment(self):
        """EnrichmentPlugin actually enriches prompts through the registry."""
        registry = PluginRegistry()
        plugin = PromptEnrichmentOnly()

        registry.register_plugin(plugin)

        result = registry.enrich_prompt("Hello")
        assert result.prompt == "Hello [enriched]"
        assert result.metadata.get("prompt_enricher", {}).get("enriched") is True

    def test_enrichment_plugin_system_instruction_enrichment(self):
        """EnrichmentPlugin enriches system instructions through the registry."""
        registry = PluginRegistry()
        plugin = SystemInstructionEnrichmentOnly()

        registry.register_plugin(plugin)

        result = registry.enrich_system_instructions("Be helpful")
        assert result.instructions == "Be helpful [sys-enriched]"

    def test_enrichment_plugin_tool_result_enrichment(self):
        """EnrichmentPlugin enriches tool results through the registry."""
        registry = PluginRegistry()
        plugin = ToolResultEnrichmentOnly()

        registry.register_plugin(plugin)

        result = registry.enrich_tool_result("some_tool", "output")
        assert result.result == "output [result-enriched]"

    def test_enrichment_plugin_not_in_exposed_schemas(self):
        """EnrichmentPlugin tools are NOT in exposed tool schemas."""
        registry = PluginRegistry()
        plugin = PromptEnrichmentOnly()

        registry.register_plugin(plugin)

        schemas = registry.get_exposed_tool_schemas()
        assert len(schemas) == 0

    def test_enrichment_plugin_not_in_exposed_executors(self):
        """EnrichmentPlugin executors are NOT in exposed executors."""
        registry = PluginRegistry()
        plugin = PromptEnrichmentOnly()

        registry.register_plugin(plugin)

        executors = registry.get_exposed_executors()
        assert len(executors) == 0

    def test_multi_enrichment_plugin_all_types(self):
        """Multi-enrichment plugin participates in all enrichment types."""
        registry = PluginRegistry()
        plugin = MultiEnrichmentPlugin()

        registry.register_plugin(plugin)

        prompt_result = registry.enrich_prompt("Hello")
        assert prompt_result.prompt == "Hello [multi-prompt]"

        sys_result = registry.enrich_system_instructions("Be helpful")
        assert sys_result.instructions == "Be helpful [multi-sys]"

        tool_result = registry.enrich_tool_result("tool", "output")
        assert tool_result.result == "output [multi-result]"

    def test_enrichment_plugin_priority_ordering(self):
        """Enrichment plugins are called in priority order."""
        registry = PluginRegistry()

        # Register in reverse priority order to test sorting
        high_priority = PriorityEnrichmentPlugin("high", 10, " [high]")
        low_priority = PriorityEnrichmentPlugin("low", 90, " [low]")

        registry.register_plugin(low_priority)
        registry.register_plugin(high_priority)

        result = registry.enrich_prompt("Hello")
        # high priority (10) runs first, then low priority (90)
        assert result.prompt == "Hello [high] [low]"

    def test_enrichment_plugin_get_plugin(self):
        """EnrichmentPlugin can be retrieved via get_plugin()."""
        registry = PluginRegistry()
        plugin = PromptEnrichmentOnly()

        registry.register_plugin(plugin)

        retrieved = registry.get_plugin("prompt_enricher")
        assert retrieved is plugin

    def test_mixed_tool_and_enrichment_plugins(self):
        """Tool plugins and enrichment plugins can coexist in enrichment pipeline."""
        registry = PluginRegistry()

        # Register an enrichment-only plugin
        enrichment_plugin = PriorityEnrichmentPlugin("enrichment", 10, " [enrichment]")
        registry.register_plugin(enrichment_plugin)

        # Also register a ToolPlugin that subscribes to enrichment
        # (simulated by registering enrichment_only=True with a ToolPlugin-shaped object)
        tool_enricher = PriorityEnrichmentPlugin("tool_enricher", 90, " [tool]")
        registry._plugins[tool_enricher.name] = tool_enricher
        registry._enrichment_only.add(tool_enricher.name)

        result = registry.enrich_prompt("Hello")
        assert result.prompt == "Hello [enrichment] [tool]"

    def test_expose_all_skips_enrichment_plugins(self):
        """expose_all() must not add enrichment-only plugins to _exposed.

        Enrichment plugins don't implement get_tool_schemas()/get_executors(),
        so if they end up in _exposed, get_exposed_tool_schemas() and
        get_plugin_for_tool() will raise AttributeError.
        """
        registry = PluginRegistry()

        # Register an enrichment-only plugin
        enrichment_plugin = PromptEnrichmentOnly()
        registry.register_plugin(enrichment_plugin)

        # expose_all iterates _plugins and calls expose_tool on each
        registry.expose_all()

        # Enrichment plugin must NOT be in _exposed
        assert enrichment_plugin.name not in registry._exposed
        assert enrichment_plugin.name in registry._enrichment_only

        # These must not raise AttributeError
        schemas = registry.get_exposed_tool_schemas()
        executors = registry.get_exposed_executors()
        assert isinstance(schemas, list)
        assert isinstance(executors, dict)
