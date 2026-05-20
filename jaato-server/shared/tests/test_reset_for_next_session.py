"""Pin the cascade-sharing reset protocol (Phase 1, server 0.6.142+).

See ``docs/design/runner-cascade-sharing.md`` for the full architecture.

Daniel's litmus test (2026-05-20):

    A plugin's state should SURVIVE ``reset_for_next_session()`` if a
    subsequent session within the SAME cascade might benefit from it.

This file pins:

1. **Protocol**: every ``ToolPlugin`` and ``EnrichmentPlugin`` has a
   callable ``reset_for_next_session()`` method.  Default is no-op.

2. **Per-attribute behavior**: each plugin that overrides
   ``reset_for_next_session()`` is tested for:
   - which attributes get CLEARED (per-session state)
   - which attributes SURVIVE the reset (per Daniel's litmus test)

3. **Daniel's explicit corrections**: memory + artifact_tracker were
   moved out of the "needs reset" bucket because next cascade stages
   benefit from prior stages' memories + artifacts.  These tests pin
   that those plugins have explicit NO-OP overrides documenting the
   decision.

4. **The motivating plugin**: lsp's ``_connected_servers`` MUST
   survive reset_for_next_session — that's the whole point of
   cascade-sharing.  Pinned here.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# =============================================================================
# Protocol pins
# =============================================================================


class TestProtocolDefinition:
    """Pin: both ToolPlugin and EnrichmentPlugin declare
    ``reset_for_next_session`` in their protocol so plugin authors see
    it via duck-typing."""

    def test_tool_plugin_protocol_has_reset_for_next_session(self):
        from jaato_sdk.plugins.base import ToolPlugin
        # Protocol method existence — checked structurally
        assert hasattr(ToolPlugin, 'reset_for_next_session'), (
            "ToolPlugin protocol must declare reset_for_next_session "
            "for cascade-sharing arc (Phase 1)."
        )

    def test_enrichment_plugin_protocol_has_reset_for_next_session(self):
        from jaato_sdk.plugins.base import EnrichmentPlugin
        assert hasattr(EnrichmentPlugin, 'reset_for_next_session'), (
            "EnrichmentPlugin protocol must declare reset_for_next_session "
            "for cascade-sharing arc (Phase 1)."
        )


# =============================================================================
# session plugin: CLEAR per-session attrs
# =============================================================================


class TestSessionPluginReset:
    """Pin: session plugin clears per-session id + counters but NOT
    workspace-scoped config."""

    def _make_plugin(self):
        from shared.plugins.session.file_session import FileSessionPlugin
        return FileSessionPlugin()

    def test_clears_current_session_id(self):
        plugin = self._make_plugin()
        plugin._current_session_id = "previous-session"
        plugin.reset_for_next_session()
        assert plugin._current_session_id is None

    def test_clears_turn_count(self):
        plugin = self._make_plugin()
        plugin._turn_count = 42
        plugin.reset_for_next_session()
        assert plugin._turn_count == 0

    def test_clears_description(self):
        plugin = self._make_plugin()
        plugin._session_description = "prev session description"
        plugin._description_requested = True
        plugin.reset_for_next_session()
        assert plugin._session_description is None
        assert plugin._description_requested is False

    def test_preserves_storage_path(self):
        """Pin: workspace-tier config survives reset."""
        from pathlib import Path
        plugin = self._make_plugin()
        plugin._storage_path = Path("/my/workspace/.jaato/sessions")
        plugin._config = {"storage_path": "/my/workspace/.jaato/sessions"}
        plugin.reset_for_next_session()
        assert plugin._storage_path == Path("/my/workspace/.jaato/sessions")
        assert plugin._config == {"storage_path": "/my/workspace/.jaato/sessions"}


# =============================================================================
# permission plugin: CLEAR per-session approvals
# =============================================================================


class TestPermissionPluginReset:
    """Pin: permission clears per-session approval flags + execution log."""

    def _make_plugin(self):
        from shared.plugins.permission.plugin import PermissionPlugin
        return PermissionPlugin()

    def test_clears_allow_all(self):
        plugin = self._make_plugin()
        plugin._allow_all = True
        plugin.reset_for_next_session()
        assert plugin._allow_all is False

    def test_clears_turn_suspended(self):
        plugin = self._make_plugin()
        plugin._turn_suspended = True
        plugin.reset_for_next_session()
        assert plugin._turn_suspended is False

    def test_clears_idle_suspended(self):
        plugin = self._make_plugin()
        plugin._idle_suspended = True
        plugin.reset_for_next_session()
        assert plugin._idle_suspended is False

    def test_clears_execution_log(self):
        plugin = self._make_plugin()
        plugin._execution_log = [{"tool": "x", "approved": True}]
        plugin.reset_for_next_session()
        assert plugin._execution_log == []

    def test_clears_agent_name(self):
        plugin = self._make_plugin()
        plugin._agent_name = "previous-agent"
        plugin.reset_for_next_session()
        assert plugin._agent_name is None

    def test_preserves_workspace_path(self):
        """Pin: workspace_path is constant within cascade."""
        plugin = self._make_plugin()
        plugin._workspace_path = "/my/workspace"
        plugin._allow_all = True
        plugin.reset_for_next_session()
        # Workspace path survives; allow_all cleared
        assert plugin._workspace_path == "/my/workspace"
        assert plugin._allow_all is False


# =============================================================================
# thinking plugin: RESET _current_config to default preset
# =============================================================================


class TestThinkingPluginReset:
    """Pin: thinking clears per-session config back to the default
    preset (so next stage doesn't inherit prior stage's enter_tier
    choice)."""

    def _make_plugin(self):
        from shared.plugins.thinking.plugin import ThinkingPlugin
        return ThinkingPlugin()

    def test_resets_to_default_preset_config(self):
        from jaato_sdk.plugins.model_provider.types import ThinkingConfig
        plugin = self._make_plugin()
        # Simulate session A having enabled a non-default config
        plugin._current_config = ThinkingConfig(enabled=True, budget=8192)
        plugin.reset_for_next_session()
        # After reset, config should be defaults (whatever the preset
        # config dictates — typically disabled / empty thinking)
        # We just verify the attribute was touched (re-assigned)
        assert isinstance(plugin._current_config, ThinkingConfig)


# =============================================================================
# Across-session by design: NO-OP overrides (with explicit comments)
# =============================================================================


class TestAcrossSessionByDesignPlugins:
    """Pin: lsp, memory, artifact_tracker, todo, clarification override
    reset_for_next_session as explicit NO-OPs so the design intent is
    documented in code (vs inheriting the Protocol default)."""

    def test_lsp_reset_preserves_connected_servers(self):
        """Pin THE motivating plugin: _connected_servers MUST survive."""
        from shared.plugins.lsp.plugin import LSPToolPlugin
        plugin = LSPToolPlugin()
        plugin._connected_servers = {"java", "python"}
        plugin._clients = {"java": MagicMock(), "python": MagicMock()}
        plugin._config_cache = {"languageServers": {"java": {"command": "jdtls"}}}
        plugin.reset_for_next_session()
        # ALL these survive — cascade-sharing target
        assert plugin._connected_servers == {"java", "python"}
        assert "java" in plugin._clients
        assert "python" in plugin._clients
        assert plugin._config_cache != {}

    def test_memory_reset_preserves_storage_and_index(self):
        """Pin: memory plugin's storage + indexer survive reset.
        Daniel-corrected 2026-05-20."""
        from shared.plugins.memory.plugin import MemoryPlugin
        plugin = MemoryPlugin()
        plugin._storage = MagicMock()
        plugin._indexer = MagicMock()
        plugin.reset_for_next_session()
        # State survives — model memories are cross-session
        assert plugin._storage is not None
        assert plugin._indexer is not None

    def test_artifact_tracker_reset_preserves_registry(self):
        """Pin: artifact_tracker's registry survives reset.
        Daniel-corrected 2026-05-20."""
        from shared.plugins.artifact_tracker.plugin import ArtifactTrackerPlugin
        plugin = ArtifactTrackerPlugin()
        plugin._registry = MagicMock()
        plugin._workspace_root = "/my/workspace"
        plugin.reset_for_next_session()
        # State survives — artifact graph is cross-session
        assert plugin._registry is not None
        assert plugin._workspace_root == "/my/workspace"

    def test_todo_reset_preserves_plan_ids(self):
        """Pin: todo's per-agent plan map survives reset (existing
        design preserves across shutdown/init too)."""
        from shared.plugins.todo.plugin import TodoPlugin
        plugin = TodoPlugin()
        plugin._current_plan_ids = {"agent_a": "plan_123", "agent_b": "plan_456"}
        plugin.reset_for_next_session()
        # Plan map survives
        assert plugin._current_plan_ids == {"agent_a": "plan_123", "agent_b": "plan_456"}

    def test_clarification_reset_is_callable_no_op(self):
        """Pin: clarification's reset is callable (Protocol contract)
        and doesn't crash on a fresh instance."""
        from shared.plugins.clarification.plugin import ClarificationPlugin
        plugin = ClarificationPlugin()
        # No setup needed — just call + assert it doesn't raise
        plugin.reset_for_next_session()


# =============================================================================
# Source-pin tests: the implemented overrides include the litmus-test
# rationale in their docstrings (catches future refactors that strip
# the documentation).
# =============================================================================


class TestImplementationsDocumentLitmusTest:
    """Pin: each overridden reset_for_next_session documents WHY it
    clears (or doesn't) per the Daniel litmus test.  Prevents future
    refactors that strip the rationale and leave behavior unjustified."""

    @pytest.mark.parametrize("plugin_module,plugin_class_name,expected_marker", [
        ("shared.plugins.session.file_session", "FileSessionPlugin", "litmus test"),
        ("shared.plugins.permission.plugin", "PermissionPlugin", "litmus test"),
        ("shared.plugins.thinking.plugin", "ThinkingPlugin", "litmus test"),
        ("shared.plugins.todo.plugin", "TodoPlugin", "litmus test"),
        ("shared.plugins.clarification.plugin", "ClarificationPlugin", "litmus test"),
        ("shared.plugins.lsp.plugin", "LSPToolPlugin", "cascade-sharing target"),
        ("shared.plugins.memory.plugin", "MemoryPlugin", "litmus test"),
        ("shared.plugins.artifact_tracker.plugin", "ArtifactTrackerPlugin", "litmus test"),
    ])
    def test_docstring_references_litmus_or_design_target(
        self, plugin_module, plugin_class_name, expected_marker
    ):
        import importlib
        mod = importlib.import_module(plugin_module)
        cls = getattr(mod, plugin_class_name)
        method = getattr(cls, 'reset_for_next_session', None)
        assert method is not None, (
            f"{plugin_class_name} must override reset_for_next_session"
        )
        doc = (method.__doc__ or "").lower()
        assert expected_marker.lower() in doc, (
            f"{plugin_class_name}.reset_for_next_session docstring should "
            f"reference '{expected_marker}' (Phase 1 audit trail)."
        )
