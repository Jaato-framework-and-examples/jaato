"""Tests for the generic plugin state persistence mechanism.

Tests the SessionManager's generic loop that iterates all exposed plugins
and calls get_persistence_state/restore_persistence_state on them.
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock


class FakePlugin:
    """A minimal plugin that implements persistence."""

    def __init__(self, plugin_name: str, state: Optional[Dict[str, Any]] = None):
        self._name = plugin_name
        self._state = state or {}
        self.restore_called_with = None

    @property
    def name(self) -> str:
        return self._name

    def get_persistence_state(self) -> Dict[str, Any]:
        return self._state

    def restore_persistence_state(self, state: Dict[str, Any]) -> None:
        self.restore_called_with = state
        self._state = state


class FakePluginNoPersistence:
    """A plugin that does NOT implement persistence."""

    def __init__(self, plugin_name: str):
        self._name = plugin_name

    @property
    def name(self) -> str:
        return self._name


class FakePluginBrokenSave:
    """A plugin whose get_persistence_state raises."""

    def __init__(self, plugin_name: str):
        self._name = plugin_name

    @property
    def name(self) -> str:
        return self._name

    def get_persistence_state(self) -> Dict[str, Any]:
        raise RuntimeError("save failed")

    def restore_persistence_state(self, state: Dict[str, Any]) -> None:
        pass


class FakeRegistry:
    """Minimal registry for testing the generic loop."""

    def __init__(self, plugins: Dict[str, Any]):
        self._plugins = plugins

    def list_exposed(self) -> List[str]:
        return list(self._plugins.keys())

    def get_plugin(self, name: str):
        return self._plugins.get(name)


class TestGenericPluginPersistenceLoop:
    """Test the generic save/restore loop logic used by SessionManager.

    These tests validate the loop algorithm in isolation, without needing
    a full SessionManager or JaatoServer.
    """

    def _save_loop(self, registry: FakeRegistry) -> Dict[str, Any]:
        """Simulate SessionManager._save_session's generic plugin loop."""
        plugin_states = {}
        _DEDICATED_PLUGINS = {'subagent', 'todo'}

        for plugin_name in registry.list_exposed():
            if plugin_name in _DEDICATED_PLUGINS:
                continue
            plugin = registry.get_plugin(plugin_name)
            if plugin and hasattr(plugin, 'get_persistence_state'):
                try:
                    pstate = plugin.get_persistence_state()
                    if pstate:
                        plugin_states[plugin_name] = pstate
                except Exception:
                    pass  # Silently skip broken plugins

        return plugin_states

    def _restore_loop(
        self, registry: FakeRegistry, plugin_states: Dict[str, Any]
    ) -> None:
        """Simulate SessionManager._load_session's generic plugin loop."""
        for plugin_name, plugin_state in plugin_states.items():
            plugin = registry.get_plugin(plugin_name)
            if plugin and hasattr(plugin, 'restore_persistence_state'):
                try:
                    plugin.restore_persistence_state(plugin_state)
                except Exception:
                    pass

    def test_save_collects_state_from_persistent_plugins(self):
        """Plugins with state are collected."""
        registry = FakeRegistry({
            "service_connector": FakePlugin("service_connector", {"services": ["gh"]}),
            "reliability": FakePlugin("reliability", {"turn_index": 5}),
        })

        states = self._save_loop(registry)

        assert "service_connector" in states
        assert states["service_connector"] == {"services": ["gh"]}
        assert "reliability" in states
        assert states["reliability"] == {"turn_index": 5}

    def test_save_skips_dedicated_plugins(self):
        """Subagent and todo are skipped (they have dedicated persistence)."""
        registry = FakeRegistry({
            "subagent": FakePlugin("subagent", {"agents": []}),
            "todo": FakePlugin("todo", {"plans": []}),
            "reliability": FakePlugin("reliability", {"turn_index": 1}),
        })

        states = self._save_loop(registry)

        assert "subagent" not in states
        assert "todo" not in states
        assert "reliability" in states

    def test_save_skips_plugins_without_persistence(self):
        """Plugins without get_persistence_state are ignored."""
        registry = FakeRegistry({
            "cli": FakePluginNoPersistence("cli"),
            "reliability": FakePlugin("reliability", {"turn_index": 1}),
        })

        states = self._save_loop(registry)

        assert "cli" not in states
        assert "reliability" in states

    def test_save_skips_plugins_with_empty_state(self):
        """Plugins returning empty dict are not stored."""
        registry = FakeRegistry({
            "service_connector": FakePlugin("service_connector", {}),
        })

        states = self._save_loop(registry)

        assert "service_connector" not in states

    def test_save_handles_broken_plugin_gracefully(self):
        """A plugin that raises during save doesn't break other plugins."""
        registry = FakeRegistry({
            "broken": FakePluginBrokenSave("broken"),
            "reliability": FakePlugin("reliability", {"turn_index": 3}),
        })

        states = self._save_loop(registry)

        assert "broken" not in states
        assert "reliability" in states

    def test_restore_calls_plugins(self):
        """Restore calls restore_persistence_state on each plugin."""
        svc = FakePlugin("service_connector")
        rel = FakePlugin("reliability")
        registry = FakeRegistry({
            "service_connector": svc,
            "reliability": rel,
        })

        plugin_states = {
            "service_connector": {"services": ["gh"]},
            "reliability": {"turn_index": 5},
        }

        self._restore_loop(registry, plugin_states)

        assert svc.restore_called_with == {"services": ["gh"]}
        assert rel.restore_called_with == {"turn_index": 5}

    def test_restore_skips_missing_plugins(self):
        """Plugins in state but not in registry are skipped."""
        registry = FakeRegistry({})

        plugin_states = {
            "service_connector": {"services": ["gh"]},
        }

        # Should not raise
        self._restore_loop(registry, plugin_states)

    def test_restore_skips_plugins_without_restore_method(self):
        """Plugins without restore_persistence_state are skipped."""
        registry = FakeRegistry({
            "cli": FakePluginNoPersistence("cli"),
        })

        plugin_states = {
            "cli": {"some": "state"},
        }

        # Should not raise
        self._restore_loop(registry, plugin_states)

    def test_full_roundtrip(self):
        """Save then restore preserves plugin state."""
        svc = FakePlugin("service_connector", {"services": ["gh", "stripe"]})
        rel = FakePlugin("reliability", {"turn_index": 10, "tool_states": {}})

        registry = FakeRegistry({
            "service_connector": svc,
            "reliability": rel,
            "cli": FakePluginNoPersistence("cli"),
            "todo": FakePlugin("todo", {"plans": ["should-be-skipped"]}),
        })

        # Save
        states = self._save_loop(registry)

        # Create fresh plugins (simulating session reload)
        svc2 = FakePlugin("service_connector")
        rel2 = FakePlugin("reliability")
        registry2 = FakeRegistry({
            "service_connector": svc2,
            "reliability": rel2,
            "cli": FakePluginNoPersistence("cli"),
        })

        # Restore
        self._restore_loop(registry2, states)

        assert svc2._state == {"services": ["gh", "stripe"]}
        assert rel2._state == {"turn_index": 10, "tool_states": {}}
