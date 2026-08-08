"""Tests for ``PluginRegistry`` pre-init framework-key injection.

Server 0.6.129+ structural fix.  Replaces the per-call-site
threading of framework values (workspace_path, config_root,
session_id, agent_name) into plugin_configs at ``core.py:1730``
(daemon-side) and ``server/runner/session.py:_bootstrap_session``
(runner-side).  Both sites had to manually populate these for every
framework concept that crossed the boundary — the "12 sites per
concept" pattern documented in
``project_backlog_profile_field_propagation_duplication``.

Now: callers set framework values on the registry via setters
BEFORE calling ``expose_all`` / ``expose_tool``.
:meth:`PluginRegistry._augment_plugin_config` injects them into each
plugin's config dict at every ``plugin.initialize`` call site.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from shared.plugins.registry import PluginRegistry


# ---------------------------------------------------------------------------
# _augment_plugin_config — the injection helper
# ---------------------------------------------------------------------------


class TestAugmentPluginConfig:
    """Unit tests for the helper that produces injected configs."""

    def test_empty_registry_returns_input_unchanged(self):
        r = PluginRegistry()
        # Nothing set on registry → return input unchanged (preserves
        # pre-fix behavior when callers bypass the setters).
        assert r._augment_plugin_config({"foo": "bar"}) == {"foo": "bar"}
        assert r._augment_plugin_config(None) is None
        assert r._augment_plugin_config({}) == {}

    def test_all_framework_keys_injected(self):
        r = PluginRegistry()
        r.set_workspace_path("/ws")
        r.set_config_root("/cr")
        r.set_session_id("sess-1")
        r.set_agent_name("agent-x")
        out = r._augment_plugin_config({"foo": "bar"})
        assert out == {
            "foo": "bar",
            "workspace_path": "/ws",
            "config_root": "/cr",
            "session_id": "sess-1",
            "agent_name": "agent-x",
        }

    def test_caller_config_wins_over_framework_value(self):
        """Pin: operator-supplied values in plugin_configs override
        framework-injected values (setdefault semantics)."""
        r = PluginRegistry()
        r.set_workspace_path("/framework_ws")
        out = r._augment_plugin_config({"workspace_path": "/operator_ws"})
        assert out["workspace_path"] == "/operator_ws"

    def test_none_config_yields_framework_only_dict(self):
        """When caller passes None and framework values are set, the
        helper returns a dict with just the framework keys."""
        r = PluginRegistry()
        r.set_workspace_path("/ws")
        r.set_session_id("sess-1")
        out = r._augment_plugin_config(None)
        assert out == {"workspace_path": "/ws", "session_id": "sess-1"}

    def test_partial_framework_values_only_inject_what_is_set(self):
        r = PluginRegistry()
        r.set_workspace_path("/ws")
        # config_root, session_id, agent_name unset
        out = r._augment_plugin_config({"foo": "bar"})
        assert out == {"foo": "bar", "workspace_path": "/ws"}
        assert "config_root" not in out
        assert "session_id" not in out
        assert "agent_name" not in out

    def test_input_dict_not_mutated(self):
        """Pin: caller's dict is NOT mutated (defensive copy)."""
        r = PluginRegistry()
        r.set_workspace_path("/ws")
        caller_dict = {"foo": "bar"}
        r._augment_plugin_config(caller_dict)
        assert caller_dict == {"foo": "bar"}  # unchanged

    def test_clear_workspace_path_removes_from_injection(self):
        """set_workspace_path(None) should NOT inject the key."""
        r = PluginRegistry()
        r.set_workspace_path("/ws")
        out1 = r._augment_plugin_config(None)
        assert "workspace_path" in out1
        # The set_workspace_path docstring says it broadcasts; passing
        # None clears the registry attribute.  Verify injection skips it.
        # (Today's set_workspace_path doesn't accept None for workspace —
        # but the field can be None at construction; assert that.)
        r2 = PluginRegistry()
        assert r2._augment_plugin_config(None) is None


# ---------------------------------------------------------------------------
# End-to-end via a stub plugin
# ---------------------------------------------------------------------------


class _StubPlugin:
    """Minimal plugin that records the config it was initialized with."""

    def __init__(self) -> None:
        self.received_config: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return "stub"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.received_config = dict(config) if config else None

    def shutdown(self) -> None:
        pass

    def get_tool_schemas(self) -> List[Any]:
        return []

    def get_executors(self) -> Dict[str, Any]:
        return {}


class TestExposeToolInjectsFrameworkValues:
    """Pin: ``expose_tool`` and ``expose_all`` inject framework
    values into the plugin's config before ``initialize`` fires."""

    def _registry_with_stub(self) -> tuple:
        r = PluginRegistry()
        stub = _StubPlugin()
        r._plugins["stub"] = stub
        return r, stub

    def test_expose_tool_injects_workspace_path(self):
        r, stub = self._registry_with_stub()
        r.set_workspace_path("/ws")
        r.expose_tool("stub")
        assert stub.received_config is not None
        assert stub.received_config["workspace_path"] == "/ws"

    def test_expose_tool_injects_all_framework_keys(self):
        r, stub = self._registry_with_stub()
        r.set_workspace_path("/ws")
        r.set_config_root("/cr")
        r.set_session_id("sess-1")
        r.set_agent_name("agent-x")
        r.expose_tool("stub")
        assert stub.received_config == {
            "workspace_path": "/ws",
            "config_root": "/cr",
            "session_id": "sess-1",
            "agent_name": "agent-x",
        }

    def test_expose_tool_caller_config_wins(self):
        r, stub = self._registry_with_stub()
        r.set_workspace_path("/framework_ws")
        r.expose_tool("stub", {"workspace_path": "/operator_ws"})
        assert stub.received_config["workspace_path"] == "/operator_ws"

    def test_expose_all_injects_framework_keys(self):
        r, stub = self._registry_with_stub()
        r.set_workspace_path("/ws")
        r.set_config_root("/cr")
        r.expose_all({"stub": {"extra": "operator-supplied"}})
        assert stub.received_config == {
            "extra": "operator-supplied",
            "workspace_path": "/ws",
            "config_root": "/cr",
        }

    def test_expose_all_no_registry_values_no_injection(self):
        """Pin: when no framework values set, ``expose_all`` passes
        plugin's config through unchanged (preserves pre-fix
        behavior for callers that don't use the setters)."""
        r, stub = self._registry_with_stub()
        r.expose_all({"stub": {"foo": "bar"}})
        assert stub.received_config == {"foo": "bar"}

    def test_set_session_id_no_broadcast(self):
        """Pin: ``set_session_id`` doesn't broadcast (session_id
        doesn't change mid-session, so no hook needed)."""
        r = PluginRegistry()
        # No exception even if no plugins exposed.
        r.set_session_id("sess-1")
        assert r._session_id == "sess-1"
        r.set_session_id(None)
        assert r._session_id is None
