"""Tests for per-session tool-scope filtering on JaatoSession.

The profile ``plugins`` modifier ``tools:[...]`` declares a per-plugin
allow-list.  ``JaatoSession`` enforces it per-session (NOT via the shared
registry's ``disable_tool`` — that would leak a scope onto sibling
subagents sharing the same runtime) so a tool outside its plugin's
allow-list is absent from the wire body AND the provider's grammar
surface.

The load-bearing property — the whole reason this exists for
context-shaving — is that the scoped-out tool is **absent from the list
``_get_tools_for_provider`` returns** (the wire body), not merely
dispatch-blocked.  These tests assert that directly.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..jaato_session import JaatoSession


def _schema(name):
    """A minimal stand-in for a ToolSchema (only ``.name`` is read by
    the scope filter)."""
    return SimpleNamespace(name=name)


def _session_with_scopes(tool_scopes, plugin_of):
    """Build a JaatoSession whose registry maps tool name → plugin via
    ``plugin_of`` (a dict tool_name→plugin_name; missing names → no
    owning plugin, like lifecycle tools)."""
    runtime = MagicMock()

    def get_plugin_for_tool(tool_name):
        pname = plugin_of.get(tool_name)
        return SimpleNamespace(name=pname) if pname is not None else None

    runtime.registry.get_plugin_for_tool.side_effect = get_plugin_for_tool
    session = JaatoSession(runtime, "test-model")
    session._tool_scopes = tool_scopes
    return session, runtime


class TestApplyToolScopes:

    def test_noop_without_scopes(self):
        session, _ = _session_with_scopes({}, {})
        schemas = [_schema("readFile"), _schema("updateFile")]
        assert session._apply_tool_scopes(schemas) is schemas

    def test_drops_tool_outside_allowlist(self):
        plugin_of = {
            "readFile": "file_edit",
            "updateFile": "file_edit",
            "writeNewFile": "file_edit",
        }
        session, _ = _session_with_scopes(
            {"file_edit": ["readFile"]}, plugin_of
        )
        schemas = [
            _schema("readFile"),
            _schema("updateFile"),
            _schema("writeNewFile"),
        ]
        kept = {s.name for s in session._apply_tool_scopes(schemas)}
        assert kept == {"readFile"}

    def test_lifecycle_tool_without_plugin_survives(self):
        # signal_completion has no owning plugin → no scope → always kept
        plugin_of = {"readFile": "file_edit", "updateFile": "file_edit"}
        session, _ = _session_with_scopes(
            {"file_edit": ["readFile"]}, plugin_of
        )
        schemas = [
            _schema("readFile"),
            _schema("updateFile"),
            _schema("signal_completion"),  # plugin_of miss → survives
        ]
        kept = {s.name for s in session._apply_tool_scopes(schemas)}
        assert kept == {"readFile", "signal_completion"}

    def test_unscoped_plugin_untouched(self):
        # todo has no scope entry → all its tools survive
        plugin_of = {
            "readFile": "file_edit",
            "updateFile": "file_edit",
            "addTodo": "todo",
        }
        session, _ = _session_with_scopes(
            {"file_edit": ["readFile"]}, plugin_of
        )
        schemas = [
            _schema("readFile"),
            _schema("updateFile"),
            _schema("addTodo"),
        ]
        kept = {s.name for s in session._apply_tool_scopes(schemas)}
        assert kept == {"readFile", "addTodo"}


class TestWireBodyAbsence:
    """The definitive property: scoped-out tools are ABSENT from the
    list ``_get_tools_for_provider`` hands the provider — not merely
    dispatch-blocked."""

    def _wire_session(self, tool_scopes, plugin_of, tools):
        session, runtime = _session_with_scopes(tool_scopes, plugin_of)
        # Provider uses external tools; no visibility-gating plugins.
        session._provider = SimpleNamespace(
            uses_external_tools=lambda: True
        )
        session._tools = tools
        runtime.registry.list_exposed.return_value = []
        return session

    def test_scoped_out_tool_absent_from_wire_body(self):
        plugin_of = {
            "readFile": "file_edit",
            "updateFile": "file_edit",
            "writeNewFile": "file_edit",
        }
        tools = [
            _schema("readFile"),
            _schema("updateFile"),
            _schema("writeNewFile"),
        ]
        session = self._wire_session(
            {"file_edit": ["readFile"]}, plugin_of, tools
        )
        wire = session._get_tools_for_provider()
        wire_names = {s.name for s in wire}
        # The whole point: updateFile / writeNewFile are GONE from the
        # array sent to the provider, not just blocked at dispatch.
        assert "updateFile" not in wire_names
        assert "writeNewFile" not in wire_names
        assert wire_names == {"readFile"}

    def test_no_scopes_passes_all_tools_to_wire(self):
        plugin_of = {"readFile": "file_edit", "updateFile": "file_edit"}
        tools = [_schema("readFile"), _schema("updateFile")]
        session = self._wire_session({}, plugin_of, tools)
        wire_names = {s.name for s in session._get_tools_for_provider()}
        assert wire_names == {"readFile", "updateFile"}


class TestActivateRespectsScope:
    """A scoped-out discoverable tool must not be activatable even when
    the model discovers it via introspection."""

    def test_activate_skips_scoped_out_tool(self):
        runtime = MagicMock()
        plugin_of = {"readFile": "file_edit", "updateFile": "file_edit"}

        def get_plugin_for_tool(tool_name):
            pname = plugin_of.get(tool_name)
            return SimpleNamespace(name=pname) if pname is not None else None

        runtime.registry.get_plugin_for_tool.side_effect = get_plugin_for_tool
        runtime.registry.get_exposed_tool_schemas.return_value = [
            _schema("readFile"),
            _schema("updateFile"),
        ]
        session = JaatoSession(runtime, "test-model")
        session._provider = SimpleNamespace(uses_external_tools=lambda: True)
        session._tools = []
        session._tool_plugins = None  # no plugin-level filter
        session._tool_scopes = {"file_edit": ["readFile"]}

        activated = session.activate_discovered_tools(["readFile", "updateFile"])
        assert "readFile" in activated
        assert "updateFile" not in activated
