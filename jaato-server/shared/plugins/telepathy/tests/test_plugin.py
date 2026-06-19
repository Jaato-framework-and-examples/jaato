"""Tests for the TelepathyPlugin extracted from
``shared/jaato_session.py`` 2026-06-07.

Pins:
    1. Schema shape (name, parameters) — preserve verbatim from the
       pre-extraction session built-in so existing prose / agent
       memories that reference ``share_context`` still match.
    2. ``is_tool_visible`` predicate (PR #241 hook):
       - share_context HIDDEN when host session has no parent
       - share_context VISIBLE when host session has a parent
       - other tool names: predicate has no opinion (returns True)
    3. Executor behavior:
       - empty payload → ``No context to share`` error
       - no parent → ``No parent session available`` error
         (defensive — visibility filter should have hidden the tool;
         only fires on edge cases like stale cached message history)
       - happy path → calls parent_session.inject_prompt with the
         formatted body
    4. ``_format_shared_context`` produces the same XML-tagged shape
       as the pre-extraction version (subagent plugin's prose
       depends on it).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from shared.session_context import _current_session, set_current_session

from ..plugin import TelepathyPlugin, create_plugin


@pytest.fixture(autouse=True)
def _isolate_session_context():
    """Reset the per-execution session ContextVar around every test.

    Telepathy resolves the host session from
    ``shared.session_context.get_current_session()`` (NOT from ``self``,
    to avoid cross-subagent leakage), so tests wire a session by calling
    ``set_current_session(...)`` instead of the old ``plugin.set_session``.
    Seed it to ``None`` so "no session" tests see a clean context, and
    reset after so nothing leaks between tests.
    """
    token = _current_session.set(None)
    try:
        yield
    finally:
        _current_session.reset(token)


# ============================================================
# Factory + lifecycle
# ============================================================


class TestFactoryAndLifecycle:

    def test_create_plugin_returns_instance(self):
        plugin = create_plugin()
        assert isinstance(plugin, TelepathyPlugin)
        assert plugin.name == "telepathy"

    def test_initialize_is_idempotent(self):
        plugin = TelepathyPlugin()
        plugin.initialize()
        plugin.initialize({"some": "config"})  # config accepted but ignored
        assert plugin._initialized is True

    def test_shutdown_resets_initialized(self):
        # Telepathy holds no per-session state on self (the session is
        # resolved per-execution from the ContextVar), so shutdown only
        # needs to flip _initialized back.
        plugin = TelepathyPlugin()
        plugin.initialize()
        plugin.shutdown()
        assert plugin._initialized is False
        assert not hasattr(plugin, "_session")


# ============================================================
# Schema
# ============================================================


class TestSchema:

    def test_share_context_schema_present(self):
        plugin = TelepathyPlugin()
        schemas = plugin.get_tool_schemas()
        names = [s.name for s in schemas]
        assert names == ["share_context"]

    def test_schema_parameters_shape(self):
        plugin = TelepathyPlugin()
        schema = plugin.get_tool_schemas()[0]
        props = schema.parameters["properties"]
        assert set(props.keys()) == {"files", "findings", "notes"}
        assert props["files"]["type"] == "object"
        assert props["findings"]["type"] == "array"
        assert props["notes"]["type"] == "string"
        # None required — all three keys are optional
        assert schema.parameters["required"] == []

    def test_share_context_auto_approved(self):
        """Subagents have no human in their loop — no permission
        prompts for share_context."""
        plugin = TelepathyPlugin()
        assert "share_context" in plugin.get_auto_approved_tools()


# ============================================================
# is_tool_visible (PR #241 hook)
# ============================================================


class TestVisibilityPredicate:

    def test_hidden_when_no_session(self):
        """Before set_session has been called the plugin has no
        session context — hide share_context defensively."""
        plugin = TelepathyPlugin()
        assert plugin.is_tool_visible("share_context") is False

    def test_hidden_when_session_has_no_parent(self):
        """The vLLM-smoke case: root session, no parent set."""
        plugin = TelepathyPlugin()
        session = MagicMock(_parent_session=None)
        set_current_session(session)
        assert plugin.is_tool_visible("share_context") is False

    def test_visible_when_session_has_parent(self):
        """Subagent case: parent_session wired by SubagentPlugin."""
        plugin = TelepathyPlugin()
        parent = MagicMock()
        session = MagicMock(_parent_session=parent)
        set_current_session(session)
        assert plugin.is_tool_visible("share_context") is True

    def test_other_tool_names_pass_through(self):
        """The predicate has no opinion on tools owned by other
        plugins — must return True so they're not silently hidden."""
        plugin = TelepathyPlugin()
        set_current_session(MagicMock(_parent_session=None))
        assert plugin.is_tool_visible("cli_based_tool") is True
        assert plugin.is_tool_visible("signal_completion") is True
        assert plugin.is_tool_visible("createPlan") is True


# ============================================================
# Executor
# ============================================================


class TestExecutor:

    def _make_plugin_with_parent(self, parent_running: bool = True):
        plugin = TelepathyPlugin()
        parent = MagicMock()
        parent.is_running = parent_running
        session = MagicMock(_parent_session=parent, _agent_id="test-agent")
        set_current_session(session)
        return plugin, parent

    def test_no_session_returns_programmer_error(self):
        plugin = TelepathyPlugin()
        # No session in the ContextVar (autouse fixture seeds it to None).
        result = plugin._execute_share_context({"notes": "x"})
        assert result["success"] is False
        assert "no session in context" in result["error"].lower()

    def test_empty_payload_returns_error(self):
        plugin, _ = self._make_plugin_with_parent()
        result = plugin._execute_share_context({})
        assert result["success"] is False
        assert "No context to share" in result["error"]

    def test_no_parent_returns_error(self):
        plugin = TelepathyPlugin()
        set_current_session(MagicMock(_parent_session=None))
        result = plugin._execute_share_context({"notes": "anything"})
        assert result["success"] is False
        assert "No parent session" in result["error"]

    def test_happy_path_parent_busy_queues(self):
        plugin, parent = self._make_plugin_with_parent(parent_running=True)

        result = plugin._execute_share_context({
            "files": {"src/foo.py": "print('hello')"},
            "findings": ["foo is the entrypoint"],
            "notes": "first pass on the codebase",
        })

        assert result["success"] is True
        assert result["status"] == "queued"
        assert "files" in result["shared"]
        assert result["shared"]["files"] == ["src/foo.py"]
        assert result["shared"]["findings_count"] == 1
        assert result["shared"]["has_notes"] is True

        # inject_prompt was called with the formatted body
        parent.inject_prompt.assert_called_once()
        call_kwargs = parent.inject_prompt.call_args.kwargs
        assert call_kwargs["source_id"] == "test-agent"
        # body is the first positional arg
        body = parent.inject_prompt.call_args.args[0]
        assert "src/foo.py" in body
        assert "print('hello')" in body
        assert "foo is the entrypoint" in body

    def test_happy_path_parent_idle_sends(self):
        plugin, parent = self._make_plugin_with_parent(parent_running=False)

        result = plugin._execute_share_context({"notes": "ack"})

        assert result["success"] is True
        assert result["status"] == "sent"
        parent.inject_prompt.assert_called_once()

    def test_inject_prompt_exception_surfaces(self):
        plugin, parent = self._make_plugin_with_parent(parent_running=True)
        parent.inject_prompt.side_effect = RuntimeError("queue exploded")

        result = plugin._execute_share_context({"notes": "x"})

        assert result["success"] is False
        assert "Failed to share context" in result["error"]
        assert "queue exploded" in result["error"]


# ============================================================
# _format_shared_context
# ============================================================


class TestFormatSharedContext:

    def test_only_notes(self):
        body = TelepathyPlugin._format_shared_context({}, [], "just a note")
        assert "<shared_context" in body
        assert "<notes>" in body
        assert "just a note" in body
        assert "<files>" not in body
        assert "<findings>" not in body

    def test_only_findings(self):
        body = TelepathyPlugin._format_shared_context({}, ["A", "B"], "")
        assert "<findings>" in body
        assert "  - A" in body
        assert "  - B" in body

    def test_files_include_re_read_warning(self):
        """Pin: when files are shared the prefix instructs the
        parent NOT to re-read them."""
        body = TelepathyPlugin._format_shared_context(
            {"src/foo.py": "X"}, [], "",
        )
        assert "DO NOT re-read" in body
        assert '<file path="src/foo.py">' in body
        assert "X" in body


# ============================================================
# Plugin discovery contract
# ============================================================


class TestPluginDiscovery:
    """Pin: the package exposes the constants the registry walks."""

    def test_plugin_kind_is_tool(self):
        from .. import PLUGIN_KIND
        assert PLUGIN_KIND == "tool"

    def test_plugin_tier_is_runner(self):
        from .. import PLUGIN_TIER
        assert PLUGIN_TIER == "runner"
