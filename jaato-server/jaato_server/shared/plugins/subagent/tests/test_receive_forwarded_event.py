"""Tests for ``SubagentPlugin.receive_forwarded_event`` (Phase 4 §4.3.6b).

This is the runner-side endpoint that the daemon dispatches to when
forwarding an event from an isolated sub-runner.  Mirrors the
default-share path's ``_parent_session.inject_prompt`` contract so
the parent model sees isolated + in-runner subagents identically.

Test surfaces:
1. event_kind="output" → inject_prompt with text content.
2. event_kind="status" → inject_prompt + ui_hooks.on_agent_status_changed.
3. event_kind="error" → inject_prompt with ERROR header + ui_hooks.
4. event_kind unknown → return ok=False with descriptive error.
5. No parent session → return ok=False; nothing injected.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ..plugin import SubagentPlugin


def _make_plugin_with_parent():
    """SubagentPlugin with mocked parent session + ui_hooks for
    inject_prompt assertions."""
    plugin = SubagentPlugin()
    parent_session = MagicMock()
    plugin._parent_session = parent_session
    plugin._ui_hooks = MagicMock()
    return plugin, parent_session


class TestOutputEvent:
    def test_inject_prompt_called_with_text(self):
        plugin, parent = _make_plugin_with_parent()
        result = plugin.receive_forwarded_event(
            subagent_id="agent-1",
            event_kind="output",
            event_payload={
                "text": "hello world",
                "source": "assistant",
            },
        )
        assert result == {"ok": True}
        parent.inject_prompt.assert_called_once()
        call_kwargs = parent.inject_prompt.call_args
        injected_text = call_kwargs[0][0]
        assert "hello world" in injected_text
        assert "agent-1" in injected_text
        assert "source=assistant" in injected_text

    def test_missing_text_field_injects_empty(self):
        """Defensive: empty text still injects (parent's queue
        de-dupes / ignores empties)."""
        plugin, parent = _make_plugin_with_parent()
        result = plugin.receive_forwarded_event(
            subagent_id="agent-1",
            event_kind="output",
            event_payload={},
        )
        assert result == {"ok": True}
        parent.inject_prompt.assert_called_once()


class TestStatusEvent:
    def test_inject_prompt_and_ui_hook_fire(self):
        plugin, parent = _make_plugin_with_parent()
        result = plugin.receive_forwarded_event(
            subagent_id="agent-1",
            event_kind="status",
            event_payload={"status": "done"},
        )
        assert result == {"ok": True}
        parent.inject_prompt.assert_called_once()
        plugin._ui_hooks.on_agent_status_changed.assert_called_once_with(
            agent_id="agent-1", status="done",
        )

    def test_ui_hook_exception_does_not_propagate(self):
        """Mirror default-share resilience — UI-hook crashes don't
        block the inject_prompt path."""
        plugin, parent = _make_plugin_with_parent()
        plugin._ui_hooks.on_agent_status_changed.side_effect = RuntimeError(
            "ui boom",
        )
        result = plugin.receive_forwarded_event(
            subagent_id="agent-1",
            event_kind="status",
            event_payload={"status": "done"},
        )
        assert result == {"ok": True}
        parent.inject_prompt.assert_called_once()


class TestErrorEvent:
    def test_inject_prompt_has_error_header(self):
        plugin, parent = _make_plugin_with_parent()
        result = plugin.receive_forwarded_event(
            subagent_id="agent-1",
            event_kind="error",
            event_payload={"message": "subprocess crashed"},
        )
        assert result == {"ok": True}
        injected = parent.inject_prompt.call_args[0][0]
        assert "ERROR" in injected
        assert "subprocess crashed" in injected
        # ui_hooks reflects error status.
        plugin._ui_hooks.on_agent_status_changed.assert_called_once_with(
            agent_id="agent-1", status="error",
        )


class TestUnknownEventKind:
    def test_returns_ok_false(self):
        plugin, parent = _make_plugin_with_parent()
        result = plugin.receive_forwarded_event(
            subagent_id="agent-1",
            event_kind="unknown_kind",
            event_payload={},
        )
        assert result["ok"] is False
        assert "unrecognized" in result["error"].lower()
        # Nothing injected.
        parent.inject_prompt.assert_not_called()


class TestNoParentSession:
    def test_returns_ok_false_when_no_parent_wired(self):
        plugin = SubagentPlugin()
        # _parent_session stays None by default.
        result = plugin.receive_forwarded_event(
            subagent_id="agent-1",
            event_kind="output",
            event_payload={"text": "hello"},
        )
        assert result["ok"] is False
        assert "no parent session" in result["error"].lower()
