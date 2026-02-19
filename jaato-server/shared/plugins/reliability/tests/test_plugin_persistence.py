"""Tests for reliability plugin session persistence."""

import pytest
from datetime import datetime

from ..types import (
    EscalationRule,
    ReliabilityConfig,
    ToolReliabilityState,
    TrustState,
)
from ..persistence import SessionSettings
from ..plugin import ReliabilityPlugin


class TestReliabilityPluginPersistence:
    """Tests for get_persistence_state / restore_persistence_state."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin instance."""
        config = ReliabilityConfig(
            default_rule=EscalationRule(
                count_threshold=3,
                window_seconds=3600,
            )
        )
        return ReliabilityPlugin(config)

    def test_empty_state_returns_empty_dict(self, plugin):
        """No tool states and zero turn index -> empty state."""
        state = plugin.get_persistence_state()
        assert state == {}

    def test_state_captures_tool_states(self, plugin):
        """Tool reliability states are serialized."""
        plugin._tool_states["cli|cmd=curl"] = ToolReliabilityState(
            failure_key="cli|cmd=curl",
            tool_name="cli",
            state=TrustState.ESCALATED,
            consecutive_failures=3,
            total_failures=5,
            total_successes=10,
        )
        plugin._turn_index = 7

        state = plugin.get_persistence_state()

        assert state["version"] == 1
        assert state["turn_index"] == 7
        assert "cli|cmd=curl" in state["tool_states"]
        ts = state["tool_states"]["cli|cmd=curl"]
        assert ts["state"] == "escalated"
        assert ts["consecutive_failures"] == 3

    def test_state_captures_session_settings(self, plugin):
        """Non-default session settings are included."""
        plugin._session_settings = SessionSettings(
            nudge_level="gentle",
            recovery_mode="ask",
        )
        plugin._turn_index = 1

        state = plugin.get_persistence_state()

        assert "session_settings" in state
        assert state["session_settings"]["nudge_level"] == "gentle"

    def test_state_omits_default_session_settings(self, plugin):
        """Default session settings are not included."""
        plugin._turn_index = 1  # Need non-empty state

        state = plugin.get_persistence_state()

        assert "session_settings" not in state

    def test_restore_tool_states(self, plugin):
        """Tool states are restored from dict."""
        state = {
            "version": 1,
            "tool_states": {
                "cli|cmd=curl": {
                    "failure_key": "cli|cmd=curl",
                    "tool_name": "cli",
                    "state": "escalated",
                    "consecutive_failures": 3,
                    "total_failures": 5,
                    "total_successes": 10,
                    "failures_in_window": 3.0,
                },
            },
            "turn_index": 7,
            "session_id": "test-session",
        }

        plugin.restore_persistence_state(state)

        assert plugin._turn_index == 7
        assert plugin._session_id == "test-session"
        assert "cli|cmd=curl" in plugin._tool_states
        ts = plugin._tool_states["cli|cmd=curl"]
        assert ts.state == TrustState.ESCALATED
        assert ts.consecutive_failures == 3

    def test_restore_session_settings(self, plugin):
        """Session settings are restored."""
        state = {
            "version": 1,
            "turn_index": 0,
            "session_settings": {
                "nudge_level": "gentle",
                "recovery_mode": "ask",
            },
        }

        plugin.restore_persistence_state(state)

        assert plugin._session_settings.nudge_level == "gentle"
        assert plugin._session_settings.recovery_mode == "ask"

    def test_roundtrip(self, plugin):
        """State survives a save/restore cycle."""
        plugin._session_id = "sess-42"
        plugin._turn_index = 12
        plugin._tool_states["http|domain=api.github.com"] = ToolReliabilityState(
            failure_key="http|domain=api.github.com",
            tool_name="http_request",
            state=TrustState.RECOVERING,
            consecutive_failures=0,
            total_failures=4,
            total_successes=20,
            successes_since_recovery=2,
            successes_needed=3,
        )
        plugin._session_settings = SessionSettings(
            nudge_level="firm",
        )

        state = plugin.get_persistence_state()

        # Restore into fresh plugin
        plugin2 = ReliabilityPlugin()
        plugin2.restore_persistence_state(state)

        assert plugin2._session_id == "sess-42"
        assert plugin2._turn_index == 12
        assert "http|domain=api.github.com" in plugin2._tool_states
        ts = plugin2._tool_states["http|domain=api.github.com"]
        assert ts.state == TrustState.RECOVERING
        assert ts.successes_since_recovery == 2
        assert plugin2._session_settings.nudge_level == "firm"

    def test_restore_with_empty_state(self, plugin):
        """Restoring empty state doesn't crash."""
        plugin.restore_persistence_state({})

        assert plugin._turn_index == 0
        assert plugin._tool_states == {}

    def test_restore_preserves_existing_session_id(self, plugin):
        """If state has no session_id, existing one is preserved."""
        plugin._session_id = "existing-session"

        plugin.restore_persistence_state({"version": 1, "turn_index": 5})

        assert plugin._session_id == "existing-session"
        assert plugin._turn_index == 5
