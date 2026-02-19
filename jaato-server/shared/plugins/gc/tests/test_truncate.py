"""Tests for TruncateGCPlugin."""

import pytest

from shared.plugins.gc import GCConfig, GCTriggerReason
from shared.plugins.gc_truncate import TruncateGCPlugin, create_plugin
from jaato import Message, Part, Role


def make_message(role: str, text: str) -> Message:
    """Helper to create Message objects."""
    r = Role.USER if role == "user" else Role.MODEL
    return Message(
        role=r,
        parts=[Part(text=text)]
    )


def make_history(num_turns: int) -> list:
    """Create a history with N turns (user+model pairs)."""
    history = []
    for i in range(num_turns):
        history.append(make_message("user", f"User message {i}"))
        history.append(make_message("model", f"Model response {i}"))
    return history


class TestTruncateGCPlugin:
    def test_create_plugin(self):
        plugin = create_plugin()
        assert plugin.name == "gc_truncate"

    def test_initialize(self):
        plugin = TruncateGCPlugin()
        plugin.initialize({"preserve_recent_turns": 10})
        assert plugin._initialized
        assert plugin._config.get("preserve_recent_turns") == 10

    def test_shutdown(self):
        plugin = TruncateGCPlugin()
        plugin.initialize({"key": "value"})
        plugin.shutdown()
        assert not plugin._initialized
        assert plugin._config == {}


class TestShouldCollect:
    def test_auto_trigger_disabled(self):
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(auto_trigger=False, threshold_percent=50.0)
        context = {"percent_used": 90.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert not should_collect
        assert reason is None

    def test_threshold_triggered(self):
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(threshold_percent=75.0)
        context = {"percent_used": 80.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert should_collect
        assert reason == GCTriggerReason.THRESHOLD

    def test_threshold_not_reached(self):
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(threshold_percent=75.0)
        context = {"percent_used": 50.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert not should_collect

    def test_turn_limit_triggered(self):
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(threshold_percent=90.0, max_turns=10)
        context = {"percent_used": 50.0, "turns": 15}

        should_collect, reason = plugin.should_collect(context, config)
        assert should_collect
        assert reason == GCTriggerReason.TURN_LIMIT


class TestCollect:
    def test_nothing_to_collect(self):
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 10})

        history = make_history(5)  # Only 5 turns
        config = GCConfig(preserve_recent_turns=10)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.success
        assert result.items_collected == 0
        assert new_history == history

    def test_truncates_old_turns(self):
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 3})

        history = make_history(10)  # 10 turns
        config = GCConfig(preserve_recent_turns=3)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.success
        assert result.items_collected == 7  # Removed 7 turns
        assert result.tokens_freed > 0
        assert len(new_history) == 6  # 3 turns * 2 content each

    def test_preserves_pinned_indices(self):
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 2})

        history = make_history(10)
        config = GCConfig(
            preserve_recent_turns=2,
            pinned_turn_indices=[0, 3]  # Pin first and fourth turn
        )
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.success
        # Preserved: 0 (pinned), 3 (pinned), 8, 9 (recent) = 4 turns
        assert result.items_collected == 6

    def test_plugin_config_overrides_gc_config(self):
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 5})  # Plugin says 5

        history = make_history(10)
        config = GCConfig(preserve_recent_turns=2)  # GCConfig says 2
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        # Plugin config wins, so 5 turns preserved
        assert result.items_collected == 5

    def test_notification_when_enabled(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 2,
            "notify_on_gc": True
        })

        history = make_history(5)
        config = GCConfig(preserve_recent_turns=2)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.notification is not None
        assert "removed" in result.notification
        # Notification content should be prepended
        assert "[System:" in new_history[0].parts[0].text

    def test_custom_notification_template(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 2,
            "notify_on_gc": True,
            "notification_template": "Cleared {removed} turns!"
        })

        history = make_history(5)
        config = GCConfig(preserve_recent_turns=2)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.notification == "Cleared 3 turns!"

    def test_result_details(self):
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 3})

        history = make_history(10)
        config = GCConfig(preserve_recent_turns=3)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.details["turns_before"] == 10
        assert result.details["turns_after"] == 3
        assert result.details["preserve_count"] == 3
        assert result.plugin_name == "gc_truncate"
        assert result.trigger_reason == GCTriggerReason.THRESHOLD
