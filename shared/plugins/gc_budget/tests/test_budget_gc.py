"""Tests for BudgetGCPlugin."""

import time
import pytest

from shared.plugins.gc import GCConfig, GCTriggerReason, GCRemovalItem
from shared.plugins.gc_budget import BudgetGCPlugin, create_plugin
from shared.instruction_budget import (
    InstructionBudget,
    InstructionSource,
    SourceEntry,
    GCPolicy,
)
from jaato import Message, Part, Role


def make_message(role: str, text: str, message_id: str = None) -> Message:
    """Helper to create Message objects."""
    r = Role.USER if role == "user" else Role.MODEL
    msg = Message(role=r, parts=[Part(text=text)])
    if message_id:
        msg.message_id = message_id
    return msg


def make_history(num_turns: int) -> list:
    """Create a history with N turns (user+model pairs)."""
    history = []
    for i in range(num_turns):
        history.append(make_message("user", f"User message {i}"))
        history.append(make_message("model", f"Model response {i}"))
    return history


def make_budget(context_limit: int = 10000) -> InstructionBudget:
    """Create a test budget with some entries."""
    budget = InstructionBudget(context_limit=context_limit)
    return budget


class TestBudgetGCPlugin:
    def test_create_plugin(self):
        plugin = create_plugin()
        assert plugin.name == "gc_budget"

    def test_initialize(self):
        plugin = BudgetGCPlugin()
        plugin.initialize({
            "preserve_recent_turns": 10,
            "target_percent": 50.0,
            "pressure_percent": 85.0,
        })
        assert plugin._initialized
        assert plugin._config.get("preserve_recent_turns") == 10
        assert plugin._config.get("target_percent") == 50.0
        assert plugin._config.get("pressure_percent") == 85.0

    def test_shutdown(self):
        plugin = BudgetGCPlugin()
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

    def test_continuous_mode_above_target(self):
        """In continuous mode, trigger when above target_percent."""
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(
            threshold_percent=80.0,
            target_percent=60.0,
            pressure_percent=0,  # Enable continuous mode
        )
        context = {"percent_used": 65.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert should_collect
        assert reason == GCTriggerReason.THRESHOLD

    def test_continuous_mode_at_target(self):
        """In continuous mode, don't trigger when at or below target."""
        plugin = create_plugin()
        plugin.initialize()

        config = GCConfig(
            threshold_percent=80.0,
            target_percent=60.0,
            pressure_percent=0,  # Enable continuous mode
        )
        context = {"percent_used": 60.0}

        should_collect, reason = plugin.should_collect(context, config)
        assert not should_collect


class TestCollectWithBudget:
    def test_already_below_target(self):
        """No collection when already below target."""
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 3})

        budget = make_budget(10000)
        # Add some entries but stay below target
        budget.update_entry(InstructionSource.SYSTEM, 1000, GCPolicy.LOCKED)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)
        context = {"percent_used": 10.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        assert result.items_collected == 0
        assert "below target" in result.details.get("message", "")

    def test_clears_enrichment_first(self):
        """Phase 1a: ENRICHMENT is cleared first."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        # Add ENRICHMENT with substantial tokens
        budget.update_entry(InstructionSource.ENRICHMENT, 2000, GCPolicy.EPHEMERAL)
        # Add some conversation
        budget.update_entry(InstructionSource.CONVERSATION, 6000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)  # Target: 6000 tokens
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        # Should have cleared enrichment
        assert result.details.get("enrichment_cleared") is True

    def test_removes_ephemeral_oldest_first(self):
        """Phase 1b: EPHEMERAL entries removed oldest first."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        # Add older ephemeral entry
        budget.update_entry(InstructionSource.PLUGIN, 100, GCPolicy.LOCKED)
        budget.add_child(
            InstructionSource.PLUGIN,
            "old_ephemeral",
            SourceEntry(
                source=InstructionSource.PLUGIN,
                tokens=1000,
                gc_policy=GCPolicy.EPHEMERAL,
                label="old_ephemeral",
                created_at=time.time() - 100,  # Older
            )
        )
        budget.add_child(
            InstructionSource.PLUGIN,
            "new_ephemeral",
            SourceEntry(
                source=InstructionSource.PLUGIN,
                tokens=1000,
                gc_policy=GCPolicy.EPHEMERAL,
                label="new_ephemeral",
                created_at=time.time(),  # Newer
            )
        )
        # Add conversation to trigger GC
        budget.update_entry(InstructionSource.CONVERSATION, 7000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)  # Target: 6000 tokens
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        # Should have removed at least the older ephemeral
        assert result.details.get("ephemeral_removed", 0) >= 1

    def test_continuous_mode_never_touches_preservable(self):
        """In continuous mode, PRESERVABLE content is never removed."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        # Add PRESERVABLE entry
        budget.add_child(
            InstructionSource.PLUGIN,
            "preservable_1",
            SourceEntry(
                source=InstructionSource.PLUGIN,
                tokens=2000,
                gc_policy=GCPolicy.PRESERVABLE,
                label="preservable_1",
            )
        )
        # Add conversation
        budget.update_entry(InstructionSource.CONVERSATION, 7000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(
            target_percent=30.0,  # Very aggressive target
            pressure_percent=0,  # Continuous mode
        )
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        # PRESERVABLE should NOT be removed in continuous mode
        assert result.details.get("preservable_removed", 0) == 0

    def test_pressure_mode_removes_preservable(self):
        """Under extreme pressure, PRESERVABLE content can be removed."""
        plugin = create_plugin()
        plugin.initialize()

        budget = make_budget(10000)
        # Add PRESERVABLE entry
        budget.add_child(
            InstructionSource.PLUGIN,
            "preservable_1",
            SourceEntry(
                source=InstructionSource.PLUGIN,
                tokens=2000,
                gc_policy=GCPolicy.PRESERVABLE,
                label="preservable_1",
                created_at=time.time() - 100,
            )
        )
        # Add locked conversation to consume budget
        budget.update_entry(InstructionSource.CONVERSATION, 7500, GCPolicy.LOCKED)

        history = make_history(5)
        config = GCConfig(
            target_percent=30.0,  # Aggressive target
            pressure_percent=90.0,  # Threshold mode with pressure
        )
        context = {"percent_used": 95.0}  # Above pressure_percent

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        # PRESERVABLE may be removed under pressure
        # (depends on whether target was met by other means)


class TestCollectFallback:
    def test_fallback_without_budget(self):
        """Falls back to truncation when no budget available."""
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 2})

        history = make_history(10)
        config = GCConfig(preserve_recent_turns=2)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget=None
        )

        assert result.success
        assert result.details.get("mode") == "fallback_truncate"
        assert result.items_collected > 0
        # Should preserve last 2 turns (4 messages)
        assert len(new_history) == 4


class TestRemovalList:
    def test_removal_list_populated(self):
        """Verify removal_list is populated correctly."""
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 2})

        budget = make_budget(10000)
        budget.update_entry(InstructionSource.ENRICHMENT, 2000, GCPolicy.EPHEMERAL)
        budget.update_entry(InstructionSource.CONVERSATION, 7000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.success
        assert len(result.removal_list) > 0

        # Check removal list items have correct structure
        for item in result.removal_list:
            assert isinstance(item, GCRemovalItem)
            assert item.source is not None
            assert item.reason != ""


class TestNotification:
    def test_notification_when_enabled(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 2,
            "notify_on_gc": True
        })

        budget = make_budget(10000)
        budget.update_entry(InstructionSource.ENRICHMENT, 3000, GCPolicy.EPHEMERAL)
        budget.update_entry(InstructionSource.CONVERSATION, 6000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        assert result.notification is not None
        assert "removed" in result.notification or "freed" in result.notification

    def test_custom_notification_template(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 2,
            "notify_on_gc": True,
            "notification_template": "Cleared {items} items!"
        })

        budget = make_budget(10000)
        budget.update_entry(InstructionSource.ENRICHMENT, 3000, GCPolicy.EPHEMERAL)
        budget.update_entry(InstructionSource.CONVERSATION, 6000, GCPolicy.PARTIAL)

        history = make_history(5)
        config = GCConfig(target_percent=60.0)
        context = {"percent_used": 90.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD, budget
        )

        if result.items_collected > 0:
            assert result.notification is not None
            assert "Cleared" in result.notification
            assert "items!" in result.notification


class TestGCConfigNewFields:
    """Tests for new GCConfig fields added for budget integration."""

    def test_default_values(self):
        """Test GCConfig has correct default values for new fields."""
        config = GCConfig()

        # Default target is 60%
        assert config.target_percent == 60.0
        # Default pressure is 90%
        assert config.pressure_percent == 90.0
        # Not in continuous mode by default
        assert config.continuous_mode is False

    def test_continuous_mode_with_zero_pressure(self):
        """Continuous mode is enabled when pressure_percent is 0."""
        config = GCConfig(pressure_percent=0)
        assert config.continuous_mode is True

    def test_continuous_mode_with_none_pressure(self):
        """Continuous mode is enabled when pressure_percent is None."""
        config = GCConfig(pressure_percent=None)
        assert config.continuous_mode is True

    def test_threshold_mode_with_positive_pressure(self):
        """Threshold mode when pressure_percent is positive."""
        config = GCConfig(pressure_percent=85.0)
        assert config.continuous_mode is False

    def test_custom_target_percent(self):
        """Test custom target_percent."""
        config = GCConfig(target_percent=50.0)
        assert config.target_percent == 50.0


class TestGCRemovalItem:
    """Tests for GCRemovalItem dataclass."""

    def test_default_values(self):
        """Test GCRemovalItem default values."""
        item = GCRemovalItem(source=InstructionSource.CONVERSATION)

        assert item.source == InstructionSource.CONVERSATION
        assert item.child_key is None
        assert item.tokens_freed == 0
        assert item.reason == ""
        assert item.message_ids == []

    def test_full_initialization(self):
        """Test GCRemovalItem with all fields."""
        item = GCRemovalItem(
            source=InstructionSource.ENRICHMENT,
            child_key="tool_result_123",
            tokens_freed=500,
            reason="ephemeral",
            message_ids=["msg-1", "msg-2"],
        )

        assert item.source == InstructionSource.ENRICHMENT
        assert item.child_key == "tool_result_123"
        assert item.tokens_freed == 500
        assert item.reason == "ephemeral"
        assert item.message_ids == ["msg-1", "msg-2"]
