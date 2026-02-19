"""Tests for HybridGCPlugin."""

import pytest

from shared.plugins.gc import GCConfig, GCTriggerReason
from shared.plugins.gc_hybrid import HybridGCPlugin, create_plugin
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


def mock_summarizer(conversation: str) -> str:
    """Simple mock summarizer for testing."""
    return f"Summary of {len(conversation)} chars"


class TestHybridGCPlugin:
    def test_create_plugin(self):
        plugin = create_plugin()
        assert plugin.name == "gc_hybrid"

    def test_initialize(self):
        plugin = HybridGCPlugin()
        plugin.initialize({
            "preserve_recent_turns": 5,
            "summarize_middle_turns": 10,
            "summarizer": mock_summarizer
        })
        assert plugin._initialized
        assert plugin._summarizer is mock_summarizer


class TestCollectWithoutSummarizer:
    def test_truncates_without_summarizer(self):
        """Without summarizer, hybrid behaves like truncate."""
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 3,
            "summarize_middle_turns": 5
        })

        history = make_history(20)
        config = GCConfig(preserve_recent_turns=3)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.success
        assert result.details["turns_truncated"] == 17  # All non-recent truncated
        assert result.details["turns_summarized"] == 0
        assert result.details["had_summarizer"] is False
        # Only recent turns remain
        assert len(new_history) == 6  # 3 turns * 2 content each


class TestCollectWithSummarizer:
    def test_summarizes_old_and_middle(self):
        """With summarizer, old+middle turns get summarized."""
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 3,
            "summarize_middle_turns": 5,
            "summarizer": mock_summarizer
        })

        history = make_history(20)  # 20 turns
        config = GCConfig(preserve_recent_turns=3)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.success
        # All 17 non-recent turns should be summarized
        assert result.details["turns_summarized"] == 17
        assert result.details["turns_truncated"] == 0
        assert result.details["had_summarizer"] is True
        # Summary + recent turns
        assert "[Context Summary" in new_history[0].parts[0].text

    def test_nothing_to_collect(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 10,
            "summarizer": mock_summarizer
        })

        history = make_history(5)  # Only 5 turns
        config = GCConfig(preserve_recent_turns=10)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.success
        assert result.items_collected == 0
        assert new_history == history

    def test_summarizer_exception_handled(self):
        def failing_summarizer(text: str) -> str:
            raise RuntimeError("Summarization failed!")

        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 3,
            "summarize_middle_turns": 5,
            "summarizer": failing_summarizer
        })

        history = make_history(20)
        config = GCConfig(preserve_recent_turns=3)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert not result.success
        assert "Summarization failed" in result.error
        assert new_history == history

    def test_result_details(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 5,
            "summarize_middle_turns": 10,
            "summarizer": mock_summarizer
        })

        history = make_history(30)
        config = GCConfig(preserve_recent_turns=5)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.details["turns_before"] == 30
        assert result.details["preserve_recent"] == 5
        assert result.details["summarize_middle"] == 10
        assert result.plugin_name == "gc_hybrid"


class TestGenerationalLayout:
    def test_generational_boundaries(self):
        """Test that turns are correctly divided into generations."""
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 5,
            "summarize_middle_turns": 10,
            "summarizer": mock_summarizer
        })

        # 30 turns total:
        # - Recent (young): turns 25-29 (5 turns)
        # - Middle (old): turns 15-24 (10 turns)
        # - Ancient: turns 0-14 (15 turns)
        history = make_history(30)
        config = GCConfig(preserve_recent_turns=5)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.success
        # All ancient + middle = 25 turns summarized
        assert result.details["turns_summarized"] == 25
        # Result: summary + 5 recent turns (10 content items)
        assert len(new_history) == 11  # 1 summary + 10 content

    def test_all_middle_no_ancient(self):
        """Test when there are middle turns but no ancient."""
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 5,
            "summarize_middle_turns": 10,
            "summarizer": mock_summarizer
        })

        # 12 turns:
        # - Recent: turns 7-11 (5 turns)
        # - Middle: turns 0-6 (7 turns, less than summarize_middle_turns)
        # - Ancient: none
        history = make_history(12)
        config = GCConfig(preserve_recent_turns=5)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.success
        assert result.details["turns_summarized"] == 7


class TestNotification:
    def test_notification_when_enabled(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 3,
            "summarize_middle_turns": 5,
            "summarizer": mock_summarizer,
            "notify_on_gc": True
        })

        history = make_history(20)
        config = GCConfig(preserve_recent_turns=3)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.notification is not None
        assert "summarized" in result.notification

    def test_custom_notification_template(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 3,
            "summarize_middle_turns": 5,
            "notify_on_gc": True,
            "notification_template": "Collected: {truncated}T, {summarized}S, {kept}K"
        })

        history = make_history(20)
        config = GCConfig(preserve_recent_turns=3)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert "17T" in result.notification or "0T" in result.notification
