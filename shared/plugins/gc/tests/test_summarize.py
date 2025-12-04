"""Tests for SummarizeGCPlugin."""

import pytest
from google.genai import types

from shared.plugins.gc import GCConfig, GCTriggerReason
from shared.plugins.gc_summarize import SummarizeGCPlugin, create_plugin


def make_content(role: str, text: str) -> types.Content:
    """Helper to create Content objects."""
    return types.Content(
        role=role,
        parts=[types.Part(text=text)]
    )


def make_history(num_turns: int) -> list:
    """Create a history with N turns (user+model pairs)."""
    history = []
    for i in range(num_turns):
        history.append(make_content("user", f"User message {i}"))
        history.append(make_content("model", f"Model response {i}"))
    return history


def mock_summarizer(conversation: str) -> str:
    """Simple mock summarizer for testing."""
    return f"Summary of {len(conversation)} chars"


class TestSummarizeGCPlugin:
    def test_create_plugin(self):
        plugin = create_plugin()
        assert plugin.name == "gc_summarize"

    def test_initialize_with_summarizer(self):
        plugin = SummarizeGCPlugin()
        plugin.initialize({"summarizer": mock_summarizer})
        assert plugin._initialized
        assert plugin._summarizer is mock_summarizer


class TestCollectWithoutSummarizer:
    def test_fails_without_summarizer(self):
        plugin = create_plugin()
        plugin.initialize({"preserve_recent_turns": 2})

        history = make_history(10)
        config = GCConfig(preserve_recent_turns=2)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert not result.success
        assert result.error is not None
        assert "summarizer" in result.error.lower()
        assert new_history == history  # Unchanged


class TestCollectWithSummarizer:
    def test_summarizes_old_turns(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 2,
            "summarizer": mock_summarizer
        })

        history = make_history(10)
        config = GCConfig(preserve_recent_turns=2)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.success
        assert result.items_collected == 8  # Summarized 8 turns
        # New history: summary + 2 recent turns (4 content items)
        assert len(new_history) == 5  # 1 summary + 4 content items
        # First item should be the summary
        assert "[Context Summary" in new_history[0].parts[0].text

    def test_nothing_to_summarize(self):
        plugin = create_plugin()
        plugin.initialize({
            "preserve_recent_turns": 10,
            "summarizer": mock_summarizer
        })

        history = make_history(5)
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
            "preserve_recent_turns": 2,
            "summarizer": failing_summarizer
        })

        history = make_history(10)
        config = GCConfig(preserve_recent_turns=2)
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
            "preserve_recent_turns": 3,
            "summarizer": mock_summarizer
        })

        history = make_history(10)
        config = GCConfig(preserve_recent_turns=3)
        context = {"percent_used": 80.0}

        new_history, result = plugin.collect(
            history, context, config, GCTriggerReason.THRESHOLD
        )

        assert result.details["turns_before"] == 10
        assert result.details["turns_after"] == 4  # 1 summary + 3 preserved
        assert result.details["turns_summarized"] == 7
        assert result.details["preserve_count"] == 3
        assert result.plugin_name == "gc_summarize"


class TestFormatTurnsForSummary:
    def test_formats_text_content(self):
        plugin = create_plugin()
        plugin.initialize()

        from shared.plugins.gc.utils import split_into_turns
        history = [
            make_content("user", "Hello"),
            make_content("model", "Hi there!"),
        ]
        turns = split_into_turns(history)

        formatted = plugin._format_turns_for_summary(turns)

        assert "USER: Hello" in formatted
        assert "MODEL: Hi there!" in formatted

    def test_formats_function_calls(self):
        plugin = create_plugin()
        plugin.initialize()

        from shared.plugins.gc.utils import split_into_turns

        history = [
            make_content("user", "Call function"),
            types.Content(role="model", parts=[
                types.Part(function_call=types.FunctionCall(
                    name="test_func",
                    args={}
                ))
            ]),
        ]
        turns = split_into_turns(history)

        formatted = plugin._format_turns_for_summary(turns)

        assert "USER: Call function" in formatted
        assert "[Called test_func]" in formatted
