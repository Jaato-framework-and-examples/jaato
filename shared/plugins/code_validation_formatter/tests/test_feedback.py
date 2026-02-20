"""Tests for code_validation_formatter get_turn_feedback()."""

import pytest

from ..plugin import CodeValidationFormatterPlugin, create_plugin


class TestGetTurnFeedback:
    """Tests for get_turn_feedback() — model self-correction loop."""

    def test_no_issues_returns_none(self):
        plugin = CodeValidationFormatterPlugin()
        assert plugin.get_turn_feedback() is None

    def test_syntax_error_produces_feedback(self):
        """Python syntax error should produce turn feedback."""
        plugin = CodeValidationFormatterPlugin()
        # Disable excerpt threshold so short blocks are still validated
        plugin.initialize({"enabled": True, "min_lines_for_validation": 0})

        # Process a code block with a syntax error
        code = "```python\ndef foo(\n```\n"
        list(plugin.process_chunk(code))
        list(plugin.flush())

        feedback = plugin.get_turn_feedback()
        assert feedback is not None
        assert "Code Validation Feedback" in feedback
        assert "error" in feedback.lower()

    def test_feedback_clears_accumulated_issues(self):
        """get_turn_feedback() should clear accumulated issues."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True, "min_lines_for_validation": 0})

        code = "```python\ndef foo(\n```\n"
        list(plugin.process_chunk(code))
        list(plugin.flush())

        assert plugin.get_turn_feedback() is not None
        assert plugin.get_turn_feedback() is None  # Cleared

    def test_valid_code_no_feedback(self):
        """Valid Python code should not produce feedback."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True, "min_lines_for_validation": 0})

        code = "```python\ndef foo():\n    return 42\n```\n"
        list(plugin.process_chunk(code))
        list(plugin.flush())

        assert plugin.get_turn_feedback() is None

    def test_non_python_passthrough(self):
        """Non-validatable languages should not produce feedback."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True})

        code = "```markdown\n# Hello\n```\n"
        list(plugin.process_chunk(code))
        list(plugin.flush())

        assert plugin.get_turn_feedback() is None

    def test_feedback_without_callback(self):
        """Feedback works without the old callback mechanism."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True, "min_lines_for_validation": 0})
        # No set_feedback_callback — should still work via get_turn_feedback

        code = "```python\ndef foo(\n```\n"
        list(plugin.process_chunk(code))
        list(plugin.flush())

        feedback = plugin.get_turn_feedback()
        assert feedback is not None
        assert "error" in feedback.lower()

    def test_reset_clears_issues_and_feedback(self):
        """reset() should clear accumulated issues."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True, "min_lines_for_validation": 0})

        code = "```python\ndef foo(\n```\n"
        list(plugin.process_chunk(code))
        list(plugin.flush())

        plugin.reset()

        assert plugin.get_turn_feedback() is None
