# shared/plugins/formatter_pipeline/tests/test_pipeline.py
"""Tests for FormatterPipeline.get_system_instructions()."""

import pytest

from ..pipeline import FormatterPipeline


class MockFormatter:
    """Mock formatter without system instructions."""

    def __init__(self, name="mock", priority=50):
        self._name = name
        self._priority = priority

    @property
    def name(self):
        return self._name

    @property
    def priority(self):
        return self._priority

    def process_chunk(self, chunk):
        yield chunk

    def flush(self):
        return iter([])

    def reset(self):
        pass


class MockFormatterWithInstructions(MockFormatter):
    """Mock formatter that provides system instructions."""

    def __init__(self, name="instructing", priority=50, instructions=None):
        super().__init__(name, priority)
        self._instructions = instructions

    def get_system_instructions(self):
        return self._instructions


class TestPipelineGetSystemInstructions:
    """Tests for FormatterPipeline.get_system_instructions()."""

    def test_no_formatters_returns_none(self):
        pipeline = FormatterPipeline()
        assert pipeline.get_system_instructions() is None

    def test_formatters_without_method_returns_none(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatter("a", 10))
        pipeline.register(MockFormatter("b", 20))
        assert pipeline.get_system_instructions() is None

    def test_formatter_returning_none_is_skipped(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithInstructions("a", 10, instructions=None))
        assert pipeline.get_system_instructions() is None

    def test_formatter_returning_empty_is_skipped(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithInstructions("a", 10, instructions=""))
        assert pipeline.get_system_instructions() is None

    def test_single_formatter_with_instructions(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithInstructions(
            "mermaid", 28, instructions="Use mermaid diagrams."
        ))
        result = pipeline.get_system_instructions()
        assert result == "Use mermaid diagrams."

    def test_multiple_formatters_combined(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithInstructions(
            "fmt_a", 10, instructions="Instruction A."
        ))
        pipeline.register(MockFormatter("plain", 20))
        pipeline.register(MockFormatterWithInstructions(
            "fmt_b", 30, instructions="Instruction B."
        ))
        result = pipeline.get_system_instructions()
        assert "Instruction A." in result
        assert "Instruction B." in result
        assert result == "Instruction A.\n\nInstruction B."

    def test_mixed_formatters_only_contributing_included(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatter("no_instr", 10))
        pipeline.register(MockFormatterWithInstructions(
            "has_instr", 20, instructions="I contribute."
        ))
        pipeline.register(MockFormatterWithInstructions(
            "none_instr", 30, instructions=None
        ))
        result = pipeline.get_system_instructions()
        assert result == "I contribute."


class MockFormatterWithFeedback(MockFormatter):
    """Mock formatter that provides turn feedback."""

    def __init__(self, name="feedback", priority=50, feedback=None):
        super().__init__(name, priority)
        self._feedback = feedback

    def get_turn_feedback(self):
        fb = self._feedback
        self._feedback = None
        return fb


class TestPipelineCollectTurnFeedback:
    """Tests for FormatterPipeline.collect_turn_feedback()."""

    def test_no_formatters_returns_none(self):
        pipeline = FormatterPipeline()
        assert pipeline.collect_turn_feedback() is None

    def test_formatters_without_method_returns_none(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatter("a", 10))
        pipeline.register(MockFormatter("b", 20))
        assert pipeline.collect_turn_feedback() is None

    def test_formatter_returning_none_is_skipped(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithFeedback("a", 10, feedback=None))
        assert pipeline.collect_turn_feedback() is None

    def test_formatter_returning_empty_is_skipped(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithFeedback("a", 10, feedback=""))
        assert pipeline.collect_turn_feedback() is None

    def test_single_formatter_with_feedback(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithFeedback(
            "mermaid", 28, feedback="Syntax error in diagram."
        ))
        result = pipeline.collect_turn_feedback()
        assert result == "Syntax error in diagram."

    def test_multiple_formatters_combined(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithFeedback(
            "fmt_a", 10, feedback="Feedback A."
        ))
        pipeline.register(MockFormatter("plain", 20))
        pipeline.register(MockFormatterWithFeedback(
            "fmt_b", 30, feedback="Feedback B."
        ))
        result = pipeline.collect_turn_feedback()
        assert "Feedback A." in result
        assert "Feedback B." in result
        assert result == "Feedback A.\n\nFeedback B."

    def test_collect_stores_in_pending(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithFeedback(
            "fmt", 10, feedback="Some feedback."
        ))
        pipeline.collect_turn_feedback()
        assert pipeline.get_pending_feedback() == "Some feedback."

    def test_get_pending_clears_feedback(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithFeedback(
            "fmt", 10, feedback="Some feedback."
        ))
        pipeline.collect_turn_feedback()
        pipeline.get_pending_feedback()
        assert pipeline.get_pending_feedback() is None

    def test_no_feedback_no_pending(self):
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatter("plain", 10))
        pipeline.collect_turn_feedback()
        assert pipeline.get_pending_feedback() is None

    def test_one_shot_feedback_drain(self):
        """get_turn_feedback() is one-shot — second collect returns None."""
        pipeline = FormatterPipeline()
        pipeline.register(MockFormatterWithFeedback(
            "fmt", 10, feedback="Once only."
        ))
        first = pipeline.collect_turn_feedback()
        assert first == "Once only."
        second = pipeline.collect_turn_feedback()
        assert second is None
