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
