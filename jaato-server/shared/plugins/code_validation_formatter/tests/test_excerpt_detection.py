"""Tests for code excerpt detection (line-threshold skipping).

Short code blocks are treated as excerpts/snippets and skip validation,
since incomplete code almost always produces false-positive diagnostics
(undefined variables, missing imports, etc.).
"""

import pytest

from ..plugin import (
    CodeValidationFormatterPlugin,
    DEFAULT_MIN_LINES_FOR_VALIDATION,
)


def _make_code_block(language: str, lines: list[str]) -> str:
    """Build a fenced code block string from a list of code lines."""
    body = "\n".join(lines)
    return f"```{language}\n{body}\n```\n"


def _make_broken_python(n_lines: int) -> list[str]:
    """Generate n_lines of Python with a syntax error on the last line.

    Returns a list like:
        ['x = 1', 'y = 2', ..., 'def broken(']
    """
    filler = [f"x{i} = {i}" for i in range(n_lines - 1)]
    filler.append("def broken(")  # SyntaxError: incomplete function def
    return filler


class TestExcerptDetection:
    """Excerpt detection skips validation for short code blocks."""

    def test_short_block_skipped_by_default(self):
        """A code block shorter than the default threshold should not be validated."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True})

        # 3-line block — well below the default threshold
        code = _make_code_block("python", ["def foo(", "    pass", ")"])
        list(plugin.process_chunk(code))
        list(plugin.flush())

        assert plugin.get_turn_feedback() is None
        assert plugin.get_accumulated_issues() == []

    def test_long_block_validated(self):
        """A code block at or above the threshold should be validated."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True})

        lines = _make_broken_python(DEFAULT_MIN_LINES_FOR_VALIDATION)
        code = _make_code_block("python", lines)
        list(plugin.process_chunk(code))
        list(plugin.flush())

        feedback = plugin.get_turn_feedback()
        assert feedback is not None
        assert "error" in feedback.lower()

    def test_block_just_below_threshold_skipped(self):
        """A block one line below the threshold should be skipped."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True})

        lines = _make_broken_python(DEFAULT_MIN_LINES_FOR_VALIDATION - 1)
        code = _make_code_block("python", lines)
        list(plugin.process_chunk(code))
        list(plugin.flush())

        assert plugin.get_turn_feedback() is None

    def test_block_at_threshold_validated(self):
        """A block exactly at the threshold should be validated."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True})

        lines = _make_broken_python(DEFAULT_MIN_LINES_FOR_VALIDATION)
        code = _make_code_block("python", lines)
        list(plugin.process_chunk(code))
        list(plugin.flush())

        assert plugin.get_turn_feedback() is not None

    def test_custom_threshold_via_config(self):
        """min_lines_for_validation config overrides the default."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True, "min_lines_for_validation": 5})

        # 4 lines — below custom threshold
        lines_short = _make_broken_python(4)
        code_short = _make_code_block("python", lines_short)
        list(plugin.process_chunk(code_short))
        list(plugin.flush())
        assert plugin.get_turn_feedback() is None

        plugin.reset()

        # 5 lines — at custom threshold
        lines_at = _make_broken_python(5)
        code_at = _make_code_block("python", lines_at)
        list(plugin.process_chunk(code_at))
        list(plugin.flush())
        assert plugin.get_turn_feedback() is not None

    def test_threshold_zero_validates_all(self):
        """Setting threshold to 0 disables excerpt detection (validates everything)."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True, "min_lines_for_validation": 0})

        # Even a 1-line broken block should be validated
        code = _make_code_block("python", ["def broken("])
        list(plugin.process_chunk(code))
        list(plugin.flush())

        assert plugin.get_turn_feedback() is not None

    def test_output_unchanged_for_skipped_block(self):
        """Skipped excerpt blocks should pass through unmodified."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True})

        code = _make_code_block("python", ["def broken("])
        chunks = list(plugin.process_chunk(code))
        chunks.extend(plugin.flush())
        output = "".join(chunks)

        # Output should be the original code block, no diagnostics appended
        assert "Code Validation" not in output
        assert output == code

    def test_diagnostics_appended_for_long_block(self):
        """Long blocks with errors should have diagnostics appended to output."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True})

        lines = _make_broken_python(DEFAULT_MIN_LINES_FOR_VALIDATION)
        code = _make_code_block("python", lines)
        chunks = list(plugin.process_chunk(code))
        chunks.extend(plugin.flush())
        output = "".join(chunks)

        assert "Code Validation" in output

    def test_unsupported_language_skipped_regardless(self):
        """Unsupported languages are skipped regardless of line count."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True, "min_lines_for_validation": 0})

        lines = [f"line {i}" for i in range(20)]
        code = _make_code_block("markdown", lines)
        list(plugin.process_chunk(code))
        list(plugin.flush())

        assert plugin.get_turn_feedback() is None

    def test_valid_long_block_no_feedback(self):
        """A long block with valid code should produce no feedback."""
        plugin = CodeValidationFormatterPlugin()
        plugin.initialize({"enabled": True})

        lines = [f"x{i} = {i}" for i in range(DEFAULT_MIN_LINES_FOR_VALIDATION)]
        code = _make_code_block("python", lines)
        list(plugin.process_chunk(code))
        list(plugin.flush())

        assert plugin.get_turn_feedback() is None
