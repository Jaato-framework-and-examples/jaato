# shared/plugins/diff_formatter/tests/test_plugin.py
"""Tests for the diff formatter plugin with adaptive rendering."""

import pytest

from ..plugin import DiffFormatterPlugin, create_plugin
from ..renderers.base import ColorScheme, NO_COLOR_SCHEME


# Sample diff for testing
SAMPLE_DIFF = """--- a/test.py
+++ b/test.py
@@ -1,4 +1,5 @@
 def hello():
-    print('hello')
+    print('hello world')
+    print('goodbye')
     return True
"""

SAMPLE_DIFF_MULTIPLE_HUNKS = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    print('hello')
+    print('hi')
     return True
@@ -10,3 +10,4 @@
 def goodbye():
     print('bye')
+    print('farewell')
     return False
"""


class TestDiffFormatterPlugin:
    """Tests for the main plugin class."""

    def test_create_plugin(self):
        plugin = create_plugin()
        assert isinstance(plugin, DiffFormatterPlugin)
        assert plugin.name == "diff_formatter"

    def test_priority(self):
        plugin = create_plugin()
        assert plugin.priority == 20  # Structural formatting range

    def test_should_format_with_hint(self):
        plugin = create_plugin()
        assert plugin.should_format("any text", format_hint="diff")

    def test_should_format_detects_diff(self):
        plugin = create_plugin()
        assert plugin.should_format(SAMPLE_DIFF)

    def test_should_format_rejects_non_diff(self):
        plugin = create_plugin()
        assert not plugin.should_format("just some text")
        assert not plugin.should_format("def hello():\n    pass")

    def test_is_diff_alias(self):
        plugin = create_plugin()
        assert plugin.is_diff(SAMPLE_DIFF)
        assert not plugin.is_diff("not a diff")


class TestModeSelection:
    """Tests for automatic mode selection based on terminal width."""

    def test_wide_terminal_uses_side_by_side(self):
        plugin = create_plugin()
        plugin.set_console_width(140)
        assert plugin.get_current_mode() == "side_by_side"

    def test_medium_terminal_uses_compact(self):
        plugin = create_plugin()
        plugin.set_console_width(100)
        assert plugin.get_current_mode() == "compact"

    def test_narrow_terminal_uses_unified(self):
        plugin = create_plugin()
        plugin.set_console_width(60)
        assert plugin.get_current_mode() == "unified"

    def test_boundary_120_is_side_by_side(self):
        plugin = create_plugin()
        plugin.set_console_width(120)
        assert plugin.get_current_mode() == "side_by_side"

    def test_boundary_119_is_compact(self):
        plugin = create_plugin()
        plugin.set_console_width(119)
        assert plugin.get_current_mode() == "compact"

    def test_boundary_80_is_compact(self):
        plugin = create_plugin()
        plugin.set_console_width(80)
        assert plugin.get_current_mode() == "compact"

    def test_boundary_79_is_unified(self):
        plugin = create_plugin()
        plugin.set_console_width(79)
        assert plugin.get_current_mode() == "unified"


class TestSideBySideOutput:
    """Tests for side-by-side rendering mode."""

    def test_side_by_side_has_box_drawing(self):
        plugin = create_plugin()
        plugin.set_console_width(140)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        # Should contain box drawing characters
        assert "┌" in output
        assert "┐" in output
        assert "└" in output
        assert "┘" in output
        assert "│" in output

    def test_side_by_side_has_column_headers(self):
        plugin = create_plugin()
        plugin.set_console_width(140)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        assert "OLD" in output
        assert "NEW" in output

    def test_side_by_side_has_file_path(self):
        plugin = create_plugin()
        plugin.set_console_width(140)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        assert "test.py" in output

    def test_side_by_side_has_stats(self):
        plugin = create_plugin()
        plugin.set_console_width(140)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        # Should have line counts in stats
        assert "line" in output.lower()


class TestCompactOutput:
    """Tests for compact rendering mode."""

    def test_compact_has_box_drawing(self):
        plugin = create_plugin()
        plugin.set_console_width(100)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        # Should contain box drawing characters
        assert "┌" in output
        assert "└" in output
        assert "│" in output

    def test_compact_has_file_path(self):
        plugin = create_plugin()
        plugin.set_console_width(100)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        assert "test.py" in output

    def test_compact_uses_markers(self):
        plugin = create_plugin()
        plugin.set_console_width(100)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        # Should use +/- or arrow markers
        # At least one of these should be present
        has_markers = "+" in output or "-" in output or "→" in output
        assert has_markers


class TestUnifiedOutput:
    """Tests for unified (fallback) rendering mode."""

    def test_unified_has_hunk_header(self):
        plugin = create_plugin()
        plugin.set_console_width(60)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        assert "@@" in output

    def test_unified_has_plus_minus_markers(self):
        plugin = create_plugin()
        plugin.set_console_width(60)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        lines = output.split("\n")
        has_plus = any(l.startswith("+") for l in lines)
        has_minus = any(l.startswith("-") for l in lines)

        assert has_plus
        assert has_minus

    def test_unified_has_file_path(self):
        plugin = create_plugin()
        plugin.set_console_width(60)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        assert "test.py" in output


class TestColorConfiguration:
    """Tests for color scheme configuration."""

    def test_disable_colors(self):
        plugin = create_plugin()
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        # Should not contain ANSI escape codes
        assert "\033[" not in output

    def test_enable_colors(self):
        plugin = create_plugin()
        plugin.disable_colors()
        plugin.enable_colors()

        output = plugin.format_output(SAMPLE_DIFF)

        # Should contain ANSI escape codes
        assert "\033[" in output

    def test_custom_color_scheme(self):
        plugin = create_plugin()

        custom = ColorScheme(
            added="[GREEN]",
            deleted="[RED]",
            reset="[RESET]",
        )
        plugin.set_colors(custom)
        plugin.set_console_width(60)  # Use unified for simpler output

        output = plugin.format_output(SAMPLE_DIFF)

        assert "[GREEN]" in output or "[RED]" in output

    def test_initialize_with_no_colors(self):
        plugin = create_plugin()
        plugin.initialize({"colors": False})

        output = plugin.format_output(SAMPLE_DIFF)

        assert "\033[" not in output


class TestInitialization:
    """Tests for plugin initialization."""

    def test_initialize_with_width(self):
        plugin = create_plugin()
        plugin.initialize({"console_width": 80})

        assert plugin.get_current_mode() == "compact"

    def test_initialize_with_priority(self):
        plugin = create_plugin()
        plugin.initialize({"priority": 50})

        assert plugin.priority == 50


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_diff(self):
        plugin = create_plugin()
        plugin.disable_colors()

        output = plugin.format_output("")

        # Should return something, not crash
        assert output is not None

    def test_malformed_diff(self):
        plugin = create_plugin()
        plugin.disable_colors()

        # Not a valid diff
        output = plugin.format_output("not a diff at all")

        # Should fall back to raw colorization (which does nothing without colors)
        assert output is not None

    def test_diff_with_only_additions(self):
        diff = """--- /dev/null
+++ b/new.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
"""
        plugin = create_plugin()
        plugin.set_console_width(140)
        plugin.disable_colors()

        output = plugin.format_output(diff)

        assert "new.py" in output

    def test_diff_with_only_deletions(self):
        diff = """--- a/old.py
+++ /dev/null
@@ -1,3 +0,0 @@
-line1
-line2
-line3
"""
        plugin = create_plugin()
        plugin.set_console_width(140)
        plugin.disable_colors()

        output = plugin.format_output(diff)

        assert "old.py" in output

    def test_multiple_hunks(self):
        plugin = create_plugin()
        plugin.set_console_width(140)
        plugin.disable_colors()

        output = plugin.format_output(SAMPLE_DIFF_MULTIPLE_HUNKS)

        # Should contain content from both hunks
        assert "hello" in output or "hi" in output
        assert "goodbye" in output or "farewell" in output


class TestLegacyCompatibility:
    """Tests for backwards compatibility with old color properties."""

    def test_legacy_color_properties(self):
        plugin = create_plugin()

        # These should not raise
        _ = plugin._color_additions
        _ = plugin._color_deletions
        _ = plugin._color_hunks
        _ = plugin._color_headers

    def test_colorize_diff_alias(self):
        plugin = create_plugin()
        plugin.disable_colors()

        output1 = plugin.format_output(SAMPLE_DIFF)
        output2 = plugin.colorize_diff(SAMPLE_DIFF)

        assert output1 == output2
