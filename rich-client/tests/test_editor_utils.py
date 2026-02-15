"""Tests for the diff format in editor_utils.

Covers unified-diff generation, parsing, round-trip, and the
retry-on-error loop inside ``edit_tool_content``.
"""

import textwrap

import pytest

from rich_client.editor_utils import (
    DiffParseError,
    _format_as_unified_diff,
    _looks_like_unified_diff,
    _parse_unified_diff,
    format_for_editing,
    parse_edited_content,
)


# ---------------------------------------------------------------------------
# _format_as_unified_diff
# ---------------------------------------------------------------------------

class TestFormatAsUnifiedDiff:
    """Tests for unified-diff generation from tool arguments."""

    def test_simple_replacement(self):
        args = {"path": "app.py", "old": "pass\n", "new": "return 42\n"}
        diff = _format_as_unified_diff(args)

        assert "--- a/app.py" in diff
        assert "+++ b/app.py" in diff
        assert "-pass" in diff
        assert "+return 42" in diff

    def test_multiline_replacement(self):
        args = {
            "path": "app.py",
            "old": "def foo():\n    pass\n",
            "new": "def foo():\n    return 42\n",
        }
        diff = _format_as_unified_diff(args)

        assert "-    pass" in diff
        assert "+    return 42" in diff
        # "def foo():" is common to both → context line
        assert " def foo():" in diff

    def test_prologue_becomes_context(self):
        args = {
            "path": "app.py",
            "old": "    pass\n",
            "new": "    return 42\n",
            "prologue": "class Foo:\n",
        }
        diff = _format_as_unified_diff(args)

        # Prologue should appear as a context line (space-prefixed)
        assert " class Foo:" in diff
        assert "-    pass" in diff
        assert "+    return 42" in diff

    def test_epilogue_becomes_context(self):
        args = {
            "path": "app.py",
            "old": "    pass\n",
            "new": "    return 42\n",
            "epilogue": "\nclass Bar:\n",
        }
        diff = _format_as_unified_diff(args)

        assert " class Bar:" in diff
        assert "-    pass" in diff

    def test_both_prologue_and_epilogue(self):
        args = {
            "path": "app.py",
            "old": "    x = 1\n",
            "new": "    x = 2\n",
            "prologue": "def setup():\n",
            "epilogue": "    return x\n",
        }
        diff = _format_as_unified_diff(args)

        assert " def setup():" in diff
        assert " " + "    return x" in diff or "     return x" in diff
        assert "-    x = 1" in diff
        assert "+    x = 2" in diff

    def test_pure_insertion(self):
        args = {"path": "app.py", "old": "", "new": "import os\n"}
        diff = _format_as_unified_diff(args)

        assert "+import os" in diff

    def test_pure_deletion(self):
        args = {"path": "app.py", "old": "import os\n", "new": ""}
        diff = _format_as_unified_diff(args)

        assert "-import os" in diff

    def test_no_trailing_newline(self):
        """Content without trailing newline should still produce valid diff."""
        args = {"path": "f.txt", "old": "hello", "new": "world"}
        diff = _format_as_unified_diff(args)

        assert "-hello" in diff
        assert "+world" in diff


# ---------------------------------------------------------------------------
# _looks_like_unified_diff / _parse_unified_diff
# ---------------------------------------------------------------------------

class TestLooksLikeUnifiedDiff:
    def test_real_diff(self):
        diff = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n"
        assert _looks_like_unified_diff(diff) is True

    def test_plain_text(self):
        assert _looks_like_unified_diff("just some text\n") is False

    def test_empty(self):
        assert _looks_like_unified_diff("") is False


class TestParseUnifiedDiff:
    """Tests for parsing a unified diff back into old/new text."""

    def test_simple_diff(self):
        diff = textwrap.dedent("""\
            --- a/f.py
            +++ b/f.py
            @@ -1,2 +1,2 @@
             context
            -old line
            +new line
        """)
        result = _parse_unified_diff(diff)
        assert result["old"] == "old line"
        assert result["new"] == "new line"

    def test_multiline_diff(self):
        diff = textwrap.dedent("""\
            --- a/f.py
            +++ b/f.py
            @@ -1,4 +1,4 @@
             prologue
            -old line 1
            -old line 2
            +new line 1
            +new line 2
             epilogue
        """)
        result = _parse_unified_diff(diff)
        assert result["old"] == "old line 1\nold line 2"
        assert result["new"] == "new line 1\nnew line 2"

    def test_pure_insertion(self):
        diff = textwrap.dedent("""\
            --- a/f.py
            +++ b/f.py
            @@ -0,0 +1,2 @@
            +line 1
            +line 2
        """)
        result = _parse_unified_diff(diff)
        assert result["old"] == ""
        assert result["new"] == "line 1\nline 2"

    def test_pure_deletion(self):
        diff = textwrap.dedent("""\
            --- a/f.py
            +++ b/f.py
            @@ -1,2 +0,0 @@
            -line 1
            -line 2
        """)
        result = _parse_unified_diff(diff)
        assert result["old"] == "line 1\nline 2"
        assert result["new"] == ""

    def test_no_hunk_raises(self):
        with pytest.raises(DiffParseError, match="No diff hunks found"):
            _parse_unified_diff("just text\n")

    def test_no_newline_marker_skipped(self):
        diff = textwrap.dedent("""\
            --- a/f
            +++ b/f
            @@ -1 +1 @@
            -hello
            \\ No newline at end of file
            +world
            \\ No newline at end of file
        """)
        result = _parse_unified_diff(diff)
        assert result["old"] == "hello"
        assert result["new"] == "world"

    def test_context_lines_ignored(self):
        diff = textwrap.dedent("""\
            --- a/f
            +++ b/f
            @@ -1,5 +1,5 @@
             line1
             line2
            -old
            +new
             line4
        """)
        result = _parse_unified_diff(diff)
        assert result["old"] == "old"
        assert result["new"] == "new"


# ---------------------------------------------------------------------------
# Round-trip: format → parse
# ---------------------------------------------------------------------------

class TestDiffFormatRoundTrip:
    """Verify that formatting as diff then parsing back recovers old/new."""

    def test_simple_round_trip(self):
        args = {"path": "f.py", "old": "pass", "new": "return 1"}
        diff = _format_as_unified_diff(args)
        result = _parse_unified_diff(diff)
        assert result["old"] == "pass"
        assert result["new"] == "return 1"

    def test_multiline_round_trip(self):
        old = "def foo():\n    pass"
        new = "def foo():\n    return 42"
        args = {"path": "f.py", "old": old, "new": new}
        diff = _format_as_unified_diff(args)
        result = _parse_unified_diff(diff)
        assert result["old"] == "    pass"
        assert result["new"] == "    return 42"
        # NOTE: "def foo():" is common to both → becomes context, not in old/new

    def test_round_trip_with_prologue_epilogue(self):
        """Prologue/epilogue become context lines and are not in parsed old/new."""
        args = {
            "path": "f.py",
            "old": "    x = 1\n",
            "new": "    x = 2\n",
            "prologue": "def setup():\n",
            "epilogue": "    return x\n",
        }
        diff = _format_as_unified_diff(args)
        result = _parse_unified_diff(diff)
        assert result["old"] == "    x = 1"
        assert result["new"] == "    x = 2"

    def test_round_trip_pure_insertion(self):
        args = {"path": "f.py", "old": "", "new": "import os"}
        diff = _format_as_unified_diff(args)
        result = _parse_unified_diff(diff)
        assert result["old"] == ""
        assert result["new"] == "import os"


# ---------------------------------------------------------------------------
# format_for_editing / parse_edited_content with format="diff"
# ---------------------------------------------------------------------------

class TestFormatForEditingDiff:
    """Tests for the top-level format_for_editing with format='diff'."""

    def test_targeted_edit_produces_diff(self):
        args = {"path": "f.py", "old": "pass\n", "new": "return 1\n"}
        content = format_for_editing(args, ["old", "new", "new_content"], "diff")
        assert "@@" in content
        assert "-pass" in content
        assert "+return 1" in content

    def test_full_replacement_produces_text(self):
        args = {"path": "f.py", "new_content": "full file content\n"}
        content = format_for_editing(args, ["old", "new", "new_content"], "diff")
        # No diff markers — just the raw text
        assert "@@" not in content
        assert "full file content" in content

    def test_template_prepended_for_full_replacement(self):
        args = {"path": "f.py", "new_content": "content"}
        content = format_for_editing(
            args, ["old", "new", "new_content"], "diff",
            template="# Header\n",
        )
        assert content.startswith("# Header\n")


class TestParseEditedContentDiff:
    """Tests for parse_edited_content with format='diff'."""

    def test_parse_diff_content(self):
        diff = textwrap.dedent("""\
            --- a/f.py
            +++ b/f.py
            @@ -1 +1 @@
            -old
            +new
        """)
        parsed, error = parse_edited_content(
            diff, ["old", "new", "new_content"], "diff"
        )
        assert error is None
        assert parsed["old"] == "old"
        assert parsed["new"] == "new"

    def test_parse_plain_text_fallback(self):
        """When diff format receives plain text, falls back to new_content."""
        parsed, error = parse_edited_content(
            "plain file content", ["old", "new", "new_content"], "diff"
        )
        assert error is None
        assert parsed["new_content"] == "plain file content"

    def test_parse_plain_text_fallback_first_param(self):
        """Falls back to first parameter if new_content not in list."""
        parsed, error = parse_edited_content(
            "some text", ["content"], "diff"
        )
        assert error is None
        assert parsed["content"] == "some text"
