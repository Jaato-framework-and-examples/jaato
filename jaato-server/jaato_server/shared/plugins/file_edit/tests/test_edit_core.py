"""Tests for the edit_core module (targeted search-and-replace logic)."""

import pytest

from ..edit_core import (
    apply_edit,
    EditNotFoundError,
    AmbiguousEditError,
    MalformedEditError,
)


class TestApplyEditBasic:
    """Basic targeted replacement tests."""

    def test_simple_replacement(self):
        content = "Hello, World!\nGoodbye, World!\n"
        result = apply_edit(content, "Hello", "Hi")
        assert result == "Hi, World!\nGoodbye, World!\n"

    def test_multiline_replacement(self):
        content = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        result = apply_edit(content, "def foo():\n    pass", "def foo():\n    return 42")
        assert result == "def foo():\n    return 42\n\ndef bar():\n    pass\n"

    def test_replace_with_empty_string(self):
        """Replacing with empty string effectively deletes the matched text."""
        content = "line1\nline2\nline3\n"
        result = apply_edit(content, "line2\n", "")
        assert result == "line1\nline3\n"

    def test_replace_at_start_of_file(self):
        content = "first line\nsecond line\n"
        result = apply_edit(content, "first line", "new first line")
        assert result == "new first line\nsecond line\n"

    def test_replace_at_end_of_file(self):
        content = "first line\nsecond line"
        result = apply_edit(content, "second line", "new second line")
        assert result == "first line\nnew second line"

    def test_replace_entire_content(self):
        content = "all of this"
        result = apply_edit(content, "all of this", "something else")
        assert result == "something else"


class TestApplyEditMalformed:
    """Degenerate requests are refused at the input boundary (#814).

    The point of each of these is the *branch*, not the exception: before
    the guard they fell through to match handling, where every remedy on
    offer addressed a different failure.
    """

    def test_empty_old_is_rejected(self):
        with pytest.raises(MalformedEditError, match="'old' is empty"):
            apply_edit("a\nb\nc\n", "", "X")

    def test_empty_old_is_not_reported_as_ambiguous(self):
        """The regression: 7 matches on a 6-byte file, unnarrowable by anchors."""
        with pytest.raises(MalformedEditError) as exc:
            apply_edit("a\nb\nc\n", "", "X")
        message = str(exc.value)
        assert "matched" not in message
        assert "7" not in message

    def test_empty_old_rejected_even_with_anchors(self):
        """No anchor can rescue it, so anchors must not change the verdict."""
        with pytest.raises(MalformedEditError, match="'old' is empty"):
            apply_edit("a\nb\nc\n", "", "X", prologue="a\n", epilogue="b\n")

    def test_empty_old_rejected_before_reading_content(self):
        """Guard is on the input, so an empty file reaches the same verdict."""
        with pytest.raises(MalformedEditError, match="'old' is empty"):
            apply_edit("", "", "X")

    def test_old_equals_new_is_rejected(self):
        """A no-op edit is a caller mistake; silently succeeding hides it."""
        with pytest.raises(MalformedEditError, match="identical"):
            apply_edit("unchanged\n", "unchanged", "unchanged")

    def test_old_equals_new_rejected_even_when_absent(self):
        """The request is malformed regardless of whether it would match."""
        with pytest.raises(MalformedEditError, match="identical"):
            apply_edit("other\n", "unchanged", "unchanged")

    def test_whitespace_only_old_is_still_allowed(self):
        """Indentation-only edits are legitimate when anchored.

        Deliberately NOT rejected: unlike an empty 'old', a whitespace-only
        'old' can match exactly once, and where it can't the ambiguity
        advice ("add anchors") is followable.
        """
        content = "def f():\n  x = 1\n"
        result = apply_edit(content, "  ", "    ", prologue="def f():\n", epilogue="x = 1")
        assert result == "def f():\n    x = 1\n"


class TestApplyEditNotFound:
    """Tests for EditNotFoundError."""

    def test_not_found_raises(self):
        content = "Hello, World!\n"
        with pytest.raises(EditNotFoundError):
            apply_edit(content, "nonexistent", "replacement")

    def test_not_found_message_contains_search_text(self):
        content = "Hello, World!\n"
        with pytest.raises(EditNotFoundError, match="not found"):
            apply_edit(content, "missing text", "replacement")

    def test_not_found_empty_file(self):
        with pytest.raises(EditNotFoundError):
            apply_edit("", "something", "replacement")


class TestApplyEditAmbiguous:
    """Tests for AmbiguousEditError."""

    def test_ambiguous_raises(self):
        content = "x = 1\nx = 1\n"
        with pytest.raises(AmbiguousEditError):
            apply_edit(content, "x = 1", "x = 2")

    def test_ambiguous_message_contains_count(self):
        content = "abc\nabc\nabc\n"
        with pytest.raises(AmbiguousEditError, match="3 times"):
            apply_edit(content, "abc", "xyz")


class TestApplyEditPrologue:
    """Tests for prologue-based disambiguation."""

    def test_prologue_disambiguates(self):
        content = "class A:\n    x = 1\n\nclass B:\n    x = 1\n"
        result = apply_edit(content, "x = 1", "x = 2", prologue="class A:\n    ")
        assert result == "class A:\n    x = 2\n\nclass B:\n    x = 1\n"

    def test_prologue_second_match(self):
        content = "class A:\n    x = 1\n\nclass B:\n    x = 1\n"
        result = apply_edit(content, "x = 1", "x = 2", prologue="class B:\n    ")
        assert result == "class A:\n    x = 1\n\nclass B:\n    x = 2\n"

    def test_prologue_not_found(self):
        content = "class A:\n    x = 1\n"
        with pytest.raises(EditNotFoundError, match="context anchors"):
            apply_edit(content, "x = 1", "x = 2", prologue="class Z:\n    ")


class TestApplyEditEpilogue:
    """Tests for epilogue-based disambiguation."""

    def test_epilogue_disambiguates(self):
        content = "x = 1\ny = 'a'\n\nx = 1\ny = 'b'\n"
        result = apply_edit(content, "x = 1", "x = 2", epilogue="\ny = 'b'")
        assert result == "x = 1\ny = 'a'\n\nx = 2\ny = 'b'\n"

    def test_epilogue_not_found(self):
        content = "x = 1\ny = 'a'\n"
        with pytest.raises(EditNotFoundError, match="context anchors"):
            apply_edit(content, "x = 1", "x = 2", epilogue="\ny = 'z'")


class TestApplyEditPrologueAndEpilogue:
    """Tests for combined prologue + epilogue disambiguation."""

    def test_both_anchors(self):
        content = (
            "if a:\n    x = 1\n    print('a')\n"
            "if b:\n    x = 1\n    print('b')\n"
            "if c:\n    x = 1\n    print('c')\n"
        )
        result = apply_edit(
            content, "x = 1", "x = 99",
            prologue="if b:\n    ", epilogue="\n    print('b')"
        )
        assert "if a:\n    x = 1" in result
        assert "if b:\n    x = 99" in result
        assert "if c:\n    x = 1" in result

    def test_both_anchors_still_ambiguous(self):
        """If even prologue+epilogue don't narrow to one match, should raise."""
        content = "ctx:\n    x = 1\n    end\nctx:\n    x = 1\n    end\n"
        with pytest.raises(AmbiguousEditError, match="2 times"):
            apply_edit(content, "x = 1", "x = 2", prologue="ctx:\n    ", epilogue="\n    end")


class TestAnchorMismatchDiagnostic:
    """The not-found message must name what went wrong, not only what to supply.

    ``prologue``/``epilogue`` are concatenated with ``old``, so they have to
    be byte-adjacent to it.  A caller that supplied distant *landmarks*
    instead used to get the three pieces echoed back with no hint that
    adjacency was the broken contract, and would refine the same wrong
    thing indefinitely (#813).
    """

    CONTENT = (
        '"""Retry utilities."""\n'      # line 1
        "\n"                            # line 2
        "import os\n"                   # line 3
        "import time\n"                 # line 4
        "\n"                            # line 5
        "# Type alias\n"                # line 6
        "Callback = None\n"             # line 7
    )

    def test_distant_prologue_names_adjacency_and_locations(self):
        """The issue's own reproduction: a docstring line used as a landmark."""
        with pytest.raises(EditNotFoundError) as exc:
            apply_edit(
                self.CONTENT,
                "import os\nimport time",
                "import os",
                prologue='"""Retry utilities."""',
                epilogue="# Type alias",
            )
        message = str(exc.value)
        # Says where 'old' actually is ...
        assert "'old' occurs once, at line 3" in message
        # ... that adjacency is the contract ...
        assert "IMMEDIATELY adjacent" in message
        # ... and where each anchor really sits.
        assert "'prologue' occurs once, at line 1" in message
        assert "'epilogue' occurs once, at line 6" in message

    def test_absent_old_says_the_anchors_are_not_the_problem(self):
        """When 'old' isn't there at all, refining anchors is wasted effort."""
        with pytest.raises(EditNotFoundError) as exc:
            apply_edit(
                self.CONTENT, "import sys", "import os",
                prologue="import os\n",
            )
        message = str(exc.value)
        assert "does not occur in the file on its own" in message
        assert "anchors are not the problem" in message

    def test_anchor_absent_from_file_is_called_out_separately(self):
        """A non-verbatim anchor is a different fix from a non-adjacent one."""
        with pytest.raises(EditNotFoundError) as exc:
            apply_edit(
                self.CONTENT, "import os", "import io",
                prologue="import typing\n",
            )
        message = str(exc.value)
        assert "'prologue' does not occur in the file at all" in message
        assert "verbatim" in message

    def test_only_the_failing_side_is_blamed(self):
        """An adjacent prologue with a distant epilogue blames the epilogue."""
        with pytest.raises(EditNotFoundError) as exc:
            apply_edit(
                self.CONTENT, "import time", "import perf",
                prologue="import os\n",
                epilogue="Callback = None",
            )
        message = str(exc.value)
        assert "'epilogue' occurs once, at line 7" in message
        assert "'prologue' occurs" not in message
        assert "'prologue' does not occur" not in message

    def test_anchors_on_different_occurrences_of_old(self):
        """Both sides adjacent, but to different matches — say exactly that."""
        content = "A\nx\nB\nC\nx\nD\n"
        with pytest.raises(EditNotFoundError) as exc:
            apply_edit(content, "x", "y", prologue="A\n", epilogue="\nD")
        message = str(exc.value)
        assert "each adjacent to a different" in message

    def test_repeated_old_reports_several_line_numbers(self):
        content = "x = 1\nfoo\nx = 1\nbar\n"
        with pytest.raises(EditNotFoundError) as exc:
            apply_edit(content, "x = 1", "x = 2", prologue="baz\n")
        assert "'old' occurs 2 times, at lines 1, 3" in str(exc.value)

    def test_occurrence_list_is_capped(self):
        content = "x\n" * 10
        with pytest.raises(EditNotFoundError) as exc:
            apply_edit(content, "x", "y", prologue="zzz")
        message = str(exc.value)
        assert "'old' occurs 10 times, at lines 1, 2, 3 and 7 more" in message

    def test_pieces_are_still_echoed_for_comparison(self):
        """The verbatim dump stays — the diagnosis is added to it, not swapped in."""
        with pytest.raises(EditNotFoundError) as exc:
            apply_edit(self.CONTENT, "import os", "import io", prologue="nope")
        message = str(exc.value)
        assert "prologue: 'nope'" in message
        assert "old:      'import os'" in message

    def test_unanchored_not_found_message_is_unchanged(self):
        """No anchors in play means there is nothing extra to diagnose."""
        with pytest.raises(EditNotFoundError) as exc:
            apply_edit(self.CONTENT, "import sys", "import os")
        assert str(exc.value) == "Search text not found: 'import sys'"
