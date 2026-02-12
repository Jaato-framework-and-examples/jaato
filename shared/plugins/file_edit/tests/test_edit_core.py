"""Tests for the edit_core module (targeted search-and-replace logic)."""

import pytest

from ..edit_core import apply_edit, EditNotFoundError, AmbiguousEditError


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

    def test_old_equals_new(self):
        """Replacing text with identical text should be a no-op."""
        content = "unchanged\n"
        result = apply_edit(content, "unchanged", "unchanged")
        assert result == "unchanged\n"


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
