"""Tests for FuzzyMatcher and fuzzy completion in completers."""

import os
import tempfile

from rich_client.file_completer import FuzzyMatcher, AtFileCompleter, PercentPromptCompleter
from prompt_toolkit.document import Document


class TestFuzzyMatcher:
    """Tests for FuzzyMatcher utility class."""

    def test_empty_pattern_matches_everything(self):
        """Empty pattern should match everything with score 0."""
        matches, score = FuzzyMatcher.match("", "anything")
        assert matches is True
        assert score == 0

    def test_exact_match(self):
        """Exact match should return True with high score."""
        matches, score = FuzzyMatcher.match("utils", "utils")
        assert matches is True
        assert score > 0  # Should have consecutive bonuses

    def test_prefix_match(self):
        """Prefix match should return True."""
        matches, score = FuzzyMatcher.match("util", "utils")
        assert matches is True

    def test_fuzzy_match_non_consecutive(self):
        """Non-consecutive characters should still match."""
        matches, score = FuzzyMatcher.match("utl", "utils")
        assert matches is True

    def test_no_match(self):
        """Pattern that doesn't match should return False."""
        matches, score = FuzzyMatcher.match("xyz", "utils")
        assert matches is False
        assert score == 0

    def test_out_of_order_no_match(self):
        """Characters out of order should not match."""
        matches, score = FuzzyMatcher.match("ltu", "utils")
        assert matches is False

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        matches, score = FuzzyMatcher.match("UTL", "utils")
        assert matches is True

        matches2, score2 = FuzzyMatcher.match("utl", "UTILS")
        assert matches2 is True

    def test_start_bonus(self):
        """Match at start should get bonus."""
        matches1, score1 = FuzzyMatcher.match("u", "utils")
        matches2, score2 = FuzzyMatcher.match("t", "utils")

        assert matches1 is True
        assert matches2 is True
        assert score1 > score2  # 'u' at start gets bonus

    def test_boundary_bonus(self):
        """Match after word boundary should get bonus."""
        # 'r' after '-' in code-review gets boundary bonus
        matches1, score1 = FuzzyMatcher.match("cr", "code-review")
        # Both 'c' and 'r' match at boundaries
        assert matches1 is True
        assert score1 > 0

    def test_consecutive_bonus(self):
        """Consecutive matches should score higher."""
        # "ut" consecutive in "utils"
        matches1, score1 = FuzzyMatcher.match("ut", "utils")
        # "ul" not consecutive in "utils"
        matches2, score2 = FuzzyMatcher.match("ul", "utils")

        assert matches1 is True
        assert matches2 is True
        assert score1 > score2

    def test_gap_penalty(self):
        """Large gaps should result in lower scores."""
        # Small gap - neither at start, so we isolate gap effect
        matches1, score1 = FuzzyMatcher.match("bc", "xbcd")
        # Large gap
        matches2, score2 = FuzzyMatcher.match("bc", "xb___c")

        assert matches1 is True
        assert matches2 is True
        assert score1 > score2  # Consecutive bc scores higher than b___c

    def test_filter_and_sort_basic(self):
        """filter_and_sort should filter and sort by score."""
        items = [
            ("utils", "data1"),
            ("test", "data2"),
            ("unit_test", "data3"),
        ]

        result = FuzzyMatcher.filter_and_sort("ut", items)

        # Should match "utils" and "unit_test", not "test"
        names = [r[0] for r in result]
        assert "utils" in names
        assert "unit_test" in names
        assert "test" not in names

    def test_filter_and_sort_empty_pattern(self):
        """Empty pattern should return all items."""
        items = [
            ("apple", "data1"),
            ("banana", "data2"),
        ]

        result = FuzzyMatcher.filter_and_sort("", items)

        assert len(result) == 2

    def test_filter_and_sort_no_matches(self):
        """No matches should return empty list."""
        items = [
            ("apple", "data1"),
            ("banana", "data2"),
        ]

        result = FuzzyMatcher.filter_and_sort("xyz", items)

        assert len(result) == 0

    def test_filter_and_sort_sorting(self):
        """Results should be sorted by score descending."""
        items = [
            ("unit_test_lib", "data1"),  # ut at boundaries
            ("utilities", "data2"),       # ut at start, consecutive
            ("output", "data3"),          # ut not at start
        ]

        result = FuzzyMatcher.filter_and_sort("ut", items)

        # "utilities" should be first (start bonus + consecutive)
        assert result[0][0] == "utilities"


class TestAtFileCompleterFuzzy:
    """Tests for fuzzy matching in AtFileCompleter."""

    def test_fuzzy_file_completion(self):
        """Should fuzzy match files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "utils.py").touch()
            Path(tmpdir, "tests.py").touch()
            Path(tmpdir, "main.py").touch()

            completer = AtFileCompleter(base_path=tmpdir)
            doc = Document(text="@utl", cursor_position=4)
            completions = list(completer.get_completions(doc, None))

            # Should find utils.py with fuzzy match
            names = [c.text for c in completions]
            assert "utils.py" in names

    def test_fuzzy_respects_order(self):
        """Fuzzy match should require correct character order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "utils.py").touch()

            completer = AtFileCompleter(base_path=tmpdir)
            # "ltu" is out of order for "utils"
            doc = Document(text="@ltu", cursor_position=4)
            completions = list(completer.get_completions(doc, None))

            assert len(completions) == 0

    def test_directory_navigation_with_fuzzy(self):
        """Should support directory navigation with fuzzy filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory with files
            subdir = Path(tmpdir, "src")
            subdir.mkdir()
            Path(subdir, "utils.py").touch()
            Path(subdir, "main.py").touch()

            completer = AtFileCompleter(base_path=tmpdir)
            doc = Document(text="@src/utl", cursor_position=8)
            completions = list(completer.get_completions(doc, None))

            names = [c.text for c in completions]
            assert "utils.py" in names

    def test_empty_pattern_lists_all(self):
        """Empty pattern after @ should list all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").touch()
            Path(tmpdir, "file2.txt").touch()

            completer = AtFileCompleter(base_path=tmpdir)
            doc = Document(text="@", cursor_position=1)
            completions = list(completer.get_completions(doc, None))

            # Should list both files
            assert len(completions) >= 2

    def test_skips_hidden_files(self):
        """Should skip hidden files (starting with .)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "visible.txt").touch()
            Path(tmpdir, ".hidden.txt").touch()

            completer = AtFileCompleter(base_path=tmpdir)
            doc = Document(text="@", cursor_position=1)
            completions = list(completer.get_completions(doc, None))

            names = [c.text for c in completions]
            assert "visible.txt" in names
            assert ".hidden.txt" not in names

    def test_directory_display_with_slash(self):
        """Directories should display with trailing slash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "subdir").mkdir()

            completer = AtFileCompleter(base_path=tmpdir)
            doc = Document(text="@sub", cursor_position=4)
            completions = list(completer.get_completions(doc, None))

            assert len(completions) == 1
            # Display can be FormattedText or string - convert to string for check
            display = completions[0].display
            display_str = display[0][1] if hasattr(display, '__getitem__') and not isinstance(display, str) else str(display)
            assert display_str.endswith("/")
            # Text should not have trailing slash (user types it)
            assert not completions[0].text.endswith("/")


class TestPercentPromptCompleterFuzzy:
    """Tests for fuzzy matching in PercentPromptCompleter."""

    def test_fuzzy_prompt_completion(self):
        """Should fuzzy match prompt names."""
        prompts = [
            {'name': 'code-review', 'description': 'Review code'},
            {'name': 'create-test', 'description': 'Create tests'},
            {'name': 'summarize', 'description': 'Summarize'},
        ]
        completer = PercentPromptCompleter(lambda: prompts)

        # "cr" should match "code-review" and "create-test"
        doc = Document(text="%cr", cursor_position=3)
        completions = list(completer.get_completions(doc, None))

        names = [c.text for c in completions]
        assert "code-review" in names
        assert "create-test" in names
        assert "summarize" not in names

    def test_fuzzy_non_consecutive(self):
        """Should match non-consecutive characters."""
        prompts = [
            {'name': 'code-review', 'description': 'Review code'},
        ]
        completer = PercentPromptCompleter(lambda: prompts)

        # "cw" matches "code-revieW"
        doc = Document(text="%cw", cursor_position=3)
        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        assert completions[0].text == "code-review"

    def test_fuzzy_sorting_by_score(self):
        """Results should be sorted by fuzzy match score."""
        prompts = [
            {'name': 'unit-test', 'description': 'Unit tests'},
            {'name': 'utilities', 'description': 'Utility functions'},
            {'name': 'output-format', 'description': 'Output formatting'},
        ]
        completer = PercentPromptCompleter(lambda: prompts)

        doc = Document(text="%ut", cursor_position=3)
        completions = list(completer.get_completions(doc, None))

        names = [c.text for c in completions]
        # "utilities" should be first (ut at start, consecutive)
        assert names[0] == "utilities"

    def test_empty_pattern_shows_all(self):
        """Empty pattern should show all prompts."""
        prompts = [
            {'name': 'prompt1', 'description': 'First'},
            {'name': 'prompt2', 'description': 'Second'},
        ]
        completer = PercentPromptCompleter(lambda: prompts)

        doc = Document(text="%", cursor_position=1)
        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 2

    def test_case_insensitive_fuzzy(self):
        """Fuzzy matching should be case-insensitive."""
        prompts = [
            {'name': 'CodeReview', 'description': 'Review code'},
        ]
        completer = PercentPromptCompleter(lambda: prompts)

        doc = Document(text="%cr", cursor_position=3)
        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        assert completions[0].text == "CodeReview"
