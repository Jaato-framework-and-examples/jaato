"""``toggle_gitignore_pattern`` / ``validate_ignore_pattern`` — the one edit two callers share."""
import pytest

from jaato_sdk.gitignore_toggle import toggle_gitignore_pattern, validate_ignore_pattern


class TestToggle:
    def test_adds_to_an_empty_file(self):
        assert toggle_gitignore_pattern("", "build/") == ("build/\n", True)

    def test_appends_after_a_missing_trailing_newline(self):
        text, ignored = toggle_gitignore_pattern("*.pyc", "build/")
        assert text == "*.pyc\nbuild/\n" and ignored is True

    def test_removes_every_exact_occurrence_and_nothing_else(self):
        existing = "# comment\nbuild/\n  build/  \nbuild/dist\n*.log\n"
        text, ignored = toggle_gitignore_pattern(existing, "build/")
        assert text == "# comment\nbuild/dist\n*.log\n" and ignored is False

    def test_a_glob_covering_the_path_is_not_a_match(self):
        # The toggle is about THIS entry; a wider pattern stays and the entry
        # is written beside it, exactly as the TUI has always done.
        text, ignored = toggle_gitignore_pattern("*.log\n", "debug.log")
        assert text == "*.log\ndebug.log\n" and ignored is True

    def test_a_commented_copy_is_never_touched(self):
        text, ignored = toggle_gitignore_pattern("#build/\n", "build/")
        assert text == "#build/\nbuild/\n" and ignored is True

    def test_round_trip_restores_the_file(self):
        original = "a\nb/\n"
        added, _ = toggle_gitignore_pattern(original, "c")
        removed, _ = toggle_gitignore_pattern(added, "c")
        assert removed == original

    def test_removing_the_only_line_leaves_an_empty_file(self):
        assert toggle_gitignore_pattern("c\n", "c") == ("", False)


class TestValidate:
    @pytest.mark.parametrize("ok", ["src/app.py", "build/", ".jaato/logs/", "a b.txt"])
    def test_a_workspace_entry_is_accepted(self, ok):
        assert validate_ignore_pattern(ok) is None

    @pytest.mark.parametrize("bad", ["", "   ", "a\nb", "a\rb", "a\0b", "/etc/passwd",
                                     "C:\\x", "#note", "!keep", "  !keep"])
    def test_what_is_not_an_entry_is_refused_with_a_reason(self, bad):
        reason = validate_ignore_pattern(bad)
        assert reason, bad
