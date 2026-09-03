"""Tests for line-ending preservation across ``file_edit`` writes (#805).

The bug these pin down: a targeted ``old``/``new`` edit rewrote *every* line
ending in the file, silently converting a CRLF file to LF and turning a
one-line change into a whole-file diff.  The rule now is that a write
reproduces what git would put in the working tree, and preserves the file's
own convention when git has no opinion.

Git state is built by writing ``.git/config`` and ``.gitattributes`` directly
rather than shelling out to ``git init``: the tests then need no git binary
and exercise exactly the files the resolver reads.  ``HOME`` and
``XDG_CONFIG_HOME`` are redirected for every test in this module, so a
developer's own ``core.autocrlf`` cannot change the outcome.
"""

import os
from pathlib import Path

import pytest

from ..git_eol import CRLF, LF, GitEolResolver
from ..line_endings import (
    LineEndingPolicy,
    detect_line_ending,
    normalize,
    restore,
)
from ..plugin import FileEditPlugin


@pytest.fixture(autouse=True)
def isolated_git_home(tmp_path, monkeypatch):
    """Point git's global-config lookup at an empty directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def make_repo(root: Path, *, config: str = "", attributes: str = None) -> Path:
    """Materialise a minimal git repository at *root*.

    Args:
        root: Directory to turn into a repository (created if absent).
        config: Body of ``.git/config``.
        attributes: Body of the root ``.gitattributes``, or ``None`` for none.

    Returns:
        *root*, for chaining.
    """
    (root / ".git").mkdir(parents=True, exist_ok=True)
    (root / ".git" / "config").write_text(config, encoding="utf-8")
    if attributes is not None:
        (root / ".gitattributes").write_text(attributes, encoding="utf-8")
    return root


def make_plugin(workspace: Path) -> FileEditPlugin:
    """A plugin rooted at *workspace*, with backups inside it."""
    plugin = FileEditPlugin()
    plugin.initialize({
        "workspace_root": str(workspace),
        "backup_dir": str(workspace / ".backups"),
    })
    plugin.set_workspace_path(str(workspace))
    return plugin


class TestDetectLineEnding:
    """Which ending a file is judged to use."""

    def test_crlf(self):
        assert detect_line_ending("a\r\nb\r\n") == CRLF

    def test_lf(self):
        assert detect_line_ending("a\nb\n") == LF

    def test_cr_only(self):
        assert detect_line_ending("a\rb\r") == "\r"

    def test_no_line_ending_at_all(self):
        assert detect_line_ending("no newline here") is None
        assert detect_line_ending("") is None

    def test_mixed_picks_the_majority(self):
        assert detect_line_ending("a\r\nb\r\nc\n") == CRLF
        assert detect_line_ending("a\nb\nc\r\n") == LF

    def test_tie_breaks_towards_crlf(self):
        """Nothing adds a CR by accident, so a tie is more likely a CRLF file."""
        assert detect_line_ending("a\r\nb\n") == CRLF


class TestNormalizeAndRestore:
    """The LF round trip that matching happens inside."""

    def test_normalize_collapses_every_flavour(self):
        assert normalize("a\r\nb\nc\rd") == "a\nb\nc\nd"

    def test_restore_lf_is_identity(self):
        assert restore("a\nb\n", LF) == "a\nb\n"

    def test_restore_crlf(self):
        assert restore("a\nb\n", CRLF) == "a\r\nb\r\n"

    def test_restore_is_idempotent_on_crlf_input(self):
        """Content that already holds CRLF must not gain a second CR."""
        assert restore("a\r\nb\r\n", CRLF) == "a\r\nb\r\n"

    def test_round_trip_preserves_content(self):
        original = "alpha\r\nbeta\r\n"
        assert restore(normalize(original), CRLF) == original


class TestGitEolResolver:
    """The repository setting, which outranks everything else."""

    def test_no_repository_means_no_opinion(self, tmp_path):
        assert GitEolResolver().ending_for(tmp_path / "a.txt") is None

    def test_default_config_converts_nothing(self, tmp_path):
        """autocrlf=false with no text attribute is git doing nothing at all."""
        make_repo(tmp_path)
        assert GitEolResolver().ending_for(tmp_path / "a.txt") is None

    def test_attributes_eol_crlf(self, tmp_path):
        make_repo(tmp_path, attributes="* text=auto eol=crlf\n")
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == CRLF

    def test_attributes_eol_lf(self, tmp_path):
        make_repo(tmp_path, attributes="* text=auto eol=lf\n")
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == LF

    def test_autocrlf_true(self, tmp_path):
        make_repo(tmp_path, config="[core]\n\tautocrlf = true\n")
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == CRLF

    def test_autocrlf_input(self, tmp_path):
        make_repo(tmp_path, config="[core]\n\tautocrlf = input\n")
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == LF

    def test_core_eol_applies_only_with_a_text_attribute(self, tmp_path):
        make_repo(tmp_path, config="[core]\n\teol = crlf\n")
        resolver = GitEolResolver()
        # No text attribute: git converts nothing, whatever core.eol says.
        assert resolver.ending_for(tmp_path / "a.txt") is None

        make_repo(tmp_path, config="[core]\n\teol = crlf\n", attributes="* text\n")
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == CRLF

    def test_minus_text_beats_autocrlf(self, tmp_path):
        make_repo(
            tmp_path,
            config="[core]\n\tautocrlf = true\n",
            attributes="*.bin -text\n",
        )
        resolver = GitEolResolver()
        assert resolver.ending_for(tmp_path / "a.bin") is None
        assert resolver.ending_for(tmp_path / "a.txt") == CRLF

    def test_binary_macro_means_no_conversion(self, tmp_path):
        make_repo(
            tmp_path,
            config="[core]\n\tautocrlf = true\n",
            attributes="*.png binary\n",
        )
        assert GitEolResolver().ending_for(tmp_path / "img.png") is None

    def test_attributes_eol_beats_autocrlf(self, tmp_path):
        make_repo(
            tmp_path,
            config="[core]\n\tautocrlf = true\n",
            attributes="* text eol=lf\n",
        )
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == LF

    def test_last_matching_line_wins(self, tmp_path):
        make_repo(tmp_path, attributes="* text eol=lf\n*.txt text eol=crlf\n")
        resolver = GitEolResolver()
        assert resolver.ending_for(tmp_path / "a.txt") == CRLF
        assert resolver.ending_for(tmp_path / "a.md") == LF

    def test_deeper_attributes_file_wins(self, tmp_path):
        make_repo(tmp_path, attributes="* text eol=lf\n")
        nested = tmp_path / "win"
        nested.mkdir()
        (nested / ".gitattributes").write_text("* text eol=crlf\n", encoding="utf-8")
        resolver = GitEolResolver()
        assert resolver.ending_for(nested / "a.txt") == CRLF
        assert resolver.ending_for(tmp_path / "a.txt") == LF

    def test_pattern_without_slash_matches_at_any_depth(self, tmp_path):
        make_repo(tmp_path, attributes="*.txt text eol=crlf\n")
        nested = tmp_path / "deep" / "deeper"
        nested.mkdir(parents=True)
        assert GitEolResolver().ending_for(nested / "a.txt") == CRLF

    def test_pattern_with_slash_is_anchored(self, tmp_path):
        make_repo(tmp_path, attributes="docs/*.txt text eol=crlf\n")
        (tmp_path / "docs").mkdir()
        (tmp_path / "src").mkdir()
        resolver = GitEolResolver()
        assert resolver.ending_for(tmp_path / "docs" / "a.txt") == CRLF
        assert resolver.ending_for(tmp_path / "src" / "a.txt") is None

    def test_double_star_crosses_directories(self, tmp_path):
        make_repo(tmp_path, attributes="docs/**/*.txt text eol=crlf\n")
        deep = tmp_path / "docs" / "a" / "b"
        deep.mkdir(parents=True)
        assert GitEolResolver().ending_for(deep / "x.txt") == CRLF

    def test_comments_and_blank_lines_are_skipped(self, tmp_path):
        make_repo(tmp_path, attributes="# a comment\n\n* text eol=crlf\n")
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == CRLF

    def test_inline_config_value_comment_is_stripped(self, tmp_path):
        make_repo(tmp_path, config="[core]\n\tautocrlf = input # for CI\n")
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == LF

    def test_repository_config_overrides_global(self, tmp_path, isolated_git_home):
        (isolated_git_home / ".gitconfig").write_text(
            "[core]\n\tautocrlf = true\n", encoding="utf-8"
        )
        make_repo(tmp_path, config="[core]\n\tautocrlf = input\n")
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == LF

    def test_global_config_applies_when_the_repository_is_silent(
        self, tmp_path, isolated_git_home
    ):
        (isolated_git_home / ".gitconfig").write_text(
            "[core]\n\tautocrlf = true\n", encoding="utf-8"
        )
        make_repo(tmp_path)
        assert GitEolResolver().ending_for(tmp_path / "a.txt") == CRLF

    def test_edited_attributes_file_is_picked_up(self, tmp_path):
        """A long-lived daemon must not cache a stale .gitattributes."""
        make_repo(tmp_path, attributes="* text eol=lf\n")
        resolver = GitEolResolver()
        assert resolver.ending_for(tmp_path / "a.txt") == LF

        attributes = tmp_path / ".gitattributes"
        attributes.write_text("* text eol=crlf\n", encoding="utf-8")
        os.utime(attributes, (0, 0))  # force a different mtime
        assert resolver.ending_for(tmp_path / "a.txt") == CRLF

    def test_unreadable_repository_resolves_to_no_opinion(self, tmp_path):
        """A line-ending preference must never be why a write fails."""
        make_repo(tmp_path, config="[core\n\tbroken", attributes="[[[ text\n")
        assert GitEolResolver().ending_for(tmp_path / "a.txt") is None


class TestLineEndingPolicy:
    """Precedence: repository setting, then the file's own ending, then LF."""

    def test_git_wins_over_the_files_own_ending(self, tmp_path):
        make_repo(tmp_path, attributes="* text eol=lf\n")
        assert LineEndingPolicy().ending_for(tmp_path / "a.txt", CRLF) == LF

    def test_file_ending_used_when_git_is_silent(self, tmp_path):
        assert LineEndingPolicy().ending_for(tmp_path / "a.txt", CRLF) == CRLF

    def test_lf_when_there_is_nothing_to_preserve(self, tmp_path):
        assert LineEndingPolicy().ending_for(tmp_path / "a.txt", None) == LF

    def test_load_returns_normalised_content_and_the_real_ending(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_bytes(b"one\r\ntwo\r\n")
        content, ending = LineEndingPolicy().load(target)
        assert content == "one\ntwo\n"
        assert ending == CRLF

    def test_ending_for_file_sniffs_an_unread_file(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_bytes(b"one\r\ntwo\r\n")
        assert LineEndingPolicy().ending_for_file(target) == CRLF

    def test_ending_for_file_on_a_missing_path_is_lf(self, tmp_path):
        assert LineEndingPolicy().ending_for_file(tmp_path / "nope.txt") == LF


class TestUpdateFilePreservesEndings:
    """The reproduction from the issue, and its neighbours."""

    def test_targeted_edit_leaves_other_endings_alone(self, tmp_path):
        """The issue's repro: one line edited, three endings rewritten."""
        target = tmp_path / "crlf.txt"
        target.write_bytes(b"line one\r\nline two\r\nline three\r\n")

        result = make_plugin(tmp_path)._execute_update_file(
            {"path": "crlf.txt", "old": "line two", "new": "LINE TWO"}
        )

        assert result.get("success") is True
        assert target.read_bytes() == b"line one\r\nLINE TWO\r\nline three\r\n"

    def test_lf_file_stays_lf(self, tmp_path):
        target = tmp_path / "lf.txt"
        target.write_bytes(b"line one\nline two\n")

        make_plugin(tmp_path)._execute_update_file(
            {"path": "lf.txt", "old": "line two", "new": "LINE TWO"}
        )

        assert target.read_bytes() == b"line one\nLINE TWO\n"

    def test_multi_line_old_written_in_lf_matches_a_crlf_file(self, tmp_path):
        """The model never sees CRs, so its ``old`` must not have to carry them."""
        target = tmp_path / "crlf.txt"
        target.write_bytes(b"one\r\ntwo\r\nthree\r\n")

        result = make_plugin(tmp_path)._execute_update_file(
            {"path": "crlf.txt", "old": "two\nthree", "new": "TWO\nTHREE"}
        )

        assert result.get("success") is True
        assert target.read_bytes() == b"one\r\nTWO\r\nTHREE\r\n"

    def test_new_text_containing_lf_gets_the_files_ending(self, tmp_path):
        target = tmp_path / "crlf.txt"
        target.write_bytes(b"one\r\ntwo\r\n")

        make_plugin(tmp_path)._execute_update_file(
            {"path": "crlf.txt", "old": "two", "new": "two\nthree"}
        )

        assert target.read_bytes() == b"one\r\ntwo\r\nthree\r\n"

    def test_full_replacement_keeps_the_files_ending(self, tmp_path):
        target = tmp_path / "crlf.txt"
        target.write_bytes(b"one\r\ntwo\r\n")

        make_plugin(tmp_path)._execute_update_file(
            {"path": "crlf.txt", "new_content": "alpha\nbeta\n"}
        )

        assert target.read_bytes() == b"alpha\r\nbeta\r\n"

    def test_full_replacement_does_not_double_the_cr(self, tmp_path):
        target = tmp_path / "crlf.txt"
        target.write_bytes(b"one\r\ntwo\r\n")

        make_plugin(tmp_path)._execute_update_file(
            {"path": "crlf.txt", "new_content": "alpha\r\nbeta\r\n"}
        )

        assert target.read_bytes() == b"alpha\r\nbeta\r\n"

    def test_mixed_file_is_repaired_to_its_dominant_ending(self, tmp_path):
        """Deliberate: rule 5 picks one ending, so the minority lines convert.

        This is the one case where an edit still changes a line it was not
        asked to change — bounded to the minority, where the whole file used
        to convert.  Pinned so it cannot happen by accident.
        """
        target = tmp_path / "mixed.txt"
        target.write_bytes(b"one\r\ntwo\nthree\r\n")

        make_plugin(tmp_path)._execute_update_file(
            {"path": "mixed.txt", "old": "one", "new": "ONE"}
        )

        assert target.read_bytes() == b"ONE\r\ntwo\r\nthree\r\n"

    def test_repository_setting_overrides_the_file(self, tmp_path):
        make_repo(tmp_path, attributes="* text=auto eol=lf\n")
        target = tmp_path / "stale.txt"
        target.write_bytes(b"one\r\ntwo\r\n")

        make_plugin(tmp_path)._execute_update_file(
            {"path": "stale.txt", "old": "two", "new": "TWO"}
        )

        assert target.read_bytes() == b"one\nTWO\n"

    def test_repository_mandating_crlf_converts_an_lf_file(self, tmp_path):
        make_repo(tmp_path, config="[core]\n\tautocrlf = true\n")
        target = tmp_path / "a.txt"
        target.write_bytes(b"one\ntwo\n")

        make_plugin(tmp_path)._execute_update_file(
            {"path": "a.txt", "old": "two", "new": "TWO"}
        )

        assert target.read_bytes() == b"one\r\nTWO\r\n"


class TestWriteNewFile:
    """A new file has no convention of its own, so only the repository speaks."""

    def test_defaults_to_lf(self, tmp_path):
        make_plugin(tmp_path)._execute_write_new_file(
            {"path": "new.txt", "content": "a\nb\n"}
        )
        assert (tmp_path / "new.txt").read_bytes() == b"a\nb\n"

    def test_follows_the_repository(self, tmp_path):
        make_repo(tmp_path, attributes="* text eol=crlf\n")
        make_plugin(tmp_path)._execute_write_new_file(
            {"path": "new.txt", "content": "a\nb\n"}
        )
        assert (tmp_path / "new.txt").read_bytes() == b"a\r\nb\r\n"


class TestMultiFileEdit:
    """The batch path reads bytes directly, so it needed its own fix."""

    def test_edit_preserves_crlf(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_bytes(b"one\r\ntwo\r\nthree\r\n")

        result = make_plugin(tmp_path)._execute_multi_file_edit({
            "operations": [
                {"action": "edit", "path": "a.txt", "old": "two", "new": "TWO"}
            ]
        })

        assert result.get("success") is True
        assert target.read_bytes() == b"one\r\nTWO\r\nthree\r\n"

    def test_multi_line_lf_old_matches_a_crlf_file(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_bytes(b"one\r\ntwo\r\nthree\r\n")

        result = make_plugin(tmp_path)._execute_multi_file_edit({
            "operations": [
                {
                    "action": "edit",
                    "path": "a.txt",
                    "old": "two\nthree",
                    "new": "TWO\nTHREE",
                }
            ]
        })

        assert result.get("success") is True
        assert target.read_bytes() == b"one\r\nTWO\r\nTHREE\r\n"

    def test_create_follows_the_repository(self, tmp_path):
        make_repo(tmp_path, attributes="* text eol=crlf\n")

        make_plugin(tmp_path)._execute_multi_file_edit({
            "operations": [
                {"action": "create", "path": "new.txt", "content": "a\nb\n"}
            ]
        })

        assert (tmp_path / "new.txt").read_bytes() == b"a\r\nb\r\n"

    def test_rollback_restores_the_original_bytes(self, tmp_path):
        """A failed batch must put the CRs back exactly as they were."""
        target = tmp_path / "a.txt"
        target.write_bytes(b"one\r\ntwo\r\n")

        result = make_plugin(tmp_path)._execute_multi_file_edit({
            "operations": [
                {"action": "edit", "path": "a.txt", "old": "two", "new": "TWO"},
                {"action": "edit", "path": "a.txt", "old": "absent", "new": "x"},
            ]
        })

        assert result.get("success") is False
        assert target.read_bytes() == b"one\r\ntwo\r\n"


class TestFindAndReplace:
    """Regex replacement across files keeps each file's own ending."""

    def test_crlf_preserved(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_bytes(b"foo\r\nbar\r\nfoo\r\n")

        result = make_plugin(tmp_path)._execute_find_and_replace(
            {"pattern": "foo", "replacement": "baz", "paths": "*.txt"}
        )

        assert result.get("success") is True
        assert target.read_bytes() == b"baz\r\nbar\r\nbaz\r\n"

    def test_replacement_introducing_a_newline_gets_the_files_ending(self, tmp_path):
        """A "\n" the caller wrote becomes the file's own ending, not a bare LF."""
        target = tmp_path / "a.txt"
        target.write_bytes(b"foo\r\nbar\r\n")

        result = make_plugin(tmp_path)._execute_find_and_replace(
            {"pattern": "bar", "replacement": "bar\nextra", "paths": "*.txt"}
        )

        assert result.get("success") is True
        assert target.read_bytes() == b"foo\r\nbar\r\nextra\r\n"


class TestReadFileStillReportsLf:
    """What the model sees is unchanged: always LF, whatever the file holds."""

    def test_crlf_file_reads_back_as_lf(self, tmp_path):
        """readFile renders a header plus the content, as a single string."""
        (tmp_path / "a.txt").write_bytes(b"one\r\ntwo\r\n")
        rendered = make_plugin(tmp_path)._execute_read_file({"path": "a.txt"})
        assert isinstance(rendered, str)
        assert "\r" not in rendered
        assert "one\ntwo\n" in rendered
