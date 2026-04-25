"""Tests for the workspace-panel ``openers`` module.

Covers config loading (project + user merge, malformed input handling)
and pattern resolution (basename vs path matching, longest-pattern
tie-breaking, ``$EDITOR``/``$VISUAL`` placeholder expansion).
"""

import json
import os

import pytest

from openers import load_openers, resolve_opener


# ---------------------------------------------------------------------------
# load_openers
# ---------------------------------------------------------------------------

class TestLoadOpeners:
    """Tests for the JSON config loader."""

    def test_missing_files_returns_empty(self, tmp_path):
        merged = load_openers(
            project_path=str(tmp_path / "missing-project.json"),
            user_path=str(tmp_path / "missing-user.json"),
        )
        assert merged == {}

    def test_user_only(self, tmp_path):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({"*.md": "glow"}))
        merged = load_openers(
            project_path=str(tmp_path / "missing.json"),
            user_path=str(user),
        )
        assert merged == {"*.md": "glow"}

    def test_project_overrides_user(self, tmp_path):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({"*.md": "less", "*.png": "feh"}))
        project = tmp_path / "project.json"
        project.write_text(json.dumps({"*.md": "glow"}))

        merged = load_openers(
            project_path=str(project),
            user_path=str(user),
        )
        # *.md overridden, *.png inherited from user
        assert merged == {"*.md": "glow", "*.png": "feh"}

    def test_underscore_keys_skipped(self, tmp_path):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({
            "_comment": "this is a comment",
            "_schema": "1",
            "*.md": "glow",
        }))
        merged = load_openers(
            project_path=str(tmp_path / "missing.json"),
            user_path=str(user),
        )
        assert merged == {"*.md": "glow"}

    def test_invalid_json_skipped(self, tmp_path, caplog):
        bad = tmp_path / "user.json"
        bad.write_text("{not valid json")
        good = tmp_path / "project.json"
        good.write_text(json.dumps({"*.md": "glow"}))

        merged = load_openers(
            project_path=str(good),
            user_path=str(bad),
        )
        # Bad file silently dropped; good file still loads.
        assert merged == {"*.md": "glow"}

    def test_non_dict_root_skipped(self, tmp_path):
        bad = tmp_path / "user.json"
        bad.write_text(json.dumps(["*.md", "glow"]))  # array, not object

        merged = load_openers(
            project_path=str(tmp_path / "missing.json"),
            user_path=str(bad),
        )
        assert merged == {}

    def test_non_string_command_skipped(self, tmp_path):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({
            "*.md": "glow",
            "*.png": ["chafa"],   # list, not string → skipped
            "*.jpg": 42,           # int, not string → skipped
        }))
        merged = load_openers(
            project_path=str(tmp_path / "missing.json"),
            user_path=str(user),
        )
        assert merged == {"*.md": "glow"}


# ---------------------------------------------------------------------------
# resolve_opener
# ---------------------------------------------------------------------------

class TestResolveOpener:
    """Tests for pattern resolution and command construction."""

    def test_no_match_uses_default_editor(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener("foo.txt", {"*.md": "glow"})
        assert argv == ["nano"]

    def test_empty_openers_uses_default_editor(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener("foo.md", {})
        assert argv == ["nano"]

    def test_basename_match(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener("README.md", {"*.md": "glow"})
        assert argv == ["glow"]

    def test_path_match_when_no_basename_match(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        # The basename is `foo.txt` (no match for docs/*); the path
        # `docs/foo.txt` matches.
        argv = resolve_opener(
            "docs/foo.txt",
            {"docs/*": "less"},
        )
        assert argv == ["less"]

    def test_longest_pattern_wins(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener(
            "README.md",
            {
                "*": "less",          # length 1
                "*.md": "glow",       # length 4 → wins
            },
        )
        assert argv == ["glow"]

    def test_basename_beats_path_on_length_tie(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        # Both patterns are length 5 and both match `notes/x.md`:
        #   `*x.md` matches the basename `x.md` (length 5)
        #   `*x.md` also matches `notes/x.md` as a path
        # Add a path-only pattern of equal length to verify basename wins.
        argv = resolve_opener(
            "notes/a.md",
            {
                "n*/*md": "less",   # length 7, path match only
                "*.md":   "glow",   # length 4, basename match
            },
        )
        # path pattern is longer → wins regardless of priority
        assert argv == ["less"]

        # Now equal length: basename should beat path
        argv = resolve_opener(
            "notes/a.md",
            {
                "no*/*": "less",   # length 5, path match
                "*a.md": "glow",   # length 5, basename match → wins
            },
        )
        assert argv == ["glow"]

    def test_command_with_arguments(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener("README.md", {"*.md": "glow -p --width 80"})
        assert argv == ["glow", "-p", "--width", "80"]

    def test_quoted_arguments_preserved(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener(
            "README.md",
            {"*.md": "myviewer --title 'Hello World'"},
        )
        assert argv == ["myviewer", "--title", "Hello World"]

    def test_editor_placeholder_expansion(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nvim")
        argv = resolve_opener("foo.txt", {"*": "$EDITOR"})
        assert argv == ["nvim"]

    def test_visual_placeholder_expansion(self, monkeypatch):
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setenv("VISUAL", "code")
        argv = resolve_opener("foo.txt", {"*": "$VISUAL"})
        assert argv == ["code"]

    def test_editor_placeholder_falls_back_to_vi(self, monkeypatch):
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)
        argv = resolve_opener("foo.txt", {"*": "$EDITOR"})
        assert argv == ["vi"]

    def test_other_env_vars_expanded(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        monkeypatch.setenv("MY_PAGER", "less")
        argv = resolve_opener("foo.txt", {"*.txt": "$MY_PAGER -R"})
        assert argv == ["less", "-R"]

    def test_unknown_env_var_remains_literal(self, monkeypatch):
        # If a variable isn't set, expandvars leaves it as-is.  Documented
        # behavior — users get a recognisable failure rather than silent
        # substitution.
        monkeypatch.setenv("EDITOR", "nano")
        monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
        argv = resolve_opener("foo.txt", {"*.txt": "tool $DEFINITELY_NOT_SET"})
        assert argv == ["tool", "$DEFINITELY_NOT_SET"]

    def test_absolute_path_basename_matched(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        # Sandbox-monitored files are tracked with absolute paths; the
        # opener resolver should still pattern-match the basename.
        argv = resolve_opener("/tmp/sandbox/notes.md", {"*.md": "glow"})
        assert argv == ["glow"]
