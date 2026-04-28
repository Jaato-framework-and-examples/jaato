"""Tests for the workspace-panel ``openers`` module.

Covers config loading (project + user merge, malformed input handling,
backward-compatible string shorthand, new action-dict shape) and pattern
resolution (basename vs path matching, longest-pattern tie-breaking,
per-action fallthrough across matching patterns, ``$EDITOR``/``$VISUAL``
placeholder expansion).
"""

import json

import pytest

from openers import ACTION_DIFF, ACTION_RAW, load_openers, resolve_opener


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

    def test_user_only_string_shorthand(self, tmp_path):
        # Bare string is normalised to {"raw": <string>} for backward compat.
        user = tmp_path / "user.json"
        user.write_text(json.dumps({"*.md": "glow"}))
        merged = load_openers(
            project_path=str(tmp_path / "missing.json"),
            user_path=str(user),
        )
        assert merged == {"*.md": {"raw": "glow"}}

    def test_user_only_action_dict(self, tmp_path):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({
            "*.md": {"raw": "glow", "diff": "git diff HEAD --"},
        }))
        merged = load_openers(
            project_path=str(tmp_path / "missing.json"),
            user_path=str(user),
        )
        assert merged == {
            "*.md": {"raw": "glow", "diff": "git diff HEAD --"},
        }

    def test_project_overrides_user_per_action(self, tmp_path):
        # User declares both raw + diff.  Project overrides only raw.
        # The user's diff entry must survive.
        user = tmp_path / "user.json"
        user.write_text(json.dumps({
            "*.md": {"raw": "less", "diff": "diff -u"},
            "*.png": "feh",
        }))
        project = tmp_path / "project.json"
        project.write_text(json.dumps({
            "*.md": {"raw": "glow"},
        }))

        merged = load_openers(
            project_path=str(project),
            user_path=str(user),
        )
        assert merged == {
            "*.md": {"raw": "glow", "diff": "diff -u"},
            "*.png": {"raw": "feh"},
        }

    def test_project_string_replaces_user_raw_only(self, tmp_path):
        # A bare string in the project file means "set raw to this".
        # User's diff entry should still survive.
        user = tmp_path / "user.json"
        user.write_text(json.dumps({
            "*.md": {"raw": "less", "diff": "diff -u"},
        }))
        project = tmp_path / "project.json"
        project.write_text(json.dumps({"*.md": "glow"}))

        merged = load_openers(
            project_path=str(project),
            user_path=str(user),
        )
        assert merged == {
            "*.md": {"raw": "glow", "diff": "diff -u"},
        }

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
        assert merged == {"*.md": {"raw": "glow"}}

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
        assert merged == {"*.md": {"raw": "glow"}}

    def test_non_dict_root_skipped(self, tmp_path):
        bad = tmp_path / "user.json"
        bad.write_text(json.dumps(["*.md", "glow"]))  # array, not object

        merged = load_openers(
            project_path=str(tmp_path / "missing.json"),
            user_path=str(bad),
        )
        assert merged == {}

    def test_unsupported_pattern_value_skipped(self, tmp_path):
        # Lists, ints, etc. at the pattern level are not valid.
        user = tmp_path / "user.json"
        user.write_text(json.dumps({
            "*.md": "glow",
            "*.png": ["chafa"],   # list, not string or object → skipped
            "*.jpg": 42,           # int → skipped
        }))
        merged = load_openers(
            project_path=str(tmp_path / "missing.json"),
            user_path=str(user),
        )
        assert merged == {"*.md": {"raw": "glow"}}

    def test_non_string_action_command_skipped(self, tmp_path):
        # Inside an action dict, non-string commands are individually skipped.
        user = tmp_path / "user.json"
        user.write_text(json.dumps({
            "*.md": {"raw": "glow", "diff": 42},
        }))
        merged = load_openers(
            project_path=str(tmp_path / "missing.json"),
            user_path=str(user),
        )
        assert merged == {"*.md": {"raw": "glow"}}


# ---------------------------------------------------------------------------
# resolve_opener
# ---------------------------------------------------------------------------

class TestResolveOpenerRaw:
    """Tests for the default ``raw`` action."""

    def test_no_match_uses_default_editor(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener("foo.txt", {"*.md": {"raw": "glow"}})
        assert argv == ["nano"]

    def test_empty_openers_uses_default_editor(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener("foo.md", {})
        assert argv == ["nano"]

    def test_basename_match(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener("README.md", {"*.md": {"raw": "glow"}})
        assert argv == ["glow"]

    def test_path_match_when_no_basename_match(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener(
            "docs/foo.txt",
            {"docs/*": {"raw": "less"}},
        )
        assert argv == ["less"]

    def test_longest_pattern_wins(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener(
            "README.md",
            {
                "*":    {"raw": "less"},   # length 1
                "*.md": {"raw": "glow"},   # length 4 → wins
            },
        )
        assert argv == ["glow"]

    def test_basename_beats_path_on_length_tie(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        # path pattern is longer → wins regardless of priority
        argv = resolve_opener(
            "notes/a.md",
            {
                "n*/*md": {"raw": "less"},   # length 7, path match only
                "*.md":   {"raw": "glow"},   # length 4, basename match
            },
        )
        assert argv == ["less"]

        # Equal length: basename should beat path
        argv = resolve_opener(
            "notes/a.md",
            {
                "no*/*": {"raw": "less"},   # length 5, path match
                "*a.md": {"raw": "glow"},   # length 5, basename match → wins
            },
        )
        assert argv == ["glow"]

    def test_command_with_arguments(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener(
            "README.md",
            {"*.md": {"raw": "glow -p --width 80"}},
        )
        assert argv == ["glow", "-p", "--width", "80"]

    def test_quoted_arguments_preserved(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener(
            "README.md",
            {"*.md": {"raw": "myviewer --title 'Hello World'"}},
        )
        assert argv == ["myviewer", "--title", "Hello World"]

    def test_editor_placeholder_expansion(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nvim")
        argv = resolve_opener("foo.txt", {"*": {"raw": "$EDITOR"}})
        assert argv == ["nvim"]

    def test_visual_placeholder_expansion(self, monkeypatch):
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setenv("VISUAL", "code")
        argv = resolve_opener("foo.txt", {"*": {"raw": "$VISUAL"}})
        assert argv == ["code"]

    def test_editor_placeholder_falls_back_to_vi(self, monkeypatch):
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)
        argv = resolve_opener("foo.txt", {"*": {"raw": "$EDITOR"}})
        assert argv == ["vi"]

    def test_other_env_vars_expanded(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        monkeypatch.setenv("MY_PAGER", "less")
        argv = resolve_opener(
            "foo.txt",
            {"*.txt": {"raw": "$MY_PAGER -R"}},
        )
        assert argv == ["less", "-R"]

    def test_unknown_env_var_remains_literal(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
        argv = resolve_opener(
            "foo.txt",
            {"*.txt": {"raw": "tool $DEFINITELY_NOT_SET"}},
        )
        assert argv == ["tool", "$DEFINITELY_NOT_SET"]

    def test_absolute_path_basename_matched(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        # Sandbox-monitored files are tracked with absolute paths; the
        # opener resolver should still pattern-match the basename.
        argv = resolve_opener(
            "/tmp/sandbox/notes.md",
            {"*.md": {"raw": "glow"}},
        )
        assert argv == ["glow"]


class TestResolveOpenerDiff:
    """Tests for the ``diff`` action and per-action fallthrough."""

    def test_no_match_returns_empty(self, monkeypatch):
        # Unlike raw, diff has no built-in fallback.
        monkeypatch.setenv("EDITOR", "nano")
        argv = resolve_opener(
            "foo.txt",
            {"*.md": {"raw": "glow"}},
            action=ACTION_DIFF,
        )
        assert argv == []

    def test_action_specific_match(self, monkeypatch):
        monkeypatch.setenv("EDITOR", "nano")
        openers = {
            "*.md": {"raw": "glow", "diff": "git diff HEAD --"},
        }
        assert resolve_opener("README.md", openers, action=ACTION_RAW) == ["glow"]
        assert resolve_opener("README.md", openers, action=ACTION_DIFF) == [
            "git", "diff", "HEAD", "--",
        ]

    def test_fallthrough_when_specific_pattern_lacks_action(self, monkeypatch):
        # *.py defines only raw.  *.md defines only diff.  *  defines diff.
        # For foo.py + diff, *.py is the most-specific match but lacks diff;
        # the resolver should fall through to *.
        monkeypatch.setenv("EDITOR", "nano")
        openers = {
            "*.py": {"raw": "code"},
            "*.md": {"diff": "git diff HEAD --"},
            "*":    {"diff": "diff -u"},
        }
        argv = resolve_opener("foo.py", openers, action=ACTION_DIFF)
        assert argv == ["diff", "-u"]

    def test_specific_pattern_wins_when_it_defines_action(self, monkeypatch):
        # Once a more-specific pattern defines the action, fallthrough stops.
        monkeypatch.setenv("EDITOR", "nano")
        openers = {
            "*.py": {"diff": "specific-diff"},
            "*":    {"diff": "fallback-diff"},
        }
        argv = resolve_opener("foo.py", openers, action=ACTION_DIFF)
        assert argv == ["specific-diff"]

    def test_action_independence(self, monkeypatch):
        # raw and diff resolve independently — overriding one doesn't drag
        # in the other.
        monkeypatch.setenv("EDITOR", "nano")
        openers = {
            "*.py": {"raw": "code"},
            "*":    {"raw": "$EDITOR", "diff": "diff -u"},
        }
        # raw: *.py wins for foo.py
        assert resolve_opener("foo.py", openers, action=ACTION_RAW) == ["code"]
        # diff: *.py has no diff → falls through to *
        assert resolve_opener("foo.py", openers, action=ACTION_DIFF) == [
            "diff", "-u",
        ]

    def test_diff_with_empty_command_returns_empty(self, monkeypatch):
        # Edge case: a pattern explicitly defines an empty diff command.
        # For non-raw actions an empty argv after expansion yields no-op.
        monkeypatch.setenv("EDITOR", "nano")
        openers = {"*.md": {"diff": ""}}
        argv = resolve_opener("README.md", openers, action=ACTION_DIFF)
        assert argv == []
