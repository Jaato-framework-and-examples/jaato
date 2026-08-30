"""Tests for :mod:`shared.command_analysis`.

The module is a security boundary input, so the tests are organised around
the two properties that matter: the analyzer must see *every* command a
string would run, and it must refuse — never guess — when the string cannot
be modelled the way ``/bin/sh`` would parse it.
"""

import pytest

from shared.command_analysis import (
    Segment,
    UnanalyzableCommand,
    analyze_command,
    resolve_command_names,
)


def names_of(command: str):
    """Effective command name of every segment, in analysis order."""
    return [seg.command_name for seg in analyze_command(command)]


class TestSegmentation:
    """Compound commands are split into the simple commands shell runs."""

    @pytest.mark.parametrize("command,expected", [
        ("cat README.md", ["cat"]),
        ("cat README.md && rm -rf notes/", ["cat", "rm"]),
        ("cat a || rm b", ["cat", "rm"]),
        ("cat a; rm b", ["cat", "rm"]),
        ("cat a | tee b", ["cat", "tee"]),
        ("cat a & rm b", ["cat", "rm"]),
        ("cat a\nrm b", ["cat", "rm"]),
        ("( rm b )", ["rm"]),
        ("! rm b", ["rm"]),
        ("cat a |& rm b", ["cat", "rm"]),
    ])
    def test_separators_split_segments(self, command, expected):
        assert names_of(command) == expected

    def test_separator_inside_quotes_does_not_split(self):
        assert names_of("echo 'a && b'") == ["echo"]
        assert names_of('echo "a; b"') == ["echo"]

    def test_escaped_separator_does_not_split(self):
        assert names_of(r"echo a \&\& b") == ["echo"]

    def test_line_continuation_is_not_a_separator(self):
        # A backslash-newline joins the line; it must not read as a command
        # boundary, and the continued word belongs to the same segment.
        assert names_of("cat \\\n /etc/passwd") == ["cat"]

    def test_basename_is_reported(self):
        assert names_of("/usr/bin/rm -rf x") == ["rm"]


class TestCommandSubstitution:
    """``$(...)``/backticks/process substitution are commands, not text."""

    @pytest.mark.parametrize("command", [
        "echo $(rm -rf /etc)",
        "echo `rm -rf /etc`",
        'echo "$(rm -rf /etc)"',
        "echo $(echo $(rm -rf /etc))",
    ])
    def test_substituted_command_is_visible(self, command):
        assert "rm" in names_of(command)

    def test_process_substitution_body_is_analyzed(self):
        assert names_of("diff <(cat a) <(rm b)") == ["diff", "cat", "rm"]

    def test_arithmetic_expansion_is_not_a_command(self):
        # $((...)) is arithmetic, not command substitution: it must not
        # invent a segment, and it must not be mistaken for one either.
        assert names_of("echo $((RANDOM=2+2))") == ["echo"]

    def test_parameter_expansion_stays_a_word(self):
        segments = analyze_command("cat ${HOME}/x")
        assert [s.words for s in segments] == [["cat", "${HOME}/x"]]

    def test_nesting_depth_is_bounded(self):
        deep = "echo " + "$(" * 12 + "rm" + ")" * 12
        with pytest.raises(UnanalyzableCommand):
            analyze_command(deep)


class TestRedirections:
    """The full FD-redirect grammar, each tagged with the access it grants."""

    @pytest.mark.parametrize("command,op,fd,target,mode", [
        ("cmd > f", ">", None, "f", "write"),
        ("cmd >> f", ">>", None, "f", "write"),
        ("cmd >|f", ">|", None, "f", "write"),
        ("cmd 2>f", ">", "2", "f", "write"),
        ("cmd 1>>f", ">>", "1", "f", "write"),
        ("cmd &>f", "&>", None, "f", "write"),
        ("cmd &>>f", "&>>", None, "f", "write"),
        ("cmd >&f", ">&", None, "f", "write"),
        ("cmd <>f", "<>", None, "f", "write"),
        ("cmd <f", "<", None, "f", "read"),
        ("cmd <&f", "<&", None, "f", "read"),
        ("cmd >&2", ">&", None, "2", "none"),
        ("cmd 2>&1", ">&", "2", "1", "none"),
        ("cmd <&-", "<&", None, "-", "none"),
        ("cmd <<<hello", "<<<", None, "hello", "none"),
    ])
    def test_redirect_forms(self, command, op, fd, target, mode):
        (segment,) = analyze_command(command)
        (redirect,) = segment.redirects
        assert (redirect.op, redirect.fd, redirect.target, redirect.mode) == (
            op, fd, target, mode
        )

    def test_digits_only_prefix_is_an_fd(self):
        # ``echo foo2>bar`` redirects stdout; ``foo2`` stays a word.
        (segment,) = analyze_command("echo foo2>bar")
        assert segment.words == ["echo", "foo2"]
        assert segment.redirects[0].fd is None

    def test_quoted_redirect_is_literal_text(self):
        (segment,) = analyze_command('echo ">/etc/passwd"')
        assert segment.words == ["echo", ">/etc/passwd"]
        assert segment.redirects == []

    def test_heredoc_body_is_not_lexed_as_commands(self):
        # The body is data. Lexing it would invent an ``rm`` segment that
        # never runs (and, worse, path tokens that never get opened).
        segments = analyze_command("cat <<EOF\nrm -rf /\nEOF\nls")
        assert [s.command_name for s in segments] == ["cat", "ls"]

    def test_heredoc_with_tab_stripping(self):
        segments = analyze_command("cat <<-END\n\tbody\n\tEND\nls")
        assert [s.command_name for s in segments] == ["cat", "ls"]

    def test_unterminated_heredoc_consumes_to_end(self):
        segments = analyze_command("cat <<EOF\nrm -rf /\n")
        assert [s.command_name for s in segments] == ["cat"]


class TestFailClosed:
    """Anything the analyzer cannot model is refused, never guessed."""

    @pytest.mark.parametrize("command", [
        'cat "/etc/passwd',      # unbalanced double quote
        "cat '/etc/passwd",      # unbalanced single quote
        "echo $(rm -rf /",       # unbalanced substitution
        "echo `rm -rf /",        # unbalanced backtick
        "cat /etc/passwd \\",    # dangling backslash
        "cat >",                 # redirection with no target
        "cat > | wc",            # redirection target eaten by an operator
        "cat a &&",              # dangling chain operator
        "cat a ||",
        "cat a |",
        "&& rm -rf /",           # chain operator with nothing before it
    ])
    def test_refused(self, command):
        with pytest.raises(UnanalyzableCommand):
            analyze_command(command)

    def test_unanalyzable_is_a_value_error(self):
        # Callers that historically wrapped ``shlex.split`` in
        # ``except ValueError`` keep working unchanged.
        assert issubclass(UnanalyzableCommand, ValueError)


class TestResolveCommandNames:
    """Assignments, keywords and wrappers do not hide the real command."""

    @pytest.mark.parametrize("words,expected", [
        (["cat", "x"], ["cat"]),
        (["FOO=bar", "rm", "x"], ["rm"]),
        (["FOO=bar", "BAZ=qux", "rm"], ["rm"]),
        (["arr[0]=1", "rm"], ["rm"]),
        (["FOO+=bar", "rm"], ["rm"]),
        (["sudo", "rm", "-rf", "x"], ["sudo", "rm"]),
        (["env", "FOO=bar", "rm"], ["env", "rm"]),
        (["xargs", "-0", "rm"], ["xargs", "rm"]),
        (["FOO=bar"], []),
        ([], []),
    ])
    def test_resolution(self, words, expected):
        assert resolve_command_names(words) == expected

    def test_segment_command_name_is_innermost(self):
        assert Segment(words=["sudo", "rm"]).command_name == "rm"
        assert Segment(words=["FOO=bar"]).command_name == ""


class TestScale:
    """Long inputs stay analyzable rather than degrading to a guess."""

    def test_very_long_command_still_segments(self):
        # Claude Code's fix for this row was "always prompt above 10k
        # characters"; ours is stronger — the hidden segment is still seen.
        filler = " ".join(["x"] * 4000)
        command = f"echo {filler} && rm -rf /etc"
        assert names_of(command) == ["echo", "rm"]
