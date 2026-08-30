"""Structural analysis of shell command strings for permission analyzers.

Permission and sandbox checks that reason about *what a command does* have
to model how the shell will actually parse it.  Taking ``shlex.split(cmd)[0]``
as "the command" is the classic mistake: in ``cat README.md && rm -rf notes/``
the command that matters is ``rm``, not ``cat``, and every analyzer built on
``tokens[0]`` silently judges the whole string by its most harmless part.

This module provides the missing layer:

* :func:`analyze_command` splits a command string into :class:`Segment`
  objects -- one per shell-level simple command -- so each is classified on
  its own name, its own arguments and its own redirections.
* Command substitutions (``$(...)``, backticks, ``<(...)``/``>(...)``) are
  extracted and analyzed as segments too, so a command hidden inside a
  substitution is no longer inert text to the analyzer.
* Redirections are parsed with the full file-descriptor grammar
  (``2>f``, ``&>f``, ``>&f``, ``<>f``, ``>|f``, heredocs, here-strings, ...)
  rather than a ``>``/``>>`` regex, and each is tagged with the access mode
  it implies.

**Fail-closed contract.**  Everything here is a security boundary input.
When the lexer meets something it cannot model the way ``/bin/sh`` would --
an unbalanced quote or substitution, a dangling backslash, a redirection
with no target, a chain operator with nothing on one side -- it raises
:exc:`UnanalyzableCommand` rather than guessing.
Callers are expected to refuse such a command, not to degrade to a looser
parse.  :exc:`UnanalyzableCommand` subclasses :exc:`ValueError` so existing
``except ValueError`` handlers around ``shlex.split`` keep working.

The lexer models POSIX ``sh`` word splitting.  It deliberately does *not*
perform expansion: ``$VAR``/``${VAR}``/``$((...))`` are kept as literal text
in the containing word, because their runtime value is unknowable here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = [
    "UnanalyzableCommand",
    "Redirect",
    "Segment",
    "analyze_command",
    "resolve_command_names",
    "WRAPPER_COMMANDS",
]


class UnanalyzableCommand(ValueError):
    """A command could not be modelled the way the shell would parse it.

    Raised by :func:`analyze_command` for unbalanced quotes/substitutions,
    dangling escapes, redirection forms without a target, and dangling
    ``&&``/``||``/``|`` chain operators.  Callers must
    treat this as "deny", never as "analyze less carefully".
    """


# Maximum nesting depth for command substitutions.  A deeper nest is refused
# rather than recursed into: legitimate commands do not need it, and an
# unbounded recursion here is itself a denial-of-service surface.
_MAX_SUBSTITUTION_DEPTH = 8

# ``NAME=value`` prefixes are variable assignments, not the command name.
# Missing this is a documented bypass class: ``FOO=bar rm -rf x`` reads as
# command ``FOO=bar`` to a naive analyzer, so ``rm`` semantics never fire.
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?\+?=")

# Shell reserved words / control tokens that occupy a command slot without
# being a command.
_SHELL_KEYWORDS = frozenset({
    "!", "{", "}", "[[", "]]", "((", "))",
    "if", "then", "elif", "else", "fi",
    "for", "while", "until", "do", "done",
    "case", "esac", "in", "select", "function",
    "time", "coproc",
})

# Commands that run *another* command given in their own arguments.  When one
# of these heads a segment, the analyzer keeps scanning for the real command
# instead of stopping at the wrapper.
WRAPPER_COMMANDS = frozenset({
    "env", "sudo", "doas", "nohup", "command", "builtin", "exec",
    "setsid", "timeout", "xargs", "nice", "ionice", "stdbuf",
    "unbuffer", "strace", "ltrace", "watch", "script",
})

# Redirection operators grouped by the access they grant to their target.
_WRITE_OPS = frozenset({">", ">>", ">|", "&>", "&>>", "<>"})
_READ_OPS = frozenset({"<"})
# Operators whose "target" is not a filesystem path at all: heredoc
# delimiters and here-strings are literal data, not files.
_NON_PATH_OPS = frozenset({"<<", "<<-", "<<<"})
# Duplication operators: ``>&2`` targets a descriptor, ``>&file`` a file.
_DUP_OPS = frozenset({">&", "<&"})

# A ``>&``/``<&`` target that names a descriptor rather than a file.
_FD_TARGET_RE = re.compile(r"^\d+-?$|^-$")

# Operators that terminate a simple command.  Each starts a new segment.
_SEGMENT_OPERATORS = ("&&", "||", ";;", "|&", ";", "|", "&", "\n", "(", ")")

# Operators that require a command on BOTH sides.  A command left dangling on
# either side of one is a shell syntax error, so the analyzer refuses it
# rather than reasoning about half a pipeline.
_CHAIN_OPERATORS = frozenset({"&&", "||", "|", "|&"})


@dataclass(frozen=True)
class Redirect:
    """One parsed redirection.

    Attributes:
        op: The normalised operator text (``>``, ``>>``, ``2>`` is recorded
            as op ``>`` with ``fd='2'``, ``&>``, ``>&``, ``<<-``, ...).
        fd: The explicit file-descriptor prefix, if the source had one
            (``2>file`` -> ``'2'``).  ``None`` when unspecified.
        target: The word following the operator, with quoting removed.
        mode: Access the redirection grants to ``target`` -- ``"write"``,
            ``"read"``, or ``"none"`` when the target is not a filesystem
            path (a heredoc delimiter, here-string body, or descriptor).
    """

    op: str
    fd: Optional[str]
    target: str
    mode: str


@dataclass
class Segment:
    """One shell-level simple command with its own words and redirections.

    A segment is what the shell would execute as a single command: the text
    between two control operators (``&&``, ``||``, ``;``, ``|``, ``&``,
    newline, subshell parens).  Analyzers classify each segment
    independently and union the results, which is the whole point of this
    module -- ``cat x && rm -rf y`` yields a ``cat`` segment and an ``rm``
    segment rather than one ``cat`` command.

    Attributes:
        words: Command name and arguments, quoting removed, redirections
            stripped out.
        redirects: Redirections attached to this segment, in source order.
    """

    words: List[str] = field(default_factory=list)
    redirects: List[Redirect] = field(default_factory=list)

    def is_empty(self) -> bool:
        """True when the segment carries neither words nor redirections."""
        return not self.words and not self.redirects

    @property
    def command_names(self) -> List[str]:
        """Candidate command names for this segment, outermost first.

        See :func:`resolve_command_names` for the resolution rules.
        """
        return resolve_command_names(self.words)

    @property
    def command_name(self) -> str:
        """The innermost resolved command name, or ``''`` for none."""
        names = self.command_names
        return names[-1] if names else ""

    def targets_with_mode(self, mode: str) -> List[str]:
        """Redirection targets granted ``mode`` ("write" or "read")."""
        return [r.target for r in self.redirects if r.mode == mode]


def resolve_command_names(words: List[str]) -> List[str]:
    """Resolve the command name(s) a word list invokes.

    Skips leading variable assignments (``FOO=bar cmd``) and shell reserved
    words, then returns the command name.  When that name is a known wrapper
    (``sudo``, ``env``, ``xargs``, ...) resolution continues past it -- and
    past its option flags -- so the wrapped command is reported too.  The
    list is outermost-first: ``sudo rm -rf x`` yields ``['sudo', 'rm']``.

    Basenames are returned (``/bin/rm`` -> ``rm``) so callers can compare
    against plain command-name sets.

    Args:
        words: Words of a single segment, quoting already removed.

    Returns:
        Resolved command basenames, outermost first.  Empty when the segment
        contains only assignments or reserved words.
    """
    names: List[str] = []
    saw_wrapper = False
    for word in words:
        if _ASSIGNMENT_RE.match(word) or word in _SHELL_KEYWORDS:
            continue
        if saw_wrapper and word.startswith("-"):
            continue
        base = os.path.basename(word)
        if not base:
            continue
        names.append(base)
        if base in WRAPPER_COMMANDS:
            saw_wrapper = True
            continue
        break
    return names


def _redirect_mode(op: str, target: str) -> str:
    """Access mode a redirection operator grants to its target.

    Args:
        op: Normalised operator text.
        target: The redirection target word.

    Returns:
        ``"write"``, ``"read"``, or ``"none"``.

    Raises:
        UnanalyzableCommand: For an operator this module does not model --
            the fail-closed path, so a new bash redirect form can never be
            silently treated as read-only.
    """
    if op in _WRITE_OPS:
        return "write"
    if op in _READ_OPS:
        return "read"
    if op in _NON_PATH_OPS:
        return "none"
    if op in _DUP_OPS:
        if _FD_TARGET_RE.match(target):
            return "none"
        # ``>&file`` is bash's synonym for ``&>file``; ``<&file`` reads it.
        return "write" if op == ">&" else "read"
    raise UnanalyzableCommand(f"unmodelled redirection operator {op!r}")


class _ShellLexer:
    """Character-level lexer producing :class:`Segment` objects.

    Tracks quoting, escapes, command substitution and redirection the way
    POSIX ``sh`` does, so the resulting segmentation matches what will
    actually run.  Substitution *bodies* are collected verbatim in
    :attr:`substitutions` for the caller to analyze recursively.

    Lifecycle: construct with the source string, call :meth:`run` once, then
    read :attr:`segments` and :attr:`substitutions`.  Instances are single
    use.
    """

    def __init__(self, src: str) -> None:
        self.src = src
        self.n = len(src)
        self.i = 0
        self.segments: List[Segment] = []
        self.substitutions: List[str] = []
        self._cur = Segment()
        self._word: List[str] = []
        self._word_started = False
        self._word_quoted = False
        self._pending_redirect: Optional[Tuple[str, Optional[str]]] = None
        self._pending_heredocs: List[Tuple[str, bool]] = []
        self._open_chain: Optional[str] = None

    # --- word accumulation -------------------------------------------------

    def _append(self, text: str) -> None:
        self._word.append(text)
        self._word_started = True

    def _reset_word(self) -> None:
        self._word = []
        self._word_started = False
        self._word_quoted = False

    def _flush_word(self) -> None:
        """Emit the pending word as an argument or a redirection target."""
        if self._pending_redirect is not None:
            if not self._word_started:
                return
            op, fd = self._pending_redirect
            target = "".join(self._word)
            self._pending_redirect = None
            self._reset_word()
            self._cur.redirects.append(
                Redirect(op=op, fd=fd, target=target, mode=_redirect_mode(op, target))
            )
            if op in ("<<", "<<-"):
                self._pending_heredocs.append((target, op == "<<-"))
            return
        if self._word_started:
            self._cur.words.append("".join(self._word))
            self._reset_word()

    def _close_segment(self) -> None:
        self._flush_word()
        if self._pending_redirect is not None:
            op, _fd = self._pending_redirect
            raise UnanalyzableCommand(f"redirection {op!r} without a target")
        if not self._cur.is_empty():
            self.segments.append(self._cur)
            self._open_chain = None
        self._cur = Segment()

    # --- quoting -----------------------------------------------------------

    def _read_escape(self) -> None:
        self.i += 1
        if self.i >= self.n:
            raise UnanalyzableCommand("dangling backslash at end of command")
        char = self.src[self.i]
        self.i += 1
        if char == "\n":
            # Line continuation: the newline disappears, and crucially does
            # NOT act as a command separator.
            return
        self._append(char)

    def _read_single_quoted(self) -> None:
        end = self.src.find("'", self.i + 1)
        if end < 0:
            raise UnanalyzableCommand("unbalanced single quote")
        self._append(self.src[self.i + 1:end])
        self._word_quoted = True
        self.i = end + 1

    def _read_double_quoted(self) -> None:
        self.i += 1
        self._word_quoted = True
        self._word_started = True
        while self.i < self.n:
            char = self.src[self.i]
            if char == '"':
                self.i += 1
                return
            if char == "\\":
                self._read_double_quoted_escape()
            elif char == "`":
                self._read_backtick()
            elif char == "$":
                self._read_dollar()
            else:
                self._append(char)
                self.i += 1
        raise UnanalyzableCommand("unbalanced double quote")

    def _read_double_quoted_escape(self) -> None:
        self.i += 1
        if self.i >= self.n:
            raise UnanalyzableCommand("dangling backslash inside double quotes")
        char = self.src[self.i]
        self.i += 1
        if char == "\n":
            return
        if char in '\\"$`':
            self._append(char)
        else:
            self._append("\\" + char)

    # --- substitutions -----------------------------------------------------

    def _read_backtick(self) -> None:
        end = self.src.find("`", self.i + 1)
        if end < 0:
            raise UnanalyzableCommand("unbalanced backtick substitution")
        self.substitutions.append(self.src[self.i + 1:end])
        self._word_started = True
        self.i = end + 1

    def _read_dollar(self) -> None:
        """Handle ``$``: substitution, expansion, or a literal dollar."""
        if self.src.startswith("$((", self.i):
            end = self._scan_balanced(self.i + 1, "(", ")")
            self._append(self.src[self.i:end])
            self.i = end
            return
        nxt = self.src[self.i + 1:self.i + 2]
        if nxt == "(":
            end = self._scan_balanced(self.i + 1, "(", ")")
            self.substitutions.append(self.src[self.i + 2:end - 1])
            self._word_started = True
            self.i = end
            return
        if nxt == "{":
            end = self._scan_balanced(self.i + 1, "{", "}")
            self._append(self.src[self.i:end])
            self.i = end
            return
        self._append("$")
        self.i += 1

    def _scan_balanced(self, start: int, opener: str, closer: str) -> int:
        """Return the index just past the ``closer`` matching ``src[start]``.

        Quoting inside the construct is honoured so that ``$(echo ")")``
        does not terminate early.

        Raises:
            UnanalyzableCommand: If the construct is never closed.
        """
        depth = 0
        idx = start
        while idx < self.n:
            char = self.src[idx]
            if char == "\\":
                idx += 2
                continue
            if char in "'\"":
                idx = self._skip_quoted(idx)
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return idx + 1
            idx += 1
        raise UnanalyzableCommand(f"unbalanced {opener!r} in command substitution")

    def _skip_quoted(self, idx: int) -> int:
        """Skip a quoted run starting at ``idx``; return the index past it."""
        quote = self.src[idx]
        idx += 1
        while idx < self.n:
            char = self.src[idx]
            if char == quote:
                return idx + 1
            if char == "\\" and quote == '"':
                idx += 2
                continue
            idx += 1
        raise UnanalyzableCommand(f"unbalanced {quote} inside substitution")

    # --- operators ---------------------------------------------------------

    def _read_operator(self) -> None:
        """Consume a control operator and start a new segment."""
        two = self.src[self.i:self.i + 2]
        op = two if two in ("&&", "||", ";;", "|&") else self.src[self.i]
        if op in _CHAIN_OPERATORS and self._cur.is_empty() and not self._word_started:
            raise UnanalyzableCommand(
                f"chain operator {op!r} without a preceding command"
            )
        self._close_segment()
        self.i += len(op)
        if op in _CHAIN_OPERATORS:
            self._open_chain = op
        if op == "\n":
            self._consume_heredocs()

    def _consume_heredocs(self) -> None:
        """Skip heredoc bodies queued by ``<<``/``<<-`` on the previous line.

        Heredoc content is data, not commands; lexing it as commands would
        invent segments (and path tokens) that never execute.
        """
        while self._pending_heredocs:
            delimiter, strip_tabs = self._pending_heredocs.pop(0)
            self.i = self._skip_heredoc_body(delimiter, strip_tabs)

    def _skip_heredoc_body(self, delimiter: str, strip_tabs: bool) -> int:
        idx = self.i
        while idx < self.n:
            end = self.src.find("\n", idx)
            line = self.src[idx:end if end >= 0 else self.n]
            candidate = line.lstrip("\t") if strip_tabs else line
            idx = self.n if end < 0 else end + 1
            if candidate == delimiter:
                break
        return idx

    # --- redirections ------------------------------------------------------

    def _read_redirect(self) -> None:
        """Parse a redirection operator and arm capture of its target."""
        fd = self._take_fd_prefix()
        op = self._read_redirect_op()
        if op is None:
            return  # process substitution: consumed as a word, not a redirect
        self._flush_word()
        if self._pending_redirect is not None:
            raise UnanalyzableCommand("redirection operator without a target")
        self._pending_redirect = (op, fd)

    def _take_fd_prefix(self) -> Optional[str]:
        """Detach a bare-digit pending word as the redirection's fd number.

        ``cmd 2>f`` has fd ``2``; ``echo foo2>f`` does not -- there the word
        is ``foo2`` and the redirection is plain ``>``.
        """
        word = "".join(self._word)
        if self._word_started and not self._word_quoted and word.isdigit():
            self._reset_word()
            return word
        return None

    def _read_redirect_op(self) -> Optional[str]:
        """Read the operator text at the cursor.

        Returns:
            The normalised operator, or ``None`` when the construct turned
            out to be a process substitution (``<(...)`` / ``>(...)``),
            which is a word rather than a redirection.
        """
        char = self.src[self.i]
        if char == "&":
            self.i += 2  # '&' and the '>' that _read_operator dispatched on
            if self.src[self.i:self.i + 1] == ">":
                self.i += 1
                return "&>>"
            return "&>"
        self.i += 1
        nxt = self.src[self.i:self.i + 1]
        if nxt == "(":
            end = self._scan_balanced(self.i, "(", ")")
            self.substitutions.append(self.src[self.i + 1:end - 1])
            # The construct occupies a word slot (bash substitutes a
            # /dev/fd path at runtime), but its literal text is not a path
            # the caller can check, so the word is left empty.
            self._word_started = True
            self.i = end
            return None
        return self._finish_redirect_op(char, nxt)

    def _finish_redirect_op(self, char: str, nxt: str) -> str:
        if char == ">":
            if nxt in (">", "|", "&"):
                self.i += 1
                return ">" + nxt
            return ">"
        if nxt == "<":
            self.i += 1
            tail = self.src[self.i:self.i + 1]
            if tail in ("<", "-"):
                self.i += 1
                return "<<" + tail
            return "<<"
        if nxt in ("&", ">"):
            self.i += 1
            return "<" + nxt
        return "<"

    # --- driver ------------------------------------------------------------

    def run(self) -> None:
        """Lex the whole command string into :attr:`segments`."""
        while self.i < self.n:
            char = self.src[self.i]
            if char == "\\":
                self._read_escape()
            elif char == "'":
                self._read_single_quoted()
            elif char == '"':
                self._read_double_quoted()
            elif char == "`":
                self._read_backtick()
            elif char == "$":
                self._read_dollar()
            elif char in "<>" or (char == "&" and self.src[self.i + 1:self.i + 2] == ">"):
                self._read_redirect()
            elif char in ";|&\n()":
                self._read_operator()
            elif char.isspace():
                self._flush_word()
                self.i += 1
            else:
                self._append(char)
                self.i += 1
        self._close_segment()
        if self._open_chain is not None:
            raise UnanalyzableCommand(
                f"chain operator {self._open_chain!r} without a following command"
            )


def analyze_command(command: str, _depth: int = 0) -> List[Segment]:
    """Split ``command`` into the simple commands the shell would run.

    Command substitutions are analyzed recursively and their segments are
    appended to the result, so a command hidden in ``$(...)`` or backticks
    is visible to callers instead of passing as inert text.

    Args:
        command: The raw shell command string.
        _depth: Internal recursion depth for substitution bodies.

    Returns:
        Non-empty segments in source order, substitution bodies last.

    Raises:
        UnanalyzableCommand: When the string cannot be modelled the way the
            shell would parse it.  Callers must deny rather than retry with
            a looser parser.
    """
    if _depth > _MAX_SUBSTITUTION_DEPTH:
        raise UnanalyzableCommand("command substitutions nested too deeply")

    lexer = _ShellLexer(command)
    lexer.run()
    segments = list(lexer.segments)
    for body in lexer.substitutions:
        segments.extend(analyze_command(body, _depth + 1))
    return segments
