"""Gitignore pattern matching utility.

Provides a .gitignore pattern parser that checks whether file paths should
be ignored based on patterns in a workspace's .gitignore file, following
git's own rules closely enough that the answer can be trusted where git
itself is not available — the workspace monitor's file panel, and
``jaato-scaffold validate`` judging a ``.gitignore`` by its effect.

Supports:
- Glob patterns (``*``, ``?``, ``[...]``), where ``*`` and ``?`` never
  match ``/`` and ``**`` does — as in git, so ``.jaato/*`` names the
  direct children of ``.jaato`` and nothing deeper
- Directory-only patterns (trailing ``/``), which match a DIRECTORY and,
  through it, everything beneath
- Anchoring: a pattern with a ``/`` anywhere but its end is matched
  against the whole workspace-relative path (a leading ``/`` is the
  explicit spelling); one without matches the basename at any depth
- Negation patterns (leading ``!``), last match wins
- Git's one asymmetry: a path beneath an EXCLUDED directory is excluded
  whatever later rules say, because git never descends into it.  So
  ``.jaato/`` followed by ``!.jaato/profiles/`` hides the profiles, and
  ``!.jaato/`` + ``.jaato/*`` + ``!.jaato/profiles/`` shows them
- Nested .gitignore is NOT supported (only root .gitignore)
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple


@dataclass(frozen=True)
class _Rule:
    """One compiled pattern.

    Attributes:
        regex: Matched against the whole workspace-relative POSIX path.
        dir_only: The pattern ended in ``/`` — it matches directories only.
        negation: The pattern began with ``!`` — a match UN-ignores.
    """

    regex: "re.Pattern"
    dir_only: bool
    negation: bool


def _glob_to_regex(pattern: str, anchored: bool) -> "re.Pattern":
    """Translate one gitignore glob into a regex over a POSIX relative path.

    ``*`` and ``?`` stop at ``/``; ``**/`` is any number of directories
    (including none); a trailing ``/**`` or a bare ``**`` is anything;
    ``[...]`` classes pass through (``[!a]`` becomes ``[^a]``).  An
    unanchored pattern may match at any depth, so it is preceded by
    ``(?:^|.*/)``.
    """
    out: List[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if pattern.startswith("**", i):
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
            else:
                out.append(".*")
                i += 2
            continue
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[" and pattern.find("]", i + 1) != -1:
            j = pattern.index("]", i + 1)
            cls = pattern[i + 1:j]
            if cls.startswith("!"):
                cls = "^" + cls[1:]
            out.append("[" + cls + "]")
            i = j + 1
            continue
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile(("^" if anchored else "(?:^|.*/)") + "".join(out) + "$")


def _compile(pattern: str, negation: bool) -> _Rule:
    """Compile one pattern as it appears in a ``.gitignore`` line (its
    leading ``!`` already stripped into *negation*)."""
    dir_only = pattern.endswith("/")
    body = pattern.rstrip("/")
    anchored = body.startswith("/") or "/" in body
    body = body.lstrip("/")
    return _Rule(_glob_to_regex(body, anchored), dir_only, negation)


class GitignoreParser:
    """.gitignore pattern parser.

    Loads patterns from a .gitignore file at the workspace root and provides
    an ``is_ignored(path)`` check for individual files/directories.

    Additionally supports a hardcoded set of default ignore patterns
    (e.g., .git/) that are always applied even when no .gitignore exists.
    Pass ``include_defaults=True`` (the default) to enable them, or
    ``False`` to rely solely on the .gitignore file.
    """

    # ``.git/`` is the only hardcoded default — surfacing the git internals
    # in the workspace panel would drown every other change in noise, and no
    # workspace would ever want to track them.  Everything else
    # (``__pycache__/``, ``node_modules/``, ``.venv/``, …) is left to the
    # workspace's own ``.gitignore`` so users have full control.
    DEFAULT_IGNORE_PATTERNS: List[str] = [
        ".git/",
    ]

    def __init__(
        self,
        workspace_root: Path,
        include_defaults: bool = True,
        extra_patterns: Optional[List[str]] = None,
    ):
        """Initialize with workspace root.

        Args:
            workspace_root: Root directory for finding .gitignore files.
            include_defaults: Whether to prepend DEFAULT_IGNORE_PATTERNS.
            extra_patterns: Additional ignore patterns to append.
        """
        self._workspace_root = workspace_root
        self._patterns: List[Tuple[str, bool]] = []  # (pattern, is_negation)

        # Load default ignore patterns first (lowest priority – can be negated
        # by .gitignore entries).
        if include_defaults:
            for pat in self.DEFAULT_IGNORE_PATTERNS:
                self._patterns.append((pat, False))

        # Load .gitignore
        self._load_gitignore()

        # Append extra patterns (highest priority).
        if extra_patterns:
            for pat in extra_patterns:
                is_negation = pat.startswith("!")
                if is_negation:
                    pat = pat[1:]
                self._patterns.append((pat, is_negation))

        self._rules: List[_Rule] = [_compile(p, neg) for p, neg in self._patterns]

    def _load_gitignore(self) -> None:
        """Load patterns from .gitignore file."""
        gitignore_path = self._workspace_root / ".gitignore"
        if not gitignore_path.exists():
            return

        try:
            content = gitignore_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Handle negation
                is_negation = line.startswith("!")
                if is_negation:
                    line = line[1:]

                self._patterns.append((line, is_negation))
        except OSError:
            pass

    def _verdict(self, rel: str, is_dir: bool) -> bool:
        """Last matching rule's verdict for ONE path, ancestors aside."""
        ignored = False
        for rule in self._rules:
            if rule.dir_only and not is_dir:
                continue
            if rule.regex.match(rel):
                ignored = not rule.negation
        return ignored

    def is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored.

        Git's answer: the path is ignored if any ancestor DIRECTORY is
        ignored (git never descends into an excluded directory, so no rule
        can re-include beneath one), else by the last rule matching the
        path itself.  Whether the path is a directory is read from the
        filesystem, so a path that does not exist is judged as a file —
        which is what a caller probing "would a file HERE be ignored" wants;
        its ancestors are directories by construction.

        Args:
            path: Path to check (absolute or relative to workspace_root).

        Returns:
            True if the path should be ignored.
        """
        try:
            rel_path = path.relative_to(self._workspace_root)
        except ValueError:
            rel_path = path

        parts = [p for p in str(rel_path).replace("\\", "/").split("/") if p]
        for depth in range(1, len(parts)):
            if self._verdict("/".join(parts[:depth]), is_dir=True):
                return True
        return self._verdict("/".join(parts), is_dir=path.is_dir())

    def filter_paths(self, paths: Set[str]) -> Set[str]:
        """Return only paths that are NOT ignored.

        Args:
            paths: Set of relative path strings.

        Returns:
            Subset of *paths* that pass the ignore filter.
        """
        result: Set[str] = set()
        for p in paths:
            full = self._workspace_root / p
            if not self.is_ignored(full):
                result.add(p)
        return result
