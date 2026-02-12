"""Gitignore pattern matching utility.

Provides a simple .gitignore pattern parser that checks whether file paths
should be ignored based on patterns in a workspace's .gitignore file.

Supports:
- Glob patterns (*, ?, [...])
- Directory-only patterns (trailing /)
- Negation patterns (leading !)
- Full-path and basename matching
- Nested .gitignore is NOT supported (only root .gitignore)
"""

import fnmatch
import os
from pathlib import Path
from typing import List, Optional, Set, Tuple


class GitignoreParser:
    """Simple .gitignore pattern parser.

    Loads patterns from a .gitignore file at the workspace root and provides
    an ``is_ignored(path)`` check for individual files/directories.

    Additionally supports a hardcoded set of default ignore patterns
    (e.g., .git/, __pycache__/, .venv/) that are always applied even when
    no .gitignore exists. Pass ``include_defaults=True`` (the default) to
    enable them, or ``False`` to rely solely on the .gitignore file.
    """

    # Patterns that are always ignored regardless of .gitignore content.
    DEFAULT_IGNORE_PATTERNS: List[str] = [
        ".git/",
        "__pycache__/",
        ".venv/",
        "venv/",
        "node_modules/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".jaato/",
        "*.pyc",
        "*.pyo",
        ".DS_Store",
        "*.swp",
        "*.swo",
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

    def _load_gitignore(self) -> None:
        """Load patterns from .gitignore file."""
        gitignore_path = self._workspace_root / ".gitignore"
        if not gitignore_path.exists():
            return

        try:
            content = gitignore_path.read_text()
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

    def is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored.

        Args:
            path: Path to check (absolute or relative to workspace_root).

        Returns:
            True if the path should be ignored.
        """
        # Make path relative to workspace
        try:
            rel_path = path.relative_to(self._workspace_root)
        except ValueError:
            rel_path = path

        path_str = str(rel_path)
        path_parts = path_str.split(os.sep)

        ignored = False
        for pattern, is_negation in self._patterns:
            # Handle directory-only patterns (ending with /)
            if pattern.endswith("/"):
                pattern = pattern[:-1]
                # Only match if it's a directory
                if not path.is_dir():
                    # Check if any parent matches
                    matches = any(
                        fnmatch.fnmatch(part, pattern)
                        for part in path_parts[:-1]
                    )
                else:
                    matches = fnmatch.fnmatch(path_parts[-1], pattern)
            else:
                # Match against full path or just filename
                if "/" in pattern or "\\" in pattern:
                    matches = fnmatch.fnmatch(path_str, pattern)
                else:
                    matches = (
                        fnmatch.fnmatch(path_str, pattern)
                        or fnmatch.fnmatch(path_parts[-1], pattern)
                        or any(fnmatch.fnmatch(part, pattern) for part in path_parts)
                    )

            if matches:
                ignored = not is_negation

        return ignored

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
