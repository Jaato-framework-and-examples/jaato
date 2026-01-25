"""Find and replace across multiple files.

Provides regex-based find and replace functionality with glob pattern
support for file selection, .gitignore respect, and dry-run preview.

Example usage:
    executor = FindReplaceExecutor(workspace_root, resolve_path_fn, is_path_allowed_fn)
    result = executor.execute(
        pattern=r"old_function_name",
        replacement="new_function_name",
        paths="src/**/*.py",
        dry_run=False
    )
"""

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from shared.ui_utils import ellipsize_path

# Default maximum width for file paths in find/replace previews
DEFAULT_MAX_PATH_WIDTH = 50


@dataclass
class FileMatch:
    """Represents matches found in a single file."""
    path: str
    matches: List[Tuple[int, str, str]]  # (line_number, original_line, replaced_line)
    match_count: int


@dataclass
class FindReplaceResult:
    """Result of a find and replace operation."""
    success: bool
    pattern: str
    replacement: str
    paths_pattern: str
    dry_run: bool
    total_matches: int
    files_affected: int
    files_searched: int
    file_matches: List[FileMatch] = field(default_factory=list)
    error: Optional[str] = None
    rollback_data: Optional[Dict[str, bytes]] = None  # path -> original content (for undo)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for tool response."""
        result = {
            "success": self.success,
            "pattern": self.pattern,
            "replacement": self.replacement,
            "paths_pattern": self.paths_pattern,
            "dry_run": self.dry_run,
            "total_matches": self.total_matches,
            "files_affected": self.files_affected,
            "files_searched": self.files_searched,
        }

        if self.file_matches:
            result["matches_by_file"] = [
                {
                    "path": fm.path,
                    "match_count": fm.match_count,
                    "preview": [
                        {
                            "line": line_num,
                            "before": before,
                            "after": after
                        }
                        for line_num, before, after in fm.matches[:10]  # Limit preview
                    ]
                }
                for fm in self.file_matches
            ]
            if any(len(fm.matches) > 10 for fm in self.file_matches):
                result["preview_truncated"] = True

        if self.error:
            result["error"] = self.error

        return result


class GitignoreParser:
    """Simple .gitignore pattern parser."""

    def __init__(self, workspace_root: Path):
        """Initialize with workspace root.

        Args:
            workspace_root: Root directory for finding .gitignore files
        """
        self._workspace_root = workspace_root
        self._patterns: List[Tuple[str, bool]] = []  # (pattern, is_negation)
        self._load_gitignore()

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
            path: Path to check (relative to workspace_root)

        Returns:
            True if the path should be ignored
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
                        fnmatch.fnmatch(path_str, pattern) or
                        fnmatch.fnmatch(path_parts[-1], pattern) or
                        any(fnmatch.fnmatch(part, pattern) for part in path_parts)
                    )

            if matches:
                ignored = not is_negation

        return ignored


class FindReplaceExecutor:
    """Executes find and replace operations across multiple files."""

    # Patterns to always ignore (binary files, etc.)
    ALWAYS_IGNORE_PATTERNS = [
        "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll", "*.exe",
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg",
        "*.pdf", "*.zip", "*.tar", "*.gz", "*.bz2",
        "*.woff", "*.woff2", "*.ttf", "*.eot",
        "*.min.js", "*.min.css",
        ".git/*", ".svn/*", ".hg/*",
        "node_modules/*", "__pycache__/*", ".venv/*", "venv/*",
        ".jaato/*",
    ]

    def __init__(
        self,
        workspace_root: Optional[Path],
        resolve_path_fn: Callable[[str], Path],
        is_path_allowed_fn: Callable[[str], bool],
        backup_fn: Optional[Callable[[Path], Optional[Path]]] = None,
        trace_fn: Optional[Callable[[str], None]] = None
    ):
        """Initialize the find/replace executor.

        Args:
            workspace_root: Root directory for file searches
            resolve_path_fn: Function to resolve relative paths
            is_path_allowed_fn: Function to check sandbox permissions
            backup_fn: Optional function to create backups before changes
            trace_fn: Optional function for debug tracing
        """
        self._workspace_root = workspace_root
        self._resolve_path = resolve_path_fn
        self._is_path_allowed = is_path_allowed_fn
        self._backup_fn = backup_fn
        self._trace = trace_fn or (lambda msg: None)

    def _should_ignore(
        self,
        path: Path,
        gitignore: Optional[GitignoreParser],
        include_ignored: bool
    ) -> bool:
        """Check if a file should be ignored.

        Args:
            path: Path to check
            gitignore: GitignoreParser instance
            include_ignored: If True, skip gitignore check

        Returns:
            True if the file should be ignored
        """
        path_str = str(path)

        # Always check our built-in ignore patterns
        for pattern in self.ALWAYS_IGNORE_PATTERNS:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True

        # Check gitignore unless include_ignored is True
        if not include_ignored and gitignore and gitignore.is_ignored(path):
            return True

        return False

    def _find_files(
        self,
        paths_pattern: str,
        include_ignored: bool = False
    ) -> List[Path]:
        """Find files matching the glob pattern.

        Args:
            paths_pattern: Glob pattern for files (e.g., "src/**/*.py")
            include_ignored: If True, include files normally ignored by .gitignore

        Returns:
            List of matching file paths
        """
        if not self._workspace_root:
            # No workspace root, can't do glob search
            return []

        workspace = Path(self._workspace_root)

        # Initialize gitignore parser
        gitignore = GitignoreParser(workspace) if not include_ignored else None

        # Find matching files
        matched_files: List[Path] = []
        for path in workspace.glob(paths_pattern):
            if not path.is_file():
                continue

            # Check if file is allowed by sandbox
            if not self._is_path_allowed(str(path)):
                continue

            # Check if file should be ignored
            if self._should_ignore(path, gitignore, include_ignored):
                continue

            matched_files.append(path)

        return sorted(matched_files)

    def _find_matches_in_file(
        self,
        file_path: Path,
        pattern: re.Pattern,
        replacement: str
    ) -> Optional[FileMatch]:
        """Find all matches in a single file.

        Args:
            file_path: Path to the file
            pattern: Compiled regex pattern
            replacement: Replacement string

        Returns:
            FileMatch if there are matches, None otherwise
        """
        try:
            content = file_path.read_text()
        except (OSError, UnicodeDecodeError):
            # Skip files we can't read
            return None

        lines = content.splitlines()
        matches: List[Tuple[int, str, str]] = []

        for i, line in enumerate(lines, start=1):
            if pattern.search(line):
                replaced_line = pattern.sub(replacement, line)
                if replaced_line != line:
                    matches.append((i, line, replaced_line))

        if matches:
            return FileMatch(
                path=str(file_path),
                matches=matches,
                match_count=len(matches)
            )

        return None

    def execute(
        self,
        pattern: str,
        replacement: str,
        paths: str,
        dry_run: bool = False,
        include_ignored: bool = False
    ) -> FindReplaceResult:
        """Execute find and replace across files.

        Args:
            pattern: Regex pattern to search for
            replacement: Replacement string (supports backreferences like \\1)
            paths: Glob pattern for files to search
            dry_run: If True, only preview changes without applying
            include_ignored: If True, include files normally ignored by .gitignore

        Returns:
            FindReplaceResult with details of the operation
        """
        self._trace(f"find_and_replace: pattern='{pattern}', paths='{paths}', dry_run={dry_run}")

        # Compile the regex pattern
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as e:
            return FindReplaceResult(
                success=False,
                pattern=pattern,
                replacement=replacement,
                paths_pattern=paths,
                dry_run=dry_run,
                total_matches=0,
                files_affected=0,
                files_searched=0,
                error=f"Invalid regex pattern: {e}"
            )

        # Find matching files
        files = self._find_files(paths, include_ignored)
        self._trace(f"find_and_replace: found {len(files)} files to search")

        # Find matches in each file
        file_matches: List[FileMatch] = []
        for file_path in files:
            match = self._find_matches_in_file(file_path, compiled_pattern, replacement)
            if match:
                file_matches.append(match)

        total_matches = sum(fm.match_count for fm in file_matches)
        self._trace(f"find_and_replace: found {total_matches} matches in {len(file_matches)} files")

        result = FindReplaceResult(
            success=True,
            pattern=pattern,
            replacement=replacement,
            paths_pattern=paths,
            dry_run=dry_run,
            total_matches=total_matches,
            files_affected=len(file_matches),
            files_searched=len(files),
            file_matches=file_matches
        )

        # If dry run, return preview without applying
        if dry_run:
            return result

        # Apply changes
        if file_matches:
            rollback_data: Dict[str, bytes] = {}

            try:
                for fm in file_matches:
                    file_path = Path(fm.path)

                    # Save original content for rollback
                    original_content = file_path.read_bytes()
                    rollback_data[fm.path] = original_content

                    # Create backup if backup function is available
                    if self._backup_fn:
                        self._backup_fn(file_path)

                    # Apply replacement
                    content = file_path.read_text()
                    new_content = compiled_pattern.sub(replacement, content)
                    file_path.write_text(new_content)

                    self._trace(f"find_and_replace: updated {fm.path}")

                result.rollback_data = rollback_data

            except OSError as e:
                # Rollback on failure
                result.success = False
                result.error = f"Failed to write file: {e}"

                self._trace(f"find_and_replace: error, rolling back")
                for path, content in rollback_data.items():
                    try:
                        Path(path).write_bytes(content)
                    except OSError:
                        pass  # Best effort rollback

        return result


def generate_find_replace_preview(
    file_matches: List[FileMatch],
    max_matches_per_file: int = 5,
    max_files: int = 10
) -> Tuple[str, bool]:
    """Generate a preview of find/replace changes.

    Args:
        file_matches: List of FileMatch objects
        max_matches_per_file: Max matches to show per file
        max_files: Max files to show in preview

    Returns:
        Tuple of (preview_text, truncated)
    """
    lines = []
    truncated = False

    displayed_files = file_matches[:max_files]
    if len(file_matches) > max_files:
        truncated = True

    for fm in displayed_files:
        display_path = ellipsize_path(fm.path, DEFAULT_MAX_PATH_WIDTH)
        lines.append(f"File: {display_path} ({fm.match_count} matches)\n")

        displayed_matches = fm.matches[:max_matches_per_file]
        if len(fm.matches) > max_matches_per_file:
            truncated = True

        for line_num, before, after in displayed_matches:
            lines.append(f"  Line {line_num}:\n")
            lines.append(f"    - {before}\n")
            lines.append(f"    + {after}\n")

        if len(fm.matches) > max_matches_per_file:
            lines.append(f"  ... and {len(fm.matches) - max_matches_per_file} more matches\n")

        lines.append("\n")

    if len(file_matches) > max_files:
        lines.append(f"... and {len(file_matches) - max_files} more files\n")

    return "".join(lines), truncated
