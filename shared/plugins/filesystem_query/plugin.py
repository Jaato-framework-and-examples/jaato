"""Filesystem Query Plugin - Read-only tools for exploring the filesystem.

This plugin provides fast, safe, auto-approved tools for:
- Finding files with glob patterns (glob_files)
- Searching file contents with regex (grep_content)

Both tools return structured JSON output and support background execution
for large searches.
"""

import logging
import os
import re
import stat
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..background.mixin import BackgroundCapableMixin
from ..base import UserCommand
from ..model_provider.types import ToolSchema
from .config_loader import (
    FilesystemQueryConfig,
    load_config,
    DEFAULT_MAX_RESULTS,
    DEFAULT_MAX_FILE_SIZE_KB,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_CONTEXT_LINES,
)

logger = logging.getLogger(__name__)


class FilesystemQueryPlugin(BackgroundCapableMixin):
    """Plugin providing filesystem query tools (glob and grep).

    This plugin offers read-only, auto-approved tools for exploring codebases:
    - glob_files: Find files matching glob patterns
    - grep_content: Search file contents using regex

    Both tools support:
    - Configurable exclusion patterns
    - Result limits to prevent overwhelming output
    - Background execution for large searches
    - Structured JSON responses
    """

    def __init__(self):
        """Initialize the filesystem query plugin."""
        super().__init__(max_workers=2, default_timeout=300.0)
        self._config: Optional[FilesystemQueryConfig] = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "filesystem_query"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dict to override defaults.
        """
        self._config = load_config(runtime_config=config)
        self._initialized = True
        logger.info(
            "FilesystemQueryPlugin initialized (max_results=%d, timeout=%ds)",
            self._config.max_results,
            self._config.timeout_seconds,
        )

    def shutdown(self) -> None:
        """Shutdown the plugin and release resources."""
        self._shutdown_bg_executor()
        self._initialized = False
        logger.info("FilesystemQueryPlugin shutdown")

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the tool schemas for glob_files and grep_content."""
        return [
            ToolSchema(
                name="glob_files",
                description=(
                    "Find files matching a glob pattern (e.g., '**/*.py', 'src/**/*.ts'). "
                    "ALWAYS use this instead of `find` or `ls` CLI commands. "
                    "Returns structured JSON with file paths and metadata. "
                    "Automatically excludes node_modules, __pycache__, .git, etc."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Glob pattern to match files. Examples: "
                                "'**/*.py' (all Python files), "
                                "'src/**/*.ts' (TypeScript in src), "
                                "'*.json' (JSON in current dir), "
                                "'**/test_*.py' (test files)"
                            ),
                        },
                        "root": {
                            "type": "string",
                            "description": (
                                "Root directory to search from. "
                                "Defaults to current working directory."
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "description": (
                                f"Maximum files to return (default: {DEFAULT_MAX_RESULTS}). "
                                "Use smaller values for faster results."
                            ),
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "description": (
                                "Include hidden files/directories (starting with '.'). "
                                "Default: false"
                            ),
                        },
                    },
                    "required": ["pattern"],
                },
                category="search",
                discoverability="core",
            ),
            ToolSchema(
                name="grep_content",
                description=(
                    "Search file contents using a regular expression pattern. "
                    "ALWAYS use this instead of `grep`, `rg`, or `ack` CLI commands. "
                    "Returns structured JSON with file paths, line numbers, and context. "
                    "Automatically excludes node_modules, __pycache__, .git, and binary files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Regular expression pattern to search for. Examples: "
                                "'def\\s+my_function' (function definition), "
                                "'TODO|FIXME' (comments), "
                                "'import\\s+requests' (imports)"
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "File or directory to search in. "
                                "Defaults to current working directory."
                            ),
                        },
                        "file_glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Only search files matching these glob patterns. "
                                "Examples: ['*.py'], ['**/*.java', '**/*.kt', '**/*.scala']"
                            ),
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": (
                                f"Lines of context before and after match "
                                f"(default: {DEFAULT_CONTEXT_LINES})"
                            ),
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Case-sensitive search (default: true)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": (
                                f"Maximum matches to return (default: {DEFAULT_MAX_RESULTS})"
                            ),
                        },
                    },
                    "required": ["pattern"],
                },
                category="search",
                discoverability="core",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor functions for each tool."""
        return {
            "glob_files": self._execute_glob_files,
            "grep_content": self._execute_grep_content,
        }

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that don't require permission (read-only operations)."""
        return ["glob_files", "grep_content"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands (none for this plugin)."""
        return []

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the model."""
        return """You have access to filesystem query tools for exploring the codebase.

IMPORTANT: Always prefer these tools over CLI commands:
- Use `glob_files` instead of `find`, `ls -R`, or shell globbing
- Use `grep_content` instead of `grep`, `rg`, or `ack`

These tools provide structured JSON output, automatic exclusion of noise directories
(node_modules, __pycache__, .git, etc.), and don't require shell escaping.

## glob_files
Find files by name pattern. Use glob syntax:
- `**/*.py` - all Python files recursively
- `src/**/*.ts` - TypeScript files in src/
- `**/test_*.py` - test files anywhere
- `*.json` - JSON files in current directory

## grep_content
Search file contents with regex:
- `def\\s+function_name` - find function definitions
- `class\\s+ClassName` - find class definitions
- `import\\s+module` - find imports
- `TODO|FIXME|HACK` - find code comments

The file_glob parameter accepts an array of glob patterns:
- Single type: `file_glob=["*.py"]`
- Multiple types: `file_glob=["**/*.java", "**/*.kt", "**/*.scala"]`

Tips:
- Use glob_files first to locate files, then grep_content to search within them
- Use file_glob parameter in grep_content to limit search to specific file types
- Results are limited to prevent overwhelming output; adjust max_results if needed
"""

    # --- BackgroundCapable overrides ---

    def supports_background(self, tool_name: str) -> bool:
        """Check if a tool supports background execution."""
        return tool_name in ("glob_files", "grep_content")

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        """Return timeout threshold for automatic backgrounding."""
        if self._config:
            return float(self._config.timeout_seconds)
        return float(DEFAULT_TIMEOUT_SECONDS)

    def estimate_duration(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate execution duration based on arguments."""
        # Rough estimates based on typical usage
        if tool_name == "glob_files":
            # Large patterns like "**/*" take longer
            pattern = arguments.get("pattern", "")
            if "**" in pattern:
                return 5.0  # Recursive search
            return 1.0
        elif tool_name == "grep_content":
            # Searching directories is slower than single files
            path = arguments.get("path", ".")
            if Path(path).is_dir() if Path(path).exists() else True:
                return 10.0  # Directory search
            return 2.0
        return None

    # --- Tool implementations ---

    def _execute_glob_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the glob_files tool.

        Args:
            args: Tool arguments containing pattern, root, max_results, etc.

        Returns:
            Dict with files list, total count, and metadata.
        """
        if not self._config:
            self._config = load_config()

        pattern = args.get("pattern", "")
        root = args.get("root", os.getcwd())
        max_results = args.get("max_results", self._config.max_results)
        include_hidden = args.get("include_hidden", False)

        if not pattern:
            return {"error": "Pattern is required", "files": [], "total": 0}

        root_path = Path(root).resolve()
        if not root_path.exists():
            return {
                "error": f"Root path does not exist: {root}",
                "files": [],
                "total": 0,
            }

        if not root_path.is_dir():
            return {
                "error": f"Root path is not a directory: {root}",
                "files": [],
                "total": 0,
            }

        try:
            files: List[Dict[str, Any]] = []
            total_found = 0
            truncated = False

            for match in root_path.glob(pattern):
                # Skip directories
                if match.is_dir():
                    continue

                # Skip hidden files unless requested
                if not include_hidden:
                    if any(part.startswith(".") for part in match.relative_to(root_path).parts):
                        continue

                # Check exclusions
                rel_path = str(match.relative_to(root_path))
                if self._config.should_exclude(rel_path):
                    continue

                total_found += 1

                if len(files) >= max_results:
                    truncated = True
                    continue  # Keep counting but don't add more

                # Get file info
                try:
                    stat_info = match.stat()
                    file_info = {
                        "path": rel_path,
                        "absolute_path": str(match),
                        "size": stat_info.st_size,
                        "modified": datetime.fromtimestamp(
                            stat_info.st_mtime
                        ).isoformat(),
                    }
                    files.append(file_info)
                except (OSError, PermissionError) as e:
                    logger.debug("Could not stat file %s: %s", match, e)
                    # Still include the path even if we can't get stats
                    files.append({
                        "path": rel_path,
                        "absolute_path": str(match),
                        "size": None,
                        "modified": None,
                        "error": str(e),
                    })

            # Sort by modification time (newest first) if we have the data
            files.sort(
                key=lambda f: f.get("modified") or "",
                reverse=True,
            )

            return {
                "files": files,
                "total": total_found,
                "returned": len(files),
                "truncated": truncated,
                "root": str(root_path),
                "pattern": pattern,
            }

        except Exception as e:
            logger.exception("Error in glob_files: %s", e)
            return {
                "error": f"Search failed: {str(e)}",
                "files": [],
                "total": 0,
            }

    def _execute_grep_content(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the grep_content tool.

        Args:
            args: Tool arguments containing pattern, path, file_glob, etc.

        Returns:
            Dict with matches list, counts, and metadata.
        """
        if not self._config:
            self._config = load_config()

        pattern = args.get("pattern", "")
        path = args.get("path", os.getcwd())
        file_glob = args.get("file_glob")
        context_lines = args.get("context_lines", self._config.context_lines)
        case_sensitive = args.get("case_sensitive", True)
        max_results = args.get("max_results", self._config.max_results)

        if not pattern:
            return {"error": "Pattern is required", "matches": [], "total_matches": 0}

        # Compile regex
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {
                "error": f"Invalid regex pattern: {e}",
                "matches": [],
                "total_matches": 0,
            }

        search_path = Path(path).resolve()
        if not search_path.exists():
            return {
                "error": f"Path does not exist: {path}",
                "matches": [],
                "total_matches": 0,
            }

        # Determine files to search
        files_to_search: List[Path] = []
        if search_path.is_file():
            files_to_search = [search_path]
        else:
            # Use provided patterns or default to all files
            glob_patterns = file_glob if file_glob else ["**/*"]

            # Use a set to avoid duplicate files when patterns overlap
            seen_files: set = set()

            for glob_pattern in glob_patterns:
                for match in search_path.glob(glob_pattern):
                    if match.is_file() and match not in seen_files:
                        seen_files.add(match)
                        rel_path = str(match.relative_to(search_path))

                        # Check exclusions
                        if self._config.should_exclude(rel_path):
                            continue

                        # Skip files that are too large
                        try:
                            if match.stat().st_size > self._config.max_file_size_kb * 1024:
                                continue
                        except (OSError, PermissionError):
                            continue

                        files_to_search.append(match)

        # Search files
        matches: List[Dict[str, Any]] = []
        total_matches = 0
        files_with_matches = 0
        files_searched = 0
        truncated = False

        max_file_size = self._config.max_file_size_kb * 1024

        for file_path in files_to_search:
            if len(matches) >= max_results:
                truncated = True
                break

            try:
                # Skip binary files
                if self._is_binary_file(file_path):
                    continue

                files_searched += 1

                # Read file content
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except (OSError, PermissionError) as e:
                    logger.debug("Could not read file %s: %s", file_path, e)
                    continue

                lines = content.splitlines()
                file_has_match = False

                for line_num, line in enumerate(lines, start=1):
                    match_obj = regex.search(line)
                    if match_obj:
                        if not file_has_match:
                            file_has_match = True
                            files_with_matches += 1

                        total_matches += 1

                        if len(matches) >= max_results:
                            truncated = True
                            break

                        # Get context
                        context_before = []
                        context_after = []

                        if context_lines > 0:
                            start = max(0, line_num - 1 - context_lines)
                            end = min(len(lines), line_num + context_lines)
                            context_before = lines[start : line_num - 1]
                            context_after = lines[line_num : end]

                        # Determine relative path
                        if search_path.is_dir():
                            rel_path = str(file_path.relative_to(search_path))
                        else:
                            rel_path = file_path.name

                        matches.append({
                            "file": rel_path,
                            "absolute_path": str(file_path),
                            "line": line_num,
                            "column": match_obj.start() + 1,
                            "text": line.rstrip(),
                            "match": match_obj.group(),
                            "context_before": context_before,
                            "context_after": context_after,
                        })

            except Exception as e:
                logger.debug("Error searching file %s: %s", file_path, e)
                continue

        return {
            "matches": matches,
            "total_matches": total_matches,
            "files_with_matches": files_with_matches,
            "files_searched": files_searched,
            "truncated": truncated,
            "pattern": pattern,
            "path": str(search_path),
            "file_glob": file_glob,
        }

    def _is_binary_file(self, path: Path, sample_size: int = 8192) -> bool:
        """Check if a file is binary by reading a sample.

        Args:
            path: Path to the file.
            sample_size: Number of bytes to sample.

        Returns:
            True if the file appears to be binary.
        """
        try:
            with open(path, "rb") as f:
                sample = f.read(sample_size)

            # Check for null bytes (common in binary files)
            if b"\x00" in sample:
                return True

            # Check if file is mostly printable/whitespace
            text_chars = bytearray(
                {7, 8, 9, 10, 12, 13, 27}
                | set(range(0x20, 0x100))
                - {0x7F}
            )
            non_text = len(sample.translate(None, text_chars))

            # If more than 30% non-text, consider binary
            if len(sample) > 0 and non_text / len(sample) > 0.30:
                return True

            return False

        except (OSError, PermissionError):
            return True  # Can't read, treat as binary


def create_plugin() -> FilesystemQueryPlugin:
    """Factory function for plugin discovery.

    Returns:
        A new FilesystemQueryPlugin instance.
    """
    return FilesystemQueryPlugin()


__all__ = ["FilesystemQueryPlugin", "create_plugin"]
