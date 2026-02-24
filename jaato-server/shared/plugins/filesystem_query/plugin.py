"""Filesystem Query Plugin - Read-only tools for exploring the filesystem.

This plugin provides fast, safe, auto-approved tools for:
- Finding files with glob patterns (glob_files)
- Searching file contents with regex (grep_content)

Both tools return structured JSON output and support:
- Background execution for large searches
- Streaming execution for incremental result delivery (via :stream suffix)
- Workspace sandboxing with /tmp access support
"""

import asyncio
import logging
import os
import re
import stat
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from ..background.mixin import BackgroundCapableMixin
from jaato_sdk.plugins.base import UserCommand
from jaato_sdk.plugins.model_provider.types import ToolSchema
from ..sandbox_utils import check_path_with_jaato_containment, detect_jaato_symlink
from shared.path_utils import msys2_to_windows_path, normalize_result_path
from shared.utils.gitignore import GitignoreParser
from ..streaming.protocol import StreamingCapable, StreamChunk, ChunkCallback
from .config_loader import (
    FilesystemQueryConfig,
    load_config,
    DEFAULT_MAX_RESULTS,
    DEFAULT_MAX_FILE_SIZE_KB,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_CONTEXT_LINES,
)

logger = logging.getLogger(__name__)


def _is_relative_pattern(pattern: str) -> bool:
    """Check whether a glob pattern is relative (not an absolute path).

    Detects POSIX absolute paths (``/foo``), Windows drive-letter paths
    (``C:\\foo``, ``C:/foo``), and UNC paths (``\\\\server``).

    Returns:
        True if the pattern is relative and safe for ``Path.glob()``.
    """
    if os.path.isabs(pattern):
        return False
    # os.path.isabs on POSIX won't flag Windows drive letters — check
    # explicitly so cross-platform tool calls are handled.
    if len(pattern) >= 2 and pattern[0].isalpha() and pattern[1] == ':':
        return False
    return True


def _detect_workspace_root() -> Optional[str]:
    """Auto-detect workspace root from environment variables.

    Checks JAATO_WORKSPACE_ROOT first, then workspaceRoot.

    Returns:
        Absolute path to workspace root, or None if not configured.
    """
    workspace = os.environ.get('JAATO_WORKSPACE_ROOT')
    if workspace:
        return os.path.realpath(os.path.abspath(workspace))
    workspace = os.environ.get('workspaceRoot')
    if workspace:
        return os.path.realpath(os.path.abspath(workspace))
    return None


class FilesystemQueryPlugin(BackgroundCapableMixin, StreamingCapable):
    """Plugin providing filesystem query tools (glob and grep).

    This plugin offers read-only, auto-approved tools for exploring codebases:
    - glob_files: Find files matching glob patterns
    - grep_content: Search file contents using regex

    Both tools support:
    - Configurable exclusion patterns
    - Result limits to prevent overwhelming output
    - Background execution for large searches
    - Structured JSON responses
    - Streaming execution via :stream suffix (e.g., glob_files:stream)
    """

    def __init__(self):
        """Initialize the filesystem query plugin."""
        super().__init__(max_workers=2, default_timeout=300.0)
        self._config: Optional[FilesystemQueryConfig] = None
        self._initialized = False
        # Workspace root for path sandboxing
        self._workspace_root: Optional[str] = None
        # Whether to allow /tmp paths
        self._allow_tmp: bool = True
        # Plugin registry for checking external path authorization
        self._plugin_registry = None

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "filesystem_query"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dict to override defaults.
                - workspace_root: Path to workspace root for sandboxing.
                                  Auto-detected from JAATO_WORKSPACE_ROOT or
                                  workspaceRoot env vars if not specified.
                - allow_tmp: Whether to allow /tmp paths (default: True).
        """
        config = config or {}
        self._config = load_config(runtime_config=config)

        # Configure workspace root for path sandboxing
        workspace_root = config.get("workspace_root")
        if workspace_root:
            self._workspace_root = os.path.realpath(os.path.abspath(workspace_root))
        else:
            self._workspace_root = _detect_workspace_root()

        # Whether to allow /tmp paths (default: True)
        self._allow_tmp = config.get("allow_tmp", True)

        # Integrate .gitignore patterns from the workspace root.
        # include_defaults=False avoids duplicating the patterns already
        # covered by DEFAULT_EXCLUDE_PATTERNS.
        self._setup_gitignore()

        self._initialized = True
        logger.info(
            "FilesystemQueryPlugin initialized (max_results=%d, timeout=%ds, workspace=%s, allow_tmp=%s)",
            self._config.max_results,
            self._config.timeout_seconds,
            self._workspace_root or "none",
            self._allow_tmp,
        )

        # Log .jaato symlink detection for visibility
        if self._workspace_root:
            is_symlink, target = detect_jaato_symlink(self._workspace_root)
            if is_symlink:
                logger.info("FilesystemQueryPlugin: .jaato is symlink -> %s", target)

    def shutdown(self) -> None:
        """Shutdown the plugin and release resources."""
        self._shutdown_bg_executor()
        self._initialized = False
        self._workspace_root = None
        self._allow_tmp = True
        self._plugin_registry = None
        logger.info("FilesystemQueryPlugin shutdown")

    def set_plugin_registry(self, registry) -> None:
        """Set the plugin registry for checking external path authorization.

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry

    def set_workspace_path(self, path: Optional[str]) -> None:
        """Update the workspace root path.

        Called when a client connects with a different working directory.
        Re-creates the ``GitignoreParser`` for the new workspace so that the
        correct ``.gitignore`` is consulted.

        Args:
            path: The new workspace root path, or None to disable sandboxing.
        """
        if path:
            self._workspace_root = os.path.realpath(os.path.abspath(path))
        else:
            self._workspace_root = None
        self._setup_gitignore()
        logger.debug("FilesystemQueryPlugin workspace_root=%s", self._workspace_root)

    def _setup_gitignore(self) -> None:
        """Create and inject a ``GitignoreParser`` for the current workspace.

        If a workspace root is configured and a config object exists, a new
        ``GitignoreParser`` is created with ``include_defaults=False`` (the
        plugin's own ``DEFAULT_EXCLUDE_PATTERNS`` already cover those) and
        injected into the config via :meth:`FilesystemQueryConfig.set_gitignore`.

        When no workspace root is available the gitignore parser is cleared.
        """
        if self._config is None:
            return
        if self._workspace_root:
            try:
                parser = GitignoreParser(
                    Path(self._workspace_root),
                    include_defaults=False,
                )
                self._config.set_gitignore(parser)
                logger.debug(
                    "FilesystemQueryPlugin: loaded .gitignore from %s",
                    self._workspace_root,
                )
            except Exception:
                logger.debug(
                    "FilesystemQueryPlugin: failed to load .gitignore from %s",
                    self._workspace_root,
                    exc_info=True,
                )
        else:
            self._config.set_gitignore(None)  # type: ignore[arg-type]

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path, making relative paths relative to workspace_root.

        Under MSYS2, also converts /c/Users/... paths to C:/Users/... so
        that Python can resolve them via Windows APIs.

        Args:
            path: Path string (absolute or relative, Windows or MSYS2 format).

        Returns:
            Resolved Path object. Relative paths are resolved against
            workspace_root if configured, otherwise against CWD.
        """
        # Convert MSYS2 drive paths (/c/...) to Windows (C:/...) for Python
        path = msys2_to_windows_path(path)
        p = Path(path)
        if p.is_absolute():
            return p
        if self._workspace_root:
            return Path(self._workspace_root) / p
        return p

    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is allowed for read access.

        A path is allowed if:
        1. The path is within the workspace_root (or CWD if not configured)
        2. The path is under .jaato and within the .jaato containment boundary
           (see sandbox_utils.py for .jaato contained symlink escape rules)
        3. The path is authorized via the plugin registry (read or readwrite)

        This plugin is read-only, so it always checks with mode="read".
        Both "readonly" and "readwrite" authorized paths allow read access.

        Args:
            path: Path to check.

        Returns:
            True if access is allowed, False otherwise.
        """
        # Use workspace_root if configured, otherwise fall back to CWD
        # This ensures absolute paths outside the boundary are always blocked
        if not self._workspace_root:
            # No workspace — cannot sandbox, allow explicitly provided paths
            return True
        workspace = self._workspace_root

        # Resolve path relative to workspace first
        resolved = self._resolve_path(path)
        abs_path = str(resolved.absolute())

        # Use shared sandbox utility with .jaato containment support and /tmp access
        # Always mode="read" since this plugin is read-only
        allowed = check_path_with_jaato_containment(
            abs_path,
            workspace,
            self._plugin_registry,
            allow_tmp=getattr(self, '_allow_tmp', True),
            mode="read"
        )

        if not allowed:
            logger.debug("FilesystemQueryPlugin: path blocked (outside sandbox): %s", path)
        return allowed

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
                                "Relative glob pattern to match files. "
                                "Must be relative, NOT an absolute path. "
                                "Examples: "
                                "'**/*.py' (all Python files), "
                                "'src/**/*.ts' (TypeScript in src), "
                                "'*.json' (JSON in current dir), "
                                "'**/test_*.py' (test files). "
                                "To search a specific directory, set 'root' "
                                "to that directory and use a relative pattern."
                            ),
                        },
                        "root": {
                            "type": "string",
                            "description": (
                                "Root directory to search from (can be absolute). "
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
                discoverability="discoverable",
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
                                "File or directory to search in (can be absolute). "
                                "Defaults to current working directory."
                            ),
                        },
                        "file_glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Only search files matching these relative glob patterns. "
                                "Must be relative, NOT absolute paths. "
                                "Examples: ['*.py'], ['**/*.java', '**/*.kt', '**/*.scala']. "
                                "To search a specific directory, set 'path' to that "
                                "directory and use relative patterns here."
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
                discoverability="discoverable",
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

These tools provide structured JSON output, respect the project's .gitignore,
automatically exclude common noise directories (node_modules, __pycache__, .git, etc.),
and don't require shell escaping.

## glob_files
Find files by name pattern. Use glob syntax:
- `**/*.py` - all Python files recursively
- `src/**/*.ts` - TypeScript files in src/
- `**/test_*.py` - test files anywhere
- `*.json` - JSON files in current directory

The `pattern` must always be **relative** (e.g. `**/*.py`), never an absolute path.
To search a specific directory, set `root` to that absolute path and use a relative `pattern`.

## grep_content
Search file contents with regex:
- `def\\s+function_name` - find function definitions
- `class\\s+ClassName` - find class definitions
- `import\\s+module` - find imports
- `TODO|FIXME|HACK` - find code comments

The file_glob parameter accepts an array of **relative** glob patterns:
- Single type: `file_glob=["*.py"]`
- Multiple types: `file_glob=["**/*.java", "**/*.kt", "**/*.scala"]`

IMPORTANT: `file_glob` patterns must be relative, never absolute paths.
To search a specific directory, set `path` to that absolute directory and keep `file_glob` relative.

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
        root = args.get("root") or self._workspace_root
        max_results = args.get("max_results", self._config.max_results)
        include_hidden = args.get("include_hidden", False)

        if not pattern:
            return {"error": "Pattern is required", "files": [], "total": 0}

        if not root:
            return {"error": "No root specified and no workspace configured", "files": [], "total": 0}

        # Normalize trailing ** — Python pathlib.glob("dir/**") only matches
        # the directory itself, not files inside it. Append /* to match files.
        if pattern.endswith("**"):
            pattern = pattern + "/*"

        # Sandbox check: ensure root path is within allowed workspace (with /tmp support)
        if not self._is_path_allowed(root):
            return {
                "error": f"Path not allowed: {root}",
                "files": [],
                "total": 0,
            }

        root_path = self._resolve_path(root).resolve()
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

        if not _is_relative_pattern(pattern):
            return {
                "error": (
                    f"Absolute patterns are not supported in glob_files. "
                    f"The pattern must be relative to the root directory. "
                    f"Got pattern: '{pattern}'. "
                    f"Instead, set root to the absolute directory and use "
                    f"a relative pattern (e.g., root='{pattern.split('**')[0].rstrip('/')}', "
                    f"pattern='**/*')."
                ),
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
                # Normalize paths for MSYS2 compatibility (forward slashes)
                norm_rel = normalize_result_path(rel_path)
                norm_abs = normalize_result_path(str(match))
                try:
                    stat_info = match.stat()
                    file_info = {
                        "path": norm_rel,
                        "absolute_path": norm_abs,
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
                        "path": norm_rel,
                        "absolute_path": norm_abs,
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
                "root": normalize_result_path(str(root_path)),
                "pattern": pattern,
            }

        except NotImplementedError:
            return {
                "error": (
                    f"Absolute patterns are not supported in glob_files. "
                    f"The pattern must be relative to the root directory. "
                    f"Got pattern: '{pattern}'. "
                    f"Instead, set root to the absolute directory and use "
                    f"a relative pattern (e.g., root='{pattern.split('**')[0].rstrip('/')}', "
                    f"pattern='**/*')."
                ),
                "files": [],
                "total": 0,
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
        path = args.get("path") or self._workspace_root
        file_glob = args.get("file_glob")
        context_lines = args.get("context_lines", self._config.context_lines)
        case_sensitive = args.get("case_sensitive", True)
        max_results = args.get("max_results", self._config.max_results)

        if not pattern:
            return {"error": "Pattern is required", "matches": [], "total_matches": 0}

        # Compile regex early to catch syntax errors before path validation
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {
                "error": f"Invalid regex pattern: {e}",
                "matches": [],
                "total_matches": 0,
            }

        if not path:
            return {"error": "No path specified and no workspace configured", "matches": [], "total_matches": 0}

        # Sandbox check: ensure search path is within allowed workspace (with /tmp support)
        if not self._is_path_allowed(path):
            return {
                "error": f"Path not allowed: {path}",
                "matches": [],
                "total_matches": 0,
            }

        search_path = self._resolve_path(path).resolve()
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
                if not _is_relative_pattern(glob_pattern):
                    bad = [p for p in glob_patterns if not _is_relative_pattern(p)]
                    return {
                        "error": (
                            f"file_glob patterns must be relative, not absolute paths. "
                            f"Got absolute pattern(s): {bad}. "
                            f"Use the 'path' parameter for the search directory and "
                            f"keep file_glob as relative patterns "
                            f"(e.g., path='{glob_pattern.split('**')[0].rstrip('/')}', "
                            f"file_glob=['**/*'])."
                        ),
                        "matches": [],
                        "total_matches": 0,
                    }
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
                            "file": normalize_result_path(rel_path),
                            "absolute_path": normalize_result_path(str(file_path)),
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
            "path": normalize_result_path(str(search_path)),
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

        except (OSError, PermissionError) as exc:
            logger.debug(f"Cannot read file {path} for binary check: {exc}")
            return True  # Can't read, treat as binary

    # --- StreamingCapable implementation ---

    def supports_streaming(self, tool_name: str) -> bool:
        """Check if a tool supports streaming execution.

        Both glob_files and grep_content support streaming.
        """
        return tool_name in ("glob_files", "grep_content")

    def get_streaming_tool_names(self) -> List[str]:
        """Get list of tools that support streaming."""
        return ["glob_files", "grep_content"]

    async def execute_streaming(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        on_chunk: Optional[ChunkCallback] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute tool with streaming results.

        Yields results as they are found, allowing the model to act on
        partial results while the operation continues.

        Args:
            tool_name: "glob_files" or "grep_content".
            arguments: Tool arguments.
            on_chunk: Optional callback for each chunk.

        Yields:
            StreamChunk objects for each result found.
        """
        if tool_name == "glob_files":
            async for chunk in self._stream_glob_files(arguments, on_chunk):
                yield chunk
            return
        elif tool_name != "grep_content":
            raise ValueError(f"Streaming not supported for tool: {tool_name}")

        if not self._config:
            self._config = load_config()

        pattern = arguments.get("pattern", "")
        path = arguments.get("path") or self._workspace_root
        file_glob = arguments.get("file_glob")
        context_lines = arguments.get("context_lines", self._config.context_lines)
        case_sensitive = arguments.get("case_sensitive", True)
        max_results = arguments.get("max_results", self._config.max_results)

        if not pattern:
            yield StreamChunk(
                content="Error: Pattern is required",
                chunk_type="error"
            )
            return

        if not path:
            yield StreamChunk(content="Error: No path specified and no workspace configured", chunk_type="error")
            return

        # Compile regex
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
        except re.error as e:
            yield StreamChunk(
                content=f"Error: Invalid regex pattern: {e}",
                chunk_type="error"
            )
            return

        search_path = Path(path).resolve()
        if not search_path.exists():
            yield StreamChunk(
                content=f"Error: Path does not exist: {path}",
                chunk_type="error"
            )
            return

        # Determine files to search
        files_to_search: List[Path] = []
        if search_path.is_file():
            files_to_search = [search_path]
        else:
            glob_patterns = file_glob if file_glob else ["**/*"]
            seen_files: set = set()

            for glob_pattern in glob_patterns:
                for match in search_path.glob(glob_pattern):
                    if match.is_file() and match not in seen_files:
                        seen_files.add(match)
                        rel_path = str(match.relative_to(search_path))

                        if self._config.should_exclude(rel_path):
                            continue

                        try:
                            if match.stat().st_size > self._config.max_file_size_kb * 1024:
                                continue
                        except (OSError, PermissionError):
                            continue

                        files_to_search.append(match)

        # Stream matches as they are found
        match_count = 0
        files_searched = 0

        for file_path in files_to_search:
            if match_count >= max_results:
                break

            try:
                if self._is_binary_file(file_path):
                    continue

                files_searched += 1

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except (OSError, PermissionError):
                    continue

                lines = content.splitlines()

                for line_num, line in enumerate(lines, start=1):
                    if match_count >= max_results:
                        break

                    match_obj = regex.search(line)
                    if match_obj:
                        match_count += 1

                        # Determine relative path
                        if search_path.is_dir():
                            rel_path = str(file_path.relative_to(search_path))
                        else:
                            rel_path = file_path.name

                        # Format match for streaming
                        norm_rel_path = normalize_result_path(rel_path)
                        match_text = f"{norm_rel_path}:{line_num}: {line.strip()}"

                        chunk = StreamChunk(
                            content=match_text,
                            chunk_type="match",
                            sequence=match_count,
                            metadata={
                                "file": norm_rel_path,
                                "line": line_num,
                                "column": match_obj.start() + 1,
                                "match": match_obj.group(),
                            }
                        )

                        if on_chunk:
                            on_chunk(chunk)

                        yield chunk

                        # Allow other tasks to run
                        await asyncio.sleep(0)

            except Exception as e:
                logger.debug("Error searching file %s: %s", file_path, e)
                continue

        # Final summary chunk
        yield StreamChunk(
            content=f"Search complete: {match_count} matches in {files_searched} files",
            chunk_type="summary",
            metadata={
                "total_matches": match_count,
                "files_searched": files_searched,
            }
        )

    async def _stream_glob_files(
        self,
        arguments: Dict[str, Any],
        on_chunk: Optional[ChunkCallback] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream glob_files results as files are found.

        Args:
            arguments: Tool arguments (pattern, root, max_results, etc.).
            on_chunk: Optional callback for each chunk.

        Yields:
            StreamChunk objects for each file found.
        """
        if not self._config:
            self._config = load_config()

        pattern = arguments.get("pattern", "")
        root = arguments.get("root") or self._workspace_root
        max_results = arguments.get("max_results", self._config.max_results)
        include_hidden = arguments.get("include_hidden", False)

        if not pattern:
            yield StreamChunk(
                content="Error: Pattern is required",
                chunk_type="error"
            )
            return

        if not root:
            yield StreamChunk(content="Error: No root specified and no workspace configured", chunk_type="error")
            return

        # Normalize trailing ** — Python pathlib.glob("dir/**") only matches
        # the directory itself, not files inside it. Append /* to match files.
        if pattern.endswith("**"):
            pattern = pattern + "/*"

        # Sandbox check
        if not self._is_path_allowed(root):
            yield StreamChunk(
                content=f"Error: Path not allowed: {root}",
                chunk_type="error"
            )
            return

        root_path = self._resolve_path(root).resolve()
        if not root_path.exists():
            yield StreamChunk(
                content=f"Error: Root path does not exist: {root}",
                chunk_type="error"
            )
            return

        if not root_path.is_dir():
            yield StreamChunk(
                content=f"Error: Root path is not a directory: {root}",
                chunk_type="error"
            )
            return

        # Stream files as they are found
        file_count = 0
        total_found = 0

        try:
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

                if file_count >= max_results:
                    continue  # Keep counting but don't stream more

                file_count += 1

                # Get file info
                # Normalize paths for MSYS2 compatibility (forward slashes)
                norm_rel = normalize_result_path(rel_path)
                norm_abs = normalize_result_path(str(match))
                try:
                    stat_info = match.stat()
                    size_str = self._format_size(stat_info.st_size)
                    modified = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M")
                    content = f"{norm_rel} ({size_str}, {modified})"
                    metadata = {
                        "path": norm_rel,
                        "absolute_path": norm_abs,
                        "size": stat_info.st_size,
                        "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                    }
                except (OSError, PermissionError):
                    content = norm_rel
                    metadata = {
                        "path": norm_rel,
                        "absolute_path": norm_abs,
                    }

                chunk = StreamChunk(
                    content=content,
                    chunk_type="file",
                    sequence=file_count,
                    metadata=metadata,
                )

                if on_chunk:
                    on_chunk(chunk)

                yield chunk

                # Allow other tasks to run
                await asyncio.sleep(0)

        except Exception as e:
            logger.exception("Error in streaming glob_files: %s", e)
            yield StreamChunk(
                content=f"Error during search: {str(e)}",
                chunk_type="error"
            )
            return

        # Final summary chunk
        truncated = total_found > max_results
        yield StreamChunk(
            content=f"Found {total_found} files{' (truncated)' if truncated else ''}",
            chunk_type="summary",
            metadata={
                "total_found": total_found,
                "returned": file_count,
                "truncated": truncated,
                "root": normalize_result_path(str(root_path)),
                "pattern": pattern,
            }
        )

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable form."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}" if unit != "B" else f"{size}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


def create_plugin() -> FilesystemQueryPlugin:
    """Factory function for plugin discovery.

    Returns:
        A new FilesystemQueryPlugin instance.
    """
    return FilesystemQueryPlugin()


__all__ = ["FilesystemQueryPlugin", "create_plugin"]
