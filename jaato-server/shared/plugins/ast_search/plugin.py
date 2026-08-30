"""AST Search Plugin - Structural code search using AST patterns.

This plugin provides semantic code search capabilities using ast-grep-py,
allowing searches based on code structure rather than just text patterns.

Features:
- Multi-language support (Python, JavaScript, TypeScript, Go, Rust, Java, C, etc.)
- Pattern-based search using code-like syntax (e.g., "def $NAME($$$): $$$")
- Node type filtering (function_definition, class_definition, etc.)
- Context lines for surrounding code
- Streaming support for incremental results (ast_search:stream)
"""

import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

from ..background.mixin import BackgroundCapableMixin
from jaato_sdk.plugins.base import UserCommand
from jaato_sdk.plugins.model_provider.types import (
    ToolSchema,
    DISCOVERABILITY_DEFERRED,
)
from shared.plugins.runner_forwarding import RunnerForwardingMixin
from ..path_safety import describe_special
from ..sandbox_utils import check_path_with_jaato_containment
from ..streaming import StreamingCapable, StreamChunk, ChunkCallback

logger = logging.getLogger(__name__)

# Supported languages and their file extensions
LANGUAGE_EXTENSIONS: Dict[str, List[str]] = {
    "python": [".py", ".pyi"],
    "javascript": [".js", ".mjs", ".cjs"],
    "typescript": [".ts", ".mts", ".cts"],
    "tsx": [".tsx"],
    "jsx": [".jsx"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"],
    "csharp": [".cs"],
    "ruby": [".rb"],
    "kotlin": [".kt", ".kts"],
    "swift": [".swift"],
    "scala": [".scala"],
    "php": [".php"],
    "lua": [".lua"],
    "html": [".html", ".htm"],
    "css": [".css"],
    "json": [".json"],
    "yaml": [".yaml", ".yml"],
    "toml": [".toml"],
    "bash": [".sh", ".bash"],
    "sql": [".sql"],
}

# Reverse mapping: extension -> language
EXTENSION_TO_LANGUAGE: Dict[str, str] = {}
for lang, exts in LANGUAGE_EXTENSIONS.items():
    for ext in exts:
        EXTENSION_TO_LANGUAGE[ext] = lang

# Default exclusion patterns (directories to skip)
DEFAULT_EXCLUDE_DIRS: List[str] = [
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "target",
    ".idea",
    ".vscode",
]

# Default limits
DEFAULT_MAX_RESULTS = 100
DEFAULT_CONTEXT_LINES = 2
DEFAULT_TIMEOUT_SECONDS = 60


def _check_ast_grep_available() -> bool:
    """Check if ast-grep-py is available."""
    try:
        import ast_grep_py  # noqa: F401
        return True
    except ImportError:
        return False


class ASTSearchPlugin(BackgroundCapableMixin, StreamingCapable, RunnerForwardingMixin):
    """Plugin providing AST-based structural code search.

    This plugin offers semantic code search using ast-grep-py:
    - ast_search: Search code by AST patterns
    - ast_search:stream: Streaming variant for incremental results

    Supports multi-language codebases with pattern-based queries
    that understand code structure rather than just text.

    Streaming mode yields matches as they're found, allowing the model
    to act on partial results while the search continues.
    """

    def __init__(self):
        """Initialize the AST search plugin."""
        super().__init__(max_workers=2, default_timeout=DEFAULT_TIMEOUT_SECONDS)
        self._initialized = False
        self._ast_grep_available: Optional[bool] = None
        self._max_results = DEFAULT_MAX_RESULTS
        self._context_lines = DEFAULT_CONTEXT_LINES
        self._exclude_dirs = list(DEFAULT_EXCLUDE_DIRS)
        self._workspace_path: Optional[str] = None
        self._allow_tmp: bool = True
        self._plugin_registry = None

    @property
    def name(self) -> str:
        """Return the plugin name."""
        return "ast_search"

    def _ensure_ast_grep(self) -> bool:
        """Ensure ast-grep-py is available, caching the result."""
        if self._ast_grep_available is None:
            self._ast_grep_available = _check_ast_grep_available()
            if not self._ast_grep_available:
                logger.warning(
                    "ast-grep-py not installed. Install with: pip install ast-grep-py"
                )
        return self._ast_grep_available

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Optional configuration dict to override defaults.
                - max_results: Maximum matches to return (default: 100)
                - context_lines: Lines of context around matches (default: 2)
                - exclude_dirs: Additional directories to exclude
        """
        if config:
            self._max_results = config.get("max_results", DEFAULT_MAX_RESULTS)
            self._context_lines = config.get("context_lines", DEFAULT_CONTEXT_LINES)
            extra_excludes = config.get("exclude_dirs", [])
            if extra_excludes:
                self._exclude_dirs = list(set(self._exclude_dirs + extra_excludes))
            workspace_path = config.get("workspace_root")
            if workspace_path:
                self._workspace_path = os.path.realpath(os.path.abspath(workspace_path))
            self._allow_tmp = config.get("allow_tmp", True)

        self._initialized = True
        logger.info(
            "ASTSearchPlugin initialized (max_results=%d, context_lines=%d)",
            self._max_results,
            self._context_lines,
        )

    def set_workspace_path(self, path: str) -> None:
        """Update the workspace path used as the search root and sandbox bound.

        Called by PluginRegistry.set_workspace_path() when a session binds
        to a specific workspace.  The path is canonicalised here so that
        :meth:`_is_path_allowed` compares two symlink-free paths.

        Args:
            path: The new workspace root, or None to disable sandboxing.
        """
        self._workspace_path = (
            os.path.realpath(os.path.abspath(path)) if path else None
        )

    def set_plugin_registry(self, registry) -> None:
        """Receive the plugin registry, used for external-path authorization.

        Auto-wired by ``PluginRegistry.expose_tool()``.  The registry is what
        knows which paths outside the workspace the user has granted via
        ``sandbox add``, so ``ast_search`` needs it to answer path questions
        the same way ``filesystem_query`` and ``file_edit`` do.

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry

    def _is_path_allowed(self, path: str) -> bool:
        """Check whether ``path`` may be read by this (read-only) plugin.

        Delegates to the shared sandbox utility so ``ast_search`` enforces the
        same boundary as ``filesystem_query``: inside the workspace, inside
        the ``.jaato`` containment boundary when explicitly authorized, under
        a system temp directory, or registered via ``sandbox add``.

        Args:
            path: Path to check.

        Returns:
            True if reading the path is allowed.
        """
        if not self._workspace_path:
            # No workspace configured — nothing to sandbox against.
            return True
        return check_path_with_jaato_containment(
            os.path.abspath(path),
            self._workspace_path,
            self._plugin_registry,
            allow_tmp=self._allow_tmp,
            mode="read",
        )

    def _make_result_guard(self) -> Callable[[Path], bool]:
        """Build a containment filter to apply to every *result* of a search.

        ``rglob`` follows symlinked directories, so validating only the search
        root lets a link committed into the repository (``data/logs -> /etc``)
        pull files from outside the workspace into the result set — and, since
        ast_search prints matching source, into the transcript.  This
        re-applies :meth:`_is_path_allowed` per hit, caching by parent
        directory because each check costs a :func:`os.path.realpath` and a
        walk yields many files per directory.  A leaf is checked individually
        only when it is itself a symlink, the one way it can escape a parent
        that is already contained.  Special files (FIFOs, devices) are dropped
        outright — parsing one would block the worker.

        Returns:
            A predicate over candidate paths.  Not thread-safe — build one per
            invocation.
        """
        dir_cache: Dict[str, bool] = {}

        def allowed(candidate: Path) -> bool:
            path_str = str(candidate)

            if describe_special(candidate) is not None:
                logger.debug("ASTSearchPlugin: skipping special file: %s", path_str)
                return False

            parent = os.path.dirname(path_str)
            verdict = dir_cache.get(parent)
            if verdict is None:
                verdict = self._is_path_allowed(parent)
                dir_cache[parent] = verdict
            if not verdict:
                logger.debug(
                    "ASTSearchPlugin: result outside sandbox (via directory %s): %s",
                    parent,
                    path_str,
                )
                return False

            try:
                is_link = candidate.is_symlink()
            except OSError:
                return False
            if is_link and not self._is_path_allowed(path_str):
                logger.debug(
                    "ASTSearchPlugin: result outside sandbox (symlinked file): %s",
                    path_str,
                )
                return False

            return True

        return allowed

    def shutdown(self) -> None:
        """Shutdown the plugin and release resources."""
        self._shutdown_bg_executor()
        self._initialized = False
        logger.info("ASTSearchPlugin shutdown")

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset — NO-OP for this plugin.

        Phase 1 hotfix (server 0.6.148+): added to satisfy the
        ``ToolPlugin`` / ``EnrichmentPlugin`` protocol's runtime
        ``isinstance`` check.  Per Daniel's litmus test (see
        ``docs/design/runner-cascade-sharing.md`` §4.3), this
        plugin holds no per-session state that the next cascade
        session would benefit from having cleared.  Override in
        future PRs if the litmus test changes.
        """
        pass


    def get_config_schema(self) -> dict:
        """Return JSON Schema for this plugin's configuration."""
        return {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "default": 100,
                    "description": "Maximum matches to return",
                },
                "context_lines": {
                    "type": "integer",
                    "default": 2,
                    "description": "Lines of context around each match",
                },
                "exclude_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": list(DEFAULT_EXCLUDE_DIRS),
                    "description": "Directories to exclude from search",
                },
            },
        }

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the tool schemas for AST search.

        The ast_search tool is categorized as "search" and marked as "discoverable"
        to support the deferred tool loading mechanism. Models can discover it via
        list_tools(category='search') and request full schema via get_tool_schemas().
        """
        return [
            ToolSchema(
                name="ast_search",
                description=(
                    "Search code by AST (Abstract Syntax Tree) patterns. "
                    "Unlike text-based grep, this understands code structure and can find "
                    "patterns like 'all functions with 3 parameters' or 'all try/except blocks'. "
                    "Use code-like patterns with metavariables: $NAME matches single node, "
                    "$$$ matches multiple nodes. "
                    "Example patterns: 'def $FUNC($$$): $$$' (Python functions), "
                    "'function $NAME($$$) { $$$ }' (JS functions), "
                    "'if err != nil { $$$ }' (Go error handling)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": (
                                "AST pattern to search for. Use metavariables:\n"
                                "- $NAME: matches any single AST node (identifier, expression, etc.)\n"
                                "- $$$: matches zero or more nodes (for parameters, statements, etc.)\n"
                                "Examples:\n"
                                "- 'def $FUNC($$$): $$$' - Python function definitions\n"
                                "- 'class $NAME($$$): $$$' - Python classes with any inheritance\n"
                                "- 'try: $$$ except $E: $$$' - Python try/except blocks\n"
                                "- 'import $MODULE' - Python imports\n"
                                "- 'function $NAME($$$) { $$$ }' - JavaScript functions\n"
                                "- 'if err != nil { $$$ }' - Go error handling pattern"
                            ),
                        },
                        "language": {
                            "type": "string",
                            "description": (
                                "Programming language. If not specified, auto-detected from "
                                "file extension. Supported: python, javascript, typescript, "
                                "tsx, jsx, go, rust, java, c, cpp, csharp, ruby, kotlin, "
                                "swift, scala, php, lua, html, css, json, yaml, toml, bash, sql. "
                                "Languages like COBOL, Perl, Haskell are NOT supported."
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "File or directory to search in. "
                                "Defaults to current working directory."
                            ),
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": (
                                "Glob pattern to filter files (e.g., '*.py', 'src/**/*.ts'). "
                                "If not specified, uses language-appropriate extensions."
                            ),
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": (
                                f"Lines of context before/after match (default: {DEFAULT_CONTEXT_LINES})"
                            ),
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
                discoverability=DISCOVERABILITY_DEFERRED,
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor functions for each tool.

        Phase 3 §3.4 wave 1: forwards via runner-RPC when a runner
        is attached; falls through to in-process otherwise.
        """
        return self.wrap_executors_for_runner_forwarding({
            "ast_search": self._execute_ast_search,
        })

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that don't require permission (read-only operations)."""
        return ["ast_search"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands (none for this plugin)."""
        return []

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the model."""
        return """You have access to AST-based structural code search via the `ast_search` tool.

## When to use ast_search vs grep_content

Use `ast_search` for **structural/semantic** queries:
- Find all function/method definitions
- Find all classes inheriting from a specific base
- Find all try/except/catch blocks
- Find specific code patterns (e.g., Go error handling: `if err != nil`)
- Find all imports of a specific module
- Find all function calls with specific argument patterns

Use `grep_content` for **text-based** queries:
- Find specific strings or identifiers
- Find comments containing certain text
- Find TODO/FIXME markers
- Simple text searches

## Supported Languages

ast_search supports 22 languages with their file extensions:

| Language | Extensions |
|----------|------------|
| python | .py, .pyi |
| javascript | .js, .mjs, .cjs |
| typescript | .ts, .mts, .cts |
| tsx | .tsx |
| jsx | .jsx |
| go | .go |
| rust | .rs |
| java | .java |
| c | .c, .h |
| cpp | .cpp, .cc, .cxx, .hpp, .hh, .hxx |
| csharp | .cs |
| ruby | .rb |
| kotlin | .kt, .kts |
| swift | .swift |
| scala | .scala |
| php | .php |
| lua | .lua |
| html | .html, .htm |
| css | .css |
| json | .json |
| yaml | .yaml, .yml |
| toml | .toml |
| bash | .sh, .bash |
| sql | .sql |

Languages like COBOL, Perl, and Haskell are **not supported**. For unsupported languages, use `grep_content` for text-based search instead.

## Pattern Syntax

Patterns use code-like syntax with metavariables:
- `$NAME` - matches any single AST node (identifier, expression, literal, etc.)
- `$$$` - matches zero or more nodes (parameters, statements, arguments)

## Examples by Language

### Python
- `def $FUNC($$$): $$$` - all function definitions
- `class $NAME: $$$` - classes without explicit inheritance
- `class $NAME($BASE): $$$` - classes with single base
- `@$DECORATOR def $FUNC($$$): $$$` - decorated functions
- `import $MODULE` - import statements
- `from $MODULE import $$$` - from imports
- `try: $$$ except $E: $$$` - try/except blocks
- `$OBJ.$METHOD($$$)` - method calls

### JavaScript/TypeScript
- `function $NAME($$$) { $$$ }` - function declarations
- `const $NAME = ($$$) => $$$` - arrow functions
- `class $NAME extends $BASE { $$$ }` - classes with inheritance
- `async function $NAME($$$) { $$$ }` - async functions
- `try { $$$ } catch ($E) { $$$ }` - try/catch blocks

### Go
- `func $NAME($$$) $$$` - function definitions
- `func ($RECV) $NAME($$$) $$$` - method definitions
- `if err != nil { $$$ }` - error handling pattern
- `type $NAME struct { $$$ }` - struct definitions
- `type $NAME interface { $$$ }` - interface definitions

### Rust
- `fn $NAME($$$) -> $$$ { $$$ }` - function definitions
- `impl $TRAIT for $TYPE { $$$ }` - trait implementations
- `struct $NAME { $$$ }` - struct definitions
- `match $EXPR { $$$ }` - match expressions

## Tips
- Start with broader patterns, then refine
- Use language parameter when searching mixed-language codebases
- Use file_pattern to limit search scope for better performance

## Streaming vs Non-streaming

| Aspect | Non-Streaming (`ast_search`) | Streaming (`ast_search:stream`) |
|--------|------------------------------|--------------------------------|
| Initial response | Waits for completion | Immediate (first results) |
| Large searches | May auto-background (60s+) | Returns partial results instantly |
| Result format | Full JSON with metavariables, context | Simplified `file:line: code` |
| Use case | Need complete data | Need quick feedback |

Guidelines:
1. **Use streaming for exploration**: When checking if a pattern matches anything
2. **Use non-streaming for complete data**: When you need metavariables and context
3. **Dismiss streams early**: If you find what you need, dismiss to save resources
"""

    # --- BackgroundCapable overrides ---

    def supports_background(self, tool_name: str) -> bool:
        """Check if a tool supports background execution."""
        return tool_name == "ast_search"

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        """Return timeout threshold for automatic backgrounding."""
        return float(DEFAULT_TIMEOUT_SECONDS)

    def estimate_duration(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate execution duration based on arguments.

        Uses heuristics to estimate whether this is likely a directory
        search (longer) or a single file search (shorter).
        """
        if tool_name == "ast_search":
            path = arguments.get("path", ".")
            # Use heuristics: if path has a file extension, it's likely a file
            # Otherwise assume directory search (which takes longer)
            path_obj = Path(path)
            has_extension = bool(path_obj.suffix)
            if has_extension:
                return 2.0  # Single file search
            return 15.0  # Directory search takes longer
        return None

    # --- Tool implementation ---

    def _select_files(
        self,
        search_path: Path,
        file_pattern: Optional[str],
        language: Optional[str],
    ) -> "tuple[List[Path], Optional[str], Optional[str]]":
        """Pick the files a search should read, applying containment per hit.

        Shared by the blocking and streaming variants, which had carried
        identical copies of this selection logic; keeping one copy is also
        what keeps the per-result sandbox check from drifting between them.

        Three selection modes, in the order the tool's arguments imply:
        an explicit ``file_pattern`` glob, a ``language``'s extensions, or
        every extension the plugin knows how to parse.  Every candidate must
        clear :meth:`_make_result_guard` — the walk follows symlinked
        directories, so a link out of the workspace would otherwise pull
        outside files into the results.

        Args:
            search_path: Resolved root of the search (file or directory).
            file_pattern: Optional glob, relative to ``search_path``.
            language: Optional language name; auto-detected for a single file.

        Returns:
            ``(files, language, error)``.  ``language`` is the resolved
            language (auto-detected when a single file was named).  ``error``
            is a message for the caller to surface, or None; when set,
            ``files`` is empty.
        """
        if search_path.is_file():
            return self._select_single_file(search_path, language)

        if not file_pattern and language and not LANGUAGE_EXTENSIONS.get(language):
            return [], language, f"Unknown language: {language}"

        result_allowed = self._make_result_guard()
        files = [
            match
            for match in self._candidate_paths(search_path, file_pattern, language)
            if match.is_file()
            and not self._should_exclude(match, search_path)
            and result_allowed(match)
        ]
        return files, language, None

    def _select_single_file(
        self,
        search_path: Path,
        language: Optional[str],
    ) -> "tuple[List[Path], Optional[str], Optional[str]]":
        """Handle the case where the search path names one file.

        Args:
            search_path: The resolved file.
            language: Requested language, or None to auto-detect.

        Returns:
            ``(files, language, error)``, as for :meth:`_select_files`.
        """
        result_allowed = self._make_result_guard()
        files = [search_path] if result_allowed(search_path) else []

        if not language:
            ext = search_path.suffix.lower()
            language = EXTENSION_TO_LANGUAGE.get(ext)
            if not language:
                return [], None, f"Could not detect language for extension: {ext}"

        return files, language, None

    def _candidate_paths(
        self,
        search_path: Path,
        file_pattern: Optional[str],
        language: Optional[str],
    ) -> Iterator[Path]:
        """Yield the raw walk candidates for a directory search.

        Pure enumeration — exclusions and the containment guard are applied by
        :meth:`_select_files`, which is the only caller.

        Args:
            search_path: Resolved directory to walk.
            file_pattern: Explicit glob, relative to ``search_path``.
            language: Language whose extensions to walk, when no glob is given.

        Yields:
            Candidate paths, not yet filtered.
        """
        if file_pattern:
            yield from search_path.glob(file_pattern)
        elif language:
            for ext in LANGUAGE_EXTENSIONS.get(language, []):
                yield from search_path.rglob(f"*{ext}")
        else:
            for match in search_path.rglob("*"):
                if match.suffix.lower() in EXTENSION_TO_LANGUAGE:
                    yield match

    def _execute_ast_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ast_search tool.

        Args:
            args: Tool arguments containing pattern, language, path, etc.

        Returns:
            Dict with matches list, counts, and metadata.
        """
        # Check if ast-grep is available
        if not self._ensure_ast_grep():
            return {
                "error": (
                    "ast-grep-py is not installed. "
                    "Install with: pip install ast-grep-py"
                ),
                "matches": [],
                "total_matches": 0,
            }

        from ast_grep_py import SgRoot

        pattern = args.get("pattern", "")
        language = args.get("language")
        path = args.get("path") or self._workspace_path
        file_pattern = args.get("file_pattern")
        context_lines = args.get("context_lines", self._context_lines)
        max_results = args.get("max_results", self._max_results)

        if not pattern:
            return {"error": "Pattern is required", "matches": [], "total_matches": 0}

        if not path:
            return {"error": "No path specified and no workspace configured", "matches": [], "total_matches": 0}

        # Sandbox check on the search root.  Without it any absolute path is
        # readable through this auto-approved tool.
        if not self._is_path_allowed(path):
            return {
                "error": f"Path not allowed: {path}",
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

        # Determine files to search (containment applied per result)
        files_to_search, language, selection_error = self._select_files(
            search_path, file_pattern, language
        )
        if selection_error:
            return {"error": selection_error, "matches": [], "total_matches": 0}

        # Search files
        matches: List[Dict[str, Any]] = []
        total_matches = 0
        files_with_matches = 0
        files_searched = 0
        truncated = False
        errors: List[str] = []

        for file_path in files_to_search:
            if len(matches) >= max_results:
                truncated = True
                break

            try:
                # Determine language for this file
                file_lang = language
                if not file_lang:
                    ext = file_path.suffix.lower()
                    file_lang = EXTENSION_TO_LANGUAGE.get(ext)
                    if not file_lang:
                        continue

                # Read file content
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except (OSError, PermissionError) as e:
                    logger.debug("Could not read file %s: %s", file_path, e)
                    continue

                files_searched += 1

                # Parse and search using ast-grep
                try:
                    root = SgRoot(content, file_lang)
                    node = root.root()
                    found_nodes = node.find_all(pattern=pattern)
                except Exception as e:
                    # Pattern might not be valid for this language
                    logger.debug(
                        "Search failed for %s (%s): %s", file_path, file_lang, e
                    )
                    continue

                if not found_nodes:
                    continue

                files_with_matches += 1
                lines = content.splitlines()

                for found in found_nodes:
                    total_matches += 1

                    if len(matches) >= max_results:
                        truncated = True
                        break

                    # Get match location
                    range_info = found.range()
                    start_line = range_info.start.line + 1  # 1-indexed
                    end_line = range_info.end.line + 1
                    start_col = range_info.start.column + 1

                    # Get context
                    context_before = []
                    context_after = []
                    match_lines = []

                    if context_lines > 0:
                        ctx_start = max(0, start_line - 1 - context_lines)
                        ctx_end = min(len(lines), end_line + context_lines)
                        context_before = lines[ctx_start : start_line - 1]
                        context_after = lines[end_line : ctx_end]

                    # Get match text
                    match_lines = lines[start_line - 1 : end_line]

                    # Determine relative path
                    if search_path.is_dir():
                        rel_path = str(file_path.relative_to(search_path))
                    else:
                        rel_path = file_path.name

                    # Extract metavariable matches
                    metavars = {}
                    # Try common metavariable names
                    for var_name in ["NAME", "FUNC", "CLASS", "BASE", "MODULE",
                                     "DECORATOR", "METHOD", "OBJ", "E", "RECV",
                                     "TRAIT", "TYPE", "EXPR"]:
                        try:
                            match_node = found.get_match(var_name)
                            if match_node:
                                metavars[f"${var_name}"] = match_node.text()
                        except Exception:
                            pass

                    matches.append({
                        "file": rel_path,
                        "absolute_path": str(file_path),
                        "start_line": start_line,
                        "end_line": end_line,
                        "column": start_col,
                        "text": "\n".join(match_lines),
                        "node_kind": found.kind(),
                        "metavariables": metavars if metavars else None,
                        "context_before": context_before if context_before else None,
                        "context_after": context_after if context_after else None,
                    })

            except Exception as e:
                logger.debug("Error processing file %s: %s", file_path, e)
                errors.append(f"{file_path}: {str(e)}")
                continue

        result = {
            "matches": matches,
            "total_matches": total_matches,
            "files_with_matches": files_with_matches,
            "files_searched": files_searched,
            "truncated": truncated,
            "pattern": pattern,
            "language": language,
            "path": str(search_path),
            "_telemetry": {
                "jaato.ast.operation": "search",
                "jaato.ast.total_matches": total_matches,
                "jaato.ast.files_with_matches": files_with_matches,
                "jaato.ast.files_searched": files_searched,
                "jaato.ast.language": language,
            },
        }

        if errors and len(errors) <= 5:
            result["errors"] = errors

        return result

    def _should_exclude(self, path: Path, root: Path) -> bool:
        """Check if a path should be excluded from search.

        Args:
            path: File path to check.
            root: Root search directory.

        Returns:
            True if the path should be excluded.
        """
        try:
            rel_parts = path.relative_to(root).parts
            for part in rel_parts:
                if part in self._exclude_dirs:
                    return True
                if part.startswith("."):
                    return True
        except ValueError:
            pass
        return False

    # --- StreamingCapable implementation ---

    def supports_streaming(self, tool_name: str) -> bool:
        """Check if a tool supports streaming execution.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if ast_search (streaming yields matches incrementally).
        """
        return tool_name == "ast_search"

    def get_streaming_tool_names(self) -> List[str]:
        """Get list of tools that support streaming.

        Returns:
            List containing 'ast_search'.
        """
        return ["ast_search"]

    async def execute_streaming(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        on_chunk: Optional[ChunkCallback] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute AST search with streaming results.

        Yields matches as they're found, allowing the model to act on
        partial results while the search continues through files.

        Args:
            tool_name: Must be "ast_search".
            arguments: Tool arguments (pattern, language, path, etc.).
            on_chunk: Optional callback invoked for each chunk.

        Yields:
            StreamChunk objects for each match found.
        """
        if tool_name != "ast_search":
            yield StreamChunk(
                content=f"Streaming not supported for tool: {tool_name}",
                chunk_type="error",
            )
            return

        # Check if ast-grep is available
        if not self._ensure_ast_grep():
            yield StreamChunk(
                content="ast-grep-py is not installed. Install with: pip install ast-grep-py",
                chunk_type="error",
            )
            return

        from ast_grep_py import SgRoot

        pattern = arguments.get("pattern", "")
        language = arguments.get("language")
        path = arguments.get("path") or self._workspace_path
        file_pattern = arguments.get("file_pattern")
        context_lines = arguments.get("context_lines", self._context_lines)
        max_results = arguments.get("max_results", self._max_results)

        if not pattern:
            yield StreamChunk(content="Error: Pattern is required", chunk_type="error")
            return

        if not path:
            yield StreamChunk(content="Error: No path specified and no workspace configured", chunk_type="error")
            return

        # Sandbox check on the search root — see the blocking variant.
        if not self._is_path_allowed(path):
            yield StreamChunk(
                content=f"Error: Path not allowed: {path}",
                chunk_type="error",
            )
            return

        search_path = Path(path).resolve()
        if not search_path.exists():
            yield StreamChunk(
                content=f"Error: Path does not exist: {path}",
                chunk_type="error",
            )
            return

        # Yield initial progress chunk
        start_chunk = StreamChunk(
            content=f"Starting AST search for pattern: {pattern}",
            chunk_type="progress",
            metadata={"pattern": pattern, "path": str(search_path)},
        )
        if on_chunk:
            on_chunk(start_chunk)
        yield start_chunk

        # Determine files to search (containment applied per result)
        files_to_search, language, selection_error = self._select_files(
            search_path, file_pattern, language
        )
        if selection_error:
            yield StreamChunk(
                content=f"Error: {selection_error}",
                chunk_type="error",
            )
            return

        # Stream matches as they're found
        match_count = 0
        files_searched = 0
        files_with_matches = 0
        sequence = 1

        for file_path in files_to_search:
            if match_count >= max_results:
                break

            try:
                file_lang = language
                if not file_lang:
                    ext = file_path.suffix.lower()
                    file_lang = EXTENSION_TO_LANGUAGE.get(ext)
                    if not file_lang:
                        continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except (OSError, PermissionError):
                    continue

                files_searched += 1

                try:
                    root = SgRoot(content, file_lang)
                    node = root.root()
                    found_nodes = node.find_all(pattern=pattern)
                except Exception:
                    continue

                if not found_nodes:
                    continue

                files_with_matches += 1
                lines = content.splitlines()

                for found in found_nodes:
                    if match_count >= max_results:
                        break

                    match_count += 1

                    range_info = found.range()
                    start_line = range_info.start.line + 1
                    end_line = range_info.end.line + 1
                    start_col = range_info.start.column + 1

                    context_before = []
                    context_after = []
                    if context_lines > 0:
                        ctx_start = max(0, start_line - 1 - context_lines)
                        ctx_end = min(len(lines), end_line + context_lines)
                        context_before = lines[ctx_start : start_line - 1]
                        context_after = lines[end_line : ctx_end]

                    match_lines = lines[start_line - 1 : end_line]

                    if search_path.is_dir():
                        rel_path = str(file_path.relative_to(search_path))
                    else:
                        rel_path = file_path.name

                    # Extract metavariables
                    metavars = {}
                    for var_name in ["NAME", "FUNC", "CLASS", "BASE", "MODULE",
                                     "DECORATOR", "METHOD", "OBJ", "E", "RECV",
                                     "TRAIT", "TYPE", "EXPR"]:
                        try:
                            match_node = found.get_match(var_name)
                            if match_node:
                                metavars[f"${var_name}"] = match_node.text()
                        except Exception:
                            pass

                    # Build match chunk
                    match_data = {
                        "file": rel_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "column": start_col,
                        "text": "\n".join(match_lines),
                        "node_kind": found.kind(),
                    }
                    if metavars:
                        match_data["metavariables"] = metavars
                    if context_before:
                        match_data["context_before"] = context_before
                    if context_after:
                        match_data["context_after"] = context_after

                    chunk = StreamChunk(
                        content=f"{rel_path}:{start_line}: {match_lines[0] if match_lines else ''}",
                        chunk_type="match",
                        sequence=sequence,
                        metadata=match_data,
                    )
                    sequence += 1

                    if on_chunk:
                        on_chunk(chunk)
                    yield chunk

            except Exception as e:
                logger.debug("Error processing file %s: %s", file_path, e)
                continue

        # Yield final summary chunk
        summary_chunk = StreamChunk(
            content=f"Search complete: {match_count} matches in {files_with_matches} files ({files_searched} files searched)",
            chunk_type="summary",
            sequence=sequence,
            metadata={
                "total_matches": match_count,
                "files_with_matches": files_with_matches,
                "files_searched": files_searched,
                "truncated": match_count >= max_results,
            },
        )
        if on_chunk:
            on_chunk(summary_chunk)
        yield summary_chunk


def create_plugin() -> ASTSearchPlugin:
    """Factory function for plugin discovery.

    Returns:
        A new ASTSearchPlugin instance.
    """
    return ASTSearchPlugin()


__all__ = ["ASTSearchPlugin", "create_plugin"]
