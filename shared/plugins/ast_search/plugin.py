"""AST Search Plugin - Structural code search using AST patterns.

This plugin provides semantic code search capabilities using ast-grep-py,
allowing searches based on code structure rather than just text patterns.

Features:
- Multi-language support (Python, JavaScript, TypeScript, Go, Rust, Java, C, etc.)
- Pattern-based search using code-like syntax (e.g., "def $NAME($$$): $$$")
- Node type filtering (function_definition, class_definition, etc.)
- Context lines for surrounding code
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..background.mixin import BackgroundCapableMixin
from ..base import UserCommand
from ..model_provider.types import ToolSchema

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


class ASTSearchPlugin(BackgroundCapableMixin):
    """Plugin providing AST-based structural code search.

    This plugin offers semantic code search using ast-grep-py:
    - ast_search: Search code by AST patterns

    Supports multi-language codebases with pattern-based queries
    that understand code structure rather than just text.
    """

    def __init__(self):
        """Initialize the AST search plugin."""
        super().__init__(max_workers=2, default_timeout=DEFAULT_TIMEOUT_SECONDS)
        self._initialized = False
        self._ast_grep_available: Optional[bool] = None
        self._max_results = DEFAULT_MAX_RESULTS
        self._context_lines = DEFAULT_CONTEXT_LINES
        self._exclude_dirs = list(DEFAULT_EXCLUDE_DIRS)

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

        self._initialized = True
        logger.info(
            "ASTSearchPlugin initialized (max_results=%d, context_lines=%d)",
            self._max_results,
            self._context_lines,
        )

    def shutdown(self) -> None:
        """Shutdown the plugin and release resources."""
        self._shutdown_bg_executor()
        self._initialized = False
        logger.info("ASTSearchPlugin shutdown")

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
                                "go, rust, java, c, cpp, ruby, kotlin, swift, etc."
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
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor functions for each tool."""
        return {
            "ast_search": self._execute_ast_search,
        }

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
        """Estimate execution duration based on arguments."""
        if tool_name == "ast_search":
            path = arguments.get("path", ".")
            if Path(path).is_dir():
                return 15.0  # Directory search takes longer
            return 2.0
        return None

    # --- Tool implementation ---

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
        path = args.get("path", os.getcwd())
        file_pattern = args.get("file_pattern")
        context_lines = args.get("context_lines", self._context_lines)
        max_results = args.get("max_results", self._max_results)

        if not pattern:
            return {"error": "Pattern is required", "matches": [], "total_matches": 0}

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
            # Auto-detect language from extension if not specified
            if not language:
                ext = search_path.suffix.lower()
                language = EXTENSION_TO_LANGUAGE.get(ext)
                if not language:
                    return {
                        "error": f"Could not detect language for extension: {ext}",
                        "matches": [],
                        "total_matches": 0,
                    }
        else:
            # Directory search
            if file_pattern:
                # Use provided glob pattern
                for match in search_path.glob(file_pattern):
                    if match.is_file() and not self._should_exclude(match, search_path):
                        files_to_search.append(match)
            elif language:
                # Use language-specific extensions
                extensions = LANGUAGE_EXTENSIONS.get(language, [])
                if not extensions:
                    return {
                        "error": f"Unknown language: {language}",
                        "matches": [],
                        "total_matches": 0,
                    }
                for ext in extensions:
                    for match in search_path.rglob(f"*{ext}"):
                        if match.is_file() and not self._should_exclude(match, search_path):
                            files_to_search.append(match)
            else:
                # Search all supported file types
                for match in search_path.rglob("*"):
                    if match.is_file():
                        ext = match.suffix.lower()
                        if ext in EXTENSION_TO_LANGUAGE:
                            if not self._should_exclude(match, search_path):
                                files_to_search.append(match)

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


def create_plugin() -> ASTSearchPlugin:
    """Factory function for plugin discovery.

    Returns:
        A new ASTSearchPlugin instance.
    """
    return ASTSearchPlugin()


__all__ = ["ASTSearchPlugin", "create_plugin"]
