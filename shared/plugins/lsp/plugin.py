"""LSP tool plugin for code intelligence via Language Server Protocol."""

import asyncio
import json
import os
import queue
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..base import (
    UserCommand, CommandParameter, CommandCompletion,
    ToolResultEnrichmentResult
)
from ..model_provider.types import ToolSchema
from ..subagent.config import expand_variables
from .lsp_client import (
    LSPClient, ServerConfig, Location, Diagnostic, Hover,
    TextEdit, WorkspaceEdit, CodeAction, Range, Position
)


# Tools that write/modify files and should trigger LSP diagnostics
FILE_WRITING_TOOLS = {
    'updateFile',
    'writeNewFile',
    'lsp_rename_symbol',
    'lsp_apply_code_action',
}

# Symbol kinds that represent exportable/referenceable entities
# Used by get_file_dependents() to find symbols worth checking for external references
# See LSP SymbolKind enum: https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#symbolKind
DEPENDENCY_SYMBOL_KINDS = {
    2,   # Module
    5,   # Class
    6,   # Method
    10,  # Enum
    11,  # Interface
    12,  # Function
    14,  # Constant
    23,  # Struct
}

# Mapping of file extensions to language IDs for LSP server matching
EXT_TO_LANGUAGE = {
    '.py': 'python',
    '.pyw': 'python',
    '.pyi': 'python',
    '.js': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    '.jsx': 'javascriptreact',
    '.ts': 'typescript',
    '.mts': 'typescript',
    '.cts': 'typescript',
    '.tsx': 'typescriptreact',
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.hxx': 'cpp',
    '.cs': 'csharp',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.scala': 'scala',
    '.lua': 'lua',
    '.r': 'r',
    '.R': 'r',
    '.zig': 'zig',
    '.vue': 'vue',
    '.svelte': 'svelte',
}


# Message types for background thread communication
MSG_CALL_METHOD = 'call_method'
MSG_CONNECT_SERVER = 'connect_server'
MSG_DISCONNECT_SERVER = 'disconnect_server'
MSG_RELOAD_CONFIG = 'reload_config'

# Log levels
LOG_INFO = 'INFO'
LOG_DEBUG = 'DEBUG'
LOG_ERROR = 'ERROR'
LOG_WARN = 'WARN'

MAX_LOG_ENTRIES = 500


def _uri_to_file_path(uri: str) -> str:
    """Convert a file URI to a local file path."""
    if uri.startswith('file://'):
        path = uri[7:]
        if os.name == 'nt' and path.startswith('/'):
            path = path[1:]
        return path
    return uri


def _apply_text_edits_to_content(content: str, edits: List[TextEdit]) -> str:
    """Apply a list of text edits to content.

    Edits are applied in reverse order (bottom-to-top, right-to-left)
    to preserve position validity.
    """
    lines = content.split('\n')

    # Sort edits in reverse order to apply from bottom to top
    sorted_edits = sorted(
        edits,
        key=lambda e: (e.range.start.line, e.range.start.character),
        reverse=True
    )

    for edit in sorted_edits:
        start_line = edit.range.start.line
        start_char = edit.range.start.character
        end_line = edit.range.end.line
        end_char = edit.range.end.character

        # Ensure line indices are within bounds
        if start_line >= len(lines):
            continue

        if end_line >= len(lines):
            end_line = len(lines) - 1
            end_char = len(lines[end_line]) if lines else 0

        # Get the parts we're keeping
        before = lines[start_line][:start_char] if start_line < len(lines) else ""
        after = lines[end_line][end_char:] if end_line < len(lines) else ""

        # Split the new text into lines
        new_text_lines = edit.new_text.split('\n')

        if len(new_text_lines) == 1:
            # Single line replacement
            lines[start_line] = before + new_text_lines[0] + after
            # Remove any lines between start and end
            del lines[start_line + 1:end_line + 1]
        else:
            # Multi-line replacement
            new_text_lines[0] = before + new_text_lines[0]
            new_text_lines[-1] = new_text_lines[-1] + after

            # Replace the range with new lines
            lines[start_line:end_line + 1] = new_text_lines

    return '\n'.join(lines)


def apply_workspace_edit(
    workspace_edit: WorkspaceEdit,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Apply a workspace edit to files on disk.

    Args:
        workspace_edit: The WorkspaceEdit to apply
        dry_run: If True, validate but don't actually write files

    Returns:
        Dict with:
            - success: bool indicating overall success
            - files_modified: list of file paths that were modified
            - changes: list of change descriptions per file
            - errors: list of any errors encountered
    """
    result: Dict[str, Any] = {
        "success": True,
        "files_modified": [],
        "changes": [],
        "errors": []
    }

    for uri, edits in workspace_edit.changes.items():
        file_path = _uri_to_file_path(uri)

        try:
            # Read the current file content
            if not os.path.isfile(file_path):
                result["errors"].append({
                    "file": file_path,
                    "error": "File not found"
                })
                result["success"] = False
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Apply edits
            new_content = _apply_text_edits_to_content(original_content, edits)

            # Count changes for reporting
            change_info = {
                "file": file_path,
                "edits_applied": len(edits),
                "lines_before": len(original_content.split('\n')),
                "lines_after": len(new_content.split('\n'))
            }

            if not dry_run:
                # Write the modified content back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                result["files_modified"].append(file_path)

            result["changes"].append(change_info)

        except IOError as e:
            result["errors"].append({
                "file": file_path,
                "error": str(e)
            })
            result["success"] = False
        except Exception as e:
            result["errors"].append({
                "file": file_path,
                "error": f"Unexpected error: {e}"
            })
            result["success"] = False

    return result


@dataclass
class LogEntry:
    """A single log entry for LSP interactions."""
    timestamp: datetime
    level: str
    server: Optional[str]
    event: str
    details: Optional[str] = None

    def format(self, include_timestamp: bool = True) -> str:
        parts = []
        if include_timestamp:
            parts.append(self.timestamp.strftime('%H:%M:%S.%f')[:-3])
        parts.append(f"[{self.level}]")
        if self.server:
            parts.append(f"[{self.server}]")
        parts.append(self.event)
        if self.details:
            parts.append(f"- {self.details}")
        return ' '.join(parts)


class LogCapture:
    """File-like object that captures LSP server stderr and routes to log buffer.

    This class uses an OS pipe to provide a real file descriptor that can be
    passed to subprocess stderr. A background thread reads from the pipe and
    routes messages to the LSP plugin's internal log buffer via a callback.

    The asyncio subprocess requires a file-like object with a valid fileno()
    for stderr redirection. Pure Python wrappers don't work because subprocess
    needs a real file descriptor.
    """

    def __init__(self, log_callback: Callable[[str, str, Optional[str], Optional[str]], None]):
        """Initialize the log capture with an OS pipe.

        Args:
            log_callback: Function to call with (level, event, server, details).
                         Should match the signature of LSPToolPlugin._log_event.
        """
        self._log_callback = log_callback
        # Create a pipe - write end for subprocess, read end for our thread
        self._read_fd, self._write_fd = os.pipe()
        # Wrap write end as a file object (this is what fileno() returns)
        self._write_file = os.fdopen(self._write_fd, 'w', encoding='utf-8')
        self._closed = False
        self._reader_thread: Optional[threading.Thread] = None
        # Start background thread to read from pipe
        self._start_reader()

    def _start_reader(self) -> None:
        """Start background thread to read from the pipe."""
        def reader():
            try:
                # Wrap read end as file for line-by-line reading
                with os.fdopen(self._read_fd, 'r', encoding='utf-8', errors='replace') as read_file:
                    for line in read_file:
                        line = line.rstrip('\n\r')
                        if line:
                            self._log_callback(LOG_DEBUG, "Server output", None, line)
            except (OSError, ValueError):
                # Pipe closed or other error during shutdown
                pass

        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._reader_thread.start()

    def write(self, text: str) -> int:
        """Write text to the pipe (called for compatibility)."""
        if self._closed:
            return 0
        try:
            self._write_file.write(text)
            self._write_file.flush()
            return len(text)
        except (OSError, ValueError):
            return 0

    def flush(self) -> None:
        """Flush the write buffer."""
        if not self._closed:
            try:
                self._write_file.flush()
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        """Close the log capture and stop the reader thread."""
        if self._closed:
            return
        self._closed = True
        try:
            self._write_file.close()
        except (OSError, ValueError):
            pass
        # Reader thread will exit when it sees the pipe closed

    def fileno(self) -> int:
        """Return the write end file descriptor for subprocess redirection."""
        return self._write_fd


class LSPToolPlugin:
    """Plugin that provides LSP (Language Server Protocol) tool execution.

    This plugin connects to LSP servers defined in .lsp.json and exposes
    code intelligence tools to the AI model. It runs a background thread
    with an asyncio event loop to handle the async LSP protocol.
    """

    def __init__(self):
        self._clients: Dict[str, LSPClient] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._request_queue: Optional[queue.Queue] = None
        self._response_queue: Optional[queue.Queue] = None
        self._initialized = False
        self._config_path: Optional[str] = None  # Explicit config path from plugin_configs
        self._custom_config_path: Optional[str] = None  # User-specified path
        self._workspace_path: Optional[str] = None  # Client's working directory
        self._config_cache: Dict[str, Any] = {}
        self._connected_servers: set = set()
        self._failed_servers: Dict[str, str] = {}
        self._log: deque = deque(maxlen=MAX_LOG_ENTRIES)
        self._log_lock = threading.Lock()
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Stderr capture for LSP server output
        self._errlog: Optional[LogCapture] = None

    def _log_event(
        self,
        level: str,
        event: str,
        server: Optional[str] = None,
        details: Optional[str] = None
    ) -> None:
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            server=server,
            event=event,
            details=details
        )
        with self._log_lock:
            self._log.append(entry)

    @property
    def name(self) -> str:
        return "lsp"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging.

        Uses JAATO_TRACE_LOG env var, or defaults to /tmp/rich_client_trace.log.
        Silently skips if trace file cannot be written.
        """
        import tempfile
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [LSP{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass  # Silently skip if trace file cannot be written

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the LSP plugin by starting the background thread.

        Args:
            config: Optional configuration dict. Supports:
                - config_path: Path to .lsp.json file (overrides default search)
                - workspace_path: Client's working directory for finding .lsp.json
                - agent_name: Name for trace logging
        """
        if self._initialized:
            return

        # Expand variables in config values (e.g., ${projectPath}, ${workspaceRoot})
        config = expand_variables(config) if config else {}

        # Extract config values
        self._agent_name = config.get('agent_name')
        self._custom_config_path = config.get('config_path')
        self._workspace_path = config.get('workspace_path')

        self._trace("initialize: starting background thread")
        self._ensure_thread()
        self._initialized = True
        self._trace(f"initialize: connected_servers={list(self._connected_servers)}")

    def set_workspace_path(self, path: str) -> None:
        """Set the workspace path for finding config files.

        This should be called when the client's working directory changes.
        It will trigger a reload of the config file from the new location.
        """
        if path != self._workspace_path:
            self._workspace_path = path
            self._trace(f"workspace_path changed to: {path}")
            # Force reload config on next access
            if self._initialized:
                self._load_config_cache(force=True)

    def shutdown(self) -> None:
        """Shutdown the LSP plugin and clean up resources."""
        self._trace("shutdown: cleaning up resources")
        if self._request_queue:
            self._request_queue.put((None, None))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        # Close stderr capture
        if self._errlog:
            self._errlog.close()
            self._errlog = None
        self._clients = {}
        self._loop = None
        self._thread = None
        self._request_queue = None
        self._response_queue = None
        self._initialized = False
        self._connected_servers = set()
        self._failed_servers = {}

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return ToolSchemas for LSP tools."""
        if not self._initialized:
            self.initialize()

        return [
            ToolSchema(
                name="lsp_goto_definition",
                description=(
                    "Find the definition of a symbol (class, method, variable, etc.). "
                    "Returns the file path and line number where the symbol is defined. "
                    "Useful for navigating to where something is implemented."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Name of the symbol to find (e.g., 'UserService', 'processOrder')"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional: file to search in for context (helps with disambiguation)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            ToolSchema(
                name="lsp_find_references",
                description=(
                    "Find all references to a symbol across the codebase. "
                    "Use for impact analysis before modifying a method or class - "
                    "shows all callers/usages. More accurate than grep for understanding "
                    "true dependencies (understands scope, not just text matching)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Name of the symbol to find references for"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional: file where the symbol is defined (helps with disambiguation)"
                        },
                        "include_declaration": {
                            "type": "boolean",
                            "description": "Include the declaration in results (default: true)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            ToolSchema(
                name="lsp_hover",
                description=(
                    "Get type information and documentation for a symbol. "
                    "Use to verify method signatures, parameter types, and return types "
                    "when integrating with existing code - faster than reading source files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Name of the symbol to get info for"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional: file containing the symbol (helps with disambiguation)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            ToolSchema(
                name="lsp_get_diagnostics",
                description=(
                    "**CODE VALIDATOR/LINTER**: Get errors, warnings, and issues for a file. "
                    "Use this to validate generated or modified code before reporting success. "
                    "This IS your linting tool - do not request a separate linter. "
                    "Returns syntax errors, type errors, missing imports, and style issues in milliseconds. "
                    "ALWAYS call this after writing code and BEFORE reporting completion."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the source file"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            ToolSchema(
                name="lsp_document_symbols",
                description="Get all symbols (functions, classes, variables) defined in a file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the source file"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            ToolSchema(
                name="lsp_workspace_symbols",
                description="Search for symbols across the entire workspace/project.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for symbol names"
                        }
                    },
                    "required": ["query"]
                }
            ),
            ToolSchema(
                name="lsp_rename_symbol",
                description=(
                    "Rename a symbol across all files in the workspace. "
                    "By default performs a dry-run showing what would change. "
                    "Set apply=true to actually apply the changes. "
                    "Returns detailed information about which files were modified."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Current name of the symbol to rename"
                        },
                        "new_name": {
                            "type": "string",
                            "description": "New name for the symbol"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional: file where the symbol is defined (helps with disambiguation)"
                        },
                        "apply": {
                            "type": "boolean",
                            "description": "If true, apply the rename. If false (default), preview only."
                        }
                    },
                    "required": ["symbol", "new_name"]
                }
            ),
            ToolSchema(
                name="lsp_get_code_actions",
                description=(
                    "Get available code actions (refactorings, quick fixes) for a code region. "
                    "Returns a list of available actions that can be applied with lsp_apply_code_action. "
                    "Use this to discover what refactoring operations the language server supports "
                    "(e.g., extract method, extract variable, inline, organize imports)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the source file"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Start line of the selection (1-indexed)"
                        },
                        "start_column": {
                            "type": "integer",
                            "description": "Start column of the selection (1-indexed)"
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "End line of the selection (1-indexed)"
                        },
                        "end_column": {
                            "type": "integer",
                            "description": "End column of the selection (1-indexed)"
                        },
                        "only_refactorings": {
                            "type": "boolean",
                            "description": "If true, only return refactoring actions (not quick fixes)"
                        }
                    },
                    "required": ["file_path", "start_line", "start_column", "end_line", "end_column"]
                }
            ),
            ToolSchema(
                name="lsp_apply_code_action",
                description=(
                    "Apply a code action (refactoring or quick fix) by its title. "
                    "First use lsp_get_code_actions to discover available actions, "
                    "then call this tool with the exact title of the action to apply. "
                    "Returns details of files modified."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the source file"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Start line of the selection (1-indexed)"
                        },
                        "start_column": {
                            "type": "integer",
                            "description": "Start column of the selection (1-indexed)"
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "End line of the selection (1-indexed)"
                        },
                        "end_column": {
                            "type": "integer",
                            "description": "End column of the selection (1-indexed)"
                        },
                        "action_title": {
                            "type": "string",
                            "description": "Exact title of the code action to apply (from lsp_get_code_actions)"
                        }
                    },
                    "required": ["file_path", "start_line", "start_column", "end_line", "end_column", "action_title"]
                }
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mappings for LSP tools."""
        if not self._initialized:
            self.initialize()

        return {
            "lsp_goto_definition": self._exec_goto_definition,
            "lsp_find_references": self._exec_find_references,
            "lsp_hover": self._exec_hover,
            "lsp_get_diagnostics": self._exec_get_diagnostics,
            "lsp_document_symbols": self._exec_document_symbols,
            "lsp_workspace_symbols": self._exec_workspace_symbols,
            "lsp_rename_symbol": self._exec_rename_symbol,
            "lsp_get_code_actions": self._exec_get_code_actions,
            "lsp_apply_code_action": self._exec_apply_code_action,
            "lsp": lambda args: self.execute_user_command('lsp', args),
        }

    def get_system_instructions(self) -> Optional[str]:
        return """## CODE VALIDATION / LINTING (AUTOMATIC + MANUAL)

**AUTOMATIC DIAGNOSTICS**: When you use file-writing tools (updateFile, writeNewFile),
LSP diagnostics are automatically run and appended to the tool result. Look for the
"LSP Diagnostics (auto-check)" section after file operations - if errors are found,
fix them immediately before proceeding.

**MANUAL DIAGNOSTICS**: For files not covered by automatic checks, or to re-check:
  lsp_get_diagnostics(file_path="/path/to/file.py")

This returns:
- Syntax errors
- Type errors
- Warnings
- Code style issues
- Any problems the language server detects

**Important**: If you see "❌ Error(s) - MUST FIX" in the automatic diagnostics,
you MUST fix those errors before reporting success to the user.

**Validation workflow:**
1. Generate or modify code (automatic diagnostics will run)
2. Check the appended "LSP Diagnostics" section for errors
3. If errors found, fix them immediately
4. Only report success when diagnostics are clean (or only warnings remain)

---

## LSP Tools Reference

Symbol-based tools (just provide the symbol name):
- lsp_goto_definition(symbol): Find where a symbol is defined
- lsp_find_references(symbol): Find all usages across the codebase
- lsp_hover(symbol): Get type info and documentation

Refactoring tools:
- lsp_rename_symbol(symbol, new_name, apply=True): Rename symbol across all files
  - Set apply=False (default) to preview changes, apply=True to apply them
- lsp_get_code_actions(file_path, start_line, start_column, end_line, end_column):
  - Discover available refactorings for a code region (extract method, inline, etc.)
- lsp_apply_code_action(file_path, ..., action_title): Apply a discovered code action

File-based tools:
- lsp_get_diagnostics(file_path): **YOUR LINTER** - Get errors/warnings for validation.
  Use AFTER writing code and BEFORE reporting completion. This IS the validator tool.
- lsp_document_symbols(file_path): List all symbols in a file

Query-based tools:
- lsp_workspace_symbols(query): Search for symbols across the project

Use 'lsp status' to see connected language servers and their capabilities."""

    def get_auto_approved_tools(self) -> List[str]:
        # Read-only tools are auto-approved
        # lsp_rename_symbol and lsp_apply_code_action modify files - NOT auto-approved
        return [
            "lsp_goto_definition",
            "lsp_find_references",
            "lsp_hover",
            "lsp_get_diagnostics",
            "lsp_document_symbols",
            "lsp_workspace_symbols",
            "lsp_get_code_actions",  # Read-only: just lists available actions
            "lsp",
        ]

    # ==================== Tool Result Enrichment ====================

    def subscribes_to_tool_result_enrichment(self) -> bool:
        """Subscribe to tool result enrichment to auto-run diagnostics after file writes.

        When enabled, the LSP plugin will automatically run diagnostics on files
        that are modified by file-writing tools (updateFile, writeNewFile, etc.)
        and append diagnostic information to the tool result.
        """
        return True

    def get_tool_result_enrichment_priority(self) -> int:
        """Run after basic file operations but before other enrichment.

        Priority 30 ensures diagnostics are added early in the enrichment chain.
        """
        return 30

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str
    ) -> ToolResultEnrichmentResult:
        """Enrich file-writing tool results with LSP diagnostics.

        If the tool wrote or modified a file that is supported by an LSP server,
        this method automatically runs diagnostics on the file and appends the
        results to the tool output. This enables the model to see any errors
        immediately and react in the same turn.

        Args:
            tool_name: Name of the tool that produced the result.
            result: The tool's output as a string (JSON-serialized dict).

        Returns:
            ToolResultEnrichmentResult with diagnostics appended if applicable.
        """
        # Only process file-writing tools
        if tool_name not in FILE_WRITING_TOOLS:
            return ToolResultEnrichmentResult(result=result)

        self._trace(f"enrich_tool_result: checking {tool_name}")

        # Skip if no LSP servers are connected
        if not self._connected_servers:
            self._trace(f"enrich_tool_result: skipped - no servers connected")
            return ToolResultEnrichmentResult(result=result)

        # Parse the result to extract file paths
        file_paths = self._extract_file_paths_from_result(tool_name, result)
        if not file_paths:
            self._trace(f"enrich_tool_result: no file paths found in result")
            return ToolResultEnrichmentResult(result=result)

        self._trace(f"enrich_tool_result: found files {file_paths}")

        # Filter to files that have LSP support
        supported_files = self._filter_supported_files(file_paths)
        if not supported_files:
            self._trace(f"enrich_tool_result: no supported file types")
            return ToolResultEnrichmentResult(result=result)

        self._trace(f"enrich_tool_result: checking diagnostics for {supported_files}")

        # Run diagnostics on each file and collect results
        all_diagnostics = {}
        for file_path in supported_files:
            diags = self._get_diagnostics_for_file(file_path)
            if diags:
                all_diagnostics[file_path] = diags

        self._trace(f"enrich_tool_result: found {len(all_diagnostics)} files with diagnostics")

        # Build enriched result with diagnostic summary
        enriched_result = self._build_enriched_result(result, all_diagnostics)
        metadata = {
            "files_checked": list(supported_files),
            "files_with_diagnostics": list(all_diagnostics.keys()),
            "total_errors": sum(
                sum(1 for d in diags if d.get("severity") == "Error")
                for diags in all_diagnostics.values()
            ),
            "total_warnings": sum(
                sum(1 for d in diags if d.get("severity") == "Warning")
                for diags in all_diagnostics.values()
            ),
        }

        return ToolResultEnrichmentResult(result=enriched_result, metadata=metadata)

    def _extract_file_paths_from_result(
        self,
        tool_name: str,
        result: str
    ) -> List[str]:
        """Extract file paths from a tool result.

        Different tools return file paths in different formats:
        - updateFile/writeNewFile: {"path": "...", "success": true}
        - lsp_rename_symbol: {"files_modified": [...], "changes": [...]}
        - lsp_apply_code_action: {"files_modified": [...], "changes": [...]}

        Args:
            tool_name: The tool that produced the result.
            result: The JSON-serialized result string.

        Returns:
            List of file paths found in the result.
        """
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return []

        # Check for error result
        if isinstance(data, dict) and data.get("error"):
            return []

        file_paths = []

        if tool_name in ('updateFile', 'writeNewFile'):
            # Single file operations
            path = data.get("path")
            if path:
                file_paths.append(path)

        elif tool_name in ('lsp_rename_symbol', 'lsp_apply_code_action'):
            # Workspace edit operations - multiple files
            files_modified = data.get("files_modified", [])
            file_paths.extend(files_modified)

            # Also check changes array for file paths
            changes = data.get("changes", [])
            for change in changes:
                if isinstance(change, dict) and change.get("file"):
                    file_paths.append(change["file"])

        return file_paths

    def _filter_supported_files(self, file_paths: List[str]) -> List[str]:
        """Filter file paths to those supported by connected LSP servers.

        Args:
            file_paths: List of file paths to check.

        Returns:
            List of file paths that have LSP support.
        """
        supported = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in EXT_TO_LANGUAGE:
                # Check if we have a server for this language
                lang = EXT_TO_LANGUAGE[ext]
                if self._has_server_for_language(lang):
                    supported.append(file_path)
        return supported

    def _has_server_for_language(self, language: str) -> bool:
        """Check if we have a connected LSP server for a language.

        Args:
            language: Language ID (e.g., 'python', 'typescript').

        Returns:
            True if a server is connected that supports this language.
        """
        for name in self._connected_servers:
            client = self._clients.get(name)
            if client:
                # Check by language_id config
                if client.config.language_id == language:
                    return True
                # Also check by server name
                if language.lower() in name.lower():
                    return True
        return False

    def _get_diagnostics_for_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Get LSP diagnostics for a file.

        Args:
            file_path: Path to the file to check.

        Returns:
            List of diagnostic dictionaries with severity, message, line, etc.
        """
        try:
            result = self._execute_method('get_diagnostics', {'file_path': file_path})
            if isinstance(result, dict) and result.get("error"):
                self._trace(f"_get_diagnostics_for_file: error for {file_path}: {result['error']}")
                return []
            if isinstance(result, list):
                return result
            return []
        except Exception as e:
            self._trace(f"_get_diagnostics_for_file: exception for {file_path}: {e}")
            return []

    def _build_enriched_result(
        self,
        original_result: str,
        diagnostics: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Build an enriched result string with diagnostic information.

        Args:
            original_result: The original tool result (JSON string).
            diagnostics: Dict mapping file paths to their diagnostics.

        Returns:
            Enriched result string with diagnostic summary appended.
        """
        if not diagnostics:
            return original_result

        # Count by severity
        errors = []
        warnings = []
        infos = []

        for file_path, diags in diagnostics.items():
            for d in diags:
                severity = d.get("severity", "Unknown")
                entry = {
                    "file": file_path,
                    "line": d.get("line"),
                    "message": d.get("message"),
                    "source": d.get("source"),
                }
                if severity == "Error":
                    errors.append(entry)
                elif severity == "Warning":
                    warnings.append(entry)
                else:
                    infos.append(entry)

        # Build diagnostic summary
        lines = [original_result, "\n\n---\n## LSP Diagnostics (auto-check)"]

        if errors:
            lines.append(f"\n### ❌ {len(errors)} Error(s) - MUST FIX:")
            for e in errors[:10]:  # Limit to first 10
                lines.append(f"- {e['file']}:{e['line']}: {e['message']}")
            if len(errors) > 10:
                lines.append(f"  ... and {len(errors) - 10} more errors")

        if warnings:
            lines.append(f"\n### ⚠️ {len(warnings)} Warning(s):")
            for w in warnings[:5]:  # Limit to first 5
                lines.append(f"- {w['file']}:{w['line']}: {w['message']}")
            if len(warnings) > 5:
                lines.append(f"  ... and {len(warnings) - 5} more warnings")

        if not errors and not warnings:
            lines.append("\n✅ No errors or warnings detected.")

        if errors:
            lines.append("\n**ACTION REQUIRED**: Fix the errors above before proceeding.")

        return "\n".join(lines)

    # ==================== Dependency Discovery ====================

    def get_file_dependents(self, file_path: str) -> List[str]:
        """Find files that depend on the given file via exported symbols.

        Uses LSP to discover which other files reference symbols defined in
        this file. This is useful for understanding the impact of changes
        and tracking related artifacts.

        Algorithm:
        1. Get all document symbols for the file
        2. Filter to "exportable" symbol kinds (Class, Function, etc.)
        3. For each symbol, find all references across the codebase
        4. Collect and deduplicate the files that contain those references

        Args:
            file_path: Path to the source file to analyze.

        Returns:
            List of file paths that depend on (reference) this file.
            Returns empty list if LSP is not available or file has no
            exported symbols with external references.
        """
        self._trace(f"get_file_dependents: analyzing {file_path}")

        if not self._initialized:
            self.initialize()

        if not self._connected_servers:
            self._trace("get_file_dependents: no servers connected")
            return []

        # Check if file type is supported
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in EXT_TO_LANGUAGE:
            self._trace(f"get_file_dependents: unsupported file type {ext}")
            return []

        # Ensure all files of the same type in the workspace are indexed
        # This is needed for find_references to work across files
        workspace_dir = os.path.dirname(os.path.abspath(file_path))
        self._trace(f"get_file_dependents: ensuring workspace indexed at {workspace_dir}")
        self._execute_method('_ensure_workspace_indexed', {
            'directory': workspace_dir,
            'extension': ext,  # Pass the file extension to filter by language
            'file_path': file_path  # Pass file_path so correct LSP server is selected
        })

        # Get document symbols
        symbols_result = self._execute_method('document_symbols', {'file_path': file_path})
        if isinstance(symbols_result, dict) and symbols_result.get("error"):
            self._trace(f"get_file_dependents: failed to get symbols: {symbols_result['error']}")
            return []

        if not isinstance(symbols_result, list):
            self._trace("get_file_dependents: no symbols found")
            return []

        # Filter to exportable symbol kinds
        exportable_symbols = [
            s for s in symbols_result
            if self._get_symbol_kind_value(s.get('kind', '')) in DEPENDENCY_SYMBOL_KINDS
        ]

        self._trace(f"get_file_dependents: found {len(exportable_symbols)} exportable symbols")

        if not exportable_symbols:
            return []

        # Collect all files that reference any of these symbols
        dependent_files: set = set()

        # Read file content once to check for import lines
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                file_lines = f.readlines()
        except (IOError, OSError):
            file_lines = []

        for symbol in exportable_symbols:
            symbol_name = symbol.get('name', '')
            if not symbol_name:
                continue

            # Parse location to get line and character (format: "path:line:character")
            location = symbol.get('location', '')
            parts = location.split(':')
            if len(parts) >= 2:
                try:
                    line = int(parts[1]) - 1  # Convert to 0-indexed
                    # Character position from LSP - may point to start of definition, not symbol name
                    character = int(parts[2]) if len(parts) >= 3 else 0
                except (ValueError, IndexError):
                    continue
            else:
                continue

            # Skip imported symbols - they are not defined in this file
            # Check if the symbol's line is an import statement
            if 0 <= line < len(file_lines):
                line_content = file_lines[line].strip()
                if line_content.startswith('import ') or line_content.startswith('from '):
                    self._trace(f"get_file_dependents: skipping imported symbol '{symbol_name}' (line: {line_content[:50]})")
                    continue

            # For SymbolInformation format, the character position often points to the
            # start of the entire definition (e.g., "def" in "def hello():") rather than
            # the symbol name. We need to find the actual position of the symbol name.
            character = self._find_symbol_name_in_line(file_path, line, symbol_name, character)

            self._trace(f"get_file_dependents: checking references for {symbol_name} at line {line}, char {character}")

            # Find references to this symbol
            refs_result = self._execute_method('find_references', {
                'file_path': file_path,
                'line': line,
                'character': character,
                'include_declaration': False  # Skip the definition itself
            })

            self._trace(f"get_file_dependents: find_references for {symbol_name} returned: {type(refs_result).__name__}, value={refs_result}")

            if isinstance(refs_result, dict) and refs_result.get("error"):
                # No references found is not an error for our purposes
                self._trace(f"get_file_dependents: find_references error: {refs_result.get('error')}")
                continue

            if isinstance(refs_result, list):
                self._trace(f"get_file_dependents: find_references returned {len(refs_result)} references")
                for ref in refs_result:
                    ref_file = ref.get('file', '')
                    self._trace(f"get_file_dependents: ref_file={ref_file}")
                    if ref_file and ref_file != file_path:
                        dependent_files.add(ref_file)

        self._trace(f"get_file_dependents: found {len(dependent_files)} dependent files")
        return list(dependent_files)

    def _get_symbol_kind_value(self, kind_name: str) -> int:
        """Convert a symbol kind name back to its numeric value.

        Args:
            kind_name: Human-readable kind name (e.g., 'Function', 'Class').

        Returns:
            The LSP SymbolKind numeric value, or 0 if not recognized.
        """
        kind_map = {
            "File": 1, "Module": 2, "Namespace": 3, "Package": 4,
            "Class": 5, "Method": 6, "Property": 7, "Field": 8,
            "Constructor": 9, "Enum": 10, "Interface": 11, "Function": 12,
            "Variable": 13, "Constant": 14, "String": 15, "Number": 16,
            "Boolean": 17, "Array": 18, "Object": 19, "Key": 20,
            "Null": 21, "EnumMember": 22, "Struct": 23, "Event": 24,
            "Operator": 25, "TypeParameter": 26
        }
        return kind_map.get(kind_name, 0)

    def _find_symbol_name_in_line(
        self,
        file_path: str,
        line_num: int,
        symbol_name: str,
        default_char: int
    ) -> int:
        """Find the character position of a symbol name within a specific line.

        For SymbolInformation format, the LSP range often covers the entire
        definition (e.g., the whole "def hello():" line) rather than just
        the symbol name. This method finds where the symbol name actually
        appears in the line.

        Args:
            file_path: Path to the source file.
            line_num: 0-indexed line number.
            symbol_name: Name of the symbol to find.
            default_char: Default character position to return if not found.

        Returns:
            The 0-indexed character position of the symbol name in the line.
        """
        import re

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            if 0 <= line_num < len(lines):
                line_content = lines[line_num]
                # Search for symbol name as a word boundary match
                pattern = re.compile(r'\b' + re.escape(symbol_name) + r'\b')
                match = pattern.search(line_content)
                if match:
                    self._trace(f"_find_symbol_name_in_line: found '{symbol_name}' at char {match.start()} (was {default_char})")
                    return match.start()
        except (IOError, OSError) as e:
            self._trace(f"_find_symbol_name_in_line: error reading {file_path}: {e}")

        return default_char

    def get_user_commands(self) -> List[UserCommand]:
        return [
            UserCommand(
                name="lsp",
                description="Manage LSP language servers",
                share_with_model=False,
                parameters=[
                    CommandParameter("subcommand", "Subcommand (list, status, connect, disconnect, reload)", required=False),
                    CommandParameter("rest", "Additional arguments", required=False, capture_rest=True),
                ]
            )
        ]

    def get_command_completions(self, command: str, args: List[str]) -> List[CommandCompletion]:
        if command != 'lsp':
            return []

        subcommands = [
            CommandCompletion('list', 'List configured LSP servers'),
            CommandCompletion('status', 'Show connection status'),
            CommandCompletion('connect', 'Connect to a server'),
            CommandCompletion('disconnect', 'Disconnect from a server'),
            CommandCompletion('reload', 'Reload configuration'),
            CommandCompletion('logs', 'Show interaction logs'),
            CommandCompletion('help', 'Show help'),
        ]

        if not args:
            return subcommands

        if len(args) == 1:
            partial = args[0].lower()
            return [c for c in subcommands if c.value.startswith(partial)]

        subcommand = args[0].lower()
        if subcommand in ('connect', 'disconnect', 'show'):
            self._load_config_cache()
            servers = self._config_cache.get('languageServers', {})
            partial = args[1].lower() if len(args) > 1 else ''
            completions = []
            for name in servers:
                if name.lower().startswith(partial):
                    if subcommand == 'connect' and name in self._connected_servers:
                        continue
                    if subcommand == 'disconnect' and name not in self._connected_servers:
                        continue
                    completions.append(CommandCompletion(name, f'{subcommand.capitalize()} {name}'))
            return completions

        return []

    def execute_user_command(self, command: str, args: Dict[str, Any]) -> str:
        if command != 'lsp':
            return f"Unknown command: {command}"

        subcommand = args.get('subcommand', '').lower()
        rest = args.get('rest', '').strip()

        if subcommand == 'list':
            return self._cmd_list()
        elif subcommand == 'status':
            return self._cmd_status()
        elif subcommand == 'connect':
            return self._cmd_connect(rest)
        elif subcommand == 'disconnect':
            return self._cmd_disconnect(rest)
        elif subcommand == 'reload':
            return self._cmd_reload()
        elif subcommand == 'logs':
            return self._cmd_logs(rest)
        elif subcommand == 'help' or subcommand == '':
            return self._cmd_help()
        else:
            return f"Unknown subcommand: {subcommand}\n\n{self._cmd_help()}"

    def _cmd_help(self) -> str:
        return """LSP Server Commands:

  lsp list              - List all configured LSP servers
  lsp status            - Show connection status of all servers
  lsp connect <name>    - Connect to a configured server
  lsp disconnect <name> - Disconnect from a running server
  lsp reload            - Reload configuration from .lsp.json
  lsp logs [clear]      - Show interaction logs

Configuration file: .lsp.json
Example:
{
  "languageServers": {
    "python": {
      "command": "pyright-langserver",
      "args": ["--stdio"],
      "languageId": "python"
    }
  }
}"""

    def _cmd_list(self) -> str:
        self._load_config_cache()
        servers = self._config_cache.get('languageServers', {})
        if not servers:
            return "No LSP servers configured. Create .lsp.json to configure servers."

        lines = ["Configured LSP servers:"]
        for name, spec in servers.items():
            status = "connected" if name in self._connected_servers else "disconnected"
            if name in self._failed_servers:
                status = f"failed: {self._failed_servers[name]}"
            cmd = spec.get('command', 'N/A')
            lines.append(f"  {name}: {cmd} [{status}]")
        return '\n'.join(lines)

    def _cmd_status(self) -> str:
        self._load_config_cache()
        servers = self._config_cache.get('languageServers', {})
        if not servers:
            return "No LSP servers configured."

        lines = ["LSP Server Status:", "-" * 50]
        for name in servers:
            if name in self._connected_servers:
                client = self._clients.get(name)
                caps = client.capabilities if client else None
                cap_list = []
                if caps:
                    if caps.definition:
                        cap_list.append("definition")
                    if caps.references:
                        cap_list.append("references")
                    if caps.hover:
                        cap_list.append("hover")
                    if caps.completion:
                        cap_list.append("completion")
                    if caps.rename:
                        cap_list.append("rename")
                lines.append(f"  {name}: CONNECTED")
                if cap_list:
                    lines.append(f"    Capabilities: {', '.join(cap_list)}")
            elif name in self._failed_servers:
                lines.append(f"  {name}: FAILED")
                lines.append(f"    Error: {self._failed_servers[name]}")
            else:
                lines.append(f"  {name}: DISCONNECTED")
        return '\n'.join(lines)

    def _cmd_connect(self, server_name: str) -> str:
        if not server_name:
            return "Usage: lsp connect <server_name>"

        if not self._initialized:
            self.initialize()

        self._load_config_cache()
        servers = self._config_cache.get('languageServers', {})

        if server_name not in servers:
            return f"Server '{server_name}' not found. Use 'lsp list' to see configured servers."

        if server_name in self._connected_servers:
            return f"Server '{server_name}' is already connected."

        try:
            spec = servers[server_name]
            self._request_queue.put((MSG_CONNECT_SERVER, {
                'name': server_name,
                'spec': spec,
            }))

            status, result = self._response_queue.get(timeout=30)
            if status == 'error':
                self._failed_servers[server_name] = result
                return f"Failed to connect to '{server_name}': {result}"

            self._connected_servers.add(server_name)
            self._failed_servers.pop(server_name, None)
            return f"Connected to '{server_name}'"
        except queue.Empty:
            return f"Connection to '{server_name}' timed out"
        except Exception as e:
            return f"Error connecting to '{server_name}': {e}"

    def _cmd_disconnect(self, server_name: str) -> str:
        if not server_name:
            return "Usage: lsp disconnect <server_name>"

        if server_name not in self._connected_servers:
            return f"Server '{server_name}' is not connected."

        try:
            self._request_queue.put((MSG_DISCONNECT_SERVER, {'name': server_name}))
            status, result = self._response_queue.get(timeout=10)

            self._connected_servers.discard(server_name)
            return f"Disconnected from '{server_name}'"
        except Exception as e:
            return f"Error disconnecting: {e}"

    def _cmd_reload(self) -> str:
        if not self._initialized:
            self.initialize()

        # Force reload config from disk
        old_servers = set(self._config_cache.get('languageServers', {}).keys()) if self._config_cache else set()
        self._load_config_cache(force=True)
        servers = self._config_cache.get('languageServers', {})
        new_servers = set(servers.keys())

        lines = []
        if old_servers != new_servers:
            added = new_servers - old_servers
            removed = old_servers - new_servers
            if added:
                lines.append(f"Added servers: {', '.join(added)}")
            if removed:
                lines.append(f"Removed servers: {', '.join(removed)}")
        else:
            lines.append(f"Config unchanged ({len(servers)} server(s))")

        try:
            self._request_queue.put((MSG_RELOAD_CONFIG, {'servers': servers}))
            status, result = self._response_queue.get(timeout=60)

            if status == 'ok':
                connected = result.get('connected', [])
                failed = result.get('failed', {})
                self._connected_servers = set(connected)
                self._failed_servers = failed

                if connected:
                    lines.append(f"Connected: {', '.join(connected)}")
                if failed:
                    for name, error in failed.items():
                        lines.append(f"Failed: {name} - {error}")
                if not connected and not failed:
                    lines.append("No servers to connect")

                return '\n'.join(lines)
            return f"Reload failed: {result}"
        except queue.Empty:
            return "Reload timed out - async loop may not be running"
        except Exception as e:
            return f"Error reloading: {e}"

    def _cmd_logs(self, args: str) -> str:
        if args.lower() == 'clear':
            with self._log_lock:
                self._log.clear()
            return "Logs cleared."

        with self._log_lock:
            if not self._log:
                return "No log entries."
            entries = list(self._log)

        if args:
            entries = [e for e in entries if e.server and e.server.lower() == args.lower()]

        lines = [e.format() for e in entries[-50:]]
        return '\n'.join(lines) if lines else "No matching log entries."

    def _load_config_cache(self, force: bool = False) -> None:
        """Load LSP configuration from file.

        Search order:
        1. Custom path from plugin_configs (config_path)
        2. .lsp.json in workspace directory (client's working directory)
        3. .lsp.json in current working directory (fallback)
        4. ~/.lsp.json in home directory
        """
        if self._config_cache and not force:
            return

        # Build search paths - custom path takes priority
        paths = []
        if self._custom_config_path:
            paths.append(self._custom_config_path)
        # Use workspace_path if set, otherwise fall back to cwd
        workspace = self._workspace_path or os.getcwd()
        paths.extend([
            os.path.join(workspace, '.lsp.json'),
            os.path.expanduser('~/.lsp.json'),
        ])

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._config_cache = json.load(f)
                    self._config_path = path
                    self._log_event(LOG_INFO, f"Loaded config from {path}")
                    return
                except Exception as e:
                    self._log_event(LOG_WARN, f"Failed to load {path}: {e}")
                    continue
        self._config_cache = {}

    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def _thread_main(self) -> None:
        """Background thread running the LSP event loop."""

        # Create stderr capture that routes to internal log buffer
        self._errlog = LogCapture(self._log_event)

        async def run_lsp():
            self._log_event(LOG_INFO, "LSP plugin initializing")

            self._load_config_cache()
            servers = self._config_cache.get('languageServers', {})

            if servers:
                self._log_event(LOG_INFO, f"Found {len(servers)} server(s) in configuration")
            else:
                self._log_event(LOG_WARN, "No LSP servers configured")

            async def connect_server(name: str, spec: dict) -> bool:
                """Connect to a language server."""
                self._log_event(LOG_INFO, "Connecting to server", server=name)
                try:
                    # Expand variables in args (e.g., ${workspaceRoot})
                    raw_args = spec.get('args', [])
                    expanded_args = expand_variables(raw_args)
                    config = ServerConfig(
                        name=name,
                        command=spec.get('command', ''),
                        args=expanded_args,
                        env=spec.get('env'),
                        root_uri=spec.get('rootUri'),
                        language_id=spec.get('languageId'),
                    )
                    client = LSPClient(config, errlog=self._errlog)
                    await asyncio.wait_for(client.start(), timeout=15.0)
                    self._clients[name] = client
                    self._connected_servers.add(name)
                    self._failed_servers.pop(name, None)
                    self._log_event(LOG_INFO, "Connected successfully", server=name)
                    return True
                except asyncio.TimeoutError:
                    self._failed_servers[name] = "Connection timed out"
                    self._log_event(LOG_ERROR, "Connection timed out", server=name)
                    return False
                except Exception as e:
                    self._failed_servers[name] = str(e)
                    self._log_event(LOG_ERROR, "Connection failed", server=name, details=str(e))
                    return False

            async def disconnect_server(name: str) -> None:
                """Disconnect from a language server."""
                if name in self._clients:
                    try:
                        await self._clients[name].stop()
                    except Exception:
                        pass
                    del self._clients[name]
                self._connected_servers.discard(name)

            # Auto-connect to configured servers
            for name, spec in servers.items():
                if spec.get('autoConnect', True):
                    await connect_server(name, spec)

            self._log_event(LOG_INFO, f"Initialization complete: {len(self._connected_servers)} connected")

            # Process requests from main thread
            while True:
                try:
                    req = self._request_queue.get(timeout=0.1)
                    if req is None or req == (None, None):
                        break

                    msg_type, data = req

                    if msg_type == MSG_CALL_METHOD:
                        method = data.get('method')
                        args = data.get('args', {})
                        server = data.get('server')

                        if server and server in self._clients:
                            client = self._clients[server]
                        else:
                            # Find appropriate server based on file extension
                            client = self._find_client_for_file(args.get('file_path', ''))

                        if not client:
                            # Build informative error message
                            error_msg = self._build_no_server_error(args.get('file_path', ''))
                            self._response_queue.put(('error', error_msg))
                            continue

                        try:
                            result = await self._call_lsp_method(client, method, args)
                            self._response_queue.put(('ok', result))
                        except Exception as e:
                            self._log_event(LOG_ERROR, f"LSP call failed: {method}", details=str(e))
                            self._response_queue.put(('error', str(e)))

                    elif msg_type == MSG_CONNECT_SERVER:
                        name = data.get('name')
                        spec = data.get('spec', {})
                        success = await connect_server(name, spec)
                        if success:
                            self._response_queue.put(('ok', {}))
                        else:
                            self._response_queue.put(('error', self._failed_servers.get(name, 'Unknown error')))

                    elif msg_type == MSG_DISCONNECT_SERVER:
                        name = data.get('name')
                        await disconnect_server(name)
                        self._response_queue.put(('ok', {}))

                    elif msg_type == MSG_RELOAD_CONFIG:
                        new_servers = data.get('servers', {})

                        # Disconnect all
                        for name in list(self._clients.keys()):
                            await disconnect_server(name)

                        # Connect to new servers
                        connected = []
                        failed = {}
                        for name, spec in new_servers.items():
                            if await connect_server(name, spec):
                                connected.append(name)
                            else:
                                failed[name] = self._failed_servers.get(name, 'Unknown error')

                        self._response_queue.put(('ok', {
                            'connected': connected,
                            'failed': failed,
                        }))

                except queue.Empty:
                    await asyncio.sleep(0.01)

            # Cleanup
            for name in list(self._clients.keys()):
                await disconnect_server(name)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(run_lsp())
        except Exception as e:
            self._log_event(LOG_ERROR, "LSP thread crashed", details=str(e))
        finally:
            self._loop.close()

    def _find_client_for_file(self, file_path: str) -> Optional[LSPClient]:
        """Find an appropriate LSP client for a file."""
        if not file_path or not self._clients:
            return list(self._clients.values())[0] if self._clients else None

        ext = os.path.splitext(file_path)[1].lower()
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
        }
        lang = ext_to_lang.get(ext)

        # Try to find a server matching the language
        for name, client in self._clients.items():
            if client.config.language_id == lang:
                return client
            if lang and lang in name.lower():
                return client

        # Return first available
        return list(self._clients.values())[0] if self._clients else None

    def _build_no_server_error(self, file_path: str) -> str:
        """Build an informative error message when no LSP server is available.

        This provides helpful context about:
        - Whether any servers are configured
        - Which servers failed to start and why
        - How to resolve the issue
        """
        parts = ["No LSP server available"]

        # Check if any servers are configured
        self._load_config_cache()
        servers = self._config_cache.get('languageServers', {})

        if not servers:
            parts.append("No LSP servers configured in .lsp.json")
            parts.append("Create .lsp.json with server configuration to enable LSP features")
            return ". ".join(parts)

        # Check for failed servers
        if self._failed_servers:
            parts.append(f"{len(self._failed_servers)} server(s) failed to start:")
            for name, error in self._failed_servers.items():
                # Simplify common errors
                if "FileNotFoundError" in error or "No such file" in error:
                    cmd = servers.get(name, {}).get('command', 'unknown')
                    parts.append(f"  - {name}: command '{cmd}' not found (not installed?)")
                elif "timed out" in error.lower():
                    parts.append(f"  - {name}: connection timed out")
                else:
                    parts.append(f"  - {name}: {error}")

        # Suggest file-specific server if applicable
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            lang = EXT_TO_LANGUAGE.get(ext)
            if lang:
                # Check if there's a configured server for this language
                matching_servers = [
                    name for name, spec in servers.items()
                    if spec.get('languageId') == lang or lang in name.lower()
                ]
                if matching_servers:
                    failed_matching = [s for s in matching_servers if s in self._failed_servers]
                    if failed_matching:
                        parts.append(f"Server for {lang} files ({', '.join(failed_matching)}) failed to start")
                        parts.append("Install the language server or check 'lsp status' for details")

        return ". ".join(parts)

    def _build_empty_result_error(self, file_path: str, operation: str, detail: str = "") -> str:
        """Build a specific error message when an LSP operation returns no results.

        Detects the actual cause:
        - No server configured for this file type
        - Server configured but failed to start (with reason)
        - Server connected but returned empty results
        """
        ext = os.path.splitext(file_path)[1].lower() if file_path else ''
        lang = EXT_TO_LANGUAGE.get(ext)

        # Load config to check server configuration
        self._load_config_cache()
        servers = self._config_cache.get('languageServers', {})

        # Find servers that might handle this file type
        matching_servers = []
        if lang:
            for name, spec in servers.items():
                if spec.get('languageId') == lang or lang in name.lower():
                    matching_servers.append(name)

        # Case 1: No servers configured at all
        if not servers:
            return f"{operation}{detail}. No LSP servers configured - create .lsp.json to enable LSP features."

        # Case 2: No server configured for this language
        if lang and not matching_servers:
            return f"{operation}{detail}. No LSP server configured for {lang} files in .lsp.json."

        # Case 3: Server configured but failed to start
        failed_matching = [s for s in matching_servers if s in self._failed_servers]
        if failed_matching:
            server_name = failed_matching[0]
            error = self._failed_servers[server_name]
            # Simplify common errors
            if "FileNotFoundError" in error or "No such file" in error:
                cmd = servers.get(server_name, {}).get('command', 'unknown')
                return f"{operation}{detail}. LSP server '{server_name}' failed: command '{cmd}' not found (not installed?)."
            elif "timed out" in error.lower():
                return f"{operation}{detail}. LSP server '{server_name}' failed: connection timed out."
            else:
                return f"{operation}{detail}. LSP server '{server_name}' failed: {error}."

        # Case 4: Server connected but returned empty - genuine empty result
        connected_matching = [s for s in matching_servers if s in self._connected_servers]
        if connected_matching:
            server_name = connected_matching[0]
            return f"{operation}{detail}. Server '{server_name}' is connected but returned no results."

        # Case 5: Server configured but not connected (unknown state)
        if matching_servers:
            return f"{operation}{detail}. LSP server '{matching_servers[0]}' is not connected. Run 'lsp status' for details."

        # Fallback
        return f"{operation}{detail}. Run 'lsp status' to check server state."

    async def _call_lsp_method(self, client: LSPClient, method: str, args: Dict[str, Any]) -> Any:
        """Call an LSP method on the client."""
        file_path = args.get('file_path')

        # Methods that require full parsing need more time
        # get_diagnostics also needs time for server to analyze and publish diagnostics
        needs_parsing = method in ('hover', 'document_symbols', 'goto_definition', 'find_references', 'get_diagnostics')

        # Ensure document is open if needed
        if file_path and method not in ('workspace_symbols',):
            await client.open_document(file_path)
            # Wait for server to process the document
            # Longer delay for operations that need full parsing/diagnostics
            await asyncio.sleep(0.8 if needs_parsing else 0.2)

        if method == 'goto_definition':
            locations = await client.goto_definition(
                file_path, args['line'], args['character']
            )
            if not locations:
                pos = f" at {file_path}:{args['line']+1}:{args['character']}"
                return {"error": self._build_empty_result_error(file_path, "No definition found", pos)}
            return self._format_locations(locations)

        elif method == 'find_references':
            locations = await client.find_references(
                file_path, args['line'], args['character'],
                args.get('include_declaration', True)
            )
            if not locations:
                pos = f" at {file_path}:{args['line']+1}:{args['character']}"
                return {"error": self._build_empty_result_error(file_path, "No references found", pos)}
            return self._format_locations(locations)

        elif method == 'hover':
            # Retry hover a few times - server might still be indexing
            for attempt in range(3):
                hover = await client.hover(file_path, args['line'], args['character'])
                if hover and hover.contents:
                    return {"contents": hover.contents}
                if attempt < 2:
                    await asyncio.sleep(0.3)  # Brief wait before retry
            pos = f" at {file_path}:{args['line']+1}:{args['character']}"
            return {"error": self._build_empty_result_error(file_path, "No hover information", pos)}

        elif method == 'get_diagnostics':
            diagnostics = client.get_diagnostics(file_path)
            return self._format_diagnostics(diagnostics)

        elif method == '_ensure_workspace_indexed':
            # Internal method to index all files of a type in a directory
            directory = args.get('directory', '')
            extension = args.get('extension')
            if directory:
                # Pass extension as a list if provided
                extensions = [extension] if extension else None
                self._trace(f"_ensure_workspace_indexed: configuring extra_paths=[{directory}] for {client.config.language_id}")
                await client.ensure_workspace_indexed(directory, extensions)
                # Give the LSP server time to process config and re-analyze files
                # pylsp/Jedi needs time to analyze cross-file imports after:
                # 1. Configuration update (extra_paths)
                # 2. Documents being closed and reopened
                await asyncio.sleep(2.0)
            return {"success": True}

        elif method == 'document_symbols':
            symbols = await client.get_document_symbols(file_path)
            if not symbols:
                return {"error": self._build_empty_result_error(file_path, "No symbols found", f" in {file_path}")}
            result = []
            for s in symbols:
                self._trace(f"document_symbols: {s.name} location.range.start = line {s.location.range.start.line}, char {s.location.range.start.character}")
                result.append({
                    "name": s.name,
                    "kind": s.kind_name,
                    "location": f"{self._uri_to_path(s.location.uri)}:{s.location.range.start.line + 1}:{s.location.range.start.character}"
                })
            return result

        elif method == 'workspace_symbols':
            query = args['query']
            symbols = await client.workspace_symbols(query)
            if not symbols:
                # For workspace symbols, use the working directory to determine language context
                return {"error": self._build_empty_result_error("", f"No symbols matching '{query}' found")}
            return [
                {
                    "name": s.name,
                    "kind": s.kind_name,
                    "location": f"{self._uri_to_path(s.location.uri)}:{s.location.range.start.line + 1}"
                }
                for s in symbols
            ]

        elif method == 'rename_symbol':
            workspace_edit = await client.rename(
                file_path, args['line'], args['character'], args['new_name']
            )
            return workspace_edit

        elif method == 'get_code_actions':
            actions = await client.get_code_actions(
                file_path,
                args['start_line'],
                args['start_char'],
                args['end_line'],
                args['end_char'],
                only_kinds=args.get('only_kinds')
            )
            return actions

        elif method == 'resolve_code_action':
            # Resolve a code action to get its edit
            action = args.get('action')
            if action:
                resolved = await client.resolve_code_action(action)
                return resolved
            return None

        elif method == 'execute_command':
            # Execute a workspace command
            command = args.get('command')
            arguments = args.get('arguments')
            result = await client.execute_command(command, arguments)
            return result

        else:
            raise ValueError(f"Unknown method: {method}")

    def _format_locations(self, locations: List[Location]) -> List[Dict[str, Any]]:
        """Format locations for output."""
        return [
            {
                "file": self._uri_to_path(loc.uri),
                "line": loc.range.start.line + 1,
                "character": loc.range.start.character
            }
            for loc in locations
        ]

    def _format_diagnostics(self, diagnostics: List[Diagnostic]) -> List[Dict[str, Any]]:
        """Format diagnostics for output."""
        return [
            {
                "severity": d.severity_name,
                "message": d.message,
                "line": d.range.start.line + 1,
                "character": d.range.start.character,
                "source": d.source,
                "code": d.code,
            }
            for d in diagnostics
        ]

    def _uri_to_path(self, uri: str) -> str:
        """Convert a file URI to a path."""
        if uri.startswith('file://'):
            path = uri[7:]
            if os.name == 'nt' and path.startswith('/'):
                path = path[1:]
            return path
        return uri

    def _find_symbol_position(
        self, symbol: str, file_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the position of a symbol in the codebase.

        This enables symbol-based tool calls instead of requiring exact positions.
        The model can say "find references to UserService" instead of providing
        line/character coordinates.

        Args:
            symbol: Name of the symbol to find (class, method, variable, etc.)
            file_path: Optional file to search in. If not provided, searches workspace.

        Returns:
            Dict with 'file_path', 'line', 'character' if found, or None.
        """
        import re

        # If file_path is provided, search in that file
        if file_path and os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()

                # Search for symbol as a word boundary match
                pattern = re.compile(r'\b' + re.escape(symbol) + r'\b')

                for line_num, line in enumerate(lines):
                    match = pattern.search(line)
                    if match:
                        return {
                            'file_path': file_path,
                            'line': line_num,
                            'character': match.start()
                        }
            except (IOError, OSError) as e:
                self._trace(f"_find_symbol_position: error reading {file_path}: {e}")

        # Fall back to workspace symbols search
        result = self._execute_method('workspace_symbols', {'query': symbol})
        if isinstance(result, list) and len(result) > 0:
            # Find exact match first, then prefix match
            for sym in result:
                if sym.get('name') == symbol:
                    loc = sym.get('location', {})
                    return {
                        'file_path': loc.get('file_path') or self._uri_to_path(loc.get('uri', '')),
                        'line': loc.get('line', 0),
                        'character': loc.get('character', 0)
                    }

            # If no exact match, use first result
            sym = result[0]
            loc = sym.get('location', {})
            return {
                'file_path': loc.get('file_path') or self._uri_to_path(loc.get('uri', '')),
                'line': loc.get('line', 0),
                'character': loc.get('character', 0)
            }

        return None

    # Tool executor methods

    def _execute_method(self, method: str, args: Dict[str, Any]) -> Any:
        """Execute an LSP method synchronously."""
        self._trace(f"execute: {method} args={args}")
        if not self._initialized:
            self.initialize()

        if not self._connected_servers:
            self._trace(f"execute: {method} FAILED - no servers connected")
            error_msg = self._build_no_server_error(args.get('file_path', ''))
            return {"error": error_msg}

        try:
            self._request_queue.put((MSG_CALL_METHOD, {'method': method, 'args': args}))
            status, result = self._response_queue.get(timeout=30)

            if status == 'error':
                self._trace(f"execute: {method} ERROR - {result}")
                return {"error": result}
            # Check if result indicates an error (e.g., no definitions found)
            if isinstance(result, dict) and 'error' in result:
                self._trace(f"execute: {method} EMPTY - {result.get('error', 'no results')}")
            else:
                self._trace(f"execute: {method} OK")
            return result
        except queue.Empty:
            self._trace(f"execute: {method} TIMEOUT")
            return {"error": "LSP request timed out"}
        except Exception as e:
            self._trace(f"execute: {method} EXCEPTION - {e}")
            return {"error": str(e)}

    def _exec_goto_definition(self, args: Dict[str, Any]) -> Any:
        """Find definition of a symbol."""
        symbol = args.get('symbol')
        file_path = args.get('file_path')

        if not symbol:
            return {"error": "symbol parameter is required"}

        pos = self._find_symbol_position(symbol, file_path)
        if not pos:
            return {"error": f"Symbol '{symbol}' not found in codebase"}

        return self._execute_method('goto_definition', {
            'file_path': pos['file_path'],
            'line': pos['line'],
            'character': pos['character']
        })

    def _exec_find_references(self, args: Dict[str, Any]) -> Any:
        """Find all references to a symbol."""
        symbol = args.get('symbol')
        file_path = args.get('file_path')
        include_declaration = args.get('include_declaration', True)

        if not symbol:
            return {"error": "symbol parameter is required"}

        pos = self._find_symbol_position(symbol, file_path)
        if not pos:
            return {"error": f"Symbol '{symbol}' not found in codebase"}

        return self._execute_method('find_references', {
            'file_path': pos['file_path'],
            'line': pos['line'],
            'character': pos['character'],
            'include_declaration': include_declaration
        })

    def _exec_hover(self, args: Dict[str, Any]) -> Any:
        """Get hover information for a symbol."""
        symbol = args.get('symbol')
        file_path = args.get('file_path')

        if not symbol:
            return {"error": "symbol parameter is required"}

        pos = self._find_symbol_position(symbol, file_path)
        if not pos:
            return {"error": f"Symbol '{symbol}' not found in codebase"}

        return self._execute_method('hover', {
            'file_path': pos['file_path'],
            'line': pos['line'],
            'character': pos['character']
        })

    def _exec_get_diagnostics(self, args: Dict[str, Any]) -> Any:
        """Get diagnostics for a file (unchanged - already file-based)."""
        return self._execute_method('get_diagnostics', args)

    def _exec_document_symbols(self, args: Dict[str, Any]) -> Any:
        """Get symbols in a file (unchanged - already file-based)."""
        return self._execute_method('document_symbols', args)

    def _exec_workspace_symbols(self, args: Dict[str, Any]) -> Any:
        """Search for symbols in workspace (unchanged - already query-based)."""
        return self._execute_method('workspace_symbols', args)

    def _exec_rename_symbol(self, args: Dict[str, Any]) -> Any:
        """Rename a symbol across all files.

        If apply=True, applies the changes to files. Otherwise returns a preview.
        """
        symbol = args.get('symbol')
        new_name = args.get('new_name')
        file_path = args.get('file_path')
        apply = args.get('apply', False)

        if not symbol:
            return {"error": "symbol parameter is required"}
        if not new_name:
            return {"error": "new_name parameter is required"}

        pos = self._find_symbol_position(symbol, file_path)
        if not pos:
            return {"error": f"Symbol '{symbol}' not found in codebase"}

        # Get the workspace edit from LSP
        result = self._execute_method('rename_symbol', {
            'file_path': pos['file_path'],
            'line': pos['line'],
            'character': pos['character'],
            'new_name': new_name
        })

        # Check for errors
        if isinstance(result, dict) and 'error' in result:
            return result

        # Handle case where rename returns None or empty
        if result is None:
            return {"error": f"LSP server could not rename symbol '{symbol}'"}

        # result should be a WorkspaceEdit object
        if isinstance(result, WorkspaceEdit):
            workspace_edit = result
        elif isinstance(result, dict):
            # Fallback if somehow we got a raw dict
            workspace_edit = WorkspaceEdit.from_dict(result)
        else:
            return {"error": f"Unexpected response from LSP server: {type(result)}"}

        # Prepare the result info
        affected_files = workspace_edit.get_affected_files()
        file_info = []
        for uri in affected_files:
            path = self._uri_to_path(uri)
            edits = workspace_edit.changes.get(uri, [])
            file_info.append({
                "file": path,
                "edits": len(edits)
            })

        if not apply:
            # Preview mode - return what would be changed
            return {
                "mode": "preview",
                "symbol": symbol,
                "new_name": new_name,
                "files_affected": len(affected_files),
                "changes": file_info,
                "message": f"Would rename '{symbol}' to '{new_name}' in {len(affected_files)} file(s). Set apply=true to apply."
            }
        else:
            # Apply the changes
            apply_result = apply_workspace_edit(workspace_edit, dry_run=False)

            return {
                "mode": "applied",
                "symbol": symbol,
                "new_name": new_name,
                "success": apply_result["success"],
                "files_modified": apply_result["files_modified"],
                "changes": apply_result["changes"],
                "errors": apply_result["errors"] if apply_result["errors"] else None
            }

    def _exec_get_code_actions(self, args: Dict[str, Any]) -> Any:
        """Get available code actions for a code region."""
        file_path = args.get('file_path')
        start_line = args.get('start_line')
        start_column = args.get('start_column')
        end_line = args.get('end_line')
        end_column = args.get('end_column')
        only_refactorings = args.get('only_refactorings', False)

        if not file_path:
            return {"error": "file_path parameter is required"}
        if start_line is None or start_column is None:
            return {"error": "start_line and start_column are required"}
        if end_line is None or end_column is None:
            return {"error": "end_line and end_column are required"}

        # Convert 1-indexed to 0-indexed
        start_line_0 = start_line - 1
        start_char_0 = start_column - 1
        end_line_0 = end_line - 1
        end_char_0 = end_column - 1

        # Build filter for code action kinds
        only_kinds = None
        if only_refactorings:
            only_kinds = ["refactor", "refactor.extract", "refactor.inline", "refactor.rewrite"]

        result = self._execute_method('get_code_actions', {
            'file_path': file_path,
            'start_line': start_line_0,
            'start_char': start_char_0,
            'end_line': end_line_0,
            'end_char': end_char_0,
            'only_kinds': only_kinds
        })

        if isinstance(result, dict) and 'error' in result:
            return result

        if not result:
            return {
                "actions": [],
                "message": "No code actions available for this selection"
            }

        # Format actions for output
        actions_list = []
        if isinstance(result, list):
            for action in result:
                if isinstance(action, CodeAction):
                    actions_list.append(action.to_summary())
                elif isinstance(action, dict):
                    # Fallback for raw dict
                    actions_list.append({
                        "title": action.get("title", "Unknown"),
                        "kind": action.get("kind", "unknown")
                    })

        return {
            "actions": actions_list,
            "count": len(actions_list)
        }

    def _exec_apply_code_action(self, args: Dict[str, Any]) -> Any:
        """Apply a code action by its title."""
        file_path = args.get('file_path')
        start_line = args.get('start_line')
        start_column = args.get('start_column')
        end_line = args.get('end_line')
        end_column = args.get('end_column')
        action_title = args.get('action_title')

        if not file_path:
            return {"error": "file_path parameter is required"}
        if start_line is None or start_column is None:
            return {"error": "start_line and start_column are required"}
        if end_line is None or end_column is None:
            return {"error": "end_line and end_column are required"}
        if not action_title:
            return {"error": "action_title parameter is required"}

        # Convert 1-indexed to 0-indexed
        start_line_0 = start_line - 1
        start_char_0 = start_column - 1
        end_line_0 = end_line - 1
        end_char_0 = end_column - 1

        # First, get all available code actions
        actions_result = self._execute_method('get_code_actions', {
            'file_path': file_path,
            'start_line': start_line_0,
            'start_char': start_char_0,
            'end_line': end_line_0,
            'end_char': end_char_0
        })

        if isinstance(actions_result, dict) and 'error' in actions_result:
            return actions_result

        if not actions_result:
            return {"error": "No code actions available for this selection"}

        # Find the action with matching title
        matching_action = None
        for action in actions_result:
            if isinstance(action, CodeAction):
                if action.title == action_title:
                    matching_action = action
                    break
            elif isinstance(action, dict) and action.get('title') == action_title:
                matching_action = CodeAction.from_dict(action)
                break

        if not matching_action:
            available = [a.title if isinstance(a, CodeAction) else a.get('title', '?') for a in actions_result[:5]]
            return {
                "error": f"Code action '{action_title}' not found",
                "available_actions": available
            }

        # Check if action is disabled
        if matching_action.disabled:
            return {"error": f"Code action is disabled: {matching_action.disabled}"}

        # If action doesn't have an edit, try to resolve it
        if matching_action.edit is None and matching_action.data is not None:
            resolved = self._execute_method('resolve_code_action', {'action': matching_action})
            if isinstance(resolved, CodeAction):
                matching_action = resolved

        # Apply the workspace edit if present
        if matching_action.edit:
            apply_result = apply_workspace_edit(matching_action.edit, dry_run=False)
            result = {
                "action": action_title,
                "success": apply_result["success"],
                "files_modified": apply_result["files_modified"],
                "changes": apply_result["changes"]
            }
            if apply_result["errors"]:
                result["errors"] = apply_result["errors"]
            return result

        # Execute command if present (some actions only have commands)
        if matching_action.command:
            cmd = matching_action.command
            cmd_result = self._execute_method('execute_command', {
                'command': cmd.get('command'),
                'arguments': cmd.get('arguments')
            })

            return {
                "action": action_title,
                "command_executed": cmd.get('command'),
                "result": cmd_result
            }

        return {"error": f"Code action '{action_title}' has no edit or command to apply"}


def create_plugin() -> LSPToolPlugin:
    """Factory function for plugin discovery."""
    return LSPToolPlugin()
