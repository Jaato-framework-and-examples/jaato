"""LSP tool plugin for code intelligence via Language Server Protocol."""

import asyncio
import atexit
import json
import logging
import os
import queue
import shutil
import signal
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.plugins.base import (
    UserCommand, CommandParameter, CommandCompletion,
    ToolResultEnrichmentResult, HelpLines, PluginSetting
)
from jaato_sdk.plugins.model_provider.types import (
    ToolSchema,
    TRAIT_FILE_WRITER,
    DISCOVERABILITY_DEFERRED,
)
from ..subagent.config import expand_variables
from .lsp_client import (
    LSPClient, ServerConfig, Location, Diagnostic, Hover,
    TextEdit, WorkspaceEdit, CodeAction, Range, Position
)

from shared.plugins.runner_forwarding import RunnerForwardingMixin
from shared.trace import trace as _trace_write

# Module logger — lands in the process's standard log (e.g. the daemon's
# /tmp/jaato.log), unlike self._log_event/_trace which route to the LSP
# debug log under the workspace (daemon-side writes there don't surface).
# Used for the #284 daemon-process gate logging (thread suppression + the
# connect_server defense guard).
logger = logging.getLogger(__name__)


def _running_in_daemon_process() -> bool:
    """True when this process is the daemon (``python -m server``), False in a
    runner/slot (``python -m server.runner``).

    The daemon must NEVER host a language server.  In the seat-flip
    architecture the per-session runner hosts LSP in a reapable slot; a jdtls
    spawned in the long-lived daemon has no owning slot, is never reaped, and
    accumulates resident until the daemon OOMs (#284).  The daemon-side LSP
    instance exists only as a :class:`RunnerForwardingMixin` executor stub
    (forwards tool calls via RPC) — it never needs the LSP background thread.

    **Why process identity (not the registry):** the #285 diagnostic proved the
    daemon-side LSP instance has ``_plugin_registry is None`` at connect time
    (it is never wired), so the earlier ``registry.runner_rpc`` gate was
    structurally unreachable and never fired.  Process identity is the only
    signal available at LSP init/connect time.  Detected via the ``__main__``
    module package (``server`` = daemon, ``server.runner`` = runner), with an
    ``argv[0]`` script-path fallback.  Unknown contexts (tests, odd launchers)
    return ``False`` — i.e. "not the daemon, host LSP" — so suppression only
    ever triggers when we are positively sure this is the daemon.
    """
    main_mod = sys.modules.get("__main__")
    pkg = getattr(main_mod, "__package__", "") or ""
    if pkg == "server.runner" or pkg.startswith("server.runner"):
        return False
    if pkg == "server":
        return True
    argv0 = (sys.argv[0] if sys.argv else "").replace("\\", "/")
    if argv0.endswith("server/runner/__main__.py"):
        return False
    return argv0.endswith("server/__main__.py")


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
# PR-157 (server 0.6.140): retry connect for any server not yet
# in self._connected_servers.  Dispatched by `set_workspace_path()`
# when it lands AFTER the initial auto-connect loop has already
# run with workspace_path=None.  Fire-and-forget — no response
# pushed to response_queue (the sender doesn't wait).
MSG_RETRY_AUTOCONNECT = 'retry_autoconnect'

# Log levels
LOG_INFO = 'INFO'
LOG_DEBUG = 'DEBUG'
LOG_ERROR = 'ERROR'
LOG_WARN = 'WARN'

MAX_LOG_ENTRIES = 500

# Default per-server LSP connect timeout (seconds).  Lightweight servers
# (pyright / typescript-language-server) initialize in under 5s, but
# Eclipse JDT LS (jdtls) on a cold Maven / Gradle workspace routinely
# takes 60-120s+ for `initialize` + dependency download + workspace
# import.  A too-short default makes `connect_server` time out WHILE
# jdtls is still legitimately starting; that timeout cancels the connect
# coroutine but leaves the spawned subprocess alive (now reaped via
# `_reap_failed_client`, #284) and the retry-autoconnect then spawns a
# fresh one — so every premature timeout cost one jdtls cold-start of
# wasted CPU/RAM.  180s comfortably covers a cold jdtls import while
# still bounding a genuinely hung server.  Operators can raise further
# via `plugin_configs.lsp.connect_timeout_seconds` (capped at
# MAX_CONNECT_TIMEOUT_SECONDS).
DEFAULT_CONNECT_TIMEOUT_SECONDS = 180.0
MIN_CONNECT_TIMEOUT_SECONDS = 1.0
MAX_CONNECT_TIMEOUT_SECONDS = 300.0

# Bounded-poll knobs for `await_diagnostics` (replaces the pre-0.6.134
# hard-coded 0.8s sleep at `_call_lsp_method`).  The framework now
# waits on a per-URI asyncio.Event signalled by the JSON-RPC reader
# when the server pushes `textDocument/publishDiagnostics`.  The
# defaults are sized for the empirically-measured worst case (jdtls
# on a cold Maven workspace ~3-8s first batch); operators can shrink
# them per-profile for lightweight servers.
DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS = 5.0
DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS = 0.5
# Convergence-loop window — seconds the bounded poll keeps listening
# for follow-up publishDiagnostics after the first one lands, so
# multi-stage jdtls analysis (parser → compiler → linter → import
# resolver) has a chance to overwrite the cache with the SETTLED
# state before the caller reads it.  ``0.0`` disables the loop and
# returns on first publish (legacy semantics).  Default ``3.0`` is
# evidence-grounded in the 2026-06-05 instrumented cascade analysis
# (91 adjacent-publish races across 30 distinct ``.java`` URIs;
# p50 = 1.46 s, p90 = 17.9 s — p90 dominated by edit-cycle re-races
# which a window can't fix; 3.0 s captures the fresh-render cluster).
# See ``/tmp/converge2.py`` (preserved in the PR-224 description).
DEFAULT_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS = 3.0
MIN_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS = 0.0
MAX_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS = 30.0
MIN_DIAGNOSTICS_MAX_WAIT_SECONDS = 0.0  # 0 = legacy "no wait, read cache now"
MAX_DIAGNOSTICS_MAX_WAIT_SECONDS = 60.0

# Default path for the lsp plugin's cross-session diagnostic log.
# Pre-0.6.136 this was hard-coded to ``tempfile.gettempdir()/lsp_debug.log``
# (e.g. ``/tmp/lsp_debug.log``) which broke silently on apparmor-confined
# runners (PR-148 made the write fail; the outer try/except in
# ``_load_config_cache`` misclassified the failure as a config-load error
# and reset ``_config_cache = {}`` → "No LSP servers configured" → 100%
# enrichment dead).  The new default resolves under the workspace root
# (where apparmor already covers writes via the per-session profile
# composed by ``get_apparmor_rules``), and operators can override the
# path via ``plugin_configs.lsp.debug_log_path``.
#
# Path semantics: relative values are resolved against ``workspace_path``
# at write time + at apparmor-fragment-composition time (same resolver
# either side, so the granted path always matches what the plugin
# actually writes to).  Absolute values pass through unchanged.
DEFAULT_DEBUG_LOG_PATH = ".jaato/logs/lsp_debug.log"


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


class LSPToolPlugin(RunnerForwardingMixin):
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
        # atexit backstop: reap jdtls when THIS (slot/runner) process exits.
        # Registered lazily in _ensure_thread.  See _atexit_reap_jdtls / #284.
        self._atexit_registered: bool = False
        self._log: deque = deque(maxlen=MAX_LOG_ENTRIES)
        self._log_lock = threading.Lock()
        # Agent context for trace logging
        self._agent_name: Optional[str] = None
        # Session ID for multi-session log disambiguation
        self._session_id: Optional[str] = None
        # Stderr capture for LSP server output
        self._errlog: Optional[LogCapture] = None
        # Per-server connect timeout (seconds). Configurable via
        # plugin_configs.lsp.connect_timeout_seconds; jdtls on a cold
        # Maven workspace + first-time mvn resolve typically needs 30-60s,
        # while pyright / typescript-language-server start in under 5s.
        self._connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
        # Bounded-poll knobs for awaiting `publishDiagnostics` after
        # didOpen / didChange (consumed by `_call_lsp_method` via
        # `LSPClient.await_diagnostics`).  Configurable via
        # plugin_configs.lsp.diagnostics_{max,min}_wait_seconds.
        self._diagnostics_max_wait_seconds: float = DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS
        self._diagnostics_min_wait_seconds: float = DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS
        self._diagnostics_convergence_window_seconds: float = (
            DEFAULT_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS
        )
        # Operator-configurable path for the cross-session diagnostic
        # log.  Relative paths resolve against `workspace_path`;
        # absolute paths pass through.  Default is workspace-relative
        # so the apparmor fragment can grant write access through the
        # standard per-session profile.  See DEFAULT_DEBUG_LOG_PATH +
        # `get_apparmor_rules` below for the symmetric composition.
        self._debug_log_path_raw: str = DEFAULT_DEBUG_LOG_PATH

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
        """Write trace message to log file for debugging."""
        _trace_write("LSP", msg)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the LSP plugin by starting the background thread.

        Args:
            config: Optional configuration dict. Supports:
                - config_path: Path to .lsp.json file (overrides default search)
                - workspace_path: Client's working directory for finding .lsp.json
                - session_id: Session identifier for log disambiguation
                - agent_name: Name for trace logging
        """
        if self._initialized:
            return

        # Expand variables in config values (e.g., ${projectPath}, ${workspaceRoot})
        config = expand_variables(config) if config else {}

        # Extract config values
        self._agent_name = config.get('agent_name')
        self._session_id = config.get('session_id')
        self._custom_config_path = config.get('config_path')
        self._workspace_path = config.get('workspace_path')

        # Resolve per-server connect timeout knob.  Values outside the
        # supported range are clamped (and logged) rather than rejected —
        # a profile typo should not break LSP entirely; the trace makes
        # the clamp visible at startup so it gets caught.
        raw_timeout = config.get(
            'connect_timeout_seconds', DEFAULT_CONNECT_TIMEOUT_SECONDS
        )
        try:
            timeout_value = float(raw_timeout)
        except (TypeError, ValueError):
            self._trace(
                f"initialize: connect_timeout_seconds={raw_timeout!r} is not a "
                f"number — falling back to default "
                f"{DEFAULT_CONNECT_TIMEOUT_SECONDS}s"
            )
            timeout_value = DEFAULT_CONNECT_TIMEOUT_SECONDS
        clamped = max(
            MIN_CONNECT_TIMEOUT_SECONDS,
            min(timeout_value, MAX_CONNECT_TIMEOUT_SECONDS),
        )
        if clamped != timeout_value:
            self._trace(
                f"initialize: connect_timeout_seconds={timeout_value} "
                f"clamped to {clamped} "
                f"(range [{MIN_CONNECT_TIMEOUT_SECONDS}, "
                f"{MAX_CONNECT_TIMEOUT_SECONDS}])"
            )
        self._connect_timeout_seconds = clamped

        # Same parse + clamp pattern for the diagnostics-wait knobs.
        self._diagnostics_max_wait_seconds = self._parse_wait_knob(
            config.get(
                'diagnostics_max_wait_seconds',
                DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS,
            ),
            DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS,
            MIN_DIAGNOSTICS_MAX_WAIT_SECONDS,
            MAX_DIAGNOSTICS_MAX_WAIT_SECONDS,
            'diagnostics_max_wait_seconds',
        )
        # min_wait is clamped to [0, max_wait] so the floor cannot
        # exceed the upper bound the operator just set.
        self._diagnostics_min_wait_seconds = self._parse_wait_knob(
            config.get(
                'diagnostics_min_wait_seconds',
                DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS,
            ),
            DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS,
            0.0,
            self._diagnostics_max_wait_seconds,
            'diagnostics_min_wait_seconds',
        )
        # Convergence window — how long the bounded poll keeps
        # listening for follow-up publishDiagnostics after the first
        # one lands.  See module-level constant for the empirical
        # rationale.  Clamped to [0, MAX_CONVERGENCE].
        self._diagnostics_convergence_window_seconds = self._parse_wait_knob(
            config.get(
                'diagnostics_convergence_window_seconds',
                DEFAULT_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS,
            ),
            DEFAULT_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS,
            MIN_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS,
            MAX_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS,
            'diagnostics_convergence_window_seconds',
        )

        # Diagnostic-log path knob.  Operator-set strings pass through
        # verbatim (resolved against workspace at use site); unset
        # falls back to the workspace-relative default.  Empty string
        # disables the diagnostic log entirely.
        debug_log_raw = config.get('debug_log_path', DEFAULT_DEBUG_LOG_PATH)
        if debug_log_raw is None:
            debug_log_raw = ''
        self._debug_log_path_raw = str(debug_log_raw)

        self._trace("initialize: starting background thread")
        self._ensure_thread()
        self._initialized = True
        self._trace(f"initialize: connected_servers={list(self._connected_servers)}")

    @staticmethod
    def _resolve_debug_log_path(
        raw_path: str,
        workspace_path: Optional[str],
    ) -> Optional[str]:
        """Resolve the diagnostic-log path knob to a concrete path.

        Symmetric helper used at TWO sites:

        - **Write site** (`_load_config_cache`) — picks where the
          plugin actually appends the log line.
        - **Apparmor-fragment site** (`get_apparmor_rules` classmethod)
          — picks what path to grant write access to.

        Both sites MUST agree on the resolved path, otherwise the
        composer would grant access to a path the plugin doesn't use
        (or vice-versa).  Centralising the resolution here avoids
        that drift class.

        Semantics:
            - Empty string → returns None (operator disabled the log).
            - Absolute path → returned verbatim.
            - Relative path + workspace_path set → joined.
            - Relative path + no workspace_path → returns None
              (writing to a workspace-relative path is meaningless
              without a workspace; better to suppress the diagnostic
              than write to the daemon's cwd silently).

        Args:
            raw_path: The operator-configured value (or default).
            workspace_path: Session's workspace root if known.

        Returns:
            Absolute path the plugin will write to, or None to skip
            the diagnostic write entirely.
        """
        if not raw_path:
            return None
        if os.path.isabs(raw_path):
            return raw_path
        if not workspace_path:
            return None
        return os.path.join(workspace_path, raw_path)

    @classmethod
    def get_apparmor_rules(
        cls,
        *,
        workspace_path: str,
        session_id: str,
        config_root: Optional[str],
        plugin_config: Dict[str, Any],
    ) -> List[str]:
        """Contribute the diagnostic-log apparmor fragment.

        The lsp plugin appends per-session config-load events to a
        shared diagnostic log.  Pre-0.6.136 the path was hard-coded
        to ``tempfile.gettempdir()/lsp_debug.log`` (e.g.
        ``/tmp/lsp_debug.log``), which apparmor-confined runners
        couldn't write — and the outer try/except in
        ``_load_config_cache`` misclassified the failure as a
        config-load error.  Symptom (v141-v144): every LSP enrichment
        call returned "no servers connected" because ``_config_cache``
        was reset to ``{}`` by the misclassified handler.

        This fragment grants ``rw`` on whichever path
        ``plugin_configs.lsp.debug_log_path`` resolves to (default
        ``<workspace>/.jaato/logs/lsp_debug.log``).  Two rules cover
        the full mkdir chain following the file_edit pattern (PR-147):

        - ``<parent_dir>/   rw,`` — the dir entry itself (for the
          mkdir of ``.jaato/logs/`` if it doesn't exist yet).
        - ``<parent_dir>/** rw,`` — all descendants (the log file
          plus any future siblings the plugin might write).

        When the operator sets ``debug_log_path: ""`` or to an
        absolute path outside any reachable directory, no rules are
        emitted — the operator owns that consequence per Daniel's
        rule against hardcoded fallbacks.
        """
        rules: List[str] = []

        # Part 1 (PR-153, server 0.6.136): debug_log_path grants.
        # The diagnostic log location is operator-configurable; the
        # fragment grants rw on the resolved parent dir (mkdir +
        # subtree pattern, same shape as file_edit PR-147).
        raw_path = plugin_config.get('debug_log_path', DEFAULT_DEBUG_LOG_PATH)
        if raw_path is None:
            raw_path = ''
        raw_path = str(raw_path)
        resolved = cls._resolve_debug_log_path(raw_path, workspace_path)
        if resolved:
            parent_dir = os.path.dirname(resolved)
            if parent_dir:
                rules.append(f"{parent_dir}/    rw,")
                rules.append(f"{parent_dir}/**  rw,")

        # Part 2 (PR-154, server 0.6.137): server-binary exec grants.
        # The lsp plugin's `connect_server` coroutine spawns each
        # configured server via `asyncio.create_subprocess_exec` —
        # under PR-148 confinement, the per-session apparmor profile
        # must grant `ix` (inherit-exec) on the canonical binary
        # path or the spawn fails with EACCES.  Symptom (v145):
        # `lsp logs` shows `Connection failed - [Errno 13] Permission
        # denied: jdtls` despite the daemon-side (unconfined)
        # instance happily spawning the same binary.  Closed by
        # reading the runtime .lsp.json at composer time + emitting
        # matching grants.
        rules.extend(
            cls._compose_lsp_server_exec_rules(workspace_path, plugin_config)
        )

        # Part 3 (PR-158, server 0.6.141): operator-supplied extra
        # apparmor rules from `plugin_configs.lsp.apparmor_extra_rules`
        # in the profile YAML.  Closes the chained-exec layer (e.g.
        # jdtls's Python wrapper execs `java` from the system JDK;
        # PR-154 covers `jdtls ix,` + `python3 ix,` but not `java ix,`
        # or `/usr/lib/jvm/** r,`).
        #
        # **CRITICAL TRUST BOUNDARY**: this knob is read from
        # `plugin_config` (= the profile YAML, `.jaato/profiles/*.yaml`)
        # — NOT from the workspace's `.lsp.json`.  Probe-confirmed
        # 2026-05-20: `.lsp.json` lives at workspace root and is
        # writable from BOTH runner main AND //child sub-profile (per
        # the `cascade_smoke/** rwkl,` rule emitted by the workspace-
        # tier fragment).  An LLM-driven tool could inject arbitrary
        # apparmor rules into `.lsp.json`, and the framework would
        # honor them on the next session — cross-session privilege
        # escalation.
        #
        # `.jaato/profiles/**` has `audit deny ... wlk,` on BOTH main
        # AND //child layers — operator-only territory, runtime-
        # tamper-proof.  So profile YAML is the safe location for
        # operator-supplied apparmor rules.
        rules.extend(
            cls._compose_lsp_apparmor_extra_rules(workspace_path, plugin_config)
        )

        return rules

    @classmethod
    def _compose_lsp_apparmor_extra_rules(
        cls,
        workspace_path: Optional[str],
        plugin_config: Dict[str, Any],
    ) -> List[str]:
        """Emit operator-supplied apparmor rules from profile YAML.

        Reads ``plugin_config["apparmor_extra_rules"]`` — a list of raw
        apparmor rule strings the operator wants spliced into the
        per-session profile.  Each rule is:

        1. **Type-checked** — non-string entries are silently skipped.
        2. **Stripped + emptiness-checked** — whitespace-only entries
           skipped.
        3. **Variable-expanded** via ``expand_variables(rule,
           workspace_root_override=workspace_path)`` so
           ``${workspaceRoot}`` and friends resolve to the session
           workspace (symmetric with PR-155 + PR-157 args resolution).
        4. **Emitted verbatim** — the operator-supplied rule is the
           final string the apparmor composer splices in.

        **Trust model** (recorded above in `get_apparmor_rules`): this
        knob is ONLY safe because profile YAMLs (`.jaato/profiles/`)
        are write-protected by the per-session apparmor profile's
        `audit deny .../.jaato/profiles/** wlk,` rules.  An attacker
        in the runner cannot modify the profile YAML to inject rules.

        If `apparmor_extra_rules` is absent / empty / non-list →
        returns empty list (degrades silently).

        Typical operator use case (jdtls JVM exec chain):

        ```yaml
        # .jaato/profiles/_base_codegen.yaml
        plugin_configs:
          lsp:
            apparmor_extra_rules:
              - "/usr/bin/java ix,"
              - "/usr/lib/jvm/** r,"
        ```
        """
        extra = plugin_config.get('apparmor_extra_rules')
        if not isinstance(extra, list):
            return []

        try:
            from ..subagent.config import expand_variables
        except ImportError:
            expand_variables = lambda v, **_kw: v  # noqa: E731

        rules: List[str] = []
        seen: set = set()  # de-dupe identical rules
        for rule in extra:
            if not isinstance(rule, str):
                continue
            stripped = rule.strip()
            if not stripped:
                continue
            try:
                expanded = expand_variables(
                    stripped,
                    workspace_root_override=workspace_path,
                )
            except Exception:  # noqa: BLE001 — composer boundary
                continue
            if not isinstance(expanded, str) or not expanded.strip():
                continue
            expanded = expanded.strip()
            if expanded in seen:
                continue
            seen.add(expanded)
            rules.append(expanded)
        return rules

    @classmethod
    def _load_lsp_config_static(
        cls,
        workspace_path: Optional[str],
        plugin_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Read `.lsp.json` at apparmor composer time.

        Replicates the runtime path-search of `_load_config_cache`
        (lsp/plugin.py:1815-1834) WITHOUT touching instance state and
        WITHOUT writing the diagnostic log entry (that's a runtime
        concern; the composer only needs to know which servers will
        be configured so it can emit exec grants for them).

        Search order matches the instance method exactly:
            1. ``plugin_config["config_path"]`` if set
            2. ``<workspace_path>/.lsp.json`` if workspace set
            3. ``~/.lsp.json`` fallback

        Returns the parsed JSON dict on first success, or None if no
        config file was found / loadable.  Composer callers MUST
        tolerate None (operator may legitimately not have a
        `.lsp.json` yet — apparmor fragment just emits no server
        rules in that case).
        """
        candidate_paths: List[str] = []
        custom = plugin_config.get('config_path')
        if custom:
            candidate_paths.append(str(custom))
        if workspace_path:
            candidate_paths.append(os.path.join(workspace_path, '.lsp.json'))
        candidate_paths.append(os.path.expanduser('~/.lsp.json'))

        for path in candidate_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (FileNotFoundError, IsADirectoryError):
                continue
            except (OSError, json.JSONDecodeError):
                # Unreadable / malformed — skip silently at composer
                # time.  The runtime _load_config_cache will hit the
                # same error and log it via _log_event there.
                continue
        return None

    @classmethod
    def _compose_lsp_server_exec_rules(
        cls,
        workspace_path: Optional[str],
        plugin_config: Dict[str, Any],
    ) -> List[str]:
        """Emit `ix` grants for each LSP server configured in `.lsp.json`.

        For each server entry:

        1. Resolve `spec["command"]` to a canonical absolute path via
           ``shutil.which`` + ``os.path.realpath`` (handles relative
           command names + symlinks).
        2. Emit ``<canonical> ix,`` so the runner can exec the binary.
        3. Emit ``<install-dir>/** r,`` for read-only access to
           bundled plugins, jars, config files (derived as the
           binary's grandparent dir — typical layout
           ``<install-dir>/bin/<command>``).
        4. If the binary's shebang is Python, emit the Python
           interpreter's path with ``ix,`` too — Python-wrapper
           scripts (e.g. jdtls) need both the wrapper AND the
           interpreter to be exec-permitted.

        Limitations (documented in README):

        - Apparmor profiles are composed per-session at bootstrap.
          Operator changes to `.lsp.json` mid-session don't update
          the profile.  Session restart required to pick up new
          servers.
        - Servers whose entry-point execs further binaries (e.g.
          jdtls's Python wrapper execs `java`) need additional
          grants for those execs.  This composer covers entry +
          (if Python wrapper) interpreter; further chain depths
          require operator-supplied grants via a future
          `apparmor_extra_rules` knob.
        """
        config = cls._load_lsp_config_static(workspace_path, plugin_config)
        if not isinstance(config, dict):
            return []
        servers = config.get('languageServers')
        if not isinstance(servers, dict):
            return []

        rules: List[str] = []
        seen_canonicals: set = set()  # de-dupe shared interpreters

        for _name, spec in servers.items():
            if not isinstance(spec, dict):
                continue
            command = spec.get('command', '')
            if not isinstance(command, str) or not command:
                continue

            canonical = cls._resolve_command_canonical(command)
            if not canonical or canonical in seen_canonicals:
                continue
            seen_canonicals.add(canonical)

            rules.append(f"{canonical} ix,")

            # Install-dir glob: binary's grandparent ("bin/" pattern).
            # If layout doesn't match (binary at /usr/bin/...) fall
            # back to the binary's parent dir for the read grant.
            install_dir = os.path.dirname(os.path.dirname(canonical))
            if install_dir and install_dir not in ('/', '/usr', '/usr/bin', '/usr/local'):
                rules.append(f"{install_dir}/** r,")

            # Shebang detection for Python-wrapper scripts.  jdtls
            # is the canonical case (`#!/usr/bin/env python3`); also
            # covers pylsp, pyright wrapper variants, etc.
            interpreter_path = cls._detect_shebang_interpreter(canonical)
            if interpreter_path and interpreter_path not in seen_canonicals:
                seen_canonicals.add(interpreter_path)
                rules.append(f"{interpreter_path} ix,")

            # Server-supplied data-directory grants (PR-155, server
            # 0.6.138).  When the operator passes ``-data <path>`` or
            # ``--data-dir <path>`` in the server's args, the binary
            # expects to write to that path — we auto-emit the
            # matching rw fragment so apparmor permits the writes.
            # Variable expansion (`${workspaceRoot}` etc.) matches
            # the runtime's `expand_variables` call so the granted
            # path always lines up with the path the binary actually
            # writes to.
            rules.extend(
                cls._compose_lsp_server_data_dir_rules(
                    spec.get('args', []), workspace_path
                )
            )

        return rules

    @classmethod
    def _compose_lsp_server_data_dir_rules(
        cls,
        args: Any,
        workspace_path: Optional[str],
    ) -> List[str]:
        """Detect ``-data <path>`` style flags in args + emit rw grants.

        Motivating case (v146-redo, server 0.6.137): the jdtls Python
        wrapper crashes at ``tempfile.gettempdir()`` before it can
        even invoke ``java``, because the apparmor-confined runner
        has no writable temp dir in the search list (``/tmp``,
        ``/var/tmp``, ``/usr/tmp`` all denied).  The wrapper would
        normally compute a data dir at ``/tmp/jdtls-<sha1(cwd)>``
        but can't reach /tmp.

        Operator fix: pass ``args: ["-data",
        "${workspaceRoot}/.jaato/jdtls-data"]`` in ``.lsp.json`` so
        the wrapper uses an explicit data dir.  This composer then
        auto-emits the matching apparmor grant (no separate
        operator action on the apparmor side).

        Recognised flags (covers the common LSP server conventions):
        - ``-data <path>`` — jdtls
        - ``--data-dir <path>`` — pyright, several others
        - ``--data <path>`` — alternate jdtls syntax

        Both forms are also accepted (apparmor-spec'd) when the
        flag uses ``=`` (``-data=<path>`` / ``--data-dir=<path>``)
        — operator convenience for shell-quoting.

        Variable expansion at composer time mirrors runtime
        `expand_variables` (workspace_root_override=workspace_path)
        so the granted path always matches what the binary writes.

        Emits two rules per recognised path following the file_edit
        PR-147 subtree pattern: ``<path>/ rw,`` + ``<path>/** rw,``.
        """
        if not isinstance(args, list) or not workspace_path:
            return []

        # Import here (not at module top) to avoid pulling subagent
        # plumbing into the lsp plugin's import path unnecessarily.
        try:
            from ..subagent.config import expand_variables
        except ImportError:
            # Older deployments without subagent in import path —
            # degrade gracefully (no variable expansion, raw value
            # used as-is if it happens to be absolute).
            expand_variables = lambda v, **_kw: v  # noqa: E731

        DATA_FLAGS = {'-data', '--data-dir', '--data'}
        rules: List[str] = []
        seen_paths: set = set()

        i = 0
        while i < len(args):
            arg = args[i]
            if not isinstance(arg, str):
                i += 1
                continue

            data_path_raw: Optional[str] = None

            # Form 1: `-data <path>` (next arg is the value)
            if arg in DATA_FLAGS:
                if i + 1 < len(args) and isinstance(args[i + 1], str):
                    data_path_raw = args[i + 1]
                    i += 2
                    if data_path_raw is None:
                        continue
                else:
                    i += 1
                    continue
            # Form 2: `-data=<path>` / `--data-dir=<path>` (combined)
            elif '=' in arg:
                flag, sep, value = arg.partition('=')
                if flag in DATA_FLAGS and value:
                    data_path_raw = value
                    i += 1
                else:
                    i += 1
                    continue
            else:
                i += 1
                continue

            # Expand ${workspaceRoot}, ${HOME}, etc. — same as
            # runtime so the apparmor grant matches what the binary
            # writes to.
            try:
                expanded = expand_variables(
                    data_path_raw,
                    workspace_root_override=workspace_path,
                )
            except Exception:  # noqa: BLE001 — composer boundary
                continue
            if not isinstance(expanded, str) or not expanded:
                continue

            # Resolve relative paths against workspace_path.
            if os.path.isabs(expanded):
                resolved = expanded
            else:
                resolved = os.path.join(workspace_path, expanded)

            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            rules.append(f"{resolved}/    rw,")
            rules.append(f"{resolved}/**  rw,")

        return rules

    @staticmethod
    def _extract_first_data_dir_from_args(
        args: Any,
        workspace_path: Optional[str],
    ) -> Optional[str]:
        """Return the first resolved ``-data`` / ``--data-dir`` path.

        Used at runtime by ``connect_server`` to compute the TMPDIR
        value injected into the LSP server's subprocess environment
        (PR-156, server 0.6.139).  Sidesteps the upstream jdtls
        eager-tempdir bug — ``jdtls.py:74`` computes
        ``tempfile.gettempdir()`` as the default value for the
        ``-data`` argparse arg BEFORE argparse parses CLI input.
        Under apparmor confinement that gettempdir() call crashes
        because /tmp / /var/tmp / /usr/tmp aren't reachable.
        Python's ``tempfile.gettempdir()`` honors TMPDIR first per
        its documented precedence — so injecting TMPDIR makes the
        wrapper's line 74 succeed.

        Pairs with :meth:`_compose_lsp_server_data_dir_rules` (PR-155,
        composer side): same args walk, same expansion, same
        resolution.  Composer emits an rw grant; runtime injects
        TMPDIR.  Both reach the same path so the wrapper's first
        write succeeds inside the apparmor profile.

        Differences from the composer helper:
        - Returns the FIRST recognized data path (TMPDIR is a
          single value; multiple ``-data`` flags would only
          confuse the wrapper anyway).
        - Returns ``None`` if no recognized flag is present OR
          workspace_path is None.

        Variable expansion mirrors the composer's call exactly via
        ``expand_variables(value, workspace_root_override=workspace_path)``.

        Forward-compatibility: upstream jdtls fixed this at commit
        ``d871e83`` (Oct 2025) by replacing gettempdir() with
        ``$HOME/.cache`` on Linux.  After that fix is widespread,
        our TMPDIR injection becomes a no-op for jdtls on Linux
        (the wrapper no longer reads TMPDIR) — harmless.  See
        ``feedback_lsp_jdtls_tempdir_eager_default_d871e83``.
        """
        if not isinstance(args, list) or not workspace_path:
            return None

        try:
            from ..subagent.config import expand_variables
        except ImportError:
            expand_variables = lambda v, **_kw: v  # noqa: E731

        DATA_FLAGS = {'-data', '--data-dir', '--data'}

        i = 0
        while i < len(args):
            arg = args[i]
            if not isinstance(arg, str):
                i += 1
                continue

            candidate: Optional[str] = None
            if arg in DATA_FLAGS:
                if i + 1 < len(args) and isinstance(args[i + 1], str):
                    candidate = args[i + 1]
            elif '=' in arg:
                flag, _, value = arg.partition('=')
                if flag in DATA_FLAGS and value:
                    candidate = value

            if candidate:
                try:
                    expanded = expand_variables(
                        candidate,
                        workspace_root_override=workspace_path,
                    )
                except Exception:  # noqa: BLE001 — runtime boundary
                    return None
                if not isinstance(expanded, str) or not expanded:
                    return None
                if os.path.isabs(expanded):
                    return expanded
                return os.path.join(workspace_path, expanded)

            i += 1

        return None

    @staticmethod
    def _resolve_command_canonical(command: str) -> Optional[str]:
        """Resolve an LSP server command to a canonical absolute path.

        Handles:
            - Absolute path commands — returned via realpath.
            - Bare names (`jdtls`, `pyright-langserver`) — resolved
              via shutil.which against the composer's PATH.
            - Returns None if the command cannot be resolved (the
              composer skips that server's grants; the runtime
              connect will fail loudly with the same EACCES it
              currently does, which is more diagnosable than a
              silent missing grant).
        """
        if os.path.isabs(command):
            target = command
        else:
            target = shutil.which(command)
            if not target:
                return None
        try:
            return os.path.realpath(target)
        except OSError:
            return None

    @staticmethod
    def _detect_shebang_interpreter(script_path: str) -> Optional[str]:
        """Read the script's shebang and resolve its interpreter.

        Returns the canonical path to the interpreter if the script
        starts with `#!` AND the interpreter is Python (covers the
        common LSP-wrapper-script case).  Returns None for binaries,
        non-Python scripts, unreadable files, etc.

        Why Python-specific: the standing motivating case is jdtls
        (Eclipse JDT LS) which ships as a `#!/usr/bin/env python3`
        wrapper script.  Other interpreter languages would need
        analogous handling but are deferred until evidence supports
        them — bare-name LSP servers in the wild are overwhelmingly
        either compiled binaries (rust-analyzer, gopls, clangd) or
        Python wrappers (jdtls, pylsp).
        """
        try:
            with open(script_path, 'rb') as f:
                first_line = f.readline()
        except OSError:
            return None
        if not first_line.startswith(b'#!'):
            return None
        try:
            shebang = first_line[2:].decode('utf-8', errors='replace').strip()
        except UnicodeDecodeError:
            return None
        # Handle `#!/usr/bin/env python3` and `#!/usr/bin/python3`
        if 'python' not in shebang.lower():
            return None
        # If the shebang uses /usr/bin/env, the actual interpreter
        # name is the next token.  Otherwise the shebang itself is
        # the interpreter path.
        parts = shebang.split()
        if parts and parts[0].endswith('/env') and len(parts) > 1:
            interpreter_name = parts[1].split('=')[0]  # strip flags
            return shutil.which(interpreter_name)
        if parts:
            return os.path.realpath(parts[0]) if os.path.exists(parts[0]) else parts[0]
        return None

    def _parse_wait_knob(
        self,
        raw: Any,
        default: float,
        floor: float,
        ceiling: float,
        knob_name: str,
    ) -> float:
        """Parse + clamp a `plugin_configs.lsp.*_wait_seconds` value.

        Non-numeric input falls back to the default (with a trace);
        in-range numeric input passes through unchanged; out-of-range
        values are clamped (with a trace).  Mirrors the connect_timeout
        knob shape so the operator-facing contract stays uniform.
        """
        try:
            value = float(raw)
        except (TypeError, ValueError):
            self._trace(
                f"initialize: {knob_name}={raw!r} is not a number — "
                f"falling back to default {default}s"
            )
            return default
        clamped = max(floor, min(value, ceiling))
        if clamped != value:
            self._trace(
                f"initialize: {knob_name}={value} clamped to {clamped} "
                f"(range [{floor}, {ceiling}])"
            )
        return clamped

    def set_workspace_path(self, path: str) -> None:
        """Set the workspace path for finding config files.

        This should be called when the client's working directory changes.
        It will trigger a reload of the config file from the new location.

        PR-157 (server 0.6.140): also dispatches MSG_RETRY_AUTOCONNECT
        to the background thread so any server NOT yet in
        ``self._connected_servers`` gets a fresh connect attempt
        with the now-correct ``self._workspace_path``.  Closes the
        lifecycle gap where the initial auto-connect loop runs
        BEFORE the framework's ``set_workspace_path()`` broadcast
        fires (broadcast happens after ``expose_all()`` which calls
        ``initialize()``; the lsp background thread starts inside
        ``initialize()``).  Without this retry, the initial connect
        attempts use ``self._workspace_path = None`` →
        ``expand_variables(...)`` auto-detects to daemon cwd → wrong
        TMPDIR + wrong `-data` arg path.  See
        feedback_advisor_outcome_pings_push_not_pull (kb-side) and
        the v148 evidence chain.
        """
        if path != self._workspace_path:
            self._workspace_path = path
            self._trace(f"workspace_path changed to: {path}")
            # Force reload config on next access
            if self._initialized:
                self._load_config_cache(force=True)
                # PR-157: trigger connect-retry for any not-yet-connected
                # server.  Fire-and-forget — the request loop handles
                # MSG_RETRY_AUTOCONNECT without pushing to response_queue
                # (see _thread_main's request loop).
                if self._request_queue is not None:
                    self._trace(
                        "set_workspace_path: dispatching MSG_RETRY_AUTOCONNECT "
                        "to retry any not-yet-connected server"
                    )
                    self._request_queue.put((MSG_RETRY_AUTOCONNECT, {}))

    def _resolve_path(self, path: str) -> str:
        """Resolve a path to an absolute path.

        If path is relative, resolves it against workspace_path (if set)
        or falls back to os.path.abspath (resolves against cwd).

        Args:
            path: Path to resolve (can be relative or absolute).

        Returns:
            Absolute path.
        """
        if os.path.isabs(path):
            return path
        if self._workspace_path:
            return os.path.abspath(os.path.join(self._workspace_path, path))
        # No workspace set — return path as-is (cannot resolve against CWD
        # in daemon mode as it would leak the server's directory)
        return path

    def shutdown(self) -> None:
        """Shutdown the LSP plugin and clean up resources.

        Sending the ``(None, None)`` sentinel makes the server thread
        reap every connected LSP server (``await client.stop()`` —
        terminate → wait → kill) BEFORE it exits its event loop, so the
        jdtls subprocesses don't leak.  The join timeout (15s) bounds the
        reap: ``client.stop()`` waits up to 5s per server for graceful
        termination before SIGKILL, so a single jdtls finishes well within
        the window.  See #284 (per-slot jdtls leak)."""
        self._trace("shutdown: cleaning up resources")
        if self._request_queue:
            self._request_queue.put((None, None))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
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

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset (Phase 1, server 0.6.142+) — NO-OP.

        **This is THE cascade-sharing target plugin.**  Per Daniel's
        litmus test, EVERY piece of this plugin's state must survive
        between sessions of the same cascade because the next session
        DIRECTLY BENEFITS from:

        - ``_clients``: ALREADY-CONNECTED LSP server clients.  Closing
          them between sessions would re-trigger the multi-minute
          jdtls cold-start tax every cascade stage pays today.  The
          v141-v151 onion-peeling exists precisely because per-session
          teardown was the wrong shape.
        - ``_connected_servers``: the membership set the enrichment
          chain consults.  Clearing it would re-create the v151
          multi-instance state-isolation symptom (enrichment instance
          sees empty set while connect instance had the server
          registered).
        - ``_config_cache``: parsed ``.lsp.json`` — the workspace
          config doesn't change within a cascade.
        - Background thread + request_queue + response_queue: the
          machinery owning LSP client lifecycles — tearing them down
          + re-creating would discard exactly the LSP state we want
          to preserve.
        - ``_diagnostics_events`` (on each LSPClient): the per-URI
          asyncio.Event signal store — cleared per-URI on each
          ``await_diagnostics()`` consume cycle (PR-151), not
          per-session.

        Nothing held by this plugin instance is per-session-only.
        ``reset_for_next_session()`` is the framework's contract for
        "between cascade sessions"; for lsp specifically, the contract
        is "don't touch anything".  ``shutdown()`` is the final
        cascade-end teardown — still tears down clients + thread.
        """
        self._trace(
            "reset_for_next_session: NO-OP — LSP client connections + "
            "_connected_servers set + config cache MUST survive across "
            "cascade sessions (cascade-sharing target plugin)"
        )

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
                },
                category="code",
                discoverability=DISCOVERABILITY_DEFERRED,
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
                },
                category="code",
                discoverability=DISCOVERABILITY_DEFERRED,
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
                },
                category="code",
                discoverability=DISCOVERABILITY_DEFERRED,
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
                },
                category="code",
                discoverability=DISCOVERABILITY_DEFERRED,
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
                },
                category="code",
                discoverability=DISCOVERABILITY_DEFERRED,
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
                },
                category="code",
                discoverability=DISCOVERABILITY_DEFERRED,
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
                },
                category="code",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_FILE_WRITER}),
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
                },
                category="code",
                discoverability=DISCOVERABILITY_DEFERRED,
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
                },
                category="code",
                discoverability=DISCOVERABILITY_DEFERRED,
                traits=frozenset({TRAIT_FILE_WRITER}),
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mappings for LSP tools.

        Phase 3 §3.5 wave 2: forwards via runner-RPC when a runner
        is attached so spawned LSP-server subprocesses inherit the
        runner's AppArmor profile.  Falls through to in-process
        otherwise.
        """
        if not self._initialized:
            self.initialize()

        return self.wrap_executors_for_runner_forwarding({
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
        })

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

    def _daemon_must_not_host_lsp(self) -> bool:
        """True when this LSP instance runs in the daemon process and must
        therefore NOT host a language server (suppress the lifecycle).

        Delegates to :func:`_running_in_daemon_process`.  See that function for
        the full rationale; in short: the daemon-side LSP is a forwarding stub
        for executor RPC, and any jdtls it spawns leaks resident in the
        long-lived daemon → OOM (#284).  The per-session runner hosts LSP
        instead (in a reapable slot).

        **History (#285):** this used to read ``registry.runner_rpc`` via
        :meth:`_runner_rpc_handle`, but the diagnostic proved the daemon-side
        instance has no registry reference at connect time, so that gate never
        fired.  Process identity is the only reliable signal.
        """
        return _running_in_daemon_process()

    def subscribes_to_tool_result_enrichment(self) -> bool:
        """Subscribe to tool result enrichment to auto-run diagnostics after file writes.

        When enabled, the LSP plugin will automatically run diagnostics on files
        that are modified by file-writing tools (updateFile, writeNewFile, etc.)
        and append diagnostic information to the tool result.

        Returns ``False`` in the daemon process (#284): diagnostics enrichment
        must run on the runner-side instance that owns the workspace and hosts
        jdtls in a reapable slot — never daemon-side, where jdtls would
        accumulate resident and OOM the daemon.  See
        :meth:`_daemon_must_not_host_lsp`.
        """
        if self._daemon_must_not_host_lsp():
            return False
        return True

    def get_tool_result_enrichment_priority(self) -> int:
        """Run after basic file operations but before other enrichment.

        Priority 30 ensures diagnostics are added early in the enrichment chain.
        """
        return 30

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str,
        tool_args: Optional[Dict[str, Any]] = None
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
        total_errors = sum(
            sum(1 for d in diags if d.get("severity") == "Error")
            for diags in all_diagnostics.values()
        )
        total_warnings = sum(
            sum(1 for d in diags if d.get("severity") == "Warning")
            for diags in all_diagnostics.values()
        )
        metadata = {
            "files_checked": list(supported_files),
            "files_with_diagnostics": list(all_diagnostics.keys()),
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            # Per-file structured diagnostics — same dicts the
            # ``## LSP Diagnostics`` markdown summary is built from, but
            # surfaced structured so completion-gate processors (consumers
            # of ``context.tool_calls[i].enrichment_metadata["lsp"]``) can
            # quote specific Error messages back to the agent for
            # actionable retry instead of just "you have N errors".  Each
            # entry has ``severity`` (str), ``line`` (int), ``message``
            # (str), ``source`` (str), ``character`` (int), as produced
            # by ``_format_diagnostics``.
            "diagnostics": dict(all_diagnostics),
            "_telemetry": {
                "jaato.enrichment.lsp.files_checked": len(supported_files),
                "jaato.enrichment.lsp.total_errors": total_errors,
                "jaato.enrichment.lsp.total_warnings": total_warnings,
            },
        }

        return ToolResultEnrichmentResult(result=enriched_result, metadata=metadata)

    def _extract_file_paths_from_result(
        self,
        tool_name: str,
        result: str
    ) -> List[str]:
        """Extract file paths from a tool result.

        Uses generic key inspection that matches the ``file_writer`` trait
        contract (see :data:`TRAIT_FILE_WRITER`):

        - ``"path"``  — single-file operations.
        - ``"files_modified"`` — multi-file operations.
        - ``"changes"[].file`` — detailed change records.

        Args:
            tool_name: The tool that produced the result (unused, kept for
                signature compatibility with enrichment callers).
            result: The JSON-serialized result string.

        Returns:
            List of file paths found in the result.
        """
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return []

        if not isinstance(data, dict) or data.get("error"):
            return []

        file_paths = []

        # Single-file key
        path = data.get("path")
        if path:
            file_paths.append(path)

        # Multi-file key
        files_modified = data.get("files_modified", [])
        file_paths.extend(files_modified)

        # Detailed changes array
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

        # Resolve path against workspace
        abs_file_path = self._resolve_path(file_path)

        # Check if file type is supported
        ext = os.path.splitext(abs_file_path)[1].lower()
        if ext not in EXT_TO_LANGUAGE:
            self._trace(f"get_file_dependents: unsupported file type {ext}")
            return []

        # Ensure all files of the same type in the workspace are indexed
        # This is needed for find_references to work across files
        workspace_dir = os.path.dirname(abs_file_path)
        self._trace(f"get_file_dependents: ensuring workspace indexed at {workspace_dir}")
        self._execute_method('_ensure_workspace_indexed', {
            'directory': workspace_dir,
            'extension': ext,  # Pass the file extension to filter by language
            'file_path': abs_file_path  # Pass resolved path so correct LSP server is selected
        })

        # Get document symbols
        symbols_result = self._execute_method('document_symbols', {'file_path': abs_file_path})
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
            with open(abs_file_path, 'r', encoding='utf-8', errors='replace') as f:
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

    def get_config_schema(self) -> List[PluginSetting]:
        return [
            PluginSetting(
                name="config_path",
                type="str",
                default="",
                description=(
                    "Override path to .lsp.json. Defaults to "
                    "<workspace>/.lsp.json then ~/.lsp.json."
                ),
            ),
            PluginSetting(
                name="connect_timeout_seconds",
                type="float",
                default=DEFAULT_CONNECT_TIMEOUT_SECONDS,
                description=(
                    "Per-server LSP `initialize` handshake timeout. "
                    "Raise for heavy-init servers (jdtls on a cold Maven / "
                    "Gradle workspace routinely needs 60-120s+). Clamped to "
                    f"[{MIN_CONNECT_TIMEOUT_SECONDS}, "
                    f"{MAX_CONNECT_TIMEOUT_SECONDS}]."
                ),
            ),
            PluginSetting(
                name="diagnostics_max_wait_seconds",
                type="float",
                default=DEFAULT_DIAGNOSTICS_MAX_WAIT_SECONDS,
                description=(
                    "Upper bound on the post-didOpen / post-didChange "
                    "wait for the server's first "
                    "`textDocument/publishDiagnostics` batch. Returns "
                    "as soon as the batch arrives. Raise for heavy "
                    "servers (jdtls cold Maven: 3-8s). "
                    f"Clamped to [{MIN_DIAGNOSTICS_MAX_WAIT_SECONDS}, "
                    f"{MAX_DIAGNOSTICS_MAX_WAIT_SECONDS}]; 0 = no "
                    "wait, read cache as-is."
                ),
            ),
            PluginSetting(
                name="diagnostics_min_wait_seconds",
                type="float",
                default=DEFAULT_DIAGNOSTICS_MIN_WAIT_SECONDS,
                description=(
                    "Floor on the post-didOpen wait. Even when an "
                    "early `publishDiagnostics` arrives, we wait at "
                    "least this long so multi-stage analysis pipelines "
                    "(parser → compiler → linter) deliver later "
                    "batches before the cache read. Clamped to "
                    "[0, diagnostics_max_wait_seconds]."
                ),
            ),
            PluginSetting(
                name="diagnostics_convergence_window_seconds",
                type="float",
                default=DEFAULT_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS,
                description=(
                    "After the first `publishDiagnostics` lands, keep "
                    "listening for follow-up batches that overwrite "
                    "the cache. Each follow-up resets the window. "
                    "Closes the convergence race where the first "
                    "publish carries transient errors (e.g. jdtls's "
                    "intra-project imports still resolving) and a "
                    "follow-up publish 1-3s later carries the settled "
                    "state. `0.0` disables the loop (legacy "
                    "first-publish semantics). Empirical default "
                    "3.0s (2026-06-05 instrumented cascade analysis). "
                    f"Clamped to [{MIN_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS}, "
                    f"{MAX_DIAGNOSTICS_CONVERGENCE_WINDOW_SECONDS}]."
                ),
            ),
            PluginSetting(
                name="debug_log_path",
                type="str",
                default=DEFAULT_DEBUG_LOG_PATH,
                description=(
                    "Path to the lsp plugin's diagnostic log "
                    "(append-only).  Relative paths resolve against "
                    "the session's workspace_path; absolute paths "
                    "pass through.  Default is workspace-relative so "
                    "the per-session apparmor profile composed by "
                    "`get_apparmor_rules` covers the write.  Empty "
                    "string disables the diagnostic log entirely."
                ),
            ),
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
            return f"Unknown subcommand: {subcommand}\n\nUse 'lsp help' for available commands."

    def _cmd_help(self) -> HelpLines:
        """Return detailed help text for pager display."""
        return HelpLines(lines=[
            ("LSP Command", "bold"),
            ("", ""),
            ("Manage Language Server Protocol (LSP) servers. LSP servers provide language", ""),
            ("intelligence features like diagnostics, completions, and go-to-definition.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    lsp [subcommand] [args]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    list              List all configured LSP servers with their status", "dim"),
            ("                      (this is the default when no subcommand is given)", "dim"),
            ("", ""),
            ("    status            Show detailed connection status of all servers", "dim"),
            ("                      Includes capabilities and error information", "dim"),
            ("", ""),
            ("    connect <name>    Connect to a configured but disconnected server", "dim"),
            ("                      Server must be defined in .lsp.json", "dim"),
            ("", ""),
            ("    disconnect <name> Disconnect from a running server", "dim"),
            ("                      Keeps configuration, just stops the connection", "dim"),
            ("", ""),
            ("    reload            Reload configuration from .lsp.json", "dim"),
            ("                      Picks up external changes to the config file", "dim"),
            ("", ""),
            ("    logs [clear]      Show interaction logs for debugging", "dim"),
            ("                      Use 'clear' to reset the log buffer", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    lsp                       List all configured servers", "dim"),
            ("    lsp status                Show detailed server status", "dim"),
            ("    lsp connect python        Connect to Python language server", "dim"),
            ("    lsp disconnect python     Disconnect Python server", "dim"),
            ("    lsp reload                Reload .lsp.json config", "dim"),
            ("    lsp logs                  Show interaction logs", "dim"),
            ("    lsp logs clear            Clear all logs", "dim"),
            ("", ""),
            ("CONFIGURATION FILE", "bold"),
            ("    LSP servers are configured in .lsp.json:", ""),
            ("", ""),
            ('    {', "dim"),
            ('      "languageServers": {', "dim"),
            ('        "python": {', "dim"),
            ('          "command": "pyright-langserver",', "dim"),
            ('          "args": ["--stdio"],', "dim"),
            ('          "languageId": "python",', "dim"),
            ('          "rootUri": "${workspaceFolder}"', "dim"),
            ('        },', "dim"),
            ('        "typescript": {', "dim"),
            ('          "command": "typescript-language-server",', "dim"),
            ('          "args": ["--stdio"],', "dim"),
            ('          "languageId": "typescript"', "dim"),
            ('        }', "dim"),
            ('      }', "dim"),
            ('    }', "dim"),
            ("", ""),
            ("SERVER CONFIGURATION OPTIONS", "bold"),
            ("    command           Path or name of the language server executable", "dim"),
            ("    args              Command-line arguments (usually [\"--stdio\"])", "dim"),
            ("    languageId        Language identifier (e.g., \"python\", \"typescript\")", "dim"),
            ("    rootUri           Workspace root (default: ${workspaceFolder})", "dim"),
            ("    initializationOptions", "dim"),
            ("                      Server-specific initialization options", "dim"),
            ("", ""),
            ("COMMON LANGUAGE SERVERS", "bold"),
            ("    Python:           pyright-langserver, pylsp, python-lsp-server", "dim"),
            ("    TypeScript/JS:    typescript-language-server", "dim"),
            ("    Rust:             rust-analyzer", "dim"),
            ("    Go:               gopls", "dim"),
            ("    C/C++:            clangd", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - Servers auto-connect on startup if configured in .lsp.json", "dim"),
            ("    - LSP provides diagnostics, hover info, and completions to the model", "dim"),
            ("    - Failed servers show error details in 'lsp status'", "dim"),
            ("    - Use 'lsp logs' to debug communication issues", "dim"),
        ])

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
        # Use workspace_path if set; skip workspace-relative path otherwise
        if self._workspace_path:
            paths.append(os.path.join(self._workspace_path, '.lsp.json'))
        paths.append(os.path.expanduser('~/.lsp.json'))

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._config_cache = json.load(f)
                    self._config_path = path
                    self._log_event(LOG_INFO, f"Loaded config from {path}")
                except (OSError, json.JSONDecodeError) as e:
                    # Real config-load failure (file unreadable / not
                    # JSON).  Reset the cache that the json.load may
                    # have partially populated, then try the next
                    # path.  This branch is now scoped to actual JSON
                    # errors — diagnostic-log write failures used to
                    # be swallowed here too (pre-0.6.136) which made
                    # apparmor-blocked writes look like config-load
                    # failures and silently broke the entire
                    # enrichment chain.
                    self._config_cache = {}
                    self._log_event(LOG_WARN, f"Failed to load {path}: {e}")
                    continue

                # Diagnostic side-channel: write the load event to
                # the operator-configured diagnostic log.  This MUST
                # NOT abort config loading if it fails — the log is
                # for human inspection, not load-bearing.  Pre-0.6.136
                # this was wrapped in the same outer try/except as the
                # json.load, which silently broke the entire LSP
                # enrichment chain when apparmor denied the write.
                debug_path = self._resolve_debug_log_path(
                    self._debug_log_path_raw, self._workspace_path
                )
                if debug_path:
                    try:
                        parent_dir = os.path.dirname(debug_path)
                        if parent_dir:
                            os.makedirs(parent_dir, exist_ok=True)
                        session_tag = f":{self._session_id}" if self._session_id else ""
                        with open(debug_path, "a") as df:
                            df.write(f"[LSP{session_tag}] Config loaded from: {path}\n")
                            servers = self._config_cache.get('languageServers', {})
                            for name, spec in servers.items():
                                df.write(
                                    f"[LSP{session_tag}]   Server '{name}': "
                                    f"command={spec.get('command')}, "
                                    f"args={spec.get('args', [])}\n"
                                )
                            df.flush()
                    except OSError as e:
                        # Apparmor-denied, disk full, parent missing,
                        # etc.  Surface the failure via the in-memory
                        # log so `lsp logs` shows it, but do NOT abort
                        # the load — the config is already cached.
                        self._log_event(
                            LOG_WARN,
                            f"Failed to write debug log to {debug_path}: {e}",
                        )
                return
        self._config_cache = {}

    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        # #284/#285: the daemon process must NOT run the LSP background thread.
        # The thread's auto-connect spawns a jdtls that leaks resident in the
        # long-lived daemon (no owning slot, never reaped) → OOM.  Executor
        # forwarding (RunnerForwardingMixin) does not use this thread, so the
        # daemon-side stub keeps forwarding tool calls to the runner; only the
        # LSP server lifecycle (connect + diagnostics) is suppressed.  The
        # per-session runner hosts jdtls in a reapable slot instead.  This is
        # the earliest, single chokepoint — no thread means no auto-connect and
        # no connect_server.
        if self._daemon_must_not_host_lsp():
            logger.info(
                "LSP background thread suppressed in daemon process pid=%d "
                "(runner-side hosts jdtls; #284)", os.getpid(),
            )
            return

        # Register the atexit reaper the first time a server thread starts
        # (i.e. the first time jdtls et al become spawnable).  Idempotent.
        self._register_atexit_reaper()

        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def _register_atexit_reaper(self) -> None:
        """Register the process-exit jdtls reaper exactly once."""
        if self._atexit_registered:
            return
        try:
            atexit.register(self._atexit_reap_jdtls)
            self._atexit_registered = True
        except Exception:  # pragma: no cover — atexit.register never raises
            pass

    def _atexit_reap_jdtls(self) -> None:
        """Process-exit backstop: SIGKILL every still-running LSP server
        subprocess when THIS process (a runner / pre-warm pool slot) exits.

        Root cause (#284 residual, per-slot jdtls leak): the pool-slot
        teardown path — daemon closes the slot socket → the runner's RPC
        ``serve()`` returns on EOF → the slot ``sys.exit(0)`` — does NOT
        call ``plugin.shutdown()``.  So a connected jdtls (0.5-1.5 GB each)
        was abandoned, re-parented to the daemon subreaper, and accumulated
        one-per-slot across cascade stages until the daemon OOMed.

        ``atexit`` fires on the clean ``sys.exit()`` the slot uses, so the
        subprocess dies WITH its slot.  We SIGKILL by pid (not the async
        ``client.stop()``) because the LSP event loop is already gone at
        interpreter exit.  Warm cascade-reuse is unaffected: this only
        fires when the slot PROCESS itself dies — exactly when a
        slot-scoped jdtls should die too.  PDEATHSIG (PR-277) remains the
        backstop for the ungraceful SIGKILL/OOM case atexit can't catch.
        """
        for client in list(self._clients.values()):
            proc = getattr(client, "_process", None)
            pid = getattr(proc, "pid", None)
            if not pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    async def _reap_failed_client(self, name: str, client) -> None:
        """Terminate a partially-started LSP client after a failed or
        timed-out ``connect_server`` so its subprocess does not leak.

        ``LSPClient.start()`` spawns the language-server subprocess
        (``create_subprocess_exec``) BEFORE the slow ``_initialize()``
        handshake.  When ``connect_server`` times out (or raises) the
        coroutine is cancelled but the spawned process is left running and
        UNTRACKED — ``self._clients[name]`` is only assigned on the success
        path.  ``client.stop()`` is the same teardown the success path uses
        in ``disconnect_server`` (cancel reader task → terminate → wait →
        kill), so it reaps regardless of whether ``_initialize()`` had
        completed.  No-op when no client was spawned yet (``client is None``,
        e.g. failure before ``LSPClient(...)``).  See #284.
        """
        if client is None:
            return
        try:
            await client.stop()
            self._trace(f"reaped failed/timed-out LSP client '{name}'")
        except Exception as e:
            self._trace(
                f"reap of failed LSP client '{name}' raised (ignored): {e}"
            )

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
                # #284 defense-in-depth: the daemon process must never spawn a
                # language server (it leaks resident → OOM).  The primary gate
                # is in _ensure_thread (the daemon never starts this background
                # thread), so reaching here in the daemon means the thread gate
                # regressed — refuse + log loudly.  Runner/slot processes
                # proceed normally.
                if self._daemon_must_not_host_lsp():
                    logger.error(
                        "LSP connect_server reached in daemon process pid=%d "
                        "despite the _ensure_thread gate — refusing (server=%s, "
                        "#284). FIX the thread gate.", os.getpid(), name,
                    )
                    return False
                self._log_event(LOG_INFO, "Connecting to server", server=name)
                client = None
                try:
                    # Expand variables in args (e.g., ${workspaceRoot}).
                    # PR-157 (server 0.6.140): pass
                    # workspace_root_override=self._workspace_path so
                    # ${workspaceRoot} resolves to the session workspace
                    # — NOT the daemon cwd auto-detect fallback.  Without
                    # this, `args: ["-data", "${workspaceRoot}/.jaato/jdtls-data"]`
                    # would expand to the daemon source tree at
                    # connect_server time (the auto-detect path), even
                    # though PR-155's composer correctly used the session
                    # workspace.  Asymmetric resolution closed.
                    raw_args = spec.get('args', [])
                    expanded_args = expand_variables(
                        raw_args,
                        workspace_root_override=self._workspace_path,
                    )
                    # Use workspace-based root_uri when config doesn't specify one
                    root_uri = spec.get('rootUri')
                    if not root_uri and self._workspace_path:
                        root_uri = f"file://{self._workspace_path}"

                    # PR-156 (server 0.6.139): auto-inject TMPDIR from
                    # an operator-supplied `-data <path>` arg.
                    # Sidesteps the upstream jdtls eager-tempdir bug
                    # (`jdtls.py:74` computes `tempfile.gettempdir()`
                    # before argparse parses `-data`; under apparmor
                    # confinement this crashes if /tmp et al denied).
                    # Python's `tempfile.gettempdir()` reads TMPDIR
                    # first per its documented precedence chain — so
                    # injecting it makes line 74 succeed.  Operator
                    # `env.TMPDIR` always wins (never overridden).
                    # See feedback_lsp_jdtls_tempdir_eager_default_d871e83
                    # memory; upstream fix landed in commit d871e83
                    # (Oct 2025) but pre-fix builds remain in the wild.
                    augmented_env = dict(spec.get('env') or {})
                    if 'TMPDIR' not in augmented_env:
                        data_dir = LSPToolPlugin._extract_first_data_dir_from_args(
                            expanded_args, self._workspace_path
                        )
                        if data_dir:
                            augmented_env['TMPDIR'] = data_dir
                            self._trace(
                                f"connect_server '{name}': "
                                f"auto-injected TMPDIR={data_dir} "
                                f"from -data arg (PR-156 jdtls eager-tempdir workaround)"
                            )

                    config = ServerConfig(
                        name=name,
                        command=spec.get('command', ''),
                        args=expanded_args,
                        env=augmented_env if augmented_env else None,
                        root_uri=root_uri,
                        language_id=spec.get('languageId'),
                        extra_paths_key=spec.get('extraPathsKey'),
                    )
                    self._trace(f"Starting LSP server '{name}': command={config.command}, args={config.args}")
                    client = LSPClient(config, errlog=self._errlog)
                    await asyncio.wait_for(
                        client.start(), timeout=self._connect_timeout_seconds
                    )
                    self._clients[name] = client
                    self._connected_servers.add(name)
                    self._failed_servers.pop(name, None)
                    self._log_event(LOG_INFO, "Connected successfully", server=name)
                    return True
                except asyncio.TimeoutError:
                    self._failed_servers[name] = "Connection timed out"
                    self._log_event(LOG_ERROR, "Connection timed out", server=name)
                    # Reap the partially-started server.  ``LSPClient.start()``
                    # spawns the subprocess BEFORE the (slow) initialize
                    # handshake, so a timeout cancels the coroutine but leaves a
                    # live, UNTRACKED process (``self._clients[name]`` was never
                    # set).  Without this the orphaned jdtls leaks AND the
                    # retry-autoconnect spawns a duplicate (it sees the server as
                    # not-connected), accumulating one jdtls per timeout until
                    # the daemon OOMs.  See #284.
                    await self._reap_failed_client(name, client)
                    return False
                except Exception as e:
                    self._failed_servers[name] = str(e)
                    self._log_event(LOG_ERROR, "Connection failed", server=name, details=str(e))
                    await self._reap_failed_client(name, client)
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

            # Auto-connect to configured servers.
            # PR-157 (server 0.6.140): defer if workspace_path isn't
            # set yet.  This thread starts inside `initialize()` —
            # BEFORE the framework's `set_workspace_path()` broadcast
            # fires (which happens after `expose_all()`).  If
            # workspace_path is None, `${workspaceRoot}` in server
            # args + the TMPDIR injection helper both fall back to
            # daemon-cwd auto-detect, producing wrong-path connects.
            # `set_workspace_path()` will dispatch
            # MSG_RETRY_AUTOCONNECT once available; the request loop
            # picks it up and retries each server with the now-correct
            # workspace_path.
            if self._workspace_path is None:
                self._log_event(
                    LOG_INFO,
                    "Auto-connect deferred: workspace_path not set yet; "
                    "will retry on set_workspace_path()",
                )
            else:
                for name, spec in servers.items():
                    if spec.get('autoConnect', True):
                        await connect_server(name, spec)

            self._log_event(LOG_INFO, f"Initialization complete: {len(self._connected_servers)} connected")

            # Process requests from main thread
            while True:
                try:
                    req = self._request_queue.get(timeout=0.1)
                    if req is None or req == (None, None):
                        # Graceful reap: stop every connected LSP server
                        # before the event loop exits so the subprocesses
                        # (jdtls) don't outlive this plugin.  shutdown()
                        # previously dropped self._clients WITHOUT stopping
                        # them (#284 residual).  Iterate captured client
                        # objects (not dict lookups) so a concurrent
                        # shutdown() clearing self._clients can't KeyError.
                        for _name, _client in list(self._clients.items()):
                            try:
                                await _client.stop()
                                self._trace(f"shutdown: reaped LSP server '{_name}'")
                            except Exception as _e:
                                self._trace(
                                    f"shutdown: reap of LSP server "
                                    f"'{_name}' raised (ignored): {_e}"
                                )
                        self._clients.clear()
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

                    elif msg_type == MSG_RETRY_AUTOCONNECT:
                        # PR-157 (server 0.6.140): set_workspace_path
                        # arrived after the initial auto-connect loop.
                        # Re-attempt connect for any server NOT in
                        # self._connected_servers using the freshly-set
                        # self._workspace_path so ${workspaceRoot}
                        # resolves correctly + TMPDIR auto-inject
                        # targets the session workspace.
                        #
                        # Clear failed_servers so a retry isn't
                        # short-circuited by a previous failure state.
                        # Fire-and-forget — no response_queue push (the
                        # sender doesn't wait).
                        retry_servers = self._config_cache.get('languageServers', {})
                        self._log_event(
                            LOG_INFO,
                            f"Retry auto-connect: workspace_path now set, "
                            f"re-attempting {len(retry_servers)} configured server(s)",
                        )
                        for name, spec in retry_servers.items():
                            if name in self._connected_servers:
                                continue
                            if not spec.get('autoConnect', True):
                                continue
                            # Clear any prior failure to ensure a
                            # clean retry attempt.
                            self._failed_servers.pop(name, None)
                            try:
                                await connect_server(name, spec)
                            except Exception as e:  # noqa: BLE001
                                self._log_event(
                                    LOG_ERROR,
                                    f"Retry connect raised: {name}: {e}",
                                )

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

        # Methods that require full parsing need to wait for the server
        # to emit `textDocument/publishDiagnostics` (or the equivalent
        # post-parse symbol index, in the case of document_symbols /
        # goto_definition / find_references / hover).  Pre-0.6.134 this
        # was a fixed `asyncio.sleep(0.8)` that timed out before
        # heavy-init servers (Eclipse JDT LS on Maven workspaces) had
        # delivered their first diagnostic batch.  Now we wait on a
        # per-URI asyncio.Event signalled by the JSON-RPC reader, so
        # fast servers return in <500ms and slow servers get up to
        # `diagnostics_max_wait_seconds` (operator-configurable).
        needs_parsing = method in (
            'hover', 'document_symbols', 'goto_definition',
            'find_references', 'get_diagnostics',
        )

        # Ensure document is open and up-to-date
        # update_document opens if not open, or sends didChange if already open
        if file_path and method not in ('workspace_symbols',):
            await client.update_document(file_path)
            # Wait for server to process the document.  Bounded poll
            # for parsing-heavy methods; small fixed delay for
            # lightweight ones (workspace_symbols already excluded
            # above).  The min_wait floor gives multi-stage analysis
            # pipelines (parser → compiler → linter) room to deliver
            # later batches before the cache read.
            if needs_parsing:
                await client.await_diagnostics(
                    file_path,
                    max_wait=self._diagnostics_max_wait_seconds,
                    min_wait=self._diagnostics_min_wait_seconds,
                    convergence_window=(
                        self._diagnostics_convergence_window_seconds
                    ),
                )
            else:
                await asyncio.sleep(0.2)

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

        elif method == 'validate_snippet':
            # Validate a code snippet by creating a temp file, opening it,
            # waiting for diagnostics, and cleaning up
            code = args.get('code', '')
            language = args.get('language', 'python')
            extension = args.get('extension', '.py')

            if not code:
                return {"error": "code parameter is required"}

            # Create temp file in a temp directory
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="lsp_validate_")
            temp_file = os.path.join(temp_dir, f"snippet{extension}")

            try:
                # Write code to temp file
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(code)

                self._trace(f"validate_snippet: created temp file {temp_file}")

                # Open document with LSP server
                await client.open_document(temp_file, code)

                # Wait for the server's first `publishDiagnostics` batch
                # via the same bounded-poll mechanism the
                # `get_diagnostics` dispatch path uses (PR-3, server
                # 0.6.134).  Pre-0.6.135 this branch had a hard-coded
                # `asyncio.sleep(0.5)` that PR-3 missed when porting
                # the dispatch path — fast servers wasted 500ms,
                # heavy-init servers (jdtls cold on Maven workspace
                # snippets) starved without ever waiting long enough.
                await client.await_diagnostics(
                    temp_file,
                    max_wait=self._diagnostics_max_wait_seconds,
                    min_wait=self._diagnostics_min_wait_seconds,
                    convergence_window=(
                        self._diagnostics_convergence_window_seconds
                    ),
                )

                # Get diagnostics
                diagnostics = client.get_diagnostics(temp_file)
                self._trace(f"validate_snippet: got {len(diagnostics)} diagnostics")

                # Close document
                await client.close_document(temp_file)

                return self._format_diagnostics(diagnostics)

            except Exception as e:
                self._trace(f"validate_snippet: error - {e}")
                return {"error": str(e)}

            finally:
                # Cleanup temp file
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except OSError:
                    pass

        elif method == '_ensure_workspace_indexed':
            # Internal method to index all files of a type in a directory
            directory = args.get('directory', '')
            extension = args.get('extension')
            if directory:
                # Pass extension as a list if provided
                extensions = [extension] if extension else None
                extra_paths_info = f" (extraPathsKey={client.config.extra_paths_key})" if client.config.extra_paths_key else ""
                self._trace(f"_ensure_workspace_indexed: indexing [{directory}] for {client.config.language_id}{extra_paths_info}")
                await client.ensure_workspace_indexed(directory, extensions)
                # Note: ensure_workspace_indexed now includes appropriate delays
                # for jedi to process file notifications and analyze documents
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

    def _exec_validate_snippet(self, args: Dict[str, Any]) -> Any:
        """Validate a code snippet by opening it as a temp file and getting diagnostics.

        This is designed for validating code blocks in model output before they're
        written to files.

        Args:
            args: Dict with:
                - code: The code snippet to validate
                - language: Language identifier (python, javascript, etc.)
                - extension: File extension (.py, .js, etc.)

        Returns:
            List of diagnostic dicts or {"error": ...}
        """
        return self._execute_method('validate_snippet', args)

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
                "message": f"Would rename '{symbol}' to '{new_name}' in {len(affected_files)} file(s). Set apply=true to apply.",
                "_telemetry": {
                    "jaato.lsp.operation": "rename_preview",
                    "jaato.lsp.files_affected": len(affected_files),
                },
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
                "errors": apply_result["errors"] if apply_result["errors"] else None,
                "_telemetry": {
                    "jaato.lsp.operation": "rename_applied",
                    "jaato.lsp.files_modified": len(apply_result.get("files_modified", [])),
                },
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
            "count": len(actions_list),
            "_telemetry": {
                "jaato.lsp.operation": "get_code_actions",
                "jaato.lsp.count": len(actions_list),
            },
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
            result['_telemetry'] = {
                'jaato.lsp.operation': 'apply_code_action',
                'jaato.lsp.success': apply_result.get('success', False),
            }
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
                "result": cmd_result,
                "_telemetry": {
                    "jaato.lsp.operation": "apply_code_action",
                },
            }

        return {"error": f"Code action '{action_title}' has no edit or command to apply"}


def create_plugin() -> LSPToolPlugin:
    """Factory function for plugin discovery."""
    return LSPToolPlugin()
