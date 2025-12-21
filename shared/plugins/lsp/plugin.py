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

from ..base import UserCommand, CommandParameter, CommandCompletion
from ..model_provider.types import ToolSchema
from ..subagent.config import expand_variables
from .lsp_client import LSPClient, ServerConfig, Location, Diagnostic, Hover


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
                - agent_name: Name for trace logging
        """
        if self._initialized:
            return

        # Expand variables in config values (e.g., ${projectPath}, ${workspaceRoot})
        config = expand_variables(config) if config else {}

        # Extract config values
        self._agent_name = config.get('agent_name')
        self._custom_config_path = config.get('config_path')

        self._trace("initialize: starting background thread")
        self._ensure_thread()
        self._initialized = True
        self._trace(f"initialize: connected_servers={list(self._connected_servers)}")

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
                description="Go to the definition of a symbol at a specific position in a file. "
                           "Returns the file path and line number where the symbol is defined.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the source file"
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number (0-indexed)"
                        },
                        "character": {
                            "type": "integer",
                            "description": "Character position in the line (0-indexed)"
                        }
                    },
                    "required": ["file_path", "line", "character"]
                }
            ),
            ToolSchema(
                name="lsp_find_references",
                description="Find all references to a symbol at a specific position. "
                           "Returns a list of locations where the symbol is used.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the source file"
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number (0-indexed)"
                        },
                        "character": {
                            "type": "integer",
                            "description": "Character position in the line (0-indexed)"
                        },
                        "include_declaration": {
                            "type": "boolean",
                            "description": "Include the declaration in results (default: true)"
                        }
                    },
                    "required": ["file_path", "line", "character"]
                }
            ),
            ToolSchema(
                name="lsp_hover",
                description="Get hover information (type info, documentation) for a symbol "
                           "at a specific position.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the source file"
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number (0-indexed)"
                        },
                        "character": {
                            "type": "integer",
                            "description": "Character position in the line (0-indexed)"
                        }
                    },
                    "required": ["file_path", "line", "character"]
                }
            ),
            ToolSchema(
                name="lsp_get_diagnostics",
                description="Get diagnostic information (errors, warnings) for a file. "
                           "The file must be opened first.",
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
                description="Rename a symbol across all files. Returns the workspace edit "
                           "that would be applied (does not apply automatically).",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the source file"
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number (0-indexed)"
                        },
                        "character": {
                            "type": "integer",
                            "description": "Character position in the line (0-indexed)"
                        },
                        "new_name": {
                            "type": "string",
                            "description": "New name for the symbol"
                        }
                    },
                    "required": ["file_path", "line", "character", "new_name"]
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
            "lsp": lambda args: self.execute_user_command('lsp', args),
        }

    def get_system_instructions(self) -> Optional[str]:
        return """LSP (Language Server Protocol) tools provide semantic code intelligence:
- lsp_goto_definition: Navigate to where a symbol is defined
- lsp_find_references: Find all usages of a symbol
- lsp_hover: Get type information and documentation
- lsp_get_diagnostics: Get errors and warnings for a file
- lsp_document_symbols: List all symbols in a file
- lsp_workspace_symbols: Search for symbols across the project
- lsp_rename_symbol: Get rename edits for a symbol

Use 'lsp status' to see connected language servers.
Line and character positions are 0-indexed."""

    def get_auto_approved_tools(self) -> List[str]:
        # All LSP tools are read-only except rename
        return [
            "lsp_goto_definition",
            "lsp_find_references",
            "lsp_hover",
            "lsp_get_diagnostics",
            "lsp_document_symbols",
            "lsp_workspace_symbols",
            "lsp",
        ]

    def get_user_commands(self) -> List[UserCommand]:
        return [
            UserCommand(
                name="lsp",
                description="Manage LSP language servers",
                share_with_model=True,
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

        self._load_config_cache(force=True)
        servers = self._config_cache.get('languageServers', {})

        try:
            self._request_queue.put((MSG_RELOAD_CONFIG, {'servers': servers}))
            status, result = self._response_queue.get(timeout=60)

            if status == 'ok':
                connected = result.get('connected', [])
                failed = result.get('failed', {})
                self._connected_servers = set(connected)
                self._failed_servers = failed
                return f"Reloaded: {len(connected)} connected, {len(failed)} failed"
            return f"Reload failed: {result}"
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
        2. .lsp.json in current working directory
        3. ~/.lsp.json in home directory
        """
        if self._config_cache and not force:
            return

        # Build search paths - custom path takes priority
        paths = []
        if self._custom_config_path:
            paths.append(self._custom_config_path)
        paths.extend([
            os.path.join(os.getcwd(), '.lsp.json'),
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
                            self._response_queue.put(('error', 'No LSP server available'))
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

    async def _call_lsp_method(self, client: LSPClient, method: str, args: Dict[str, Any]) -> Any:
        """Call an LSP method on the client."""
        file_path = args.get('file_path')

        # Ensure document is open if needed
        if file_path and method not in ('workspace_symbols',):
            await client.open_document(file_path)
            # Small delay for diagnostics to arrive
            await asyncio.sleep(0.2)

        if method == 'goto_definition':
            locations = await client.goto_definition(
                file_path, args['line'], args['character']
            )
            return self._format_locations(locations)

        elif method == 'find_references':
            locations = await client.find_references(
                file_path, args['line'], args['character'],
                args.get('include_declaration', True)
            )
            return self._format_locations(locations)

        elif method == 'hover':
            hover = await client.hover(file_path, args['line'], args['character'])
            if hover:
                return {"contents": hover.contents}
            return {"contents": "No hover information available"}

        elif method == 'get_diagnostics':
            diagnostics = client.get_diagnostics(file_path)
            return self._format_diagnostics(diagnostics)

        elif method == 'document_symbols':
            symbols = await client.get_document_symbols(file_path)
            return [
                {
                    "name": s.name,
                    "kind": s.kind_name,
                    "location": f"{self._uri_to_path(s.location.uri)}:{s.location.range.start.line + 1}"
                }
                for s in symbols
            ]

        elif method == 'workspace_symbols':
            symbols = await client.workspace_symbols(args['query'])
            return [
                {
                    "name": s.name,
                    "kind": s.kind_name,
                    "location": f"{self._uri_to_path(s.location.uri)}:{s.location.range.start.line + 1}"
                }
                for s in symbols
            ]

        elif method == 'rename_symbol':
            edits = await client.rename(
                file_path, args['line'], args['character'], args['new_name']
            )
            return edits

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

    # Tool executor methods

    def _execute_method(self, method: str, args: Dict[str, Any]) -> Any:
        """Execute an LSP method synchronously."""
        self._trace(f"execute: {method} args={args}")
        if not self._initialized:
            self.initialize()

        if not self._connected_servers:
            self._trace(f"execute: {method} FAILED - no servers connected")
            return {"error": "No LSP servers connected. Use 'lsp connect <server>' first."}

        try:
            self._request_queue.put((MSG_CALL_METHOD, {'method': method, 'args': args}))
            status, result = self._response_queue.get(timeout=30)

            if status == 'error':
                self._trace(f"execute: {method} ERROR - {result}")
                return {"error": result}
            self._trace(f"execute: {method} OK")
            return result
        except queue.Empty:
            self._trace(f"execute: {method} TIMEOUT")
            return {"error": "LSP request timed out"}
        except Exception as e:
            self._trace(f"execute: {method} EXCEPTION - {e}")
            return {"error": str(e)}

    def _exec_goto_definition(self, args: Dict[str, Any]) -> Any:
        return self._execute_method('goto_definition', args)

    def _exec_find_references(self, args: Dict[str, Any]) -> Any:
        return self._execute_method('find_references', args)

    def _exec_hover(self, args: Dict[str, Any]) -> Any:
        return self._execute_method('hover', args)

    def _exec_get_diagnostics(self, args: Dict[str, Any]) -> Any:
        return self._execute_method('get_diagnostics', args)

    def _exec_document_symbols(self, args: Dict[str, Any]) -> Any:
        return self._execute_method('document_symbols', args)

    def _exec_workspace_symbols(self, args: Dict[str, Any]) -> Any:
        return self._execute_method('workspace_symbols', args)

    def _exec_rename_symbol(self, args: Dict[str, Any]) -> Any:
        return self._execute_method('rename_symbol', args)


def create_plugin() -> LSPToolPlugin:
    """Factory function for plugin discovery."""
    return LSPToolPlugin()
