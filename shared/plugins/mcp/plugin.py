"""MCP tool plugin for executing Model Context Protocol tools."""

import asyncio
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Tuple

from ..base import UserCommand, CommandParameter, CommandCompletion
from ..model_provider.types import ToolSchema


# Message types for background thread communication
MSG_CALL_TOOL = 'call_tool'
MSG_LIST_SERVERS = 'list_servers'
MSG_SERVER_STATUS = 'server_status'
MSG_CONNECT_SERVER = 'connect_server'
MSG_DISCONNECT_SERVER = 'disconnect_server'
MSG_RELOAD_CONFIG = 'reload_config'

# Log entry levels
LOG_INFO = 'INFO'
LOG_DEBUG = 'DEBUG'
LOG_ERROR = 'ERROR'
LOG_WARN = 'WARN'

# Maximum log entries to keep
MAX_LOG_ENTRIES = 500


@dataclass
class LogEntry:
    """A single log entry for MCP interactions."""
    timestamp: datetime
    level: str
    server: Optional[str]
    event: str
    details: Optional[str] = None

    def format(self, include_timestamp: bool = True) -> str:
        """Format the log entry as a string."""
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


class MCPToolPlugin:
    """Plugin that provides MCP (Model Context Protocol) tool execution.

    This plugin connects to MCP servers defined in .mcp.json and exposes
    their tools to the AI model. It runs a background thread with an
    asyncio event loop to handle the async MCP protocol.
    """

    def __init__(self):
        # Instance state instead of module globals
        self._tool_cache: Dict[str, List[Any]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._manager: Any = None
        self._request_queue: Optional[queue.Queue] = None
        self._response_queue: Optional[queue.Queue] = None
        self._initialized = False
        self._mcp_patch_applied = False
        # Configuration state
        self._config_path: Optional[str] = None
        self._config_cache: Dict[str, Any] = {}
        self._connected_servers: set = set()
        self._failed_servers: Dict[str, str] = {}  # server -> error message
        # Interaction log
        self._log: deque = deque(maxlen=MAX_LOG_ENTRIES)
        self._log_lock = threading.Lock()

    def _log_event(
        self,
        level: str,
        event: str,
        server: Optional[str] = None,
        details: Optional[str] = None
    ) -> None:
        """Add an entry to the interaction log.

        Args:
            level: Log level (LOG_INFO, LOG_DEBUG, LOG_ERROR, LOG_WARN)
            event: Brief description of the event
            server: Optional server name this event relates to
            details: Optional additional details
        """
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
        return "mcp"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the MCP plugin by starting the background thread."""
        if self._initialized:
            return
        self._ensure_thread()
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the MCP plugin and clean up resources."""
        if self._request_queue:
            self._request_queue.put((None, None))  # Signal shutdown
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._tool_cache = {}
        self._loop = None
        self._thread = None
        self._manager = None
        self._request_queue = None
        self._response_queue = None
        self._initialized = False
        self._connected_servers = set()
        self._failed_servers = {}

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return ToolSchemas for all discovered MCP tools."""
        if not self._initialized:
            self.initialize()

        schemas = []
        for server_name, tools in self._tool_cache.items():
            for tool in tools:
                try:
                    cleaned_schema = self._clean_schema_for_vertex(tool.inputSchema)
                    schema = ToolSchema(
                        name=tool.name,
                        description=tool.description,
                        parameters=cleaned_schema
                    )
                    schemas.append(schema)
                except Exception as exc:
                    self._log_event(LOG_ERROR, f"Error creating schema for {tool.name}", server=server_name, details=str(exc))

        return schemas

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executor mappings for all discovered MCP tools and user commands."""
        if not self._initialized:
            self.initialize()

        executors = {}

        # Executors for MCP model tools
        for tools in self._tool_cache.values():
            for tool in tools:
                # Create a closure that captures the tool name
                def make_executor(toolname: str):
                    def executor(args: Dict[str, Any]) -> Dict[str, Any]:
                        return self._execute(toolname, args)
                    return executor
                executors[tool.name] = make_executor(tool.name)

        # Executor for 'mcp' user command
        def mcp_command_executor(args: Dict[str, Any]) -> str:
            return self.execute_user_command('mcp', args)
        executors['mcp'] = mcp_command_executor

        return executors

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions describing available MCP tools."""
        if not self._initialized:
            self.initialize()

        if not self._tool_cache:
            return None

        lines = ["You have access to the following MCP (Model Context Protocol) tools:"]

        for server_name, tools in self._tool_cache.items():
            lines.append(f"\nFrom '{server_name}' server:")
            for tool in tools:
                desc = tool.description or "No description"
                lines.append(f"  - {tool.name}: {desc}")

        return "\n".join(lines)

    def get_auto_approved_tools(self) -> List[str]:
        """MCP model tools require permission, but user commands are auto-approved."""
        # User commands are invoked directly by the user, so auto-approve them
        return [cmd.name for cmd in self.get_user_commands()]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for MCP server configuration."""
        return [
            UserCommand(
                name='mcp',
                description='Manage MCP server configuration (subcommands: list, show, add, remove, connect, disconnect, reload, status)',
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name='subcommand',
                        description='Subcommand: list, show, add, remove, connect, disconnect, reload, status',
                        required=True,
                    ),
                    CommandParameter(
                        name='rest',
                        description='Arguments for the subcommand',
                        required=False,
                        capture_rest=True,
                    ),
                ],
            ),
        ]

    def get_command_completions(
        self,
        command: str,
        args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for MCP command arguments.

        Args:
            command: The command name (should be 'mcp')
            args: Arguments typed so far

        Returns:
            List of CommandCompletion options.
        """
        if command != 'mcp':
            return []

        # Subcommand completions
        subcommands = [
            CommandCompletion('list', 'List all configured MCP servers'),
            CommandCompletion('status', 'Show connection status of all servers'),
            CommandCompletion('show', 'Show configuration for a specific server'),
            CommandCompletion('add', 'Add a new MCP server'),
            CommandCompletion('remove', 'Remove an MCP server'),
            CommandCompletion('connect', 'Connect to a configured server'),
            CommandCompletion('disconnect', 'Disconnect from a running server'),
            CommandCompletion('reload', 'Reload configuration from .mcp.json'),
            CommandCompletion('logs', 'Show interaction logs'),
            CommandCompletion('help', 'Show help for MCP commands'),
        ]

        if not args:
            # No args yet - show all subcommands
            return subcommands

        # First arg is partial subcommand
        if len(args) == 1:
            partial = args[0].lower()
            return [c for c in subcommands if c.value.startswith(partial)]

        # Second arg depends on subcommand
        subcommand = args[0].lower()
        if subcommand in ('show', 'connect', 'disconnect', 'remove'):
            # These take a server name as second arg
            self._load_config_cache()
            servers = self._config_cache.get('mcpServers', {})
            partial = args[1].lower() if len(args) > 1 else ''

            completions = []
            for name in servers:
                if name.lower().startswith(partial):
                    # Add status hint
                    if subcommand == 'connect':
                        if name in self._connected_servers:
                            continue  # Already connected
                        desc = 'Connect to this server'
                    elif subcommand == 'disconnect':
                        if name not in self._connected_servers:
                            continue  # Not connected
                        desc = 'Disconnect from this server'
                    elif subcommand == 'show':
                        desc = 'Show configuration'
                    elif subcommand == 'remove':
                        desc = 'Remove from configuration'
                    else:
                        desc = ''
                    completions.append(CommandCompletion(name, desc))
            return completions

        if subcommand == 'logs':
            # Logs subcommand completions: server names or 'clear'
            partial = args[1].lower() if len(args) > 1 else ''
            completions = [CommandCompletion('clear', 'Clear all logs')]
            self._load_config_cache()
            servers = self._config_cache.get('mcpServers', {})
            for name in servers:
                if name.lower().startswith(partial):
                    completions.append(CommandCompletion(name, f'Show logs for {name}'))
            return [c for c in completions if c.value.lower().startswith(partial)]

        return []

    def execute_user_command(
        self,
        command: str,
        args: Dict[str, Any]
    ) -> str:
        """Execute a user command and return the result as a string.

        Args:
            command: The command name (should be 'mcp')
            args: Parsed arguments containing 'subcommand' and optionally 'rest'

        Returns:
            Human-readable result string.
        """
        if command != 'mcp':
            return f"Unknown command: {command}"

        subcommand = args.get('subcommand', '').lower()
        rest = args.get('rest', '').strip()

        if subcommand == 'list':
            return self._cmd_list()
        elif subcommand == 'show':
            return self._cmd_show(rest)
        elif subcommand == 'add':
            return self._cmd_add(rest)
        elif subcommand == 'remove':
            return self._cmd_remove(rest)
        elif subcommand == 'connect':
            return self._cmd_connect(rest)
        elif subcommand == 'disconnect':
            return self._cmd_disconnect(rest)
        elif subcommand == 'reload':
            return self._cmd_reload()
        elif subcommand == 'status':
            return self._cmd_status()
        elif subcommand == 'logs':
            return self._cmd_logs(rest)
        elif subcommand == 'help' or subcommand == '':
            return self._cmd_help()
        else:
            return f"Unknown subcommand: {subcommand}\n\n{self._cmd_help()}"

    def _cmd_help(self) -> str:
        """Return help text for MCP commands."""
        return """MCP Server Configuration Commands:

  mcp list                  - List all configured MCP servers
  mcp status                - Show connection status of all servers
  mcp show <name>           - Show configuration for a specific server
  mcp add <name> <command> [args...] - Add a new MCP server
  mcp remove <name>         - Remove an MCP server from configuration
  mcp connect <name>        - Connect to a configured but disconnected server
  mcp disconnect <name>     - Disconnect from a running server
  mcp reload                - Reload configuration from .mcp.json
  mcp logs [server|clear]   - Show interaction logs (optionally filter by server)

Examples:
  mcp add github /usr/bin/mcp-server-github stdio
  mcp show github
  mcp disconnect github
  mcp connect github
  mcp logs                  - Show all logs
  mcp logs GitHub           - Show logs for GitHub server only
  mcp logs clear            - Clear all logs"""

    def _cmd_list(self) -> str:
        """List all configured MCP servers."""
        self._load_config_cache()

        servers = self._config_cache.get('mcpServers', {})
        if not servers:
            return "No MCP servers configured. Use 'mcp add <name> <command>' to add one."

        lines = ["Configured MCP servers:"]
        for name, spec in servers.items():
            status = "connected" if name in self._connected_servers else "disconnected"
            if name in self._failed_servers:
                status = f"failed: {self._failed_servers[name]}"
            cmd = spec.get('command', 'N/A')
            lines.append(f"  {name}: {cmd} [{status}]")

        return '\n'.join(lines)

    def _cmd_status(self) -> str:
        """Show detailed connection status of all servers."""
        self._load_config_cache()

        servers = self._config_cache.get('mcpServers', {})
        if not servers:
            return "No MCP servers configured."

        lines = ["MCP Server Status:"]
        lines.append("-" * 50)

        for name in servers:
            if name in self._connected_servers:
                tools = self._tool_cache.get(name, [])
                tool_count = len(tools)
                lines.append(f"  {name}: CONNECTED ({tool_count} tools)")
                if tools:
                    tool_names = [t.name for t in tools[:5]]
                    if len(tools) > 5:
                        tool_names.append(f"...and {len(tools) - 5} more")
                    lines.append(f"    Tools: {', '.join(tool_names)}")
            elif name in self._failed_servers:
                lines.append(f"  {name}: FAILED")
                lines.append(f"    Error: {self._failed_servers[name]}")
            else:
                lines.append(f"  {name}: DISCONNECTED")

        return '\n'.join(lines)

    def _cmd_show(self, server_name: str) -> str:
        """Show configuration for a specific server."""
        if not server_name:
            return "Usage: mcp show <server_name>"

        self._load_config_cache()
        servers = self._config_cache.get('mcpServers', {})

        if server_name not in servers:
            return f"Server '{server_name}' not found in configuration."

        spec = servers[server_name]
        lines = [f"Configuration for '{server_name}':"]
        lines.append(f"  Command: {spec.get('command', 'N/A')}")
        lines.append(f"  Args: {spec.get('args', [])}")

        env = spec.get('env', {})
        if env:
            lines.append("  Environment:")
            for k, v in env.items():
                # Mask sensitive values
                display_v = v if not v.startswith('${') else v
                lines.append(f"    {k}: {display_v}")

        # Connection status
        if server_name in self._connected_servers:
            lines.append(f"  Status: CONNECTED")
            tools = self._tool_cache.get(server_name, [])
            lines.append(f"  Tools available: {len(tools)}")
        elif server_name in self._failed_servers:
            lines.append(f"  Status: FAILED")
            lines.append(f"  Error: {self._failed_servers[server_name]}")
        else:
            lines.append(f"  Status: DISCONNECTED")

        return '\n'.join(lines)

    def _cmd_add(self, args_str: str) -> str:
        """Add a new MCP server to configuration."""
        parts = args_str.split()
        if len(parts) < 2:
            return "Usage: mcp add <name> <command> [args...]\nExample: mcp add github /usr/bin/mcp-server-github stdio"

        name = parts[0]
        command = parts[1]
        args = parts[2:] if len(parts) > 2 else []

        self._load_config_cache()
        servers = self._config_cache.get('mcpServers', {})

        if name in servers:
            return f"Server '{name}' already exists. Use 'mcp remove {name}' first to replace it."

        # Add the new server
        servers[name] = {
            'type': 'stdio',
            'command': command,
            'args': args,
        }

        self._config_cache['mcpServers'] = servers
        result = self._save_config()

        if result:
            return result  # Error message

        return f"Added MCP server '{name}' with command: {command} {' '.join(args)}\nUse 'mcp connect {name}' to connect, or 'mcp reload' to reconnect all servers."

    def _cmd_remove(self, server_name: str) -> str:
        """Remove an MCP server from configuration."""
        if not server_name:
            return "Usage: mcp remove <server_name>"

        self._load_config_cache()
        servers = self._config_cache.get('mcpServers', {})

        if server_name not in servers:
            return f"Server '{server_name}' not found in configuration."

        # If connected, disconnect first
        if server_name in self._connected_servers:
            self._cmd_disconnect(server_name)

        # Remove from config
        del servers[server_name]
        self._config_cache['mcpServers'] = servers
        result = self._save_config()

        if result:
            return result  # Error message

        # Clean up state
        self._failed_servers.pop(server_name, None)
        self._tool_cache.pop(server_name, None)

        return f"Removed MCP server '{server_name}' from configuration."

    def _cmd_connect(self, server_name: str) -> str:
        """Connect to a configured server."""
        if not server_name:
            return "Usage: mcp connect <server_name>"

        if not self._initialized:
            self.initialize()

        self._load_config_cache()
        servers = self._config_cache.get('mcpServers', {})

        if server_name not in servers:
            return f"Server '{server_name}' not found in configuration. Use 'mcp add' first."

        if server_name in self._connected_servers:
            return f"Server '{server_name}' is already connected."

        # Send connect request to background thread
        try:
            spec = servers[server_name]
            self._request_queue.put((MSG_CONNECT_SERVER, {
                'name': server_name,
                'command': spec.get('command'),
                'args': spec.get('args', []),
                'env': spec.get('env', {}),
            }))

            status, result = self._response_queue.get(timeout=30)

            if status == 'error':
                self._failed_servers[server_name] = result
                return f"Failed to connect to '{server_name}': {result}"

            # Update state
            self._connected_servers.add(server_name)
            self._failed_servers.pop(server_name, None)
            if 'tools' in result:
                self._tool_cache[server_name] = result['tools']

            tool_count = len(result.get('tools', []))
            return f"Connected to '{server_name}' successfully ({tool_count} tools available)."

        except queue.Empty:
            return f"Connection to '{server_name}' timed out."
        except Exception as exc:
            return f"Error connecting to '{server_name}': {exc}"

    def _cmd_disconnect(self, server_name: str) -> str:
        """Disconnect from a running server."""
        if not server_name:
            return "Usage: mcp disconnect <server_name>"

        if server_name not in self._connected_servers:
            return f"Server '{server_name}' is not connected."

        # Send disconnect request to background thread
        try:
            self._request_queue.put((MSG_DISCONNECT_SERVER, {'name': server_name}))

            status, result = self._response_queue.get(timeout=10)

            if status == 'error':
                return f"Failed to disconnect from '{server_name}': {result}"

            # Update state
            self._connected_servers.discard(server_name)
            self._tool_cache.pop(server_name, None)

            return f"Disconnected from '{server_name}'."

        except queue.Empty:
            return f"Disconnect from '{server_name}' timed out."
        except Exception as exc:
            return f"Error disconnecting from '{server_name}': {exc}"

    def _cmd_reload(self) -> str:
        """Reload configuration and reconnect all servers."""
        if not self._initialized:
            self.initialize()

        # Force reload from file
        self._config_cache = {}
        self._load_config_cache()

        # Send reload request to background thread
        try:
            servers = self._config_cache.get('mcpServers', {})
            self._request_queue.put((MSG_RELOAD_CONFIG, {'servers': servers}))

            status, result = self._response_queue.get(timeout=60)

            if status == 'error':
                return f"Reload failed: {result}"

            # Update state from result
            self._connected_servers = set(result.get('connected', []))
            self._failed_servers = result.get('failed', {})
            self._tool_cache = result.get('tools', {})

            connected_count = len(self._connected_servers)
            failed_count = len(self._failed_servers)
            total_tools = sum(len(t) for t in self._tool_cache.values())

            msg = f"Reloaded configuration: {connected_count} servers connected, {total_tools} tools available."
            if failed_count > 0:
                msg += f"\n{failed_count} servers failed to connect."
                for name, err in self._failed_servers.items():
                    msg += f"\n  {name}: {err}"

            return msg

        except queue.Empty:
            return "Configuration reload timed out."
        except Exception as exc:
            return f"Error reloading configuration: {exc}"

    def _cmd_logs(self, args_str: str) -> str:
        """Show interaction logs, optionally filtered by server.

        Args:
            args_str: Optional server name to filter by, or 'clear' to clear logs.
        """
        arg = args_str.strip().lower() if args_str else ''

        # Handle 'clear' command
        if arg == 'clear':
            with self._log_lock:
                count = len(self._log)
                self._log.clear()
            return f"Cleared {count} log entries."

        # Get logs, optionally filtered by server
        with self._log_lock:
            entries = list(self._log)

        if not entries:
            return "No log entries. Logs are recorded during server connections and tool calls."

        # Filter by server if specified
        filter_server = None
        if arg:
            # Check if it's a valid server name (case-insensitive match)
            self._load_config_cache()
            servers = self._config_cache.get('mcpServers', {})
            for name in servers:
                if name.lower() == arg:
                    filter_server = name
                    break
            if not filter_server:
                # Also check connected/failed servers
                for name in list(self._connected_servers) + list(self._failed_servers.keys()):
                    if name.lower() == arg:
                        filter_server = name
                        break
            if not filter_server:
                return f"Unknown server: {args_str.strip()}\nUse 'mcp logs' to see all logs or 'mcp list' to see available servers."

        # Filter and format
        if filter_server:
            entries = [e for e in entries if e.server and e.server.lower() == filter_server.lower()]
            if not entries:
                return f"No log entries for server '{filter_server}'."

        # Format output
        lines = [f"MCP Interaction Log ({len(entries)} entries)"]
        if filter_server:
            lines[0] += f" for '{filter_server}'"
        lines.append("-" * 60)

        for entry in entries:
            lines.append(entry.format())

        return '\n'.join(lines)

    def _load_config_cache(self) -> None:
        """Load configuration into cache if not already loaded."""
        if self._config_cache:
            return

        registry = self._load_mcp_registry()
        self._config_cache = registry

    def _save_config(self) -> Optional[str]:
        """Save current configuration to .mcp.json file.

        Returns:
            Error message string if save failed, None on success.
        """
        # Determine path to save to
        if self._config_path:
            path = self._config_path
        else:
            # Default to current directory
            path = os.path.join(os.getcwd(), '.mcp.json')

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._config_cache, f, indent=2)
            self._config_path = path
            return None
        except Exception as exc:
            return f"Failed to save configuration to {path}: {exc}"

    def _ensure_mcp_patch(self):
        """Lazily import mcp and apply the JSON-RPC validation patch."""
        if self._mcp_patch_applied:
            return

        from mcp import types as mcp_types

        # Store original
        _original_validate_json = mcp_types.JSONRPCMessage.model_validate_json.__func__

        # Capture self for logging in the closure
        log_event = self._log_event

        class SkipMessage(Exception):
            """Raised to signal a non-JSON-RPC message that should be skipped."""
            pass

        @classmethod
        def filtered_validate_json(cls, json_data, *args, **kwargs):
            """Wrapper that filters out non-JSON-RPC messages before validation."""
            if isinstance(json_data, bytes):
                json_data = json_data.decode('utf-8', errors='replace')

            line = json_data.strip()

            # Quick checks before expensive parsing
            if not line or not line.startswith('{'):
                log_event(LOG_DEBUG, "Filtered non-JSON message", details=line[:100])
                raise SkipMessage(line)

            # Validate it's actually JSON-RPC 2.0
            try:
                data = json.loads(line)
                if not isinstance(data, dict) or data.get('jsonrpc') != '2.0':
                    log_event(LOG_DEBUG, "Filtered non-JSONRPC message", details=line[:100])
                    raise SkipMessage(line)
            except json.JSONDecodeError:
                log_event(LOG_DEBUG, "Filtered invalid JSON", details=line[:100])
                raise SkipMessage(line)

            # It's valid JSON-RPC, let Pydantic parse it properly
            return _original_validate_json(cls, json_data, *args, **kwargs)

        # Apply patch
        mcp_types.JSONRPCMessage.model_validate_json = filtered_validate_json
        self._mcp_patch_applied = True

    def _load_mcp_registry(self, registry_path: Optional[str] = None) -> Dict[str, Any]:
        """Load MCP registry from specified path or default locations."""
        default_paths = [
            os.path.join(os.getcwd(), '.mcp.json'),
            os.path.expanduser('~/.mcp.json')
        ]
        if registry_path:
            paths = [registry_path] + default_paths
        else:
            paths = default_paths

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Track which path we loaded from
                    self._config_path = path
                    return data
                except Exception:
                    continue
        return {}

    def _clean_schema_for_vertex(self, schema: dict) -> dict:
        """Remove fields from JSON schema that Vertex AI doesn't support."""
        if not isinstance(schema, dict):
            return schema
        # Fields not supported by Vertex AI's Schema proto
        unsupported = {'$schema', '$id', '$ref', '$defs', 'definitions', 'additionalItems'}
        cleaned = {k: v for k, v in schema.items() if k not in unsupported}
        # Recursively clean nested schemas
        if 'properties' in cleaned and isinstance(cleaned['properties'], dict):
            cleaned['properties'] = {
                k: self._clean_schema_for_vertex(v) for k, v in cleaned['properties'].items()
            }
        if 'items' in cleaned:
            cleaned['items'] = self._clean_schema_for_vertex(cleaned['items'])
        return cleaned

    def _thread_main(self):
        """Background thread running the MCP event loop."""

        async def run_mcp_server():
            self._ensure_mcp_patch()
            from shared.mcp_context_manager import MCPClientManager

            self._log_event(LOG_INFO, "MCP plugin initializing")

            registry = self._load_mcp_registry()
            servers = registry.get('mcpServers', {})

            if servers:
                self._log_event(LOG_INFO, f"Found {len(servers)} server(s) in configuration",
                              details=', '.join(servers.keys()))
            else:
                self._log_event(LOG_WARN, "No MCP servers configured")

            # Expand env vars
            def expand_env(env_dict):
                result = {}
                for k, v in (env_dict or {}).items():
                    if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
                        result[k] = os.environ.get(v[2:-1], '')
                    else:
                        result[k] = v
                return result

            # Helper to connect a single server
            async def connect_server(mgr: MCPClientManager, name: str, spec: dict) -> Tuple[bool, str]:
                """Connect to a server and return (success, error_message)."""
                cmd = spec.get('command', 'unknown')
                args = spec.get('args', [])
                args_str = ' '.join(str(a) for a in args) if args else ''
                self._log_event(LOG_INFO, "Connecting to server", server=name,
                              details=f"command: {cmd} {args_str}".strip())
                try:
                    # Add timeout to prevent hanging on unresponsive servers
                    await asyncio.wait_for(
                        mgr.connect(
                            name,
                            cmd,
                            args,
                            expand_env(spec.get('env', {}))
                        ),
                        timeout=15.0  # 15 second timeout per server
                    )
                    self._connected_servers.add(name)
                    self._failed_servers.pop(name, None)

                    # Log tool discovery
                    conn = mgr.get_connection(name)
                    tool_count = len(conn.tools)
                    tool_names = [t.name for t in conn.tools[:10]]
                    if tool_count > 10:
                        tool_names.append(f"...and {tool_count - 10} more")
                    self._log_event(LOG_INFO, f"Connected successfully, discovered {tool_count} tool(s)",
                                  server=name, details=', '.join(tool_names) if tool_names else None)
                    return True, ''
                except asyncio.TimeoutError:
                    error_msg = "Connection timed out (15s)"
                    self._failed_servers[name] = error_msg
                    self._log_event(LOG_ERROR, "Connection timed out", server=name, details="Server did not respond within 15 seconds")
                    return False, error_msg
                except Exception as exc:
                    error_msg = str(exc)
                    self._failed_servers[name] = error_msg
                    self._log_event(LOG_ERROR, "Connection failed", server=name, details=error_msg)
                    return False, error_msg

            # Helper to update tool cache
            def update_tool_cache(mgr: MCPClientManager):
                self._tool_cache = {
                    name: list(conn.tools)
                    for name, conn in mgr._connections.items()
                }

            manager = MCPClientManager()
            async with manager:
                # Initial connection to all configured servers
                server_list = list(servers.items())
                for idx, (name, spec) in enumerate(server_list, 1):
                    self._log_event(LOG_DEBUG, f"Processing server {idx}/{len(server_list)}", server=name)
                    await connect_server(manager, name, spec)

                # Cache tools and log summary
                update_tool_cache(manager)
                self._manager = manager

                total_tools = sum(len(tools) for tools in self._tool_cache.values())
                self._log_event(LOG_INFO, f"Initialization complete: {len(self._connected_servers)} connected, "
                              f"{len(self._failed_servers)} failed, {total_tools} total tools")

                # Process requests from main thread
                while True:
                    try:
                        req = self._request_queue.get(timeout=0.1)
                        if req is None or req == (None, None):  # Shutdown signal
                            self._log_event(LOG_INFO, "Shutdown signal received")
                            break

                        msg_type, data = req

                        if msg_type == MSG_CALL_TOOL:
                            # Tool execution request
                            toolname = data.get('toolname')
                            args = data.get('args', {})
                            # Find which server provides this tool
                            server_name = None
                            for sname, tools in self._tool_cache.items():
                                if any(t.name == toolname for t in tools):
                                    server_name = sname
                                    break
                            self._log_event(LOG_DEBUG, f"Calling tool: {toolname}", server=server_name,
                                          details=f"args: {json.dumps(args, default=str)[:200]}")
                            try:
                                res = await manager.call_tool_auto(toolname, args)
                                is_error = getattr(res, 'isError', False)
                                if is_error:
                                    self._log_event(LOG_WARN, f"Tool returned error: {toolname}", server=server_name)
                                else:
                                    self._log_event(LOG_DEBUG, f"Tool completed: {toolname}", server=server_name)
                                self._response_queue.put(('ok', res))
                            except Exception as exc:
                                self._log_event(LOG_ERROR, f"Tool execution failed: {toolname}", server=server_name,
                                              details=str(exc))
                                self._response_queue.put(('error', str(exc)))

                        elif msg_type == MSG_CONNECT_SERVER:
                            # Connect to a specific server
                            name = data.get('name')
                            spec = {
                                'command': data.get('command'),
                                'args': data.get('args', []),
                                'env': data.get('env', {}),
                            }
                            success, error = await connect_server(manager, name, spec)
                            if success:
                                update_tool_cache(manager)
                                conn = manager.get_connection(name)
                                self._response_queue.put(('ok', {
                                    'tools': list(conn.tools),
                                }))
                            else:
                                self._response_queue.put(('error', error))

                        elif msg_type == MSG_DISCONNECT_SERVER:
                            # Disconnect from a specific server
                            name = data.get('name')
                            self._log_event(LOG_INFO, "Disconnecting from server", server=name)
                            try:
                                await manager.disconnect(name)
                                self._connected_servers.discard(name)
                                update_tool_cache(manager)
                                self._log_event(LOG_INFO, "Disconnected successfully", server=name)
                                self._response_queue.put(('ok', {}))
                            except Exception as exc:
                                self._log_event(LOG_ERROR, "Disconnect failed", server=name, details=str(exc))
                                self._response_queue.put(('error', str(exc)))

                        elif msg_type == MSG_RELOAD_CONFIG:
                            # Reload configuration - disconnect all, then reconnect
                            new_servers = data.get('servers', {})
                            connected = []
                            failed = {}

                            self._log_event(LOG_INFO, f"Reloading configuration with {len(new_servers)} server(s)")

                            # Disconnect all current servers
                            current_servers = list(manager.servers)
                            if current_servers:
                                self._log_event(LOG_INFO, f"Disconnecting {len(current_servers)} current server(s)")
                            for name in current_servers:
                                try:
                                    self._log_event(LOG_DEBUG, "Disconnecting for reload", server=name)
                                    await manager.disconnect(name)
                                except Exception as exc:
                                    self._log_event(LOG_WARN, "Disconnect during reload failed", server=name,
                                                  details=str(exc))
                            self._connected_servers.clear()
                            self._failed_servers.clear()

                            # Connect to all servers in new config
                            for name, spec in new_servers.items():
                                success, error = await connect_server(manager, name, spec)
                                if success:
                                    connected.append(name)
                                else:
                                    failed[name] = error

                            update_tool_cache(manager)
                            total_tools = sum(len(tools) for tools in self._tool_cache.values())
                            self._log_event(LOG_INFO, f"Reload complete: {len(connected)} connected, "
                                          f"{len(failed)} failed, {total_tools} total tools")

                            self._response_queue.put(('ok', {
                                'connected': connected,
                                'failed': failed,
                                'tools': self._tool_cache,
                            }))

                        elif msg_type == MSG_LIST_SERVERS:
                            # Return list of servers and their status
                            self._response_queue.put(('ok', {
                                'connected': list(self._connected_servers),
                                'failed': dict(self._failed_servers),
                            }))

                        elif msg_type == MSG_SERVER_STATUS:
                            # Return detailed status for a specific server
                            name = data.get('name')
                            if name in self._connected_servers:
                                conn = manager.get_connection(name)
                                self._response_queue.put(('ok', {
                                    'status': 'connected',
                                    'tools': [t.name for t in conn.tools],
                                }))
                            elif name in self._failed_servers:
                                self._response_queue.put(('ok', {
                                    'status': 'failed',
                                    'error': self._failed_servers[name],
                                }))
                            else:
                                self._response_queue.put(('ok', {
                                    'status': 'disconnected',
                                }))

                        else:
                            self._response_queue.put(('error', f'Unknown message type: {msg_type}'))

                    except queue.Empty:
                        await asyncio.sleep(0.01)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(run_mcp_server())
        except Exception as exc:
            self._log_event(LOG_ERROR, "MCP thread crashed", details=str(exc))
        finally:
            self._loop.close()

    def _ensure_thread(self):
        """Start the MCP background thread if not already running."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

        # Wait for tools to be discovered
        for _ in range(100):  # 10 second timeout
            if self._tool_cache:
                break
            time.sleep(0.1)

    def _execute(self, toolname: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool via the background connection.

        Args:
            toolname: Name of the MCP tool to invoke.
            args: Dict containing tool parameters.

        Returns:
            Dict with 'result' containing tool invocation details or 'error'.
        """
        # Ensure MCP thread is running
        if not self._initialized:
            self.initialize()

        if not self._tool_cache:
            return {'error': 'MCP tools not available'}

        # Verify tool exists
        found = False
        for tools in self._tool_cache.values():
            for t in tools:
                if t.name == toolname:
                    found = True
                    break
            if found:
                break

        if not found:
            return {'error': f"Tool '{toolname}' not found on any MCP server"}

        try:
            # Send request to MCP thread using new message format
            self._request_queue.put((MSG_CALL_TOOL, {'toolname': toolname, 'args': args}))

            # Wait for response (30 second timeout)
            status, result = self._response_queue.get(timeout=30)

            if status == 'error':
                return {'error': result}

            out = {
                'tool': toolname,
                'isError': getattr(result, 'isError', False),
                'structured': getattr(result, 'structuredContent', None),
                'content': [getattr(c, 'text', None) for c in getattr(result, 'content', [])],
            }
            return {'result': out}

        except queue.Empty:
            return {'error': 'MCP tool call timed out'}
        except Exception as exc:
            return {'error': f"{type(exc).__name__}: {exc}"}


def create_plugin() -> MCPToolPlugin:
    """Factory function to create the MCP plugin instance."""
    return MCPToolPlugin()
