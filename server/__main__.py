#!/usr/bin/env python3
"""Jaato Server - Multi-client AI assistant backend.

This is the main entry point for the Jaato server, which provides:
- IPC socket for local clients (rich-client, IDE extensions)
- WebSocket for remote/web clients
- Multi-session management
- Daemon mode for background operation

Usage:
    # Start with IPC socket only (local)
    python -m server --ipc-socket /tmp/jaato.sock

    # Start with WebSocket only (remote)
    python -m server --web-socket :8080

    # Start with both
    python -m server --ipc-socket /tmp/jaato.sock --web-socket :8080

    # Daemon mode (background)
    python -m server --ipc-socket /tmp/jaato.sock --daemon

    # Check if running
    python -m server --status

    # Stop daemon
    python -m server --stop

    # Restart with same parameters (useful during development)
    python -m server --restart
"""

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.session_manager import SessionManager
from server.session_logging import (
    configure_session_logging,
    set_logging_context,
    clear_logging_context,
)
from jaato_sdk.events import Event


# Default paths (use platform-appropriate temp directory)
_TEMP_DIR = Path(tempfile.gettempdir())
# IPC path is platform-specific: Unix socket on Unix, named pipe on Windows
if sys.platform == "win32":
    DEFAULT_SOCKET_PATH = "jaato"  # Will become \\.\pipe\jaato
else:
    DEFAULT_SOCKET_PATH = str(_TEMP_DIR / "jaato.sock")
DEFAULT_PID_FILE = str(_TEMP_DIR / "jaato.pid")
DEFAULT_LOG_FILE = str(_TEMP_DIR / "jaato.log")
DEFAULT_CONFIG_FILE = str(_TEMP_DIR / "jaato.config.json")

# Log rotation settings
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
LOG_BACKUP_COUNT = 5  # Keep 5 backup files


def configure_logging(
    log_file: Optional[str] = None,
    verbose: bool = False,
    enable_session_logging: bool = True,
) -> None:
    """Configure logging with optional file rotation and per-session routing.

    Args:
        log_file: Path to log file. If provided, uses RotatingFileHandler.
        verbose: If True, use DEBUG level; otherwise INFO.
        enable_session_logging: If True, also route logs to per-session files.
            Session logs go to {workspace}/JAATO_SESSION_LOG_DIR/ based on
            the JAATO_SESSION_LOG_DIR env var in each workspace's .env file.
    """
    central_level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Remove any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Root logger passes everything; each handler filters its own level.
    # This allows session logs to capture DEBUG while the central log
    # stays at INFO (unless --verbose).
    root.setLevel(logging.DEBUG)

    if log_file:
        # Use rotating file handler to prevent unbounded log growth
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8',
        )
        handler.setLevel(central_level)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    else:
        # Console logging
        handler = logging.StreamHandler()
        handler.setLevel(central_level)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)

    # Add session routing handler for per-session/client log files
    # Always at DEBUG level for full session visibility.
    if enable_session_logging:
        configure_session_logging(
            level=logging.DEBUG,
            formatter=logging.Formatter(fmt),
        )


logger = logging.getLogger(__name__)


class JaatoDaemon:
    """Main server daemon managing IPC and WebSocket servers."""

    def __init__(
        self,
        ipc_socket: Optional[str] = None,
        web_socket: Optional[str] = None,
        pid_file: str = DEFAULT_PID_FILE,
        config_file: str = DEFAULT_CONFIG_FILE,
        log_file: str = DEFAULT_LOG_FILE,
    ):
        """Initialize the daemon.

        Args:
            ipc_socket: Path to Unix domain socket (None to disable).
            web_socket: WebSocket address as "host:port" or ":port" (None to disable).
            pid_file: Path to PID file for daemon mode.
            config_file: Path to config file for restart support.
            log_file: Path to log file for daemon mode.
        """
        self.ipc_socket = ipc_socket
        self.web_socket = web_socket
        self.pid_file = pid_file
        self.config_file = config_file
        self.log_file = log_file

        # Components
        self._session_manager: Optional[SessionManager] = None
        self._ipc_server = None
        self._ws_server = None

        # Session-independent plugins (auth plugins loaded at daemon startup)
        # These provide user commands that work without an active session/provider.
        self._daemon_plugins: dict = {}  # name -> plugin instance

        # Pending post-auth setup requests: client_id -> {request_id, provider_name}
        self._pending_post_auth: dict = {}

        # Shutdown flag
        self._shutdown_event = asyncio.Event()

        # Pending workspace mismatch requests: client_id -> {request_id, session_id, ...}
        self._pending_workspace_mismatch: dict = {}

    async def start(self) -> None:
        """Start the daemon and run until shutdown."""
        # Note: We don't load_dotenv here - the daemon is provider-agnostic.
        # Each session gets its config from the client's workspace .env file.
        # The server's env_file is only used as a fallback for sessions
        # without their own env file.

        # Initialize session manager
        # Note: Sessions are workspace-bound. Each session gets its env_file and
        # provider from the client's workspace, not from server-level config.
        self._session_manager = SessionManager()

        # Set up event routing
        self._session_manager.set_event_callback(self._route_event)

        # Discover session-independent plugins (auth plugins).
        # These are loaded at daemon startup so their commands are available
        # before any session/provider connection exists.
        self._discover_daemon_plugins()

        tasks = []

        # Start IPC server if configured
        # Uses Unix domain sockets on Unix/Linux/macOS, named pipes on Windows
        if self.ipc_socket:
            from server.ipc import JaatoIPCServer, _get_display_path

            self._ipc_server = JaatoIPCServer(
                socket_path=self.ipc_socket,
                on_session_request=self._handle_session_request,
                on_command_list_request=self._get_command_list,
            )
            tasks.append(asyncio.create_task(self._ipc_server.start()))
            display_path = _get_display_path(self.ipc_socket)
            logger.info(f"IPC server will listen on {display_path}")

        # Start WebSocket server if configured
        if self.web_socket:
            from server.websocket import JaatoWSServer

            # Parse host:port
            if ':' in self.web_socket:
                if self.web_socket.startswith(':'):
                    host = "0.0.0.0"
                    port = int(self.web_socket[1:])
                else:
                    host, port_str = self.web_socket.rsplit(':', 1)
                    port = int(port_str)
            else:
                host = "0.0.0.0"
                port = int(self.web_socket)

            self._ws_server = JaatoWSServer(
                host=host,
                port=port,
            )
            tasks.append(asyncio.create_task(self._ws_server.start()))
            logger.info(f"WebSocket server will listen on ws://{host}:{port}")

        if not tasks:
            logger.error("No servers configured. Use --ipc-socket and/or --web-socket")
            return

        # Write PID and config files
        self._write_pid()
        self._write_config()

        # Set up signal handlers (not supported on Windows)
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        logger.info("Jaato server started")

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Cancel all tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cleanup
        if self._session_manager:
            self._session_manager.shutdown()

        self._remove_pid()
        # Note: Don't remove config on normal stop - needed for restart
        logger.info("Jaato server stopped")

    async def stop(self) -> None:
        """Signal shutdown."""
        logger.info("Shutdown requested...")
        self._shutdown_event.set()

        if self._ipc_server:
            await self._ipc_server.stop()
        if self._ws_server:
            await self._ws_server.stop()

    def _write_pid(self) -> None:
        """Write PID file."""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            logger.warning(f"Could not write PID file: {e}")

    def _remove_pid(self) -> None:
        """Remove PID file."""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
        except Exception as e:
            logger.warning(f"Could not remove PID file: {e}")

    def _write_config(self) -> None:
        """Write startup config for restart support."""
        config = {
            "ipc_socket": self.ipc_socket,
            "web_socket": self.web_socket,
            "pid_file": self.pid_file,
            "log_file": self.log_file,
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            logger.warning(f"Could not write config file: {e}")

    def _remove_config(self) -> None:
        """Remove config file."""
        try:
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
        except Exception as e:
            logger.warning(f"Could not remove config file: {e}")

    def _handle_session_request(
        self,
        client_id: str,
        session_id: str,
        event: Event,
    ) -> None:
        """Handle a request from an IPC client."""
        if not self._session_manager:
            return

        # Set logging context from existing session (if any) so all logger
        # calls in this thread are routed to per-session log files.
        existing_session = self._session_manager.get_client_session(client_id)
        workspace_from_ipc = (
            self._ipc_server.get_client_workspace(client_id)
            if self._ipc_server else None
        )
        if existing_session and existing_session.server:
            set_logging_context(
                session_id=existing_session.session_id,
                client_id=client_id,
                workspace_path=existing_session.workspace_path or workspace_from_ipc,
                session_env=existing_session.server.get_all_session_env(),
            )
        elif session_id and workspace_from_ipc:
            # No loaded session yet but we have identifiers (e.g. attach path)
            set_logging_context(
                session_id=session_id,
                client_id=client_id,
                workspace_path=workspace_from_ipc,
            )

        try:
            self._handle_session_request_inner(client_id, session_id, event)
        finally:
            clear_logging_context()

    def _handle_session_request_inner(
        self,
        client_id: str,
        session_id: str,
        event: Event,
    ) -> None:
        """Inner handler with logging context already set."""
        # Handle tool disable request (direct registry call, no response events)
        from jaato_sdk.events import ToolDisableRequest
        if isinstance(event, ToolDisableRequest):
            session = self._session_manager.get_client_session(client_id)
            if session and session.server and session.server.registry:
                session.server.registry.disable_tool(event.tool_name)
            return

        # Handle session management commands
        from jaato_sdk.events import CommandRequest

        if isinstance(event, CommandRequest):
            cmd = event.command.lower()

            # Handle set_workspace command (sent by client on connect)
            if cmd == "set_workspace":
                workspace_path = event.args[0] if event.args else None
                if workspace_path and self._ipc_server:
                    self._ipc_server.set_client_workspace(client_id, workspace_path)
                    logger.debug(f"Client {client_id} workspace set to: {workspace_path}")
                return

            # Get client's workspace path for session operations
            workspace_path = None
            if self._ipc_server:
                workspace_path = self._ipc_server.get_client_workspace(client_id)

            if cmd == "session.new":
                name = event.args[0] if event.args else None
                new_session_id = self._session_manager.create_session(
                    client_id, name, workspace_path=workspace_path
                )
                if new_session_id:
                    # Update logging context now that session_id is known.
                    # create_session() loaded the .env, so fetch session_env.
                    new_session = self._session_manager.get_client_session(client_id)
                    session_env = (
                        new_session.server.get_all_session_env()
                        if new_session and new_session.server else {}
                    )
                    set_logging_context(
                        session_id=new_session_id,
                        client_id=client_id,
                        workspace_path=workspace_path,
                        session_env=session_env,
                    )
                    logger.info(f"Session {new_session_id} created and context set")
                    if self._ipc_server:
                        self._ipc_server.set_client_session(client_id, new_session_id)
                return

            elif cmd == "session.attach":
                if event.args:
                    target_session_id = event.args[0]
                    # Check for workspace mismatch
                    mismatch = self._session_manager.check_workspace_mismatch(
                        target_session_id, workspace_path
                    )
                    if mismatch:
                        session_workspace, client_workspace = mismatch
                        # Emit mismatch event and wait for user response
                        import uuid
                        request_id = str(uuid.uuid4())
                        self._pending_workspace_mismatch[client_id] = {
                            "request_id": request_id,
                            "session_id": target_session_id,
                            "session_workspace": session_workspace,
                            "client_workspace": client_workspace,
                        }
                        from jaato_sdk.events import WorkspaceMismatchRequestedEvent
                        self._route_event(client_id, WorkspaceMismatchRequestedEvent(
                            request_id=request_id,
                            session_id=target_session_id,
                            session_workspace=session_workspace,
                            client_workspace=client_workspace,
                            response_options=[
                                {"key": "s", "label": "switch", "action": "switch",
                                 "description": f"Switch to session workspace: {session_workspace}"},
                                {"key": "c", "label": "cancel", "action": "cancel",
                                 "description": "Stay in current session"},
                            ],
                            prompt_lines=[
                                f"Workspace mismatch detected:",
                                f"  Session workspace: {session_workspace}",
                                f"  Your workspace:    {client_workspace}",
                                f"",
                                f"Choose an option:",
                                f"  [s] Switch to session's workspace",
                                f"  [c] Cancel and stay in current session",
                            ],
                        ))
                        return
                    # No mismatch, proceed with attach
                    # Set context before attach so initialization logs are routed
                    set_logging_context(
                        session_id=target_session_id,
                        client_id=client_id,
                        workspace_path=workspace_path,
                    )
                    if self._session_manager.attach_session(
                        client_id, target_session_id, workspace_path=workspace_path
                    ):
                        # Update context with session_env now that server is loaded
                        attached = self._session_manager.get_client_session(client_id)
                        if attached and attached.server:
                            set_logging_context(
                                session_env=attached.server.get_all_session_env(),
                            )
                        if self._ipc_server:
                            self._ipc_server.set_client_session(client_id, target_session_id)
                return

            elif cmd == "session.list":
                sessions = self._session_manager.list_sessions()
                from jaato_sdk.events import SessionListEvent

                # Get client's current session to mark it in the list
                current_session_id = session_id  # From the event

                # Send structured session data - client handles formatting
                session_data = [{
                    "id": s.session_id,
                    "name": s.name or "",
                    "description": s.description or "",
                    "model_provider": s.model_provider or "",
                    "model_name": s.model_name or "",
                    "is_loaded": s.is_loaded,
                    "is_current": s.session_id == current_session_id,
                    "client_count": s.client_count,
                    "turn_count": s.turn_count,
                    "workspace_path": s.workspace_path or "",
                } for s in sessions]

                self._route_event(client_id, SessionListEvent(sessions=session_data))
                return

            elif cmd == "session.default":
                default_session_id = self._session_manager.get_or_create_default(
                    client_id, workspace_path=workspace_path
                )
                if default_session_id:
                    # Update context now that session exists
                    default_session = self._session_manager.get_client_session(client_id)
                    if default_session and default_session.server:
                        set_logging_context(
                            session_id=default_session_id,
                            client_id=client_id,
                            workspace_path=workspace_path,
                            session_env=default_session.server.get_all_session_env(),
                        )
                    if self._ipc_server:
                        self._ipc_server.set_client_session(client_id, default_session_id)
                else:
                    # Session creation failed (e.g., missing MODEL_NAME).
                    # List available auth providers so the user knows how to
                    # configure a provider for this workspace.
                    self._hint_available_auth_providers(client_id)
                return

            elif cmd == "session.end":
                # End session - stop agent and signal all attached clients to exit
                session = self._session_manager.get_client_session(client_id)
                if session and session.server:
                    session.server.stop()
                # Signal termination to all clients attached to this session
                from jaato_sdk.events import SystemMessageEvent
                self._session_manager._emit_to_session(
                    session_id,
                    SystemMessageEvent(message="[SESSION_TERMINATED]", style="system")
                )
                return

            elif cmd == "session.delete":
                if event.args:
                    session_id_to_delete = event.args[0]
                    if self._session_manager.delete_session(session_id_to_delete):
                        from jaato_sdk.events import SystemMessageEvent
                        self._route_event(client_id, SystemMessageEvent(
                            message=f"Session '{session_id_to_delete}' deleted.",
                            style="info",
                        ))
                    else:
                        from jaato_sdk.events import SystemMessageEvent
                        self._route_event(client_id, SystemMessageEvent(
                            message=f"Session '{session_id_to_delete}' not found.",
                            style="warning",
                        ))
                return

            elif cmd == "session.help":
                from jaato_sdk.events import HelpTextEvent
                help_lines = [
                    ("Session Command", "bold"),
                    ("", ""),
                    ("Manage multiple conversation sessions. Each session has its own", ""),
                    ("conversation history, model state, and workspace.", ""),
                    ("", ""),
                    ("USAGE", "bold"),
                    ("    session [subcommand] [args]", ""),
                    ("", ""),
                    ("SUBCOMMANDS", "bold"),
                    ("    list              List all available sessions", "dim"),
                    ("                      Shows ID, description, model, and status", "dim"),
                    ("", ""),
                    ("    new [name]        Create a new session", "dim"),
                    ("                      Optional name for easier identification", "dim"),
                    ("", ""),
                    ("    attach <id>       Attach to an existing session", "dim"),
                    ("                      Loads session from disk if not in memory", "dim"),
                    ("", ""),
                    ("    delete <id>       Delete a session permanently", "dim"),
                    ("                      Removes both memory and disk state", "dim"),
                    ("", ""),
                    ("    help              Show this help message", "dim"),
                    ("", ""),
                    ("EXAMPLES", "bold"),
                    ("    session list               List all sessions", "dim"),
                    ("    session new                Create unnamed session", "dim"),
                    ("    session new myproject      Create session named 'myproject'", "dim"),
                    ("    session attach 20251207    Attach to session by ID", "dim"),
                    ("    session delete 20251207    Delete session by ID", "dim"),
                    ("", ""),
                    ("SESSION STATES", "bold"),
                    ("    Sessions can be in different states:", ""),
                    ("    - Loaded: Currently in memory, ready for use", "dim"),
                    ("    - On disk: Saved to disk, will be loaded on attach", "dim"),
                    ("    - Processing: Currently running a model turn", "dim"),
                    ("", ""),
                    ("PERSISTENCE", "bold"),
                    ("    Sessions are automatically saved to:", ""),
                    ("        .jaato/sessions/<session_id>.json", "dim"),
                    ("", ""),
                    ("    Each session stores:", ""),
                    ("    - Conversation history", "dim"),
                    ("    - Model and provider settings", "dim"),
                    ("    - Workspace path", "dim"),
                    ("    - Session description (auto-generated)", "dim"),
                    ("", ""),
                    ("RELATED COMMANDS", "bold"),
                    ("    save              Manually save current session", "dim"),
                    ("    resume <id>       Resume a saved session (alias for attach)", "dim"),
                    ("    reset             Clear current session history", "dim"),
                ]
                self._route_event(client_id, HelpTextEvent(lines=help_lines))
                return

            # Tools commands - handled per-session
            elif cmd.startswith("tools."):
                # Get the client's session
                session = self._session_manager.get_client_session(client_id)
                if not session or not session.server:
                    from jaato_sdk.events import SystemMessageEvent
                    self._route_event(client_id, SystemMessageEvent(
                        message="No active session. Use 'session attach' first.",
                        style="warning",
                    ))
                    return

                tools_subcmd = cmd.split(".", 1)[1] if "." in cmd else "list"
                from jaato_sdk.events import ToolStatusEvent

                if tools_subcmd == "list":
                    # Get tool status from session's server - send structured data
                    tools = self._get_tool_status(session.server)
                    self._route_event(client_id, ToolStatusEvent(tools=tools))
                elif tools_subcmd == "enable" and event.args:
                    result = self._tools_enable(session.server, event.args[0])
                    tools = self._get_tool_status(session.server)
                    self._route_event(client_id, ToolStatusEvent(tools=tools, message=result))
                elif tools_subcmd == "disable" and event.args:
                    result = self._tools_disable(session.server, event.args[0])
                    tools = self._get_tool_status(session.server)
                    self._route_event(client_id, ToolStatusEvent(tools=tools, message=result))
                elif tools_subcmd == "help":
                    from jaato_sdk.events import HelpTextEvent
                    help_lines = [
                        ("Tools Command", "bold"),
                        ("", ""),
                        ("Manage tools available to the model. Tools can be enabled or disabled", ""),
                        ("to control what capabilities the model has access to.", ""),
                        ("", ""),
                        ("USAGE", "bold"),
                        ("    tools [subcommand] [args]", ""),
                        ("", ""),
                        ("SUBCOMMANDS", "bold"),
                        ("    list              List all tools with their enabled/disabled status", "dim"),
                        ("                      (this is the default when no subcommand is given)", "dim"),
                        ("", ""),
                        ("    enable <name>     Enable a specific tool by name", "dim"),
                        ("    enable all        Enable all tools at once", "dim"),
                        ("", ""),
                        ("    disable <name>    Disable a specific tool by name", "dim"),
                        ("    disable all       Disable all tools at once", "dim"),
                        ("", ""),
                        ("    help              Show this help message", "dim"),
                        ("", ""),
                        ("EXAMPLES", "bold"),
                        ("    tools                    Show all tools and their status", "dim"),
                        ("    tools list               Same as above", "dim"),
                        ("    tools enable Bash        Enable the Bash tool", "dim"),
                        ("    tools disable web_search Disable web search", "dim"),
                        ("    tools enable all         Enable all tools", "dim"),
                        ("", ""),
                        ("NOTES", "bold"),
                        ("    - Tool names are case-sensitive", "dim"),
                        ("    - Disabled tools will not be available for the model to use", "dim"),
                        ("    - Use 'tools list' to see available tool names", "dim"),
                    ]
                    self._route_event(client_id, HelpTextEvent(lines=help_lines))
                else:
                    from jaato_sdk.events import SystemMessageEvent
                    self._route_event(client_id, SystemMessageEvent(
                        message="Usage: tools list | tools enable <name> | tools disable <name> | tools help",
                        style="dim",
                    ))
                return

        # Handle WorkspaceMismatchResponseRequest
        from jaato_sdk.events import WorkspaceMismatchResponseRequest, WorkspaceMismatchResolvedEvent
        if isinstance(event, WorkspaceMismatchResponseRequest):
            pending = self._pending_workspace_mismatch.pop(client_id, None)
            if not pending or pending["request_id"] != event.request_id:
                logger.warning(f"No pending workspace mismatch request for client {client_id}")
                return

            response = event.response.lower()
            target_session_id = pending["session_id"]
            session_workspace = pending["session_workspace"]
            client_workspace = pending["client_workspace"]

            if response in ("s", "switch"):
                # User chose to switch to session's workspace
                # Attach without passing workspace_path (use session's workspace)
                if self._session_manager.attach_session(client_id, target_session_id):
                    if self._ipc_server:
                        self._ipc_server.set_client_session(client_id, target_session_id)
                    self._route_event(client_id, WorkspaceMismatchResolvedEvent(
                        request_id=event.request_id,
                        session_id=target_session_id,
                        action="switch",
                    ))
                    from jaato_sdk.events import SystemMessageEvent
                    self._route_event(client_id, SystemMessageEvent(
                        message=f"Attached to session. Working directory: {session_workspace}",
                        style="info",
                    ))

            else:
                # Cancel or unknown response
                self._route_event(client_id, WorkspaceMismatchResolvedEvent(
                    request_id=event.request_id,
                    session_id=target_session_id,
                    action="cancel",
                ))
                from jaato_sdk.events import SystemMessageEvent
                self._route_event(client_id, SystemMessageEvent(
                    message="Attach cancelled.",
                    style="dim",
                ))
            return

        # Handle HistoryRequest
        from jaato_sdk.events import HistoryRequest, HistoryEvent
        if isinstance(event, HistoryRequest):
            session = self._session_manager.get_client_session(client_id)
            if session and session.server:
                # Get history from session
                history = session.server.get_history(event.agent_id)
                turn_accounting = session.server.get_turn_accounting(event.agent_id)

                # Serialize history to dicts
                history_data = []
                for msg in history:
                    history_data.append({
                        "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                        "parts": [self._serialize_part(p) for p in (msg.parts or [])],
                    })

                self._route_event(client_id, HistoryEvent(
                    agent_id=event.agent_id,
                    history=history_data,
                    turn_accounting=turn_accounting or [],
                ))
            return

        # Handle daemon-level plugin commands (session-independent plugins).
        # These are always routed through the daemon path regardless of session
        # state, because they need daemon-level features (e.g., post-auth wizard).
        if isinstance(event, CommandRequest):
            plugin = self._find_daemon_plugin_for_command(event.command)
            if plugin:
                self._execute_daemon_command(client_id, plugin, event.command, event.args)
                return

        # Handle post-auth setup response
        from jaato_sdk.events import PostAuthSetupResponse
        if isinstance(event, PostAuthSetupResponse):
            self._handle_post_auth_response(client_id, event)
            return

        # Route to session
        self._session_manager.handle_request(client_id, session_id, event)

    def _route_event(self, client_id: str, event: Event) -> None:
        """Route an event to a client."""
        if self._ipc_server:
            self._ipc_server.queue_event(client_id, event)
        # WebSocket routing would be added here

    def _discover_daemon_plugins(self) -> None:
        """Discover session-independent plugins at daemon startup.

        Scans the plugins directory for modules with SESSION_INDEPENDENT = True.
        These plugins (typically auth plugins) provide user commands that work
        without an active session or provider connection.
        """
        import importlib
        import pkgutil
        from pathlib import Path as _Path

        plugins_dir = _Path(__file__).resolve().parents[1] / "shared" / "plugins"

        for finder, name, ispkg in pkgutil.iter_modules([str(plugins_dir)]):
            if name.startswith('_') or name in ('base', 'registry'):
                continue
            try:
                module = importlib.import_module(f"shared.plugins.{name}")
                if not getattr(module, 'SESSION_INDEPENDENT', False):
                    continue
                if not hasattr(module, 'create_plugin'):
                    continue

                plugin = module.create_plugin()
                self._daemon_plugins[plugin.name] = plugin
                logger.debug(f"Loaded daemon-level plugin: {plugin.name}")

            except Exception as exc:
                logger.warning(f"Failed to load daemon plugin '{name}': {exc}")

    def _find_daemon_plugin_for_command(self, command: str):
        """Find a daemon-level plugin that provides a user command.

        Args:
            command: The command name to find.

        Returns:
            The plugin instance or None.
        """
        for plugin in self._daemon_plugins.values():
            if hasattr(plugin, 'get_user_commands'):
                for cmd in plugin.get_user_commands():
                    if cmd.name == command:
                        return plugin
        return None

    def _execute_daemon_command(
        self,
        client_id: str,
        plugin,
        command: str,
        args: list,
    ) -> None:
        """Execute a user command on a daemon-level plugin (no session required).

        Sets up output callback to route plugin output to the client via events,
        parses arguments, and handles HelpLines results.

        Args:
            client_id: The requesting client.
            plugin: The daemon-level plugin instance.
            command: The command name.
            args: Raw argument list from the client.
        """
        from jaato_sdk.events import HelpTextEvent, SystemMessageEvent
        from shared.plugins.base import parse_command_args, HelpLines

        # Buffer plugin._emit() output — daemon commands run outside any agent
        # context, so we accumulate output and send as a SystemMessageEvent.
        output_parts = []
        if hasattr(plugin, 'set_output_callback'):
            def output_callback(source: str, text: str, mode: str) -> None:
                output_parts.append(text)
            plugin.set_output_callback(output_callback)

        try:
            # Find the UserCommand definition for arg parsing
            cmd_def = None
            for cmd in plugin.get_user_commands():
                if cmd.name == command:
                    cmd_def = cmd
                    break

            parsed_args = parse_command_args(cmd_def, ' '.join(args)) if cmd_def else {}
            result = plugin.execute_user_command(command, parsed_args)

            # Send accumulated _emit() output as a single system message
            if output_parts:
                combined = "".join(output_parts).rstrip("\n")
                if combined:
                    self._route_event(client_id, SystemMessageEvent(
                        message=combined,
                        style="info",
                    ))

            if isinstance(result, HelpLines):
                self._route_event(client_id, HelpTextEvent(lines=result.lines))
            elif isinstance(result, str) and result:
                self._route_event(client_id, SystemMessageEvent(
                    message=result,
                    style="info",
                ))

            # After auth command execution, check if credentials are now valid
            # and offer to set up a session with the provider.
            if hasattr(plugin, 'verify_credentials') and plugin.verify_credentials():
                self._offer_post_auth_setup(client_id, plugin)

        except Exception as e:
            self._route_event(client_id, SystemMessageEvent(
                message=f"Command error: {e}",
                style="error",
            ))

        finally:
            if hasattr(plugin, 'set_output_callback'):
                plugin.set_output_callback(None)

    def _hint_available_auth_providers(self, client_id: str) -> None:
        """Send a hint listing available auth providers after session creation fails.

        Iterates daemon-level plugins with the TRAIT_AUTH_PROVIDER trait and
        emits a message showing their login commands so the user knows how to
        configure a provider for this workspace.
        """
        from shared.plugins.base import TRAIT_AUTH_PROVIDER

        hints: list[str] = []
        for plugin in self._daemon_plugins.values():
            traits = getattr(plugin, 'plugin_traits', frozenset())
            if TRAIT_AUTH_PROVIDER not in traits:
                continue
            display_name = getattr(plugin, 'provider_display_name', plugin.name)
            commands = plugin.get_user_commands() if hasattr(plugin, 'get_user_commands') else []
            cmd_name = commands[0].name if commands else plugin.name
            hints.append(f"  {cmd_name} login  — {display_name}")

        if hints:
            from jaato_sdk.events import SystemMessageEvent
            msg = "Available providers:\n" + "\n".join(hints)
            self._route_event(client_id, SystemMessageEvent(
                message=msg,
                style="dim",
            ))

    def _offer_post_auth_setup(self, client_id: str, plugin) -> None:
        """Emit PostAuthSetupEvent to offer session creation after auth success."""
        import uuid
        from jaato_sdk.events import PostAuthSetupEvent

        provider_name = getattr(plugin, 'provider_name', '')
        if not provider_name:
            return

        # Check if client already has an active session
        has_active_session = False
        current_provider = ""
        current_model = ""
        if self._session_manager:
            session = self._session_manager.get_client_session(client_id)
            if session and session.server:
                has_active_session = True
                current_provider = getattr(session.server, '_provider_name', '') or ""
                current_model = getattr(session.server, '_model_name', '') or ""

        workspace_path = ""
        if self._ipc_server:
            workspace_path = self._ipc_server.get_client_workspace(client_id) or ""

        models = []
        if hasattr(plugin, 'get_default_models'):
            models = plugin.get_default_models()

        request_id = str(uuid.uuid4())
        self._pending_post_auth[client_id] = {
            "request_id": request_id,
            "provider_name": provider_name,
            "credential_env_vars": getattr(plugin, 'credential_env_vars', []),
        }

        self._route_event(client_id, PostAuthSetupEvent(
            request_id=request_id,
            provider_name=provider_name,
            provider_display_name=getattr(plugin, 'provider_display_name', provider_name),
            available_models=models,
            has_active_session=has_active_session,
            current_provider=current_provider,
            current_model=current_model,
            workspace_path=workspace_path,
        ))

    def _handle_post_auth_response(self, client_id: str, event) -> None:
        """Handle PostAuthSetupResponse from client.

        Creates/reconfigures session and optionally writes .env file.
        """
        from jaato_sdk.events import SystemMessageEvent

        pending = self._pending_post_auth.pop(client_id, None)
        if not pending or pending["request_id"] != event.request_id:
            logger.warning(f"No pending post-auth request for client {client_id}")
            return

        if not event.connect:
            return

        provider_name = pending["provider_name"]
        model_name = event.model_name

        if not model_name:
            self._route_event(client_id, SystemMessageEvent(
                message="No model selected, skipping session setup.",
                style="dim",
            ))
            return

        # Strip provider prefix from model name if present (e.g., "zhipuai/glm-4.7" -> "glm-4.7")
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]

        workspace_path = None
        if self._ipc_server:
            workspace_path = self._ipc_server.get_client_workspace(client_id)

        # Persist to .env if requested
        if event.persist_env and workspace_path:
            credential_env_vars = pending.get("credential_env_vars", [])
            self._persist_env(workspace_path, provider_name, model_name, credential_env_vars)
            self._route_event(client_id, SystemMessageEvent(
                message=f"Saved JAATO_PROVIDER={provider_name} and MODEL_NAME={model_name} to .env",
                style="info",
            ))

        # Create a new session with the authenticated provider.
        # Pass provider/model as env_overrides so they take precedence over
        # whatever the .env file currently has (user may have declined to persist).
        if self._session_manager:
            session_id = self._session_manager.create_session(
                client_id, None, workspace_path=workspace_path,
                env_overrides={
                    "JAATO_PROVIDER": provider_name,
                    "MODEL_NAME": model_name,
                },
            )
            if session_id:
                set_logging_context(
                    session_id=session_id,
                    client_id=client_id,
                    workspace_path=workspace_path,
                )
                if self._ipc_server:
                    self._ipc_server.set_client_session(client_id, session_id)

                self._route_event(client_id, SystemMessageEvent(
                    message=f"Session created with {provider_name} / {model_name}",
                    style="success",
                ))
            else:
                self._route_event(client_id, SystemMessageEvent(
                    message="Failed to create session.",
                    style="error",
                ))

    def _persist_env(
        self,
        workspace_path: str,
        provider_name: str,
        model_name: str,
        credential_env_vars: Optional[List[str]] = None,
    ) -> None:
        """Write or update JAATO_PROVIDER and MODEL_NAME in workspace .env file.

        Only replaces active (uncommented) lines. Commented-out lines like
        ``#JAATO_PROVIDER=...`` are preserved untouched.

        When *credential_env_vars* is provided (from the auth plugin), any
        commented-out lines for those vars are annotated to indicate that
        credentials are stored securely in ``.jaato/`` and managed by the
        auth command — so users know auth is configured even though the
        key doesn't appear in ``.env``.
        """
        env_path = os.path.join(workspace_path, '.env')
        lines = []
        seen_provider = False
        seen_model = False
        cred_vars = set(credential_env_vars or [])

        # Read existing .env if it exists
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('JAATO_PROVIDER='):
                        lines.append(f'JAATO_PROVIDER={provider_name}\n')
                        seen_provider = True
                    elif stripped.startswith('MODEL_NAME='):
                        lines.append(f'MODEL_NAME={model_name}\n')
                        seen_model = True
                    elif cred_vars and self._is_commented_credential(stripped, cred_vars):
                        # Annotate commented credential line to indicate
                        # it's managed by the auth plugin
                        var_name = self._extract_var_name(stripped)
                        lines.append(f'# {var_name}=<stored in .jaato/ — use {provider_name}-auth>\n')
                        cred_vars.discard(var_name)
                    else:
                        lines.append(line)

        # Append missing keys
        if not seen_provider:
            lines.append(f'JAATO_PROVIDER={provider_name}\n')
        if not seen_model:
            lines.append(f'MODEL_NAME={model_name}\n')

        with open(env_path, 'w') as f:
            f.writelines(lines)

    @staticmethod
    def _is_commented_credential(stripped: str, cred_vars: set) -> bool:
        """Check if a stripped line is a commented-out credential env var."""
        if not stripped.startswith('#'):
            return False
        # Strip leading # and whitespace: "# ZHIPUAI_API_KEY=..." -> "ZHIPUAI_API_KEY=..."
        uncommented = stripped.lstrip('#').lstrip()
        return any(uncommented.startswith(f'{var}=') for var in cred_vars)

    @staticmethod
    def _extract_var_name(stripped: str) -> str:
        """Extract the env var name from a commented line like '# ZHIPUAI_API_KEY=...'."""
        uncommented = stripped.lstrip('#').lstrip()
        return uncommented.split('=', 1)[0]

    def _get_command_list(self) -> list:
        """Get list of available commands for clients.

        Returns:
            List of {name, description} dicts.
        """
        commands = []

        # Static session management commands (handled by daemon)
        session_commands = [
            {"name": "session list", "description": "List all sessions"},
            {"name": "session new", "description": "Create a new session"},
            {"name": "session attach", "description": "Attach to an existing session"},
            {"name": "session delete", "description": "Delete a session"},
            {"name": "session help", "description": "Show detailed help for session command"},
        ]
        commands.extend(session_commands)

        # Static tools commands (handled by daemon)
        tools_commands = [
            {"name": "tools list", "description": "List all tools with status"},
            {"name": "tools enable", "description": "Enable a tool (or 'all')"},
            {"name": "tools disable", "description": "Disable a tool (or 'all')"},
            {"name": "tools help", "description": "Show detailed help for tools command"},
        ]
        commands.extend(tools_commands)

        # Session-independent plugin commands (auth plugins).
        # Available regardless of whether a session is loaded.
        for plugin in self._daemon_plugins.values():
            if hasattr(plugin, 'get_user_commands'):
                for cmd in plugin.get_user_commands():
                    if hasattr(plugin, 'get_command_completions'):
                        subcommands = plugin.get_command_completions(cmd.name, [])
                        if subcommands:
                            for sub in subcommands:
                                commands.append({
                                    "name": f"{cmd.name} {sub.value}",
                                    "description": sub.description or "",
                                })
                        else:
                            commands.append({
                                "name": cmd.name,
                                "description": cmd.description or "",
                            })
                    else:
                        commands.append({
                            "name": cmd.name,
                            "description": cmd.description or "",
                        })

        # Get commands from any active session
        # (includes model command from session, plugin commands with subcommands)
        if self._session_manager:
            sessions = self._session_manager.list_sessions()
            for session_info in sessions:
                if session_info.is_loaded:
                    session = self._session_manager.get_session(session_info.session_id)
                    if session and session.server:
                        # Get commands from server (with model subcommand expansion)
                        server_cmds = session.server.get_available_commands()
                        for name, description in server_cmds.items():
                            # Special handling for model command - expand subcommands
                            if name == "model" and hasattr(session.server, '_jaato'):
                                jaato = session.server._jaato
                                if jaato and hasattr(jaato, 'get_model_completions'):
                                    model_subs = jaato.get_model_completions([])
                                    for sub in model_subs:
                                        commands.append({
                                            "name": f"model {sub.value}",
                                            "description": sub.description or "",
                                        })
                                else:
                                    commands.append({"name": name, "description": description or ""})
                            else:
                                commands.append({
                                    "name": name,
                                    "description": description or "",
                                })

                        # Get commands from registry plugins (with two-level subcommand expansion)
                        if session.server.registry:
                            for plugin_name in session.server.registry.list_exposed():
                                plugin = session.server.registry.get_plugin(plugin_name)
                                if plugin and hasattr(plugin, 'get_user_commands'):
                                    for cmd in plugin.get_user_commands():
                                        # Check if plugin has subcommand completions
                                        if hasattr(plugin, 'get_command_completions'):
                                            subcommands = plugin.get_command_completions(cmd.name, [])
                                            if subcommands:
                                                # Skip second-level expansion for plugins with dynamic
                                                # argument completions (e.g., memory IDs and service names
                                                # change at runtime and are refreshed via dedicated events
                                                # + PluginCommandCompleter)
                                                has_dynamic_completions = (
                                                    hasattr(plugin, 'get_memory_metadata')
                                                    or hasattr(plugin, 'get_service_metadata')
                                                )

                                                # Add expanded subcommands
                                                for sub in subcommands:
                                                    commands.append({
                                                        "name": f"{cmd.name} {sub.value}",
                                                        "description": sub.description or "",
                                                    })
                                                    if not has_dynamic_completions:
                                                        # Check for second-level completions
                                                        sub_completions = plugin.get_command_completions(
                                                            cmd.name, [sub.value, ""]
                                                        )
                                                        for sub2 in sub_completions:
                                                            commands.append({
                                                                "name": f"{cmd.name} {sub.value} {sub2.value}",
                                                                "description": sub2.description or "",
                                                            })
                                            else:
                                                # No subcommands, add base command
                                                commands.append({
                                                    "name": cmd.name,
                                                    "description": cmd.description or "",
                                                })
                                        else:
                                            # No completion method, add base command
                                            commands.append({
                                                "name": cmd.name,
                                                "description": cmd.description or "",
                                            })

                        # Get commands from permission plugin (with two-level subcommand expansion)
                        if session.server.permission_plugin:
                            perm = session.server.permission_plugin
                            if hasattr(perm, 'get_user_commands'):
                                for cmd in perm.get_user_commands():
                                    if hasattr(perm, 'get_command_completions'):
                                        subcommands = perm.get_command_completions(cmd.name, [])
                                        if subcommands:
                                            for sub in subcommands:
                                                # Add first-level subcommand
                                                commands.append({
                                                    "name": f"{cmd.name} {sub.value}",
                                                    "description": sub.description or "",
                                                })
                                                # Check for second-level completions
                                                sub_completions = perm.get_command_completions(
                                                    cmd.name, [sub.value, ""]
                                                )
                                                for sub2 in sub_completions:
                                                    commands.append({
                                                        "name": f"{cmd.name} {sub.value} {sub2.value}",
                                                        "description": sub2.description or "",
                                                    })
                                        else:
                                            commands.append({
                                                "name": cmd.name,
                                                "description": cmd.description or "",
                                            })
                                    else:
                                        commands.append({
                                            "name": cmd.name,
                                            "description": cmd.description or "",
                                        })

                        # Got commands from one session, that's enough
                        break

        # Deduplicate by name
        seen = set()
        unique_commands = []
        for cmd in commands:
            if cmd["name"] not in seen:
                seen.add(cmd["name"])
                unique_commands.append(cmd)

        return unique_commands

    def _get_tool_status(self, server) -> list:
        """Get tool status as structured data.

        Args:
            server: JaatoServer instance.

        Returns:
            List of tool status dicts: {name, description, enabled, plugin}
        """
        tool_status = []

        # Get registry tools with status
        if server.registry:
            tool_status.extend(server.registry.get_tool_status())

        # Add permission plugin tools (always enabled)
        if server.permission_plugin:
            for schema in server.permission_plugin.get_tool_schemas():
                tool_status.append({
                    'name': schema.name,
                    'description': schema.description,
                    'enabled': True,
                    'plugin': 'permission',
                })

        return tool_status

    def _tools_enable(self, server, tool_name: str) -> str:
        """Enable a tool.

        Args:
            server: JaatoServer instance.
            tool_name: Tool name or 'all'.

        Returns:
            Result message.
        """
        if not server.registry:
            return "No registry available."

        if tool_name.lower() == "all":
            count = 0
            for status in server.registry.get_tool_status():
                if not status.get('enabled', True):
                    server.registry.enable_tool(status['name'])
                    count += 1
            return f"Enabled {count} tools."

        if server.registry.enable_tool(tool_name):
            return f"Enabled tool: {tool_name}"
        return f"Tool not found or already enabled: {tool_name}"

    def _tools_disable(self, server, tool_name: str) -> str:
        """Disable a tool.

        Args:
            server: JaatoServer instance.
            tool_name: Tool name or 'all'.

        Returns:
            Result message.
        """
        if not server.registry:
            return "No registry available."

        if tool_name.lower() == "all":
            count = 0
            for status in server.registry.get_tool_status():
                if status.get('enabled', True):
                    server.registry.disable_tool(status['name'])
                    count += 1
            return f"Disabled {count} tools."

        if server.registry.disable_tool(tool_name):
            return f"Disabled tool: {tool_name}"
        return f"Tool not found or already disabled: {tool_name}"

    def _serialize_part(self, part) -> dict:
        """Serialize a message part to a dict.

        Args:
            part: Message Part object.

        Returns:
            Dict with part data.
        """
        if hasattr(part, 'text') and part.text is not None:
            return {"type": "text", "text": part.text}
        elif hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            return {
                "type": "function_call",
                "name": fc.name if hasattr(fc, 'name') else str(fc),
                "args": fc.args if hasattr(fc, 'args') else {},
            }
        elif hasattr(part, 'function_response') and part.function_response:
            fr = part.function_response
            return {
                "type": "function_response",
                "name": fr.name if hasattr(fr, 'name') else str(fr),
                "response": fr.response if hasattr(fr, 'response') else str(fr),
            }
        else:
            return {"type": "unknown", "data": str(part)}


def daemonize(log_file: str = DEFAULT_LOG_FILE) -> None:
    """Daemonize the process (double-fork method on Unix, subprocess on Windows)."""
    if sys.platform == "win32":
        # Windows: use subprocess to start detached process
        import subprocess
        args = [sys.executable] + sys.argv
        # Remove --daemon from args to avoid infinite recursion
        args = [a for a in args if a not in ("--daemon", "-d")]
        # Add a marker to indicate we're already daemonized
        env = os.environ.copy()
        env["JAATO_DAEMONIZED"] = "1"
        # Start detached process without console window
        subprocess.Popen(
            args,
            stdout=open(log_file, 'a'),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=(
                subprocess.DETACHED_PROCESS |
                subprocess.CREATE_NEW_PROCESS_GROUP |
                subprocess.CREATE_NO_WINDOW
            ),
            env=env,
        )
        sys.exit(0)
    else:
        # Unix: use double-fork method
        # First fork
        pid = os.fork()
        if pid > 0:
            # Parent exits
            sys.exit(0)

        # Create new session
        os.setsid()

        # Second fork
        pid = os.fork()
        if pid > 0:
            sys.exit(0)

        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()

        with open('/dev/null', 'r') as devnull:
            os.dup2(devnull.fileno(), sys.stdin.fileno())

        with open(log_file, 'a') as log:
            os.dup2(log.fileno(), sys.stdout.fileno())
            os.dup2(log.fileno(), sys.stderr.fileno())


def check_running(pid_file: str = DEFAULT_PID_FILE) -> Optional[int]:
    """Check if a server is already running.

    Returns:
        The PID if running, None otherwise.
    """
    if not os.path.exists(pid_file):
        return None

    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())

        # Check if process exists
        if sys.platform == "win32":
            # Windows: use tasklist or ctypes to check process
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return pid
            else:
                raise ProcessLookupError("Process not found")
        else:
            os.kill(pid, 0)
            return pid

    except (ValueError, ProcessLookupError, PermissionError, OSError):
        # PID file exists but process is dead
        try:
            os.remove(pid_file)
        except Exception:
            pass
        return None


def load_config(config_file: str = DEFAULT_CONFIG_FILE) -> Optional[dict]:
    """Load saved startup config.

    Returns:
        Config dict if exists, None otherwise.
    """
    if not os.path.exists(config_file):
        return None

    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def stop_server(pid_file: str = DEFAULT_PID_FILE) -> bool:
    """Stop a running server.

    Returns:
        True if stopped, False if not running.
    """
    pid = check_running(pid_file)
    if not pid:
        return False

    try:
        import time

        if sys.platform == "win32":
            # Windows: use taskkill or TerminateProcess
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_TERMINATE = 0x0001
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(
                PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                kernel32.TerminateProcess(handle, 0)
                kernel32.CloseHandle(handle)
            # Wait for process to exit
            for _ in range(50):  # 5 seconds timeout
                time.sleep(0.1)
                if not check_running(pid_file):
                    return True
            return True
        else:
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit
            for _ in range(50):  # 5 seconds timeout
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
            # Force kill
            os.kill(pid, signal.SIGKILL)
            return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Jaato Server - Multi-client AI assistant backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with IPC socket (local clients)
  python -m server --ipc-socket /tmp/jaato.sock

  # Start with WebSocket (remote clients)
  python -m server --web-socket :8080

  # Start with both
  python -m server --ipc-socket /tmp/jaato.sock --web-socket :8080

  # Start as daemon (background)
  python -m server --ipc-socket /tmp/jaato.sock --daemon

  # Check status
  python -m server --status

  # Stop daemon
  python -m server --stop

  # Restart with same parameters (development)
  python -m server --restart
        """,
    )

    # Server endpoints
    parser.add_argument(
        "--ipc-socket",
        metavar="PATH",
        help=f"Unix domain socket path for local clients (default: {DEFAULT_SOCKET_PATH})",
    )
    parser.add_argument(
        "--web-socket",
        metavar="[HOST:]PORT",
        help="WebSocket address for remote clients (e.g., :8080 or 0.0.0.0:8080)",
    )

    # Daemon control
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as daemon (background process)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check if server is running",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop a running daemon",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart the daemon with same parameters",
    )

    # Configuration
    parser.add_argument(
        "--pid-file",
        default=DEFAULT_PID_FILE,
        help=f"PID file path (default: {DEFAULT_PID_FILE})",
    )
    parser.add_argument(
        "--log-file",
        default=DEFAULT_LOG_FILE,
        help=f"Log file for daemon mode (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging - initially to console, will switch to file for daemon mode
    configure_logging(verbose=args.verbose)

    # Handle --status
    if args.status:
        pid = check_running(args.pid_file)
        if pid:
            print(f"Jaato server is running (PID: {pid})")
            # Show socket info
            if os.path.exists(DEFAULT_SOCKET_PATH):
                print(f"  IPC socket: {DEFAULT_SOCKET_PATH}")
            sys.exit(0)
        else:
            print("Jaato server is not running")
            sys.exit(1)

    # Handle --stop
    if args.stop:
        if stop_server(args.pid_file):
            print("Jaato server stopped")
            sys.exit(0)
        else:
            print("Jaato server is not running")
            sys.exit(1)

    # Handle --restart
    if args.restart:
        config = load_config()
        if not config:
            print("Error: No saved config found. Cannot restart.")
            print("  Start the server normally first, then use --restart")
            sys.exit(1)

        # Stop current server
        pid = check_running(args.pid_file)
        if pid:
            print(f"Stopping server (PID: {pid})...")
            if not stop_server(args.pid_file):
                print("Error: Failed to stop server")
                sys.exit(1)
            print("Server stopped")
        else:
            print("Server was not running")

        # Apply saved config
        args.ipc_socket = config.get("ipc_socket")
        args.web_socket = config.get("web_socket")
        args.log_file = config.get("log_file", DEFAULT_LOG_FILE)

        # Always restart as daemon
        args.daemon = True

        print(f"Restarting server...")
        if args.ipc_socket:
            print(f"  IPC socket: {args.ipc_socket}")
        if args.web_socket:
            print(f"  WebSocket: {args.web_socket}")

    # Validate arguments
    if not args.ipc_socket and not args.web_socket:
        # Default to IPC (Unix socket on Unix, named pipe on Windows)
        args.ipc_socket = DEFAULT_SOCKET_PATH
        if sys.platform == "win32":
            print(f"No endpoint specified, using default named pipe: \\\\.\\pipe\\{args.ipc_socket}")
        else:
            print(f"No endpoint specified, using default IPC socket: {args.ipc_socket}")

    # Check if already running
    pid = check_running(args.pid_file)
    if pid:
        print(f"Error: Jaato server is already running (PID: {pid})")
        print(f"  Use 'python -m server --stop' to stop it")
        sys.exit(1)

    # Daemonize if requested (skip if already daemonized on Windows)
    if args.daemon and not os.environ.get("JAATO_DAEMONIZED"):
        print(f"Starting Jaato server as daemon...")
        print(f"  PID file: {args.pid_file}")
        print(f"  Log file: {args.log_file}")
        if args.ipc_socket:
            print(f"  IPC socket: {args.ipc_socket}")
        if args.web_socket:
            print(f"  WebSocket: {args.web_socket}")
        daemonize(args.log_file)

    # Reconfigure logging for daemon/background mode with rotating file handler
    # This ensures log files don't grow unbounded
    if args.daemon or os.environ.get("JAATO_DAEMONIZED"):
        configure_logging(log_file=args.log_file, verbose=args.verbose)

    # Create and run daemon
    daemon = JaatoDaemon(
        ipc_socket=args.ipc_socket,
        web_socket=args.web_socket,
        pid_file=args.pid_file,
        log_file=args.log_file,
    )

    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
