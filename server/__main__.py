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
import os
import signal
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from server.session_manager import SessionManager
from server.events import Event


# Default paths
DEFAULT_SOCKET_PATH = "/tmp/jaato.sock"
DEFAULT_PID_FILE = "/tmp/jaato.pid"
DEFAULT_LOG_FILE = "/tmp/jaato.log"
DEFAULT_CONFIG_FILE = "/tmp/jaato.config.json"


logger = logging.getLogger(__name__)


class JaatoDaemon:
    """Main server daemon managing IPC and WebSocket servers."""

    def __init__(
        self,
        ipc_socket: Optional[str] = None,
        web_socket: Optional[str] = None,
        env_file: str = ".env",
        provider: Optional[str] = None,
        pid_file: str = DEFAULT_PID_FILE,
        config_file: str = DEFAULT_CONFIG_FILE,
        log_file: str = DEFAULT_LOG_FILE,
    ):
        """Initialize the daemon.

        Args:
            ipc_socket: Path to Unix domain socket (None to disable).
            web_socket: WebSocket address as "host:port" or ":port" (None to disable).
            env_file: Path to .env file.
            provider: Model provider override.
            pid_file: Path to PID file for daemon mode.
            config_file: Path to config file for restart support.
            log_file: Path to log file for daemon mode.
        """
        self.ipc_socket = ipc_socket
        self.web_socket = web_socket
        self.env_file = env_file
        self.provider = provider
        self.pid_file = pid_file
        self.config_file = config_file
        self.log_file = log_file

        # Components
        self._session_manager: Optional[SessionManager] = None
        self._ipc_server = None
        self._ws_server = None

        # Shutdown flag
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the daemon and run until shutdown."""
        # Load environment
        load_dotenv(self.env_file)

        # Initialize session manager
        self._session_manager = SessionManager(
            env_file=self.env_file,
            provider=self.provider,
        )

        # Set up event routing
        self._session_manager.set_event_callback(self._route_event)

        tasks = []

        # Start IPC server if configured
        if self.ipc_socket:
            from server.ipc import JaatoIPCServer

            self._ipc_server = JaatoIPCServer(
                socket_path=self.ipc_socket,
                on_session_request=self._handle_session_request,
                on_command_list_request=self._get_command_list,
            )
            tasks.append(asyncio.create_task(self._ipc_server.start()))
            logger.info(f"IPC server will listen on {self.ipc_socket}")

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
                env_file=self.env_file,
                provider=self.provider,
            )
            # Note: WebSocket server needs to be updated to use session manager
            # For now, it runs standalone
            tasks.append(asyncio.create_task(self._ws_server.start()))
            logger.info(f"WebSocket server will listen on ws://{host}:{port}")

        if not tasks:
            logger.error("No servers configured. Use --ipc-socket and/or --web-socket")
            return

        # Write PID and config files
        self._write_pid()
        self._write_config()

        # Set up signal handlers
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
            "env_file": self.env_file,
            "provider": self.provider,
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

        # Handle session management commands
        from server.events import CommandRequest

        if isinstance(event, CommandRequest):
            cmd = event.command.lower()

            if cmd == "session.create":
                name = event.args[0] if event.args else None
                new_session_id = self._session_manager.create_session(client_id, name)
                if self._ipc_server and new_session_id:
                    self._ipc_server.set_client_session(client_id, new_session_id)
                return

            elif cmd == "session.attach":
                if event.args:
                    if self._session_manager.attach_session(client_id, event.args[0]):
                        if self._ipc_server:
                            self._ipc_server.set_client_session(client_id, event.args[0])
                return

            elif cmd == "session.list":
                sessions = self._session_manager.list_sessions()
                from server.events import SystemMessageEvent
                self._route_event(client_id, SystemMessageEvent(
                    message=json.dumps([{
                        "id": s.session_id,
                        "name": s.name,
                        "model": f"{s.model_provider}/{s.model_name}",
                        "clients": s.client_count,
                        "turns": s.turn_count,
                    } for s in sessions], indent=2),
                    style="info",
                ))
                return

            elif cmd == "session.default":
                default_session_id = self._session_manager.get_or_create_default(client_id)
                if self._ipc_server and default_session_id:
                    self._ipc_server.set_client_session(client_id, default_session_id)
                return

        # Route to session
        self._session_manager.handle_request(client_id, session_id, event)

    def _route_event(self, client_id: str, event: Event) -> None:
        """Route an event to a client."""
        if self._ipc_server:
            self._ipc_server.queue_event(client_id, event)
        # WebSocket routing would be added here

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
        ]
        commands.extend(session_commands)

        # Get commands from any active session
        if self._session_manager:
            sessions = self._session_manager.list_sessions()
            for session_info in sessions:
                if session_info.is_loaded:
                    session = self._session_manager.get_session(session_info.session_id)
                    if session and session.server:
                        # Get commands from server
                        server_cmds = session.server.get_available_commands()
                        for name, description in server_cmds.items():
                            commands.append({
                                "name": name,
                                "description": description or "",
                            })

                        # Get commands from registry plugins
                        if session.server.registry:
                            for plugin_name in session.server.registry.list_exposed():
                                plugin = session.server.registry.get_plugin(plugin_name)
                                if plugin and hasattr(plugin, 'get_user_commands'):
                                    for cmd in plugin.get_user_commands():
                                        commands.append({
                                            "name": cmd.name,
                                            "description": cmd.description or "",
                                        })

                        # Get commands from permission plugin
                        if session.server.permission_plugin:
                            perm = session.server.permission_plugin
                            if hasattr(perm, 'get_user_commands'):
                                for cmd in perm.get_user_commands():
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


def daemonize(log_file: str = DEFAULT_LOG_FILE) -> None:
    """Daemonize the process (double-fork method)."""
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
        os.kill(pid, 0)
        return pid

    except (ValueError, ProcessLookupError, PermissionError):
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
        os.kill(pid, signal.SIGTERM)
        # Wait for process to exit
        import time
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
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--provider",
        help="Model provider override",
    )
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

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
        args.env_file = config.get("env_file", ".env")
        args.provider = config.get("provider")
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
        # Default to IPC socket
        args.ipc_socket = DEFAULT_SOCKET_PATH
        print(f"No endpoint specified, using default IPC socket: {args.ipc_socket}")

    # Check if already running
    pid = check_running(args.pid_file)
    if pid:
        print(f"Error: Jaato server is already running (PID: {pid})")
        print(f"  Use 'python -m server --stop' to stop it")
        sys.exit(1)

    # Daemonize if requested
    if args.daemon:
        print(f"Starting Jaato server as daemon...")
        print(f"  PID file: {args.pid_file}")
        print(f"  Log file: {args.log_file}")
        if args.ipc_socket:
            print(f"  IPC socket: {args.ipc_socket}")
        if args.web_socket:
            print(f"  WebSocket: {args.web_socket}")
        daemonize(args.log_file)

    # Create and run daemon
    daemon = JaatoDaemon(
        ipc_socket=args.ipc_socket,
        web_socket=args.web_socket,
        env_file=args.env_file,
        provider=args.provider,
        pid_file=args.pid_file,
        log_file=args.log_file,
    )

    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
