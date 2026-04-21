#!/usr/bin/env python3
"""Jaato Server - Multi-client AI assistant backend.

This is the main entry point for the Jaato server, which provides:
- IPC socket for local clients (jaato-tui, IDE extensions)
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
import importlib.metadata
import json
import logging
import logging.handlers
import os
import secrets
import signal
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.session_manager import SessionManager
from server.session_logging import configure_session_logging


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


class _ExtensionContext:
    """Context namespace passed to daemon extension factories.

    Provides access to the daemon's core infrastructure so that extensions
    can register hooks, interceptors, and other integrations without
    depending on concrete daemon internals.

    This class is intentionally simple — a plain namespace with typed
    attributes.  Extensions should treat it as read-only.

    Attributes:
        session_manager: The daemon's ``SessionManager`` instance.
            Extensions typically call ``session_manager.add_session_hook()``
            to register a callback invoked after each session is initialized.
        ws_server: The ``JaatoWSServer`` instance if ``--web-socket`` was
            passed, or ``None`` otherwise.  Extensions can call
            ``ws_server.set_connection_interceptor(check, handler)`` to
            route special WebSocket connections to custom handlers, or
            ``ws_server.register_message_handler(type, callback)`` to
            handle custom message types on client connections.
        web_socket: The raw ``--web-socket`` CLI argument string (e.g.
            ``:8080``, ``0.0.0.0:8080``), or ``None``.
        ipc_socket: The raw ``--ipc-socket`` CLI argument string, or ``None``.
        server_name: The ``--server-name`` CLI argument, or ``None``.
        dashboard_port: The ``--dashboard-port`` CLI argument (int), or ``None``.
        available_plugins: Frozen set of plugin names that the server can
            load.  Discovered once at daemon startup via
            ``PluginRegistry.discover()``.  Names match what profiles use
            (e.g. ``"cli"``, ``"references"``, ``"todo"``).
        plugin_registry: The ``PluginRegistry`` instance used for discovery.
            Extensions can call ``plugin_registry.get_plugin_config_schema(name)``
            to introspect a plugin's configurable settings.
        available_gc_plugins: Frozen set of GC plugin names discovered at
            startup (e.g. ``"gc_truncate"``, ``"gc_budget"``).
        gc_plugin_factories: Dict mapping GC plugin names to their factory
            functions.  Extensions can instantiate a GC plugin to call
            ``get_config_schema()`` for settings introspection.
    """

    __slots__ = (
        "session_manager", "ws_server", "web_socket",
        "ipc_socket", "server_name", "dashboard_port",
        "available_plugins", "plugin_registry",
        "available_gc_plugins", "gc_plugin_factories",
    )

    def __init__(
        self,
        session_manager,
        ws_server,
        web_socket: Optional[str],
        ipc_socket: Optional[str],
        server_name: Optional[str],
        dashboard_port: Optional[int],
        available_plugins: frozenset = frozenset(),
        plugin_registry=None,
        available_gc_plugins: frozenset = frozenset(),
        gc_plugin_factories: dict = None,
    ):
        self.session_manager = session_manager
        self.ws_server = ws_server
        self.web_socket = web_socket
        self.ipc_socket = ipc_socket
        self.server_name = server_name
        self.dashboard_port = dashboard_port
        self.available_plugins = available_plugins
        self.plugin_registry = plugin_registry
        self.available_gc_plugins = available_gc_plugins
        self.gc_plugin_factories = gc_plugin_factories or {}


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
        socket_mode: int = 0o666,
        dashboard_port: Optional[int] = None,
        server_name: Optional[str] = None,
        ws_token: Optional[str] = None,
        ws_token_file: Optional[str] = None,
        ws_unsafe_no_auth: bool = False,
    ):
        """Initialize the daemon.

        Args:
            ipc_socket: Path to Unix domain socket (None to disable).
            web_socket: WebSocket address as "host:port" or ":port" (None to disable).
            pid_file: Path to PID file for daemon mode.
            config_file: Path to config file for restart support.
            log_file: Path to log file for daemon mode.
            socket_mode: Unix file permissions for the IPC socket (default: 0o666).
            dashboard_port: TCP port for the dashboard and health HTTP endpoint
                (None to disable).
            server_name: Explicit server name for self-identification.
                Passed to daemon extensions via ``_ExtensionContext``.
            ws_token: Bearer token clients must present in the WS Upgrade
                request. ``None`` means WS auth is disabled (open accept
                — only acceptable on a trusted network or behind a
                terminating reverse proxy that does its own auth).
        """
        self.ipc_socket = ipc_socket
        self.web_socket = web_socket
        self.socket_mode = socket_mode
        self.pid_file = pid_file
        self.config_file = config_file
        self.log_file = log_file
        self._dashboard_port = dashboard_port
        self._server_name = server_name
        self._ws_token = ws_token
        # Stored only so they can round-trip through _write_config for
        # --restart. The plaintext _ws_token is never serialised.
        self._ws_token_file = ws_token_file
        self._ws_unsafe_no_auth = ws_unsafe_no_auth

        # Components
        self._session_manager: Optional[SessionManager] = None
        self._ipc_server = None
        self._ws_server = None

        # Daemon extensions loaded via ``jaato.extensions`` entry points.
        # See ``_load_extensions()`` for the discovery and lifecycle protocol.
        self._extensions: list = []

        # Session-independent plugins (auth plugins loaded at daemon startup)
        # These provide user commands that work without an active session/provider.
        self._daemon_plugins: dict = {}  # name -> plugin instance

        # CommandRouter — created in start(), owns pending-state dicts
        self._command_router = None

        # Shutdown flag
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the daemon and run until shutdown.

        Wiring sequence:
        1. Create ``SessionManager``
        2. Discover daemon-level plugins (auth providers)
        3. Create transport servers (IPC, WS)
        4. Build ``CompositeEventSink`` from transport sinks
        5. Create ``CommandRouter`` with composite sink
        6. Wire router into transports and session manager
        7. Load daemon extensions
        8. Run until shutdown
        """
        from server.event_sink import CompositeEventSink
        from server.command_router import CommandRouter

        # Write PID and config files early so that clients checking
        # _check_server_running() see this daemon before initialization
        # completes (avoids race where TUI auto-starts a second server).
        self._write_pid()
        self._write_config()

        # Initialize session manager
        self._session_manager = SessionManager()

        # Discover session-independent plugins (auth plugins).
        self._discover_daemon_plugins()

        tasks = []

        def _on_task_done(task: asyncio.Task) -> None:
            """Log task failures immediately and trigger shutdown."""
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.error("Server task failed: %s", exc)
                asyncio.create_task(self.stop())

        # Build CompositeEventSink from transport servers
        composite_sink = CompositeEventSink()

        # Start IPC server if configured
        if self.ipc_socket:
            from server.ipc import JaatoIPCServer, _get_display_path

            self._ipc_server = JaatoIPCServer(
                socket_path=self.ipc_socket,
                socket_mode=self.socket_mode,
                # Will be set after CommandRouter is created
                on_session_request=None,
                on_command_list_request=None,
            )
            composite_sink.add_sink(self._ipc_server)
            t = asyncio.create_task(self._ipc_server.start())
            t.add_done_callback(_on_task_done)
            tasks.append(t)
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

            # Provision a workspace root so WS clients get isolated,
            # non-persistent workspaces (same as standalone WS mode).
            from pathlib import Path as _Path
            _default_ws_root = str(_Path.home() / ".jaato" / "workspaces")

            # Load TLS context from servers.json if available
            from server.websocket import load_tls_context
            ws_ssl_ctx = load_tls_context()

            self._ws_server = JaatoWSServer(
                host=host,
                port=port,
                workspace_root=_default_ws_root,
                ssl_context=ws_ssl_ctx,
                required_token=self._ws_token,
            )
            ws_adapter = self._ws_server.get_event_sink_adapter()
            ws_adapter.bind_loop(asyncio.get_running_loop())
            composite_sink.add_sink(ws_adapter)
            t = asyncio.create_task(self._ws_server.start())
            t.add_done_callback(_on_task_done)
            tasks.append(t)
            scheme = "wss" if ws_ssl_ctx else "ws"
            logger.info(f"WebSocket server will listen on {scheme}://{host}:{port}")

        if not tasks:
            logger.error("No servers configured. Use --ipc-socket and/or --web-socket")
            return

        # Create CommandRouter — unified dispatch for all transports
        self._command_router = CommandRouter(
            session_manager=self._session_manager,
            event_sink=composite_sink,
            daemon_plugins=self._daemon_plugins,
        )

        # Wire router into transports
        if self._ipc_server:
            self._ipc_server._on_session_request = self._command_router.handle_request
            self._ipc_server._on_command_list_request = self._command_router.get_command_list
        if self._ws_server:
            self._ws_server.set_command_router(self._command_router)

        # Wire composite sink as session manager's event callback
        self._session_manager.set_event_callback(composite_sink.send_event)

        # Load daemon extensions (e.g., gossip clustering from jaato-premium)
        self._load_extensions()

        # Set up signal handlers (not supported on Windows)
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        # Start daemon extensions after transport servers are up
        for ext in self._extensions:
            await ext.start()

        logger.info("Jaato server started")

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Cancel all tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass  # Failures already logged by _on_task_done

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

        for ext in reversed(self._extensions):
            try:
                await ext.stop()
            except Exception as exc:
                logger.warning("Extension stop failed: %s", exc)
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
        """Write startup config for restart support.

        Only references that don't leak secrets are persisted:
        ``ws_token_file`` (a path) yes, plaintext ``ws_token`` no, and
        the auto-generated banner token is intentionally not saved so
        each restart issues a fresh value.
        """
        config = {
            "ipc_socket": self.ipc_socket,
            "web_socket": self.web_socket,
            "pid_file": self.pid_file,
            "log_file": self.log_file,
            "socket_mode": self.socket_mode,
            "dashboard_port": self._dashboard_port,
            "server_name": self._server_name,
            "ws_token_file": self._ws_token_file,
            "ws_unsafe_no_auth": self._ws_unsafe_no_auth,
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

    # ------------------------------------------------------------------
    # Command dispatch is handled by CommandRouter (server/command_router.py).
    # The router is wired in start() and delegates to SessionManager.
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Daemon Extensions (entry-point-based lifecycle objects)
    # ------------------------------------------------------------------

    def _load_extensions(self) -> None:
        """Discover and instantiate daemon extensions from entry points.

        Extensions are loaded from the ``jaato.extensions`` entry-point group.
        Each entry point must resolve to a **factory function** with the
        signature::

            def create_extension(context: ExtensionContext) -> Extension

        The factory receives an ``_ExtensionContext`` (a simple namespace)
        with the following attributes:

        =================== ============================== ================
        Attribute           Type                           Description
        =================== ============================== ================
        ``session_manager`` ``SessionManager``             The daemon's
                                                           session manager
                                                           instance.
        ``ws_server``       ``JaatoWSServer | None``       WebSocket server
                                                           (``None`` when
                                                           ``--web-socket``
                                                           was not passed).
        ``web_socket``      ``str | None``                 Raw CLI arg
                                                           (e.g. ``:8080``).
        ``ipc_socket``      ``str | None``                 Raw CLI arg.
        ``server_name``     ``str | None``                 ``--server-name``
                                                           CLI argument.
        ``dashboard_port``  ``int | None``                 ``--dashboard-port``
                                                           CLI argument.
        =================== ============================== ================

        The returned extension object must implement two **async** methods:

        * ``async start()`` — called **after** transport servers (IPC and
          WebSocket) are up and listening.  Extensions should perform their
          startup work here (e.g., start background tasks, register hooks).

        * ``async stop()``  — called **before** the daemon shuts down.
          Extensions should clean up resources here.

        Extensions are started in discovery order and stopped in reverse
        order.

        **Registering an extension** (in ``pyproject.toml``)::

            [project.entry-points."jaato.extensions"]
            my_ext = "my_package.ext:create_extension"

        **Minimal extension skeleton**::

            class MyExtension:
                def __init__(self, ctx):
                    self._ctx = ctx

                async def start(self):
                    # Register a session hook so we get called for each
                    # new session:
                    self._ctx.session_manager.add_session_hook(self._hook)

                async def stop(self):
                    pass  # clean up

                def _hook(self, server, session_id):
                    # ``server`` is the JaatoServer for the new session.
                    # ``session_id`` is its unique identifier.
                    plugin = server.registry.get_plugin("environment")
                    if plugin and hasattr(plugin, 'register_aspect'):
                        plugin.register_aspect("my_aspect", self._handler)

        Extensions typically combine several hooks:

        1. **Session hooks** (``session_manager.add_session_hook``) — run
           after each session is initialized to wire per-session plugins.
        2. **WS connection interceptors** (``ws_server.set_connection_interceptor``)
           — route special WebSocket connections to custom handlers.
        3. **Custom environment aspects** (``env_plugin.register_aspect``)
           — add dynamic aspects to the ``get_environment`` tool.
        4. **Remote spawn handlers** (``subagent_plugin.register_remote_handler``)
           — enable remote subagent delegation.
        """
        eps = importlib.metadata.entry_points()
        if sys.version_info >= (3, 12):
            ext_eps = list(eps.select(group="jaato.extensions"))
        elif sys.version_info >= (3, 10):
            ext_eps = list(eps.select(group="jaato.extensions"))
        else:
            ext_eps = list(eps.get("jaato.extensions", []))

        if not ext_eps:
            return

        # Discover available plugins once so extensions can validate
        # profile definitions against the actual plugin set.
        from shared.plugins.registry import PluginRegistry
        _discovery_registry = PluginRegistry()
        _discovery_registry.discover()
        _available = frozenset(_discovery_registry.list_available())
        logger.info("Discovered %d available plugins for extensions", len(_available))

        # Discover GC plugins so extensions can list/introspect them.
        from shared.plugins.gc import discover_gc_plugins
        _gc_factories = discover_gc_plugins()
        _available_gc = frozenset(_gc_factories.keys())
        logger.info("Discovered %d available GC plugins for extensions", len(_available_gc))

        # Build the context namespace passed to every extension factory.
        context = _ExtensionContext(
            session_manager=self._session_manager,
            ws_server=self._ws_server,
            web_socket=self.web_socket,
            ipc_socket=self.ipc_socket,
            server_name=self._server_name,
            dashboard_port=self._dashboard_port,
            available_plugins=_available,
            plugin_registry=_discovery_registry,
            available_gc_plugins=_available_gc,
            gc_plugin_factories=_gc_factories,
        )

        for ep in ext_eps:
            try:
                factory = ep.load()
                ext = factory(context)
                self._extensions.append(ext)
                logger.info("Loaded daemon extension: %s", ep.name)
            except Exception:
                logger.warning(
                    "Failed to load extension %s", ep.name, exc_info=True,
                )

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




def daemonize(log_file: str = DEFAULT_LOG_FILE) -> None:
    """Daemonize the process (double-fork method on Unix, subprocess on Windows).

    On Windows, re-execs as ``python -m server`` so the daemon starts
    correctly regardless of whether the caller used ``python -m server``
    or the ``jaato-server`` console-script entry point.  The entry-point
    launcher (``.exe``) is not a valid Python script, so passing
    ``sys.argv[0]`` to the subprocess would fail.
    """
    if sys.platform == "win32":
        # Windows: use subprocess to start detached process
        import subprocess
        # Always re-exec via ``-m server`` so it works whether the caller
        # invoked ``python -m server --daemon`` or ``jaato-server --daemon``.
        # The console-script .exe is not a Python file, so we cannot just
        # re-use sys.argv[0].
        args = [sys.executable, "-m", "server"] + sys.argv[1:]
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


def check_pipe_exists(pipe_name: str) -> bool:
    """Check if a Windows named pipe already exists.

    Uses ``WaitNamedPipeW`` with a minimal timeout to probe without consuming
    a pipe instance.

    Args:
        pipe_name: Bare pipe name (e.g. ``"jaato"``) or full path
            (e.g. ``r"\\\\.\pipe\\jaato"``).

    Returns:
        True if the pipe exists, False otherwise.
    """
    import ctypes

    WINDOWS_PIPE_PREFIX = "\\\\.\\pipe\\"

    if pipe_name.startswith(WINDOWS_PIPE_PREFIX):
        pipe_path = pipe_name
    else:
        pipe_path = f"{WINDOWS_PIPE_PREFIX}{pipe_name}"

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.WaitNamedPipeW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong]
    kernel32.WaitNamedPipeW.restype = ctypes.c_int

    result = kernel32.WaitNamedPipeW(pipe_path, 1)
    if result:
        return True

    error = ctypes.get_last_error()
    ERROR_SEM_TIMEOUT = 121
    return error == ERROR_SEM_TIMEOUT


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

        # If the PID file points to our own process, it is a stale leftover
        # (e.g. a container restarted and got the same PID 1).  Clean it up.
        if pid == os.getpid():
            os.remove(pid_file)
            return None

        # Check if process exists
        if sys.platform == "win32":
            # Windows: use ctypes to check process.  Explicit argtypes /
            # restype are required so that the 64-bit HANDLE return value
            # is not silently truncated to a 32-bit c_int.
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.argtypes = [
                ctypes.c_ulong,   # DWORD dwDesiredAccess
                ctypes.c_int,     # BOOL  bInheritHandle
                ctypes.c_ulong,   # DWORD dwProcessId
            ]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
            )
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


def _resolve_ws_token(args) -> Optional[str]:
    """Decide which bearer token (if any) the WS server should require.

    Resolution order:

    1. ``--web-socket`` not set → ``None`` (no WS server, no token).
    2. ``--ws-unsafe-no-auth`` → ``None``, with a startup WARNING.
    3. ``--ws-token-file`` → read first non-empty line. The file must
       have mode 0600 or stricter (group/other readable is rejected) so
       a leaked token can't be silently grabbed by another local user.
    4. ``--ws-token`` → use as-is. Visible in process listings; the
       help text discourages it for production.
    5. Otherwise → auto-generate a 32-byte url-safe token and print it
       to stderr (Jupyter-style). The generated value is not persisted;
       restarting the daemon issues a fresh one.

    Conflicting flags exit with a clear error rather than picking a
    silent winner.
    """
    if not args.web_socket:
        return None

    if args.ws_unsafe_no_auth:
        if args.ws_token or args.ws_token_file:
            print(
                "Error: --ws-unsafe-no-auth cannot be combined with "
                "--ws-token / --ws-token-file",
                file=sys.stderr,
            )
            sys.exit(2)
        logger.warning(
            "WS bearer auth disabled (--ws-unsafe-no-auth). The "
            "WebSocket endpoint accepts ALL connections — only safe on "
            "a trusted network or behind an auth-terminating proxy."
        )
        return None

    if args.ws_token and args.ws_token_file:
        print(
            "Error: pass either --ws-token or --ws-token-file, not both",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.ws_token_file:
        path = Path(args.ws_token_file).expanduser()
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            print(f"Error: cannot read --ws-token-file {path}: {exc}", file=sys.stderr)
            sys.exit(2)
        # Reject world/group readable files. Same check ssh applies to
        # private keys — leaked tokens are private keys for the daemon.
        if sys.platform != "win32" and mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(
                f"Error: --ws-token-file {path} is group/other accessible "
                f"(mode {oct(mode & 0o777)}); restrict to 0600",
                file=sys.stderr,
            )
            sys.exit(2)
        token = path.read_text().splitlines()[0].strip() if path.read_text() else ""
        if not token:
            print(f"Error: --ws-token-file {path} is empty", file=sys.stderr)
            sys.exit(2)
        return token

    if args.ws_token:
        return args.ws_token

    # Auto-generate. Print once to stderr in a banner so it's visible
    # even when stdout is redirected. Not persisted to the restart
    # config — each daemon start gets a fresh token.
    token = secrets.token_urlsafe(32)
    banner_line = "─" * 64
    print(banner_line, file=sys.stderr)
    print("WS bearer token (auto-generated, not persisted):", file=sys.stderr)
    print(f"  {token}", file=sys.stderr)
    print("Pass to clients via:", file=sys.stderr)
    print("  Authorization: Bearer <token>   (Python / curl / proxies)", file=sys.stderr)
    print("  ws://host:port/?token=<token>   (browsers)", file=sys.stderr)
    print(
        "Set --ws-token-file PATH to use a stable token across restarts.",
        file=sys.stderr,
    )
    print(banner_line, file=sys.stderr)
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Jaato Server - Multi-client AI assistant backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
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
    parser.add_argument(
        "--ws-token",
        metavar="TOKEN",
        default=None,
        help="Bearer token clients must present in the WS Upgrade request. "
             "Discouraged for production (visible in process list); prefer "
             "--ws-token-file. Cannot be combined with --ws-unsafe-no-auth.",
    )
    parser.add_argument(
        "--ws-token-file",
        metavar="PATH",
        default=None,
        help="Path to a file containing the bearer token (one line). "
             "File must be mode 0600 or stricter.",
    )
    parser.add_argument(
        "--ws-unsafe-no-auth",
        action="store_true",
        help="Disable WS bearer auth entirely. Required to keep the legacy "
             "open-accept behaviour. Logs a WARNING at startup.",
    )
    parser.add_argument(
        "--socket-mode",
        metavar="MODE",
        default="666",
        help="Unix file permissions for the IPC socket in octal (default: 666). "
             "Use 660 to restrict to owner and group only.",
    )
    parser.add_argument(
        "--dashboard-port",
        metavar="PORT",
        type=int,
        default=None,
        help="TCP port for the dashboard and health HTTP endpoint. "
             "Disabled by default.",
    )
    parser.add_argument(
        "--server-name",
        metavar="NAME",
        default=None,
        help="Explicit server name for self-identification. "
             "When set, the server matches itself by name in servers.json "
             "instead of by address.",
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
        args.socket_mode = oct(config["socket_mode"])[2:] if "socket_mode" in config else "666"
        args.dashboard_port = config.get("dashboard_port")
        args.server_name = config.get("server_name")
        # Only --ws-token-file is persisted (path, not secret). Inline
        # tokens and auto-generated tokens are deliberately not saved.
        args.ws_token_file = config.get("ws_token_file")
        args.ws_token = None
        args.ws_unsafe_no_auth = bool(config.get("ws_unsafe_no_auth", False))

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

    # On Windows, also check whether the named pipe already exists.
    # The PID-file check can miss a running server (e.g. stale PID, ctypes
    # issues on 64-bit, or daemon that hasn't written its PID yet).
    if sys.platform == "win32" and args.ipc_socket:
        try:
            if check_pipe_exists(args.ipc_socket):
                print(
                    f"Error: Named pipe already exists "
                    f"(another server is listening)"
                )
                print(f"  Use 'python -m server --stop' to stop it")
                sys.exit(1)
        except Exception:
            pass  # Best-effort; ctypes may not be available

    # Daemonize if requested (skip if already daemonized on Windows)
    if args.daemon and not os.environ.get("JAATO_DAEMONIZED"):
        print(f"Starting Jaato server as daemon...")
        print(f"  PID file: {args.pid_file}")
        print(f"  Log file: {args.log_file}")
        if args.ipc_socket:
            print(f"  IPC socket: {args.ipc_socket}")
        if args.web_socket:
            print(f"  WebSocket: {args.web_socket}")
        if args.dashboard_port:
            print(f"  Dashboard: :{args.dashboard_port}")
        daemonize(args.log_file)

    # Reconfigure logging for daemon/background mode with rotating file handler
    # This ensures log files don't grow unbounded
    if args.daemon or os.environ.get("JAATO_DAEMONIZED"):
        configure_logging(log_file=args.log_file, verbose=args.verbose)

    # Resolve WS bearer token (only when --web-socket is configured).
    ws_token = _resolve_ws_token(args)

    # Create and run daemon
    socket_mode = int(args.socket_mode, 8)
    daemon = JaatoDaemon(
        ipc_socket=args.ipc_socket,
        web_socket=args.web_socket,
        pid_file=args.pid_file,
        log_file=args.log_file,
        socket_mode=socket_mode,
        dashboard_port=args.dashboard_port,
        server_name=args.server_name,
        ws_token=ws_token,
        ws_token_file=args.ws_token_file,
        ws_unsafe_no_auth=args.ws_unsafe_no_auth,
    )

    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
