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
        ipc_server: The ``JaatoIPCServer`` instance if ``--ipc-socket``
            was passed, or ``None`` otherwise.  Extensions can call
            ``ipc_server.register_message_handler(type, callback)`` to
            handle custom message types on IPC connections, mirroring
            the WS surface.  Most extensions should prefer the
            unified :meth:`register_message_handler` on this context
            object — it fans out to both transports in one call.
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
        "session_manager", "ws_server", "ipc_server", "web_socket",
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
        ipc_server=None,
    ):
        self.session_manager = session_manager
        self.ws_server = ws_server
        self.ipc_server = ipc_server
        self.web_socket = web_socket
        self.ipc_socket = ipc_socket
        self.server_name = server_name
        self.dashboard_port = dashboard_port
        self.available_plugins = available_plugins
        self.plugin_registry = plugin_registry
        self.available_gc_plugins = available_gc_plugins
        self.gc_plugin_factories = gc_plugin_factories or {}

    def broadcast_event(self, event) -> None:
        """Broadcast a daemon-wide event to every connected IPC + WS client.

        Used by extensions for events that don't belong to a specific
        session — currently the jaato-premium reactor framework's
        HandoffGate transitions (``gate.announced`` / ``gate.released`` /
        ``gates.snapshot``).

        Thin wrapper over ``self.session_manager.broadcast_event(...)``.
        Provided as a stable extension-facing API so the underlying
        implementation can move without breaking extensions.
        """
        self.session_manager.broadcast_event(event)

    def emit_to_client(self, client_id: str, event) -> None:
        """Send an event to a single client, addressed by ``client_id``.

        Targeted-emit complement to :meth:`broadcast_event`.  Used by
        extension message handlers (registered via
        :meth:`register_message_handler`) that need to reply to the
        sender — e.g. a ``gates.list`` request expects only its sender
        to receive the snapshot, not every connected client.

        Routes through the SessionManager's per-client emit path so
        the event fans out to the right transport (IPC or WS) for the
        ``client_id``.  Silently no-ops when the ``client_id`` is
        unknown (caller doesn't have to track client lifecycles).
        """
        try:
            self.session_manager._emit_to_client(client_id, event)
        except Exception:
            # Don't let a client-targeted send failure propagate into
            # extension code — handlers shouldn't have to wrap their
            # own emit calls in try/except.  The session manager
            # already logs send failures internally.
            pass

    def register_message_handler(self, message_type: str, handler) -> None:
        """Register a custom message-type handler for both transports.

        Convenience wrapper that fans out a single registration to
        both ``ws_server`` and ``ipc_server`` (whichever is present),
        so an extension can wire a verb like ``gates.list`` once and
        have it reach both IPC and WS clients.

        Handler signature is the transport-agnostic shape used by
        :meth:`JaatoIPCServer.register_message_handler`:

            async def handler(message: dict, client_id: str, user: Optional[str]) -> None

        Reply via :meth:`emit_to_client` for targeted responses or
        :meth:`broadcast_event` for fan-out.

        For WS-only handlers that need raw-socket access (rare —
        e.g. streaming protocols that bypass the typed-event bus),
        register against ``ws_server.register_message_handler``
        directly with the WS-specific
        ``(ws, raw_dict, user, client_id)`` signature.

        Args:
            message_type: The ``type`` field value to match
                (e.g., ``"gates.list"``).
            handler: Async callback with the signature above.
        """
        if self.ipc_server is not None:
            self.ipc_server.register_message_handler(message_type, handler)
        if self.ws_server is not None:
            # The WS path uses a different handler signature
            # ``(ws, raw_dict, user, client_id)``.  Wrap the unified
            # handler so the same callable works on both sides:
            # we emulate the unified shape by ignoring ``ws``
            # (handlers reply via ``ctx.emit_to_client`` rather than
            # raw ``ws.send``) and reordering the kwargs.
            async def _ws_adapter(ws, raw, user, client_id, _h=handler):
                await _h(raw, client_id, user)
            self.ws_server.register_message_handler(message_type, _ws_adapter)


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

        # Phase 2 (confined runner): the daemon never confines its own
        # threads to per-session AppArmor profiles — confinement happens
        # in the per-session runner subprocess (see server/runner/).
        # The previous SafeThreadPoolExecutor + per-task defensive-reset
        # hook was load-bearing only because daemon threads ran tools
        # under apparmor_confine; after Phase 2 the daemon is unconfined
        # and the asyncio default executor is fine.

        # Write PID and config files early so that clients checking
        # _check_server_running() see this daemon before initialization
        # completes (avoids race where TUI auto-starts a second server).
        self._write_pid()
        self._write_config()

        # Initialize session manager
        self._session_manager = SessionManager()

        # Register the IPC-aware AppArmor session hook.  Fires for any
        # session whose creating client opted in via
        # ``ClientConfigRequest.apparmor=True`` (the default is
        # ``False``, preserving today's IPC behavior).  WS-provisioned
        # sessions are still handled by the WS server's own hook —
        # this one stays out of their lane by checking that the
        # session's workspace is NOT under the WS workspace_root.
        self._register_ipc_apparmor_hook()

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
            self._ipc_server._on_client_disconnect = self._command_router.handle_client_disconnect
        if self._ws_server:
            self._ws_server.set_command_router(self._command_router)

        # Wire composite sink as session manager's event callback
        self._session_manager.set_event_callback(composite_sink.send_event)
        # Also wire broadcast — daemon-wide events (HandoffGate transitions)
        # fan out across all transports via CompositeEventSink.broadcast_event.
        self._session_manager.set_broadcast_callback(composite_sink.broadcast_event)

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

        # Periodic health checks (server 0.6.54+) — currently inotify
        # pressure.  Cheap proc/self/fd scan every 5 minutes; warns
        # operators before the kernel limit is hit and sessions start
        # failing with Errno 24.  Linux-only; check is a silent no-op
        # on other platforms.
        health_task = asyncio.create_task(self._run_health_checks())
        health_task.add_done_callback(_on_task_done)
        tasks.append(health_task)

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

    async def _run_health_checks(self) -> None:
        """Periodic health-check loop (server 0.6.54+).

        Runs ``check_inotify_pressure()`` every 5 minutes; logs a
        WARNING when usage crosses 80% of the kernel limit so
        operators see EAGAIN-class failures coming before
        ``inotify_init1()`` returns ``EMFILE`` and sessions fail to
        spawn.  The check is a silent no-op on non-Linux platforms.

        Cancellable via ``self._shutdown_event``: ``wait_for`` with
        the 300s timeout returns immediately on shutdown signal,
        otherwise raises ``TimeoutError`` (i.e. interval elapsed)
        and the loop runs the check.
        """
        from .health_check import check_inotify_pressure

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=300.0,
                )
                return  # shutdown signaled
            except asyncio.TimeoutError:
                pass  # interval elapsed; run the check
            try:
                msg = check_inotify_pressure()
            except Exception:
                logger.exception("Health check failed; continuing")
                continue
            if msg:
                logger.warning("Health: %s", msg)

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
    # IPC AppArmor opt-in hook
    # ------------------------------------------------------------------

    def _register_ipc_apparmor_hook(self) -> None:
        """Register a session hook that confines IPC sessions when opted in.

        Fires for every session created on this daemon.  Skips sessions
        whose creating client did not set
        ``ClientConfigRequest.apparmor=True`` (the default).  Skips
        WS-provisioned sessions (handled by the WS server's own hook).
        When AppArmor is unavailable on the host (non-Linux, kernel
        module not loaded, ``apparmor_parser`` missing), the session
        falls back to ``sandbox_mode = "soft"`` and the hook emits a
        ``SystemMessageEvent`` (style ``"warning"``, prefix
        ``[apparmor]``) to the client so the user can see at a glance
        that confinement was requested but not applied — silent
        fallbacks make security regressions invisible.

        The AppArmor manager is constructed lazily on first opt-in so
        deployments that never opt in pay zero startup cost.  The
        manager is process-singleton: subsequent opt-ins reuse it.

        Profile cleanup is handled per-session via
        ``apparmor.teardown_profile`` from the session-end path; the
        WS workspace reaper handles WS-provisioned sessions, while
        IPC sessions tear down at session.end (see
        ``SessionManager.end_session``).
        """
        from server.apparmor import AppArmorManager

        # Captured by the closure: lazily-allocated AppArmor manager.
        # Wrap in a list for write-from-closure (avoids ``nonlocal`` —
        # this method is on a class but the hook doesn't need
        # ``self``; if the manager is reused across daemon restarts
        # that's fine because it's stateless beyond cached
        # availability detection).
        apparmor_holder: list = [None]

        # Capture the daemon's main asyncio loop here (this method
        # runs from ``async start()``).  Passed to AppArmorManager so
        # that mutations triggered from confined worker threads can be
        # dispatched onto this loop for execution in an unconfined
        # context — eliminates the daemon-restart workaround that was
        # otherwise needed when ``selectReferences`` (or any other
        # confined-worker-driven AppArmor mutation) hit EACCES on the
        # ``/etc/apparmor.d/jaato/`` file write.
        try:
            daemon_loop = asyncio.get_running_loop()
        except RuntimeError:
            daemon_loop = None

        def _ipc_apparmor_pre_init_hook(
            server,
            session_id: str,
            workspace_path: Optional[str],
            client_id: Optional[str],
        ) -> None:
            """Pre-init hook: provision AppArmor profile + spawn runner.

            Phase 2 task 2.3 (post-rebase): converted from a session
            hook to a pre-init hook so the runner is up and
            ``registry.runner_rpc`` is wired BEFORE
            ``server.initialize()`` runs.  Plugins that read
            ``runner_rpc`` from the registry at configure-time
            (``set_plugin_registry``) now find it set; Phase 2 cli
            still reads lazily at execute-time, but Phase 3 plugins
            may rely on the configure-time hand-off.

            Stashes the planned sandbox_mode on the JaatoServer (via
            ``server._planned_sandbox_mode``); ``_create_session_impl``
            reads it back when building the ``Session`` object.

            Skipped for sessions without ``client_id`` (the
            ``_load_session`` disk-restore path) — Phase 2 explicitly
            defers ``_load_session_impl``, ``run_ephemeral_session``,
            and the standalone WS bootstrap to Phase 3 (per the
            review feedback on the §2.3 wiring scope).
            """
            from jaato_sdk.events import SystemMessageEvent

            sm = self._session_manager
            if sm is None:
                return

            def _notify(message: str, style: str) -> None:
                """Surface an apparmor-status line to the client terminal.

                Hook fires pre-init, BEFORE the session id is mapped
                to a client (``_client_to_session``) and BEFORE the
                Session record exists in ``_sessions``, so we route
                via ``_emit_to_client(client_id, ...)`` directly.
                Falls through to the daemon log when no client_id is
                available (non-IPC bootstrap paths).
                """
                logger.info("[apparmor] %s", message)
                if client_id is None:
                    return
                try:
                    sm._emit_to_client(client_id, SystemMessageEvent(
                        message=f"[apparmor] {message}",
                        style=style,
                    ))
                except Exception:
                    # Emit failure must not break session creation.
                    logger.warning(
                        "Failed to emit apparmor status event for %s",
                        session_id, exc_info=True,
                    )

            # Phase 2 scope: only IPC apparmor opt-in triggers spawn.
            # Non-client-driven bootstrap paths (loaded-from-disk,
            # ephemeral, standalone WS) → Phase 3.
            if client_id is None:
                return

            client_config = sm._client_config.get(client_id, {})
            if not client_config.get("apparmor"):
                return

            if not workspace_path:
                _notify(
                    "requested but session has no workspace_path — "
                    "running unconfined",
                    style="warning",
                )
                return

            # If a WS server is running and this workspace is under
            # its workspace_root, the WS hook owns confinement for
            # this session.  Don't double-provision.
            if self._ws_server is not None:
                ws_root = getattr(self._ws_server, "_workspace_root", None)
                if ws_root:
                    try:
                        ws_root_real = os.path.realpath(ws_root)
                        sess_real = os.path.realpath(workspace_path)
                        if (
                            sess_real == ws_root_real
                            or sess_real.startswith(ws_root_real + os.sep)
                        ):
                            return
                    except OSError:
                        pass

            # Lazy-init the AppArmor manager.  ``workspace_root`` on
            # the manager is unused for profile rendering today (the
            # template doesn't reference it — sibling-deny is implicit
            # via AppArmor's default-deny policy), so passing the
            # session's own workspace_path is fine and keeps the
            # interface uniform across IPC + WS use.
            if apparmor_holder[0] is None:
                # ``daemon_loop`` was captured above when this method
                # was invoked from ``async start()``.  Passing it lets
                # AppArmorManager dispatch confined-worker mutations
                # back onto the unconfined main loop.
                apparmor_holder[0] = AppArmorManager(
                    workspace_root=workspace_path,
                    loop=daemon_loop,
                )

            apparmor = apparmor_holder[0]
            if not apparmor.is_available():
                server._planned_sandbox_mode = "soft"
                _notify(
                    "requested but AppArmor is unavailable on this "
                    "host (non-Linux, kernel module not loaded, or "
                    "apparmor_parser missing) — running unconfined",
                    style="warning",
                )
                return

            config_root = client_config.get("config_root")
            env_file = client_config.get("env_file")
            if not apparmor.provision_profile(
                session_id,
                workspace_path,
                config_root=config_root,
                env_file=env_file,
            ):
                server._planned_sandbox_mode = "soft"
                _notify(
                    "profile provisioning failed (see daemon log) — "
                    "running unconfined",
                    style="warning",
                )
                return

            # Phase 2 (confined runner): the kernel-level profile is
            # provisioned above (apparmor.provision_profile) but the
            # daemon's own threads stay unconfined.  Per-session
            # confinement is applied by the runner subprocess, which
            # self-confines via aa_change_profile against this
            # already-loaded profile.  See docs/design/per_session_confined_runner.md
            # §4.6 (daemon apparmor-state constraint).
            server._planned_sandbox_mode = "apparmor"

            # Spawn the per-session runner subprocess.  The cli plugin
            # stub (and Phase 3's runner-tier plugins) will route
            # tool.execute calls to this runner via
            # ``registry.runner_rpc`` (the registry-attribute pattern
            # picked in plan §5.4).
            try:
                _spawn_session_runner(
                    server=server,
                    session_id=session_id,
                    workspace_path=workspace_path,
                    profile_name=apparmor.get_profile_name(session_id),
                    daemon_loop=daemon_loop,
                )
            except Exception as exc:  # noqa: BLE001 — boundary
                # If the runner fails to spawn, downgrade to soft mode
                # rather than killing the session — Phase 3 will make
                # this strict (apparmor=on requires runner=on), but
                # Phase 2 keeps the fallback path so a host-level
                # apparmor mishap doesn't break IPC sessions outright.
                server._planned_sandbox_mode = "soft"
                _notify(
                    f"runner spawn failed ({type(exc).__name__}: {exc}) "
                    "— falling back to in-process tool execution; "
                    "session is NOT kernel-confined",
                    style="warning",
                )
                logger.exception(
                    "runner spawn failed for session %s", session_id,
                )
                return

            _notify(
                f"profile provisioned (workspace={workspace_path}, "
                f"config_root={config_root or '(none)'}); runner spawned",
                style="info",
            )

        self._session_manager.add_pre_initialize_hook(_ipc_apparmor_pre_init_hook)

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
            ipc_server=self._ipc_server,
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




def _spawn_session_runner(
    *,
    server,
    session_id: str,
    workspace_path: str,
    profile_name: str,
    daemon_loop,
) -> None:
    """Spawn the per-session runner subprocess and wire its RPC handle
    onto the JaatoServer.

    Called from the IPC AppArmor session hook AFTER
    ``apparmor.provision_profile`` returns successfully.  The runner
    self-confines via ``aa_change_profile`` against the just-loaded
    profile (see :mod:`server.runner.bootstrap`).

    Args:
        server: The session's ``JaatoServer`` instance.
        session_id: Session identifier (passed via env to the runner).
        workspace_path: Session workspace; used both as the runner's
            cwd and as the prefix for the per-session log file path
            (plan §5.1).
        profile_name: AppArmor profile name (already loaded in the
            kernel).
        daemon_loop: The daemon's main asyncio loop — needed to run
            ``RunnerRPCClient.start()`` since it's async.

    Raises on any failure.  The caller (the session hook) catches
    and downgrades to ``sandbox_mode = "soft"`` per the §4.6 fallback
    contract.
    """
    import asyncio
    import os

    from server.runner_spawner import RunnerSpawner
    from server.runner_rpc_client import RunnerRPCClient

    if daemon_loop is None:
        raise RuntimeError(
            "_spawn_session_runner: daemon loop unavailable; cannot "
            "start RunnerRPCClient"
        )

    spawner = RunnerSpawner()

    log_path: Optional[str] = None
    if workspace_path:
        log_dir = os.path.join(workspace_path, ".jaato", "logs")
        log_path = os.path.join(log_dir, f"runner-{session_id}.log")

    spawned = spawner.spawn(
        profile_name=profile_name,
        session_id=session_id,
        workspace_path=workspace_path,
        log_path=log_path,
    )

    rpc = RunnerRPCClient(
        spawned.parent_socket,
        runner_pid=spawned.pid,
        loop=daemon_loop,
    )

    fut = asyncio.run_coroutine_threadsafe(rpc.start(), daemon_loop)
    fut.result(timeout=10.0)

    server.set_runner_rpc(rpc, spawned)
    logger.info(
        "runner spawned for session %s: pid=%d profile=%s log=%s",
        session_id, spawned.pid, profile_name, log_path or "(inherited)",
    )

    # Phase 3 §3.3c part 2: when JAATO_RUNNER_HOSTS_SESSION is set,
    # also send the session.bootstrap envelope so the runner-side
    # JaatoSession host gets exercised.  Daemon-side JaatoSession is
    # NOT removed in this commit — that's part 3 (the seat-flip).
    # Coexistence is intentional during the §3.3 review window.
    if os.environ.get("JAATO_RUNNER_HOSTS_SESSION", "").strip().lower() in (
        "1", "true", "yes",
    ):
        try:
            envelope = _build_session_envelope(
                server=server,
                session_id=session_id,
                workspace_path=workspace_path,
                profile_name=profile_name,
            )
            result = rpc.bootstrap_session_threadsafe(envelope, timeout=30.0)
            logger.info(
                "runner session.bootstrap acknowledged for %s: %s",
                session_id, result,
            )
        except Exception as exc:  # noqa: BLE001 — boundary surface
            # Bootstrap failure does NOT kill the session — the
            # daemon-side JaatoSession is still authoritative until
            # part 3.  Log loudly so the operator notices the runner
            # host isn't actually populated.
            logger.warning(
                "runner session.bootstrap failed for %s: %s — "
                "daemon-side JaatoSession remains authoritative",
                session_id, exc, exc_info=True,
            )


def _build_session_envelope(
    *,
    server,  # JaatoServer (forward-typed; importing the real type
             # creates a cycle through server/core.py).
    session_id: str,
    workspace_path: Optional[str],
    profile_name: str,
) -> "SessionInitEnvelope":
    """Build a :class:`SessionInitEnvelope` from a pre-init JaatoServer.

    Phase 3 §3.3c part 2.  Reads the resolved profile from the
    server (set in ``SessionManager._create_session_impl`` before
    the pre-init hooks fire) and constructs the envelope the
    runner-side host needs.

    Defaults applied for fields that would otherwise be empty:
    - ``provider_name`` → ``"anthropic"`` (the framework default).
    - ``model_name`` → ``""`` (the runner-side validate stage will
      reject; surfaced loudly).

    Args:
        server: The :class:`JaatoServer` instance — has ``_profile``
            set to a :class:`SubagentProfile` if a profile was
            resolved.  ``None`` for inline-spec / no-profile sessions.
        session_id: Stable session identifier.
        workspace_path: Session's workspace; ``None`` for headless.
        profile_name: AppArmor profile name (informational; the
            envelope's ``profile_name`` field carries it for
            audit attribution).

    Returns:
        A :class:`SessionInitEnvelope` ready for
        :meth:`RunnerRPCClient.bootstrap_session_threadsafe`.
    """
    from shared.session_envelope import SessionInitEnvelope

    profile = getattr(server, "_profile", None)
    provider_name = ""
    model_name = ""
    plugin_specs: list = []
    plugin_configs_dict: dict = {}
    preloaded: set = set()
    system_instructions: Optional[str] = None
    gc_dict: Optional[dict] = None
    env_overrides: dict = {}

    if profile is not None:
        provider_name = getattr(profile, "provider", None) or ""
        model_name = getattr(profile, "model", None) or ""
        # plugins is a list of clean names (strings); preloaded_plugins
        # is a set of names; plugin_configs is a dict.
        names = list(getattr(profile, "plugins", []) or [])
        preloaded = set(getattr(profile, "preloaded_plugins", set()) or set())
        plugin_configs_dict = dict(
            getattr(profile, "plugin_configs", {}) or {},
        )
        for name in names:
            entry: dict = {"name": name, "preload": name in preloaded}
            cfg = plugin_configs_dict.get(name)
            if cfg:
                entry["config"] = dict(cfg)
            plugin_specs.append(entry)
        system_instructions = getattr(profile, "system_instructions", None)
        gc_obj = getattr(profile, "gc", None)
        if gc_obj is not None:
            # GCProfileConfig has ``type`` + ``config`` (dict).  Flatten
            # to a single dict for the envelope.
            gc_type = getattr(gc_obj, "type", None)
            gc_config = getattr(gc_obj, "config", None) or {}
            if gc_type:
                gc_dict = {"type": gc_type, **dict(gc_config)}
        env_overrides = dict(getattr(profile, "env", {}) or {})

    # Provider fallback — the JaatoRuntime default is "google_genai"
    # but Phase 3's runner-tier plugins are most-tested against
    # anthropic.  When neither the profile nor the env explicitly
    # specifies, fall back to anthropic which has the broadest
    # plugin compat coverage.
    if not provider_name:
        provider_name = "anthropic"

    return SessionInitEnvelope(
        session_id=session_id,
        workspace_path=workspace_path,
        profile_name=profile_name,
        provider_name=provider_name,
        model_name=model_name,
        plugins=plugin_specs,
        system_instructions=system_instructions,
        agent_id="main",
        gc=gc_dict,
        agent_params={},
        config_root=getattr(server, "config_root", None),
        env_overrides=env_overrides,
    )


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


DEFAULT_WS_TOKEN_FILE = str(Path.home() / ".jaato" / "ws.token")


def _load_token_file(path: Path) -> str:
    """Read a bearer token from ``path``, enforcing 0600-or-stricter mode.

    Used by both the explicit ``--ws-token-file`` path and the default
    ``~/.jaato/ws.token`` path. Exits (code 2) on any failure rather
    than falling back — a leaked token is a security defect, not a
    recoverable error.
    """
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        print(f"Error: cannot read WS token file {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    # Reject world/group readable files. Same check ssh applies to
    # private keys — leaked tokens are private keys for the daemon.
    if sys.platform != "win32" and mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(
            f"Error: WS token file {path} is group/other accessible "
            f"(mode {oct(mode & 0o777)}); restrict to 0600",
            file=sys.stderr,
        )
        sys.exit(2)
    content = path.read_text()
    token = content.splitlines()[0].strip() if content else ""
    if not token:
        print(f"Error: WS token file {path} is empty", file=sys.stderr)
        sys.exit(2)
    return token


def _create_default_token_file(path: Path) -> str:
    """Generate a fresh token and persist it at ``path`` with mode 0600.

    Used on first WS-bound daemon start when no explicit flag was given.
    The parent directory is created if missing; the file is written
    atomically via ``os.open`` with ``O_CREAT|O_EXCL`` so a concurrent
    process racing to create the same file can't clobber our token.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(token + "\n")
    except Exception:
        # If writing failed after opening, remove the empty file so the
        # next start can retry instead of erroring on a zero-byte token.
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return token


def _resolve_ws_token(args) -> Optional[str]:
    """Decide which bearer token (if any) the WS server should require.

    Resolution order:

    1. ``--web-socket`` not set → ``None`` (no WS server, no token).
    2. ``--ws-unsafe-no-auth`` → ``None``, with a startup WARNING.
    3. ``--ws-token`` → use as-is (discouraged — visible in process list).
    4. ``--ws-token-file PATH`` → read that path, enforce mode 0600.
    5. **Default:** read/create ``~/.jaato/ws.token``. If the file
       exists, use it (mode 0600 check applies). If it doesn't exist,
       generate a fresh 32-byte token, write it with 0600, and use it.
       This gives a stable token across restarts without any CLI flag
       and matches Jupyter/SSH-style "well-known config file" UX.

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

    if args.ws_token:
        return args.ws_token

    token_path = Path(args.ws_token_file).expanduser() if args.ws_token_file \
        else Path(DEFAULT_WS_TOKEN_FILE)

    if token_path.exists():
        return _load_token_file(token_path)

    # Default path doesn't exist yet → first-time WS daemon start on
    # this host. Create the file so subsequent starts and local clients
    # see a stable value.
    if args.ws_token_file:
        # User explicitly named a file that doesn't exist — that's a
        # config error, not a request to create one.
        print(
            f"Error: --ws-token-file {token_path} does not exist",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        token = _create_default_token_file(token_path)
    except OSError as exc:
        print(
            f"Error: could not create default WS token file "
            f"{token_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    banner_line = "─" * 64
    print(banner_line, file=sys.stderr)
    print(f"WS bearer token created at {token_path} (mode 0600)", file=sys.stderr)
    print("Clients on this host can read it from the same path.", file=sys.stderr)
    print("For cross-host clients, pass the token via:", file=sys.stderr)
    print("  Authorization: Bearer <token>   (Python / curl / proxies)", file=sys.stderr)
    print("  ws://host:port/?token=<token>   (browsers)", file=sys.stderr)
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
        help=f"Path to a file containing the bearer token (one line). "
             f"File must be mode 0600 or stricter. "
             f"Defaults to {DEFAULT_WS_TOKEN_FILE} — created automatically "
             f"on first WS-bound daemon start if it does not exist.",
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
