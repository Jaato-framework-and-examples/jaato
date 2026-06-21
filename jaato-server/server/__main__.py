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
        socket_mode: int = 0o660,
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
            socket_mode: Unix file permissions for the IPC socket (default: 0o660,
                owner and group only — the IPC transport is unauthenticated).
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

        # Pre-warm runner template (pool PR 2).  Spawned in :meth:`start`
        # AFTER extensions load + before IPC accepts connections.  Sits
        # idle through the daemon's lifetime with runner-tier plugin
        # modules pre-imported, ready for pool slots (PR 3) to fork
        # from.  ``None`` when template-mode is disabled (no pool work
        # consumes the template yet in PR 2, so failure to spawn is
        # logged but non-fatal — sessions fall back to cold-spawn
        # session-mode).
        from server.runner_template import TemplateManager
        from server.runner_pool import PoolManager
        self._template_manager: TemplateManager = TemplateManager()
        # Pool manager (pool PR 3).  Asks the template to fork N pool
        # slots at daemon startup.  Slots sit idle through the daemon's
        # lifetime waiting for bootstrap envelopes (PR 4).  Pool size
        # configurable via ``JAATO_RUNNER_POOL_SIZE`` env var (default
        # 2).  Pool-empty fallback path = today's cold-spawn session-
        # mode (preserved through PR 4's flag-gated rollout).
        _pool_size_raw = os.environ.get("JAATO_RUNNER_POOL_SIZE", "2")
        try:
            _pool_size = int(_pool_size_raw)
        except ValueError:
            logger.warning(
                "JAATO_RUNNER_POOL_SIZE=%r is not an int; defaulting "
                "to 2", _pool_size_raw,
            )
            _pool_size = 2
        self._pool_manager: PoolManager = PoolManager(
            self._template_manager, target_size=_pool_size,
        )

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

        # Phase 3 §3.13: wire the IPC AppArmor + runner-spawn
        # dependencies onto the session manager.  Phase 2 §2.3
        # registered an inline pre-init hook from this method;
        # §3.13 relocates the logic into
        # ``SessionManager._provision_ipc_apparmor_and_spawn_runner``
        # called inline from ``_bootstrap_session``.  All this
        # method does now is pass through the WS-server reference
        # (for the workspace-overlap precedence check) and the
        # daemon's main asyncio loop (needed by AppArmorManager and
        # the runner-RPC start coroutine).
        self._wire_ipc_apparmor_dependencies()

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

            # ``JAATO_CGROUPS_ROOT`` env var overrides the default
            # ``/sys/fs/cgroup/jaato`` parent cgroup directory.  Useful
            # on hosts where the jaato cgroup tree is at a different
            # path (e.g., a dev host with subtree_control delegated
            # under ``/sys/fs/cgroup/jaato-test``).  Defaults to the
            # ``JaatoWSServer.cgroups_root`` default when unset.
            ws_server_kwargs = dict(
                host=host,
                port=port,
                workspace_root=_default_ws_root,
                ssl_context=ws_ssl_ctx,
                required_token=self._ws_token,
            )
            _cgroups_root_env = os.environ.get("JAATO_CGROUPS_ROOT", "").strip()
            if _cgroups_root_env:
                ws_server_kwargs["cgroups_root"] = _cgroups_root_env
                logger.info(
                    "JAATO_CGROUPS_ROOT=%s — using non-default cgroup parent",
                    _cgroups_root_env,
                )
            self._ws_server = JaatoWSServer(**ws_server_kwargs)
            # Pool PR 4: thread the daemon's PoolManager into the WS
            # server so the WS apparmor pre-init hook can route session
            # bootstrap through pre-warm slots when the session opts out
            # of AppArmor (disable_confine=True) AND doesn't need a
            # cgroup_attach.  The reference is a plain attribute write
            # because the WS server's pre-init hook reads it via
            # ``getattr(ws_server, "_pool_manager_ref", None)`` — no
            # constructor parameter churn required.
            self._ws_server._pool_manager_ref = self._pool_manager
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
            # Give the IPC transport a SessionManager handle so it can register
            # client-provided ("host") tools onto a session's registry (mirrors
            # the WS transport's _command_router access).
            self._ipc_server._session_manager = self._session_manager
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

        # Pool PR 5b: claim subreaper role for the daemon.  Pool slots
        # are template-children (forked from the template, no exec).
        # By default, when the template dies, slots get re-parented to
        # PID 1 (init/systemd) — daemon loses visibility, can't
        # ``waitpid`` them.  ``prctl(PR_SET_CHILD_SUBREAPER, 1)``
        # claims the role for the daemon: orphaned descendants
        # re-parent to the daemon instead of init.
        #
        # This unlocks: (1) clean ``waitpid(slot_pid)`` in pool
        # teardown (no more ``ChildProcessError`` swallow);
        # (2) the template watchdog (also PR 5b) — when template
        # dies, slots come back to the daemon so the watchdog can
        # reap + drain + respawn cleanly.
        #
        # Linux-only.  Non-Linux hosts log a debug message and skip.
        # Failure is non-fatal — pool still works via the existing
        # ChildProcessError-swallow path, just slightly less tidy.
        self._configure_subreaper()

        # Pool PR 2: spawn the pre-warm runner template.  The template
        # subprocess imports runner-tier plugin modules at startup so
        # pool slots (PR 3) can fork from it and inherit warm imports
        # — eliminating the per-session 50s plugin-discovery cost
        # observed on v62 step 6.
        try:
            self._template_manager.spawn()
        except Exception as exc:  # noqa: BLE001 — boundary surface
            logger.warning(
                "Failed to spawn runner template (pool PR 2): %s; "
                "sessions will continue to cold-spawn via session-mode",
                exc,
            )

        # Pool PR 3: fork N idle slots from the template.  Slots sit
        # idle through the daemon's lifetime waiting for bootstrap
        # envelopes (PR 4 routed sessions through the pool).
        #
        # Pool PR 5c: ``TemplateManager.spawn`` now blocks for the
        # template's READY signal (sent after plugin discovery
        # completes), so no time.sleep is needed here — the template
        # is guaranteed-ready by the time we hit ``is_alive`` below.
        if self._template_manager.is_alive():
            try:
                self._pool_manager.spawn_initial_slots()
            except Exception as exc:  # noqa: BLE001 — boundary surface
                logger.warning(
                    "Failed to spawn pool slots (pool PR 3): %s; "
                    "sessions will continue to cold-spawn via session-"
                    "mode (pool empty)",
                    exc,
                )
            # Pool PR 4: start the background replenishment thread.
            # Keeps the pool topped up between session bootstrap calls;
            # without it a cascade with target_size=2 cold-spawns every
            # step past the second.  Idempotent + clean-stop on
            # ``shutdown_all`` (which the daemon shutdown path calls).
            try:
                self._pool_manager.start_replenishment()
            except Exception as exc:  # noqa: BLE001 — boundary surface
                logger.warning(
                    "Failed to start pool replenishment thread "
                    "(pool PR 4): %s; pool will not refill after "
                    "sessions consume slots", exc,
                )

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

        # Pool PR 3: tear down idle pool slots BEFORE template.  Slots
        # were forked from template; killing template first orphans
        # slots without their SHUTDOWN command.  Order matters here.
        try:
            self._pool_manager.shutdown_all()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            logger.warning(
                "Pool teardown raised: %s; daemon will exit anyway",
                exc,
            )

        # Pool PR 2: tear down the runner template as part of daemon
        # exit.  The template is daemon-lifetime — alive across all
        # sessions, never shut down on "idle" or "no active sessions".
        # This call fires ONLY here, when the daemon itself is
        # exiting.  Polite SHUTDOWN command first, SIGTERM after 5s
        # timeout.  Idempotent.
        try:
            self._template_manager.shutdown()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            logger.warning(
                "Template teardown raised: %s; daemon will exit anyway",
                exc,
            )

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
    # IPC AppArmor opt-in wiring (Phase 3 §3.13 — relocated)
    # ------------------------------------------------------------------

    def _wire_ipc_apparmor_dependencies(self) -> None:
        """Wire the IPC AppArmor + runner-spawn dependencies onto the
        session manager (Phase 3 §3.13).

        The actual provisioning logic now lives in
        :meth:`SessionManager._provision_ipc_apparmor_and_spawn_runner`,
        called inline from
        :meth:`SessionManager._bootstrap_session`.  This method
        captures the daemon's main asyncio loop (needed by
        :class:`AppArmorManager` for confined-worker dispatch and by
        :func:`server.runner_spawn.spawn_session_runner` for
        ``RunnerRPCClient.start()``) and passes through the WS-server
        reference so the relocated method's workspace-under-WS-root
        precedence check still works.

        Phase 2 §2.3 registered an inline pre-init hook here.
        §3.13 collapses that indirection — the helper is now a
        normal SessionManager method, not a closure passed through
        a hook list.
        """
        try:
            daemon_loop = asyncio.get_running_loop()
        except RuntimeError:
            daemon_loop = None
        if self._session_manager is not None:
            self._session_manager.set_apparmor_dependencies(
                ws_server=self._ws_server,
                daemon_loop=daemon_loop,
                # Pool PR 4: thread the daemon's pool manager so IPC
                # sessions can route through pre-warm slots when the
                # session opts out of AppArmor.  WS sessions get the
                # pool manager through the WS server attribute set in
                # ``start()`` (see the ``_pool_manager_ref`` write).
                pool_manager=self._pool_manager,
            )

    # ------------------------------------------------------------------
    # Pool PR 5b: subreaper setup for pool slot re-parenting
    # ------------------------------------------------------------------

    def _configure_subreaper(self) -> None:
        """Set ``PR_SET_CHILD_SUBREAPER`` on the daemon process.

        Pool slots are template-children (forked from the template
        subprocess, no exec).  When the template dies, slots get
        orphaned — by default re-parented to PID 1 (init / systemd),
        out of the daemon's reach for ``waitpid`` + lifecycle
        management.

        ``prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)`` claims that role
        for the daemon: any descendant of the daemon (template,
        slots, slots' children) that gets orphaned will re-parent to
        the daemon instead of to init.

        After this call:

        - Daemon teardown calls ``os.waitpid(slot_pid)`` directly
          (slots are template-children; if template dies first or
          we're in the brief window between template's death and
          init's reap, slots have re-parented to daemon).
        - The template watchdog (also PR 5b) can detect template
          death, reap any orphaned slots that came back to the
          daemon, and respawn the template + refill the pool.

        Linux-only — uses Linux-specific ``prctl(2)``.  Other
        platforms log debug and skip.  Failure is non-fatal: the
        pool still works via the existing ``ChildProcessError``
        catch-and-ignore in ``shutdown_all``; daemon just can't
        recover cleanly from template death (slots leak to init).
        """
        if sys.platform != "linux":
            logger.debug(
                "subreaper setup: skipped on non-Linux platform (%s) — "
                "Linux-only prctl call", sys.platform,
            )
            return

        import ctypes
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError as exc:
            logger.warning(
                "subreaper setup: libc.so.6 unavailable (%s); "
                "skipping prctl call.  Pool will continue working but "
                "without subreaper cleanup on template death.", exc,
            )
            return

        # PR_SET_CHILD_SUBREAPER is option 36 in <linux/prctl.h>.
        # ``prctl(int option, ulong arg2, ulong arg3, ulong arg4,
        # ulong arg5)`` — arg2=1 to enable, args 3-5 unused.
        PR_SET_CHILD_SUBREAPER = 36
        rc = libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
        if rc != 0:
            err = ctypes.get_errno()
            logger.warning(
                "subreaper setup: prctl(PR_SET_CHILD_SUBREAPER, 1) "
                "returned rc=%d errno=%d.  Pool will continue working "
                "but without subreaper cleanup on template death.",
                rc, err,
            )
            return

        logger.info(
            "subreaper setup: daemon (pid=%d) claimed PR_SET_CHILD_SUBREAPER; "
            "orphaned descendants will re-parent to this process instead "
            "of init.", os.getpid(),
        )

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
    """Spawn the per-session runner subprocess + dispatch the bootstrap
    envelope (IPC path).

    Phase 3 §3.12 + §7c step 2: composes the shared
    :func:`server.runner_spawn.spawn_session_runner` (process spawn) +
    :func:`server.runner_spawn.dispatch_bootstrap_envelope` (RPC dispatch
    of ``session.bootstrap``).  The WS hook calls the same two
    helpers in sequence — see ``server/websocket.py``'s apparmor
    pre-init hook.

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

    Raises on spawn failure.  The bootstrap-dispatch leg is failure-
    tolerant (logs WARNING + returns).  The caller (the session
    hook) catches spawn failures and downgrades to
    ``sandbox_mode = "soft"`` per the §4.6 fallback contract.
    """
    from server.runner_spawn import (
        spawn_session_runner,
        dispatch_bootstrap_envelope,
    )

    spawn_session_runner(
        server=server,
        session_id=session_id,
        workspace_path=workspace_path,
        profile_name=profile_name,
        daemon_loop=daemon_loop,
    )

    dispatch_bootstrap_envelope(
        server=server,
        session_id=session_id,
        workspace_path=workspace_path,
        profile_name=profile_name,
    )


# ``_build_session_envelope`` was relocated to ``server.runner_spawn``
# in §7c step 2 (where it sits next to ``dispatch_bootstrap_envelope``,
# the only caller).  Re-export under the legacy name so existing test
# imports + any external callers keep working without churn.  Marked
# private; remove the alias when the broader §7c seat-flip is done.
from server.runner_spawn import (  # noqa: E402, F401 — back-compat re-export
    build_session_envelope as _build_session_envelope,
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
    r"""Check if a Windows named pipe already exists.

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


def _pid_on_socket(socket_path: Optional[str]) -> Optional[int]:
    """PID of the jaato daemon bound to a Unix socket (the authoritative signal).

    Fallback for when the pidfile is missing/stale but a daemon is still
    listening — e.g. an ephemeral client-autostarted daemon that began teardown
    (removed its pidfile) but survived.  Without this, ``--stop`` trusts the
    pidfile alone and reports "not running" while orphaning a live daemon (the
    gap peer-reported 2026-06-20: ``jaato-server --stop`` missing an
    autostarted daemon on the default socket).

    Verifies the owner is actually a jaato server process (cmdline contains
    ``-m server`` / ``jaato-server``) before returning it, so a process that
    merely happens to bind the path is never targeted.  Returns None on
    Windows / no listener / permission / non-jaato owner.
    """
    if sys.platform == "win32" or not socket_path:
        return None
    if not os.path.exists(socket_path):
        return None
    try:
        import psutil
        for conn in psutil.net_connections(kind="unix"):
            if conn.laddr == socket_path and conn.pid:
                try:
                    cmdline = " ".join(psutil.Process(conn.pid).cmdline())
                except Exception:
                    continue
                if "-m server" in cmdline or "jaato-server" in cmdline:
                    return conn.pid
    except Exception:
        return None
    return None


def check_running(
    pid_file: str = DEFAULT_PID_FILE,
    ipc_socket: Optional[str] = None,
) -> Optional[int]:
    """Check if a server is already running — pidfile first, socket fallback.

    Resolves via the pidfile; if that yields nothing (missing / stale / points
    at a dead PID) but a daemon is still bound to ``ipc_socket``, falls back to
    the authoritative socket signal (:func:`_pid_on_socket`).  This is what lets
    ``--stop`` / ``--status`` find a client-autostarted daemon whose ephemeral
    teardown removed the pidfile but survived.

    Returns:
        The PID if running, None otherwise.
    """
    pid = _check_running_via_pidfile(pid_file)
    if pid is not None:
        return pid
    return _pid_on_socket(ipc_socket) if ipc_socket else None


def _check_running_via_pidfile(pid_file: str = DEFAULT_PID_FILE) -> Optional[int]:
    """Check the pidfile alone for a running server (no socket fallback).

    Returns:
        The PID if the pidfile names a live process, None otherwise.
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


def _snapshot_daemon_descendants(pid: int) -> List[int]:
    """Snapshot every descendant PID of the daemon process tree.

    Walked BEFORE the daemon is killed, so when ``--stop`` later
    reaps any orphans (e.g. runner subprocesses whose upstream
    session was killed mid-cascade and pruned from the daemon's
    in-memory session→runner registry), it can target THIS
    daemon's descendants specifically — never a sibling daemon's
    runners on the same host.

    Closes the orphan-bleed gap diagnosed by peer 7:1 on 2026-06-07:
    after ``jaato-server --stop`` cleanly reaped a daemon + 2
    current runners, a 3rd runner (PID 1985236, etime 1h17m,
    orphaned from session ``20260607_215922`` killed mid-cascade)
    stayed alive POSTing to the upstream vLLM endpoint until
    manually killed.  See
    ``project_backlog_jaato_server_stop_orphan_runner_reaper.md``
    for the full diagnosis + the (a) structural follow-up that
    eventually obviates this defensive sweep.

    psutil's ``children(recursive=True)`` enumerates BOTH:
    - Direct children spawned by the daemon (normal case).
    - Orphans re-parented to the daemon via the daemon's
      ``prctl(PR_SET_CHILD_SUBREAPER, 1)`` startup invariant
      documented at ``jaato/CLAUDE.md:493`` — the kernel makes
      the daemon the new parent of any descendant whose original
      parent died.

    Returns an empty list when psutil can't see the daemon
    (already-dead, permission denied) — caller treats absence as
    "no orphans to sweep" rather than an error.
    """
    try:
        import psutil
    except ImportError:
        return []
    try:
        proc = psutil.Process(pid)
        return [child.pid for child in proc.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _reap_orphan_descendants(pids: List[int]) -> int:
    """Send SIGTERM + SIGKILL escalation to each PID still alive
    after the daemon exit.  Captured via
    :func:`_snapshot_daemon_descendants` BEFORE the daemon was
    killed.

    Returns the count of processes that were actually killed
    (skipped if already dead).  Best-effort: per-PID errors are
    swallowed silently because the daemon is already down and
    failed reaps just leak processes — they don't corrupt state.
    """
    import time as _time

    killed = 0
    for pid in pids:
        try:
            # Skip if already dead.
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue
        # Wait up to 2s for graceful exit (orphan runners may have
        # in-flight HTTP requests to flush).
        gone = False
        for _ in range(20):
            _time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                gone = True
                break
        if not gone:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        killed += 1
    return killed


def stop_server(
    pid_file: str = DEFAULT_PID_FILE,
    ipc_socket: Optional[str] = None,
) -> bool:
    """Stop a running server.

    Snapshots the daemon's descendant PIDs BEFORE sending SIGTERM
    so orphan runners — runner subprocesses whose upstream session
    was killed mid-cascade and pruned from the in-memory
    session→runner registry — can be reaped after the daemon
    exits.  See :func:`_snapshot_daemon_descendants` for the
    diagnosis behind this defensive sweep.

    Returns:
        True if stopped, False if not running.
    """
    pid = check_running(pid_file, ipc_socket)
    if not pid:
        return False

    try:
        import time

        if sys.platform == "win32":
            # Windows: use taskkill or TerminateProcess.  Orphan-reap
            # sweep is Unix-only for now (psutil works on Windows but
            # the runner subprocess model is POSIX-fork-specific; if a
            # Windows port lands, add the snapshot + reap here too).
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
            # Snapshot descendant PIDs BEFORE killing the daemon.
            # After SIGTERM the kernel re-parents to PID 1, which
            # would defeat any post-kill child enumeration.
            descendants = _snapshot_daemon_descendants(pid)
            os.kill(pid, signal.SIGTERM)
            # Wait for daemon to exit
            daemon_gone = False
            for _ in range(50):  # 5 seconds timeout
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    daemon_gone = True
                    break
            if not daemon_gone:
                # Force kill the daemon if it didn't exit gracefully.
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # Orphan sweep: reap any captured descendants the daemon
            # didn't reach itself.  Always runs — even on graceful
            # daemon exit — because the daemon's reaper walks an
            # in-memory session→runner registry that gets pruned when
            # sessions are killed mid-cascade.  Orphans whose session
            # entry was pruned stay alive past the daemon's own reap.
            if descendants:
                reaped = _reap_orphan_descendants(descendants)
                if reaped:
                    print(
                        f"  Reaped {reaped} orphan runner "
                        f"{'process' if reaped == 1 else 'processes'} "
                        f"left behind by killed mid-cascade sessions",
                        file=sys.stderr,
                    )
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
        default="660",
        help="Unix file permissions for the IPC socket in octal (default: 660, "
             "owner and group only). The IPC transport is unauthenticated, so "
             "any principal that can open the socket can fully drive the agent. "
             "Pass 666 to opt into world-accessible (e.g. cross-user containers "
             "on a trusted host).",
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
        pid = check_running(args.pid_file, args.ipc_socket or DEFAULT_SOCKET_PATH)
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
        if stop_server(args.pid_file, args.ipc_socket or DEFAULT_SOCKET_PATH):
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
        _probe = args.ipc_socket or DEFAULT_SOCKET_PATH
        pid = check_running(args.pid_file, _probe)
        if pid:
            print(f"Stopping server (PID: {pid})...")
            if not stop_server(args.pid_file, _probe):
                print("Error: Failed to stop server")
                sys.exit(1)
            print("Server stopped")
        else:
            print("Server was not running")

        # Apply saved config
        args.ipc_socket = config.get("ipc_socket")
        args.web_socket = config.get("web_socket")
        args.log_file = config.get("log_file", DEFAULT_LOG_FILE)
        args.socket_mode = oct(config["socket_mode"])[2:] if "socket_mode" in config else "660"
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

    # Check if already running.  Pass the socket being started so a daemon
    # already bound to it is detected even when its pidfile is missing/stale
    # — preventing a duplicate daemon (the Unix analog of the Windows
    # named-pipe fallback below).
    pid = check_running(args.pid_file, args.ipc_socket)
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
