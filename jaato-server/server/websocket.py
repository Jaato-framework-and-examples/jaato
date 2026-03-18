"""WebSocket Server for Jaato.

This module provides a WebSocket server that wraps JaatoServer,
enabling real-time bidirectional communication with multiple clients.

Usage:
    from server.websocket import JaatoWSServer

    server = JaatoWSServer(host="localhost", port=8080)
    await server.start()  # Blocks until shutdown
"""

import asyncio
import errno
import json
import logging
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set
import threading

try:
    import websockets
    from websockets import ServerConnection
    from websockets.exceptions import ConnectionClosed
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    ServerConnection = Any

from .core import JaatoServer
from .workspace_provisioner import WorkspaceProvisioner, ProvisionedWorkspace
from .apparmor import AppArmorManager
from .session_logging import set_logging_context, clear_logging_context
from jaato_sdk.events import (
    Event,
    EventType,
    ConnectedEvent,
    ErrorEvent,
    SystemMessageEvent,
    serialize_event,
    deserialize_event,
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    ClarificationBatchResponseEvent,
    ReferenceSelectionResponseRequest,
    StopRequest,
    CommandRequest,
    # Workspace management events
    WorkspaceListRequest,
    WorkspaceListEvent,
    WorkspaceCreateRequest,
    WorkspaceCreatedEvent,
    WorkspaceSelectRequest,
    ConfigStatusEvent,
    ConfigUpdateRequest,
    ConfigUpdatedEvent,
)
from .workspace_manager import WorkspaceManager
from .event_sink import EventSink


logger = logging.getLogger(__name__)

# Default path for servers.json (contains TLS config)
_SERVERS_JSON = Path.home() / ".jaato" / "servers.json"


def load_tls_context(servers_json: Optional[Path] = None) -> Optional[ssl.SSLContext]:
    """Build an ``ssl.SSLContext`` from the ``tls`` section in servers.json.

    Args:
        servers_json: Path to servers.json. Defaults to ``~/.jaato/servers.json``.

    Returns:
        An ``ssl.SSLContext`` configured for TLS server mode, or ``None``
        if the file is missing, has no ``tls`` section, or cert files don't exist.
    """
    path = servers_json or _SERVERS_JSON
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    tls = data.get("tls")
    if not tls:
        return None

    cert = Path(tls.get("cert", "")).expanduser()
    key = Path(tls.get("key", "")).expanduser()
    ca_cert = Path(tls.get("ca_cert", "")).expanduser()

    if not cert.exists() or not key.exists():
        logger.warning(
            "TLS cert (%s) or key (%s) not found — falling back to plain ws://",
            cert, key,
        )
        return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert), str(key))
    if ca_cert.exists():
        ctx.load_verify_locations(str(ca_cert))
    # Do not require client certificates — browser dashboard uses SSO,
    # not mTLS.  Peer gossip connections handle mTLS separately.
    ctx.verify_mode = ssl.CERT_NONE

    logger.info("TLS context loaded: cert=%s, ca=%s", cert, ca_cert)
    return ctx


class WSEventSinkAdapter:
    """Adapts ``JaatoWSServer`` to the ``EventSink`` protocol.

    ``CommandRouter`` and ``SessionManager`` call ``send_event()`` from
    synchronous model/session threads.  This adapter bridges that into
    the async WebSocket world by scheduling coroutines on the WS
    server's event loop via ``asyncio.run_coroutine_threadsafe()``.

    ``client_id`` values that don't belong to any connected WebSocket
    client are silently ignored, which is the contract of ``EventSink``
    and allows ``CompositeEventSink`` to fan-out safely.

    Per-client session/workspace state is tracked locally so the
    ``CommandRouter`` can query workspace paths for WS clients.
    """

    def __init__(self, ws_server: "JaatoWSServer") -> None:
        self._ws = ws_server
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        # Per-client tracking (mirrors IPC server's client fields)
        self._client_sessions: Dict[str, str] = {}        # client_id -> session_id
        self._client_workspaces: Dict[str, Optional[str]] = {}  # client_id -> workspace

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the event loop for thread-safe scheduling.

        Must be called from the async context (e.g., inside ``start()``).
        """
        self._event_loop = loop

    def send_event(self, client_id: str, event) -> None:
        """Send an event to a WebSocket client (thread-safe).

        Silently ignores unknown ``client_id`` values.
        """
        if client_id not in self._ws._clients:
            return
        if not self._event_loop:
            return

        async def _send():
            await self._ws._send_to_client(client_id, event)

        asyncio.run_coroutine_threadsafe(_send(), self._event_loop)

    def set_client_session(self, client_id: str, session_id: str) -> None:
        """Associate a client with a session."""
        self._client_sessions[client_id] = session_id

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        """Get the workspace path for a client."""
        return self._client_workspaces.get(client_id)

    def set_client_workspace(self, client_id: str, workspace_path: str) -> None:
        """Associate a workspace path with a client."""
        self._client_workspaces[client_id] = workspace_path

    def remove_client(self, client_id: str) -> None:
        """Clean up tracking state when a client disconnects."""
        self._client_sessions.pop(client_id, None)
        self._client_workspaces.pop(client_id, None)


def _get_server_version() -> str:
    """Read the jaato-server package version from installed metadata."""
    from importlib.metadata import version as pkg_version
    return pkg_version("jaato-server")


@dataclass
class ClientConnection:
    """Represents a connected client."""
    websocket: ServerConnection
    client_id: str
    connected_at: str
    subscriptions: Set[str]  # Event types to receive (empty = all)


class JaatoWSServer:
    """WebSocket server wrapping JaatoServer.

    Handles:
    - Multiple client connections
    - Event broadcasting
    - Request routing
    - Connection lifecycle

    Example:
        server = JaatoWSServer(host="localhost", port=8080)

        # Option 1: Run standalone
        asyncio.run(server.start())

        # Option 2: Start in background
        await server.start_background()
        # ... do other things ...
        await server.stop()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        workspace_root: Optional[str] = None,
        apparmor: Optional[bool] = None,
        default_template: str = "default",
        workspace_max_age: int = 86400,
        ssl_context: Optional[ssl.SSLContext] = None,
    ):
        """Initialize the WebSocket server.

        Args:
            host: Host to bind to.
            port: Port to bind to.
            workspace_root: Root directory for workspaces. Remote clients select
                from subdirectories; each workspace has its own .env file that
                determines the provider.
            apparmor: Enable AppArmor confinement for provisioned workspaces.
                ``None`` (default) auto-detects availability.  ``True`` requires
                AppArmor.  ``False`` disables confinement.
            default_template: Name of the default workspace template to copy
                when auto-provisioning (default: ``"default"``).
            workspace_max_age: Maximum age in seconds for provisioned workspaces
                before the reaper removes them (default: 86400 = 24h).
            ssl_context: Optional ``ssl.SSLContext`` for TLS. When provided the
                server listens on ``wss://`` instead of ``ws://``. Use
                :func:`load_tls_context` to build one from ``servers.json``.
        """
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets package required. Install with: pip install websockets"
            )

        self.host = host
        self.port = port
        self._ssl_context = ssl_context
        self._workspace_root = workspace_root

        # Connection interceptors registered by daemon extensions.
        # See ``set_connection_interceptor()`` for the protocol.
        self._interceptors: list = []

        # Server state
        self._server: Optional[Any] = None
        self._clients: Dict[str, ClientConnection] = {}
        self._client_counter = 0
        self._lock = asyncio.Lock()

        # Workspace manager (if workspace_root provided)
        self._workspace_manager: Optional[WorkspaceManager] = None

        # Workspace provisioner for auto-provisioning session workspaces
        self._provisioner: Optional[WorkspaceProvisioner] = None

        # AppArmor manager for per-session confinement
        self._apparmor: Optional[AppArmorManager] = None
        self._apparmor_mode = apparmor  # None=auto, True=required, False=disabled
        self._default_template = default_template
        self._workspace_max_age = workspace_max_age

        # Per-client provisioned workspace tracking
        self._client_provisioned: Dict[str, ProvisionedWorkspace] = {}

        # Core server (runs in thread)
        self._jaato_server: Optional[JaatoServer] = None
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()

        # Daemon-mode command routing.
        # When running as part of JaatoDaemon, the command router handles
        # session/tool/auth commands.  When None, the WS server handles
        # commands directly via JaatoServer (standalone mode).
        self._command_router = None  # Set by set_command_router()
        self._event_sink_adapter: Optional[WSEventSinkAdapter] = None

        # Shutdown flag
        self._shutdown_event = asyncio.Event()

    def set_command_router(self, router) -> None:
        """Set the daemon-mode command router.

        When set, incoming ``CommandRequest``, ``SendMessageRequest``, etc.
        are delegated to the ``CommandRouter`` instead of being handled
        directly by a per-WS ``JaatoServer``.

        Called by ``JaatoDaemon.start()`` after constructing the router.

        Args:
            router: ``CommandRouter`` instance.
        """
        self._command_router = router

    def get_event_sink_adapter(self) -> WSEventSinkAdapter:
        """Return (create if needed) the ``WSEventSinkAdapter`` for this server.

        The adapter implements ``EventSink`` and is registered with the
        ``CompositeEventSink`` in ``JaatoDaemon.start()``.
        """
        if self._event_sink_adapter is None:
            self._event_sink_adapter = WSEventSinkAdapter(self)
        return self._event_sink_adapter

    async def start(self) -> None:
        """Start the server and block until shutdown.

        This method:
        1. Initializes WorkspaceManager (workspace_root required)
        2. Starts the WebSocket server
        3. Runs event broadcasting loop
        4. Blocks until stop() is called

        JaatoServer initialization is deferred until a workspace is selected
        and configured by the client.
        """
        # Initialize workspace manager if root is provided.
        # When running in daemon mode without workspace_root, the WS server
        # still accepts peer gossip connections and IPC-attached client events.
        if self._workspace_root:
            self._workspace_manager = WorkspaceManager(self._workspace_root)
            self._workspace_manager.discover_workspaces()
            logger.info(f"Workspace mode enabled, root: {self._workspace_root}")

            # Initialize workspace provisioner for auto-provisioning
            self._provisioner = WorkspaceProvisioner(
                self._workspace_root,
                default_template=self._default_template,
            )

            # Initialize AppArmor manager
            self._apparmor = AppArmorManager(
                workspace_root=self._workspace_root,
            )
            if self._apparmor_mode is False:
                logger.info("AppArmor confinement disabled by configuration")
                self._apparmor = None
            elif self._apparmor_mode is True and not self._apparmor.is_available():
                logger.warning(
                    "AppArmor confinement required but not available — "
                    "workspace isolation will rely on directory sandboxing only"
                )
            elif self._apparmor and self._apparmor.is_available():
                logger.info("AppArmor confinement enabled")

            # Start workspace reaper
            def _on_workspace_reaped(session_id: str) -> None:
                if self._apparmor and self._apparmor.is_available():
                    self._apparmor.teardown_profile(session_id)

            self._provisioner.start_reaper(
                interval_seconds=3600,
                max_age_seconds=self._workspace_max_age,
                on_teardown=_on_workspace_reaped,
            )

        # Bind event loop for the WSEventSinkAdapter (thread-safe scheduling)
        if self._event_sink_adapter:
            self._event_sink_adapter.bind_loop(asyncio.get_running_loop())

        # Start WebSocket server
        try:
            serve_kwargs: Dict[str, Any] = dict(
                ping_interval=30,
                ping_timeout=10,
            )
            if self._ssl_context:
                serve_kwargs["ssl"] = self._ssl_context

            async with websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                **serve_kwargs,
            ) as server:
                self._server = server
                scheme = "wss" if self._ssl_context else "ws"
                logger.info(f"WebSocket server listening on {scheme}://{self.host}:{self.port}")

                # Run event broadcaster and wait for shutdown
                broadcast_task = asyncio.create_task(self._broadcast_loop())

                try:
                    await self._shutdown_event.wait()
                finally:
                    broadcast_task.cancel()
                    try:
                        await broadcast_task
                    except asyncio.CancelledError:
                        pass
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                raise OSError(
                    e.errno,
                    f"Cannot start WebSocket server: port {self.port} is already in use",
                ) from None
            raise

        # Cleanup
        if self._jaato_server:
            self._jaato_server.shutdown()

        logger.info("Server stopped")

    async def start_background(self) -> None:
        """Start the server in a background task.

        Returns immediately. Use stop() to shut down.
        """
        asyncio.create_task(self.start())
        # Give server time to start
        await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop the server gracefully."""
        self._shutdown_event.set()

        # Stop workspace reaper
        if self._provisioner:
            self._provisioner.stop_reaper()

        # Close all client connections
        async with self._lock:
            for client in list(self._clients.values()):
                try:
                    await client.websocket.close(1001, "Server shutting down")
                except Exception:
                    pass
            self._clients.clear()

    def _on_server_event(self, event: Event) -> None:
        """Callback from JaatoServer - queue event for broadcasting."""
        # This is called from a different thread (model thread)
        # Use asyncio.run_coroutine_threadsafe to safely queue
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                self._event_queue.put(event),
                loop
            )
        except RuntimeError:
            # No event loop running yet - server not started
            pass

    async def _broadcast_loop(self) -> None:
        """Continuously broadcast events to all clients."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for event with timeout (to check shutdown)
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                # Broadcast to all clients
                await self._broadcast(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    async def _broadcast(self, event: Event) -> None:
        """Broadcast an event to all connected clients."""
        if not self._clients:
            return

        message = serialize_event(event)

        async with self._lock:
            disconnected = []

            for client_id, client in self._clients.items():
                try:
                    await client.websocket.send(message)
                except ConnectionClosed:
                    disconnected.append(client_id)
                except Exception as e:
                    logger.error(f"Send error to {client_id}: {e}")
                    disconnected.append(client_id)

            # Remove disconnected clients
            for client_id in disconnected:
                del self._clients[client_id]
                logger.info(f"Client disconnected: {client_id}")

    def set_connection_interceptor(
        self,
        check: Callable,
        handler: Callable,
    ) -> None:
        """Register an interceptor for incoming WebSocket connections.

        Interceptors are evaluated in registration order **before** normal
        client handling.  When ``check(websocket)`` returns ``True``, the
        connection is handed off to ``handler(websocket)`` and never enters
        the regular client flow.

        This is the primary mechanism for daemon extensions (e.g., gossip
        clustering) to route special connections to custom handlers.

        Args:
            check: A callable ``(websocket) -> bool`` that inspects the
                inbound connection (e.g., checking request headers) and
                returns ``True`` if this interceptor should handle it.
            handler: An async callable ``(websocket) -> None`` that takes
                over the connection when ``check`` returns ``True``.
                The handler is responsible for the full connection lifecycle.

        Example (from a daemon extension's ``start()`` method)::

            ws_server.set_connection_interceptor(
                check=lambda ws: (
                    ws.request
                    and ws.request.headers.get("X-My-Header") == "true"
                ),
                handler=self._handle_special_connection,
            )
        """
        self._interceptors.append((check, handler))

    async def _handle_client(self, websocket: ServerConnection) -> None:
        """Handle a single client connection.

        Before normal client handling, registered interceptors are checked.
        If any interceptor's ``check`` returns ``True``, the connection is
        handed off to that interceptor's ``handler`` and this method returns.
        """
        # Check registered interceptors (e.g., peer gossip connections)
        for check, handler in self._interceptors:
            try:
                if check(websocket):
                    await handler(websocket)
                    return
            except Exception as exc:
                logger.error("Connection interceptor failed: %s", exc)
                return

        # Assign client ID
        async with self._lock:
            self._client_counter += 1
            client_id = f"client_{self._client_counter}"

            client = ClientConnection(
                websocket=websocket,
                client_id=client_id,
                connected_at=datetime.now(timezone.utc).isoformat(),
                subscriptions=set(),
            )
            self._clients[client_id] = client

        logger.info(f"Client connected: {client_id} from {websocket.remote_address}")

        # Send connected event
        try:
            server_info = {
                "client_id": client_id,
                "workspace_mode": self._workspace_manager is not None,
                "server_version": _get_server_version(),
            }

            if self._jaato_server:
                server_info["model_provider"] = self._jaato_server.model_provider
                server_info["model_name"] = self._jaato_server.model_name

            connected_event = ConnectedEvent(
                protocol_version="1.0",
                server_info=server_info,
            )
            await websocket.send(serialize_event(connected_event))

            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(client_id, message)

        except ConnectionClosed:
            pass
        except Exception as e:
            import traceback as _tb
            logger.error(f"Client error {client_id}: {e}\n{''.join(_tb.format_exception(e))}")
        finally:
            # Remove client
            async with self._lock:
                if client_id in self._clients:
                    del self._clients[client_id]
            # Clean up per-client state
            if self._workspace_manager:
                self._workspace_manager.remove_client(client_id)
            self._client_provisioned.pop(client_id, None)
            if self._event_sink_adapter:
                self._event_sink_adapter.remove_client(client_id)
            logger.info(f"Client disconnected: {client_id}")

    async def _handle_message(self, client_id: str, message: str) -> None:
        """Handle an incoming message from a client.

        Args:
            client_id: The client's ID.
            message: The JSON message.
        """
        try:
            event = deserialize_event(message)
        except json.JSONDecodeError as e:
            await self._send_error(client_id, f"Invalid JSON: {e}")
            return
        except ValueError as e:
            await self._send_error(client_id, str(e))
            return

        # --- Workspace management (transport-level, all modes) ---
        # Workspace negotiation is a transport concern, not a command concern.
        # These events are handled by the WS server regardless of whether a
        # CommandRouter is present (daemon mode) or not (standalone mode).
        is_workspace_request = isinstance(event, (
            WorkspaceListRequest,
            WorkspaceCreateRequest,
            WorkspaceSelectRequest,
            ConfigUpdateRequest,
        ))
        if is_workspace_request:
            await self._handle_workspace_event(client_id, event)
            return

        # --- External events (from <jaato-task> web component) ---
        from jaato_sdk.events import ExternalEventRequest
        if isinstance(event, ExternalEventRequest):
            await self._handle_external_event(client_id, event)
            return

        # --- Daemon-mode delegation ---
        # When running as part of JaatoDaemon, route session/command events
        # through the CommandRouter for unified dispatch across transports.
        if self._command_router:
            await self._handle_message_daemon(client_id, event)
            return

        # --- Standalone mode ---
        if not self._jaato_server:
            await self._send_error(client_id, "No workspace selected")
            return

        # Set logging context for session-specific log routing
        if self._jaato_server and self._workspace_manager:
            selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
            workspace_path = selected.path if selected else None
            session_env = self._jaato_server.get_all_session_env()
            # Use workspace name as session_id for WebSocket mode
            session_id = selected.name if selected else "websocket"
            set_logging_context(
                session_id=session_id,
                client_id=client_id,
                workspace_path=workspace_path,
                session_env=session_env,
            )

        # Route by event type
        if isinstance(event, SendMessageRequest):
            # Capture context for thread (ContextVars don't propagate to threads)
            if self._jaato_server and self._workspace_manager:
                selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
                ctx_workspace = selected.path if selected else None
                ctx_session_env = self._jaato_server.get_all_session_env()
                ctx_session_id = selected.name if selected else "websocket"
                ctx_client_id = client_id

                def run_with_context():
                    set_logging_context(
                        session_id=ctx_session_id,
                        client_id=ctx_client_id,
                        workspace_path=ctx_workspace,
                        session_env=ctx_session_env,
                    )
                    try:
                        self._jaato_server.send_message(
                            event.text,
                            event.attachments if event.attachments else None
                        )
                    finally:
                        clear_logging_context()

                await asyncio.get_event_loop().run_in_executor(None, run_with_context)
            else:
                # Fallback without context
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._jaato_server.send_message(
                        event.text,
                        event.attachments if event.attachments else None
                    )
                )

        elif isinstance(event, PermissionResponseRequest):
            self._jaato_server.respond_to_permission(
                event.request_id,
                event.response,
                edited_arguments=event.edited_arguments,
            )

        elif isinstance(event, ClarificationResponseRequest):
            self._jaato_server.respond_to_clarification(
                event.request_id,
                event.response
            )

        elif isinstance(event, ClarificationBatchResponseEvent):
            self._jaato_server.respond_to_clarification_batch(
                event.request_id,
                event.answers
            )

        elif isinstance(event, ReferenceSelectionResponseRequest):
            self._jaato_server.respond_to_reference_selection(
                event.request_id,
                event.response
            )

        elif isinstance(event, StopRequest):
            self._jaato_server.stop()

        elif isinstance(event, CommandRequest):
            result = self._jaato_server.execute_command(
                event.command,
                event.args
            )
            # HelpLines results are already emitted via HelpTextEvent, skip
            if not (isinstance(result, dict) and "_pager" in result):
                # Send result as system message
                await self._send_to_client(
                    client_id,
                    SystemMessageEvent(
                        message=json.dumps(result),
                        style="info",
                    )
                )

        else:
            await self._send_error(client_id, f"Unknown request type: {event.type}")

    async def _handle_external_event(self, client_id: str, event) -> None:
        """Handle an ``ExternalEventRequest`` from the web component.

        Publishes the external event on the session's ``EventBus`` so that
        agents subscribed via ``subscribeToEvents(event_types=['external_event'])``
        are woken via ``inject_prompt()``.

        Args:
            client_id: The WS client that sent the message.
            event: Deserialized ``ExternalEventRequest``.
        """
        from jaato_sdk.event_bus import Event as BusEvent, EventType as BusEventType

        # Resolve the session attached to this client
        session_id = ""
        if self._event_sink_adapter:
            session_id = self._event_sink_adapter._client_sessions.get(client_id, "")

        if not session_id:
            await self._send_error(client_id, "No session attached — cannot deliver external event")
            return

        # Find the session's EventBus.
        # In daemon mode: SessionManager → Session → JaatoServer → JaatoClient → session → runtime
        # In standalone mode: self._jaato_server → JaatoClient → session → runtime
        bus = None
        if self._command_router:
            sm = self._command_router._session_manager
            session_obj = sm.get_session(session_id) if hasattr(sm, 'get_session') else None
            if session_obj and session_obj.server and session_obj.server._jaato:
                jaato_session = session_obj.server._jaato.get_session()
                if jaato_session:
                    bus = jaato_session._runtime.event_bus
        elif self._jaato_server and self._jaato_server._jaato:
            jaato_session = self._jaato_server._jaato.get_session()
            if jaato_session:
                bus = jaato_session._runtime.event_bus

        if not bus:
            await self._send_error(client_id, "Session event bus not available")
            return

        # Build and publish the bus event
        timestamp = event.timestamp or datetime.now(timezone.utc).isoformat()

        bus_event = BusEvent(
            event_id=f"ext_{timestamp}",
            event_type=BusEventType.EXTERNAL_EVENT,
            timestamp=timestamp,
            source_agent="_external",
            payload={
                "source": "websocket",
                "event_type": event.name,
                "data": event.data,
            },
        )
        notified = bus.publish(bus_event)
        logger.debug(
            "External event '%s' published to session %s, notified %d subscriber(s)",
            event.name, session_id, notified,
        )

    async def _handle_workspace_event(self, client_id: str, event: Event) -> None:
        """Handle workspace management events (transport-level concern).

        Routes to the existing workspace handlers and, when a workspace is
        selected, bridges the resolved path to the ``WSEventSinkAdapter``
        so the ``CommandRouter`` can query it via ``get_client_workspace()``.
        """
        if isinstance(event, WorkspaceListRequest):
            await self._handle_workspace_list(client_id)
        elif isinstance(event, WorkspaceCreateRequest):
            await self._handle_workspace_create(client_id, event.name)
        elif isinstance(event, WorkspaceSelectRequest):
            await self._handle_workspace_select(client_id, event.name)
            # Bridge selected workspace path to the event sink adapter
            # so CommandRouter can resolve it via get_client_workspace()
            if self._event_sink_adapter and self._workspace_manager:
                selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
                if selected and selected.path:
                    self._event_sink_adapter.set_client_workspace(client_id, selected.path)
        elif isinstance(event, ConfigUpdateRequest):
            await self._handle_config_update(
                client_id, event.provider, event.model, event.api_key,
            )

    async def _handle_message_daemon(self, client_id: str, event: Event) -> None:
        """Handle a message when running in daemon mode.

        Delegates all events to the ``CommandRouter`` via
        ``run_in_executor`` so the async event loop is not blocked.

        The router uses the ``EventSink`` to send responses back to this
        client (via ``WSEventSinkAdapter``).

        Exceptions from the router are caught and sent back to the client
        as ``ErrorEvent`` so that a single bad request does not tear down
        the entire WebSocket connection.

        Args:
            client_id: The client's ID.
            event: The deserialized event.
        """
        from jaato_sdk.events import (
            ClientConfigRequest, CommandListRequest, CommandListEvent,
            ToolsRegisterClientRequest, ToolExecuteResultEvent,
        )

        # Handle CommandListRequest directly (same as IPC path)
        if isinstance(event, CommandListRequest):
            if self._command_router:
                commands = self._command_router.get_command_list()
                await self._send_to_client(
                    client_id, CommandListEvent(commands=commands),
                )
            return

        # Handle client-side tool registration
        if isinstance(event, ToolsRegisterClientRequest):
            # Buffer tools if no session yet (registered after session.new)
            if not self._event_sink_adapter or not self._event_sink_adapter._client_sessions.get(client_id):
                if not hasattr(self, '_pending_client_tools'):
                    self._pending_client_tools = {}
                self._pending_client_tools[client_id] = event.tools
                logger.info("Buffered %d client tools for %s (session pending)", len(event.tools), client_id)
            else:
                self._register_client_tools(client_id, event.tools)
            return

        # Handle client-side tool execution result
        if isinstance(event, ToolExecuteResultEvent):
            self._handle_tool_execute_result(client_id, event)
            return

        # Resolve session_id from the adapter's tracking
        session_id = ""
        if self._event_sink_adapter:
            session_id = self._event_sink_adapter._client_sessions.get(client_id, "")

        # Auto-provision a workspace for WS clients that don't have one
        # when they request a new session.  This mirrors the standalone WS
        # flow but plugs into the daemon command-router path.
        from jaato_sdk.events import CommandRequest
        if (isinstance(event, CommandRequest)
                and event.command.lower() == "session.new"
                and self._provisioner
                and self._event_sink_adapter
                and not self._event_sink_adapter.get_client_workspace(client_id)):
            import uuid as _uuid
            provisioned = await self.provision_workspace(
                session_id=f"ws_{_uuid.uuid4().hex[:8]}",
                client_id=client_id,
            )
            if provisioned:
                # Copy staged artifacts into the workspace if present.
                # The dashboard passes --artifacts <staging_id> in the
                # session.new args after uploading via /api/task/artifacts.
                args = event.args if hasattr(event, 'args') else []
                staging_id = None
                for i, arg in enumerate(args):
                    if arg == "--artifacts" and i + 1 < len(args):
                        staging_id = args[i + 1]
                        break
                if staging_id:
                    import shutil
                    import tempfile
                    from pathlib import Path as _Path
                    staging_dir = _Path(tempfile.gettempdir()) / f"jaato-artifacts-{staging_id}"
                    if staging_dir.is_dir():
                        ws_path = _Path(provisioned.path)
                        for item in staging_dir.iterdir():
                            dest = ws_path / item.name
                            if item.is_dir():
                                shutil.copytree(item, dest, dirs_exist_ok=True)
                            else:
                                shutil.copy2(item, dest)
                        shutil.rmtree(staging_dir, ignore_errors=True)
                        logger.info(
                            "Copied staged artifacts %s into workspace %s",
                            staging_id, provisioned.path,
                        )

                self._event_sink_adapter.set_client_workspace(
                    client_id, provisioned.path,
                )
                self._client_provisioned[client_id] = provisioned
                logger.info(
                    "Auto-provisioned workspace for WS client %s: %s",
                    client_id, provisioned.path,
                )

        try:
            # ClientConfigRequest must be processed synchronously (same as IPC)
            if isinstance(event, ClientConfigRequest):
                self._command_router.handle_request(client_id, session_id, event)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    self._command_router.handle_request,
                    client_id,
                    session_id,
                    event,
                )
            # After session.new completes, register any buffered client tools
            if (isinstance(event, CommandRequest)
                    and event.command.lower() == "session.new"
                    and hasattr(self, '_pending_client_tools')
                    and client_id in self._pending_client_tools):
                pending = self._pending_client_tools.pop(client_id)
                self._register_client_tools(client_id, pending)
                # (runtime refresh happens inside _register_client_tools)

        except Exception as exc:
            logger.error(
                "Command routing failed for client %s: %s", client_id, exc,
                exc_info=True,
            )
            await self._send_error(client_id, f"Internal error: {exc}")

    async def _send_to_client(self, client_id: str, event: Event) -> None:
        """Send an event to a specific client."""
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                try:
                    await client.websocket.send(serialize_event(event))
                except Exception as e:
                    logger.error(f"Send error to {client_id}: {e}")

    async def _send_error(self, client_id: str, error: str) -> None:
        """Send an error event to a client."""
        await self._send_to_client(
            client_id,
            ErrorEvent(error=error, error_type="RequestError")
        )

    # =========================================================================
    # Client-side tool execution
    # =========================================================================

    def _register_client_tools(self, client_id: str, tools: list) -> None:
        """Register client-provided tools as proxies in the session's registry.

        Each tool becomes a real tool in the model's tool list. When the model
        calls it, the executor sends a ``tool.execute_request`` to the WS
        client and waits for ``tool.execute_result``.
        """
        if not self._event_sink_adapter:
            return
        session_id = self._event_sink_adapter._client_sessions.get(client_id, "")
        if not session_id or not self._command_router:
            return

        session = self._command_router._session_manager.get_session(session_id)
        if not session or not session.server or not session.server.registry:
            return

        import threading
        from jaato_sdk.plugins.model_provider.types import ToolSchema
        from jaato_sdk.events import ToolExecuteRequestEvent

        registry = session.server.registry

        for tool_def in tools:
            tool_name = tool_def.get('name', '')
            if not tool_name:
                continue

            description = tool_def.get('description', '')
            parameters = tool_def.get('parameters', {})
            category = tool_def.get('category', '')
            timeout = tool_def.get('timeout', 30000) / 1000.0  # ms -> seconds
            auto_approve = tool_def.get('auto_approve', True)

            # Create a waiting mechanism for this client's responses
            if not hasattr(self, '_client_tool_waiters'):
                self._client_tool_waiters = {}

            # Build the proxy executor
            ws_server = self
            cid = client_id

            def make_executor(tname, tout):
                def executor(args):
                    import uuid
                    call_id = str(uuid.uuid4())[:8]

                    # Create a threading Event to wait for the result
                    waiter = threading.Event()
                    result_holder = {'result': None, 'error': None}
                    ws_server._client_tool_waiters[call_id] = (waiter, result_holder)

                    # Send execute request to client
                    import asyncio
                    if ws_server._event_sink_adapter and ws_server._event_sink_adapter._event_loop:
                        asyncio.run_coroutine_threadsafe(
                            ws_server._send_to_client(cid, ToolExecuteRequestEvent(
                                call_id=call_id,
                                agent_id='',
                                tool_name=tname,
                                tool_args=args,
                            )),
                            ws_server._event_sink_adapter._event_loop,
                        )

                    # Wait for result with timeout
                    if waiter.wait(timeout=tout):
                        if result_holder['error']:
                            return {'error': result_holder['error']}
                        return {'result': result_holder['result']}
                    else:
                        return {'error': f'Client tool {tname} timed out after {tout}s'}
                return executor

            executor = make_executor(tool_name, timeout)

            # Register as a tool in the session's registry
            schema = ToolSchema(
                name=tool_name,
                description=description + ' [client-provided]',
                parameters=parameters,
                category=category or None,
            )
            registry.register_core_tool(schema, executor, auto_approved=auto_approve)

            logger.info(
                "Registered client tool '%s' for client %s (timeout=%ss, auto_approve=%s)",
                tool_name, client_id, timeout, auto_approve,
            )

        # Refresh the runtime's tool schema list so the model sees new tools
        if session.server and session.server._jaato:
            runtime = session.server._jaato.get_runtime()
            if runtime and hasattr(runtime, '_all_tool_schemas'):
                existing = {s.name for s in runtime._all_tool_schemas}
                for name, schema in registry._core_tools.items():
                    if name not in existing:
                        runtime._all_tool_schemas.append(schema)
                logger.info("Refreshed runtime tool list for client %s", client_id)

    def _handle_tool_execute_result(self, client_id: str, event) -> None:
        """Route a tool execution result back to the waiting executor thread."""
        if not hasattr(self, '_client_tool_waiters'):
            return
        call_id = event.call_id
        entry = self._client_tool_waiters.pop(call_id, None)
        if not entry:
            logger.warning("No waiter for tool result call_id=%s", call_id)
            return
        waiter, result_holder = entry
        result_holder['result'] = event.result
        result_holder['error'] = event.error
        waiter.set()

    # =========================================================================
    # Workspace Management Handlers
    # =========================================================================

    async def _handle_workspace_list(self, client_id: str) -> None:
        """Handle workspace list request."""
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        workspaces = self._workspace_manager.list_workspaces()
        await self._send_to_client(
            client_id,
            WorkspaceListEvent(
                workspaces=[ws.to_dict() for ws in workspaces],
            )
        )

    async def _handle_workspace_create(self, client_id: str, name: str) -> None:
        """Handle workspace creation request."""
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        try:
            ws_info = self._workspace_manager.create_workspace(name)
            await self._send_to_client(
                client_id,
                WorkspaceCreatedEvent(workspace=ws_info.to_dict())
            )
        except ValueError as e:
            await self._send_error(client_id, str(e))

    async def _handle_workspace_select(self, client_id: str, name: str) -> None:
        """Handle workspace selection request.

        This selects the workspace and returns its configuration status.
        Per-client workspace tracking is used so multiple clients can
        select different workspaces simultaneously.
        """
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        try:
            ws_info = self._workspace_manager.select_workspace(name, client_id=client_id)
            config_status = self._workspace_manager.get_config_status(name)

            # Send config status to client
            await self._send_to_client(
                client_id,
                ConfigStatusEvent(
                    workspace=name,
                    configured=ws_info.configured,
                    provider=ws_info.provider,
                    model=ws_info.model,
                    available_providers=config_status.get("available_providers", []),
                    missing_fields=config_status.get("missing_fields", []),
                )
            )

        except ValueError as e:
            await self._send_error(client_id, str(e))

    async def provision_workspace(
        self,
        client_id: str,
        session_id: str,
        template: Optional[str] = None,
    ) -> Optional[ProvisionedWorkspace]:
        """Auto-provision an isolated workspace for a remote session.

        Creates a new workspace directory, applies a template, and
        optionally sets up AppArmor confinement.

        Args:
            client_id: The client requesting the workspace.
            session_id: Session identifier (used as workspace directory name).
            template: Template name to apply (default: server's default_template).

        Returns:
            The provisioned workspace, or None on failure.
        """
        if not self._provisioner:
            logger.warning("Cannot provision workspace: no provisioner configured")
            return None

        try:
            workspace = self._provisioner.provision(
                session_id=session_id,
                client_id=client_id,
                template=template,
            )
        except ValueError as e:
            logger.error("Workspace provision failed: %s", e)
            return None

        # Set up AppArmor confinement
        if self._apparmor and self._apparmor.is_available():
            self._apparmor.provision_profile(session_id, workspace.path)

        self._client_provisioned[client_id] = workspace
        return workspace

    def get_apparmor_wrappers(
        self,
        session_id: str,
    ) -> tuple:
        """Get AppArmor command wrappers for a session.

        Returns a (argv_wrapper, shell_wrapper) tuple suitable for
        passing to ``JaatoServer.set_apparmor_wrapper()``.  Both are
        None if AppArmor is not available or not configured.

        Args:
            session_id: Session identifier.

        Returns:
            Tuple of (argv_wrapper, shell_wrapper) callables, or
            (None, None) if AppArmor is not active.
        """
        if not self._apparmor or not self._apparmor.is_available():
            return None, None

        def argv_wrapper(cmd):
            return self._apparmor.wrap_command(session_id, cmd)

        def shell_wrapper(cmd):
            return self._apparmor.wrap_shell_command(session_id, cmd)

        return argv_wrapper, shell_wrapper

    async def _handle_config_update(
        self,
        client_id: str,
        provider: str,
        model: Optional[str],
        api_key: Optional[str],
    ) -> None:
        """Handle workspace configuration update request.

        After successfully updating the workspace config, initializes a
        ``JaatoServer`` for the workspace so the client can start sending
        messages.  If auto-provisioning is active, the workspace is
        provisioned first and AppArmor confinement is applied.
        """
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        selected = self._workspace_manager.get_selected_workspace(client_id=client_id)
        if not selected:
            await self._send_error(client_id, "No workspace selected")
            return

        try:
            result = self._workspace_manager.update_config(
                provider=provider,
                model=model,
                api_key=api_key,
                name=selected.name,
            )

            await self._send_to_client(
                client_id,
                ConfigUpdatedEvent(
                    workspace=result["workspace"],
                    provider=result["provider"],
                    model=result["model"],
                    success=result["success"],
                )
            )

            # Initialize JaatoServer now that the workspace is configured
            if result["success"]:
                await self._initialize_server_for_workspace(client_id, selected)

        except ValueError as e:
            await self._send_error(client_id, str(e))

    async def _initialize_server_for_workspace(
        self,
        client_id: str,
        workspace_info: Any,
    ) -> None:
        """Initialize a JaatoServer for the selected workspace.

        Creates the server from the workspace's ``.env`` file, initializes
        it in a background thread, and optionally applies AppArmor
        confinement.

        This is called after ``_handle_config_update()`` succeeds, meaning
        the workspace has a valid provider configuration.

        Args:
            client_id: The requesting client.
            workspace_info: The ``WorkspaceInfo`` for the selected workspace.
        """
        env_file = self._workspace_manager.get_env_file(workspace_info.name)
        if not env_file or not env_file.exists():
            await self._send_error(client_id, "Workspace .env file not found")
            return

        # Auto-provision an isolated workspace directory if provisioner is
        # configured.  This creates a session-specific subdirectory under
        # {root}/sessions/ with template contents and AppArmor confinement.
        session_id = workspace_info.name  # Use workspace name as session ID
        provisioned_ws = None
        if self._provisioner:
            provisioned_ws = await self.provision_workspace(
                client_id=client_id,
                session_id=session_id,
            )
            if provisioned_ws:
                # Use the provisioned workspace's .env instead
                provisioned_env = Path(provisioned_ws.path) / ".env"
                if provisioned_env.exists():
                    env_file = provisioned_env

        # Create JaatoServer
        server = JaatoServer(
            env_file=str(env_file),
            on_event=self._on_server_event,
            workspace_path=provisioned_ws.path if provisioned_ws else workspace_info.path,
        )

        # Initialize in executor (blocking call)
        def _init():
            return server.initialize()

        success = await asyncio.get_event_loop().run_in_executor(None, _init)

        if not success:
            await self._send_error(client_id, "Failed to initialize server")
            return

        self._jaato_server = server

        # Apply AppArmor confinement to CLI and interactive shell plugins
        if provisioned_ws:
            argv_wrapper, shell_wrapper = self.get_apparmor_wrappers(session_id)
            if argv_wrapper or shell_wrapper:
                server.set_apparmor_wrapper(
                    argv_wrapper=argv_wrapper,
                    shell_wrapper=shell_wrapper,
                )
                logger.info(
                    "AppArmor confinement applied to session %s",
                    session_id,
                )

        await self._send_to_client(
            client_id,
            SystemMessageEvent(
                message=f"Server initialized: {server.model_provider}/{server.model_name}",
                style="info",
            ),
        )

    # =========================================================================
    # Status Methods
    # =========================================================================

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._server is not None and not self._shutdown_event.is_set()

    def get_server_info(self) -> Dict[str, Any]:
        """Get server status information."""
        info = {
            "host": self.host,
            "port": self.port,
            "is_running": self.is_running,
            "client_count": self.client_count,
            "workspace_mode": self._workspace_manager is not None,
            "model_provider": self._jaato_server.model_provider if self._jaato_server else None,
            "model_name": self._jaato_server.model_name if self._jaato_server else None,
            "is_processing": self._jaato_server.is_processing if self._jaato_server else False,
        }

        if self._workspace_manager:
            selected = self._workspace_manager.get_selected_workspace()
            info["workspace_root"] = str(self._workspace_manager.workspace_root)
            info["selected_workspace"] = selected.name if selected else None

        if self._provisioner:
            info["provisioned_workspaces"] = len(self._provisioner.list_workspaces())
            info["available_templates"] = self._provisioner.list_templates()

        if self._apparmor:
            info["apparmor_available"] = self._apparmor.is_available()

        return info


# =============================================================================
# Standalone Entry Point
# =============================================================================

async def main():
    """Run the WebSocket server standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Jaato WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--workspace-root",
        metavar="PATH",
        required=True,
        help="Root directory for workspaces (remote clients select from subdirectories)",
    )
    parser.add_argument(
        "--apparmor",
        default=None,
        action="store_true",
        dest="apparmor",
        help="Enable AppArmor confinement (default: auto-detect)",
    )
    parser.add_argument(
        "--no-apparmor",
        action="store_false",
        dest="apparmor",
        help="Disable AppArmor confinement",
    )
    parser.add_argument(
        "--workspace-template",
        default="default",
        help="Default template for auto-provisioned workspaces (default: 'default')",
    )
    parser.add_argument(
        "--workspace-max-age",
        type=int,
        default=86400,
        help="Max age in seconds for provisioned workspaces (default: 86400)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    server = JaatoWSServer(
        host=args.host,
        port=args.port,
        workspace_root=args.workspace_root,
        apparmor=args.apparmor,
        default_template=args.workspace_template,
        workspace_max_age=args.workspace_max_age,
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
