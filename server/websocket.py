"""WebSocket Server for Jaato.

This module provides a WebSocket server that wraps JaatoServer,
enabling real-time bidirectional communication with multiple clients.

Usage:
    from server.websocket import JaatoWSServer

    server = JaatoWSServer(host="localhost", port=8080)
    await server.start()  # Blocks until shutdown
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Set
import threading

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosed
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketServerProtocol = Any

from .core import JaatoServer
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


logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Represents a connected client."""
    websocket: WebSocketServerProtocol
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
    ):
        """Initialize the WebSocket server.

        Args:
            host: Host to bind to.
            port: Port to bind to.
            workspace_root: Root directory for workspaces. Remote clients select
                from subdirectories; each workspace has its own .env file that
                determines the provider.
        """
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets package required. Install with: pip install websockets"
            )

        self.host = host
        self.port = port
        self._workspace_root = workspace_root

        # Server state
        self._server: Optional[Any] = None
        self._clients: Dict[str, ClientConnection] = {}
        self._client_counter = 0
        self._lock = asyncio.Lock()

        # Workspace manager (if workspace_root provided)
        self._workspace_manager: Optional[WorkspaceManager] = None

        # Core server (runs in thread)
        self._jaato_server: Optional[JaatoServer] = None
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()

        # Shutdown flag
        self._shutdown_event = asyncio.Event()

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
        # WebSocket server requires workspace_root for remote clients
        if not self._workspace_root:
            raise RuntimeError(
                "WebSocket server requires --workspace-root. "
                "Remote clients select workspaces from the server."
            )

        self._workspace_manager = WorkspaceManager(self._workspace_root)
        self._workspace_manager.discover_workspaces()
        logger.info(f"Workspace mode enabled, root: {self._workspace_root}")

        # Start WebSocket server
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
        ) as server:
            self._server = server
            logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")

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

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a single client connection."""
        # Assign client ID
        async with self._lock:
            self._client_counter += 1
            client_id = f"client_{self._client_counter}"

            client = ClientConnection(
                websocket=websocket,
                client_id=client_id,
                connected_at=datetime.utcnow().isoformat(),
                subscriptions=set(),
            )
            self._clients[client_id] = client

        logger.info(f"Client connected: {client_id} from {websocket.remote_address}")

        # Send connected event
        try:
            server_info = {
                "client_id": client_id,
                "workspace_mode": self._workspace_manager is not None,
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
            logger.error(f"Client error {client_id}: {e}")
        finally:
            # Remove client
            async with self._lock:
                if client_id in self._clients:
                    del self._clients[client_id]
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

        # Workspace requests work without JaatoServer
        is_workspace_request = isinstance(event, (
            WorkspaceListRequest,
            WorkspaceCreateRequest,
            WorkspaceSelectRequest,
            ConfigUpdateRequest,
        ))

        if not self._jaato_server and not is_workspace_request:
            await self._send_error(client_id, "No workspace selected")
            return

        # Set logging context for session-specific log routing
        # WebSocket server uses a single workspace at a time
        if self._jaato_server and self._workspace_manager:
            selected = self._workspace_manager.get_selected_workspace()
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
                selected = self._workspace_manager.get_selected_workspace()
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

        # Workspace management requests
        elif isinstance(event, WorkspaceListRequest):
            await self._handle_workspace_list(client_id)

        elif isinstance(event, WorkspaceCreateRequest):
            await self._handle_workspace_create(client_id, event.name)

        elif isinstance(event, WorkspaceSelectRequest):
            await self._handle_workspace_select(client_id, event.name)

        elif isinstance(event, ConfigUpdateRequest):
            await self._handle_config_update(
                client_id,
                event.provider,
                event.model,
                event.api_key,
            )

        else:
            await self._send_error(client_id, f"Unknown request type: {event.type}")

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
        """
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        try:
            ws_info = self._workspace_manager.select_workspace(name)
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

    async def _handle_config_update(
        self,
        client_id: str,
        provider: str,
        model: Optional[str],
        api_key: Optional[str],
    ) -> None:
        """Handle workspace configuration update request."""
        if not self._workspace_manager:
            await self._send_error(client_id, "Workspace mode not enabled")
            return

        selected = self._workspace_manager.get_selected_workspace()
        if not selected:
            await self._send_error(client_id, "No workspace selected")
            return

        try:
            result = self._workspace_manager.update_config(
                provider=provider,
                model=model,
                api_key=api_key,
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

        except ValueError as e:
            await self._send_error(client_id, str(e))

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
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
