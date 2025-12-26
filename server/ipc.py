"""IPC Server using Unix Domain Sockets.

This module provides a local IPC server for fast, secure communication
with local clients (rich-client, IDE extensions, etc.).

Unix domain sockets are:
- Faster than TCP (no network stack overhead)
- Inherently secure (filesystem permissions)
- Local-only (no remote access)

Usage:
    from server.ipc import JaatoIPCServer

    server = JaatoIPCServer(socket_path="/tmp/jaato.sock")
    await server.start()
"""

import asyncio
import json
import logging
import os
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from .events import (
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
    StopRequest,
    CommandRequest,
)


logger = logging.getLogger(__name__)


# Message framing: 4-byte length prefix (big-endian) + JSON payload
HEADER_SIZE = 4
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB max


@dataclass
class IPCClientConnection:
    """Represents a connected IPC client."""
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    client_id: str
    session_id: Optional[str]
    connected_at: str


class JaatoIPCServer:
    """IPC server using Unix domain sockets.

    Provides fast local communication for:
    - rich-client TUI
    - IDE extensions
    - CLI tools
    - Local scripts

    Protocol:
    - Each message is framed: 4-byte length (big-endian) + JSON payload
    - Same event protocol as WebSocket (server/events.py)
    - Supports session multiplexing (client specifies session_id)
    """

    def __init__(
        self,
        socket_path: str = "/tmp/jaato.sock",
        on_session_request: Optional[Callable[[str, str, Event], None]] = None,
    ):
        """Initialize the IPC server.

        Args:
            socket_path: Path to the Unix domain socket.
            on_session_request: Callback for session requests.
                Called with (client_id, session_id, event).
        """
        self.socket_path = socket_path
        self._on_session_request = on_session_request

        # Server state
        self._server: Optional[asyncio.Server] = None
        self._clients: Dict[str, IPCClientConnection] = {}
        self._client_counter = 0
        self._lock = asyncio.Lock()

        # Event queue for broadcasting
        self._event_queues: Dict[str, asyncio.Queue[Event]] = {}

        # Shutdown flag
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the IPC server.

        Creates the Unix domain socket and listens for connections.
        """
        # Remove existing socket file
        socket_file = Path(self.socket_path)
        if socket_file.exists():
            socket_file.unlink()

        # Ensure parent directory exists
        socket_file.parent.mkdir(parents=True, exist_ok=True)

        # Start server
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=self.socket_path,
        )

        # Set socket permissions (owner read/write only)
        os.chmod(self.socket_path, 0o600)

        logger.info(f"IPC server listening on {self.socket_path}")

        # Run until shutdown
        async with self._server:
            await self._shutdown_event.wait()

        # Cleanup
        if socket_file.exists():
            socket_file.unlink()

        logger.info("IPC server stopped")

    async def start_background(self) -> None:
        """Start the server in a background task."""
        asyncio.create_task(self.start())
        await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop the server gracefully."""
        self._shutdown_event.set()

        # Close all client connections
        async with self._lock:
            for client in list(self._clients.values()):
                try:
                    client.writer.close()
                    await client.writer.wait_closed()
                except Exception:
                    pass
            self._clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection."""
        # Assign client ID
        async with self._lock:
            self._client_counter += 1
            client_id = f"ipc_{self._client_counter}"

            client = IPCClientConnection(
                reader=reader,
                writer=writer,
                client_id=client_id,
                session_id=None,
                connected_at=datetime.utcnow().isoformat(),
            )
            self._clients[client_id] = client
            self._event_queues[client_id] = asyncio.Queue()

        peer = writer.get_extra_info('peername') or 'unknown'
        logger.info(f"IPC client connected: {client_id} from {peer}")

        # Send connected event
        try:
            connected_event = ConnectedEvent(
                protocol_version="1.0",
                server_info={
                    "client_id": client_id,
                    "transport": "ipc",
                    "socket_path": self.socket_path,
                },
            )
            await self._send_to_client(client_id, connected_event)

            # Start broadcast task for this client
            broadcast_task = asyncio.create_task(
                self._broadcast_to_client(client_id)
            )

            # Read messages
            while not self._shutdown_event.is_set():
                try:
                    message = await self._read_message(reader)
                    if message is None:
                        break  # Connection closed
                    await self._handle_message(client_id, message)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error reading from {client_id}: {e}")
                    break

            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            logger.error(f"Client error {client_id}: {e}")
        finally:
            # Cleanup
            async with self._lock:
                if client_id in self._clients:
                    del self._clients[client_id]
                if client_id in self._event_queues:
                    del self._event_queues[client_id]

            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

            logger.info(f"IPC client disconnected: {client_id}")

    async def _read_message(self, reader: asyncio.StreamReader) -> Optional[str]:
        """Read a length-prefixed message from the stream.

        Returns:
            The message string, or None if connection closed.
        """
        # Read length header
        header = await reader.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            return None  # Connection closed

        length = struct.unpack(">I", header)[0]
        if length > MAX_MESSAGE_SIZE:
            raise ValueError(f"Message too large: {length} bytes")

        # Read payload
        payload = await reader.read(length)
        if len(payload) < length:
            return None  # Connection closed

        return payload.decode("utf-8")

    async def _write_message(
        self,
        writer: asyncio.StreamWriter,
        message: str,
    ) -> None:
        """Write a length-prefixed message to the stream."""
        payload = message.encode("utf-8")
        header = struct.pack(">I", len(payload))
        writer.write(header + payload)
        await writer.drain()

    async def _handle_message(self, client_id: str, message: str) -> None:
        """Handle an incoming message from a client."""
        try:
            event = deserialize_event(message)
        except json.JSONDecodeError as e:
            await self._send_error(client_id, f"Invalid JSON: {e}")
            return
        except ValueError as e:
            await self._send_error(client_id, str(e))
            return

        # Get client's session
        async with self._lock:
            client = self._clients.get(client_id)
            session_id = client.session_id if client else None

        # Handle session selection
        if hasattr(event, 'session_id') and event.session_id:
            session_id = event.session_id
            async with self._lock:
                if client:
                    client.session_id = session_id

        # Route to session handler
        # CommandRequest is allowed without session_id (for session.create, session.default, etc.)
        if self._on_session_request:
            if session_id or isinstance(event, CommandRequest):
                self._on_session_request(client_id, session_id or "", event)
            else:
                await self._send_error(
                    client_id,
                    "No session selected. Use session.create or session.attach first."
                )

    async def _send_to_client(self, client_id: str, event: Event) -> None:
        """Send an event to a specific client."""
        async with self._lock:
            client = self._clients.get(client_id)
            if not client:
                return

        try:
            await self._write_message(client.writer, serialize_event(event))
        except Exception as e:
            logger.error(f"Send error to {client_id}: {e}")

    async def _send_error(self, client_id: str, error: str) -> None:
        """Send an error event to a client."""
        await self._send_to_client(
            client_id,
            ErrorEvent(error=error, error_type="RequestError")
        )

    def queue_event(self, client_id: str, event: Event) -> None:
        """Queue an event for delivery to a client.

        This is thread-safe and can be called from session handlers.
        """
        logger.debug(f"queue_event: {client_id} <- {type(event).__name__}")
        if client_id in self._event_queues:
            try:
                self._event_queues[client_id].put_nowait(event)
                logger.debug(f"  queued successfully")
            except asyncio.QueueFull:
                logger.warning(f"Event queue full for {client_id}")
        else:
            logger.warning(f"queue_event: client {client_id} not in queues")

    def set_client_session(self, client_id: str, session_id: str) -> None:
        """Set the session ID for a client.

        This is thread-safe and should be called after session.create/attach/default.
        """
        if client_id in self._clients:
            self._clients[client_id].session_id = session_id

    def broadcast_to_session(self, session_id: str, event: Event) -> None:
        """Broadcast an event to all clients attached to a session."""
        for client_id, client in self._clients.items():
            if client.session_id == session_id:
                self.queue_event(client_id, event)

    async def _broadcast_to_client(self, client_id: str) -> None:
        """Continuously send queued events to a client."""
        logger.debug(f"_broadcast_to_client started for {client_id}")
        queue = self._event_queues.get(client_id)
        if not queue:
            logger.warning(f"_broadcast_to_client: no queue for {client_id}")
            return

        while not self._shutdown_event.is_set():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
                logger.debug(f"_broadcast_to_client: sending {type(event).__name__} to {client_id}")
                await self._send_to_client(client_id, event)
                logger.debug(f"_broadcast_to_client: sent successfully")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.debug(f"_broadcast_to_client cancelled for {client_id}")
                break
            except Exception as e:
                logger.error(f"Broadcast error to {client_id}: {e}")
        logger.debug(f"_broadcast_to_client exiting for {client_id}")

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

    def get_clients(self) -> Dict[str, Dict[str, Any]]:
        """Get info about connected clients."""
        return {
            client_id: {
                "session_id": client.session_id,
                "connected_at": client.connected_at,
            }
            for client_id, client in self._clients.items()
        }
