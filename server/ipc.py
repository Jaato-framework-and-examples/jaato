r"""IPC Server using Unix Domain Sockets or Windows Named Pipes.

This module provides a local IPC server for fast, secure communication
with local clients (rich-client, IDE extensions, etc.).

On Unix/Linux/macOS:
- Uses Unix domain sockets
- Faster than TCP (no network stack overhead)
- Inherently secure (filesystem permissions)
- Local-only (no remote access)

On Windows:
- Uses named pipes (\\.\pipe\pipename)
- Native Windows IPC mechanism
- Secure and local-only

Usage:
    from server.ipc import JaatoIPCServer

    # On Unix: uses socket path
    server = JaatoIPCServer(socket_path="/tmp/jaato.sock")

    # On Windows: uses named pipe (socket_path is pipe name)
    server = JaatoIPCServer(socket_path="jaato")  # becomes \\.\pipe\jaato

    await server.start()
"""

import asyncio
import json
import logging
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set


# Windows named pipe prefix (\\.\pipe\)
WINDOWS_PIPE_PREFIX = "\\\\.\\pipe\\"


def _get_default_ipc_path() -> str:
    """Get the platform-appropriate default IPC path.

    Returns:
        On Unix: Path to socket file in temp directory
        On Windows: Named pipe name (will be prefixed with \\\\.\\pipe\\)
    """
    if sys.platform == "win32":
        # On Windows, use a named pipe
        return "jaato"
    else:
        # On Unix, use a socket file in temp directory
        temp_dir = Path(tempfile.gettempdir())
        return str(temp_dir / "jaato.sock")


def _get_display_path(path: str) -> str:
    """Get a display-friendly path for logging.

    Args:
        path: The IPC path (socket path or pipe name)

    Returns:
        Full path including Windows pipe prefix if applicable
    """
    if sys.platform == "win32" and not path.startswith(WINDOWS_PIPE_PREFIX):
        return f"{WINDOWS_PIPE_PREFIX}{path}"
    return path

from jaato_sdk.events import (
    Event,
    EventType,
    ConnectedEvent,
    ErrorEvent,
    SystemMessageEvent,
    CommandListEvent,
    serialize_event,
    deserialize_event,
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    StopRequest,
    CommandRequest,
    CommandListRequest,
    ClientConfigRequest,
    PostAuthSetupResponse,
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
    workspace_path: Optional[str] = None  # Client's working directory


class _PipeServerProtocol(asyncio.StreamReaderProtocol):
    """Protocol handler for Windows named pipe connections.

    Extends StreamReaderProtocol to properly handle StreamReader/StreamWriter
    pairs with all required methods (like _drain_helper for drain()).
    """

    def __init__(self, server: "JaatoIPCServer"):
        self._server = server
        self._pipe_reader = asyncio.StreamReader()
        super().__init__(self._pipe_reader, self._client_connected_cb)

    def connection_made(self, transport):
        """Called when a pipe client connects."""
        logger.info("Windows pipe: connection_made called")
        super().connection_made(transport)

    def connection_lost(self, exc):
        """Called when the pipe connection is lost."""
        logger.info(f"Windows pipe: connection_lost called, exc={exc}")
        super().connection_lost(exc)

    def _client_connected_cb(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Called when the stream reader/writer are ready."""
        logger.info("Windows pipe: _client_connected_cb called")
        # Handle the client in a task with error handling
        task = asyncio.create_task(
            self._handle_client_with_logging(reader, writer)
        )
        # Log any exceptions from the task
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task):
        """Log any exceptions from the client handler task."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"Windows pipe client handler error: {exc}", exc_info=exc)
        except asyncio.CancelledError:
            pass

    async def _handle_client_with_logging(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client with exception logging."""
        try:
            logger.info("Windows pipe: starting client handler")
            await self._server._handle_pipe_client(reader, writer)
        except Exception as e:
            logger.error(f"Windows pipe client handler exception: {e}", exc_info=True)
            raise


class JaatoIPCServer:
    r"""IPC server using Unix domain sockets or Windows named pipes.

    Provides fast local communication for:
    - rich-client TUI
    - IDE extensions
    - CLI tools
    - Local scripts

    Protocol:
    - Each message is framed: 4-byte length (big-endian) + JSON payload
    - Same event protocol as WebSocket (server/events.py)
    - Supports session multiplexing (client specifies session_id)

    Platform support:
    - Unix/Linux/macOS: Unix domain sockets
    - Windows: Named pipes (\\.\pipe\pipename)
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        socket_mode: int = 0o660,
        on_session_request: Optional[Callable[[str, str, Event], None]] = None,
        on_command_list_request: Optional[Callable[[], list]] = None,
    ):
        """Initialize the IPC server.

        Args:
            socket_path: Path to the Unix domain socket or Windows pipe name.
                Defaults to platform-appropriate path.
            socket_mode: Unix file permissions for the socket (default: 0o660,
                owner+group read/write). Use 0o666 to allow any local user
                to connect.
            on_session_request: Callback for session requests.
                Called with (client_id, session_id, event).
            on_command_list_request: Callback to get list of available commands.
                Returns list of {name, description} dicts.
        """
        self.socket_path = socket_path or _get_default_ipc_path()
        self.socket_mode = socket_mode
        self._on_session_request = on_session_request
        self._on_command_list_request = on_command_list_request

        # Server state
        self._server: Optional[asyncio.Server] = None
        self._pipe_server = None  # Windows named pipe server
        self._clients: Dict[str, IPCClientConnection] = {}
        self._client_counter = 0
        self._lock = asyncio.Lock()

        # Event loop reference for thread-safe operations
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Event queue for broadcasting
        self._event_queues: Dict[str, asyncio.Queue[Event]] = {}

        # Shutdown flag
        self._shutdown_event = asyncio.Event()

    def _get_pipe_path(self) -> str:
        """Get the full Windows named pipe path."""
        if self.socket_path.startswith(WINDOWS_PIPE_PREFIX):
            return self.socket_path
        return f"{WINDOWS_PIPE_PREFIX}{self.socket_path}"

    async def start(self) -> None:
        """Start the IPC server.

        On Unix: Creates a Unix domain socket and listens for connections.
        On Windows: Creates a named pipe and listens for connections.
        """
        # Capture event loop for thread-safe operations
        self._event_loop = asyncio.get_running_loop()

        display_path = _get_display_path(self.socket_path)

        if sys.platform == "win32":
            # Windows: Use named pipes
            await self._start_windows_pipe_server()
        else:
            # Unix: Use Unix domain sockets
            await self._start_unix_socket_server()

        logger.info(f"IPC server listening on {display_path}")

        # Run until shutdown
        await self._shutdown_event.wait()

        # Cleanup handled in stop()
        logger.info("IPC server stopped")

    async def _start_unix_socket_server(self) -> None:
        """Start the Unix domain socket server."""
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

        os.chmod(self.socket_path, self.socket_mode)

        # Run server in background
        asyncio.create_task(self._run_unix_server())

    async def _run_unix_server(self) -> None:
        """Run the Unix socket server until shutdown."""
        async with self._server:
            await self._shutdown_event.wait()

        # Cleanup socket file
        socket_file = Path(self.socket_path)
        if socket_file.exists():
            socket_file.unlink()

    async def _start_windows_pipe_server(self) -> None:
        """Start the Windows named pipe server."""
        pipe_path = self._get_pipe_path()
        logger.info(f"Starting Windows pipe server at {pipe_path}")

        # Use proactor event loop's pipe server
        loop = asyncio.get_running_loop()
        logger.info(f"Event loop type: {type(loop).__name__}")

        def protocol_factory():
            logger.info("Protocol factory called - creating new protocol instance")
            return _PipeServerProtocol(self)

        # Start serving on the named pipe
        try:
            self._pipe_server = await loop.start_serving_pipe(
                protocol_factory,
                pipe_path,
            )
            logger.info(f"Pipe server created: {self._pipe_server}")
        except Exception as e:
            logger.error(f"Failed to start pipe server: {e}", exc_info=True)
            raise

        # Run server in background
        asyncio.create_task(self._run_windows_pipe_server())

    async def _run_windows_pipe_server(self) -> None:
        """Run the Windows pipe server until shutdown."""
        await self._shutdown_event.wait()

        # Close pipe server
        if self._pipe_server:
            self._pipe_server.close()

    async def _handle_pipe_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a Windows named pipe client connection.

        This is called from the pipe protocol when a client connects.
        """
        await self._handle_client(reader, writer)

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

        # Close Unix socket server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Close Windows pipe server
        if self._pipe_server:
            self._pipe_server.close()

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
            logger.info(f"Sending ConnectedEvent to {client_id}...")
            await self._send_to_client(client_id, connected_event)
            logger.info(f"ConnectedEvent sent to {client_id}")

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
        try:
            # Read length header - use readexactly for reliable framed reading
            header = await reader.readexactly(HEADER_SIZE)

            length = struct.unpack(">I", header)[0]
            if length > MAX_MESSAGE_SIZE:
                raise ValueError(f"Message too large: {length} bytes")

            # Read payload
            payload = await reader.readexactly(length)
            return payload.decode("utf-8")

        except asyncio.IncompleteReadError:
            # Connection closed before complete message was read
            return None
        except ConnectionResetError:
            # Connection was reset by peer
            return None

    async def _write_message(
        self,
        writer: asyncio.StreamWriter,
        message: str,
    ) -> None:
        """Write a length-prefixed message to the stream."""
        payload = message.encode("utf-8")
        header = struct.pack(">I", len(payload))
        logger.debug(f"_write_message: writing {len(payload)} bytes")
        writer.write(header + payload)
        logger.debug("_write_message: calling drain()")
        await writer.drain()
        logger.debug("_write_message: drain() completed")

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

        # Handle CommandListRequest directly - no session needed
        if isinstance(event, CommandListRequest):
            commands = []
            if self._on_command_list_request:
                commands = self._on_command_list_request()
            await self._send_to_client(
                client_id,
                CommandListEvent(commands=commands)
            )
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
        # CommandRequest, ClientConfigRequest, and PostAuthSetupResponse are
        # allowed without session_id — they are handled at daemon level.
        if self._on_session_request:
            if session_id or isinstance(event, (CommandRequest, ClientConfigRequest, PostAuthSetupResponse)):
                # ClientConfigRequest must be processed synchronously to ensure
                # client's env_file is registered before session creation
                if isinstance(event, ClientConfigRequest):
                    self._on_session_request(client_id, session_id or "", event)
                else:
                    # Run other requests in executor to not block the event loop.
                    # This allows _broadcast_to_client to send events while
                    # the session request (e.g., initialization) is running.
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,  # Default executor
                        self._on_session_request,
                        client_id,
                        session_id or "",
                        event,
                    )
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

        This is thread-safe and can be called from session handlers
        running in thread pool executors.
        """
        logger.debug(f"queue_event: {client_id} <- {type(event).__name__}")
        if client_id not in self._event_queues:
            logger.warning(f"queue_event: client {client_id} not in queues")
            return

        queue = self._event_queues[client_id]

        def _do_put():
            try:
                queue.put_nowait(event)
                logger.debug(f"  queued successfully")
            except asyncio.QueueFull:
                logger.warning(f"Event queue full for {client_id}")

        # Use call_soon_threadsafe for thread-safe queue operations
        if self._event_loop:
            self._event_loop.call_soon_threadsafe(_do_put)
        else:
            # Fallback for when called before start() - direct put
            _do_put()

    def set_client_session(self, client_id: str, session_id: str) -> None:
        """Set the session ID for a client.

        This is thread-safe and should be called after session.create/attach/default.
        """
        if client_id in self._clients:
            self._clients[client_id].session_id = session_id

    def set_client_workspace(self, client_id: str, workspace_path: str) -> None:
        """Set the workspace path for a client.

        Args:
            client_id: The client ID.
            workspace_path: The client's working directory.
        """
        if client_id in self._clients:
            self._clients[client_id].workspace_path = workspace_path

    def get_client_workspace(self, client_id: str) -> Optional[str]:
        """Get the workspace path for a client.

        Args:
            client_id: The client ID.

        Returns:
            The client's working directory, or None if not set.
        """
        client = self._clients.get(client_id)
        return client.workspace_path if client else None

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
                event = await asyncio.wait_for(queue.get(), timeout=0.01)
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
