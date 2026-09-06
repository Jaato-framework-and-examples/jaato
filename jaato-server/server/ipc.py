r"""IPC Server using Unix Domain Sockets or Windows Named Pipes.

This module provides a local IPC server for fast, secure communication
with local clients (jaato-tui, IDE extensions, etc.).

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
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from shared.framing import (
    HEADER_SIZE,
    MAX_MESSAGE_SIZE,
    FrameTooLargeError,
    read_frame,
    write_frame,
)
from shared.session_id import is_safe_session_id
from shared.plugins.path_safety import ensure_private_dir


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
    PROTOCOL_VERSION,
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
    ToolsRegisterClientRequest,
    ToolExecuteResultEvent,
    ToolOutputEvent,
)


logger = logging.getLogger(__name__)


def _get_server_version() -> str:
    """Read the jaato-server package version from installed metadata."""
    from importlib.metadata import version as pkg_version
    return pkg_version("jaato-server")


# ---------------------------------------------------------------------------
# Per-client event-queue backpressure
# ---------------------------------------------------------------------------
#
# Sizing: PCM16 24 kHz mono is ~48 KB/s, so 100 ms audio chunks arrive at
# ~10/sec.  A 2048-event bound is therefore minutes of audio backlog for a
# client that has stopped reading -- far past the point where dropping is
# the right answer -- while staying small enough that the worst case is
# bounded memory rather than an OOM.  Override for clients that legitimately
# burst (a cascade driver replaying a long transcript).
def _resolve_event_queue_max() -> int:
    """Resolve the per-client event-queue bound from the environment.

    A non-numeric or non-positive value falls back to the default rather
    than disabling the bound, because "unbounded" is the bug this exists
    to fix and a typo should not silently reintroduce it.
    """
    raw = os.environ.get("JAATO_IPC_EVENT_QUEUE_MAX", "").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
        logger.warning(
            f"Ignoring invalid JAATO_IPC_EVENT_QUEUE_MAX={raw!r}; "
            f"using default {_DEFAULT_EVENT_QUEUE_MAX}"
        )
    return _DEFAULT_EVENT_QUEUE_MAX


_DEFAULT_EVENT_QUEUE_MAX = 2048
_EVENT_QUEUE_MAX = _resolve_event_queue_max()


def _is_lossy_event(event: Any) -> bool:
    """Whether ``event`` may be dropped under backpressure.

    Lossy means "a gap degrades the experience but does not desynchronise
    the client".  Only live tool-output chunks qualify: they are the
    firehose, and a client that misses one still has a consistent view of
    the session.

    Deliberately NOT lossy: agent output text.  It looks like a similar
    token firehose, but a client accumulates it into the visible
    transcript, so a dropped chunk silently corrupts what the user reads
    rather than merely thinning it.

    ``isinstance`` rather than a class-name match, so a subclass is
    classified with its base instead of silently becoming undroppable.
    """
    return isinstance(event, ToolOutputEvent)


def _event_is_media(event: Any) -> bool:
    """Whether ``event`` carries a binary payload."""
    return bool(
        getattr(event, "mime_type", None) and getattr(event, "data_b64", None)
    )


def _evict_one_lossy(queue: "asyncio.Queue") -> bool:
    """Evict the oldest lossy event from ``queue``; return whether one went.

    Media is preferred over text: a media chunk is orders of magnitude
    more bytes, so evicting one buys far more headroom per unit of lost
    output.  Within a class the OLDEST is evicted -- for a media stream
    recency is what matters, and a stale chunk is worthless.

    Drains and refills via the public API so the relative order of every
    retained event is preserved.  Returns ``False`` when the queue holds
    nothing droppable, leaving it untouched.
    """
    drained = []
    while True:
        try:
            drained.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    victim = None
    for index, item in enumerate(drained):
        if not _is_lossy_event(item):
            continue
        if _event_is_media(item):
            victim = index
            break
        if victim is None:
            victim = index  # oldest text chunk, kept as the fallback

    if victim is not None:
        drained.pop(victim)

    for item in drained:
        queue.put_nowait(item)
    return victim is not None


# Message framing constants are imported from shared.framing — see
# that module for the wire format (4-byte big-endian length prefix +
# UTF-8 payload).  Re-exported here as ``HEADER_SIZE`` /
# ``MAX_MESSAGE_SIZE`` for any in-tree caller still importing them
# from server.ipc.


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
    - jaato-tui TUI
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
        on_client_disconnect: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the IPC server.

        Args:
            socket_path: Path to the Unix domain socket or Windows pipe name.
                Defaults to platform-appropriate path.
            socket_mode: Unix file permissions for the socket (default: 0o660,
                owner and group only). The IPC transport has no per-message
                authentication, so any principal that can open the socket can
                fully drive the agent (send prompts, run whitelisted tools,
                read/write the workspace). 0o660 keeps that reachable only by
                the daemon's owner and group; pass 0o666 to opt into
                world-accessible (e.g. cross-user containers on a trusted host).
            on_session_request: Callback for session requests.
                Called with (client_id, session_id, event).
            on_command_list_request: Callback to get list of available commands.
                Returns list of {name, description} dicts.
            on_client_disconnect: Callback fired when an IPC client
                connection terminates (clean disconnect or error).
                Called with ``(client_id,)``.  Wired to
                ``CommandRouter.handle_client_disconnect`` so that
                ``SessionManager`` detaches the client from its session
                — without this, ``session.attached_clients`` retains
                stale ids forever and per-session resources (workspace
                monitor inotify handles) leak.
        """
        self.socket_path = socket_path or _get_default_ipc_path()
        self.socket_mode = socket_mode
        self._on_session_request = on_session_request
        self._on_command_list_request = on_command_list_request
        self._on_client_disconnect = on_client_disconnect

        # Server state
        self._server: Optional[asyncio.Server] = None
        self._pipe_server = None  # Windows named pipe server
        self._clients: Dict[str, IPCClientConnection] = {}
        self._client_counter = 0
        self._lock = asyncio.Lock()
        # Client-provided ("host") tool support.  ``_session_manager`` is wired
        # at construction (server/__main__.py) so the IPC transport can reach a
        # session's registry to register the proxy tools; ``_client_tool_waiters``
        # holds call_id -> (Event, holder) for the proxy executors' blocking wait
        # on the client's ToolExecuteResultEvent.
        self._session_manager = None
        self._client_tool_waiters: Dict[str, Any] = {}
        # Host tools registered BEFORE session.new — buffered here for the
        # post-session.new proxy-EXECUTOR registration (the SCHEMAS are buffered
        # separately in SessionManager for the pre-spawn envelope seeding).
        self._pending_client_tools_ipc: Dict[str, Any] = {}

        # Custom message-type handlers registered by daemon extensions
        # via ``register_message_handler``.  Fires when a client sends
        # a JSON message whose ``type`` field doesn't match any built-
        # in event class (parallel to the WS path's
        # ``_message_handlers`` at ``websocket.py``).  Built-in types
        # cannot be overridden — those still take the typed-event
        # dispatch in ``_handle_message``.
        self._message_handlers: Dict[str, Any] = {}

        # Event loop reference for thread-safe operations
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Event queue for broadcasting.  BOUNDED -- see
        # ``_EVENT_QUEUE_MAX`` and ``queue_event`` for the drop policy.
        # An unbounded queue meant a slow consumer grew it without limit:
        # cosmetic lag for text, but unbounded memory and monotonically
        # increasing audio drift once chunks carry media.
        self._event_queues: Dict[str, asyncio.Queue[Event]] = {}

        # Per-client count of media chunks evicted under backpressure,
        # so a client can be told its stream was lossy rather than
        # silently receiving a gap.
        self._dropped_chunks: Dict[str, int] = {}

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
        """Start the Unix domain socket server.

        The default socket path sits under a shared, world-writable temp
        directory, so both the socket's own path and the directory holding it
        are pre-plantable: another user can create either first and have the
        daemon bind somewhere they control.  Since the IPC transport is
        unauthenticated, that is a full takeover of the agent.  So the
        stale-socket cleanup unlinks only entries that are ours to remove — a
        symlink at the socket path is refused, never followed — and a parent
        directory the daemon has to create is adopted through
        ``ensure_private_dir``, which refuses a pre-planted symlink or another
        user's directory rather than binding inside it.

        Raises:
            OSError: if the socket directory cannot be used safely, or the
                path is occupied by something that must not be unlinked.
        """
        # Remove stale socket from a previous run
        socket_file = Path(self.socket_path)
        if socket_file.exists() or socket_file.is_symlink():
            if socket_file.is_symlink():
                # A symlink here is never ours: the daemon only ever creates a
                # socket.  Unlinking it and binding would place the socket at
                # the planter's chosen target, so refuse instead.
                raise OSError(
                    f"Cannot create socket at '{self.socket_path}': "
                    f"a symlink exists at that path (pre-planted?). "
                    f"Inspect it, then remove it with: rm {self.socket_path}"
                )
            if socket_file.is_socket():
                socket_file.unlink()
            elif socket_file.is_dir():
                raise OSError(
                    f"Cannot create socket at '{self.socket_path}': "
                    f"a directory exists at that path. "
                    f"Remove it with: rm -rf {self.socket_path}"
                )
            else:
                # Regular file or other non-socket — try to unlink
                socket_file.unlink()

        # Ensure the parent directory exists.  The strict private-directory
        # rules apply only to a directory *we* create: finding one already
        # there where we expected to make it is the pre-planting case, and
        # ensure_private_dir refuses a symlink or another user's directory
        # rather than binding inside it.  An operator-supplied parent that
        # already exists (``--ipc-socket /tmp/jaato.sock`` → ``/tmp``, a
        # root-owned system root) is theirs to choose, not ours to police —
        # the socket path itself is still guarded by the symlink refusal above.
        parent = socket_file.parent
        if not parent.exists():
            ensure_private_dir(parent)
        elif not parent.is_dir():
            raise OSError(
                f"Cannot create socket at '{self.socket_path}': "
                f"'{parent}' exists but is not a directory."
            )

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
                connected_at=datetime.now(timezone.utc).isoformat(),
            )
            self._clients[client_id] = client
            self._event_queues[client_id] = asyncio.Queue()
            self._dropped_chunks[client_id] = 0

        peer = writer.get_extra_info('peername') or 'unknown'
        logger.info(f"IPC client connected: {client_id} from {peer}")

        # Send connected event
        try:
            connected_event = ConnectedEvent(
                protocol_version=PROTOCOL_VERSION,
                server_info={
                    "client_id": client_id,
                    "transport": "ipc",
                    "socket_path": self.socket_path,
                    "server_version": _get_server_version(),
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
                self._dropped_chunks.pop(client_id, None)
            # Detach from session via the transport-agnostic helper
            # (CommandRouter.handle_client_disconnect → SessionManager.detach_client).
            # Without this, session.attached_clients retains the dead
            # client_id, _maybe_unload_session blocks forever, and the
            # session's workspace_monitor never releases its inotify
            # handle — leading to Errno 24 on long-lived daemons.
            if self._on_client_disconnect:
                try:
                    self._on_client_disconnect(client_id)
                except Exception as exc:
                    logger.error(
                        f"on_client_disconnect failed for {client_id}: {exc}"
                    )

            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

            logger.info(f"IPC client disconnected: {client_id}")

    async def _read_message(self, reader: asyncio.StreamReader) -> Optional[str]:
        """Read a length-prefixed message from the stream.

        Thin wrapper around :func:`shared.framing.read_frame` —
        delegates the wire format so the runner-RPC channel can share
        the same framing.  Returns ``None`` on clean EOF or peer
        reset; oversize-frame errors propagate as
        :class:`shared.framing.FrameTooLargeError`.
        """
        return await read_frame(reader)

    async def _write_message(
        self,
        writer: asyncio.StreamWriter,
        message: str,
    ) -> None:
        """Write a length-prefixed message to the stream.

        Thin wrapper around :func:`shared.framing.write_frame` —
        same wire format as the daemon-runner RPC.
        """
        await write_frame(writer, message)

    def register_message_handler(
        self,
        message_type: str,
        handler: Any,
    ) -> None:
        """Register an async handler for a custom IPC message type.

        Symmetric to :meth:`JaatoWSServer.register_message_handler`.
        Daemon extensions call this (via
        ``_ExtensionContext.ipc_server`` or the unified
        ``_ExtensionContext.register_message_handler``) to handle
        custom message types on IPC connections — the same
        connections used for built-in session/command messages.

        The handler is called when a client sends a JSON message
        whose ``type`` field matches *message_type* and no built-in
        event class deserialises from it.  Built-in types cannot be
        overridden.

        Handler signature is intentionally **transport-agnostic**
        (no raw socket passed) so the same callback works on both
        IPC and WS sides:

            async def handler(message: dict, client_id: str, user: Optional[str]) -> None

        - ``message``: the parsed JSON dict.
        - ``client_id``: server-assigned client identifier — feed
          back into ``ctx.emit_to_client(client_id, event)`` to
          target the reply.
        - ``user``: authenticated user identifier or ``None``
          (always ``None`` on IPC today; included for symmetry with
          WS, which does carry an authenticated user when SSO is
          wired up).

        Args:
            message_type: The ``type`` field value to match
                (e.g., ``"gates.list"``).
            handler: Async callback with the signature above.
        """
        if message_type in self._message_handlers:
            logger.warning(
                "Overwriting existing IPC handler for message type '%s'",
                message_type,
            )
        self._message_handlers[message_type] = handler
        logger.debug("Registered IPC message handler: %s", message_type)

    async def _handle_message(self, client_id: str, message: str) -> None:
        """Handle an incoming message from a client."""
        try:
            event = deserialize_event(message)
        except json.JSONDecodeError as e:
            await self._send_error(client_id, f"Invalid JSON: {e}")
            return
        except ValueError as e:
            # Unknown event type — check extension message handlers
            # before falling through to error.  Mirrors the WS
            # dispatch fallthrough at ``websocket.py``.
            try:
                raw = json.loads(message)
            except json.JSONDecodeError:
                await self._send_error(client_id, str(e))
                return
            msg_type = raw.get("type", "")
            handler = self._message_handlers.get(msg_type)
            if handler:
                async with self._lock:
                    client = self._clients.get(client_id)
                user = getattr(client, "user_id", None) if client else None
                try:
                    await handler(raw, client_id, user)
                except Exception:
                    logger.exception(
                        "IPC message handler for '%s' raised", msg_type,
                    )
                    await self._send_error(
                        client_id,
                        f"Handler for '{msg_type}' raised — see daemon log",
                    )
                return
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

        # Client-provided ("host") tool registration (mirrors the WS path; uses
        # the shared registration helper) — register the proxy schema/executor so
        # the agent can call it, the executor proxying back to this IPC client.
        if isinstance(event, ToolsRegisterClientRequest):
            self._register_client_tools_ipc(
                client_id, event.tools, getattr(event, "categories", None))
            return
        # Client-side tool execution result — wake the proxy executor's waiter.
        if isinstance(event, ToolExecuteResultEvent):
            entry = self._client_tool_waiters.pop(event.call_id, None)
            if entry is not None:
                waiter, holder = entry
                # Decode the JSON-encoded host-tool result to its native value
                # (symmetric with the SDK's encode) so it records like an
                # in-process tool result — see decode_client_tool_result.
                from server.client_tools import decode_client_tool_result
                holder["result"] = decode_client_tool_result(event.result)
                holder["error"] = event.error
                waiter.set()
            return

        # Get client's session
        async with self._lock:
            client = self._clients.get(client_id)
            session_id = client.session_id if client else None

        # Handle session selection
        if hasattr(event, 'session_id') and event.session_id:
            # Client-supplied id — reject a traversal / injection id at the
            # boundary before it reaches persistence / cgroup / apparmor sinks.
            if not is_safe_session_id(event.session_id):
                await self._send_to_client(
                    client_id,
                    ErrorEvent(
                        error="invalid session_id: must match [A-Za-z0-9._-] "
                              "(1-256 chars) with no '..'",
                        error_type="RequestError",
                    ),
                )
                return
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
                    # After session.new/attach, register buffered host-tool
                    # EXECUTORS now that the session (+ registry) exists.  The
                    # SCHEMAS already rode the runner envelope (seeded pre-spawn
                    # in the session.new flow); this wires the daemon-side proxy
                    # executors for execution.  Mirrors websocket's post-apply.
                    if (isinstance(event, CommandRequest)
                            and event.command.lower() in ("session.new", "session.attach")):
                        if client_id in self._pending_client_tools_ipc:
                            pending = self._pending_client_tools_ipc.pop(client_id)
                            # Wires tools + drives any deferred wake AFTER the
                            # runner push (see _register_client_tools_ipc._push).
                            self._register_client_tools_ipc(client_id, pending)
                        else:
                            # No buffered host tools to wire → drive any deferred
                            # wake for the now-attached session immediately.
                            _c = self._clients.get(client_id)
                            _sid = _c.session_id if _c else None
                            if _sid and self._session_manager is not None:
                                try:
                                    self._session_manager.drive_pending_wake(_sid)
                                except Exception:
                                    logger.exception(
                                        "deferred-wake drive on no-tools attach "
                                        "failed for %s", _sid)
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

    def _make_ipc_send_request(self, session_id: str):
        """Build the proxy executor's transport send: dispatch the
        ``ToolExecuteRequestEvent`` to every IPC client attached to
        ``session_id``.  Returns True iff at least one received it (the shared
        helper retries while none is connected)."""
        def _send(event) -> bool:
            sent = False
            for cid, client in list(self._clients.items()):
                if client.session_id == session_id:
                    self.send_event(cid, event)
                    sent = True
            return sent
        return _send

    def _register_client_tools_ipc(self, client_id, tools, categories=None) -> None:
        """Register client-provided ("host") tools for an IPC client.

        Before the session exists (registered before session.new): buffer the
        SCHEMAS via the ``SessionManager`` so the session.new flow seeds the
        runner envelope before the spawn (shared with the WS path).  Once the
        session exists: register the proxy executors (which send
        ``ToolExecuteRequestEvent`` to this IPC client) on the session registry
        via the shared helper.
        """
        sm = self._session_manager
        if sm is None:
            logger.warning("IPC client tools: no session_manager wired — ignoring")
            return
        client = self._clients.get(client_id)
        session_id = client.session_id if client else None
        session = sm.get_session(session_id) if session_id else None
        if (session is None or getattr(session, "server", None) is None
                or session.server.registry is None):
            # No session yet: buffer the SCHEMAS (SessionManager → pre-spawn
            # envelope seed) AND the tools for the post-session.new proxy-executor
            # registration below.
            sm.buffer_client_tools(client_id, tools)
            self._pending_client_tools_ipc[client_id] = tools
            logger.info("Buffered %d IPC client tools for %s (session pending)",
                        len(tools or []), client_id)
            return
        from server.client_tools import register_client_tools
        registered = register_client_tools(
            session.server.registry, session.server, tools,
            send_request=self._make_ipc_send_request(session_id),
            waiters=self._client_tool_waiters,
        )
        # Glue the schemas to the RUNNER-tier model so it SEES a tool registered
        # AFTER session.new (register_client_tools above wires daemon-side
        # execution only; the model runs in the runner).  Best-effort — see the
        # WS path for the rationale.
        runner_rpc = getattr(session.server, "_runner_rpc", None)
        if runner_rpc is not None:
            # Run the blocking runner-RPC OFF the daemon event loop: this method
            # runs in the async dispatch, and the threadsafe RPC waits on
            # future.result() against that same loop — calling it inline
            # deadlocks (the loop can't deliver the RPC response while blocked).
            # Best-effort background thread; the proxy is already registered.
            import threading

            def _push(rpc=runner_rpc, t=tools, cid=client_id, sid=session_id):
                try:
                    res = rpc.session_register_client_tools_threadsafe(t)
                    logger.info(
                        "mid-session client-tool runner push for %s: %s", cid, res)
                except Exception:
                    logger.exception(
                        "mid-session client-tool runner push failed for %s", cid)
                # Deferred-turn (Option 2): the runner now has this client's host
                # tools, so it's safe to drive any wake deferred while the session
                # was cold — the turn's schema will include these tools.  Driving
                # here (after the push) rather than at attach-end closes the
                # drive-races-tool-wiring gap.  drive_pending_wake is idempotent.
                try:
                    self._session_manager.drive_pending_wake(sid)
                except Exception:
                    logger.exception(
                        "deferred-wake drive after client-tool push failed for %s", sid)

            threading.Thread(target=_push, daemon=True).start()
        logger.info("Registered %d IPC client tools for %s: %s",
                    len(registered), client_id, registered)

    def send_event(self, client_id: str, event: Event) -> None:
        """Send an event to a specific client (``EventSink`` protocol).

        Silently ignores ``client_id`` values not owned by this transport,
        which is the ``EventSink`` contract that allows ``CompositeEventSink``
        to fan-out safely across IPC and WS sinks.

        Thread-safe.
        """
        if client_id not in self._event_queues:
            return
        self.queue_event(client_id, event)

    def broadcast_event(self, event: Event) -> None:
        """Send an event to every connected IPC client (``EventSink`` protocol).

        Used for daemon-wide events that don't belong to a specific
        session.  Iterates a snapshot of the current client list, so
        clients connecting/disconnecting during the broadcast do not
        race with the iteration.

        Per-client delivery failures are swallowed by ``queue_event``
        (log-and-continue), so one stuck client never blocks the rest.

        Thread-safe.
        """
        # Snapshot keys to avoid "dictionary changed size during iteration"
        # when a client connects/disconnects mid-broadcast.
        for client_id in list(self._event_queues.keys()):
            self.queue_event(client_id, event)

    def queue_event(self, client_id: str, event: Event) -> None:
        """Queue an event for delivery to a client, applying backpressure.

        The queue is bounded by ``_EVENT_QUEUE_MAX``.  The bound is
        enforced here rather than by ``asyncio.Queue(maxsize=...)`` so
        that an essential event can deliberately exceed it; the queue
        itself stays unbounded and ``put_nowait`` therefore never raises.

        The policy is deliberately NOT uniform, because events are not
        equally replaceable:

        - **Lossy** events (tool-output chunks) degrade gracefully.  A
          dropped audio chunk is a click; a *late* one is drift that never
          recovers, so recency beats completeness and the OLDEST lossy
          event is evicted to make room.  Media is evicted before text,
          because a media chunk is ~1000x the bytes of a text chunk, so
          evicting one buys far more headroom per unit of lost output.
        - **Essential** events (lifecycle, permission prompts, results)
          drive the client's state machine.  Losing one desynchronises the
          client permanently -- strictly worse than a bounded memory
          spike -- so an essential event is queued past the bound with a
          warning naming the client.

        Eviction drains and refills through the public queue API rather
        than touching ``queue._queue``, preserving the relative order of
        everything retained.  It is O(n), but runs only on overflow.

        This is thread-safe and can be called from session handlers
        running in thread pool executors.
        """
        logger.debug(f"queue_event: {client_id} <- {type(event).__name__}")
        if client_id not in self._event_queues:
            logger.warning(f"queue_event: client {client_id} not in queues")
            return

        queue = self._event_queues[client_id]

        def _do_put():
            if queue.qsize() >= _EVENT_QUEUE_MAX:
                if _evict_one_lossy(queue):
                    self._dropped_chunks[client_id] = (
                        self._dropped_chunks.get(client_id, 0) + 1
                    )
                elif _is_lossy_event(event):
                    # Nothing evictable and the newcomer is itself lossy:
                    # drop it rather than grow without bound.
                    dropped = self._dropped_chunks.get(client_id, 0) + 1
                    self._dropped_chunks[client_id] = dropped
                    logger.warning(
                        f"Event queue full for {client_id}; dropped a "
                        f"{type(event).__name__} ({dropped} dropped so far)"
                    )
                    return
                else:
                    logger.warning(
                        f"Event queue full for {client_id}; queueing "
                        f"essential {type(event).__name__} past the "
                        f"{_EVENT_QUEUE_MAX} bound"
                    )
            queue.put_nowait(event)
            logger.debug("  queued successfully")

        # Use call_soon_threadsafe for thread-safe queue operations
        if self._event_loop:
            self._event_loop.call_soon_threadsafe(_do_put)
        else:
            # Fallback for when called before start() - direct put
            _do_put()

    def dropped_chunk_count(self, client_id: str) -> int:
        """How many chunks were dropped for ``client_id`` under backpressure.

        An accessor AWAITING A CONSUMER.  Nothing in the tree calls it,
        so the gap it counts is currently silent: a client whose media
        stream was thinned receives fewer chunks and is told nothing,
        and only notices as a jump in ``sequence``.  This said it "lets
        a client be told", which described an intention the code does
        not carry out -- the counter exists and is maintained, but no
        path reports it.

        Wiring it means choosing a surface (a field on a lifecycle
        event, or a reply to an explicit query) and is a protocol
        decision, not an oversight to patch here.  Returns 0 for an
        unknown client.
        """
        return self._dropped_chunks.get(client_id, 0)

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

    def get_client_user(self, client_id: str) -> Optional[str]:
        """Get the authenticated user for an IPC client.

        IPC connections are local and unauthenticated — always returns
        ``None``.  User identity is a WS/SSO concept.
        """
        return None

    def set_client_user(self, client_id: str, user_id: str) -> None:
        """No-op for IPC — local connections have no user identity."""
        pass

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
