r"""IPC Client for connecting to Jaato Server.

This module provides a client for connecting to the Jaato server
via Unix domain socket (Unix/Linux/macOS) or named pipe (Windows).

Usage:
    from jaato_sdk.client import IPCClient

    # On Unix:
    client = IPCClient("/tmp/jaato.sock")

    # On Windows:
    client = IPCClient("jaato")  # connects to \\.\pipe\jaato

    await client.connect()

    # Send a message
    await client.send_message("Hello, world!")

    # Receive events
    async for event in client.events():
        print(event)
"""

import asyncio
import json
import logging
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Optional

logger = logging.getLogger(__name__)

from jaato_sdk.events import (
    Event,
    EventType,
    serialize_event,
    deserialize_event,
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    ReferenceSelectionResponseRequest,
    StopRequest,
    CommandRequest,
    CommandListRequest,
    CommandListEvent,
    ConnectedEvent,
    ErrorEvent,
    HistoryRequest,
    HistoryEvent,
    ClientConfigRequest,
    SessionInfoEvent,
)


# Message framing: 4-byte length prefix (big-endian) + JSON payload
HEADER_SIZE = 4
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB max

# Windows named pipe prefix (\\.\pipe\)
WINDOWS_PIPE_PREFIX = "\\\\.\\pipe\\"

# Platform-specific defaults
if sys.platform == "win32":
    DEFAULT_SOCKET_PATH = "jaato"  # Will become \\.\pipe\jaato
    DEFAULT_PID_FILE = str(Path(tempfile.gettempdir()) / "jaato.pid")
else:
    DEFAULT_SOCKET_PATH = "/tmp/jaato.sock"
    DEFAULT_PID_FILE = "/tmp/jaato.pid"


class IPCClient:
    r"""Client for connecting to Jaato server via IPC.

    Provides async methods for:
    - Connecting to server
    - Sending messages and commands
    - Receiving events
    - Auto-starting server if not running

    Platform support:
    - Unix/Linux/macOS: Unix domain sockets
    - Windows: Named pipes (\\.\pipe\pipename)
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        auto_start: bool = True,
        env_file: str = ".env",
        workspace_path: Optional[str] = None,
    ):
        """Initialize the IPC client.

        Args:
            socket_path: Path to Unix domain socket or Windows pipe name.
            auto_start: Whether to auto-start server if not running.
            env_file: Path to .env file for auto-started server.
            workspace_path: Working directory sent to the server for file
                operations and sandbox scoping.  Falls back to
                ``os.getcwd()`` when not provided.
        """
        self.socket_path = socket_path
        self.auto_start = auto_start
        self.env_file = env_file
        self.workspace_path = workspace_path

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._session_id: Optional[str] = None
        self._client_id: Optional[str] = None

        # Event callback
        self._on_event: Optional[Callable[[Event], None]] = None

        # Buffer for events consumed during request-response operations
        # (e.g. create_session reads events until SessionInfoEvent; any
        # other events received in the meantime are buffered here so that
        # events() can yield them later without data loss).
        self._buffered_events: list[Event] = []

        # True while events() is actively iterating.  When set,
        # create_session() must NOT read from the socket (concurrent
        # readers on a StreamReader is a RuntimeError).  It falls back
        # to fire-and-forget and lets events() pick up the response.
        self._events_active: bool = False

    def _get_pipe_path(self) -> str:
        """Get the full Windows named pipe path."""
        # Accept a variety of user inputs and normalize to the canonical
        # Windows named pipe form: \\.\pipe\<name>
        path = self.socket_path

        # If user provided an absolute-looking pipe path but using a single
        # leading backslash (e.g. \.\pipe\jaato) or accidental concatenation
        # like \.pipejaato, try to normalize it.
        # Strip surrounding whitespace
        path = path.strip()

        # If user passed the canonical prefix already, return as-is
        if path.startswith(WINDOWS_PIPE_PREFIX):
            return path

        # If the user passed some variant containing the word 'pipe' (for
        # example: "\\.pipejaato", "\\.\pipe\\jaato", "pipe\\jaato",
        # or even "\\.\\pipe\\.pipejaato"), try to extract the final
        # name after the last occurrence of "pipe" and use that as the
        # canonical pipe name.
        lower = path.lower()
        idx = lower.rfind("pipe")
        if idx != -1:
            # everything after the last 'pipe' occurrence is the name
            name = path[idx + len("pipe"):]
            # strip separators and whitespace
            name = name.lstrip("\\/ .")
            if name:
                return f"{WINDOWS_PIPE_PREFIX}{name}"

        # Fallback: treat the whole cleaned path as a bare name
        cleaned = path
        # Remove any leading slashes/backslashes or dots
        while cleaned and (cleaned[0] in "\\/."):
            cleaned = cleaned[1:]

        return f"{WINDOWS_PIPE_PREFIX}{cleaned}"

    def _is_windows_pipe(self) -> bool:
        """Check if we're using a Windows named pipe."""
        return sys.platform == "win32"

    async def _connect_windows_pipe(self, pipe_path: str):
        r"""Connect to a Windows named pipe.

        Args:
            pipe_path: Full path to the named pipe (e.g., \\.\pipe\jaato)

        Returns:
            Tuple of (reader, writer) for the pipe connection.
        """
        logger.debug(
            "Connecting to Windows pipe. socket_path=%r, resolved pipe_path=%s",
            self.socket_path, pipe_path,
        )
        loop = asyncio.get_running_loop()
        logger.debug("Client event loop type: %s", type(loop).__name__)

        # Use a Future to capture the reader/writer from the protocol callback
        connected_future: asyncio.Future = loop.create_future()

        def client_connected_cb(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            """Called when the protocol is ready with properly initialized streams."""
            logger.debug("Client protocol callback called, streams ready")
            connected_future.set_result((reader, writer))

        # Create a protocol with callback to get properly initialized writer
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader, client_connected_cb)

        # Connect to the named pipe - this triggers connection_made -> callback
        logger.debug("Client calling create_pipe_connection...")
        transport, _ = await loop.create_pipe_connection(
            lambda: protocol,
            pipe_path,
        )
        logger.debug("Client create_pipe_connection returned, transport=%s", transport)

        # Wait for the callback to provide the reader/writer
        result = await connected_future
        logger.debug("Client got reader/writer from callback")
        return result

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected and self._writer is not None

    @property
    def connection_state(self) -> str:
        """Get detailed connection state.

        Returns:
            One of: "connected", "closing", "disconnected"
        """
        if self._connected and self._writer:
            return "connected"
        elif self._writer:
            return "closing"
        else:
            return "disconnected"

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        """Set the session ID (used by recovery client)."""
        self._session_id = value

    @property
    def client_id(self) -> Optional[str]:
        """Get the client ID assigned by server."""
        return self._client_id

    def supports_reconnection(self) -> bool:
        """Check if this client supports reconnection.

        Returns True if we have enough state to attempt reconnection
        (i.e., we have a session ID to reattach to).

        Returns:
            True if reconnection is possible.
        """
        return self._session_id is not None

    def set_event_callback(self, callback: Callable[[Event], None]) -> None:
        """Set callback for received events.

        Args:
            callback: Function called with each received event.
        """
        self._on_event = callback

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self, timeout: float = 5.0) -> bool:
        """Connect to the server.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            True if connected successfully.
        """
        if self._is_windows_pipe():
            # Windows: use named pipe connection
            pipe_path = self._get_pipe_path()
            # When auto-start is available, use a short initial probe so we
            # don't waste the time budget waiting for a server that isn't
            # running yet.  If the server IS already running, 2s is plenty
            # for the pipe to respond.
            initial_timeout = min(2.0, timeout) if self.auto_start else timeout
            try:
                self._reader, self._writer = await asyncio.wait_for(
                    self._connect_windows_pipe(pipe_path),
                    timeout=initial_timeout,
                )
                self._connected = True
            except (asyncio.TimeoutError, OSError, ConnectionRefusedError, FileNotFoundError) as e:
                if self.auto_start:
                    if not await self._start_server():
                        return False
                    # Retry connection with backoff — the daemon may need
                    # a moment after pipe creation before it can accept
                    # client connections.
                    #
                    # IMPORTANT: We must NOT use short per-attempt timeouts
                    # inside wait_for().  If wait_for() cancels the coroutine
                    # after create_pipe_connection() has already established a
                    # transport, the transport leaks and the server sees a
                    # ghost client.  Instead we use the full remaining budget
                    # and only retry on errors that prove no connection was
                    # established (ConnectionRefused, FileNotFound, OSError).
                    deadline = time.time() + timeout
                    last_err: Optional[Exception] = None
                    while True:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        try:
                            self._reader, self._writer = await asyncio.wait_for(
                                self._connect_windows_pipe(pipe_path),
                                timeout=remaining,
                            )
                            self._connected = True
                            last_err = None
                            break
                        except asyncio.TimeoutError:
                            # Timeout may mean the transport was created at
                            # the OS level — stop retrying to avoid ghosts.
                            last_err = TimeoutError(
                                f"Pipe connection timed out after {timeout}s"
                            )
                            break
                        except (OSError, ConnectionRefusedError, FileNotFoundError) as e2:
                            # These errors mean no transport was created;
                            # the server is not ready yet — safe to retry.
                            last_err = e2
                            remaining = deadline - time.time()
                            if remaining <= 0:
                                break
                            await asyncio.sleep(min(0.5, remaining))
                    if last_err is not None:
                        raise ConnectionError(f"Connection failed after auto-start: {last_err}")
                else:
                    raise ConnectionError(f"Connection failed: {e}")
        else:
            # Unix: check if socket file exists
            socket_file = Path(self.socket_path)

            if not socket_file.exists():
                if self.auto_start:
                    if not await self._start_server():
                        return False
                else:
                    raise ConnectionError(f"Socket not found: {self.socket_path}")

            # Connect to socket
            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(self.socket_path),
                    timeout=timeout,
                )
                self._connected = True
            except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
                # Socket file exists but connection failed — likely a stale
                # socket from a crashed server.  Try auto-starting.
                if self.auto_start:
                    if not await self._start_server():
                        raise ConnectionError(f"Connection failed (auto-start failed): {e}")
                    # Retry connection with backoff — the daemon may need
                    # a moment after socket creation before it can accept
                    # client connections.
                    #
                    # Only retry on ConnectionRefusedError/OSError (no
                    # transport created).  On TimeoutError, stop to avoid
                    # leaking a transport that the server already accepted.
                    deadline = time.time() + timeout
                    last_err: Optional[Exception] = None
                    while True:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        try:
                            self._reader, self._writer = await asyncio.wait_for(
                                asyncio.open_unix_connection(self.socket_path),
                                timeout=remaining,
                            )
                            self._connected = True
                            last_err = None
                            break
                        except asyncio.TimeoutError:
                            last_err = TimeoutError(
                                f"Socket connection timed out after {timeout}s"
                            )
                            break
                        except (ConnectionRefusedError, OSError) as e2:
                            last_err = e2
                            remaining = deadline - time.time()
                            if remaining <= 0:
                                break
                            await asyncio.sleep(min(0.5, remaining))
                    if last_err is not None:
                        raise ConnectionError(f"Connection failed after auto-start: {last_err}")
                else:
                    raise ConnectionError(f"Connection failed: {e}")

        # Wait for connected event
        try:
            message = await self._read_message()
            if message:
                event = deserialize_event(message)
                if isinstance(event, ConnectedEvent):
                    self._client_id = event.server_info.get("client_id")
                    # Send our working directory to the server
                    import os
                    cwd = self.workspace_path or os.getcwd()
                    await self._send_event(CommandRequest(
                        command="set_workspace",
                        args=[cwd],
                    ))
                    # Send client config with env overrides
                    await self._send_client_config()
                    return True
        except Exception as e:
            await self.disconnect()
            raise ConnectionError(f"Handshake failed: {e}")

        return False

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        self._connected = False

        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

        self._reader = None
        self._writer = None
        self._session_id = None
        self._client_id = None

    async def _send_client_config(self) -> None:
        """Send client configuration to the server.

        Sends the path to the client's .env file so the server can load
        all provider-related settings when creating sessions.
        """
        import os
        import shutil
        from dotenv import dotenv_values

        # Load client's .env file (without modifying os.environ)
        # Resolve relative env_file paths against workspace_path (if set),
        # otherwise against the process cwd (the default Path behaviour).
        env_path = Path(self.env_file)
        if not env_path.is_absolute() and self.workspace_path:
            env_path = Path(self.workspace_path) / env_path
        if env_path.exists():
            client_env = dotenv_values(env_path)
        else:
            client_env = {}

        # Helper to get value from .env or shell environment
        def get_env(key: str) -> str | None:
            return client_env.get(key) or os.environ.get(key)

        # Trace paths (for backward compatibility, still sent explicitly)
        trace_log = get_env("JAATO_TRACE_LOG")
        provider_trace = get_env("PROVIDER_TRACE_LOG")

        # Send the effective content width (terminal minus client chrome)
        # so server-side formatters render to the actual available area.
        # Panel borders: 4 chars (2 per side).  Debug line gutter: 6 chars.
        terminal_width, _ = shutil.get_terminal_size()
        content_width = terminal_width - 4  # panel borders
        if os.environ.get('JAATO_DEBUG_LINE_NUMBERS', '').lower() in ('1', 'true', 'yes'):
            content_width -= 6  # debug line number gutter (4-digit num + "│ ")

        # Build presentation context describing TUI terminal capabilities.
        # This is transmitted to the server so the model can adapt its output
        # (e.g. avoid wide tables on narrow terminals).
        presentation = {
            "content_width": content_width,
            "supports_markdown": True,
            "supports_tables": True,
            "supports_code_blocks": True,
            "supports_images": False,
            "supports_rich_text": True,
            "supports_unicode": True,
            "supports_mermaid": False,
            "supports_expandable_content": False,
            "client_type": "terminal",
        }

        # Get client's working directory (for finding config files like .lsp.json)
        working_dir = self.workspace_path or os.getcwd()

        # Always resolve to absolute path - server will check if it exists
        # This allows relative paths like "../.env" to work correctly
        env_file_abs = str(env_path.resolve())

        # Log for debugging
        import logging
        logging.getLogger(__name__).info(f"Sending env_file={env_file_abs} (exists={env_path.exists()})")

        # Send config to server
        await self._send_event(ClientConfigRequest(
            trace_log_path=trace_log,
            provider_trace_log=provider_trace,
            terminal_width=content_width,
            working_dir=working_dir,
            env_file=env_file_abs,
            presentation=presentation,
        ))

    async def _start_server(self) -> bool:
        """Auto-start the server daemon.

        Checks if the server is already running (via PID file), and if not,
        launches ``python -m server --daemon``.  The env file is NOT passed
        as a CLI argument because the server is provider-agnostic — each
        client sends its own env config via ``ClientConfigRequest`` after
        connecting.

        On Unix, if a stale socket file exists from a previous crash, it is
        removed before starting the server so the new instance can bind.

        Returns:
            True if server started (or was already running) and the IPC
            endpoint became available within the timeout.
        """
        # Check if server is already running
        pid = self._check_server_running()
        if pid:
            # Server is running, just wait for socket/pipe
            return await self._wait_for_socket()

        # On Unix, clean up stale socket file left over from a crash.
        # The server also does this on startup, but removing it here avoids
        # a race where the old file tricks _wait_for_socket into returning
        # too early.
        if not self._is_windows_pipe():
            socket_file = Path(self.socket_path)
            if socket_file.exists():
                try:
                    socket_file.unlink()
                except OSError:
                    pass  # Best-effort; the server will also try to clean up

        # Start server as daemon
        print("Starting Jaato server...")

        # On Windows, pass the resolved pipe path (e.g. \\.\pipe\jaato) rather
        # than the raw socket_path which may have been mangled by the shell
        # (e.g. MSYS2 eats backslashes: \\.\pipe\jaato -> \.pipejaato).
        # The server's simpler _get_pipe_path() would create the wrong pipe
        # name from the mangled input, causing a name mismatch.
        if self._is_windows_pipe():
            ipc_arg = self._get_pipe_path()
        else:
            ipc_arg = self.socket_path

        cmd = [
            sys.executable, "-m", "server",
            "--ipc-socket", ipc_arg,
            "--daemon",
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to start server: {e}")
            return False

        # Wait for socket/pipe to appear
        return await self._wait_for_socket()

    async def _wait_for_socket(self, timeout: float = 10.0) -> bool:
        """Wait for the IPC endpoint to become available.

        For Unix sockets, this waits for the socket file to appear.  Note
        that the file may exist before the server is actually listening;
        callers should use a retry loop for the real connection attempt
        (see ``connect()``).  For Windows named pipes, it uses
        ``WaitNamedPipeW`` to check pipe availability.

        Args:
            timeout: Maximum time to wait.

        Returns:
            True if endpoint became available.
        """
        start = time.time()

        if self._is_windows_pipe():
            # Windows: use WaitNamedPipeW to check pipe availability without
            # consuming a pipe instance.  Unlike creating a full connection
            # (which uses up a server pipe instance and requires the server to
            # create a new one), WaitNamedPipeW simply checks whether a pipe
            # instance is available for connection.
            pipe_path = self._get_pipe_path()
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.WaitNamedPipeW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong]
            kernel32.WaitNamedPipeW.restype = ctypes.c_int
            loop = asyncio.get_running_loop()

            while time.time() - start < timeout:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    break
                # Wait up to 1s per probe (or remaining time, whichever is less)
                wait_ms = min(int(remaining * 1000), 1000)
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda wms=wait_ms: kernel32.WaitNamedPipeW(pipe_path, wms),
                    )
                    if result:
                        return True
                except OSError:
                    pass
                await asyncio.sleep(0.2)
            return False
        else:
            # Unix: wait for socket file to appear.  The file may exist
            # before the server is actually listening, but we intentionally
            # do NOT probe with a real connection here — that would create
            # a ghost client on the server.  The retry loop in connect()
            # handles the listen-readiness race instead.
            socket_file = Path(self.socket_path)
            while time.time() - start < timeout:
                if socket_file.exists():
                    return True
                await asyncio.sleep(0.2)
            return False

    def _check_server_running(self) -> Optional[int]:
        """Check if server is already running.

        Returns:
            PID if running, None otherwise.
        """
        import os

        pid_file = Path(DEFAULT_PID_FILE)
        if not pid_file.exists():
            return None

        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())

            if sys.platform == "win32":
                # Windows: use ctypes to check process
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
                os.kill(pid, 0)  # Check if process exists
                return pid
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            return None

    # =========================================================================
    # Message I/O
    # =========================================================================

    async def _read_message(self) -> Optional[str]:
        """Read a length-prefixed message from the socket.

        Returns:
            The message string, or None if connection closed.
        """
        if not self._reader:
            return None

        try:
            # Read length header - use readexactly for reliable framed reading
            header = await self._reader.readexactly(HEADER_SIZE)

            length = struct.unpack(">I", header)[0]
            if length > MAX_MESSAGE_SIZE:
                raise ValueError(f"Message too large: {length}")

            # Read payload
            payload = await self._reader.readexactly(length)
            return payload.decode("utf-8")

        except asyncio.IncompleteReadError:
            # Connection closed before complete message was read
            return None
        except ConnectionResetError:
            # Connection was reset by peer
            return None

    async def _write_message(self, message: str) -> None:
        """Write a length-prefixed message to the socket."""
        if not self._writer:
            raise ConnectionError("Not connected")

        payload = message.encode("utf-8")
        header = struct.pack(">I", len(payload))
        try:
            self._writer.write(header + payload)
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            logger.debug(f"_write_message: connection lost while writing: {e}")
            # Ensure we clean up the writer/reader state
            try:
                await self.disconnect()
            except Exception:
                logger.debug("_write_message: error while disconnecting after write failure", exc_info=True)
            # Surface a ConnectionError to callers if they want to handle it
            raise ConnectionError("Connection lost") from e

    async def _send_event(self, event: Event) -> None:
        """Send an event to the server."""
        try:
            await self._write_message(serialize_event(event))
        except ConnectionError as e:
            # Log and swallow the error to avoid unhandled exceptions in
            # background tasks (stop()/other commands may be called when
            # connection is already shutting down).
            logger.debug(f"_send_event: failed to send event {type(event).__name__}: {e}")
            # Ensure disconnected state
            try:
                await self.disconnect()
            except Exception:
                logger.debug("_send_event: error while disconnecting after send failure", exc_info=True)
            return

    # =========================================================================
    # Session Management
    # =========================================================================

    async def create_session(
        self,
        name: Optional[str] = None,
        timeout: float = 60.0,
    ) -> Optional[str]:
        """Create a new session on the server.

        Sends a ``session.new`` command and, when no other coroutine is
        reading from the socket (i.e. ``events()`` is not active), waits
        for the server's ``SessionInfoEvent`` confirmation.

        When ``events()`` IS already active (e.g. the TUI starts its
        event loop before requesting a session), we cannot read from the
        same socket — that would be a concurrent-reader race.  In that
        case the command is fire-and-forget: the ``SessionInfoEvent``
        will arrive via ``events()`` and update ``_session_id`` there.

        Args:
            name: Optional session name.
            timeout: Maximum seconds to wait for session creation when
                blocking.  The server may need time to initialise the
                provider, so the default is generous.

        Returns:
            The new session ID, or None if fire-and-forget / failed /
            timed out.
        """
        args = [name] if name else []
        await self._send_event(CommandRequest(
            command="session.new",
            args=args,
        ))

        # If events() is already consuming the socket, we must not read
        # here — the SessionInfoEvent will be picked up by events().
        if self._events_active:
            return None

        # No active event consumer — read events until we receive a
        # SessionInfoEvent (success) or an ErrorEvent / timeout (failure).
        # Intermediate events are buffered for events() to yield later.
        try:
            return await asyncio.wait_for(
                self._await_session_info(), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("create_session: timed out waiting for SessionInfoEvent")
            return None

    async def _await_session_info(self) -> Optional[str]:
        """Read events until a SessionInfoEvent arrives.

        Buffers any non-target events in ``_buffered_events`` so they
        can be replayed by ``events()`` later.

        Returns:
            The session ID from the SessionInfoEvent, or None on error.
        """
        while self._connected:
            message = await self._read_message()
            if message is None:
                self._connected = False
                return None

            event = deserialize_event(message)

            if isinstance(event, SessionInfoEvent) and event.session_id:
                self._session_id = event.session_id
                # Buffer the SessionInfoEvent itself too — downstream
                # consumers (e.g. IPCRecoveryClient) may need it.
                self._buffered_events.append(event)
                return event.session_id

            if isinstance(event, ErrorEvent) and not event.recoverable:
                self._buffered_events.append(event)
                return None

            # Any other event (init progress, system messages, …) —
            # buffer it so events() can yield it later.
            self._buffered_events.append(event)

        return None

    async def attach_session(self, session_id: str) -> bool:
        """Attach to an existing session.

        Args:
            session_id: The session to attach to.

        Returns:
            True if attached successfully.
        """
        await self._send_event(CommandRequest(
            command="session.attach",
            args=[session_id],
        ))
        self._session_id = session_id
        return True

    async def get_default_session(self) -> None:
        """Get or create the default session."""
        await self._send_event(CommandRequest(
            command="session.default",
            args=[],
        ))

    async def list_sessions(self) -> None:
        """Request list of sessions (response via events)."""
        await self._send_event(CommandRequest(
            command="session.list",
            args=[],
        ))

    # =========================================================================
    # Requests
    # =========================================================================

    async def send_message(
        self,
        text: str,
        attachments: Optional[list] = None,
    ) -> None:
        """Send a message to the model.

        Args:
            text: The message text.
            attachments: Optional file attachments.
        """
        await self._send_event(SendMessageRequest(
            text=text,
            attachments=attachments or [],
        ))

    async def respond_to_permission(
        self,
        request_id: str,
        response: str,
        edited_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Respond to a permission request.

        Args:
            request_id: The permission request ID.
            response: The response (y, n, a, never, etc.).
            edited_arguments: Optional edited tool arguments (when response is "e").
        """
        await self._send_event(PermissionResponseRequest(
            request_id=request_id,
            response=response,
            edited_arguments=edited_arguments,
        ))

    async def respond_to_clarification(
        self,
        request_id: str,
        response: str,
    ) -> None:
        """Respond to a clarification question.

        Args:
            request_id: The clarification request ID.
            response: The user's answer.
        """
        await self._send_event(ClarificationResponseRequest(
            request_id=request_id,
            response=response,
        ))

    async def respond_to_reference_selection(
        self,
        request_id: str,
        response: str,
    ) -> None:
        """Respond to a reference selection request.

        Args:
            request_id: The reference selection request ID.
            response: The user's selection (e.g., "1,3,4", "all", "none").
        """
        await self._send_event(ReferenceSelectionResponseRequest(
            request_id=request_id,
            response=response,
        ))

    async def stop(self) -> None:
        """Stop current operation."""
        await self._send_event(StopRequest())

    async def execute_command(
        self,
        command: str,
        args: Optional[list] = None,
    ) -> None:
        """Execute a command.

        Args:
            command: Command name.
            args: Command arguments.
        """
        await self._send_event(CommandRequest(
            command=command,
            args=args or [],
        ))

    async def disable_tool(self, tool_name: str) -> None:
        """Disable a tool directly via registry.

        This is a fire-and-forget request that doesn't generate response events.
        Used by headless mode to disable tools before starting event handling.

        Args:
            tool_name: Name of the tool to disable.
        """
        from jaato_sdk.events import ToolDisableRequest
        await self._send_event(ToolDisableRequest(tool_name=tool_name))

    async def request_command_list(self) -> None:
        """Request the list of available commands from server.

        The response will arrive as a CommandListEvent via the event stream.
        """
        await self._send_event(CommandListRequest())

    async def request_history(self, agent_id: str = "main") -> None:
        """Request conversation history from server.

        The response will arrive as a HistoryEvent via the event stream.

        Args:
            agent_id: Which agent's history to request.
        """
        await self._send_event(HistoryRequest(agent_id=agent_id))

    # =========================================================================
    # Event Stream
    # =========================================================================

    async def events(self) -> AsyncIterator[Event]:
        """Async iterator for receiving events.

        Yields events from the server until the connection is closed or
        an error occurs. When the connection is lost, the iterator exits
        cleanly (stops yielding) rather than raising an exception.

        Any events buffered by request-response methods (e.g.
        ``create_session``) are yielded first before reading from the
        socket.

        Connection loss can be detected by:
        1. The iterator stopping (connection closed cleanly)
        2. Receiving an ErrorEvent (error during read)

        Yields:
            Events from the server.
        """
        logger.debug("events(): starting event loop")
        self._events_active = True

        try:
            # Drain events that were buffered during request-response
            # operations (e.g. create_session consuming init-progress events).
            while self._buffered_events:
                event = self._buffered_events.pop(0)
                logger.debug(f"events(): yielding buffered {type(event).__name__}")
                if self._on_event:
                    self._on_event(event)
                yield event

            while self._connected:
                try:
                    message = await self._read_message()
                    if message is None:
                        # Connection closed cleanly (server shutdown, network loss)
                        logger.debug("events(): connection closed (received None)")
                        self._connected = False
                        break

                    event = deserialize_event(message)
                    logger.debug(f"events(): received {type(event).__name__}")

                    # Auto-update session_id when receiving SessionInfoEvent
                    if isinstance(event, SessionInfoEvent) and event.session_id:
                        self._session_id = event.session_id
                        logger.debug(f"events(): session_id updated to {event.session_id}")

                    # Call callback if set
                    if self._on_event:
                        self._on_event(event)

                    yield event

                except asyncio.IncompleteReadError:
                    # Connection lost mid-message
                    logger.debug("events(): incomplete read, connection lost")
                    self._connected = False
                    break

                except ConnectionResetError:
                    # Connection reset by peer
                    logger.debug("events(): connection reset by peer")
                    self._connected = False
                    break

                except asyncio.CancelledError:
                    logger.debug("events(): cancelled")
                    raise

                except Exception as e:
                    logger.error(f"events(): error: {e}")
                    yield ErrorEvent(
                        error=str(e),
                        error_type=type(e).__name__,
                    )
        finally:
            self._events_active = False

    async def receive_events(self) -> None:
        """Receive events and call the callback.

        Runs until disconnected. Use set_event_callback() first.
        """
        async for event in self.events():
            pass  # Callback is called in events()
