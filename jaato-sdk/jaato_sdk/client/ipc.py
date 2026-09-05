r"""IPC Client for connecting to Jaato Server.

This module provides a client for connecting to the Jaato server
via Unix domain socket (Unix/Linux/macOS) or named pipe (Windows).

Usage:
    from jaato_sdk import IPCClient, EventType

    # On Unix:
    client = IPCClient("/tmp/jaato.sock")

    # On Windows:
    client = IPCClient("jaato")  # connects to \\.\pipe\jaato

    await client.connect()

    # Subscribe to specific event types (typed handlers)
    client.subscribe(
        EventType.PERMISSION_REQUESTED,
        lambda e: print(f"perm: {e.tool_name}"),
    )

    # Send a message
    await client.send_message("Hello, world!")

    # Either iterate events directly, or use drain_events() to let
    # subscribed handlers do the work:
    async for event in client.events():
        print(event)
"""

import asyncio
import json
import logging
import struct
import uuid
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from jaato_sdk.client.errors import (
    SessionCreateFailed,
    SessionNotConfirmed,
    SessionNotSent,
    SessionRefused,
)
from jaato_sdk.client._handler_registry import (
    EventHandler,
    Unsubscribe,
    _HandlerRegistry,
)
from jaato_sdk.path_boundary import require_absolute_path

logger = logging.getLogger(__name__)

from jaato_sdk.events import (
    Event,
    EventType,
    serialize_event,
    deserialize_event,
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    ClarificationBatchResponseEvent,
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
    ClientType,
    PresentationContext,
    SessionInfoEvent,
    InjectPromptRequest,
    InjectPromptResultEvent,
    ReplayMessagesRequest,
    ResolveForkPointRequest,
    PermissionAddWhitelistRequest,
    PermissionAddBlacklistRequest,
    PermissionRemoveRequest,
    PermissionClearRequest,
    PermissionSetDefaultRequest,
    PermissionPolicySnapshotRequest,
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


def _parse_protocol_version(v: str) -> Tuple[int, int]:
    """Parse a ``"MAJOR.MINOR"`` semver string into a tuple.

    Lenient — extra components ("1.0.5") are tolerated; the trailing
    parts are dropped.  Non-numeric tokens yield ``ValueError``.

    A non-string is a ``ValueError`` too, not an ``AttributeError``.
    The distinction matters because this is called from inside
    :class:`IncompatibleServerError`'s constructor to build its own
    message: an exception type the caller does not expect there turns a
    clear "your daemon is too old" into an unrelated crash during error
    reporting.  A reporter must survive the worst input it describes.
    """
    if not isinstance(v, str):
        raise ValueError(f"Protocol version must be MAJOR.MINOR, got {v!r}")
    parts = v.split(".")
    if len(parts) < 2:
        raise ValueError(f"Protocol version must be MAJOR.MINOR, got {v!r}")
    return int(parts[0]), int(parts[1])


def _protocol_compatible(server_version: str, client_min: str) -> bool:
    """Return whether ``server_version`` satisfies the client's minimum.

    Compat rule (semver-flavoured):

    - Server's MAJOR must equal the client's MAJOR.  A different major
      means the server has shape changes the client cannot parse, or
      uses fields the client expects to find but doesn't.
    - Server's MINOR must be >= the client's required minor.  Server
      minor *higher* is fine — additive optional fields the client
      hasn't been taught about yet.

    Either side malformed (or ``None``) → ``False``.  Refuse rather
    than risk parsing garbage; the caller treats this as "unknown
    protocol" and surfaces ``IncompatibleServerError``.
    """
    if not isinstance(server_version, str) or not isinstance(client_min, str):
        return False
    try:
        server_major, server_minor = _parse_protocol_version(server_version)
        client_major, client_minor = _parse_protocol_version(client_min)
    except (ValueError, TypeError):
        return False
    if server_major != client_major:
        return False
    return server_minor >= client_minor


class IncompatibleServerError(Exception):
    """Raised when the server's protocol version is incompatible.

    Non-retryable: an old (or wrong-major) server will not become
    compatible on retry.  Clients should catch this and prompt the
    operator with the protocol-version mismatch — telling them the
    *protocol* version is the actionable signal, not the package
    version (which may not have changed when shapes broke).

    Attributes:
        server_protocol: ``ConnectedEvent.protocol_version`` from the
            daemon — the wire-protocol version it speaks.
        min_protocol: The client's required minimum protocol version.
        server_version: The daemon's package version (read from
            ``server_info["server_version"]``), kept for diagnostics
            only.  When the field is absent we report ``"unknown"``.
    """

    def __init__(
        self,
        server_protocol: str,
        min_protocol: str,
        server_version: Optional[str] = None,
    ):
        self.server_protocol = server_protocol
        self.min_protocol = min_protocol
        self.server_version = server_version or "unknown"
        try:
            sm, sn = _parse_protocol_version(server_protocol)
            cm, cn = _parse_protocol_version(min_protocol)
            if sm != cm:
                hint = (
                    f"major-version mismatch (server speaks {sm}.x, client "
                    f"needs {cm}.x) — wire shapes are incompatible"
                )
            elif sn < cn:
                hint = (
                    f"server minor {sn} is below client's required minor "
                    f"{cn} — daemon is missing fields the client depends on"
                )
            else:
                hint = "version mismatch"
        except (ValueError, TypeError):
            hint = "unparseable protocol version"
        super().__init__(
            f"Server protocol {server_protocol} is not supported by this "
            f"client (requires >= {min_protocol}): {hint}. "
            f"Daemon package: {self.server_version}."
        )

    # =========================================================================
    # Backwards-compat properties
    # =========================================================================
    # Pre-1.0 callers read ``.min_version`` and used the deprecated
    # ``server_version``-driven check.  Map them to the new fields so a
    # ``except IncompatibleServerError as e: print(e.min_version)`` site
    # keeps working without code changes.

    @property
    def min_version(self) -> str:
        """Alias for ``min_protocol`` (pre-1.0 compatibility)."""
        return self.min_protocol


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

    # Minimum wire-protocol version this client speaks.  See
    # ``docs/sdk-protocol-versioning.md`` for the bump policy.
    # Override per-instance via the ``min_protocol_version`` ctor arg
    # for development against unreleased daemons.
    MIN_PROTOCOL_VERSION: str = "1.0"

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        *,
        client_type: ClientType,
        auto_start: bool = True,
        env_file: str = ".env",
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
        apparmor: bool = False,
        min_protocol_version: Optional[str] = None,
        autostart_timeout: float = 120.0,
        presentation: Optional[Any] = None,
    ):
        """Initialize the IPC client.

        Args:
            socket_path: Path to Unix domain socket or Windows pipe name.
            auto_start: Whether to auto-start server if not running.
            env_file: Path to .env file for auto-started server.
            workspace_path: Working directory sent to the server for file
                operations and sandbox scoping.  Falls back to
                ``os.getcwd()`` when not provided.  **Must be absolute** —
                the daemon has its own cwd and would resolve a relative
                path against that one, silently running the session in a
                different directory from the one this process reads back
                (issue #742).  A relative value raises
                ``RelativePathAcrossBoundaryError`` here rather than being
                absolutised, because ``../proj`` and ``~/proj`` both look
                relative and mean different things; resolve it yourself
                (e.g. ``Path(p).expanduser().resolve()``).
            config_root: Optional override for where the daemon reads
                read-only framework config (profiles, agents, prompts,
                references, completion_schemas, instructions, scripts,
                services).  When unset, the daemon falls back to
                ``<workspace_path>/.jaato/``; when set, that
                workspace-anchored search is replaced with this path.
                The ``~/.jaato/`` user-tier fallback is always honored.
                Pair with a ``workspace_path`` that does **not** contain
                a ``.jaato/`` symlink to give the agent's filesystem
                tools no visibility into the framework config.  See
                ``shared/config_resolver.py`` for the resolver contract.
                **Must be absolute**, for the same reason as
                ``workspace_path``.
            apparmor: Opt-in AppArmor confinement for sessions on this
                connection.  Defaults to ``False`` to preserve the
                long-standing IPC behavior (sessions run unconfined).
                Set to ``True`` to ask the daemon to provision a per-
                session AppArmor profile that confines the agent's
                tool plugins to ``workspace_path`` (rw),
                ``config_root`` (read-only), the standard
                ``~/.jaato/`` config, and the venv / source tree.
                Useful for orchestrator-driven harnesses where the
                agent itself is the threat surface, not the local user.
                When AppArmor is unavailable on the host (non-Linux,
                kernel module not loaded, ``apparmor_parser`` missing)
                the session falls back to running unconfined.  This
                is **not** a silent fallback: the daemon always emits
                a ``SystemMessageEvent`` to the client describing the
                outcome (style ``"info"`` with prefix
                ``[apparmor] confinement applied (...)`` when
                enforcement is in effect, style ``"warning"`` with
                prefix ``[apparmor] requested but ...`` otherwise),
                so the caller can surface it in the event loop and
                the user knows whether kernel confinement is really
                active.  See ``docs/apparmor-setup.md``.
            client_type: **Required.** Identifies the kind of client for
                server-side presentation / lifecycle filters.  Pass
                ``ClientType.TERMINAL`` for interactive TUI clients,
                ``WEB`` / ``CHAT`` for chat-shaped UIs, ``API`` for
                headless orchestrators / test harnesses / cascade
                entry-points.  No default — caller must declare intent
                explicitly so the server can apply the correct
                interactive-root filter (server 0.6.61+ strips
                ``signal_completion`` for ``TERMINAL``/``WEB``/``CHAT``
                root sessions to prevent premature termination in
                interactive contexts; ``API`` keeps it for cascade
                completion).  Reactor-spawned headless sessions
                (server 0.6.67+) automatically default to ``API`` —
                this requirement only applies to clients connecting
                via IPC / WS.
            min_protocol_version: Override the class-level
                ``MIN_PROTOCOL_VERSION`` for this connection.  Use only
                for development against unreleased daemons; production
                deployments should pin a real minimum at the class
                level so the SDK refuses to talk to incompatible servers.
            presentation: Override the display-capability context sent to
                the server at connect (``ClientConfigRequest.presentation``).
                Accepts a ``PresentationContext`` or a plain ``dict``.  When
                ``None`` (default) the client auto-derives a TUI-shaped
                context from the terminal width + ``client_type``.  Pass an
                explicit context for non-terminal clients (chat / web) whose
                capabilities differ from a terminal's — e.g. a chat client
                with ``supports_tables=False`` / ``supports_images=True`` /
                ``supports_expandable_content=True`` and a fixed narrow
                ``content_width`` — so the model adapts its output format.

        Raises:
            ValueError: if ``env_file`` is None.
            RelativePathAcrossBoundaryError: if ``workspace_path`` or
                ``config_root`` is relative (a ``ValueError`` subclass).
        """
        if env_file is None:
            raise ValueError(
                "IPCClient(env_file=...) must be a path (e.g. '.env'), not "
                "None — the IPC handshake serializes it, and None raises an "
                "opaque os.PathLike TypeError mid-connect.  Pass a real .env "
                "path (a minimal one is fine if the daemon gets provider "
                "config another way)."
            )
        # A relative ``workspace_path`` / ``config_root`` is refused HERE,
        # in the sending process, because here is the only place its
        # meaning is knowable: the daemon resolves what it receives
        # against its OWN cwd, which nothing keeps equal to this one and
        # which a daemon restart can change (issue #742).  Refused rather
        # than absolutised — "resolve it yourself" is a decision the
        # caller must make explicitly, since ``~/proj`` and ``../proj``
        # both look relative and mean very different things.
        require_absolute_path(
            workspace_path, field="workspace_path",
            origin="the daemon boundary",
        )
        require_absolute_path(
            config_root, field="config_root",
            origin="the daemon boundary",
        )
        self.socket_path = socket_path
        self.auto_start = auto_start
        self.env_file = env_file
        # Cold-daemon autostart can take ~30-60s (plugin discovery + imports);
        # the connect ``timeout`` (default 5s) is for an already-running daemon.
        # When THIS client launches the daemon, the post-launch wait + connect
        # retry budget uses this longer ``autostart_timeout`` instead.
        self.autostart_timeout = autostart_timeout
        self.workspace_path = workspace_path
        self.config_root = config_root
        self.apparmor = apparmor
        self.client_type = client_type
        # Optional caller-supplied presentation override (PresentationContext
        # or dict).  When set, it REPLACES the auto-derived terminal context at
        # config-send — the SDK hook for non-terminal (chat/web) clients.
        self._presentation = presentation
        self._min_protocol_version: str = (
            min_protocol_version or self.MIN_PROTOCOL_VERSION
        )

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._session_id: Optional[str] = None
        self._client_id: Optional[str] = None
        self._server_version: Optional[str] = None
        self._server_protocol_version: Optional[str] = None

        # Event handler registry.  Owned and dispatched from this
        # client's event loop; not thread-safe.  See _handler_registry
        # for snapshot/unsubscribe semantics.
        self._registry = _HandlerRegistry()

        # Drain task + per-consumer subscriber queues (SDK 0.13.0+).
        # Replaces the previous "single-reader + _events_active gate"
        # design.  A background drain task reads every event from the
        # socket exactly once and fans it out to:
        #
        #   * each active subscriber queue (one per ``events()``
        #     iterator and per ``_await_session_info()`` call), and
        #   * ``_buffered_events`` when no subscribers exist (so a
        #     later ``events()`` call can replay events that flowed
        #     while no consumer was attached).
        #
        # This removes the deferred-aclose race that ``_events_active``
        # could not avoid: consumers no longer read the socket
        # directly, so back-to-back ``create_session`` / ``events()``
        # patterns are race-free.  See ``_drain_loop``,
        # ``_subscribe_events``, ``_unsubscribe_events``.
        self._drain_task: Optional[asyncio.Task] = None
        self._event_subscribers: list[asyncio.Queue] = []
        self._buffered_events: list[Event] = []

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

    @property
    def server_version(self) -> Optional[str]:
        """Get the server's package version, available after connect().

        Returns the ``server_version`` string from the ``ConnectedEvent``
        server_info dict, or ``None`` if the server did not report one
        (pre-0.2.28 servers).

        **Diagnostics only** — compat is checked against
        ``server_protocol_version``.  Two daemons with different
        package versions can speak the same protocol; package version
        on its own says nothing about wire compatibility.
        """
        return self._server_version

    @property
    def server_protocol_version(self) -> Optional[str]:
        """Get the server's wire-protocol version, available after connect().

        Returns the ``protocol_version`` string from ``ConnectedEvent``,
        or ``None`` if the connection hasn't completed handshake yet.
        This is the version the compat check ran against — different
        from the daemon's package version (see ``server_version``).
        """
        return self._server_protocol_version

    def supports_reconnection(self) -> bool:
        """Check if this client supports reconnection.

        Returns True if we have enough state to attempt reconnection
        (i.e., we have a session ID to reattach to).

        Returns:
            True if reconnection is possible.
        """
        return self._session_id is not None

    # =========================================================================
    # Event Subscription API
    # =========================================================================

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Unsubscribe:
        """Subscribe to events of a specific type.

        Sync handlers run inline; async handlers are scheduled
        fire-and-forget on the current event loop. Returns an idempotent
        unsubscribe callable.
        """
        return self._registry.subscribe(event_type, handler)

    def subscribe_once(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Unsubscribe:
        """Subscribe to a single event of ``event_type`` then auto-unsubscribe."""
        return self._registry.subscribe_once(event_type, handler)

    def subscribe_all(self, handler: EventHandler) -> Unsubscribe:
        """Subscribe to every event regardless of type (catchall firehose)."""
        return self._registry.subscribe_all(handler)

    def subscribe_many(
        self,
        handlers: Dict[EventType, EventHandler],
    ) -> Unsubscribe:
        """Register multiple typed handlers in one call.

        Returns a single unsubscribe that removes all of them atomically.
        """
        return self._registry.subscribe_many(handlers)

    def _dispatch(self, event: Event) -> None:
        """Forward to the embedded handler registry."""
        self._registry.dispatch(event)

    # =========================================================================
    # High-level convenience facade
    # =========================================================================

    @classmethod
    def session(cls, **kwargs):
        """Open a session with the high-level facade (additive sugar).

        Returns an async context manager yielding a
        :class:`~jaato_sdk.client.convenience.Session` that owns the
        send-and-wait recipe — so the common path never reproduces the
        ``SESSION_TERMINATED``-only hang (PR #399)::

            async with IPCClient.session(profile="researcher", agent="pirate") as s:
                print(await s.ask("Research tide pools."))

        ``profile`` (str=named / dict=inline spec), ``agent``, ``agent_params``,
        ``cascade_driver_id`` are forwarded to :meth:`create_session` unchanged
        — both declarative and programmatic styles are preserved.  Connection
        knobs (``socket_path``, ``env_file``, ``workspace_path``, ``auto_start``,
        ``client_type``, ``connect_timeout``) and ``on_permission`` have sensible
        defaults.  See ``docs/design/sdk-convenience-layer.md``.
        """
        from .convenience import open_session
        return open_session(cls, **kwargs)

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
                    # Use the cold-start budget: we just launched the daemon.
                    deadline = time.time() + self.autostart_timeout
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
                    # Use the cold-start budget: we just launched the daemon.
                    deadline = time.time() + self.autostart_timeout
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

        return await self._handshake()

    async def _handshake(self) -> bool:
        """Post-transport handshake — shared by the IPC and WebSocket transports.

        Once the transport is open (Unix socket / Windows pipe / WebSocket),
        the wire protocol is identical: read the server's unprompted
        ``ConnectedEvent``, gate on protocol compatibility, send the workspace
        + client config, and start the single drain reader. Subclasses that
        swap the transport (see ``WSClient``) reuse this verbatim — only the
        ``_read_message`` / ``_write_message`` / connection setup differ.
        """
        # Wait for connected event
        try:
            message = await self._read_message()
            if message:
                event = deserialize_event(message)
                if isinstance(event, ConnectedEvent):
                    self._client_id = event.server_info.get("client_id")
                    self._server_version = event.server_info.get("server_version")
                    self._server_protocol_version = event.protocol_version

                    # Wire-protocol compat gate.  Refuse to keep the
                    # connection alive when the server's protocol
                    # version is incompatible — the operator's right
                    # next step is to upgrade one side, not retry.
                    if not _protocol_compatible(
                        self._server_protocol_version,
                        self._min_protocol_version,
                    ):
                        # Read what the daemon told us BEFORE tearing the
                        # connection down: ``disconnect()`` clears both
                        # fields, so building the error from instance
                        # state afterwards reported ``None`` for the two
                        # versions the message exists to name -- and
                        # ``None`` then crashed the constructor itself.
                        server_protocol = self._server_protocol_version
                        server_version = self._server_version
                        await self.disconnect()
                        raise IncompatibleServerError(
                            server_protocol=server_protocol,
                            min_protocol=self._min_protocol_version,
                            server_version=server_version,
                        )

                    # Send our working directory to the server
                    import os
                    cwd = self.workspace_path or os.getcwd()
                    await self._send_event(CommandRequest(
                        command="set_workspace",
                        args=[cwd],
                    ))
                    # Send client config with env overrides
                    await self._send_client_config()

                    # Start the drain task — single reader for the
                    # connection's lifetime.  After this point, no other
                    # code path should call ``_read_message`` directly
                    # on this client (the drain task owns it).  Consumers
                    # use ``events()`` / ``_await_session_info()`` which
                    # subscribe queues to the drain loop's fan-out.
                    self._drain_task = asyncio.create_task(
                        self._drain_loop(),
                        name=f"jaato-drain-{id(self)}",
                    )
                    return True
        except IncompatibleServerError:
            raise
        except Exception as e:
            await self.disconnect()
            raise ConnectionError(f"Handshake failed: {e}")

        return False

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        self._connected = False

        # Stop the drain task BEFORE closing the writer.  Cancelling
        # the task interrupts its in-flight ``_read_message`` await;
        # the task's finally block then puts a sentinel into every
        # subscriber queue so consumers exit cleanly.
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug(f"_drain_task ended with: {exc}")
            self._drain_task = None

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
        self._server_version = None
        self._server_protocol_version = None

    # ============================================================
    # Drain task — single reader, fan-out to subscriber queues
    # ============================================================

    async def _drain_loop(self) -> None:
        """Read events from the socket and fan out to subscribers.

        Single reader for the lifetime of the connection.  Started by
        ``connect()`` after the handshake completes; stopped by
        ``disconnect()`` (cancelled) or naturally on connection close.

        Each event flows to every active subscriber queue (one per
        ``events()`` iterator and per ``_await_session_info()`` call).
        When no subscribers exist, the event is appended to
        ``_buffered_events`` so a later ``events()`` call can replay
        it — preserving the prior behaviour where events emitted
        during ``create_session`` were visible to a subsequent
        ``async for ev in events()``.

        On exit (cancellation or read failure), every active subscriber
        queue gets a ``None`` sentinel so consumers wake from
        ``q.get()`` and exit their loop.
        """
        try:
            while self._connected:
                try:
                    message = await self._read_message()
                except asyncio.IncompleteReadError:
                    logger.debug("_drain_loop: incomplete read, connection lost")
                    break
                except ConnectionResetError:
                    logger.debug("_drain_loop: connection reset by peer")
                    break

                if message is None:
                    # Clean close from server side
                    logger.debug("_drain_loop: connection closed")
                    self._connected = False
                    break

                try:
                    event = deserialize_event(message)
                except Exception as exc:
                    logger.error(f"_drain_loop: deserialize failed: {exc}")
                    continue

                # Auto-update session_id on SessionInfoEvent.  Done here
                # in the single reader so both ``events()`` consumers and
                # ``_await_session_info()`` see the same value.
                if isinstance(event, SessionInfoEvent) and event.session_id:
                    self._session_id = event.session_id

                # Handler-based dispatch (subscribe()/unsubscribe()).
                self._dispatch(event)

                # Iterator-based fan-out.  Snapshot the subscribers list
                # so a concurrent unsubscribe doesn't break the loop;
                # ``put_nowait`` is sync and won't block.
                if self._event_subscribers:
                    for q in list(self._event_subscribers):
                        try:
                            q.put_nowait(event)
                        except asyncio.QueueFull:
                            logger.warning(
                                "_drain_loop: subscriber queue full; "
                                "dropping %s",
                                type(event).__name__,
                            )
                else:
                    # No active subscriber — buffer for replay on the
                    # next ``events()`` / ``_await_session_info()`` call.
                    self._buffered_events.append(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"_drain_loop: unexpected error: {exc}", exc_info=True)
        finally:
            # Wake any active consumers so they observe disconnection.
            for q in list(self._event_subscribers):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass

    def _subscribe_events(self) -> asyncio.Queue:
        """Subscribe a fresh queue to the drain loop's fan-out.

        Drains any pending ``_buffered_events`` into the queue first so
        the consumer sees events emitted before subscription (typically
        from a recent ``create_session`` whose response events flowed
        before ``events()`` was called).

        Returns:
            A queue that receives events; ``None`` is the disconnect
            sentinel.  Callers must call ``_unsubscribe_events`` (e.g.
            in a ``finally`` block) when done.
        """
        q: asyncio.Queue = asyncio.Queue()
        for ev in self._buffered_events:
            q.put_nowait(ev)
        self._buffered_events.clear()
        self._event_subscribers.append(q)
        return q

    def _unsubscribe_events(self, q: asyncio.Queue) -> None:
        """Remove a previously-subscribed queue.  Idempotent."""
        try:
            self._event_subscribers.remove(q)
        except ValueError:
            pass

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

        # Trace paths (for backward compatibility, still sent explicitly).
        # The DAEMON opens these files, so a relative value would land in
        # the daemon's cwd rather than beside the workspace the author of
        # the .env meant (issue #742, same mechanism as workspace_path).
        # Unlike ``workspace_path`` these are resolved rather than
        # refused: they come from a file INSIDE the workspace, so "beside
        # the workspace" is what a relative entry there unambiguously
        # means — and a log path is not worth failing a connect over.
        workspace_base = Path(self.workspace_path) if self.workspace_path \
            else Path.cwd()

        def abs_env_path(key: str) -> str | None:
            raw = get_env(key)
            if not raw:
                return raw
            candidate = Path(raw).expanduser()
            if candidate.is_absolute():
                return str(candidate)
            return str((workspace_base / candidate).resolve())

        trace_log = abs_env_path("JAATO_TRACE_LOG")
        provider_trace = abs_env_path("PROVIDER_TRACE_LOG")

        # Send the effective content width (terminal minus client chrome)
        # so server-side formatters render to the actual available area.
        # Panel borders: 4 chars (2 per side).  Debug line gutter: 6 chars.
        terminal_width, _ = shutil.get_terminal_size()
        content_width = terminal_width - 4  # panel borders
        if os.environ.get('JAATO_DEBUG_LINE_NUMBERS', '').lower() in ('1', 'true', 'yes'):
            content_width -= 6  # debug line number gutter (4-digit num + "│ ")

        # Build the presentation context transmitted to the server so the model
        # can adapt its output (e.g. avoid wide tables on narrow terminals).
        # A caller-supplied override (the ``presentation=`` ctor param) wins —
        # the hook for non-terminal clients (chat/web) whose capabilities differ
        # from a TUI's.  Accepts a PresentationContext or a plain dict; falls
        # back to the auto-derived terminal context otherwise.
        if self._presentation is not None:
            presentation_payload = (
                self._presentation.to_dict()
                if isinstance(self._presentation, PresentationContext)
                else dict(self._presentation)
            )
        else:
            presentation_payload = PresentationContext(
                content_width=content_width,
                client_type=self.client_type,
            ).to_dict()

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
            working_dir=working_dir,
            config_root=self.config_root,
            env_file=env_file_abs,
            apparmor=self.apparmor,
            presentation=presentation_payload,
        ))

    def _endpoint_is_live(self) -> bool:
        """Whether the socket/pipe actually ACCEPTS a connection (not just exists).

        The authoritative "is the daemon up" signal — unlike a pidfile PID
        (which can be a recycled, unrelated process) or a socket *file* (which
        can be stale after a crash).  Gates trusting the pidfile in
        :meth:`_start_server` so a reused PID can't block auto-start on a dead
        socket.
        """
        if self._is_windows_pipe():
            try:
                return bool(self._check_pipe_exists())
            except Exception:
                return False
        import socket as _socket
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            return s.connect_ex(self.socket_path) == 0
        except OSError:
            return False
        finally:
            s.close()

    async def _start_server(self) -> bool:
        """Auto-start the server daemon.

        Checks if the server is already running (via PID file and, on
        Windows, via pipe existence probe), and if not, launches
        ``python -m server --daemon``.  The env file is NOT passed as a CLI
        argument because the server is provider-agnostic — each client sends
        its own env config via ``ClientConfigRequest`` after connecting.

        On Unix, if a stale socket file exists from a previous crash, it is
        removed before starting the server so the new instance can bind.

        Returns:
            True if server started (or was already running) and the IPC
            endpoint became available within the timeout.
        """
        # Check if server is already running.  A live pidfile PID alone is NOT
        # proof the daemon is up: ``os.kill(pid, 0)`` only confirms SOME
        # process exists at that PID, and PIDs get recycled.  Require the
        # endpoint to actually accept a connection — otherwise a stale pidfile
        # (dead daemon whose PID was reused by an unrelated process) makes us
        # wait on a dead socket forever instead of relaunching.
        pid = self._check_server_running()
        if pid:
            if self._endpoint_is_live():
                # Genuinely running — wait for the socket/pipe and attach.
                return await self._wait_for_socket()
            # PID alive but endpoint dead → stale daemon / reused PID.  Clear
            # the stale pidfile so the relaunch below starts from a clean slate.
            logger.info(
                "pidfile %s -> PID %s is alive but the endpoint is not "
                "listening (stale daemon / reused PID); relaunching",
                DEFAULT_PID_FILE, pid,
            )
            try:
                Path(DEFAULT_PID_FILE).unlink()
            except OSError:
                pass

        # On Windows, the PID-file check can fail even when the server IS
        # running (e.g. stale PID, ctypes truncation on 64-bit, or the
        # daemon hasn't written its PID file yet).  A named-pipe probe is
        # a more reliable indicator: if the pipe exists, a server owns it.
        if self._is_windows_pipe():
            try:
                if self._check_pipe_exists():
                    logger.debug(
                        "Named pipe exists — server is running, "
                        "skipping auto-start"
                    )
                    return True
            except Exception:
                pass  # Best-effort; fall through to normal start

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

        # Wait for the socket/pipe to appear.  We just launched a COLD daemon
        # (plugin discovery + imports) — use the longer autostart budget, not
        # the short already-running connect timeout.
        return await self._wait_for_socket(timeout=self.autostart_timeout)

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

    def _check_pipe_exists(self) -> bool:
        """Check if the Windows named pipe already exists.

        Uses ``WaitNamedPipeW`` with a minimal timeout to probe for the pipe
        without consuming a pipe instance.  Returns ``True`` if the pipe
        exists (server is running), even if all instances are currently busy.

        Returns:
            True if the pipe exists, False otherwise.
        """
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.WaitNamedPipeW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong]
        kernel32.WaitNamedPipeW.restype = ctypes.c_int

        pipe_path = self._get_pipe_path()

        # Probe with 1 ms timeout — fast enough for a presence check.
        result = kernel32.WaitNamedPipeW(pipe_path, 1)
        if result:
            return True

        # WaitNamedPipeW returned 0.  Distinguish "pipe exists but busy"
        # (ERROR_SEM_TIMEOUT = 121) from "pipe not found"
        # (ERROR_FILE_NOT_FOUND = 2).
        error = ctypes.get_last_error()
        ERROR_SEM_TIMEOUT = 121
        return error == ERROR_SEM_TIMEOUT

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

    async def _send_event(self, event: Event) -> bool:
        """Send an event to the server.  Returns whether it actually went.

        The swallow-and-disconnect behaviour is deliberate and unchanged:
        ``stop()`` and other commands may be called while the connection is
        already shutting down, and an unhandled exception in a background
        task is worse than a dropped command.

        What changed is that the answer is now AVAILABLE.  Returning ``None``
        made "sent" and "the socket is gone" the same outcome to every caller,
        and ``create_session`` — which had just been told the send failed, one
        line earlier, in this method — went on to wait the full 60s for a
        reply to a command that never left the process, then reported a
        TIMEOUT.  Measured: 1.00s of a 1.00s budget with the write raising
        immediately.  The blame landed on the daemon for a local socket fault.

        Existing callers ignore the return and are unaffected; a caller that
        cannot act sensibly on a silently-dropped command should check it.
        """
        try:
            await self._write_message(serialize_event(event))
            return True
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
            return False

    # =========================================================================
    # Session Management
    # =========================================================================

    async def create_session(
        self,
        name: Optional[str] = None,
        profile: Optional[Union[str, Dict[str, Any]]] = None,
        agent: Optional[str] = None,
        agent_params: Optional[Dict[str, str]] = None,
        cascade_driver_id: Optional[str] = None,
        sibling_name: Optional[str] = None,
        timeout: float = 60.0,
    ) -> str:
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
            profile: Either a profile **name** (str) referencing a
                file in ``.jaato/profiles/`` on the server, **or** an
                inline **spec dict** with the same shape — recognised
                keys include ``model`` (required), ``provider``,
                ``plugins``, ``plugin_configs``, ``system_instructions``,
                ``gc``, ``env``, ``max_turns``, ``runtime_limits``,
                ``model_tiers``, ``completion_payload_schema``.  The
                server validates the dict and rejects it with a clear
                ``ErrorEvent`` if ``model`` is missing.  The two forms
                are mutually exclusive — pass one or the other.
            agent: Optional agent name — WHO the session is.  Orthogonal
                to ``profile``: the agent is the session's persona, the
                profile its capabilities, and they compose freely.

                The agent is NOT "the session's system instructions",
                though it is easy to describe it that way.  Its rendered
                markdown is ONE LAYER of an assembly that also carries the
                ``.jaato/instructions/`` base layer, plugin instructions,
                framework constants and the untrusted-content boundary —
                and ``suppress_base_instructions`` can drop every one of
                those layers EXCEPT the agent and its plugins.  The
                instructions are how an agent reaches a turn; the agent is
                what persists across turns, sessions and profiles.
            agent_params: Parameter values for the agent's ``{{param}}``
                placeholders.  Only used when *agent* is specified.
            sibling_name: Cascade-scoped ADDRESS other sessions use to
                reach this one via ``send_to_sibling`` — the same string
                they pass, so there is no translation between what you
                set and what they address.  Shape
                ``^[a-z0-9][a-z0-9_-]{0,31}$``, unique within the
                cascade; the server refuses a malformed or already-taken
                name at creation rather than producing a session nobody
                can address.  ``None`` (default) = not sibling-addressable.
            cascade_driver_id: Phase 2 cascade-sharing (server
                0.6.144+) tenant ID identifying the cascade this
                session belongs to.  Opaque UTF-8 string; UUID
                recommended.  Sessions sharing the same ID reuse
                the same pool slot — warm imports + warm plugin
                state + warm LSP server connections survive across
                cascade stages.  ``None`` (default) = standalone
                session, no slot reuse.  Generate one ID per
                cascade (``uuid.uuid4().hex``) and pass it on every
                ``session.new`` for that cascade.  Only the top-level
                cascade-driver supplies the ID.

                Subagents inherit the runner's WARM SLOT, not this
                identifier: a subagent is a runtime-level session
                (``JaatoRuntime.create_session``, which takes no
                cascade parameter), never enters the daemon's session
                table, and carries no ``cascade_driver_id`` of its
                own.  It is therefore NOT addressable as a sibling —
                it is reached by its parent through
                ``send_to_subagent``.  The previous wording said
                subagents "inherit automatically via the shared
                runner", which reads as inheriting the ID and implies
                the opposite fact about who appears in a sibling
                roster.  See ``docs/design/runner-cascade-sharing.md``.
            timeout: Maximum seconds to wait for session creation when
                blocking.  The server may need time to initialise the
                provider, so the default is generous.

        Returns:
            The new session ID.  Never ``None`` — a failure raises.

        Raises:
            SessionNotSent: the command never left this process (the socket
                write failed).  Nothing was created; retry after
                reconnecting.
            SessionRefused: the daemon answered and refused — unknown
                profile/agent, invalid spec, failed spawn-payload validation,
                exhausted budget, provider auth.  The daemon's own reason is
                carried on the exception; it is not summarised or guessed at.
                Nothing was created; retry is futile unless the request
                changes.
            SessionNotConfirmed: the command was sent and no answer arrived
                (timeout, or the connection dropped).  **A session may exist
                on the daemon.**  ``session.new`` has no idempotency key, so
                retrying makes a SECOND session with its own runner and pool
                slot — check ``list_sessions()`` first.
            TypeError: If ``profile`` is not None, str, or dict.

        All three share the base ``SessionCreateFailed``; catch that to treat
        every creation failure alike, and ``.may_exist`` to branch on the only
        axis that changes what a caller should DO.

        IT USED TO RETURN ``None`` FOR ALL OF THEM.  Measured, the five
        failure paths returned the same ``None`` from the same call and four
        were indistinguishable even by elapsed time — so a caller could not
        tell "I never sent it" from "the daemon said no" from "a session may
        be running right now".  Two of this repository's own callers did not
        even read the value, which is how a create failure became invisible:
        headless mode went on to set policies and send prompts with no
        session.
        """
        args: List[str] = [name] if name else []
        payload: Optional[Dict[str, Any]] = None
        # Correlate this create with the event that answers it.  The wait below
        # used to accept ANY SessionInfoEvent carrying a session_id, and
        # ``_subscribe_events`` drains the buffered-event list into each new
        # subscription — so a stale event from an earlier create satisfied a
        # later wait and returned an id this call never created.
        req_id = f"req_{uuid.uuid4().hex[:16]}"

        if isinstance(profile, str):
            args.extend(["--profile", profile])
        elif isinstance(profile, dict):
            payload = {"spec": profile}
        elif profile is not None:
            raise TypeError(
                f"create_session: 'profile' must be str (profile name) or "
                f"dict (inline spec), got {type(profile).__name__}"
            )

        if agent:
            args.extend(["--agent", agent])
        if agent_params:
            for key, value in agent_params.items():
                args.append(f"{key}={value}")
        # Phase 2 cascade-sharing (server 0.6.144+): forward the
        # cascade tenant ID so the daemon's PoolManager can reuse a
        # slot already affined to this cascade.  Append AFTER agent
        # + agent_params so the argv flag sits at a stable position
        # for log diffing.  Server-side parser accepts any order.
        if cascade_driver_id:
            args.extend(["--cascade-driver-id", cascade_driver_id])
        if sibling_name:
            args.extend(["--sibling-name", sibling_name])
        # ``payload`` is the documented generic escape hatch; the request id
        # rides it so no new CommandRequest field is needed.
        payload = dict(payload or {})
        payload["request_id"] = req_id
        sent = await self._send_event(CommandRequest(
            command="session.new",
            args=args,
            payload=payload,
        ))
        if not sent:
            # FAIL FAST.  Waiting here would burn the whole timeout on a reply
            # to a command that never left the process, and then blame the
            # daemon for a local socket fault.
            raise SessionNotSent(
                "session.new was not sent — the connection dropped while "
                "writing it, so the daemon never saw the request and no "
                "session was created.  Reconnect before retrying."
            )

        # Wait for the daemon's SessionInfoEvent via the drain loop.
        # SDK 0.13.0+: no more ``_events_active`` gate — the drain task
        # delivers events to every subscriber concurrently, so a
        # back-to-back ``events() → create_session`` pattern works
        # race-free even if the previous events() iterator's aclose()
        # hasn't fully completed yet.
        try:
            return await asyncio.wait_for(
                self._await_session_info(req_id), timeout=timeout
            )
        except asyncio.TimeoutError:
            # NOT CONFIRMED, not "not created": the command WAS sent, so the
            # daemon may have made the session and only the answer was lost.
            # ``session.new`` has no idempotency key -- ``request_id`` is
            # echoed for correlation, never used to dedupe -- so a blind retry
            # makes a SECOND session with its own runner and pool slot.
            logger.warning(
                "create_session: no answer within %ss — a session MAY have "
                "been created; look for it before creating another",
                timeout,
            )
            raise SessionNotConfirmed(
                f"session.new was sent but not answered within {timeout}s. "
                "The daemon may have created the session and only the "
                "confirmation was lost — retrying may create a SECOND "
                "session. Check list_sessions() first.",
                cause="timeout",
            ) from None

    #: Wire-protocol minor from which the daemon echoes ``request_id`` on the
    #: events answering ``session.new``.  Gated on the PROTOCOL version, not the
    #: package version -- ``server_version`` is diagnostics-only and says
    #: nothing about wire shape (two daemons can differ in package and speak
    #: the same protocol).  Below 1.1 nothing echoes, so requiring the id would
    #: make every create hang.
    MIN_CORRELATION_PROTOCOL = "1.1"

    def _correlates(self, event: Any, request_id: Optional[str]) -> bool:
        """Does ``event`` answer the request identified by ``request_id``?

        Correlation is what makes the wait about THIS call.  Matching on shape
        alone -- "any SessionInfoEvent with a session_id" -- let a stale event
        from an earlier create satisfy a later wait, because
        ``_subscribe_events`` drains the buffered-event list into every new
        subscription.  A refused ``sibling_name`` reproduced it on demand; any
        two rapid ``session.new`` calls could hit it.

        Falls back to accepting an uncorrelated event ONLY when the daemon is
        too old to echo the id, and says so once.  Requiring the echo
        unconditionally would hang every call against an older daemon;
        accepting it silently would leave the bug in place with nothing to
        notice.
        """
        if request_id is None:
            return True                      # caller did not ask to correlate
        got = getattr(event, "request_id", None)
        if got is not None:
            return got == request_id
        if not _protocol_compatible(
                self.server_protocol_version, self.MIN_CORRELATION_PROTOCOL):
            if not getattr(self, "_warned_no_correlation", False):
                self._warned_no_correlation = True
                logger.warning(
                    "daemon protocol %s predates session.new request "
                    "correlation (needs >= %s); a concurrent or refused create "
                    "may return another call's session id",
                    self.server_protocol_version,
                    self.MIN_CORRELATION_PROTOCOL,
                )
            return True
        return False

    async def _await_session_info(
        self, request_id: Optional[str] = None,
    ) -> str:
        """Subscribe to the drain loop and wait for SessionInfoEvent.

        Filters the subscriber queue for ``SessionInfoEvent`` (success)
        or ``ErrorEvent`` (failure — any ``ErrorEvent`` terminates the
        wait, regardless of its ``recoverable`` flag).  The daemon
        emits ALL ``session.new`` failures (profile-not-found,
        agent-not-found, invalid spec, spawn-payload validation) with
        ``recoverable=True`` so they can be surfaced to the user
        without forcing a client reconnect, but from the caller's
        perspective they are still terminal for THIS create_session
        call.  Pre-fix the SDK silently swallowed them and returned
        ``None`` only after the asyncio timeout fired, producing the
        "daemon stalled" symptom (see project_backlog and 2026-06-06
        Bug-B investigation).

        When an ``ErrorEvent`` arrives, it is logged at WARNING with
        the daemon-supplied error type + message so the failure cause
        is visible in the SDK consumer's logs.

        When this method is the only subscriber active, non-target
        events seen along the way are saved to ``_buffered_events``
        so a subsequent ``events()`` call can replay them — preserving
        the pre-0.13.0 behaviour where ``events()`` after
        ``create_session`` could yield init-progress / system-message
        events emitted during session creation.  When ``events()`` is
        concurrently subscribed, those events are already being yielded
        directly to the consumer's iterator and are NOT re-buffered
        (avoiding duplicates).

        Returns:
            The session ID from the ``SessionInfoEvent``.

        Raises:
            SessionRefused: the daemon answered with an ``ErrorEvent``.
            SessionNotConfirmed: the connection dropped before any answer.

        It no longer returns ``None``.  It used to return it for BOTH of the
        above, and the caller then returned that same ``None`` for a timeout
        and for a failed send — collapsing five distinct outcomes, with
        opposite correct responses, into one value.
        """
        q = self._subscribe_events()
        # Events to re-buffer for a future events() call, but only when
        # this subscription is solo (no concurrent events() iterator).
        incidental: list[Event] = []
        try:
            while True:
                event = await q.get()
                if event is None:
                    # drain loop signalled disconnection.  Same reasoning as
                    # the timeout: the command was already on the wire, so the
                    # daemon may have created the session before the socket
                    # went.  Unknown, not "no".
                    if len(self._event_subscribers) == 1 and incidental:
                        self._buffered_events.extend(incidental)
                    raise SessionNotConfirmed(
                        "the connection dropped before session.new was "
                        "answered. The daemon may have created the session — "
                        "retrying may create a SECOND one.",
                        cause="disconnect",
                    )

                solo = len(self._event_subscribers) == 1

                if isinstance(event, SessionInfoEvent) and event.session_id:
                    if not self._correlates(event, request_id):
                        if solo:
                            incidental.append(event)
                        continue
                    self._session_id = event.session_id
                    if solo:
                        incidental.append(event)
                        self._buffered_events.extend(incidental)
                    return event.session_id

                if isinstance(event, ErrorEvent):
                    if not self._correlates(event, request_id):
                        if solo:
                            incidental.append(event)
                        continue
                    logger.warning(
                        "create_session: daemon reported error "
                        "(error_type=%s, recoverable=%s): %s",
                        event.error_type,
                        event.recoverable,
                        event.error,
                    )
                    if solo:
                        incidental.append(event)
                        self._buffered_events.extend(incidental)
                    # The daemon STATED the reason.  Carry it; do not
                    # summarise it into a likely cause -- the caller that
                    # used to guess "check provider auth" was wrong for
                    # every refusal that was not an auth failure.
                    raise SessionRefused(
                        f"the daemon refused session.new: {event.error}",
                        error_type=event.error_type,
                    )

                # Non-target event — track for re-buffer when solo.
                if solo:
                    incidental.append(event)
        finally:
            self._unsubscribe_events(q)

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

    async def end_session(self) -> None:
        """Terminate the currently-attached session.

        Sends ``session.end`` — the server stops the session's
        in-flight activity and emits a ``[SESSION_TERMINATED]``
        marker so attached clients know the session is no longer
        active.  The session record itself stays on disk; use
        :meth:`delete_session` to purge it.
        """
        await self._send_event(CommandRequest(
            command="session.end",
            args=[],
        ))

    async def delete_session(self, session_id: str) -> None:
        """Permanently delete a session by ID.

        Sends ``session.delete`` — the server removes both
        in-memory state and the on-disk journal for the named
        session.  Response arrives via the event stream as a
        ``SystemMessageEvent`` ("Session 'X' deleted." on success;
        "Session 'X' not found." otherwise).

        Args:
            session_id: The session to delete.  Must be a known
                session ID (visible in :meth:`list_sessions`).
        """
        await self._send_event(CommandRequest(
            command="session.delete",
            args=[session_id],
        ))

    async def list_profiles(self) -> None:
        """Request list of available agent profiles.

        The server responds with a ``SessionProfilesEvent`` containing
        profile summaries discovered from ``.jaato/profiles/``.
        """
        await self._send_event(CommandRequest(
            command="session.profiles",
            args=[],
        ))

    # =========================================================================
    # Requests
    # =========================================================================

    @staticmethod
    def _normalize_attachments(attachments: Optional[list]) -> List[Dict[str, Any]]:
        """Normalize user-message attachments to the canonical wire shape
        ``{mime_type, data: base64-str, display_name}`` (client-expanded — the
        daemon/runner can't read client-side paths, esp. cross-host WS).

        Accepts, per item:
          - a file-path ``str`` → read bytes, base64-encode, guess mime from ext
          - a ``dict`` with ``bytes`` ``data`` → base64-encode it
          - a ``dict`` with base64-``str`` ``data`` → pass through unchanged
        Unknown shapes are skipped (no fabricated content).
        """
        import base64
        import mimetypes
        import os
        out: List[Dict[str, Any]] = []
        for a in attachments or []:
            if isinstance(a, str):
                with open(a, "rb") as fh:
                    raw = fh.read()
                out.append({
                    "mime_type": mimetypes.guess_type(a)[0]
                                 or "application/octet-stream",
                    "data": base64.b64encode(raw).decode("ascii"),
                    "display_name": os.path.basename(a),
                })
            elif isinstance(a, dict):
                d = dict(a)
                data = d.get("data")
                if isinstance(data, (bytes, bytearray)):
                    d["data"] = base64.b64encode(bytes(data)).decode("ascii")
                out.append(d)
        return out

    async def send_message(
        self,
        text: str,
        attachments: Optional[list] = None,
        parallel_tools: Optional[bool] = None,
    ) -> None:
        """Send a message to the model.

        Args:
            text: The message text.
            attachments: Optional user-message attachments — each a file-path
                ``str`` OR a ``{mime_type, data, display_name}`` dict (``data``
                as raw ``bytes`` or a base64 ``str``).  Normalized client-side
                to the canonical wire shape ``{mime_type, data: base64-str,
                display_name}`` and delivered to the model's multimodal path
                (gated by the provider's vision/input modality).
            parallel_tools: Per-call override for parallel tool execution.
                ``None`` (default) keeps the env-configured behaviour
                (``JAATO_PARALLEL_TOOLS``).  ``True`` / ``False`` forces
                parallel / sequential tool execution for this turn only.
        """
        await self._send_event(SendMessageRequest(
            text=text,
            attachments=self._normalize_attachments(attachments),
            parallel_tools=parallel_tools,
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

    async def respond_to_clarification_batch(
        self,
        request_id: str,
        answers: List[str],
        *,
        cancelled: bool = False,
    ) -> None:
        """Respond to a batched clarification — all answers at once.

        The blessed public form for clients that receive every question in
        one ``ClarificationBatchEvent`` and answer them together, rather
        than calling ``respond_to_clarification`` per question.  ``answers``
        is an ordered list, one entry per question by index.

        Mandatory for a ``batch_only`` batch (runner-tier sessions): no
        per-question events follow one, so this is the only reply that
        unblocks the tool call.

        Args:
            request_id: The clarification request ID.
            answers: Ordered answers, one per question (by index).
            cancelled: Abandon the clarification instead of answering it.
                The tool returns ``{"cancelled": True}`` to the model and
                the turn continues; ``answers`` is ignored.
        """
        await self._send_event(ClarificationBatchResponseEvent(
            request_id=request_id,
            answers=answers,
            cancelled=cancelled,
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

    async def respond_to_tool_execution(
        self,
        call_id: str,
        result: str = "",
        error: str = "",
    ) -> None:
        """Return the result of a client-side tool execution.

        Sends ``ToolExecuteResultEvent`` so the server can resume the
        model loop with the tool's result.  Caller-side counterpart of
        the ``ToolExecuteRequestEvent`` the server emits when the
        model invokes a client-registered tool (see
        :meth:`register_client_tools`).

        Args:
            call_id: The ``call_id`` from the originating
                ``ToolExecuteRequestEvent``.  Server uses this to
                correlate the response with the in-flight tool call.
            result: JSON-encoded tool result.  Empty string when
                ``error`` is set.
            error: Error message when execution failed.  Empty when
                ``result`` is set.  Setting both is undefined.
        """
        from jaato_sdk.events import ToolExecuteResultEvent
        await self._send_event(ToolExecuteResultEvent(
            call_id=call_id,
            result=result,
            error=error,
        ))

    async def register_client_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Register client-provided ("host") tools the agent can call.

        Each entry: ``{"name", "description", "parameters", "handler"}`` (plus
        optional ``"timeout"`` ms / ``"auto_approve"``).  ``handler(args) -> Any``
        runs when the agent invokes the tool; its return (JSON-encoded if not a
        str) is sent back as the result.  Register **before** ``create_session``
        so the schema reaches the runner-tier model (mid-session registration
        isn't seen until a follow-up lands the runner mid-session push).
        """
        from jaato_sdk.events import ToolsRegisterClientRequest, EventType
        if not hasattr(self, "_host_tool_handlers"):
            self._host_tool_handlers: Dict[str, Any] = {}
            self.subscribe(
                EventType.TOOL_EXECUTE_REQUEST, self._on_tool_execute_request)
        for t in tools:
            if t.get("handler"):
                self._host_tool_handlers[t["name"]] = t["handler"]
        wire = [{k: v for k, v in t.items() if k != "handler"} for t in tools]
        await self._send_event(
            ToolsRegisterClientRequest(tools=wire, categories={}))

    def _on_tool_execute_request(self, event: Any) -> None:
        """Run the registered host-tool handler for an agent tool call and send
        the result back via :meth:`respond_to_tool_execution`."""
        import asyncio
        import json
        fn = getattr(self, "_host_tool_handlers", {}).get(event.tool_name)

        async def _run() -> None:
            if fn is None:
                await self.respond_to_tool_execution(
                    event.call_id,
                    error=f"no handler for host tool {event.tool_name!r}")
                return
            try:
                out = fn(event.tool_args)
                if asyncio.iscoroutine(out):
                    out = await out
                result = out if isinstance(out, str) else json.dumps(out)
                await self.respond_to_tool_execution(event.call_id, result=result)
            except Exception as exc:  # report the failure to the model
                await self.respond_to_tool_execution(event.call_id, error=str(exc))

        asyncio.create_task(_run())

    async def stop(self) -> None:
        """Stop current operation."""
        await self._send_event(StopRequest())

    async def execute_command(
        self,
        command: str,
        args: Optional[list] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Execute a command.

        Args:
            command: Command name.
            args: Command arguments.
            payload: Optional structured body for verbs that take one
                (``CommandRequest.payload``) — used where a dict is the
                natural shape and squeezing it into positional ``args``
                would be lossy, e.g. ``cascade.budget.set``.
        """
        await self._send_event(CommandRequest(
            command=command,
            args=args or [],
            payload=payload,
        ))

    # ---- typed wake-primitive methods (see _wake_client) ----
    async def bind_wake(self, wake_ref: str, trust_keys: list, *,
                        timeout: float = 30.0):
        """Declare a wake binding for this session; await the typed result.
        See :func:`jaato_sdk.client._wake_client.bind_wake`."""
        from ._wake_client import bind_wake
        return await bind_wake(self, wake_ref, trust_keys, timeout=timeout)

    async def unbind_wake(self, wake_ref: str, *, timeout: float = 30.0):
        """Remove this session's wake binding; await the typed result.
        See :func:`jaato_sdk.client._wake_client.unbind_wake`."""
        from ._wake_client import unbind_wake
        return await unbind_wake(self, wake_ref, timeout=timeout)

    async def cascade_register(self, cascade_driver_id: str,
                              role: str = "observer",
                              event_types: Optional[list] = None) -> None:
        """Register as a cascade owner/observer (event CLASSES or names).
        See :func:`jaato_sdk.client._wake_client.cascade_register`."""
        from ._wake_client import cascade_register
        await cascade_register(self, cascade_driver_id, role, event_types)

    async def cascade_budget_set(
        self,
        cascade_driver_id: str,
        limits: dict,
        degrade: Optional[list] = None,
    ) -> None:
        """Declare a cascade's AGGREGATE budget ceiling (owner-side).

        ``limits`` maps dimension -> ceiling (``usd`` / ``tokens`` /
        ``seconds`` / ``tool_calls`` / ``turns``); omit a dimension to leave
        it unbounded.  ``degrade`` is the optional rung ladder, same grammar
        as a profile's ``budget_control``.

        Every session subsequently created with this ``cascade_driver_id``
        has its own ceiling clamped to ``min(profile, cascade_remaining)``
        at spawn, and a cascade with no headroom left REFUSES the spawn
        rather than starting a child that cannot run a turn.

        Declared here rather than on a profile because a cap is a runtime
        aggregate over one live cid, not a property of a reusable template.

        Fire-and-forget; the daemon acknowledges with a
        ``SystemMessageEvent`` (and reports authoring errors as
        ``ErrorEvent``).
        """
        body: dict = {"limits": dict(limits)}
        if degrade:
            body["degrade"] = list(degrade)
        await self.execute_command(
            "cascade.budget.set", [cascade_driver_id], payload=body)

    async def cascade_budget_get(self, cascade_driver_id: str) -> None:
        """Request a cascade's remaining headroom.

        The daemon replies with a ``SystemMessageEvent`` whose ``message`` is
        JSON: ``{cascade_driver_id, declared, limits, remaining,
        usage_fraction, pressure}`` — or ``{cascade_driver_id, declared:
        false}`` when the cid has no budget.

        Same fire-and-forget shape as :meth:`list_profiles`: the reply
        arrives on the event stream rather than as a return value.  Reading
        ``remaining`` between stages is the client-side witness of the pool
        depleting — independent corroboration of the daemon's spawn-time
        clamp rather than only the framework reporting its own decision.
        """
        await self.execute_command("cascade.budget.get", [cascade_driver_id])

    async def cascade_budget_clear(self, cascade_driver_id: str) -> None:
        """Drop a cascade's budget pool (finished cascade / clean re-run)."""
        await self.execute_command("cascade.budget.clear", [cascade_driver_id])

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
    # SDK feature parity — session-primitive verbs
    #
    # Typed methods over the public-side primitives
    # ``JaatoSession.inject_prompt`` / ``replay_messages`` /
    # ``resolve_fork_point``.  Premium's ``session_ops`` plugin builds
    # higher-level model-callable tools on top of these same primitives;
    # these methods let SDK consumers reach the primitives directly
    # without going through the model loop.  See
    # ``project_backlog_sdk_feature_parity.md``.
    # =========================================================================

    #: Wire-protocol minor from which the daemon answers an inject carrying a
    #: ``request_id`` with an :class:`InjectPromptResultEvent`.  Below this,
    #: nothing answers, so waiting would hang every call.
    MIN_INJECT_RESULT_PROTOCOL = "1.3"

    async def inject_prompt(
        self,
        text: str,
        source_type: str = "user",
        source_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Optional[str]:
        """Inject a prompt into the session's message queue.

        Single verb covering both "steer" (USER priority — interrupts
        the model at the next safe point) and "follow-up" (CHILD
        priority — queued behind in-flight work) patterns via the
        ``source_type`` dimension.

        Returns the delivery status, so a caller can tell whether the
        target will ACT on the message rather than only that the call did
        not raise.  Before SDK 0.14 this returned ``None`` unconditionally:
        the runner's receipt was discarded by the daemon and this method
        returned nothing, so a driver got identical silence whether its
        target was busy, idle, stranded, or dead.  A cascade driver read
        that silence as "sent" and stalled with no way to attribute it.

        Args:
            text: Prompt text to inject.
            source_type: Queue priority — ``"user"`` (steer),
                ``"child"`` (follow-up), or ``"system"`` / ``"event"``
                / ``"parent"`` for reactor / hook callers.
            source_id: Caller identifier for telemetry / logs.
            timeout: Seconds to wait for the daemon's result event.

        Returns:
            One of ``"accepted"`` (the target was idle, so a turn was
            STARTED), ``"queued"`` (the target is mid-turn and its running
            turn will drain the message), ``"terminated"`` (loaded but dead),
            ``"no_session"`` (not loaded), ``"unreachable"`` (live, but
            nothing was sent -- re-sending is SAFE), or ``"not_confirmed"``
            (an offer was made and its answer was lost -- re-sending MAY
            DUPLICATE).

            Only ``"accepted"`` and ``"queued"`` mean the message will be
            acted on.  **Do not treat the rest as success** — a caller that
            assumes delivery and is wrong gets a silent stall.

            The two transport failures are separate words because they call
            for OPPOSITE responses on retry.  Branch on membership of the
            delivered set for correctness; branch on these two only when
            deciding whether to re-send.

            ``None`` means the status is UNKNOWN, not that delivery failed:
            either the daemon predates protocol 1.3 and cannot answer, or
            the wait timed out.  It is returned rather than a placeholder
            string so that "I was not told" stays checkable instead of
            being mistaken for a real state.
        """
        if not _protocol_compatible(
                self.server_protocol_version,
                self.MIN_INJECT_RESULT_PROTOCOL):
            # Older daemon: it still routes the prompt, it just cannot report.
            # Say so once rather than hanging until the timeout on every call.
            if not getattr(self, "_warned_no_inject_result", False):
                self._warned_no_inject_result = True
                logger.warning(
                    "daemon protocol %s predates inject_prompt delivery "
                    "reporting (needs >= %s); inject_prompt will return None "
                    "and the delivery status is unavailable",
                    self.server_protocol_version,
                    self.MIN_INJECT_RESULT_PROTOCOL,
                )
            await self._send_event(InjectPromptRequest(
                text=text,
                source_type=source_type,
                source_id=source_id,
            ))
            return None

        req_id = f"req_{uuid.uuid4().hex[:16]}"
        await self._send_event(InjectPromptRequest(
            text=text,
            source_type=source_type,
            source_id=source_id,
            request_id=req_id,
        ))
        try:
            return await asyncio.wait_for(
                self._await_inject_result(req_id), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                "inject_prompt: timed out after %ss waiting for "
                "InjectPromptResultEvent", timeout,
            )
            return None

    async def _await_inject_result(self, request_id: str) -> Optional[str]:
        """Subscribe to the drain loop and wait for this inject's result.

        Mirrors :meth:`_await_session_info`: filters the subscriber queue for
        the :class:`InjectPromptResultEvent` (or a correlated
        :class:`ErrorEvent`, which the daemon emits for a rejected
        ``source_type``) carrying ``request_id``, and re-buffers incidental
        events when this is the only active subscriber so a later ``events()``
        call can still replay them.

        Returns:
            The status string, or ``None`` on error / disconnect.
        """
        q = self._subscribe_events()
        incidental: list[Event] = []
        try:
            while True:
                event = await q.get()
                if event is None:
                    if len(self._event_subscribers) == 1 and incidental:
                        self._buffered_events.extend(incidental)
                    return None

                solo = len(self._event_subscribers) == 1

                if isinstance(event, InjectPromptResultEvent):
                    if event.request_id != request_id:
                        if solo:
                            incidental.append(event)
                        continue
                    if solo:
                        incidental.append(event)
                        self._buffered_events.extend(incidental)
                    return event.status

                if isinstance(event, ErrorEvent):
                    if getattr(event, "request_id", None) != request_id:
                        if solo:
                            incidental.append(event)
                        continue
                    logger.warning(
                        "inject_prompt: daemon reported error "
                        "(error_type=%s): %s",
                        event.error_type, event.error,
                    )
                    if solo:
                        incidental.append(event)
                        self._buffered_events.extend(incidental)
                    return None

                if solo:
                    incidental.append(event)
        finally:
            self._unsubscribe_events(q)

    async def replay_messages(
        self,
        request_id: str,
        messages: Optional[list] = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Re-run the model loop against an explicit message list.

        When ``messages`` is omitted, replays the session's current
        ``get_history()`` — semantically equivalent to "continue from
        the current state with no new user input".

        The response arrives as a ``ReplayMessagesResultEvent`` via
        the event stream, correlated by ``request_id``.

        Args:
            request_id: Caller-chosen ID to correlate the result event.
            messages: Optional explicit message list (serialised
                ``List[Message]``).  ``None`` uses session history.
            timeout_seconds: Provider-exclusion lock acquisition
                timeout.
        """
        await self._send_event(ReplayMessagesRequest(
            request_id=request_id,
            messages=messages,
            timeout_seconds=timeout_seconds,
        ))

    async def resolve_fork_point(
        self,
        request_id: str,
        after_message: Optional[int] = None,
        after_tool_call: Optional[str] = None,
        after_timestamp: Optional[str] = None,
    ) -> None:
        """Resolve a fork point in the session's history to a message index.

        Exactly one of ``after_message`` / ``after_tool_call`` /
        ``after_timestamp`` should be supplied; if none are given,
        the server returns the last message index (full-history
        fork).  The session's current ``get_history()`` is used as
        the search space — clients don't pass history over the wire.

        The response arrives as a ``ResolveForkPointResultEvent`` via
        the event stream, correlated by ``request_id``.

        Args:
            request_id: Caller-chosen ID to correlate the result event.
            after_message: Direct message index specifier.
            after_tool_call: Tool call ID specifier.
            after_timestamp: HH:MM:SS or ISO timestamp specifier.
        """
        await self._send_event(ResolveForkPointRequest(
            request_id=request_id,
            after_message=after_message,
            after_tool_call=after_tool_call,
            after_timestamp=after_timestamp,
        ))

    # =========================================================================
    # SDK feature parity — permission policy verbs
    #
    # Typed methods replacing stringly-typed
    # ``execute_command("permissions", [...])`` for SDK consumers.
    # The CLI command path stays for actual users typing.
    # =========================================================================

    async def add_whitelist_tools(
        self,
        tools: Optional[list] = None,
        patterns: Optional[list] = None,
    ) -> None:
        """Add tools / patterns to the session's permission whitelist.

        Args:
            tools: Tool names to whitelist (exact match, auto-approved).
            patterns: Glob patterns to add to the session whitelist.
        """
        await self._send_event(PermissionAddWhitelistRequest(
            tools=tools or [],
            patterns=patterns or [],
        ))

    async def add_blacklist_tools(
        self,
        tools: Optional[list] = None,
        patterns: Optional[list] = None,
    ) -> None:
        """Add tools / patterns to the session's permission blacklist.

        Args:
            tools: Tool names to blacklist (always denied).
            patterns: Glob patterns to add to the session blacklist.
        """
        await self._send_event(PermissionAddBlacklistRequest(
            tools=tools or [],
            patterns=patterns or [],
        ))

    async def remove_permission_rules(
        self,
        target: str,
        tools: Optional[list] = None,
        patterns: Optional[list] = None,
    ) -> None:
        """Remove tools / patterns from a permission list.

        Args:
            target: ``"whitelist"`` or ``"blacklist"``.
            tools: Tool names to remove.
            patterns: Patterns to remove.
        """
        await self._send_event(PermissionRemoveRequest(
            target=target,
            tools=tools or [],
            patterns=patterns or [],
        ))

    async def clear_permission_rules(self, target: str = "all") -> None:
        """Clear the session-level permission lists.

        Does NOT affect the base policy declared in
        ``permissions.json``; only the session-level overrides.

        Args:
            target: ``"whitelist"``, ``"blacklist"``, or ``"all"``
                (clears both lists and the session default).
        """
        await self._send_event(PermissionClearRequest(target=target))

    async def set_default_policy(self, policy: str) -> None:
        """Set the session-level default permission policy.

        Overrides the base default for this session only.

        Args:
            policy: ``"allow"``, ``"deny"``, or ``"ask"``.
        """
        await self._send_event(PermissionSetDefaultRequest(policy=policy))

    async def request_policy_snapshot(self, request_id: str = "") -> None:
        """Request a structured snapshot of the current permission policy.

        The response arrives as a ``PermissionPolicySnapshotEvent``
        via the event stream, correlated by ``request_id``.

        Args:
            request_id: Caller-chosen ID to correlate the snapshot.
        """
        await self._send_event(PermissionPolicySnapshotRequest(
            request_id=request_id,
        ))

    # =========================================================================
    # Event Stream
    # =========================================================================

    def open_event_stream(self) -> "_SyncSubscribedStream":
        """Subscribe SYNCHRONOUSLY (at call time) and return an event iterator.

        Unlike :meth:`events` — an async generator that subscribes lazily on its
        first ``__anext__`` — this registers the subscriber queue NOW, before it
        returns.  Use it when the subscription must be established before an
        action that triggers server-side output, e.g.::

            stream = client.open_event_stream()   # queue registered now
            await client.attach(session_id)        # driven output can't be missed
            async for ev in stream:
                ...

        This removes any need to reach into ``_subscribe_events`` /
        ``_event_subscribers`` to force + prove registration.  Same fan-out and
        ``None``-sentinel disconnect semantics as :meth:`events` (buffered events
        are replayed into the queue first).  Lifetime is caller-managed: the
        stream unsubscribes on disconnect, on ``aclose()``, or on ``async with``
        exit — a long-lived consumer should ``aclose()`` at teardown.
        """
        from ._event_stream import _SyncSubscribedStream
        return _SyncSubscribedStream(self, self._subscribe_events())

    async def events(self) -> AsyncIterator[Event]:
        """Async iterator for receiving events.

        SDK 0.13.0+: subscribes a queue to the connection's drain task
        and yields events as they arrive.  Buffered events (those that
        flowed before any consumer subscribed — typically during
        ``create_session``) are replayed first.

        Multiple iterators can be active concurrently: each gets its
        own subscriber queue and the drain task fans events out to all
        of them.  Re-entrant ``create_session`` / ``events()``
        sequences no longer race on a shared ``_events_active`` flag —
        the drain task is the single reader regardless of how many
        consumers are listening.

        When the connection is lost (drain task ends), every
        subscriber queue receives a ``None`` sentinel and this
        iterator exits cleanly without raising.

        Yields:
            Events from the server.
        """
        logger.debug("events(): subscribing")
        q = self._subscribe_events()
        try:
            while True:
                event = await q.get()
                if event is None:
                    # drain loop signalled disconnection
                    logger.debug("events(): drain loop ended; iterator exiting")
                    break
                yield event
        finally:
            self._unsubscribe_events(q)
            logger.debug("events(): unsubscribed")

    async def cascade_events(
        self,
        cascade_driver_id: str,
        event_types: Optional[List[str]] = None,
        role: str = "observer",
    ) -> AsyncIterator[Event]:
        """Phase 2 cascade-as-client (server 0.6.156+, SDK 0.13.2+):
        async iterator over events from any session stamped with
        ``cascade_driver_id``.

        Sends ``cascade.register`` to the daemon at iterator start;
        yields matching events received via the existing event
        channel; sends ``cascade.unregister`` on iterator close
        (``break``, exception, or natural completion).

        Server-side filtering: only events from sessions whose
        ``cascade_driver_id`` matches AND whose type-name is in
        ``event_types`` reach this iterator.  Other events on the
        connection (e.g., from sessions the same client also
        created interactively) still flow to ``events()`` but are
        NOT yielded by this iterator — caveat: if the same client
        subscribes to both ``events()`` AND ``cascade_events()``,
        cascade events arrive on BOTH channels (both subscribers
        see them).

        Args:
            cascade_driver_id: The cid this iterator observes.
                Sessions stamped with this cid will route their
                events to the iterator.
            event_types: Optional list of event type-names to
                filter for (e.g., ``["SessionTerminatedEvent",
                "AgentCompletedEvent"]``).  ``None`` (default)
                subscribes to all event types.  Empty list also
                subscribes to all (no filter).
            role: ``"owner"`` (lifecycle authority; single per cid)
                or ``"observer"`` (read-only; multiple allowed).
                Default ``"observer"`` for the common observe-only
                case.  See ``docs/design/cascade-as-client.md``
                Decision 5.

        Yields:
            Events received by this connection that originated from
            a session stamped with ``cascade_driver_id`` and whose
            type matches the ``event_types`` filter.

        Raises:
            ValueError: when the server rejects the registration
                (duplicate owner, invalid role, etc.).  Surfaced
                via an ``ErrorEvent`` from the daemon; this
                iterator translates it into ValueError + exits.

        Example::

            import uuid
            cid = uuid.uuid4().hex
            async for event in client.cascade_events(
                cid,
                event_types=["SessionTerminatedEvent"],
            ):
                if event.reason == "error":
                    logger.error(f"Cascade failed at session {event.session_id}")
                    break  # see "Cleanup contract" below

        **Cleanup contract**: the iterator sends ``cascade.unregister``
        in its ``finally`` block, but Python's async-generator
        semantics mean ``finally`` only runs when the generator is
        explicitly closed (``await gen.aclose()``) OR garbage-
        collected.  ``async for ... break`` does NOT trigger
        ``aclose()`` synchronously.  For deterministic cleanup the
        user can:

        - Use ``async with contextlib.aclosing(client.cascade_events(...))``
        - Or rely on the server-side disconnect-cleanup backstop
          (Phase 2.2): when the IPC connection drops, the daemon
          removes ALL cascade-client registrations for that
          connection within 50ms.

        For typical kb-cascade smoke-driver use, the server-side
        backstop is sufficient — the driver disconnects when the
        cascade ends + cleanup fires automatically.
        """
        logger.debug(
            "cascade_events(): subscribing cid=%s role=%s event_types=%s",
            cascade_driver_id, role, event_types,
        )
        # Build CommandRequest args: [cid, role, *event_types].
        # Server-side _handle_cascade_register parses this shape.
        args = [cascade_driver_id, role]
        if event_types:
            args.extend(event_types)
        # Subscribe to the event queue BEFORE sending the register
        # command so we don't miss the confirmation SystemMessageEvent
        # or any early events for fast-arriving sessions.
        q = self._subscribe_events()
        try:
            await self._send_event(CommandRequest(
                command="cascade.register",
                args=args,
            ))
            # SDK 0.14.4+: client-side type-name filter honors the
            # docstring contract.  Server-side dispatch via the
            # cascade-client callback path already filters on
            # ``event_types`` (session_manager.py:217-222
            # ``event_type_match``), but the SDK's
            # ``_subscribe_events()`` queue receives EVERY event on
            # this IPC connection — including events arriving via
            # other paths (e.g. normal session events the client also
            # observes).  Without this filter, multi-event-subscription
            # callers saw events of types they didn't subscribe to
            # leak through (peer 7:1 empirical: 42 AGENT_CREATED
            # arrived despite registering only SessionTerminatedEvent).
            filter_set = set(event_types) if event_types else None
            while True:
                event = await q.get()
                if event is None:
                    # Connection drain loop ended.
                    logger.debug(
                        "cascade_events(): drain loop ended; "
                        "iterator exiting cid=%s", cascade_driver_id,
                    )
                    break
                if filter_set is not None and (
                    type(event).__name__ not in filter_set
                ):
                    continue
                yield event
        finally:
            self._unsubscribe_events(q)
            # Best-effort unregister.  If the connection is already
            # gone (drain loop ended), the send is a no-op; the
            # server-side disconnect handler already cleaned up the
            # registration via
            # ``unregister_all_cascade_clients_for_connection``.
            try:
                await self._send_event(CommandRequest(
                    command="cascade.unregister",
                    args=[cascade_driver_id],
                ))
            except Exception as exc:  # noqa: BLE001 — cleanup boundary
                logger.debug(
                    "cascade_events(): unregister send failed "
                    "(connection likely closed): %s", exc,
                )
            logger.debug(
                "cascade_events(): unsubscribed cid=%s",
                cascade_driver_id,
            )

    async def drain_events(self) -> None:
        """Drive the event loop, dispatching to subscribed handlers.

        Runs until disconnected. Convenience wrapper for callers that
        only want handler-based delivery and don't need to iterate
        ``events()`` manually. Equivalent to::

            async for _ in client.events():
                pass
        """
        async for _ in self.events():
            pass
