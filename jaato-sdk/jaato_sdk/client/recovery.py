"""IPC Client with automatic connection recovery.

Provides a wrapper around IPCClient that handles automatic reconnection
when the server becomes unavailable (e.g., during restarts).

Features:
- Automatic detection of connection loss
- Exponential backoff with jitter for reconnection attempts
- Session reattachment after successful reconnection
- Status callbacks for UI updates
- Configurable retry behavior via RecoveryConfig

Usage:
    from jaato_sdk.client import IPCRecoveryClient, ConnectionState
    from jaato_sdk.client.config import get_recovery_config

    config = get_recovery_config()

    def on_status(status):
        if status.state == ConnectionState.RECONNECTING:
            print(f"Reconnecting... attempt {status.attempt}/{status.max_attempts}")

    client = IPCRecoveryClient(
        socket_path="/tmp/jaato.sock",
        config=config,
        on_status_change=on_status,
    )

    await client.connect()

    # Use normally - reconnection is automatic
    async for event in client.events():
        handle_event(event)
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from jaato_sdk.client._handler_registry import (
    EventHandler,
    Unsubscribe,
    _HandlerRegistry,
)
from jaato_sdk.client.config import RecoveryConfig
from jaato_sdk.client.errors import (
    SessionCreateFailed,
    SessionNotConfirmed,
    SessionNotSent,
    SessionRefused,
)
from jaato_sdk.client.ipc import DEFAULT_SOCKET_PATH, IPCClient, IncompatibleServerError
from jaato_sdk.events import ClientType
from jaato_sdk.events import (
    ConnectedEvent,
    ErrorEvent,
    Event,
    EventType,
    SessionInfoEvent,
    SystemMessageEvent,
)

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state for the recovery client.

    States:
        DISCONNECTED: Not connected, no reconnection in progress.
        CONNECTING: Initial connection attempt in progress.
        CONNECTED: Successfully connected to server.
        RECONNECTING: Connection lost, attempting to reconnect.
        DISCONNECTING: Graceful disconnect initiated.
        CLOSED: Terminal state, no more reconnection attempts.
    """
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTING = "disconnecting"
    CLOSED = "closed"


class ConnectionError(Exception):  # noqa: A001 — shadows the builtin, see below
    """Error related to IPC connection.

    .. warning::

       This SHADOWS the builtin ``ConnectionError`` and does **not** subclass
       it.  A consumer writing the obvious ``except ConnectionError`` catches
       the *builtin* and silently fails to catch this one, unless it imported
       this name explicitly.  Two out-of-tree consumers were found doing
       exactly that.

       Left as-is here because renaming is a breaking change with its own
       blast radius, and it is not this change's subject.  Recorded so the
       next reader does not have to rediscover it.
    """
    pass


class ReconnectingError(Exception):
    """Raised when an operation is attempted during reconnection."""

    def __init__(self, message: str = "Client is reconnecting"):
        super().__init__(message)


class ConnectionClosedError(Exception):
    """Raised when connection is permanently closed."""

    def __init__(self, message: str = "Connection is closed"):
        super().__init__(message)


@dataclass
class ConnectionStatus:
    """Current connection status for UI display.

    Provides all information needed to display reconnection status
    to the user.

    Attributes:
        state: Current connection state.
        attempt: Current reconnection attempt number (0 if not reconnecting).
        max_attempts: Maximum reconnection attempts configured.
        next_retry_in: Seconds until next retry attempt (None if not waiting).
        last_error: Description of the last error encountered.
        session_id: ID of the attached session (None if not attached).
        client_id: ID assigned by server (None if not connected).
    """
    state: ConnectionState
    attempt: int = 0
    max_attempts: int = 0
    next_retry_in: Optional[float] = None
    last_error: Optional[str] = None
    session_id: Optional[str] = None
    client_id: Optional[str] = None


# Type alias for status change callback
StatusCallback = Callable[[ConnectionStatus], None]


class IPCRecoveryClient:
    """IPC client with automatic connection recovery.

    Wraps IPCClient to provide automatic reconnection when the server
    becomes unavailable. Maintains session state for reattachment after
    reconnection.

    The client uses a state machine to track connection status:
    - DISCONNECTED -> CONNECTING (on connect())
    - CONNECTING -> CONNECTED (on successful handshake)
    - CONNECTED -> RECONNECTING (on connection loss)
    - RECONNECTING -> CONNECTING (on retry attempt)
    - RECONNECTING -> CLOSED (on max retries exceeded)
    - * -> CLOSED (on close())

    Attributes:
        socket_path: Path to the IPC socket.
        config: Recovery configuration.
        state: Current connection state.
        session_id: ID of the attached session.
        client_id: ID assigned by server.
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        *,
        client_type: ClientType,
        config: Optional[RecoveryConfig] = None,
        auto_start: bool = True,
        env_file: str = ".env",
        workspace_path: Optional[Path] = None,
        on_status_change: Optional[StatusCallback] = None,
        min_protocol_version: Optional[str] = None,
        presentation: Optional[Any] = None,
        config_root: Optional[str] = None,
        apparmor: bool = False,
        autostart_timeout: float = 120.0,
    ):
        """Initialize the recovery client.

        Args:
            socket_path: Path to Unix domain socket or Windows pipe name.
            config: Recovery configuration. If None, loads from config files
                and environment variables.
            auto_start: Whether to auto-start server if not running.
            env_file: Path to .env file for auto-started server.
            workspace_path: Workspace path for loading project-level config.
            on_status_change: Callback invoked on connection status changes.
                Receives a ConnectionStatus object.
            client_type: **Required.** Forwarded to the inner IPCClient —
                see ``IPCClient.__init__`` for semantics.  Pass
                ``ClientType.TERMINAL`` for the TUI, ``API`` for
                headless / batch / cascade harnesses.
            config_root: Forwarded to the inner IPCClient — where the
                daemon reads framework config for sessions on this
                connection.  Was IPCClient-ONLY, which left a
                recovery-driven session at ``config_root=None`` with no
                route to set it: AppArmor composition then silently
                dropped every plugin rule gated on config_root and
                file_edit lost its backup subtree, while the profile
                still loaded and still logged ``runner confined
                (enforce)``.  An incomplete confinement profile that
                reports success is worse than a missing kwarg.
            apparmor: Forwarded to the inner IPCClient — opt-in
                confinement for sessions on this connection.  A profile
                can also request it (``apparmor: true``), so this was
                the one half of the gap with an escape hatch;
                ``config_root`` had none.
            autostart_timeout: Forwarded to the inner IPCClient — how
                long to wait for the daemon socket when auto-starting.
                Relevant here precisely because recovery forwards
                ``auto_start``.
            min_protocol_version: Override the inner IPCClient's
                ``MIN_PROTOCOL_VERSION``. Forwarded verbatim — see
                ``IPCClient.__init__`` for semantics.
            presentation: Display-capability override forwarded verbatim to
                each inner IPCClient the recovery client constructs (initial
                connect + every reconnect), so a non-terminal (chat/web)
                presentation survives reconnection.  See
                ``IPCClient.__init__`` for semantics.
        """
        self._socket_path = socket_path
        self._client_type = client_type
        self._auto_start = auto_start
        self._env_file = env_file
        self._workspace_path = workspace_path
        self._on_status_change = on_status_change
        self._min_protocol_version = min_protocol_version
        self._presentation = presentation
        self._config_root = config_root
        self._apparmor = apparmor
        self._autostart_timeout = autostart_timeout

        # Load config if not provided
        if config is None:
            from jaato_sdk.client.config import get_recovery_config
            config = get_recovery_config(workspace_path)
        self._config = config

        # Underlying client
        self._client: Optional[IPCClient] = None

        # State management
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = asyncio.Lock()

        # Session tracking (for reattachment)
        self._session_id: Optional[str] = None
        self._client_id: Optional[str] = None
        # Host ("client") tools remembered so they're re-registered on the
        # fresh inner client after a reconnect (see register_client_tools).
        self._registered_client_tools: List[Dict[str, Any]] = []

        # Reconnection state
        self._reconnect_attempt = 0
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_cancelled = False

        # Event forwarding
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._event_task: Optional[asyncio.Task] = None
        self._events_running = False

        # Subscribers live at this layer so they survive reconnections
        # (the inner IPCClient is recreated on each reconnect, but the
        # recovery wrapper keeps the same registry).
        self._registry = _HandlerRegistry()

        # Fan-out plumbing (mirror of IPCClient's drain loop).  The background
        # event PUMP — started in ``connect`` — is the SINGLE reader of the
        # inner client's events: it dispatches every event to ``_registry``
        # (so ``subscribe()`` handlers fire WITHOUT anyone iterating
        # ``events()`` — the gap that hung the convenience facade over a
        # recovery client) AND fans out to ``events()`` consumer queues.
        self._event_subscribers: "list[asyncio.Queue]" = []
        self._event_pump_task: Optional[asyncio.Task] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def socket_path(self) -> str:
        """Get the socket path."""
        return self._socket_path

    @property
    def config(self) -> RecoveryConfig:
        """Get the recovery configuration."""
        return self._config

    @property
    def state(self) -> ConnectionState:
        """Get the current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._state == ConnectionState.CONNECTED

    @property
    def is_reconnecting(self) -> bool:
        """Check if reconnection is in progress."""
        return self._state == ConnectionState.RECONNECTING

    @property
    def is_closed(self) -> bool:
        """Check if connection is permanently closed."""
        return self._state == ConnectionState.CLOSED

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @property
    def client_id(self) -> Optional[str]:
        """Get the client ID assigned by server."""
        return self._client_id

    @property
    def server_version(self) -> Optional[str]:
        """Get the server's package version, available after connect().

        Delegates to the underlying ``IPCClient.server_version``.
        Returns ``None`` if not connected or the server did not report a version.
        """
        if self._client:
            return self._client.server_version
        return None

    # =========================================================================
    # High-level convenience facade
    # =========================================================================

    @classmethod
    def session(cls, **kwargs):
        """Open a session with the high-level facade (auto-reconnect variant).

        Same surface and semantics as :meth:`IPCClient.session` — returns an
        async context manager yielding a
        :class:`~jaato_sdk.client.convenience.Session` — but backed by an
        ``IPCRecoveryClient`` so the session survives daemon restarts.  Adds an
        ``on_status_change=`` kwarg (the reconnection-status callback) on top of
        the shared knobs::

            async with IPCRecoveryClient.session(profile="researcher",
                                                 on_status_change=print) as s:
                print(await s.ask("Long task…"))

        See ``docs/design/sdk-convenience-layer.md``.
        """
        from .convenience import open_session
        return open_session(cls, **kwargs)

    # =========================================================================
    # Connection Management
    # =========================================================================

    def _make_client(self, *, auto_start: bool) -> IPCClient:
        """Construct the inner transport client.

        The recovery state machine, reconnect loop, session reattachment, and
        event pump are all transport-agnostic — they only ``connect`` /
        ``disconnect`` / read events off whatever this returns.  Override in a
        transport subclass (e.g. ``WSRecoveryClient``) to bind a different
        transport while reusing all of that.  ``auto_start`` is
        ``self._auto_start`` on the initial connect and ``False`` on reconnect
        (a restart is never auto-started).
        """
        return IPCClient(
            socket_path=self._socket_path,
            client_type=self._client_type,
            auto_start=auto_start,
            env_file=self._env_file,
            workspace_path=str(self._workspace_path) if self._workspace_path else None,
            min_protocol_version=self._min_protocol_version,
            presentation=self._presentation,
            config_root=self._config_root,
            apparmor=self._apparmor,
            autostart_timeout=self._autostart_timeout,
        )

    async def connect(self, timeout: float = 5.0) -> bool:
        """Connect to the server.

        Args:
            timeout: Connection timeout in seconds.

        Returns:
            True if connected successfully.

        Raises:
            ConnectionError: If connection fails and recovery is disabled.
        """
        async with self._state_lock:
            if self._state == ConnectionState.CLOSED:
                raise ConnectionClosedError()

            self._transition_to(ConnectionState.CONNECTING)

        try:
            self._client = self._make_client(auto_start=self._auto_start)

            # When auto-start is enabled, the inner connect() may need to:
            # 1. Try initial connection (timeout seconds)
            # 2. Start server daemon subprocess
            # 3. Wait for socket/pipe to appear (up to 10s)
            # 4. Retry connection (timeout seconds)
            # This can take 2*timeout + 10s+, so we need a generous outer
            # timeout to avoid cancelling the inner operation prematurely.
            # On Windows named pipes this is especially important because
            # pipe creation and server initialization take longer.
            if self._auto_start:
                outer_timeout = timeout * 2 + 20.0
            else:
                outer_timeout = timeout + 1.0

            connected = await asyncio.wait_for(
                self._client.connect(timeout=timeout),
                timeout=outer_timeout,
            )

            if connected:
                self._client_id = self._client.client_id
                async with self._state_lock:
                    self._transition_to(ConnectionState.CONNECTED)
                # Start the background event pump (single reader → registry
                # dispatch + events() fan-out).  It owns ``_events_running``
                # and survives reconnections (the loop re-reads the recreated
                # inner client), so start it once.
                if self._event_pump_task is None or self._event_pump_task.done():
                    self._events_running = True
                    self._event_pump_task = asyncio.create_task(self._event_pump())
                return True

            raise ConnectionError("Connection failed")

        except IncompatibleServerError:
            # Server too old — propagate directly, retrying won't help
            async with self._state_lock:
                self._transition_to(ConnectionState.CLOSED)
            raise

        except asyncio.TimeoutError:
            logger.warning(f"Connection timeout to {self._socket_path}")
            async with self._state_lock:
                self._transition_to(ConnectionState.DISCONNECTED)
            raise ConnectionError(f"Connection timeout: {self._socket_path}")

        except Exception as e:
            logger.warning(f"Connection failed: {e}")
            async with self._state_lock:
                self._transition_to(ConnectionState.DISCONNECTED)
            raise ConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the server gracefully."""
        async with self._state_lock:
            if self._state in (ConnectionState.CLOSED, ConnectionState.DISCONNECTED):
                return

            self._transition_to(ConnectionState.DISCONNECTING)

        # Cancel any reconnection in progress
        await self._cancel_reconnection()

        # Stop event loop + the background event pump
        self._events_running = False
        await self._stop_event_pump()

        # Disconnect underlying client
        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.debug(f"Error during disconnect: {e}")

        self._client = None

        async with self._state_lock:
            self._transition_to(ConnectionState.DISCONNECTED)

    async def close(self) -> None:
        """Permanently close the connection.

        After calling close(), the client cannot be reconnected.
        """
        async with self._state_lock:
            if self._state == ConnectionState.CLOSED:
                return

            self._transition_to(ConnectionState.CLOSED)

        # Cancel any reconnection in progress
        await self._cancel_reconnection()

        # Stop event loop + the background event pump
        self._events_running = False
        await self._stop_event_pump()

        # Disconnect underlying client
        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.debug(f"Error during close: {e}")

        self._client = None

    # =========================================================================
    # Session Management
    # =========================================================================

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for reattachment after reconnection.

        This should be called when the session is first attached,
        so that the client knows which session to reattach to
        after a reconnection.

        Args:
            session_id: The session ID to track.
        """
        self._session_id = session_id
        logger.debug(f"Session ID set for recovery: {session_id}")

    async def attach_session(self, session_id: str) -> bool:
        """Attach to a session.

        Args:
            session_id: The session to attach to.

        Returns:
            True if attach command was sent.

        Raises:
            ReconnectingError: If currently reconnecting.
            ConnectionClosedError: If connection is closed.
        """
        self._check_can_send()

        if self._client:
            await self._client.attach_session(session_id)
            self._session_id = session_id
            return True
        return False

    async def create_session(
        self,
        name: Optional[str] = None,
        profile: Optional[Union[str, Dict[str, Any]]] = None,
        agent: Optional[str] = None,
        agent_params: Optional[Dict[str, str]] = None,
        cascade_driver_id: Optional[str] = None,
        sibling_name: Optional[str] = None,
        timeout: float = 60.0,
    ) -> Optional[str]:
        """Create a new session.

        Args:
            name: Optional name for the session.
            profile: Either a profile **name** (str) referencing a file
                under ``.jaato/profiles/``, **or** an inline **spec dict**
                (``model``, ``plugins``, ``system_instructions``, ...).
                See ``IPCClient.create_session`` for the full list.
                Mutually exclusive forms — pass one or the other.
            agent: Optional agent name — WHO the session is, as opposed
                to ``profile``, which is what it can do.  Its rendered
                markdown is one LAYER of the assembled system instructions,
                not the whole of them; see
                :meth:`IPCClient.create_session` for the full contract.
            agent_params: Parameter values for the agent's ``{{param}}``
                placeholders.
            sibling_name: Cascade-scoped address; see
                ``IPCClient.create_session`` for the contract.
            cascade_driver_id: Phase 2 cascade-sharing tenant ID; see
                ``IPCClient.create_session`` for the contract.  Pass
                the same opaque ID across every session of one cascade.
            timeout: Seconds to wait for the ``SessionInfoEvent`` — mirrors
                ``IPCClient.create_session`` so a plain→recovery swap is
                drop-in (forwarded to the underlying client).  The recovery
                client's own reconnect timing is governed separately (recovery
                config + ``connect(timeout=)``).

        Returns:
            The new session ID.  Never ``None`` — a failure raises.

        Raises:
            ReconnectingError: If currently reconnecting.
            ConnectionClosedError: If connection is closed.
            SessionNotSent: the command never left this process — including
                the case where this recovery client has no inner client at
                all, which used to return a bare ``None`` indistinguishable
                from a daemon refusal.
            SessionRefused: the daemon answered and refused.
            SessionNotConfirmed: sent, unanswered — a session MAY exist.
            TypeError: If ``profile`` is not None, str, or dict.

        See :meth:`IPCClient.create_session` for the full contract; this
        wrapper adds only the recovery-state gate in front of it.
        """
        self._check_can_send()

        if self._client is None:
            # Reachable when the recovery client is constructed but has not
            # built an inner client yet.  It returned ``None`` here — the same
            # answer as a daemon refusal, from a state where nothing was even
            # attempted.
            raise SessionNotSent(
                "no inner client: this recovery client has not connected, so "
                "session.new was never sent and no session was created."
            )

        session_id = await self._client.create_session(
            name, profile=profile, agent=agent,
            agent_params=agent_params,
            cascade_driver_id=cascade_driver_id,
            sibling_name=sibling_name,
            timeout=timeout,
        )
        self._session_id = session_id
        return session_id

    async def list_profiles(self) -> None:
        """Request list of available agent profiles.

        The server responds with a ``SessionProfilesEvent`` containing
        profile summaries discovered from ``.jaato/profiles/``.

        Raises:
            ReconnectingError: If currently reconnecting.
            ConnectionClosedError: If connection is closed.
        """
        self._check_can_send()

        if self._client:
            await self._client.list_profiles()

    async def get_default_session(self) -> None:
        """Get or create the default session."""
        self._check_can_send()

        if self._client:
            await self._client.get_default_session()

    # =========================================================================
    # Message Sending
    # =========================================================================

    async def send_message(
        self,
        text: str,
        attachments: Optional[list] = None,
        parallel_tools: Optional[bool] = None,
    ) -> None:
        """Send a message to the model.

        Args:
            text: The message text.
            attachments: Optional file attachments.
            parallel_tools: Per-call override for parallel tool execution.
                ``None`` (default) keeps the env-configured behaviour
                (``JAATO_PARALLEL_TOOLS``); ``True`` / ``False`` forces
                parallel / sequential tool execution for this turn only.

        Raises:
            ReconnectingError: If currently reconnecting.
            ConnectionClosedError: If connection is closed.
        """
        self._check_can_send()

        if self._client:
            await self._client.send_message(text, attachments, parallel_tools=parallel_tools)

    async def respond_to_permission(
        self,
        request_id: str,
        response: str,
        edited_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Respond to a permission request."""
        self._check_can_send()

        if self._client:
            await self._client.respond_to_permission(request_id, response,
                                                     edited_arguments=edited_arguments)

    async def respond_to_clarification(
        self,
        request_id: str,
        response: str,
    ) -> None:
        """Respond to a clarification question."""
        self._check_can_send()

        if self._client:
            await self._client.respond_to_clarification(request_id, response)

    async def respond_to_clarification_batch(
        self,
        request_id: str,
        answers: List[str],
    ) -> None:
        """Respond to a batched clarification (all answers at once) — proxied
        to the inner client (see ``IPCClient.respond_to_clarification_batch``)."""
        self._check_can_send()

        if self._client:
            await self._client.respond_to_clarification_batch(request_id, answers)

    async def register_client_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Register client-provided ("host") tools — proxied to the inner client.

        The tool set is REMEMBERED so it is re-registered automatically after a
        reconnect: a reconnect builds a fresh inner client (``_make_client``)
        that would otherwise lose the host-tool handlers, breaking a recoverable
        host-tool client on the first daemon restart.  See
        ``IPCClient.register_client_tools`` for the entry contract (register
        before ``create_session``).
        """
        self._check_can_send()
        self._registered_client_tools = list(tools)

        if self._client:
            await self._client.register_client_tools(tools)

    async def list_sessions(self) -> None:
        """Request the session list — proxied to the inner client."""
        self._check_can_send()

        if self._client:
            await self._client.list_sessions()

    async def respond_to_reference_selection(
        self,
        request_id: str,
        response: str,
    ) -> None:
        """Respond to a reference selection request."""
        self._check_can_send()

        if self._client:
            await self._client.respond_to_reference_selection(request_id, response)

    async def respond_to_tool_execution(
        self,
        call_id: str,
        result: str = "",
        error: str = "",
    ) -> None:
        """Return the result of a client-side tool execution.

        See :meth:`IPCClient.respond_to_tool_execution` for full docs.
        """
        self._check_can_send()
        if self._client:
            await self._client.respond_to_tool_execution(call_id, result, error)

    async def end_session(self) -> None:
        """Terminate the currently-attached session.

        See :meth:`IPCClient.end_session` for full docs.
        """
        self._check_can_send()
        if self._client:
            await self._client.end_session()

    async def delete_session(self, session_id: str) -> None:
        """Permanently delete a session by ID.

        See :meth:`IPCClient.delete_session` for full docs.
        """
        self._check_can_send()
        if self._client:
            await self._client.delete_session(session_id)

    async def respond_to_post_auth_setup(
        self,
        request_id: str,
        connect: bool = False,
        model_name: str = "",
        persist_env: bool = False,
    ) -> None:
        """Respond to a post-auth setup prompt."""
        self._check_can_send()

        if self._client:
            from jaato_sdk.events import PostAuthSetupResponse
            await self._client._send_event(PostAuthSetupResponse(
                request_id=request_id,
                connect=connect,
                model_name=model_name,
                persist_env=persist_env,
            ))

    async def stop(self) -> None:
        """Stop current operation."""
        # Allow stop even during reconnection
        if self._client and self._state == ConnectionState.CONNECTED:
            await self._client.stop()

    async def execute_command(
        self,
        command: str,
        args: Optional[list] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Execute a command.

        Args:
            command: Command verb (e.g. ``session.wake``).
            args: Positional arguments.
            payload: Structured payload.  Required by verbs that take one --
                ``cascade.budget.set`` errors without it, and ``session.wake``
                accepts either shape.  Omitting this parameter here (while
                :class:`IPCClient` had it) made every payload-form verb a
                TypeError on a recovery client.
        """
        self._check_can_send()

        if self._client:
            await self._client.execute_command(command, args, payload)

    # ---- typed wake-primitive methods (see _wake_client) ----
    async def bind_wake(self, wake_ref: str, trust_keys: list, *,
                        timeout: float = 30.0):
        """Declare a wake binding for this session; await the typed result.
        See :func:`jaato_sdk.client._wake_client.bind_wake`.  Subscribes on the
        recovery wrapper's registry (survives reconnects), so the result
        handler is armed before the command is sent."""
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

    @property
    def server_protocol_version(self) -> Optional[str]:
        """The server's WIRE-PROTOCOL version, once the handshake completes.

        Distinct from :attr:`server_version` (the daemon's package version).
        This is the value the compatibility check actually ran against.

        Exposed because the wrapper already ACCEPTS ``min_protocol_version``
        and forwards it to the inner client -- a caller could constrain the
        negotiation but not observe its outcome.  That matters most for the
        long-lived drivers this class exists for: reconnecting across a daemon
        restart can land on a DIFFERENT server build, and without this there
        is no way to notice.
        """
        return self._client.server_protocol_version if self._client else None

    async def cascade_budget_set(
        self,
        cascade_driver_id: str,
        limits: dict,
        degrade: Optional[list] = None,
    ) -> None:
        """Set a cascade's aggregate budget ceiling.

        See :meth:`jaato_sdk.client.ipc.IPCClient.cascade_budget_set`.  There
        is no args-only form of this verb -- the daemon requires
        ``args=[cascade_driver_id]`` AND a ``{limits, degrade}`` payload -- so
        before this existed a recovery client could not set a ceiling at all.
        """
        self._check_can_send()
        if self._client:
            await self._client.cascade_budget_set(
                cascade_driver_id, limits, degrade)

    async def cascade_budget_get(self, cascade_driver_id: str) -> None:
        """Read a cascade's budget state.
        See :meth:`jaato_sdk.client.ipc.IPCClient.cascade_budget_get`."""
        self._check_can_send()
        if self._client:
            await self._client.cascade_budget_get(cascade_driver_id)

    async def cascade_budget_clear(self, cascade_driver_id: str) -> None:
        """Drop a cascade's budget registration.
        See :meth:`jaato_sdk.client.ipc.IPCClient.cascade_budget_clear`."""
        self._check_can_send()
        if self._client:
            await self._client.cascade_budget_clear(cascade_driver_id)

    def cascade_events(
        self,
        cascade_driver_id: str,
        event_types: Optional[List[str]] = None,
        role: str = "observer",
    ):
        """Async-iterate events from sessions stamped with *cascade_driver_id*.

        See :meth:`jaato_sdk.client.ipc.IPCClient.cascade_events`.  The
        recovery wrapper already forwarded ``cascade_register`` but not this,
        so an observer could register and then have no way to read.

        NOT an ``async def``: the underlying method is an async GENERATOR, so
        returning it directly preserves ``async for`` at the call site.
        """
        self._check_can_send()
        if not self._client:
            raise RuntimeError("cascade_events: not connected")
        return self._client.cascade_events(
            cascade_driver_id, event_types=event_types, role=role)

    async def drain_events(self) -> None:
        """Drive the event loop, dispatching to subscribed handlers.
        See :meth:`jaato_sdk.client.ipc.IPCClient.drain_events`."""
        async for _ in self.events():
            pass

    async def disable_tool(self, tool_name: str) -> None:
        """Disable a tool directly via registry.

        This is a fire-and-forget request that doesn't generate response events.
        Used by headless mode to disable tools before starting event handling.
        """
        self._check_can_send()

        if self._client:
            await self._client.disable_tool(tool_name)

    async def request_command_list(self) -> None:
        """Request the list of available commands."""
        self._check_can_send()

        if self._client:
            await self._client.request_command_list()

    async def request_history(self, agent_id: str = "main") -> None:
        """Request conversation history."""
        self._check_can_send()

        if self._client:
            await self._client.request_history(agent_id)

    # =========================================================================
    # SDK feature parity — session-primitive verbs
    # =========================================================================

    async def inject_prompt(
        self,
        text: str,
        source_type: str = "user",
        source_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Optional[str]:
        """Inject a prompt into the session's message queue.

        See :meth:`IPCClient.inject_prompt` for full docs, including what
        each status means and why ``None`` is "not told" rather than
        "not delivered".

        Returns ``None`` when no underlying client is connected — the same
        unknown-status signal the delegate uses, since a recovery client
        between connections has not been told anything either.
        """
        self._check_can_send()
        if self._client:
            return await self._client.inject_prompt(
                text, source_type, source_id, timeout=timeout,
            )
        return None

    async def replay_messages(
        self,
        request_id: str,
        messages: Optional[list] = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Re-run the model loop against an explicit message list.

        See :meth:`IPCClient.replay_messages` for full docs.
        """
        self._check_can_send()
        if self._client:
            await self._client.replay_messages(request_id, messages, timeout_seconds)

    async def resolve_fork_point(
        self,
        request_id: str,
        after_message: Optional[int] = None,
        after_tool_call: Optional[str] = None,
        after_timestamp: Optional[str] = None,
    ) -> None:
        """Resolve a fork point in the session's history.

        See :meth:`IPCClient.resolve_fork_point` for full docs.
        """
        self._check_can_send()
        if self._client:
            await self._client.resolve_fork_point(
                request_id,
                after_message=after_message,
                after_tool_call=after_tool_call,
                after_timestamp=after_timestamp,
            )

    # =========================================================================
    # SDK feature parity — permission policy verbs
    # =========================================================================

    async def add_whitelist_tools(
        self,
        tools: Optional[list] = None,
        patterns: Optional[list] = None,
    ) -> None:
        """Add tools / patterns to the session's permission whitelist."""
        self._check_can_send()
        if self._client:
            await self._client.add_whitelist_tools(tools, patterns)

    async def add_blacklist_tools(
        self,
        tools: Optional[list] = None,
        patterns: Optional[list] = None,
    ) -> None:
        """Add tools / patterns to the session's permission blacklist."""
        self._check_can_send()
        if self._client:
            await self._client.add_blacklist_tools(tools, patterns)

    async def remove_permission_rules(
        self,
        target: str,
        tools: Optional[list] = None,
        patterns: Optional[list] = None,
    ) -> None:
        """Remove tools / patterns from a permission list."""
        self._check_can_send()
        if self._client:
            await self._client.remove_permission_rules(target, tools, patterns)

    async def clear_permission_rules(self, target: str = "all") -> None:
        """Clear the session-level permission lists."""
        self._check_can_send()
        if self._client:
            await self._client.clear_permission_rules(target)

    async def set_default_policy(self, policy: str) -> None:
        """Set the session-level default permission policy."""
        self._check_can_send()
        if self._client:
            await self._client.set_default_policy(policy)

    async def request_policy_snapshot(self, request_id: str = "") -> None:
        """Request a structured snapshot of the current permission policy."""
        self._check_can_send()
        if self._client:
            await self._client.request_policy_snapshot(request_id)

    # =========================================================================
    # Event Stream
    # =========================================================================

    # =========================================================================
    # Event Subscription API (mirrors IPCClient; survives reconnections)
    # =========================================================================

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Unsubscribe:
        """Subscribe to events of a specific type.

        Subscriptions live on the recovery wrapper, so they keep working
        across reconnections without the caller re-registering anything.
        """
        return self._registry.subscribe(event_type, handler)

    def subscribe_once(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> Unsubscribe:
        """Subscribe to one event of ``event_type`` then auto-unsubscribe."""
        return self._registry.subscribe_once(event_type, handler)

    def subscribe_all(self, handler: EventHandler) -> Unsubscribe:
        """Subscribe to every event regardless of type."""
        return self._registry.subscribe_all(handler)

    def subscribe_many(
        self,
        handlers: Dict[EventType, EventHandler],
    ) -> Unsubscribe:
        """Register multiple typed handlers; single unsub removes all."""
        return self._registry.subscribe_many(handlers)

    def open_event_stream(self) -> "_SyncSubscribedStream":
        """Subscribe SYNCHRONOUSLY (at call time) and return an event iterator.

        The recovery-client counterpart to :meth:`IPCClient.open_event_stream`.
        Unlike :meth:`events` (an async generator that subscribes lazily on its
        first ``__anext__``), this registers the subscriber queue NOW, before it
        returns — so a caller can guarantee the subscription is live BEFORE it
        triggers server-side output (e.g. ``attach`` a session the daemon will
        immediately drive a woken turn on).  Critical here because the recovery
        client has NO zero-subscriber replay buffer, so a not-yet-registered
        queue silently drops that output::

            stream = client.open_event_stream()   # queue registered now
            await client.attach(session_id)        # driven output can't be missed
            async for ev in stream:
                ...

        Removes any need to poll ``_event_subscribers`` to prove registration.
        Same fan-out + ``None``-sentinel semantics as :meth:`events`, and the
        queue survives transient reconnects (the pump keeps fanning out to it).
        Lifetime is caller-managed — ``aclose()`` at teardown.
        """
        from ._event_stream import _SyncSubscribedStream
        return _SyncSubscribedStream(self, self._subscribe_events())

    async def events(self) -> AsyncIterator[Event]:
        """Async iterator for receiving events.

        Fan-out model (mirror of ``IPCClient.events``): the background
        :meth:`_event_pump` is the SINGLE reader of the inner client's events;
        ``events()`` subscribes a queue to its fan-out and yields what arrives.
        Multiple iterators can run concurrently, and ``subscribe()`` handlers
        fire from the pump regardless of whether anyone iterates here — which
        is what lets the convenience facade (``subscribe`` + ``await done``)
        work over a recovery client instead of hanging.  On disconnect the
        pump pushes a ``None`` sentinel and this iterator exits cleanly.

        Example:
            async for event in client.events():
                if isinstance(event, AgentOutputEvent):
                    print(event.text)
        """
        q = self._subscribe_events()
        try:
            while True:
                event = await q.get()
                if event is None:  # pump signalled disconnect/close
                    break
                yield event
        finally:
            self._unsubscribe_events(q)

    async def _event_pump(self) -> None:
        """Background single reader: drain the inner client's events, dispatch
        to ``_registry`` (so ``subscribe()`` handlers fire without anyone
        iterating :meth:`events`), fan out to ``events()`` consumers, and drive
        reconnection across recreated inner clients.  Mirror of
        ``IPCClient._drain_loop`` for the recovery wrapper.
        """
        try:
            while self._events_running and self._state != ConnectionState.CLOSED:
                if self._state == ConnectionState.CONNECTED and self._client:
                    try:
                        async for event in self._client.events():
                            if isinstance(event, SessionInfoEvent) and event.session_id:
                                self._session_id = event.session_id
                            self._registry.dispatch(event)
                            self._fanout(event)

                        # Inner stream ended → connection lost.
                        if self._events_running and self._state == ConnectionState.CONNECTED:
                            logger.info("Connection lost, starting recovery...")
                            await self._start_reconnection()
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001 — keep the pump alive
                        logger.error(f"Error in event pump: {e}")
                        if self._events_running and self._state == ConnectionState.CONNECTED:
                            await self._start_reconnection()
                else:
                    # RECONNECTING / transient: poll for a state change.
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            # Release any events() consumers blocked on their queue.
            self._fanout(None)

    # ---- fan-out helpers (mirror IPCClient's drain fan-out) ----
    def _subscribe_events(self) -> "asyncio.Queue":
        q: asyncio.Queue = asyncio.Queue()
        self._event_subscribers.append(q)
        return q

    def _unsubscribe_events(self, q: "asyncio.Queue") -> None:
        try:
            self._event_subscribers.remove(q)
        except ValueError:
            pass

    def _fanout(self, event) -> None:
        for q in list(self._event_subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "recovery events fan-out: subscriber queue full; dropping event"
                )

    async def _stop_event_pump(self) -> None:
        """Cancel the background pump and release ``events()`` consumers."""
        task = self._event_pump_task
        self._event_pump_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 — best-effort teardown
                pass
        self._fanout(None)

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> ConnectionStatus:
        """Get current connection status.

        Returns:
            ConnectionStatus object with current state and details.
        """
        return ConnectionStatus(
            state=self._state,
            attempt=self._reconnect_attempt,
            max_attempts=self._config.max_attempts,
            next_retry_in=None,  # Updated by reconnection loop
            last_error=None,
            session_id=self._session_id,
            client_id=self._client_id,
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _check_can_send(self) -> None:
        """Check if we can send messages.

        Raises:
            ReconnectingError: If currently reconnecting.
            ConnectionClosedError: If connection is closed.
            ConnectionError: If not connected.
        """
        if self._state == ConnectionState.CLOSED:
            raise ConnectionClosedError()
        if self._state == ConnectionState.RECONNECTING:
            raise ReconnectingError()
        if self._state != ConnectionState.CONNECTED:
            raise ConnectionError("Not connected")

    def _transition_to(self, new_state: ConnectionState) -> None:
        """Transition to a new state and notify listeners.

        Must be called with _state_lock held.

        Args:
            new_state: The new state to transition to.
        """
        old_state = self._state
        self._state = new_state

        logger.debug(f"Connection state: {old_state.value} -> {new_state.value}")

        if self._on_status_change:
            try:
                self._on_status_change(self.get_status())
            except Exception as e:
                logger.warning(f"Error in status callback: {e}")

    def _notify_status(self, status: ConnectionStatus) -> None:
        """Notify listeners of a status update.

        Args:
            status: The status to report.
        """
        if self._on_status_change:
            try:
                self._on_status_change(status)
            except Exception as e:
                logger.warning(f"Error in status callback: {e}")

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff delay for a reconnection attempt.

        Uses exponential backoff with jitter.

        Args:
            attempt: Current attempt number (1-indexed).

        Returns:
            Delay in seconds before next attempt.
        """
        # Exponential backoff: base_delay * 2^(attempt-1)
        exp_delay = self._config.base_delay * (2 ** (attempt - 1))

        # Cap at max_delay
        capped_delay = min(self._config.max_delay, exp_delay)

        # Add jitter
        jitter_range = capped_delay * self._config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        delay = max(0.1, capped_delay + jitter)

        return delay

    async def _start_reconnection(self) -> None:
        """Start the reconnection process."""
        if not self._config.enabled:
            logger.info("Automatic reconnection disabled")
            async with self._state_lock:
                self._transition_to(ConnectionState.DISCONNECTED)
            return

        async with self._state_lock:
            if self._state == ConnectionState.CLOSED:
                return
            self._transition_to(ConnectionState.RECONNECTING)

        self._reconnect_attempt = 0
        self._reconnect_cancelled = False

        # Start reconnection loop in background
        self._reconnect_task = asyncio.create_task(self._reconnection_loop())

    async def _cancel_reconnection(self) -> None:
        """Cancel any ongoing reconnection."""
        self._reconnect_cancelled = True

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        self._reconnect_task = None

    async def _reconnection_loop(self) -> None:
        """Background task that handles reconnection with exponential backoff."""
        logger.info(f"Starting reconnection (max {self._config.max_attempts} attempts)")

        last_error: Optional[str] = None

        while (
            self._reconnect_attempt < self._config.max_attempts
            and not self._reconnect_cancelled
            and self._state == ConnectionState.RECONNECTING
        ):
            self._reconnect_attempt += 1
            delay = self._calculate_backoff(self._reconnect_attempt)

            logger.info(
                f"Reconnection attempt {self._reconnect_attempt}/{self._config.max_attempts} "
                f"in {delay:.1f}s"
            )

            # Notify UI of countdown
            self._notify_status(ConnectionStatus(
                state=ConnectionState.RECONNECTING,
                attempt=self._reconnect_attempt,
                max_attempts=self._config.max_attempts,
                next_retry_in=delay,
                last_error=last_error,
                session_id=self._session_id,
            ))

            # Wait before attempt
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                logger.debug("Reconnection wait cancelled")
                return

            if self._reconnect_cancelled or self._state != ConnectionState.RECONNECTING:
                return

            # Attempt reconnection
            try:
                success = await self._attempt_reconnect()
                if success:
                    logger.info("Reconnection successful!")
                    self._reconnect_attempt = 0
                    return

            except asyncio.CancelledError:
                logger.debug("Reconnection attempt cancelled")
                return

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Reconnection attempt {self._reconnect_attempt} failed: {e}"
                )

                # Classify error
                error_type = self._classify_error(e)
                if error_type == "permanent":
                    logger.error(f"Permanent error, stopping reconnection: {e}")
                    break

        # Max attempts exceeded or permanent error
        if not self._reconnect_cancelled:
            logger.error(
                f"Reconnection failed after {self._reconnect_attempt} attempts"
            )
            async with self._state_lock:
                self._transition_to(ConnectionState.CLOSED)

            self._notify_status(ConnectionStatus(
                state=ConnectionState.CLOSED,
                attempt=self._reconnect_attempt,
                max_attempts=self._config.max_attempts,
                last_error=last_error or "Max reconnection attempts exceeded",
                session_id=self._session_id,
            ))

    async def _attempt_reconnect(self) -> bool:
        """Single reconnection attempt.

        Returns:
            True if reconnection and session reattachment succeeded.
        """
        # Clean up old client
        if self._client:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

        # Create new client (never auto-start a restart)
        self._client = self._make_client(auto_start=False)

        # Connect with timeout
        try:
            connected = await asyncio.wait_for(
                self._client.connect(timeout=self._config.connection_timeout),
                timeout=self._config.connection_timeout + 1.0,
            )
        except asyncio.TimeoutError:
            raise ConnectionError("Connection timeout")

        if not connected:
            raise ConnectionError("Connection failed")

        self._client_id = self._client.client_id

        # Re-register host ("client") tools on the fresh inner client BEFORE
        # reattaching.  The reconnect built a new inner client with no host-tool
        # handlers; the reattached session still exposes those tools, so without
        # this the agent's next host-tool call would dangle.  Calls the inner
        # client directly (not the guarded proxy) since we're mid-reconnect.
        if self._registered_client_tools:
            try:
                await self._client.register_client_tools(self._registered_client_tools)
            except Exception as e:
                logger.warning(f"Failed to re-register client tools on reconnect: {e}")

        # Reattach to session if configured and we have a session ID
        if self._config.reattach_session and self._session_id:
            logger.info(f"Reattaching to session {self._session_id}")
            try:
                await self._client.attach_session(self._session_id)
            except Exception as e:
                logger.warning(f"Failed to reattach session: {e}")
                # Continue anyway - user may need to create new session

        # Success!
        async with self._state_lock:
            self._transition_to(ConnectionState.CONNECTED)

        return True

    def _classify_error(self, exc: Exception) -> str:
        """Classify a connection error for retry decisions.

        Args:
            exc: The exception to classify.

        Returns:
            "transient" for retryable errors, "permanent" for fatal errors.
        """
        exc_str = str(exc).lower()

        # Permanent errors - don't retry
        if isinstance(exc, IncompatibleServerError):
            # Server is too old; retrying won't change its version
            return "permanent"
        if isinstance(exc, FileNotFoundError):
            # Socket file deleted - server likely not restarting
            return "permanent"
        if "permission denied" in exc_str:
            return "permanent"
        if "authentication" in exc_str:
            return "permanent"

        # Transient errors - retry
        if isinstance(exc, (ConnectionRefusedError, ConnectionResetError)):
            return "transient"
        if isinstance(exc, asyncio.TimeoutError):
            return "transient"
        if "connection refused" in exc_str:
            return "transient"
        if "timeout" in exc_str:
            return "transient"

        # Default to transient (optimistic)
        return "transient"


__all__ = [
    "ConnectionClosedError",
    "ConnectionError",
    "ConnectionState",
    "ConnectionStatus",
    "IncompatibleServerError",
    "IPCRecoveryClient",
    "ReconnectingError",
    "StatusCallback",
]
