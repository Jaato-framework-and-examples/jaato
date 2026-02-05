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
    from ipc_recovery import IPCRecoveryClient, ConnectionState
    from client_config import get_recovery_config

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
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client_config import RecoveryConfig
from ipc_client import IPCClient

from server.events import (
    ConnectedEvent,
    ErrorEvent,
    Event,
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


class ConnectionError(Exception):
    """Error related to IPC connection."""
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
        socket_path: str,
        config: Optional[RecoveryConfig] = None,
        auto_start: bool = True,
        env_file: str = ".env",
        workspace_path: Optional[Path] = None,
        on_status_change: Optional[StatusCallback] = None,
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
        """
        self._socket_path = socket_path
        self._auto_start = auto_start
        self._env_file = env_file
        self._workspace_path = workspace_path
        self._on_status_change = on_status_change

        # Load config if not provided
        if config is None:
            from client_config import get_recovery_config
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

        # Reconnection state
        self._reconnect_attempt = 0
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_cancelled = False

        # Event forwarding
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._event_task: Optional[asyncio.Task] = None
        self._events_running = False

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

    # =========================================================================
    # Connection Management
    # =========================================================================

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
            self._client = IPCClient(
                socket_path=self._socket_path,
                auto_start=self._auto_start,
                env_file=self._env_file,
            )

            connected = await asyncio.wait_for(
                self._client.connect(timeout=timeout),
                timeout=timeout + 1.0,  # Slightly longer outer timeout
            )

            if connected:
                self._client_id = self._client.client_id
                async with self._state_lock:
                    self._transition_to(ConnectionState.CONNECTED)
                return True

            raise ConnectionError("Connection failed")

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

        # Stop event loop
        self._events_running = False

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

        # Stop event loop
        self._events_running = False

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

    async def create_session(self, name: Optional[str] = None) -> Optional[str]:
        """Create a new session.

        Args:
            name: Optional name for the session.

        Returns:
            Session ID if created, None otherwise.

        Raises:
            ReconnectingError: If currently reconnecting.
            ConnectionClosedError: If connection is closed.
        """
        self._check_can_send()

        if self._client:
            session_id = await self._client.create_session(name)
            if session_id:
                self._session_id = session_id
            return session_id
        return None

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
    ) -> None:
        """Send a message to the model.

        Args:
            text: The message text.
            attachments: Optional file attachments.

        Raises:
            ReconnectingError: If currently reconnecting.
            ConnectionClosedError: If connection is closed.
        """
        self._check_can_send()

        if self._client:
            await self._client.send_message(text, attachments)

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

    async def respond_to_reference_selection(
        self,
        request_id: str,
        response: str,
    ) -> None:
        """Respond to a reference selection request."""
        self._check_can_send()

        if self._client:
            await self._client.respond_to_reference_selection(request_id, response)

    async def stop(self) -> None:
        """Stop current operation."""
        # Allow stop even during reconnection
        if self._client and self._state == ConnectionState.CONNECTED:
            await self._client.stop()

    async def execute_command(
        self,
        command: str,
        args: Optional[list] = None,
    ) -> None:
        """Execute a command."""
        self._check_can_send()

        if self._client:
            await self._client.execute_command(command, args)

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
    # Event Stream
    # =========================================================================

    async def events(self) -> AsyncIterator[Event]:
        """Async iterator for receiving events with automatic reconnection.

        Yields events from the server. If the connection is lost and
        recovery is enabled, automatically attempts to reconnect.
        During reconnection, yields SystemMessageEvents with status updates.

        Yields:
            Events from the server.

        Example:
            async for event in client.events():
                if isinstance(event, AgentOutputEvent):
                    print(event.text)
        """
        self._events_running = True

        while self._events_running and self._state != ConnectionState.CLOSED:
            if self._state == ConnectionState.CONNECTED and self._client:
                try:
                    async for event in self._client.events():
                        # Track session ID from session events
                        if isinstance(event, SessionInfoEvent):
                            if event.session_id:
                                self._session_id = event.session_id

                        yield event

                    # If we get here, the event stream ended (connection lost)
                    if self._events_running and self._state == ConnectionState.CONNECTED:
                        logger.info("Connection lost, starting recovery...")
                        await self._start_reconnection()

                except asyncio.CancelledError:
                    logger.debug("Event stream cancelled")
                    break

                except Exception as e:
                    logger.error(f"Error in event stream: {e}")
                    if self._events_running and self._state == ConnectionState.CONNECTED:
                        await self._start_reconnection()

            elif self._state == ConnectionState.RECONNECTING:
                # Wait for reconnection or yield status updates
                try:
                    # Check periodically for state changes
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break

            else:
                # Not connected and not reconnecting, wait a bit
                try:
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break

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

        # Create new client
        self._client = IPCClient(
            socket_path=self._socket_path,
            auto_start=False,  # Don't auto-start during reconnection
            env_file=self._env_file,
        )

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
    "IPCRecoveryClient",
    "ReconnectingError",
    "StatusCallback",
]
