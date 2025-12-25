"""Session Manager for multi-session support.

This module manages multiple named sessions, each with its own
JaatoServer instance and conversation state.

Sessions are identified by name and can be:
- Created on demand
- Attached to by multiple clients
- Persisted and resumed
- Garbage collected when idle
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from .core import JaatoServer
from .events import (
    Event,
    EventType,
    SystemMessageEvent,
    ErrorEvent,
)


logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Metadata about a session."""
    session_id: str
    name: str
    created_at: str
    last_activity: str
    model_provider: str
    model_name: str
    is_processing: bool
    client_count: int
    turn_count: int


@dataclass
class Session:
    """A managed session with its JaatoServer."""
    session_id: str
    name: str
    server: JaatoServer
    created_at: str
    last_activity: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    attached_clients: Set[str] = field(default_factory=set)


# Event types for session management
@dataclass
class SessionCreatedEvent(Event):
    """Emitted when a session is created."""
    type: EventType = field(default=EventType.SYSTEM_MESSAGE)
    session_id: str = ""
    session_name: str = ""


@dataclass
class SessionListEvent(Event):
    """Response to session list request."""
    type: EventType = field(default=EventType.SYSTEM_MESSAGE)
    sessions: List[Dict[str, Any]] = field(default_factory=list)


class SessionManager:
    """Manages multiple named sessions.

    Each session has its own JaatoServer with isolated:
    - Conversation history
    - Agent state
    - Plugin state
    - Token accounting

    Clients can:
    - Create new sessions
    - Attach to existing sessions
    - List available sessions
    - Switch between sessions
    """

    def __init__(
        self,
        env_file: str = ".env",
        provider: Optional[str] = None,
        default_session_name: str = "default",
    ):
        """Initialize the session manager.

        Args:
            env_file: Path to .env file.
            provider: Model provider override.
            default_session_name: Name for the default session.
        """
        self._env_file = env_file
        self._provider = provider
        self._default_session_name = default_session_name

        # Session storage
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

        # Event routing
        self._client_to_session: Dict[str, str] = {}
        self._event_callback: Optional[Callable[[str, Event], None]] = None

    def set_event_callback(
        self,
        callback: Callable[[str, Event], None],
    ) -> None:
        """Set callback for routing events to clients.

        Args:
            callback: Called with (client_id, event) for each event.
        """
        self._event_callback = callback

    def _emit_to_client(self, client_id: str, event: Event) -> None:
        """Emit an event to a specific client."""
        if self._event_callback:
            self._event_callback(client_id, event)

    def _emit_to_session(self, session_id: str, event: Event) -> None:
        """Emit an event to all clients attached to a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                for client_id in session.attached_clients:
                    self._emit_to_client(client_id, event)

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    def create_session(
        self,
        client_id: str,
        session_name: Optional[str] = None,
    ) -> str:
        """Create a new session and attach the client.

        Args:
            client_id: The requesting client.
            session_name: Optional name (auto-generated if not provided).

        Returns:
            The session ID.
        """
        # Generate session ID and name
        session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self._sessions)}"
        name = session_name or f"Session {len(self._sessions) + 1}"

        # Create JaatoServer for this session
        server = JaatoServer(
            env_file=self._env_file,
            provider=self._provider,
            on_event=lambda e: self._emit_to_session(session_id, e),
        )

        # Initialize the server
        if not server.initialize():
            self._emit_to_client(client_id, ErrorEvent(
                error="Failed to initialize session",
                error_type="SessionError",
            ))
            return ""

        # Create session
        session = Session(
            session_id=session_id,
            name=name,
            server=server,
            created_at=datetime.utcnow().isoformat(),
        )

        with self._lock:
            self._sessions[session_id] = session
            session.attached_clients.add(client_id)
            self._client_to_session[client_id] = session_id

        logger.info(f"Session created: {session_id} ({name})")

        self._emit_to_client(client_id, SystemMessageEvent(
            message=f"Session created: {name} ({session_id})",
            style="info",
        ))

        return session_id

    def attach_session(
        self,
        client_id: str,
        session_id: str,
    ) -> bool:
        """Attach a client to an existing session.

        Args:
            client_id: The requesting client.
            session_id: The session to attach to.

        Returns:
            True if attached successfully.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"Session not found: {session_id}",
                    error_type="SessionError",
                ))
                return False

            # Detach from current session if any
            current = self._client_to_session.get(client_id)
            if current and current in self._sessions:
                self._sessions[current].attached_clients.discard(client_id)

            # Attach to new session
            session.attached_clients.add(client_id)
            self._client_to_session[client_id] = session_id

        logger.info(f"Client {client_id} attached to session {session_id}")

        self._emit_to_client(client_id, SystemMessageEvent(
            message=f"Attached to session: {session.name} ({session_id})",
            style="info",
        ))

        return True

    def detach_client(self, client_id: str) -> None:
        """Detach a client from its current session.

        Args:
            client_id: The client to detach.
        """
        with self._lock:
            session_id = self._client_to_session.pop(client_id, None)
            if session_id and session_id in self._sessions:
                self._sessions[session_id].attached_clients.discard(client_id)
                logger.info(f"Client {client_id} detached from session {session_id}")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and disconnect all clients.

        Args:
            session_id: The session to delete.

        Returns:
            True if deleted.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                return False

            # Notify attached clients
            for client_id in session.attached_clients:
                self._emit_to_client(client_id, SystemMessageEvent(
                    message=f"Session deleted: {session.name}",
                    style="warning",
                ))
                self._client_to_session.pop(client_id, None)

            # Shutdown the server
            session.server.shutdown()

        logger.info(f"Session deleted: {session_id}")
        return True

    def get_or_create_default(self, client_id: str) -> str:
        """Get the default session, creating if needed.

        Args:
            client_id: The requesting client.

        Returns:
            The default session ID.
        """
        # Look for existing default session
        with self._lock:
            for session in self._sessions.values():
                if session.name == self._default_session_name:
                    session.attached_clients.add(client_id)
                    self._client_to_session[client_id] = session.session_id
                    return session.session_id

        # Create default session
        return self.create_session(client_id, self._default_session_name)

    # =========================================================================
    # Session Queries
    # =========================================================================

    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions with metadata."""
        result = []
        with self._lock:
            for session in self._sessions.values():
                result.append(SessionInfo(
                    session_id=session.session_id,
                    name=session.name,
                    created_at=session.created_at,
                    last_activity=session.last_activity,
                    model_provider=session.server.model_provider,
                    model_name=session.server.model_name,
                    is_processing=session.server.is_processing,
                    client_count=len(session.attached_clients),
                    turn_count=len(session.server.get_history()),
                ))
        return result

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_client_session(self, client_id: str) -> Optional[Session]:
        """Get the session a client is attached to."""
        with self._lock:
            session_id = self._client_to_session.get(client_id)
            if session_id:
                return self._sessions.get(session_id)
        return None

    # =========================================================================
    # Request Routing
    # =========================================================================

    def handle_request(
        self,
        client_id: str,
        session_id: str,
        event: Event,
    ) -> None:
        """Route a request to the appropriate session.

        Args:
            client_id: The requesting client.
            session_id: The target session.
            event: The request event.
        """
        session = self.get_session(session_id)
        if not session:
            self._emit_to_client(client_id, ErrorEvent(
                error=f"Session not found: {session_id}",
                error_type="SessionError",
            ))
            return

        # Update activity timestamp
        session.last_activity = datetime.utcnow().isoformat()

        # Route to session's server
        server = session.server

        from .events import (
            SendMessageRequest,
            PermissionResponseRequest,
            ClarificationResponseRequest,
            StopRequest,
            CommandRequest,
        )

        if isinstance(event, SendMessageRequest):
            # Run in thread to not block
            threading.Thread(
                target=lambda: server.send_message(
                    event.text,
                    event.attachments if event.attachments else None
                ),
                daemon=True,
            ).start()

        elif isinstance(event, PermissionResponseRequest):
            server.respond_to_permission(event.request_id, event.response)

        elif isinstance(event, ClarificationResponseRequest):
            server.respond_to_clarification(event.request_id, event.response)

        elif isinstance(event, StopRequest):
            server.stop()

        elif isinstance(event, CommandRequest):
            result = server.execute_command(event.command, event.args)
            self._emit_to_client(client_id, SystemMessageEvent(
                message=str(result),
                style="info",
            ))

        else:
            self._emit_to_client(client_id, ErrorEvent(
                error=f"Unknown request type: {type(event).__name__}",
                error_type="RequestError",
            ))

    # =========================================================================
    # Cleanup
    # =========================================================================

    def shutdown(self) -> None:
        """Shutdown all sessions."""
        with self._lock:
            for session_id in list(self._sessions.keys()):
                session = self._sessions.pop(session_id, None)
                if session:
                    session.server.shutdown()
            self._client_to_session.clear()

        logger.info("Session manager shutdown complete")
