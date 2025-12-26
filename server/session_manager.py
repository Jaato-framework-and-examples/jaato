"""Session Manager for multi-session support.

This module manages multiple named sessions, each with its own
JaatoServer instance and conversation state.

Sessions are:
- Persisted to disk via the Session Plugin
- Loaded on-demand when clients attach
- Saved periodically and on shutdown
- Identified by consistent IDs across memory and disk

Integration with Session Plugin:
- SessionManager uses SessionPlugin for persistence
- SessionState from the plugin is used for save/load
- Session IDs are consistent between runtime and storage
"""

import logging
import sys
import pathlib
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.plugins.session import (
    create_plugin as create_session_plugin,
    load_session_config,
    SessionPlugin,
    SessionState,
    SessionConfig,
    SessionInfo as PluginSessionInfo,
)

from .core import JaatoServer
from .events import (
    Event,
    EventType,
    SystemMessageEvent,
    ErrorEvent,
)


logger = logging.getLogger(__name__)


@dataclass
class RuntimeSessionInfo:
    """Metadata about a session (runtime + persisted)."""
    session_id: str
    name: str
    description: Optional[str]
    created_at: str
    last_activity: str
    model_provider: str
    model_name: str
    is_processing: bool
    is_loaded: bool  # True if currently in memory
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
    description: Optional[str] = None
    is_dirty: bool = False  # True if has unsaved changes


class SessionManager:
    """Manages multiple named sessions with persistence.

    Integrates with the Session Plugin to provide:
    - Persistent storage of session history
    - Load sessions on-demand from disk
    - Save sessions periodically and on shutdown
    - Unified view of in-memory and on-disk sessions

    Each session has its own JaatoServer with isolated:
    - Conversation history
    - Agent state
    - Plugin state
    - Token accounting

    Clients can:
    - Create new sessions
    - Attach to existing sessions (loads from disk if needed)
    - List all sessions (memory + disk)
    - Save/checkpoint sessions
    """

    def __init__(
        self,
        env_file: str = ".env",
        provider: Optional[str] = None,
        default_session_name: str = "default",
        storage_path: Optional[str] = None,
    ):
        """Initialize the session manager.

        Args:
            env_file: Path to .env file.
            provider: Model provider override.
            default_session_name: Name for the default session.
            storage_path: Override for session storage path.
        """
        self._env_file = env_file
        self._provider = provider
        self._default_session_name = default_session_name

        # Initialize session plugin for persistence
        self._session_plugin: SessionPlugin = create_session_plugin()
        self._session_config: SessionConfig = load_session_config()

        if storage_path:
            self._session_config.storage_path = storage_path

        self._session_plugin.initialize({
            'storage_path': self._session_config.storage_path
        })

        # In-memory session storage
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

        # Client to session mapping
        self._client_to_session: Dict[str, str] = {}

        # Event routing callback
        self._event_callback: Optional[Callable[[str, Event], None]] = None

        logger.info(f"SessionManager initialized with storage: {self._session_config.storage_path}")

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
            The session ID (empty string on failure).
        """
        # Generate session ID (matches Session Plugin format)
        timestamp = datetime.utcnow()
        session_id = timestamp.strftime("%Y%m%d_%H%M%S")
        name = session_name or f"Session {timestamp.strftime('%Y-%m-%d %H:%M')}"

        # Check for collision with existing session
        existing = self._get_persisted_sessions()
        existing_ids = {s.session_id for s in existing}
        counter = 0
        original_id = session_id
        while session_id in existing_ids or session_id in self._sessions:
            counter += 1
            session_id = f"{original_id}_{counter}"

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

        # Create session object
        session = Session(
            session_id=session_id,
            name=name,
            server=server,
            created_at=timestamp.isoformat(),
            description=None,
            is_dirty=True,  # New session needs saving
        )

        with self._lock:
            self._sessions[session_id] = session
            session.attached_clients.add(client_id)
            self._client_to_session[client_id] = session_id

        # Save initial state to disk
        self._save_session(session)

        logger.info(f"Session created: {session_id} ({name})")

        # Emit current agent state to the newly attached client
        server.emit_current_state(lambda e: self._emit_to_client(client_id, e))

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

        If the session is not in memory, attempts to load from disk.

        Args:
            client_id: The requesting client.
            session_id: The session to attach to.

        Returns:
            True if attached successfully.
        """
        with self._lock:
            # Check if session is in memory
            session = self._sessions.get(session_id)

            if not session:
                # Try to load from disk
                session = self._load_session(session_id)
                if session:
                    self._sessions[session_id] = session

            if not session:
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"Session not found: {session_id}",
                    error_type="SessionError",
                ))
                return False

            # Detach from current session if any
            current = self._client_to_session.get(client_id)
            if current and current in self._sessions:
                old_session = self._sessions[current]
                old_session.attached_clients.discard(client_id)
                # Consider unloading if no clients
                self._maybe_unload_session(current)

            # Attach to new session
            session.attached_clients.add(client_id)
            self._client_to_session[client_id] = session_id

        logger.info(f"Client {client_id} attached to session {session_id}")

        # Emit current agent state to the newly attached client
        session.server.emit_current_state(lambda e: self._emit_to_client(client_id, e))

        self._emit_to_client(client_id, SystemMessageEvent(
            message=f"Attached to session: {session.name} ({session_id})",
            style="info",
        ))

        return True

    def _load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from disk.

        Args:
            session_id: The session ID to load.

        Returns:
            The loaded Session, or None if not found.
        """
        try:
            state = self._session_plugin.load(session_id)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

        # Create JaatoServer and restore state
        server = JaatoServer(
            env_file=self._env_file,
            provider=self._provider,
            on_event=lambda e: self._emit_to_session(session_id, e),
        )

        if not server.initialize():
            logger.error(f"Failed to initialize server for session {session_id}")
            return None

        # Restore history to the server's JaatoClient
        if state.history and server._jaato:
            server._jaato.reset_session(state.history)
            logger.info(f"Restored {len(state.history)} messages for session {session_id}")

        session = Session(
            session_id=session_id,
            name=state.description or f"Session {session_id}",
            server=server,
            created_at=state.created_at.isoformat(),
            last_activity=state.updated_at.isoformat(),
            description=state.description,
            is_dirty=False,
        )

        logger.info(f"Loaded session from disk: {session_id}")
        return session

    def _save_session(self, session: Session) -> bool:
        """Save a session to disk.

        Args:
            session: The session to save.

        Returns:
            True if saved successfully.
        """
        try:
            # Get history from server
            history = session.server.get_history() if session.server else []
            turn_accounting = []

            if session.server and "main" in session.server._agents:
                turn_accounting = session.server._agents["main"].turn_accounting

            # Create SessionState
            state = SessionState(
                session_id=session.session_id,
                history=history,
                created_at=datetime.fromisoformat(session.created_at),
                updated_at=datetime.utcnow(),
                description=session.description or session.name,
                turn_count=len(history) // 2,  # Approximate
                turn_accounting=turn_accounting,
                model=session.server.model_name if session.server else None,
            )

            self._session_plugin.save(state)
            session.is_dirty = False

            logger.debug(f"Saved session: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False

    def _maybe_unload_session(self, session_id: str) -> None:
        """Unload a session from memory if no clients attached.

        Saves to disk first if dirty.

        Args:
            session_id: The session to potentially unload.
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        if session.attached_clients:
            return  # Still has clients

        # Save before unloading
        if session.is_dirty:
            self._save_session(session)

        # Shutdown server and remove from memory
        session.server.shutdown()
        del self._sessions[session_id]
        logger.info(f"Unloaded session: {session_id}")

    def detach_client(self, client_id: str) -> None:
        """Detach a client from its current session.

        Args:
            client_id: The client to detach.
        """
        with self._lock:
            session_id = self._client_to_session.pop(client_id, None)
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.attached_clients.discard(client_id)
                logger.info(f"Client {client_id} detached from session {session_id}")

                # Maybe unload if no more clients
                self._maybe_unload_session(session_id)

    def save_session(self, session_id: str) -> bool:
        """Explicitly save a session to disk.

        Args:
            session_id: The session to save.

        Returns:
            True if saved successfully.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            return self._save_session(session)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from memory and disk.

        Args:
            session_id: The session to delete.

        Returns:
            True if deleted.
        """
        with self._lock:
            # Remove from memory
            session = self._sessions.pop(session_id, None)
            if session:
                # Notify attached clients
                for client_id in session.attached_clients:
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message=f"Session deleted: {session.name}",
                        style="warning",
                    ))
                    self._client_to_session.pop(client_id, None)

                # Shutdown the server
                session.server.shutdown()

        # Delete from disk
        deleted = self._session_plugin.delete(session_id)

        logger.info(f"Session deleted: {session_id}")
        return deleted or session is not None

    def get_or_create_default(self, client_id: str) -> str:
        """Get the default session, creating if needed.

        Tries to find/load default session first, creates if not found.

        Args:
            client_id: The requesting client.

        Returns:
            The default session ID.
        """
        # Check in-memory sessions first
        with self._lock:
            for session in self._sessions.values():
                if session.name == self._default_session_name:
                    session.attached_clients.add(client_id)
                    self._client_to_session[client_id] = session.session_id
                    # Emit current agent state to the newly attached client
                    session.server.emit_current_state(lambda e: self._emit_to_client(client_id, e))
                    return session.session_id

        # Check persisted sessions
        persisted = self._get_persisted_sessions()
        for info in persisted:
            if info.description == self._default_session_name:
                if self.attach_session(client_id, info.session_id):
                    return info.session_id

        # Create new default session
        return self.create_session(client_id, self._default_session_name)

    # =========================================================================
    # Session Queries
    # =========================================================================

    def _get_persisted_sessions(self) -> List[PluginSessionInfo]:
        """Get list of sessions from disk."""
        try:
            return self._session_plugin.list_sessions()
        except Exception as e:
            logger.error(f"Failed to list persisted sessions: {e}")
            return []

    def list_sessions(self) -> List[RuntimeSessionInfo]:
        """List all sessions (in-memory and on-disk).

        Returns merged view with runtime status for loaded sessions.
        """
        result: Dict[str, RuntimeSessionInfo] = {}

        # Add persisted sessions first
        for info in self._get_persisted_sessions():
            result[info.session_id] = RuntimeSessionInfo(
                session_id=info.session_id,
                name=info.description or info.session_id,
                description=info.description,
                created_at=info.created_at.isoformat(),
                last_activity=info.updated_at.isoformat(),
                model_provider="",
                model_name=info.model or "",
                is_processing=False,
                is_loaded=False,
                client_count=0,
                turn_count=info.turn_count,
            )

        # Overlay in-memory sessions (have more current info)
        with self._lock:
            for session in self._sessions.values():
                result[session.session_id] = RuntimeSessionInfo(
                    session_id=session.session_id,
                    name=session.name,
                    description=session.description,
                    created_at=session.created_at,
                    last_activity=session.last_activity,
                    model_provider=session.server.model_provider,
                    model_name=session.server.model_name,
                    is_processing=session.server.is_processing,
                    is_loaded=True,
                    client_count=len(session.attached_clients),
                    turn_count=len(session.server.get_history()) // 2,
                )

        # Sort by last activity
        sessions = list(result.values())
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID (in-memory only)."""
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
        session.is_dirty = True

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
            def run_message():
                server.send_message(
                    event.text,
                    event.attachments if event.attachments else None
                )
                # Auto-save after turn
                self._save_session(session)

            threading.Thread(target=run_message, daemon=True).start()

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

    def save_all(self) -> int:
        """Save all dirty sessions to disk.

        Returns:
            Number of sessions saved.
        """
        saved = 0
        with self._lock:
            for session in self._sessions.values():
                if session.is_dirty:
                    if self._save_session(session):
                        saved += 1
        return saved

    def shutdown(self) -> None:
        """Shutdown all sessions, saving to disk first."""
        logger.info("SessionManager shutting down...")

        with self._lock:
            # Save all sessions
            for session in self._sessions.values():
                self._save_session(session)
                session.server.shutdown()

            self._sessions.clear()
            self._client_to_session.clear()

        self._session_plugin.shutdown()
        logger.info("SessionManager shutdown complete")
