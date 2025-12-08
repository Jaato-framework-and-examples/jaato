"""Base types and protocol for Session Persistence plugins.

This module defines the interface that all session plugins must implement,
along with supporting types for configuration, state, and session metadata.

Session plugins provide persistence for conversation history, allowing users
to save and resume sessions across client restarts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..model_provider.types import Message


@dataclass
class SessionState:
    """Complete state of a session for persistence.

    Contains all data needed to restore a session to its previous state.
    """

    session_id: str
    """Unique identifier for this session (timestamp-based)."""

    history: List[Message]
    """Conversation history as list of Message objects."""

    created_at: datetime
    """When the session was first created."""

    updated_at: datetime
    """When the session was last saved."""

    description: Optional[str] = None
    """Model-generated description of the session (set after a few turns)."""

    turn_count: int = 0
    """Number of conversation turns in this session."""

    turn_accounting: List[Dict[str, int]] = field(default_factory=list)
    """Token usage per turn: [{'prompt': N, 'output': M, 'total': O}, ...]."""

    user_inputs: List[str] = field(default_factory=list)
    """Original user inputs for readline/prompt history restoration."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional plugin-specific metadata."""

    # Connection info for resumption
    project: Optional[str] = None
    """GCP project ID used for this session."""

    location: Optional[str] = None
    """Vertex AI location used for this session."""

    model: Optional[str] = None
    """Model name used for this session."""


@dataclass
class SessionInfo:
    """Lightweight session metadata for listing sessions.

    Used by list_sessions() to avoid loading full history.
    """

    session_id: str
    """Unique identifier for this session."""

    description: Optional[str]
    """Model-generated description, or None if not yet named."""

    created_at: datetime
    """When the session was first created."""

    updated_at: datetime
    """When the session was last saved."""

    turn_count: int
    """Number of conversation turns."""

    model: Optional[str] = None
    """Model name used for this session."""

    def display_name(self) -> str:
        """Return a display-friendly name for the session."""
        if self.description:
            return f'{self.session_id} - "{self.description}"'
        return f"{self.session_id} (unnamed)"


@dataclass
class SessionConfig:
    """Configuration for session persistence.

    Controls auto-save behavior, naming, and storage limits.
    """

    # Storage settings
    storage_path: str = ".jaato/sessions"
    """Directory for session files."""

    # Auto-save settings
    auto_save_on_exit: bool = True
    """Whether to automatically save the session on clean shutdown."""

    auto_save_interval: Optional[int] = None
    """Auto-save interval in seconds (None = disabled)."""

    checkpoint_after_turns: Optional[int] = None
    """Save checkpoint every N turns (None = disabled)."""

    # Resume settings
    auto_resume_last: bool = False
    """Whether to automatically resume the last session on connect."""

    # Naming settings
    request_description_after_turns: int = 3
    """Request model-generated description after this many turns."""

    # Cleanup settings
    max_sessions: int = 20
    """Maximum number of sessions to keep (oldest deleted first)."""

    # Plugin-specific configuration
    plugin_config: Dict[str, Any] = field(default_factory=dict)
    """Additional configuration passed to the session plugin."""


@runtime_checkable
class SessionPlugin(Protocol):
    """Protocol for Session Persistence plugins.

    Session plugins handle saving and loading conversation history,
    allowing users to resume sessions across client restarts.

    This follows the same pattern as GCPlugin - JaatoClient accepts
    any plugin implementing this interface via set_session_plugin().

    Example implementation:
        class FileSessionPlugin:
            @property
            def name(self) -> str:
                return "session_file"

            def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
                self._storage_path = config.get('storage_path', '.jaato/sessions')

            def save(self, state: SessionState) -> None:
                # Serialize and write to file
                ...

            def load(self, session_id: str) -> SessionState:
                # Read and deserialize from file
                ...
    """

    @property
    def name(self) -> str:
        """Unique identifier for this session plugin (e.g., 'session_file')."""
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with configuration.

        Args:
            config: Plugin-specific configuration dictionary.
        """
        ...

    def shutdown(self) -> None:
        """Clean up any resources held by the plugin."""
        ...

    # ==================== Core Persistence ====================

    def save(self, state: SessionState) -> None:
        """Save session state to persistent storage.

        Args:
            state: The complete session state to persist.

        Raises:
            IOError: If the session cannot be saved.
        """
        ...

    def load(self, session_id: str) -> SessionState:
        """Load session state from persistent storage.

        Args:
            session_id: The session ID to load.

        Returns:
            The loaded SessionState.

        Raises:
            FileNotFoundError: If the session does not exist.
            ValueError: If the session data is corrupted.
        """
        ...

    def list_sessions(self) -> List[SessionInfo]:
        """List all available sessions.

        Returns:
            List of SessionInfo objects, sorted by updated_at descending.
        """
        ...

    def delete(self, session_id: str) -> bool:
        """Delete a session from storage.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted, False if session didn't exist.
        """
        ...

    def get_latest(self) -> Optional[SessionInfo]:
        """Get the most recently updated session.

        Returns:
            SessionInfo for the latest session, or None if no sessions exist.
        """
        ...

    # ==================== Lifecycle Hooks ====================
    # These are called by JaatoClient at appropriate times

    def on_turn_complete(
        self,
        state: SessionState,
        config: SessionConfig
    ) -> None:
        """Called after each conversation turn completes.

        Plugins can use this for checkpoint saves or tracking.

        Args:
            state: Current session state.
            config: Session configuration.
        """
        ...

    def on_session_start(
        self,
        config: SessionConfig
    ) -> Optional[SessionState]:
        """Called when a new client session starts.

        If auto_resume_last is enabled, this should return the last
        session's state for restoration.

        Args:
            config: Session configuration.

        Returns:
            SessionState to restore, or None to start fresh.
        """
        ...

    def on_session_end(
        self,
        state: SessionState,
        config: SessionConfig
    ) -> None:
        """Called when the client session ends cleanly.

        If auto_save_on_exit is enabled, this should save the session.

        Args:
            state: Current session state.
            config: Session configuration.
        """
        ...

    # ==================== Description Management ====================

    def set_description(self, session_id: str, description: str) -> None:
        """Set the description for a session.

        Called when the model provides a session description via tool call.

        Args:
            session_id: The session ID to update.
            description: The model-generated description.
        """
        ...

    def needs_description(self, state: SessionState, config: SessionConfig) -> bool:
        """Check if the session needs a description.

        Returns True if:
        - Session has no description
        - Turn count >= config.request_description_after_turns

        Args:
            state: Current session state.
            config: Session configuration.

        Returns:
            True if description should be requested from model.
        """
        ...
