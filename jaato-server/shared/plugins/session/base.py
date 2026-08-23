"""Base types and protocol for Session Persistence plugins.

This module defines the interface that all session plugins must implement,
along with supporting types for configuration, state, and session metadata.

Session plugins provide persistence for conversation history, allowing users
to save and resume sessions across client restarts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from jaato_sdk.plugins.model_provider.types import Message


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

    profile_name: Optional[str] = None
    """Name of the SubagentProfile this session was spawned with.

    Persisted so disk-restore can re-bind the full provider recipe
    (model + provider + plugin_configs + system_instructions + GC
    strategy) at load time.  When the profile is absent / renamed /
    deleted between save and restore, the restore path raises a
    typed error rather than silently constructing a server with
    incomplete config.

    Replaces the pre-multi-provider trio of ``project`` / ``location`` /
    ``model`` which carried Google-GenAI-shaped fields directly on
    SessionState.  Those fields are tolerated on deserialize for
    backward-compat with old session JSONs but are no longer written
    or read by the framework — the profile is the authoritative
    recipe source.
    """

    profile_spec: Optional[Dict[str, Any]] = None
    """The UNRESOLVED inline-profile spec, when the session was created
    from an inline profile (``profile_name == "<inline>"``) rather than a
    named profile on disk.

    An inline profile has no re-resolvable name, so disk-restore cannot
    re-bind its recipe via the profile registry (the named-profile
    assumption behind ``profile_name``).  Persisting the original spec
    dict — the same JSON shape ``build_inline_profile`` accepts — lets
    restore reconstruct the full recipe (model + provider + plugins +
    plugin_configs + system_instructions + GC) with NO named profile.

    Stored **unresolved** (secret URIs like ``pass://`` preserved, not the
    resolved literals) so credentials never land in the on-disk session
    record; the daemon re-resolves them at restore, exactly as it does at
    create.  ``None`` for named-profile sessions (they restore via
    ``profile_name``) and for records written before this field existed.
    """

    workspace_path: Optional[str] = None
    """Workspace path (directory) where this session was created."""

    config_root: Optional[str] = None
    """Framework-config root override at session-creation time.

    Persisted so disk-restore can hand it back to
    :func:`discover_profiles` — the workspace tier and
    ``JAATO_PROFILE_SET`` subdir scans both gate on
    ``effective_config_root`` (see ``subagent/config.py:1685``).
    Without this, restoring a session spawned with a profile that
    lives under ``<config_root>/profiles/<set>/`` fails profile
    resolution because ``discover_profiles`` falls back to
    ``<workspace>/.jaato/profiles/`` which is the wrong directory
    for multi-set workspaces (canonical kb-cascade case: profiles
    live at ``<repo>/.jaato/profiles/zhipuai_glm5/``, sessions run
    under ``<repo>/tests/runs/<run>/``).

    None for sessions persisted before 2.4 OR sessions spawned
    without a config_root override (default workspace-only layout).
    """

    sandbox_mode: Optional[str] = None
    """Confinement mode at session-creation time (e.g. ``"apparmor"``).

    Persisted so disk-restore / orphan-revive re-applies the SAME
    confinement on runner re-spawn.  Without it a revived session's
    ``_load_session`` read of ``state.sandbox_mode`` was always None
    (the field didn't exist) → the re-spawned runner ran UNCONFINED
    after any idle detach — a security regression.  Mirrors the
    ``BootstrapEnvelope.sandbox_mode`` the restore path consumes;
    None on old records / never-confined sessions (unchanged behavior).
    """

    agent_name: Optional[str] = None
    """Agent/persona identity (``--agent <name>``) the session was
    spawned with.

    Persisted so disk-restore / orphan-revive rebinds the SAME persona
    (``.jaato/agents/<name>.md`` layered on the base instructions).
    Without it a revived session's ``_load_session`` built
    ``JaatoServer(agent_name=None)`` → no persona → persona-only
    guidance (e.g. "call ``enter_tier('vision')`` on user images") was
    silently dropped, so a revived multimodal session kept its
    ``enter_tier`` tool but never the instruction to use it → images
    hit the text tier and confabulated.  Mirrors
    ``BootstrapEnvelope.agent_name``; None on old records / no-persona
    sessions (unchanged behavior — the agent id falls back to "main").
    """

    budget_state: Optional[Dict[str, Any]] = None
    # Accumulated budget_control usage (usd / tokens / seconds / tool_calls /
    # turns).  DISTINCT from ``budget_state`` above, which is the
    # conversation/instruction budget -- a different subsystem entirely.
    #
    # Persisted because BudgetTracker accumulates in memory only: an unloaded
    # session came back with a zeroed tracker, so every CROSS-TURN ceiling
    # silently restarted.  Sessions unload on ORPHAN, so a suspend/resume
    # driver that disconnects during a wait is evicted every time -- the
    # longer a goal suspends, the more certainly the ceiling meant to bound it
    # resets.
    budget_usage: Optional[Dict[str, float]] = None
    """Serialized conversation budget for restoration."""

    interrupted_turn: Optional[Dict[str, Any]] = None
    """State of an interrupted turn for recovery on restart."""

    workspace_files: Optional[Dict[str, str]] = None
    """Tracked workspace file changes since session start.

    Maps relative file paths to status strings (``"created"``,
    ``"modified"``, ``"deleted"``).  Persisted so the monitor can be
    restored on session reload and the full delta replayed to
    reconnecting clients.
    """

    session_state: Optional[Dict[str, Any]] = None
    """Snapshot of session-attached state (extension-owned opaque
    storage) at save time.

    Captured from ``JaatoSession.get_all_session_state()`` — invokes
    every registered state provider so the snapshot reflects live
    values, not whatever was last pushed via ``set_session_state``.
    Persisted as JSON; the framework treats values as opaque
    (extensions encrypt before attach if confidentiality is needed).
    On resume, restored by re-attaching each key via
    ``JaatoSession.set_session_state``; consumer hooks fire and can
    re-register providers / instantiate runtime structures from the
    restored values.
    """


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

    profile_name: Optional[str] = None
    """Name of the SubagentProfile this session was spawned with.

    Denormalised at save time so session-list views can show recipe
    binding without resolving the profile registry for every entry.
    None for sessions persisted before this field landed OR for
    sessions spawned without a profile (legacy / test paths).
    """

    workspace_path: Optional[str] = None
    """Workspace path (directory) where this session was created."""

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

    def save(
        self,
        state: SessionState,
        storage_dir: Optional[Path] = None,
    ) -> None:
        """Save session state to persistent storage.

        Args:
            state: The complete session state to persist.
            storage_dir: Override storage directory. When None, uses the
                directory set during initialize(). SessionManager passes
                a workspace-resolved path; standalone JaatoClient omits it.

        Raises:
            IOError: If the session cannot be saved.
        """
        ...

    def load(
        self,
        session_id: str,
        storage_dir: Optional[Path] = None,
    ) -> SessionState:
        """Load session state from persistent storage.

        Args:
            session_id: The session ID to load.
            storage_dir: Override storage directory (see save() for details).

        Returns:
            The loaded SessionState.

        Raises:
            FileNotFoundError: If the session does not exist.
            ValueError: If the session data is corrupted.
        """
        ...

    def list_sessions(
        self,
        storage_dir: Optional[Path] = None,
    ) -> List[SessionInfo]:
        """List all available sessions.

        Args:
            storage_dir: Override storage directory (see save() for details).

        Returns:
            List of SessionInfo objects, sorted by updated_at descending.
        """
        ...

    def delete(
        self,
        session_id: str,
        storage_dir: Optional[Path] = None,
    ) -> bool:
        """Delete a session from storage.

        Args:
            session_id: The session ID to delete.
            storage_dir: Override storage directory (see save() for details).

        Returns:
            True if deleted, False if session didn't exist.
        """
        ...

    def get_latest(
        self,
        storage_dir: Optional[Path] = None,
    ) -> Optional[SessionInfo]:
        """Get the most recently updated session.

        Args:
            storage_dir: Override storage directory (see save() for details).

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

    def set_description(
        self,
        session_id: str,
        description: str,
        storage_dir: Optional[Path] = None,
    ) -> None:
        """Set the description for a session.

        Called when the model provides a session description via tool call.

        Args:
            session_id: The session ID to update.
            description: The model-generated description.
            storage_dir: Override storage directory (see save() for details).
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
