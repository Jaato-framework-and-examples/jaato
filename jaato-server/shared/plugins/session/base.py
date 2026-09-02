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

    profile_snapshot: Optional[Dict[str, Any]] = None
    """The RESOLVED profile this session actually ran under, frozen at
    creation (record version 2.8+, issue #787).

    Distinct from :attr:`profile_spec`, which persists an *inline* recipe
    because there is no name to resolve back to.  This one persists a
    *named* profile's resolved form, because re-resolving a name reads the
    profile FILES AS THEY ARE AT REVIVE TIME — so an edit between creation
    and revive silently changed what a revived session ran under, and a
    session came back with its original history under a different recipe.

    The operator decision recorded on #787 is that a revived session keeps
    what it was created with, and a session that wants new instructions is
    a new session.  This field is what makes that true rather than
    aspirational.

    Written by :func:`shared.plugins.subagent.config.profile_to_snapshot`
    and read back by ``profile_from_snapshot``.  Secret URIs
    (``pass://``, ``vault://``) are carried UNRESOLVED, exactly as
    ``profile_spec`` carries them — a resolved profile holds the URI, not
    the credential; expansion happens later on the daemon.

    ``None`` on records written before 2.8 and on inline-profile sessions
    (which restore via ``profile_spec``).  Absent, the revive falls back to
    re-resolving ``profile_name`` from disk — i.e. exactly the pre-2.8
    behaviour, so old records keep loading.
    """

    rendered_instructions: Optional[str] = None
    """The system instruction EXACTLY as rendered at session-prep — the
    prompt the model was actually given on turn 1 (record version 2.8+,
    issue #787).

    Includes the persona, the framework layers the profile did not
    suppress, plugin instructions, and the OUTPUT of every ``{{!py:...}}``
    prefetch placeholder.  Snapshotted at the end of
    ``JaatoSession.configure()`` and read back through
    ``BootstrapEnvelope.system_instruction_override``.

    WHY THIS IS PERSISTED AT ALL.  A revived session used to REBUILD its
    prompt: re-read ``.jaato/instructions/``, re-resolve the agent
    markdown, and RE-RUN the prefetch scripts.  Re-running is what broke
    #787 — ``agent_params`` were not persisted, so a mandatory prefetch
    that reads them aborted session-prep and the session could not be
    woken by anything that goes through ``_load_session``.  It was also
    wrong in two quieter ways: a prefetch is documented as running once,
    before turn 1, yet a side-effecting one (the reported case
    *materialises a git worktree*) re-ran on every revive; and the rebuilt
    prompt could differ from the original, so the session resumed with a
    history produced under one prompt and continued under another.

    Restoring the render removes all three.  ``None`` on pre-2.8 records
    and on sessions whose runner never reported one — both fall back to
    re-rendering, which is the pre-2.8 behaviour, so this change is
    backward compatible by construction.
    """

    agent_params: Optional[Dict[str, str]] = None
    """The ``agent_params`` the session was created with (record version
    2.8+, issue #787).

    These fill ``{{param}}`` placeholders in the persona and are the
    documented channel a ``{{!py:...}}`` prefetch reads its per-agent
    inputs from (``context.agent_params``).  Not persisting them is the
    proximate cause of #787: bootstrap re-ran the prefetch on revive and
    handed it an empty dict, so the script raised and blamed the task
    definition — which was correct all along.

    On the default revive path the prefetch does not re-run at all
    (``rendered_instructions`` above is restored instead), so these are
    persisted for the OPT-IN re-render path
    (``JAATO_REVIVE_PERSONA=disk``), which re-renders the persona from
    disk against the ORIGINAL params rather than against nothing.

    CONTRACT FOR AUTHORS: **never pass a credential as an agent_param.**
    ``resolve_agent`` substitutes them into the persona, so anything put
    here already reaches the model in its system prompt — and, since 2.8,
    lands on disk inside ``rendered_instructions`` whether or not this
    field exists.  Use ``profile.env`` with a ``pass://`` / ``vault://``
    URI for secrets; those stay unresolved on disk and are resolved
    daemon-side at spawn.
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
    # Why a budget ceiling STOPPED this session, if it did.  Persisted
    # alongside the usage above because usage alone was not enough: an abort
    # rung latches this, and it is what `_refuse_if_budget_exhausted` reads to
    # turn a crossed ceiling into a refused turn.  Without it a reloaded
    # session held usage AT the ceiling with no memory of being stopped, so it
    # served one more turn -- the re-assert lands in that turn's `finally`,
    # one turn too late -- and a goal finishing inside it sailed through.
    budget_exhausted_reason: Optional[str] = None
    # The EFFECTIVE ``budget_control`` config this session ran under, as the
    # dict ``BudgetControlConfig.to_dict`` produces.  DISTINCT from
    # ``budget_usage`` (what was spent) -- this is the CEILING itself.
    #
    # Needed because a budget reaches the runner ONLY via the profile, and a
    # budget declared outside the profile (``cascade_budget_set`` on a driver,
    # where limits are a per-run operator choice) leaves nothing for restore
    # to rebuild from: the revived session came back with no BudgetTracker at
    # all, so no cross-turn ceiling could fire however many resumes ran.
    # Persisting the resolved ceiling lets restore re-attach it to the
    # rebuilt profile.  ``None`` for genuinely unbudgeted sessions.
    budget_control: Optional[Dict[str, Any]] = None
    # Cascade-scoped sibling ADDRESS (design §4).  Persisted because an
    # address that does not survive a reload is not an address --
    # sessions unload on ORPHAN, and a sibling that came back nameless
    # would be unreachable by every sibling still holding its name.
    sibling_name: Optional[str] = None
    # The cascade this session belongs to.  Persisted because it was NOT, and
    # nothing restored it: ``Session.cascade_driver_id`` is set at create and
    # re-supplied by the CALLER of ``wake_session`` -- so a stage that unloaded
    # on ORPHAN and was later LOADED (attach / disk-restore rather than an
    # explicit wake) came back with ``None`` and silently left its cascade.
    #
    # Consequences beyond siblings: the cid is what ``_emit_to_session`` routes
    # a cascade's observers on, and what the durability sweep reads.
    #
    # For siblings it made ``sibling_name`` incoherent -- the ADDRESS survived
    # a reload while the MEMBERSHIP did not, so a revived sibling held a name
    # belonging to a cascade it was no longer in.
    cascade_driver_id: Optional[str] = None
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
    # Carried on the INDEX so a COLD sibling stays visible.  Sessions unload on
    # ORPHAN, so one resting on disk still owns its address; a roster or a
    # uniqueness check reading only the in-memory table cannot see it, and the
    # name can be handed to a second claimant.
    cascade_driver_id: Optional[str] = None
    sibling_name: Optional[str] = None
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
