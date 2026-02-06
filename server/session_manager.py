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

import json
import logging
import os
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
from .session_logging import set_logging_context, clear_logging_context, get_session_handler
from .events import (
    Event,
    EventType,
    SystemMessageEvent,
    ErrorEvent,
    SessionInfoEvent,
    SessionDescriptionUpdatedEvent,
    ContextUpdatedEvent,
    AgentCreatedEvent,
    InstructionBudgetEvent,
    InterruptedTurnRecoveredEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    TurnCompletedEvent,
    AgentStatusChangedEvent,
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
    workspace_path: Optional[str] = None


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
    workspace_path: Optional[str] = None  # Client's working directory
    user_inputs: List[str] = field(default_factory=list)  # Command history for prompt restoration
    interrupted_turn: Optional[Dict[str, Any]] = None  # Turn interruption state for recovery


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
        storage_path: Optional[str] = None,
    ):
        """Initialize the session manager.

        Args:
            storage_path: Override for session storage path.
        """

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
        # Use RLock (reentrant) because initialize() may emit events during session load
        self._lock = threading.RLock()

        # Client to session mapping
        self._client_to_session: Dict[str, str] = {}

        # Per-client configuration (terminal_width, etc.)
        self._client_config: Dict[str, Dict[str, Any]] = {}

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
        logger.debug(f"_emit_to_client: {client_id} <- {type(event).__name__}")
        if self._event_callback:
            logger.debug(f"  calling event_callback")
            self._event_callback(client_id, event)
        else:
            logger.warning(f"  NO event_callback set!")

    def _emit_to_session(self, session_id: str, event: Event) -> None:
        """Emit an event to all clients attached to a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # Handle session description updates - update in-memory Session
                if isinstance(event, SessionDescriptionUpdatedEvent):
                    if event.session_id == session_id:
                        session.description = event.description
                        session.is_dirty = True
                        logger.debug(f"Updated session {session_id} description: {event.description}")

                # Handle turn tracking for interrupted tool recovery
                self._handle_turn_tracking_event(session, event)

                for client_id in session.attached_clients:
                    self._emit_to_client(client_id, event)

    def _apply_client_config(self, client_id: str, event: 'ClientConfigRequest') -> None:
        """Apply client configuration settings.

        Updates environment and plugin settings based on client's config.
        This allows clients to use their own .env settings (like JAATO_TRACE_LOG)
        even when connecting to a shared server.

        Args:
            client_id: The requesting client.
            event: The client config event with settings.
        """
        import os

        # Apply trace log paths if provided
        if event.trace_log_path:
            os.environ['JAATO_TRACE_LOG'] = event.trace_log_path
            logger.info(f"Client {client_id} set JAATO_TRACE_LOG={event.trace_log_path}")

        if event.provider_trace_log:
            os.environ['PROVIDER_TRACE_LOG'] = event.provider_trace_log
            logger.info(f"Client {client_id} set PROVIDER_TRACE_LOG={event.provider_trace_log}")

        # Initialize client config dict if needed
        if client_id not in self._client_config:
            self._client_config[client_id] = {}

        # Store and apply terminal width
        if event.terminal_width:
            self._client_config[client_id]['terminal_width'] = event.terminal_width
            logger.info(f"Client {client_id} set terminal_width={event.terminal_width}")

        # Store and apply working directory
        if event.working_dir:
            self._client_config[client_id]['working_dir'] = event.working_dir
            logger.info(f"Client {client_id} set working_dir={event.working_dir}")

        # Store client's env_file path for session creation
        if event.env_file:
            self._client_config[client_id]['env_file'] = event.env_file
            logger.info(f"Client {client_id} set env_file={event.env_file}")

        # Apply to current session if client is attached to one
        session_id = self._client_to_session.get(client_id)
        if session_id:
            session = self._sessions.get(session_id)
            if session and session.server:
                if event.terminal_width:
                    session.server.terminal_width = event.terminal_width
                if event.working_dir:
                    session.server.workspace_path = event.working_dir
                    session.workspace_path = event.working_dir

    def _apply_client_config_to_server(self, client_id: str, server: 'JaatoServer') -> None:
        """Apply stored client configuration to a server.

        Called when a client creates or attaches to a session.

        Args:
            client_id: The client whose config to apply.
            server: The server to configure.
        """
        config = self._client_config.get(client_id, {})
        if 'terminal_width' in config:
            server.terminal_width = config['terminal_width']
            logger.debug(f"Applied terminal_width={config['terminal_width']} to server for client {client_id}")
        if 'working_dir' in config:
            server.workspace_path = config['working_dir']
            logger.debug(f"Applied workspace_path={config['working_dir']} to server for client {client_id}")

    def _handle_turn_tracking_event(self, session: Session, event: Event) -> None:
        """Handle events for turn tracking (interrupted tool recovery).

        Tracks tool execution state so that if the server crashes during tool
        execution, we can recover by injecting synthetic error results.

        Args:
            session: The session being tracked.
            event: The event to process.
        """
        # Track when agent becomes active (turn starts)
        if isinstance(event, AgentStatusChangedEvent):
            if event.status == "active" and event.agent_id == "main":
                # Main agent starting a turn - initialize tracking
                # Note: We don't have user_prompt here, but we can still track tool calls
                if not session.interrupted_turn:
                    session.interrupted_turn = {
                        "agent_id": event.agent_id,
                        "pending_tool_calls": [],
                        "user_prompt": "",  # Not available at this point
                        "started_at": datetime.utcnow().isoformat(),
                    }
                    session.is_dirty = True
                    logger.debug(f"Started turn tracking for session {session.session_id}")
            elif event.status == "done":
                # Agent finished - clear tracking
                if session.interrupted_turn:
                    session.interrupted_turn = None
                    session.is_dirty = True
                    logger.debug(f"Cleared turn tracking for session {session.session_id} (agent done)")

        # Track tool calls as they start
        elif isinstance(event, ToolCallStartEvent):
            if session.interrupted_turn and event.agent_id == session.interrupted_turn.get("agent_id"):
                # Add this tool call to pending list
                pending = session.interrupted_turn.get("pending_tool_calls", [])
                pending.append({
                    "id": event.call_id or "",
                    "name": event.tool_name,
                    "args": event.tool_args,
                })
                session.interrupted_turn["pending_tool_calls"] = pending
                session.is_dirty = True
                # Incremental save to persist pending tool calls before execution
                self._save_session(session)
                logger.debug(
                    f"Updated pending tool calls for session {session.session_id}: "
                    f"{len(pending)} call(s), saving incrementally"
                )

        # Remove completed tool calls from pending list
        elif isinstance(event, ToolCallEndEvent):
            if session.interrupted_turn and event.agent_id == session.interrupted_turn.get("agent_id"):
                pending = session.interrupted_turn.get("pending_tool_calls", [])
                # Remove the completed tool call by matching call_id
                original_count = len(pending)
                pending = [p for p in pending if p.get("id") != event.call_id]
                if len(pending) < original_count:
                    session.interrupted_turn["pending_tool_calls"] = pending
                    session.is_dirty = True
                    logger.debug(
                        f"Tool {event.tool_name} completed, {len(pending)} pending call(s) remain "
                        f"for session {session.session_id}"
                    )

        # Clear tracking when turn completes
        elif isinstance(event, TurnCompletedEvent):
            if session.interrupted_turn:
                session.interrupted_turn = None
                session.is_dirty = True
                logger.debug(f"Cleared turn tracking for session {session.session_id} (turn completed)")

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    def create_session(
        self,
        client_id: str,
        session_name: Optional[str] = None,
        workspace_path: Optional[str] = None,
    ) -> str:
        """Create a new session and attach the client.

        Args:
            client_id: The requesting client.
            session_name: Optional name (auto-generated if not provided).
            workspace_path: Client's working directory for file operations.

        Returns:
            The session ID (empty string on failure).
        """
        # Generate session ID (matches Session Plugin format)
        timestamp = datetime.now()
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

        # Get env_file from client config or derive from workspace path
        # Sessions are workspace-bound: the workspace determines the .env file,
        # which in turn determines the provider.
        client_config = self._client_config.get(client_id, {})
        session_env_file = client_config.get('env_file')
        if not session_env_file and workspace_path:
            # Default to workspace/.env
            import os
            workspace_env = os.path.join(workspace_path, '.env')
            if os.path.exists(workspace_env):
                session_env_file = workspace_env

        logger.info(f"Creating session for client {client_id}: env_file={session_env_file}")
        logger.info(f"  Client config: {client_config}")

        # Create JaatoServer for this session
        # Provider is determined by env_file, not passed explicitly
        server = JaatoServer(
            env_file=session_env_file,
            provider=None,  # Let env_file determine provider
            # During init, emit directly to requesting client (not yet attached to session)
            on_event=lambda e: self._emit_to_client(client_id, e),
            workspace_path=workspace_path,
            session_id=session_id,
        )

        # Initialize the server (events go directly to requesting client)
        if not server.initialize():
            self._emit_to_client(client_id, ErrorEvent(
                error="Failed to initialize session",
                error_type="SessionError",
            ))
            return ""

        logger.info(f"Server initialized successfully for session {session_id}")

        # Switch to session-based event emission now that init is complete
        server.set_event_callback(lambda e: self._emit_to_session(session_id, e))

        # Configure TODO plugin with session-scoped storage
        session_dir = pathlib.Path(self._session_config.storage_path) / session_id
        self._configure_todo_storage(server, session_dir)

        # Apply client-specific config (e.g., terminal_width)
        self._apply_client_config_to_server(client_id, server)

        # Create session object
        session = Session(
            session_id=session_id,
            name=name,
            server=server,
            created_at=timestamp.isoformat(),
            description=None,
            is_dirty=True,  # New session needs saving
            workspace_path=workspace_path,
        )

        # Register callback for when auth completes (if it was pending)
        def on_auth_complete():
            self._emit_to_session(session_id, self._build_session_info_event(session))
            self._emit_to_session(session_id, SystemMessageEvent(
                message=f"Session created: {name} ({session_id})",
                style="info",
            ))
        server.set_auth_complete_callback(on_auth_complete)

        with self._lock:
            self._sessions[session_id] = session
            session.attached_clients.add(client_id)
            self._client_to_session[client_id] = session_id

        # Save initial state to disk
        self._save_session(session)

        logger.info(f"Session created: {session_id} ({name})")

        # Note: We don't call emit_current_state() here because the client
        # already received all events during initialize() via direct emission.

        # Send complete SessionInfoEvent with state snapshot (unless auth pending)
        if not server.auth_pending:
            self._emit_to_client(client_id, self._build_session_info_event(session))

            self._emit_to_client(client_id, SystemMessageEvent(
                message=f"Session created: {name} ({session_id})",
                style="info",
            ))

        return session_id

    def get_session_workspace(self, session_id: str) -> Optional[str]:
        """Get the workspace path of a session.

        Args:
            session_id: The session ID.

        Returns:
            The session's workspace path, or None if session not found.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.workspace_path
        return None

    def check_workspace_mismatch(
        self,
        session_id: str,
        client_workspace: Optional[str],
    ) -> Optional[tuple]:
        """Check if there's a workspace mismatch between client and session.

        Args:
            session_id: The session to check.
            client_workspace: The client's workspace path.

        Returns:
            Tuple of (session_workspace, client_workspace) if there's a mismatch,
            None if no mismatch or session not found.
        """
        session_workspace: Optional[str] = None

        with self._lock:
            # First check in-memory sessions
            session = self._sessions.get(session_id)
            if session:
                session_workspace = session.workspace_path
            else:
                # Check persisted sessions on disk
                persisted = self._get_persisted_sessions()
                for s in persisted:
                    if s.session_id == session_id:
                        session_workspace = s.workspace_path
                        break

        if not session_workspace or not client_workspace:
            # No mismatch if either is not set
            return None

        # Use helper method to compare workspaces
        if not self._workspaces_match(session_workspace, client_workspace):
            return (session_workspace, client_workspace)

        return None

    def attach_session(
        self,
        client_id: str,
        session_id: str,
        workspace_path: Optional[str] = None,
    ) -> bool:
        """Attach a client to an existing session.

        If the session is not in memory, attempts to load from disk.

        Args:
            client_id: The requesting client.
            session_id: The session to attach to.
            workspace_path: Client's working directory for file operations.

        Returns:
            True if attached successfully.
        """
        # Track if session was already in memory (client missed init events)
        session_was_in_memory = False

        with self._lock:
            # Check if session is in memory
            session = self._sessions.get(session_id)
            session_was_in_memory = session is not None

            if not session:
                # Try to load from disk (pass client_id for init progress events)
                logger.debug(f"attach_session: session {session_id} not in memory, loading from disk...")
                try:
                    session = self._load_session(session_id, client_id=client_id)
                    logger.debug(f"attach_session: _load_session returned {session is not None}")
                except Exception as e:
                    logger.error(f"attach_session: _load_session raised: {type(e).__name__}: {e}")
                    import traceback
                    logger.error(f"attach_session: traceback:\n{traceback.format_exc()}")
                    session = None
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

            # Only set workspace if session doesn't have one yet.
            # If session already has a workspace, it keeps it - clients are warned
            # about workspace mismatches before attach via check_workspace_mismatch().
            if workspace_path and not session.workspace_path:
                session.workspace_path = workspace_path
                session.server.workspace_path = workspace_path

        logger.info(f"Client {client_id} attached to session {session_id}")

        # Apply client-specific config (e.g., terminal_width)
        self._apply_client_config_to_server(client_id, session.server)

        # Only emit current state if session was already in memory.
        # If we just loaded it from disk, the client received all events during init.
        if session_was_in_memory:
            session.server.emit_current_state(
                lambda e: self._emit_to_client(client_id, e),
                skip_session_info=True
            )
        else:
            # Session was loaded from disk - clear any stale pending requests
            # the client might have from before the session was saved/restored
            session.server.emit_current_state(
                lambda e: self._emit_to_client(client_id, e),
                skip_session_info=True,
                clear_stale_pending_requests=True
            )

        # Send complete SessionInfoEvent with state snapshot
        self._emit_to_client(client_id, self._build_session_info_event(session))

        # Build attach message with description if available
        desc_part = f" - {session.description}" if session.description else ""
        self._emit_to_client(client_id, SystemMessageEvent(
            message=f"Attached to session: {session_id}{desc_part}",
            style="info",
        ))

        return True

    def _load_session(
        self,
        session_id: str,
        client_id: Optional[str] = None
    ) -> Optional[Session]:
        """Load a session from disk.

        Args:
            session_id: The session ID to load.
            client_id: Optional client ID to receive init progress events.

        Returns:
            The loaded Session, or None if not found.
        """
        logger.debug(f"_load_session: attempting to load {session_id}")
        try:
            state = self._session_plugin.load(session_id)
            logger.debug(f"_load_session: loaded state for {session_id}")
        except FileNotFoundError:
            logger.debug(f"_load_session: session {session_id} not found on disk")
            return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

        # Create JaatoServer and restore state
        logger.debug(f"_load_session: creating JaatoServer for {session_id}...")

        # During init, emit directly to requesting client if provided
        if client_id:
            init_callback = lambda e: self._emit_to_client(client_id, e)
        else:
            init_callback = lambda e: self._emit_to_session(session_id, e)

        # Determine which env_file to use for this session:
        # 1. If client_id is provided, use client's env_file from their config
        # 2. If session has workspace_path, try workspace/.env
        # Sessions are workspace-bound: the workspace determines the .env file,
        # which in turn determines the provider.
        session_env_file = None
        if client_id:
            client_config = self._client_config.get(client_id, {})
            if client_config.get('env_file'):
                session_env_file = client_config['env_file']
                logger.debug(f"_load_session: using client's env_file: {session_env_file}")
        if not session_env_file and state.workspace_path:
            import os
            workspace_env = os.path.join(state.workspace_path, '.env')
            if os.path.exists(workspace_env):
                session_env_file = workspace_env
                logger.debug(f"_load_session: using workspace env_file: {session_env_file}")

        server = JaatoServer(
            env_file=session_env_file,
            provider=None,  # Let env_file determine provider
            on_event=init_callback,
            session_id=session_id,
        )
        logger.debug(f"_load_session: JaatoServer created, calling initialize()...")

        try:
            init_result = server.initialize()
            logger.debug(f"_load_session: initialize() returned {init_result}")
            if not init_result:
                logger.error(f"Failed to initialize server for session {session_id}")
                return None
        except Exception as e:
            logger.error(f"_load_session: initialize() raised exception: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"_load_session: traceback:\n{traceback.format_exc()}")
            return None

        logger.debug(f"_load_session: server initialized for {session_id}")

        # Switch to session-based event emission now that init is complete
        server.set_event_callback(lambda e: self._emit_to_session(session_id, e))

        # Configure TODO plugin with session-scoped storage
        session_dir = pathlib.Path(self._session_config.storage_path) / session_id
        self._configure_todo_storage(server, session_dir)

        # Restore history to the server's JaatoClient
        if state.history and server._jaato:
            server._jaato.reset_session(state.history)
            logger.debug(f"Restored {len(state.history)} messages for session {session_id}")

            # Restore turn accounting (reset_session clears it, so we restore after)
            if state.turn_accounting:
                jaato_session = server._jaato.get_session()
                jaato_session._turn_accounting = list(state.turn_accounting)
                logger.debug(f"Restored {len(state.turn_accounting)} turn accounting entries for session {session_id}")

                # Update server's agent state and emit context update
                if "main" in server._agents:
                    server._agents["main"].turn_accounting = list(state.turn_accounting)
                    usage = server._jaato.get_context_usage()
                    server._agents["main"].context_usage = {
                        'total_tokens': usage.get('total_tokens', 0),
                        'prompt_tokens': usage.get('prompt_tokens', 0),
                        'output_tokens': usage.get('output_tokens', 0),
                        'percent_used': usage.get('percent_used', 0.0),
                    }
                    # Emit context update so clients show correct usage
                    server.emit(ContextUpdatedEvent(
                        agent_id="main",
                        total_tokens=usage.get('total_tokens', 0),
                        prompt_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('output_tokens', 0),
                        context_limit=usage.get('context_limit', 0),
                        percent_used=usage.get('percent_used', 0.0),
                        tokens_remaining=usage.get('tokens_remaining', 0),
                        turns=usage.get('turns', 0),
                    ))
                    logger.debug(f"Emitted ContextUpdatedEvent: {usage.get('percent_used', 0.0):.1f}% used")

        # Restore conversation budget if present (other budget sources are
        # automatically populated during session recreation)
        if state.budget_state and server._jaato:
            jaato_session = server._jaato.get_session()
            if jaato_session and jaato_session.instruction_budget:
                jaato_session.instruction_budget.restore_conversation_from_snapshot(state.budget_state)
                logger.debug(f"Restored conversation budget for session {session_id}")
                # Emit budget event so clients show correct budget
                server.emit(InstructionBudgetEvent(
                    agent_id=jaato_session.agent_id,
                    budget_snapshot=jaato_session.instruction_budget.snapshot(),
                ))

        # Restore subagent state if present in metadata
        if state.metadata.get('subagents') and server._jaato:
            self._restore_subagent_states(
                session_id,
                state.metadata['subagents'],
                server
            )

        # Restore TODO plugin state (agent-plan mapping, blocked steps)
        self._load_todo_state(server, session_dir)

        # Check for and recover from interrupted turn
        recovered_count = 0
        if state.interrupted_turn:
            recovered_count = self._recover_interrupted_turn(
                session_id,
                state.interrupted_turn,
                server
            )
            if recovered_count > 0:
                logger.info(f"Recovered {recovered_count} interrupted tool calls for session {session_id}")

        session = Session(
            session_id=session_id,
            name=state.description or f"Session {session_id}",
            server=server,
            created_at=state.created_at.isoformat(),
            last_activity=state.updated_at.isoformat(),
            description=state.description,
            is_dirty=recovered_count > 0,  # Mark dirty if recovery happened
            workspace_path=state.workspace_path,
            user_inputs=state.user_inputs or [],  # Command history for prompt restoration
        )

        logger.info(f"Loaded session from disk: {session_id}")
        return session

    def _restore_subagent_states(
        self,
        session_id: str,
        subagent_registry: Dict[str, Any],
        server: JaatoServer
    ) -> int:
        """Restore subagent states from persisted data.

        Args:
            session_id: The parent session ID.
            subagent_registry: Registry dict from state.metadata["subagents"].
            server: The JaatoServer to restore subagents into.

        Returns:
            Number of subagents successfully restored.
        """
        if not server.registry:
            logger.warning("Cannot restore subagents: no registry available")
            return 0

        subagent_plugin = server.registry.get_plugin("subagent")
        if not subagent_plugin or not hasattr(subagent_plugin, 'restore_persistence_state'):
            logger.warning("Cannot restore subagents: subagent plugin not available")
            return 0

        # Load per-agent state files
        subagents_dir = pathlib.Path(
            self._session_config.storage_path
        ) / session_id / "subagents"

        agent_states: Dict[str, Dict[str, Any]] = {}
        if subagents_dir.exists():
            for agent_file in subagents_dir.glob("*.json"):
                agent_id = agent_file.stem
                try:
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        agent_states[agent_id] = json.load(f)
                    logger.debug(f"Loaded subagent state file: {agent_file}")
                except Exception as e:
                    logger.error(f"Failed to load subagent state {agent_file}: {e}")

        # Get runtime from server's jaato client
        runtime = server._jaato.get_runtime() if server._jaato else None
        if not runtime:
            logger.warning("Cannot restore subagents: no runtime available")
            return 0

        # Restore subagents
        restored = subagent_plugin.restore_persistence_state(
            subagent_registry,
            agent_states,
            runtime
        )

        # Emit AgentCreatedEvent for each restored subagent so clients see them
        for agent_id, info in subagent_plugin._active_sessions.items():
            profile = info.get('profile')
            created_at = info.get('created_at')
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()

            server.emit(AgentCreatedEvent(
                agent_id=agent_id,
                agent_name=profile.name if profile else agent_id,
                agent_type="subagent",
                profile_name=profile.name if profile else "",
                parent_agent_id="main",
                created_at=created_at,
            ))

            # Emit context update for restored subagent
            session = info.get('session')
            if session:
                usage = session.get_context_usage()
                context_limit = session.get_context_limit()
                server.emit(ContextUpdatedEvent(
                    agent_id=agent_id,
                    total_tokens=usage.get('total_tokens', 0),
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('output_tokens', 0),
                    context_limit=context_limit,
                    percent_used=usage.get('percent_used', 0.0),
                    tokens_remaining=max(0, context_limit - usage.get('total_tokens', 0)),
                    turns=usage.get('turns', 0),
                ))

        logger.info(f"Restored {restored} subagents for session {session_id}")
        return restored

    def _recover_interrupted_turn(
        self,
        session_id: str,
        interrupted_state: Dict[str, Any],
        server: JaatoServer
    ) -> int:
        """Recover from an interrupted turn by injecting synthetic tool results.

        When a session is loaded with pending tool calls (from an interrupted turn),
        this method injects synthetic error results for each pending call. This
        completes the function_call/response pairs so the model sees what happened
        and can decide whether to retry.

        Args:
            session_id: The session ID being recovered.
            interrupted_state: The interrupted_turn dict from SessionState containing:
                - agent_id: Which agent was executing
                - pending_tool_calls: List of {id, name, args}
                - user_prompt: Original user prompt
                - started_at: When the turn started
            server: The JaatoServer to inject results into.

        Returns:
            Number of pending tool calls recovered.
        """
        from shared.plugins.model_provider.types import Part, Message, Role, ToolResult

        pending_calls = interrupted_state.get('pending_tool_calls', [])
        if not pending_calls:
            logger.debug(f"No pending tool calls to recover for session {session_id}")
            return 0

        agent_id = interrupted_state.get('agent_id', 'main')

        # Build synthetic tool results for each pending call
        synthetic_parts = []
        for call in pending_calls:
            call_id = call.get('id', '')
            tool_name = call.get('name', 'unknown')

            synthetic_result = ToolResult(
                call_id=call_id,
                name=tool_name,
                result={
                    "error": "tool_interrupted",
                    "reason": "server_restart",
                    "message": f"Tool '{tool_name}' was interrupted by server restart. "
                               "You may retry this operation if appropriate."
                },
                is_error=True
            )
            synthetic_parts.append(Part.from_function_response(synthetic_result))

        # Create a TOOL message with all synthetic results
        synthetic_message = Message(role=Role.TOOL, parts=synthetic_parts)

        # Inject into history based on which agent was executing
        if agent_id == 'main':
            if server._jaato:
                jaato_session = server._jaato.get_session()
                if jaato_session:
                    # Append the synthetic tool message to history using proper API
                    current_history = jaato_session.get_history()
                    current_history.append(synthetic_message)
                    jaato_session.reset_session(current_history)
                    logger.info(
                        f"Recovered {len(pending_calls)} interrupted tool call(s) "
                        f"for main agent in session {session_id}"
                    )
        else:
            # Subagent recovery - find the subagent session
            if server.registry:
                subagent_plugin = server.registry.get_plugin("subagent")
                if subagent_plugin and hasattr(subagent_plugin, '_active_sessions'):
                    session_info = subagent_plugin._active_sessions.get(agent_id)
                    if session_info:
                        subagent_session = session_info.get('session')
                        if subagent_session:
                            # Append the synthetic tool message to history using proper API
                            current_history = subagent_session.get_history()
                            current_history.append(synthetic_message)
                            subagent_session.reset_session(current_history)
                            logger.info(
                                f"Recovered {len(pending_calls)} interrupted tool call(s) "
                                f"for subagent {agent_id} in session {session_id}"
                            )

        # Emit recovery event so clients know what happened
        server.emit(InterruptedTurnRecoveredEvent(
            session_id=session_id,
            agent_id=agent_id,
            recovered_calls=len(pending_calls),
            action_taken="synthetic_error",
        ))

        # Also emit a system message for user visibility
        tool_names = [call.get('name', 'unknown') for call in pending_calls]
        server.emit(SystemMessageEvent(
            message=f"Recovered from interrupted turn: {len(pending_calls)} tool call(s) "
                    f"({', '.join(tool_names)}) were interrupted by server restart.",
            style="warning",
        ))

        # Signal that the interrupted turn is now complete (agent is done)
        # This tells the client to stop showing the "thinking" spinner
        server.emit(AgentStatusChangedEvent(
            agent_id=agent_id,
            status="done",
        ))

        return len(pending_calls)

    def _save_session(self, session: Session) -> bool:
        """Save a session to disk.

        Args:
            session: The session to save.

        Returns:
            True if saved successfully.
        """
        try:
            # Get history directly from JaatoClient to ensure we capture
            # in-progress turns (the agent state cache is only updated at turn end)
            history = []
            if session.server and session.server._jaato:
                history = session.server._jaato.get_history()
            turn_accounting = []

            if session.server and "main" in session.server._agents:
                turn_accounting = session.server._agents["main"].turn_accounting

            # Get subagent state if subagent plugin is available
            subagent_metadata = {}
            if session.server and session.server.registry:
                subagent_plugin = session.server.registry.get_plugin("subagent")
                if subagent_plugin and hasattr(subagent_plugin, 'get_persistence_state'):
                    subagent_registry = subagent_plugin.get_persistence_state()
                    if subagent_registry.get('agents'):
                        subagent_metadata['subagents'] = subagent_registry

                        # Save per-agent state files
                        self._save_subagent_states(
                            session.session_id,
                            subagent_plugin,
                            subagent_registry.get('agents', [])
                        )

            # Save TODO plugin state
            session_dir = pathlib.Path(self._session_config.storage_path) / session.session_id
            if session.server:
                self._save_todo_state(session.server, session_dir)

            # Get conversation budget for persistence (other budget sources are
            # automatically recreated when the session is restored)
            budget_state = None
            if session.server and session.server._jaato:
                jaato_session = session.server._jaato.get_session()
                if jaato_session and jaato_session.instruction_budget:
                    budget_state = jaato_session.instruction_budget.get_conversation_snapshot()

            # Create SessionState
            state = SessionState(
                session_id=session.session_id,
                history=history,
                created_at=datetime.fromisoformat(session.created_at),
                updated_at=datetime.utcnow(),
                description=session.description or session.name,
                turn_count=len(history) // 2,  # Approximate
                turn_accounting=turn_accounting,
                user_inputs=session.user_inputs,  # Command history for prompt restoration
                model=session.server.model_name if session.server else None,
                workspace_path=session.workspace_path,
                metadata=subagent_metadata,
                budget_state=budget_state,
                interrupted_turn=session.interrupted_turn,  # For recovery on restart
            )

            self._session_plugin.save(state)
            session.is_dirty = False

            logger.debug(f"Saved session: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False

    def _get_todo_plugin(self, server: JaatoServer) -> Optional[Any]:
        """Get the TODO plugin from a server's registry.

        Args:
            server: The JaatoServer instance.

        Returns:
            The TodoPlugin instance, or None if not available.
        """
        if not server or not server.registry:
            return None
        return server.registry.get_plugin("todo")

    def _configure_todo_storage(self, server: JaatoServer, session_dir: pathlib.Path) -> None:
        """Configure TODO plugin with session-scoped file storage.

        Args:
            server: The JaatoServer instance.
            session_dir: The session's storage directory.
        """
        todo_plugin = self._get_todo_plugin(server)
        if not todo_plugin:
            return

        # Resolve to absolute path to avoid issues with CWD changes
        # (e.g., when subagents call os.chdir() in background threads)
        plans_dir = (session_dir / "plans").resolve()
        todo_plugin.initialize({
            "storage_type": "file",
            "storage_path": str(plans_dir),
            "storage_use_directory": True,  # One file per plan
        })
        logger.debug(f"Configured TODO storage at: {plans_dir}")

    def _save_todo_state(self, server: JaatoServer, session_dir: pathlib.Path) -> None:
        """Save TODO plugin state (agent-plan mapping, blocked steps).

        Args:
            server: The JaatoServer instance.
            session_dir: The session's storage directory.
        """
        todo_plugin = self._get_todo_plugin(server)
        if not todo_plugin or not hasattr(todo_plugin, 'get_persistence_state'):
            return

        state = todo_plugin.get_persistence_state()
        if not state.get('agent_plan_ids'):
            # No plans to save
            return

        # Resolve to absolute path to avoid CWD issues
        state_path = (session_dir / "plans" / "_state.json").resolve()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved TODO state: {state_path}")
        except Exception as e:
            logger.error(f"Failed to save TODO state: {e}")

    def _load_todo_state(self, server: JaatoServer, session_dir: pathlib.Path) -> None:
        """Load TODO plugin state from disk.

        Args:
            server: The JaatoServer instance.
            session_dir: The session's storage directory.
        """
        # Resolve to absolute path to avoid CWD issues
        state_path = (session_dir / "plans" / "_state.json").resolve()
        if not state_path.exists():
            return

        todo_plugin = self._get_todo_plugin(server)
        if not todo_plugin or not hasattr(todo_plugin, 'restore_persistence_state'):
            return

        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            todo_plugin.restore_persistence_state(state)
            logger.debug(f"Loaded TODO state: {state_path}")
        except Exception as e:
            logger.error(f"Failed to load TODO state: {e}")

    def _save_subagent_states(
        self,
        session_id: str,
        subagent_plugin: Any,
        agents: List[Dict[str, Any]]
    ) -> None:
        """Save per-agent state files for subagents.

        Args:
            session_id: The parent session ID.
            subagent_plugin: The SubagentPlugin instance.
            agents: List of agent info dicts from the registry.
        """
        # Create subagents directory
        subagents_dir = pathlib.Path(
            self._session_config.storage_path
        ) / session_id / "subagents"
        subagents_dir.mkdir(parents=True, exist_ok=True)

        for agent_info in agents:
            agent_id = agent_info.get('agent_id')
            if not agent_id:
                continue

            # Get full state from plugin
            full_state = subagent_plugin.get_agent_full_state(agent_id)
            if not full_state:
                continue

            # Write to file
            agent_file = subagents_dir / f"{agent_id}.json"
            try:
                with open(agent_file, 'w', encoding='utf-8') as f:
                    json.dump(full_state, f, indent=2)
                logger.debug(f"Saved subagent state: {agent_file}")
            except Exception as e:
                logger.error(f"Failed to save subagent {agent_id}: {e}")

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

        # Close session-specific log handlers
        handler = get_session_handler()
        if handler:
            handler.close_session(session_id)

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

    def _normalize_workspace(self, path: Optional[str]) -> Optional[str]:
        """Normalize a workspace path for comparison.

        Args:
            path: The path to normalize.

        Returns:
            Normalized absolute path, or None if path is None.
        """
        if not path:
            return None
        import os
        return os.path.normpath(os.path.abspath(path))

    def _workspaces_match(
        self,
        path1: Optional[str],
        path2: Optional[str],
    ) -> bool:
        """Check if two workspace paths match.

        Args:
            path1: First path.
            path2: Second path.

        Returns:
            True if both paths are set and point to the same directory.
        """
        norm1 = self._normalize_workspace(path1)
        norm2 = self._normalize_workspace(path2)
        if not norm1 or not norm2:
            return False
        return norm1 == norm2

    def get_or_create_default(
        self,
        client_id: str,
        workspace_path: Optional[str] = None,
    ) -> str:
        """Get the default session for a workspace, or create a new one.

        Finds the most recently used session for the given workspace.
        Creates a new session if no matching session exists.

        Args:
            client_id: The requesting client.
            workspace_path: Client's working directory for file operations.

        Returns:
            The session ID.
        """
        logger.debug(f"get_or_create_default called for client {client_id}, workspace={workspace_path}")

        # Check in-memory sessions first - find one matching the workspace
        with self._lock:
            if self._sessions and workspace_path:
                # Find sessions matching this workspace
                matching_sessions = [
                    s for s in self._sessions.values()
                    if self._workspaces_match(s.workspace_path, workspace_path)
                ]
                if matching_sessions:
                    # Use the first matching session (they're all for the same workspace)
                    session = matching_sessions[0]
                    logger.debug(f"  found in-memory session for workspace: {session.session_id}")
                    session.attached_clients.add(client_id)
                    self._client_to_session[client_id] = session.session_id
                    # Emit current agent state to the newly attached client
                    session.server.emit_current_state(
                        lambda e: self._emit_to_client(client_id, e),
                        skip_session_info=True
                    )
                    # Send complete SessionInfoEvent with state snapshot
                    self._emit_to_client(client_id, self._build_session_info_event(session))
                    return session.session_id

        # Check persisted sessions (already sorted by updated_at descending)
        logger.debug(f"  checking persisted sessions...")
        persisted = self._get_persisted_sessions()
        logger.debug(f"  found {len(persisted)} persisted session(s)")

        if persisted and workspace_path:
            # Find sessions matching this workspace
            matching_persisted = [
                s for s in persisted
                if self._workspaces_match(s.workspace_path, workspace_path)
            ]
            logger.debug(f"  found {len(matching_persisted)} session(s) for workspace")

            if matching_persisted:
                # Use the most recent one for this workspace
                most_recent = matching_persisted[0]
                logger.debug(f"  attaching to workspace session: {most_recent.session_id}")
                if self.attach_session(client_id, most_recent.session_id, workspace_path):
                    return most_recent.session_id

        # No matching sessions exist - create a new one for this workspace
        logger.debug(f"  creating new session for workspace...")
        return self.create_session(client_id, workspace_path=workspace_path)

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
                workspace_path=info.workspace_path,
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
                    workspace_path=session.workspace_path,
                )

        # Sort by last activity
        sessions = list(result.values())
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions

    def _build_session_info_event(self, session: "Session") -> SessionInfoEvent:
        """Build a complete SessionInfoEvent with state snapshot.

        Includes current session info plus:
        - sessions: All available sessions for completion/display
        - tools: All tools with enabled status
        - models: Available model names
        """
        # Get sessions list
        sessions_data = [{
            "id": s.session_id,
            "name": s.name or "",
            "description": s.description or "",
            "model_provider": s.model_provider or "",
            "model_name": s.model_name or "",
            "is_loaded": s.is_loaded,
            "client_count": s.client_count,
            "turn_count": s.turn_count,
            "workspace_path": s.workspace_path or "",
        } for s in self.list_sessions()]

        # Get tools list from the session's server
        tools_data = []
        if session.server:
            tools_data = session.server.get_tool_status()

        # Models list is lazy-loaded on demand to avoid API calls during init
        # Client fetches models when user requests completions
        models_data = []

        # Get memory metadata from the session's server for completion cache
        memories_data = []
        if session.server:
            mem_plugin = session.server._find_plugin_for_command("memory")
            if mem_plugin and hasattr(mem_plugin, 'get_memory_metadata'):
                memories_data = mem_plugin.get_memory_metadata()

        return SessionInfoEvent(
            session_id=session.session_id,
            session_name=session.name,
            model_provider=session.server.model_provider if session.server else "",
            model_name=session.server.model_name if session.server else "",
            sessions=sessions_data,
            tools=tools_data,
            models=models_data,
            user_inputs=session.user_inputs,  # Command history for prompt restoration
            memories=memories_data,
        )

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
    # Turn Tracking for Recovery
    # =========================================================================

    def start_turn_tracking(
        self,
        session_id: str,
        user_prompt: str,
        agent_id: str = "main"
    ) -> None:
        """Mark a turn as in-progress for recovery purposes.

        Call this when a turn starts (user sends message) to enable recovery
        if the server crashes during tool execution.

        Args:
            session_id: The session ID.
            user_prompt: The user's original prompt.
            agent_id: Which agent is executing ("main" or subagent ID).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.interrupted_turn = {
                    "agent_id": agent_id,
                    "pending_tool_calls": [],
                    "user_prompt": user_prompt,
                    "started_at": datetime.utcnow().isoformat(),
                }
                session.is_dirty = True
                logger.debug(f"Started turn tracking for session {session_id}, agent {agent_id}")

    def update_pending_tool_calls(
        self,
        session_id: str,
        function_calls: List[Dict[str, Any]]
    ) -> None:
        """Update pending tool calls after model response.

        Call this after the model returns function calls, before tool execution.
        This triggers an incremental save so the pending calls are persisted.

        Args:
            session_id: The session ID.
            function_calls: List of {id, name, args} dicts from model response.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.interrupted_turn:
                session.interrupted_turn["pending_tool_calls"] = function_calls
                session.is_dirty = True
                # Incremental save to persist pending tool calls before execution
                self._save_session(session)
                logger.debug(
                    f"Updated pending tool calls for session {session_id}: "
                    f"{len(function_calls)} call(s)"
                )

    def clear_turn_tracking(self, session_id: str) -> None:
        """Clear turn tracking on successful completion.

        Call this when a turn completes successfully (no more function calls).

        Args:
            session_id: The session ID.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.interrupted_turn = None
                session.is_dirty = True
                logger.debug(f"Cleared turn tracking for session {session_id}")

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
        from .events import ClientConfigRequest

        # Handle client config before session lookup (doesn't require session)
        if isinstance(event, ClientConfigRequest):
            self._apply_client_config(client_id, event)
            return

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

        # Set logging context for session-specific log routing
        workspace_path = session.workspace_path
        session_env = server.get_all_session_env() if server else {}
        set_logging_context(
            session_id=session_id,
            client_id=client_id,
            workspace_path=workspace_path,
            session_env=session_env,
        )

        from .events import (
            SendMessageRequest,
            PermissionResponseRequest,
            ClarificationResponseRequest,
            ReferenceSelectionResponseRequest,
            StopRequest,
            CommandRequest,
            GetInstructionBudgetRequest,
            InstructionBudgetEvent,
        )

        if isinstance(event, SendMessageRequest):
            # Track user input for command history restoration
            if event.text and event.text.strip():
                session.user_inputs.append(event.text)
                session.is_dirty = True

            # Capture context for thread (ContextVars don't propagate to threads)
            ctx_session_id = session_id
            ctx_client_id = client_id
            ctx_workspace = workspace_path
            ctx_session_env = session_env

            # Run in thread to not block
            def run_message():
                # Set logging context in thread
                set_logging_context(
                    session_id=ctx_session_id,
                    client_id=ctx_client_id,
                    workspace_path=ctx_workspace,
                    session_env=ctx_session_env,
                )
                try:
                    server.send_message(
                        event.text,
                        event.attachments if event.attachments else None
                    )
                    # Auto-save after turn
                    self._save_session(session)
                finally:
                    clear_logging_context()

            threading.Thread(target=run_message, daemon=True).start()

        elif isinstance(event, PermissionResponseRequest):
            server.respond_to_permission(
                event.request_id, event.response,
                edited_arguments=event.edited_arguments,
            )

        elif isinstance(event, ClarificationResponseRequest):
            server.respond_to_clarification(event.request_id, event.response)

        elif isinstance(event, ReferenceSelectionResponseRequest):
            server.respond_to_reference_selection(event.request_id, event.response)

        elif isinstance(event, StopRequest):
            server.stop()

        elif isinstance(event, CommandRequest):
            result = server.execute_command(event.command, event.args)
            # Format result properly
            if isinstance(result, dict):
                if "error" in result:
                    # Error result
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message=result["error"],
                        style="error",
                    ))
                elif "result" in result:
                    # Simple result - show the text directly
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message=result["result"],
                        style="info",
                    ))
                else:
                    # Dict result with multiple keys - format each
                    lines = []
                    for key, value in result.items():
                        if not key.startswith('_'):
                            if isinstance(value, list):
                                # Format lists nicely
                                if value:
                                    lines.append(f"{key}:")
                                    for item in value:
                                        # Extract short name for model paths
                                        if isinstance(item, str) and '/' in item:
                                            item = item.split('/')[-1]
                                        lines.append(f"  • {item}")
                                else:
                                    lines.append(f"{key}: (none)")
                            else:
                                lines.append(f"{key}: {value}")
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message="\n".join(lines) if lines else str(result),
                        style="info",
                    ))
            else:
                self._emit_to_client(client_id, SystemMessageEvent(
                    message=str(result),
                    style="info",
                ))

        elif isinstance(event, GetInstructionBudgetRequest):
            # Get instruction budget for the requested agent
            agent_id = event.agent_id or "main"

            if agent_id == "main":
                # Main agent budget from JaatoClient session
                jaato_session = server._jaato.get_session() if server._jaato else None
                if jaato_session and jaato_session.instruction_budget:
                    self._emit_to_client(client_id, InstructionBudgetEvent(
                        agent_id=agent_id,
                        budget_snapshot=jaato_session.instruction_budget.snapshot(),
                    ))
                else:
                    self._emit_to_client(client_id, ErrorEvent(
                        error="No instruction budget available for main agent",
                        error_type="BudgetNotFound",
                    ))
            else:
                # Subagent budget from SubagentPlugin
                subagent_plugin = server.registry.get_plugin("subagent") if server.registry else None
                if subagent_plugin and hasattr(subagent_plugin, '_active_sessions'):
                    session_info = subagent_plugin._active_sessions.get(agent_id)
                    if session_info:
                        subagent_session = session_info.get('session')
                        if subagent_session and hasattr(subagent_session, 'instruction_budget') and subagent_session.instruction_budget:
                            self._emit_to_client(client_id, InstructionBudgetEvent(
                                agent_id=agent_id,
                                budget_snapshot=subagent_session.instruction_budget.snapshot(),
                            ))
                        else:
                            self._emit_to_client(client_id, ErrorEvent(
                                error=f"No instruction budget available for agent {agent_id}",
                                error_type="BudgetNotFound",
                            ))
                    else:
                        self._emit_to_client(client_id, ErrorEvent(
                            error=f"Agent not found: {agent_id}",
                            error_type="AgentNotFound",
                        ))
                else:
                    self._emit_to_client(client_id, ErrorEvent(
                        error=f"Subagent plugin not available",
                        error_type="PluginNotFound",
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

        # Close all session log handlers
        handler = get_session_handler()
        if handler:
            handler.close()

        self._session_plugin.shutdown()
        logger.info("SessionManager shutdown complete")
