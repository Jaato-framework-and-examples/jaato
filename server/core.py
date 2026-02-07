"""JaatoServer - Core logic for multi-client support.

This module extracts the non-UI logic from RichClient into a reusable
server that can be driven by different frontends (TUI, WebSocket, HTTP).

The server emits events for all state changes, allowing clients to
subscribe and render appropriately.
"""

import contextlib
import logging
import os
import re
import sys
import pathlib
import queue
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIMPLE_CLIENT = ROOT / "simple-client"
if str(SIMPLE_CLIENT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_CLIENT))

RICH_CLIENT = ROOT / "rich-client"
if str(RICH_CLIENT) not in sys.path:
    sys.path.insert(0, str(RICH_CLIENT))

from dotenv import load_dotenv

from shared import (
    JaatoClient,
    TokenLedger,
    PluginRegistry,
    PermissionPlugin,
    TodoPlugin,
    active_cert_bundle,
)
from shared.message_queue import SourceType
from shared.plugins.session import create_plugin as create_session_plugin, load_session_config
from shared.plugins.base import parse_command_args, HelpLines
from shared.plugins.gc import load_gc_from_file

# Formatter pipeline for server-side output formatting
from shared.plugins.formatter_pipeline import FormatterRegistry, create_registry

# Reuse input handling from simple-client
from input_handler import InputHandler

# Reuse plan reporter from rich-client (already generic with callbacks)
from plan_reporter import create_live_reporter

# Import events
from .events import (
    Event,
    EventType,
    ConnectedEvent,
    AgentCreatedEvent,
    AgentOutputEvent,
    AgentStatusChangedEvent,
    AgentCompletedEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolOutputEvent,
    PermissionInputModeEvent,
    PermissionResolvedEvent,
    PermissionStatusEvent,
    ClarificationInputModeEvent,
    ClarificationResolvedEvent,
    ReferenceSelectionRequestedEvent,
    ReferenceSelectionResolvedEvent,
    ReferenceSelectionResponseRequest,
    PlanUpdatedEvent,
    PlanClearedEvent,
    ContextUpdatedEvent,
    InstructionBudgetEvent,
    TurnCompletedEvent,
    TurnProgressEvent,
    SystemMessageEvent,
    HelpTextEvent,
    InitProgressEvent,
    ErrorEvent,
    RetryEvent,
    SessionInfoEvent,
    SessionDescriptionUpdatedEvent,
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    StopRequest,
    CommandRequest,
    MidTurnPromptQueuedEvent,
    MidTurnPromptInjectedEvent,
    MidTurnInterruptEvent,
    MemoryListEvent,
    serialize_event,
    deserialize_event,
)


# Type alias for event callback
EventCallback = Callable[[Event], None]


class AgentState:
    """Tracks state for a single agent."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        agent_type: str,
        profile_name: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.profile_name = profile_name
        self.parent_agent_id = parent_agent_id
        self.status = "idle"  # idle, active, done, error
        self.created_at = datetime.utcnow().isoformat()
        self.completed_at: Optional[str] = None
        self.history: List[Any] = []
        self.turn_accounting: List[Dict] = []
        self.context_usage: Dict[str, Any] = {}
        # GC configuration (set when agent is created with GC)
        self.gc_threshold: Optional[float] = None
        self.gc_strategy: Optional[str] = None
        self.gc_target_percent: Optional[float] = None
        self.gc_continuous_mode: bool = False
        # Per-agent formatter pipeline for output formatting
        # Initialized lazily via JaatoServer._get_agent_pipeline()
        self.formatter_pipeline: Optional[Any] = None


class JaatoServer:
    """Core server logic for Jaato - UI-agnostic.

    This class manages:
    - JaatoClient and plugins
    - Agent lifecycle and state
    - Message processing
    - Permission/clarification flows
    - Event emission for clients

    Clients subscribe to events via the `on_event` callback and send
    requests via the public methods.
    """

    def __init__(
        self,
        env_file: str = ".env",
        provider: Optional[str] = None,
        on_event: Optional[EventCallback] = None,
        workspace_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the server.

        Args:
            env_file: Path to .env file.
            provider: Model provider override (e.g., 'google_genai').
            on_event: Callback for emitting events to clients.
            workspace_path: Client's working directory for file operations.
                           If provided, the server will chdir to this path
                           when processing requests.
            session_id: Unique identifier for this session (used in logs).
        """
        self.env_file = env_file
        self._provider = provider
        self._on_event = on_event or (lambda e: None)
        self._on_auth_complete: Optional[Callable[[], None]] = None
        self._workspace_path = workspace_path
        self._session_id = session_id

        # Core components
        self._jaato: Optional[JaatoClient] = None
        self.registry: Optional[PluginRegistry] = None
        self.permission_plugin: Optional[PermissionPlugin] = None
        self.todo_plugin: Optional[TodoPlugin] = None
        self.ledger = TokenLedger()

        # Agent tracking
        self._agents: Dict[str, AgentState] = {}
        self._selected_agent_id: str = "main"

        # Input handling (for file expansion)
        self._input_handler = InputHandler()

        # Track original inputs for session export
        self._original_inputs: List[Dict] = []

        # Queue for permission/clarification responses
        self._channel_input_queue: queue.Queue[str] = queue.Queue()
        self._waiting_for_channel_input: bool = False
        self._pending_permission_request_id: Optional[str] = None
        # Edited arguments from client-side editing (set before "e" is put in queue)
        self._pending_edited_arguments: Optional[Dict[str, Any]] = None
        self._pending_clarification_request_id: Optional[str] = None
        self._pending_reference_selection_request_id: Optional[str] = None

        # Track which agent is currently executing a tool (for permission/clarification routing)
        self._current_tool_agent_id: str = "main"

        # Queue for mid-turn prompts (messages sent while model is running)
        self._mid_turn_prompt_queue: queue.Queue[str] = queue.Queue()

        # Background model thread
        self._model_thread: Optional[threading.Thread] = None
        self._model_running: bool = False

        # Model info
        self._model_provider: str = ""
        self._model_name: str = ""

        # Auth state
        self._auth_pending: bool = False
        self._auth_plugin_command: Optional[str] = None  # Command name for pending auth plugin

        # Terminal width for formatting (default 80)
        self._terminal_width: int = 80

        # Session-specific environment variables (isolated per session)
        # These are loaded from the session's .env file and NOT applied to
        # global os.environ, keeping each session's configuration isolated.
        self._session_env: Dict[str, str] = {}

        # Formatter pipeline for server-side output formatting
        # Initialized in _setup_formatter_pipeline() after registry is available
        # The pipeline handles buffering internally for streaming
        self._formatter_pipeline = None

    # =========================================================================
    # Workspace Management
    # =========================================================================

    @property
    def workspace_path(self) -> Optional[str]:
        """Get the client's workspace path."""
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, path: Optional[str]) -> None:
        """Set the client's workspace path."""
        self._workspace_path = path
        # Propagate to plugins that need workspace awareness
        self._update_plugin_workspace(path)

    @property
    def terminal_width(self) -> int:
        """Get the terminal width for formatting."""
        return self._terminal_width

    @terminal_width.setter
    def terminal_width(self, width: int) -> None:
        """Set the terminal width for formatting.

        This affects enrichment notification formatting to properly
        wrap and align text for the terminal.
        """
        self._terminal_width = width
        # Propagate to JaatoClient if connected
        if self._jaato:
            self._jaato.set_terminal_width(width)
        # Propagate to main formatter pipeline if initialized
        if self._formatter_pipeline:
            self._formatter_pipeline.set_console_width(width)
        # Propagate to all agent-specific pipelines
        for agent in self._agents.values():
            if agent.formatter_pipeline:
                agent.formatter_pipeline.set_console_width(width)

    @property
    def auth_pending(self) -> bool:
        """Check if authentication is pending."""
        return self._auth_pending

    @contextlib.contextmanager
    def _in_workspace(self):
        """Context manager to temporarily change to the workspace directory.

        Also sets JAATO_WORKSPACE_ROOT env var for thread-safe token resolution
        in subagents (since os.chdir is process-wide and racy).
        """
        if not self._workspace_path:
            yield
            return

        original_cwd = os.getcwd()
        original_workspace_env = os.environ.get("JAATO_WORKSPACE_ROOT")
        try:
            os.chdir(self._workspace_path)
            os.environ["JAATO_WORKSPACE_ROOT"] = self._workspace_path
            logger.debug(f"Changed to workspace: {self._workspace_path}")
            yield
        finally:
            os.chdir(original_cwd)
            if original_workspace_env is not None:
                os.environ["JAATO_WORKSPACE_ROOT"] = original_workspace_env
            elif "JAATO_WORKSPACE_ROOT" in os.environ:
                del os.environ["JAATO_WORKSPACE_ROOT"]
            logger.debug(f"Restored to: {original_cwd}")

    @contextlib.contextmanager
    def _with_session_env(self):
        """Context manager to apply session environment variables.

        Applies session-specific environment variables to os.environ for the
        duration of the context. This is necessary for components that read
        from os.environ (like provider SDKs, telemetry, etc.).

        Since all env-dependent operations happen within session contexts,
        we simply apply the session's env vars without restoration - the next
        session will apply its own env vars when needed.

        LIMITATION: This only overlays vars present in the session's .env file.
        Variables set by the server at startup or other sessions are NOT cleared.
        If a user comments out a variable (e.g., REQUESTS_CA_BUNDLE), the stale
        value persists. Workaround: set vars to empty string instead of commenting.
        See docs/architecture.md "Environment Variable Isolation" for details.
        """
        # Apply session env vars (no restoration needed - next session will
        # apply its own env vars anyway)
        for key, value in self._session_env.items():
            if value is not None:
                os.environ[key] = value
        yield

    def get_session_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get an environment variable, checking session env first.

        Args:
            key: Environment variable name.
            default: Default value if not found.

        Returns:
            Value from session env, falling back to global os.environ,
            then to the default value.
        """
        return self._session_env.get(key) or os.environ.get(key) or default

    def get_all_session_env(self) -> Dict[str, str]:
        """Get all session-specific environment variables.

        Returns:
            Copy of session environment dictionary.
        """
        return dict(self._session_env)

    def _update_plugin_workspace(self, path: Optional[str]) -> None:
        """Update workspace-aware plugins with the new workspace path.

        This notifies plugins like LSP, MCP, file_edit, and CLI that need
        to find config files relative to the client's working directory.

        Uses registry.set_workspace_path() which broadcasts to all plugins
        implementing set_workspace_path().
        """
        if not path or not hasattr(self, 'registry') or not self.registry:
            return

        self.registry.set_workspace_path(path)
        logger.debug(f"Broadcast workspace_path to plugins: {path}")

    # =========================================================================
    # Event Emission
    # =========================================================================

    def emit(self, event: Event) -> None:
        """Emit an event to all subscribed clients."""
        self._on_event(event)

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for clients."""
        self._on_event = callback

    def set_auth_complete_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called when authentication completes.

        This is called when a session that was in auth-pending state
        successfully completes authentication.
        """
        self._on_auth_complete = callback

    def emit_current_state(
        self,
        emit_fn: Optional[EventCallback] = None,
        skip_session_info: bool = False,
        clear_stale_pending_requests: bool = False
    ) -> None:
        """Emit current agent state to a specific client or all clients.

        This is useful when a client attaches to an existing session and needs
        to receive the current agent state that was emitted before they connected.

        Args:
            emit_fn: Optional callback to emit to a specific client.
                     If None, uses the default event callback (broadcast).
            skip_session_info: If True, skip emitting SessionInfoEvent (caller will send it).
            clear_stale_pending_requests: If True, emit "resolved" events for permission/
                clarification if no request is pending on the server. This clears stale
                client state after session recovery.
        """
        logger.info(f"emit_current_state called, emit_fn={emit_fn is not None}, agents={list(self._agents.keys())}")
        emit = emit_fn or self._on_event

        # Emit session info with model details (unless caller is sending its own)
        if not skip_session_info:
            logger.info(f"  emitting SessionInfoEvent")
            emit(SessionInfoEvent(
                session_id="",  # Will be set by SessionManager if needed
                session_name="",
                model_provider=self._model_provider,
                model_name=self._model_name,
            ))

        # Emit AgentCreatedEvent for all existing agents (from _agents dict)
        for agent_id, agent in self._agents.items():
            emit(AgentCreatedEvent(
                agent_id=agent.agent_id,
                agent_name=agent.name,
                agent_type=agent.agent_type,
                profile_name=agent.profile_name,
                parent_agent_id=agent.parent_agent_id,
                created_at=agent.created_at,
            ))

            # If agent has a non-idle status, emit that too
            if agent.status != "idle":
                emit(AgentStatusChangedEvent(
                    agent_id=agent.agent_id,
                    status=agent.status,
                ))

        # Emit instruction budget for main agent
        if self._jaato:
            session = self._jaato.get_session()
            if session and session.instruction_budget:
                emit(InstructionBudgetEvent(
                    agent_id=session.agent_id,
                    budget_snapshot=session.instruction_budget.snapshot(),
                ))

        # Emit restored subagent state from SubagentPlugin
        # This handles subagents that were restored from persistence but not yet
        # tracked in _agents (since they're managed by SubagentPlugin._active_sessions)
        self._emit_subagent_state(emit)

        # Clear stale pending requests on client if requested
        # This is used after session recovery when the server has no pending requests
        # but the client might still have stale UI state from before the crash
        if clear_stale_pending_requests:
            self._emit_clear_stale_requests(emit)

    def _emit_subagent_state(self, emit: EventCallback) -> None:
        """Emit state for subagents from SubagentPlugin._active_sessions.

        This is called by emit_current_state() to ensure reconnecting clients
        see all active subagents, including those restored from persistence.

        Args:
            emit: Event callback to use for emission.
        """
        if not self.registry:
            return

        subagent_plugin = self.registry.get_plugin("subagent")
        if not subagent_plugin or not hasattr(subagent_plugin, '_active_sessions'):
            return

        from datetime import datetime

        for agent_id, info in subagent_plugin._active_sessions.items():
            # Skip if already emitted via _agents dict
            if agent_id in self._agents:
                continue

            profile = info.get('profile')
            created_at = info.get('created_at')
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()

            emit(AgentCreatedEvent(
                agent_id=agent_id,
                agent_name=profile.name if profile else agent_id,
                agent_type="subagent",
                profile_name=profile.name if profile else "",
                parent_agent_id="main",
                created_at=created_at,
            ))

            # Emit context update for the subagent
            session = info.get('session')
            if session:
                usage = session.get_context_usage()
                context_limit = session.get_context_limit()
                emit(ContextUpdatedEvent(
                    agent_id=agent_id,
                    total_tokens=usage.get('total_tokens', 0),
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('output_tokens', 0),
                    context_limit=context_limit,
                    percent_used=usage.get('percent_used', 0.0),
                    tokens_remaining=max(0, context_limit - usage.get('total_tokens', 0)),
                    turns=usage.get('turns', 0),
                ))

                # Emit instruction budget for the subagent
                if hasattr(session, 'instruction_budget') and session.instruction_budget:
                    emit(InstructionBudgetEvent(
                        agent_id=agent_id,
                        budget_snapshot=session.instruction_budget.snapshot(),
                    ))

    def _emit_clear_stale_requests(self, emit: EventCallback) -> None:
        """Emit "resolved" events to clear stale pending requests on clients.

        After session recovery, the client might still have UI state for a pending
        permission or clarification request that no longer exists on the server.
        This method emits resolved events with method="session_restored" to tell
        clients to clear their stale state.

        Args:
            emit: Event callback to use for emission.
        """
        # If no permission request is pending on server, emit a clear event
        # The client will ignore this if it has no pending request
        if not self._pending_permission_request_id:
            emit(PermissionResolvedEvent(
                agent_id="main",
                request_id="",  # Empty - client clears any pending request
                tool_name="",
                granted=False,
                method="session_restored",  # Special method indicating recovery
            ))
            logger.debug("Emitted PermissionResolvedEvent to clear stale client state")

        # Same for clarification requests
        if not self._pending_clarification_request_id:
            emit(ClarificationResolvedEvent(
                agent_id="main",
                request_id="",
                tool_name="",
                qa_pairs=[],
            ))
            logger.debug("Emitted ClarificationResolvedEvent to clear stale client state")

        # Same for reference selection requests
        if not self._pending_reference_selection_request_id:
            emit(ReferenceSelectionResolvedEvent(
                agent_id="main",
                request_id="",
                tool_name="",
                selected_ids=[],
            ))
            logger.debug("Emitted ReferenceSelectionResolvedEvent to clear stale client state")

    # =========================================================================
    # Initialization
    # =========================================================================

    def _emit_init_progress(
        self,
        step: str,
        status: str,
        step_number: int,
        total_steps: int,
        message: str = ""
    ) -> None:
        """Emit an initialization progress event."""
        self.emit(InitProgressEvent(
            step=step,
            status=status,
            step_number=step_number,
            total_steps=total_steps,
            message=message,
        ))

    def initialize(self) -> bool:
        """Initialize the server.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        total_steps = 6

        # Step 1: Load configuration
        self._emit_init_progress("Loading configuration", "running", 1, total_steps)

        # Read session's env file into session-specific storage (NOT global os.environ)
        # This keeps each session's configuration isolated from other sessions.
        from dotenv import dotenv_values
        raw_session_env = dotenv_values(self.env_file) if self.env_file else {}
        # Filter out None values and store as session env
        self._session_env = {k: v for k, v in raw_session_env.items() if v is not None}

        def get_config(key: str) -> Optional[str]:
            """Get config value from session env, falling back to global env."""
            return self._session_env.get(key) or os.environ.get(key)

        try:
            active_bundle = active_cert_bundle(verbose=False)

            model_name = get_config("MODEL_NAME")
            if not model_name:
                self._emit_init_progress("Loading configuration", "error", 1, total_steps,
                                         "Missing MODEL_NAME")
                self.emit(ErrorEvent(
                    error="Missing required environment variable: MODEL_NAME",
                    error_type="ConfigurationError",
                    recoverable=False,
                ))
                return False

            # Get provider from session env (takes precedence over constructor arg)
            session_provider = get_config("JAATO_PROVIDER")
            provider_to_use = session_provider or self._provider

            # Get provider-specific settings (may be None for non-Google providers)
            project_id = get_config("PROJECT_ID")
            location = get_config("LOCATION")
            self._emit_init_progress("Loading configuration", "done", 1, total_steps)

            # Step 2: Connect to model provider
            # Credential validation is handled by each provider during connect()
            # Use session env context so provider can access session-specific API keys
            self._emit_init_progress("Connecting to model provider", "running", 2, total_steps)
            with self._with_session_env():
                self._jaato = JaatoClient(provider_name=provider_to_use)
                # Pass project/location for providers that need them (Google/Vertex)
                # Other providers ignore these and use their own env vars
                self._jaato.connect(project_id, location, model_name)
        except Exception as e:
            self._emit_init_progress("Connecting to model provider", "error", 2, total_steps,
                                     str(e))
            self.emit(ErrorEvent(
                error=f"Failed to connect: {e}",
                error_type=type(e).__name__,
                recoverable=False,
            ))
            return False

        self._model_name = self._jaato.model_name or model_name
        self._model_provider = self._jaato.provider_name
        self._jaato.set_terminal_width(self._terminal_width)
        self._emit_init_progress("Connecting to model provider", "done", 2, total_steps)

        # Step 3: Discover and configure plugins
        self._emit_init_progress("Loading plugins", "running", 3, total_steps)
        self.registry = PluginRegistry(model_name=model_name)
        self.registry.discover()

        plugin_configs = {
            "todo": {
                "reporter_type": "memory",
                "storage_type": "memory",
            },
            "references": {
                "channel_type": "queue",
            },
            "clarification": {
                "channel_type": "queue",
            },
            # LSP and MCP need workspace_path during initialize() to find config files
            # Also include session_id for log disambiguation in multi-session mode
            "lsp": {
                "workspace_path": self._workspace_path,
                "session_id": self._session_id,
            },
            "mcp": {
                "workspace_path": self._workspace_path,
                "session_id": self._session_id,
            },
            # Pass session_id to file_edit for session-scoped backup storage
            # workspace_root is handled by set_workspace_path() broadcast
            "file_edit": {
                "session_id": self._session_id,
            },
            # Pass session_id to waypoint plugin for session-scoped waypoint storage
            "waypoint": {
                "session_id": self._session_id,
            },
        }
        self.registry.expose_all(plugin_configs)
        self.todo_plugin = self.registry.get_plugin("todo")

        # Broadcast workspace path to all plugins implementing set_workspace_path()
        # This covers: file_edit, cli, filesystem_query, lsp, mcp, and any future plugins
        if self._workspace_path:
            self.registry.set_workspace_path(self._workspace_path)

        # Note: Plugins with set_plugin_registry() are auto-wired during expose_all()
        # No manual wiring needed for artifact_tracker, file_edit, cli, references, etc.

        self.permission_plugin = PermissionPlugin()
        self.permission_plugin.initialize({
            "channel_type": "queue",
            "channel_config": {"use_colors": False},
            "policy": {
                "defaultPolicy": "ask",
                "whitelist": {"tools": [], "patterns": []},
                "blacklist": {"tools": [], "patterns": []},
            }
        })
        self._emit_init_progress("Loading plugins", "done", 3, total_steps)

        # Set up formatter pipeline for server-side output formatting
        self._setup_formatter_pipeline()

        # Step 4: Verify authentication (may trigger interactive login via plugin)
        self._emit_init_progress("Verifying authentication", "running", 4, total_steps)
        self._trace(f"[auth] Starting verify_auth for provider: {self._model_provider}")

        self._auth_pending = False  # Track if auth is still needed
        self._auth_plugin_command = None

        def auth_message(msg: str) -> None:
            """Send auth status messages to the client."""
            self._trace(f"[auth] {msg}")
            self.emit(SystemMessageEvent(message=msg, style="info"))

        try:
            # Use session env context and workspace directory so auth can access
            # session-specific credentials and save tokens to the right location
            with self._with_session_env(), self._in_workspace():
                auth_ok = self._jaato.verify_auth(allow_interactive=True, on_message=auth_message)

            if not auth_ok:
                # Credentials not found - try to use provider-specific auth plugin
                auth_plugin = self._get_auth_plugin_for_provider(self._model_provider)

                if auth_plugin:
                    self._trace(f"[auth] Using {auth_plugin.name} plugin for interactive login")
                    self._auth_pending = True
                    # Store the auth command name for provider-agnostic completion check
                    auth_commands = auth_plugin.get_user_commands()
                    self._auth_plugin_command = auth_commands[0].name if auth_commands else None

                    # Set up output callback for plugin messages
                    def plugin_output(source: str, text: str, mode: str) -> None:
                        self._trace(f"[auth][{source}] {text.rstrip()}")
                        self.emit(SystemMessageEvent(message=text.rstrip(), style="info"))

                    auth_plugin.set_output_callback(plugin_output)

                    # Run the login command in workspace context so tokens are saved
                    # to the correct location. This is a blocking call that waits
                    # for OAuth flow to complete.
                    with self._in_workspace():
                        auth_plugin.execute_user_command(auth_plugin.get_user_commands()[0].name, {"action": "login"})

                    # Check if auth completed during the plugin execution
                    # (the plugin blocks until OAuth flow completes or times out)
                    self._check_auth_completion()

                    # If still pending after check, emit waiting status
                    if self._auth_pending:
                        self._emit_init_progress("Verifying authentication", "pending", 4, total_steps,
                                                 "Waiting for authentication")
                else:
                    self._emit_init_progress("Verifying authentication", "error", 4, total_steps,
                                             "No credentials found")
                    self.emit(ErrorEvent(
                        error="Authentication failed: no credentials found and no auth plugin available",
                        error_type="AuthenticationError",
                        recoverable=False,
                    ))
                    return False

        except Exception as e:
            self._emit_init_progress("Verifying authentication", "error", 4, total_steps, str(e))
            self.emit(ErrorEvent(
                error=f"Authentication failed: {e}",
                error_type=type(e).__name__,
                recoverable=False,
            ))
            return False

        if not self._auth_pending:
            self._trace("[auth] verify_auth completed successfully")
            self._emit_init_progress("Verifying authentication", "done", 4, total_steps)
        else:
            # Auth pending - only configure plugins for user commands (no provider session)
            # Skip remaining steps until auth completes
            self._trace("[auth] Configuring plugins only (auth pending, skipping provider session)")
            self._jaato.configure_plugins_only(self.registry, self.permission_plugin, self.ledger)
            return True

        # Step 5: Configure tools (only if auth is complete)
        # Use session env and workspace context so plugins can access session-specific
        # config and tokens are loaded from the correct location
        self._emit_init_progress("Configuring tools", "running", 5, total_steps)
        with self._with_session_env(), self._in_workspace():
            self._jaato.configure_tools(self.registry, self.permission_plugin, self.ledger)

            # Wire formatter pipeline into runtime so output formatters can
            # contribute system instructions (e.g., mermaid rendering hints)
            if self._formatter_pipeline:
                runtime = self._jaato.get_runtime()
                if runtime:
                    runtime.set_formatter_pipeline(self._formatter_pipeline)

            gc_result = load_gc_from_file()
        gc_threshold = None
        gc_strategy = None
        gc_target_percent = None
        gc_continuous_mode = False
        if gc_result:
            gc_plugin, gc_config = gc_result
            self._jaato.set_gc_plugin(gc_plugin, gc_config)
            gc_threshold = gc_config.threshold_percent
            gc_target_percent = gc_config.target_percent
            gc_continuous_mode = gc_config.continuous_mode
            gc_strategy = getattr(gc_plugin, 'name', 'gc')
            if gc_strategy.startswith('gc_'):
                gc_strategy = gc_strategy[3:]  # Remove 'gc_' prefix

        # Set up instruction budget callback and emit initial budget
        # This must happen after configure_tools() which populates the budget
        session = self._jaato.get_session()
        if session:
            server = self

            def instruction_budget_callback(snapshot: dict):
                server.emit(InstructionBudgetEvent(
                    agent_id=snapshot.get('agent_id', 'main'),
                    budget_snapshot=snapshot,
                ))

            session.set_instruction_budget_callback(instruction_budget_callback)

            # Emit initial budget snapshot
            if session.instruction_budget:
                self.emit(InstructionBudgetEvent(
                    agent_id=session.agent_id,
                    budget_snapshot=session.instruction_budget.snapshot(),
                ))

        self._emit_init_progress("Configuring tools", "done", 5, total_steps)

        # Step 6: Set up session
        self._emit_init_progress("Setting up session", "running", 6, total_steps)
        self._setup_session_plugin()
        self._setup_agent_hooks()
        self._setup_permission_hooks()
        self._setup_clarification_hooks()
        self._setup_reference_selection_hooks()
        self._setup_plan_hooks()
        self._setup_queue_channels()
        self._create_main_agent()
        # Store GC config in main agent state
        if "main" in self._agents and gc_threshold is not None:
            self._agents["main"].gc_threshold = gc_threshold
            self._agents["main"].gc_strategy = gc_strategy
            self._agents["main"].gc_target_percent = gc_target_percent
            self._agents["main"].gc_continuous_mode = gc_continuous_mode

        # Emit initial context update so toolbar shows correct usage at startup
        # This must happen after _create_main_agent() so client has the agent registered
        if self._jaato:
            usage = self._jaato.get_context_usage()
            self.emit(ContextUpdatedEvent(
                agent_id="main",
                total_tokens=usage.get('total_tokens', 0),
                prompt_tokens=usage.get('prompt_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                context_limit=usage.get('context_limit', 128000),
                percent_used=usage.get('percent_used', 0.0),
                tokens_remaining=usage.get('tokens_remaining', 128000),
                turns=usage.get('turns', 0),
                gc_threshold=gc_threshold,
                gc_strategy=gc_strategy,
                gc_target_percent=gc_target_percent,
                gc_continuous_mode=gc_continuous_mode,
            ))

        self._emit_init_progress("Setting up session", "done", 6, total_steps)

        self.emit(SystemMessageEvent(
            message=f"Connected to {self._model_provider}/{self._model_name}",
            style="info",
        ))

        return True

    def _create_main_agent(self) -> None:
        """Create the main agent entry.

        Note: This only creates the local AgentState tracking. The AgentCreatedEvent
        is already emitted via the UI hooks when set_ui_hooks() is called on JaatoClient,
        which triggers on_agent_created() in ServerAgentHooks.
        """
        logger.debug("  _create_main_agent: creating AgentState...")

        # Check if agent was already created by hooks
        if "main" in self._agents:
            logger.debug("  _create_main_agent: 'main' already exists (created by hooks), skipping")
            return

        agent = AgentState(
            agent_id="main",
            name="Main Agent",
            agent_type="main",
        )
        self._agents["main"] = agent
        self._selected_agent_id = "main"
        logger.debug("  _create_main_agent: agent state created")

        # Note: AgentCreatedEvent is NOT emitted here - it's handled by
        # ServerAgentHooks.on_agent_created() when set_ui_hooks() is called

    def _setup_formatter_pipeline(self) -> None:
        """Set up the formatter pipeline for server-side output formatting.

        Uses FormatterRegistry for dynamic formatter discovery and configuration.
        Loads config from .jaato/formatters.json if present, otherwise uses defaults.

        Formatters that need tool plugins (like code_validation_formatter needing
        LSP) will wire themselves automatically via wire_dependencies().
        """
        # Create formatter registry and discover available formatters
        formatter_registry = create_registry()
        formatter_registry.discover()

        # Give formatters access to tool plugins for self-wiring
        if self.registry:
            formatter_registry.set_tool_registry(self.registry)

        # Try to load config from project directory (workspace) or user directory
        # Use workspace_path if available, otherwise fall back to relative path
        if self._workspace_path:
            project_config = os.path.join(self._workspace_path, ".jaato/formatters.json")
        else:
            project_config = ".jaato/formatters.json"

        config_loaded = (
            formatter_registry.load_config(project_config) or
            formatter_registry.load_config(os.path.expanduser("~/.jaato/formatters.json"))
        )

        if not config_loaded:
            formatter_registry.use_defaults()
            self._trace("Using default formatter configuration")
        else:
            self._trace("Loaded formatter configuration from file")

        # Create pipeline from registry (formatters wire themselves)
        self._formatter_pipeline = formatter_registry.create_pipeline(self._terminal_width)

        self._trace(f"Formatter pipeline initialized with {len(self._formatter_pipeline.list_formatters())} formatters")

    def _get_agent_pipeline(self, agent_id: str) -> Optional[Any]:
        """Get the formatter pipeline for a specific agent.

        Each agent has its own formatter pipeline to prevent cross-contamination
        of buffered content when multiple agents are active.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            The agent's formatter pipeline, or None if agent not found.
        """
        if agent_id not in self._agents:
            return None

        agent = self._agents[agent_id]
        if agent.formatter_pipeline is None and self._formatter_pipeline:
            # Lazily create a new pipeline for this agent using the same config
            # We clone the main pipeline's configuration
            from shared.plugins.formatter_pipeline import create_registry
            formatter_registry = create_registry()
            formatter_registry.discover()
            if self.registry:
                formatter_registry.set_tool_registry(self.registry)
            # Use same config loading as main pipeline
            # Use workspace_path if available for project config
            if self._workspace_path:
                project_config = os.path.join(self._workspace_path, ".jaato/formatters.json")
            else:
                project_config = ".jaato/formatters.json"
            config_loaded = (
                formatter_registry.load_config(project_config) or
                formatter_registry.load_config(os.path.expanduser("~/.jaato/formatters.json"))
            )
            if not config_loaded:
                formatter_registry.use_defaults()
            agent.formatter_pipeline = formatter_registry.create_pipeline(self._terminal_width)
            self._trace(f"Created formatter pipeline for agent {agent_id}")
        return agent.formatter_pipeline

    def _setup_session_plugin(self) -> None:
        """Set up session persistence plugin.

        Each JaatoServer has its own session plugin instance for tool operations.
        SessionManager has a separate plugin instance for persistence operations.
        """
        if not self._jaato:
            logger.debug("  _setup_session_plugin: no _jaato, returning early")
            return

        try:
            logger.debug("  _setup_session_plugin: loading session config...")
            session_config = load_session_config()
            logger.debug("  _setup_session_plugin: creating session plugin...")
            session_plugin = create_session_plugin()
            logger.debug("  _setup_session_plugin: initializing session plugin...")
            session_plugin.initialize({'storage_path': session_config.storage_path})
            logger.debug("  _setup_session_plugin: setting session plugin on jaato...")
            self._jaato.set_session_plugin(session_plugin, session_config)
            logger.debug("  _setup_session_plugin: session plugin set")

            # Set session ID on plugin so it knows the current session
            if self._session_id and hasattr(session_plugin, 'set_session_id'):
                session_plugin.set_session_id(self._session_id)
                logger.debug(f"  _setup_session_plugin: session_id set to {self._session_id}")

            # Set up callback to emit event when description changes
            if hasattr(session_plugin, 'set_description_callback'):
                def on_description_changed(session_id: str, description: str) -> None:
                    self.emit(SessionDescriptionUpdatedEvent(
                        session_id=session_id,
                        description=description,
                    ))
                session_plugin.set_description_callback(on_description_changed)
                logger.debug("  _setup_session_plugin: description callback set")

            if self.registry:
                logger.debug("  _setup_session_plugin: registering session plugin with registry...")
                self.registry.register_plugin(session_plugin, enrichment_only=True)

            if self.permission_plugin and hasattr(session_plugin, 'get_auto_approved_tools'):
                auto_approved = session_plugin.get_auto_approved_tools()
                if auto_approved:
                    logger.debug(f"  _setup_session_plugin: adding {len(auto_approved)} auto-approved tools")
                    self.permission_plugin.add_whitelist_tools(auto_approved)

            logger.debug("  _setup_session_plugin: completed successfully")
        except Exception as e:
            logger.warning(f"  _setup_session_plugin: exception: {e}")
            pass  # Session plugin is optional

    def _setup_agent_hooks(self) -> None:
        """Set up agent lifecycle hooks."""
        logger.debug("  _setup_agent_hooks: entering...")
        if not self._jaato:
            logger.debug("  _setup_agent_hooks: no _jaato, returning early")
            return

        logger.debug("  _setup_agent_hooks: defining ServerAgentHooks class...")
        server = self

        class ServerAgentHooks:
            """Agent hooks that emit events."""

            def on_agent_created(self, agent_id, agent_name, agent_type, profile_name,
                                 parent_agent_id, icon_lines, created_at):
                agent = AgentState(
                    agent_id=agent_id,
                    name=agent_name,
                    agent_type=agent_type,
                    profile_name=profile_name,
                    parent_agent_id=parent_agent_id,
                )
                if created_at:
                    # Convert datetime to isoformat string if needed
                    if hasattr(created_at, 'isoformat'):
                        agent.created_at = created_at.isoformat()
                    else:
                        agent.created_at = str(created_at)
                server._agents[agent_id] = agent

                server.emit(AgentCreatedEvent(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    agent_type=agent_type,
                    profile_name=profile_name,
                    parent_agent_id=parent_agent_id,
                    icon_lines=icon_lines,
                    created_at=agent.created_at,
                ))

            def on_agent_output(self, agent_id, source, text, mode):
                server._trace(f"ON_AGENT_OUTPUT agent={agent_id} source={source} len={len(text)} mode={mode}")
                # Get agent-specific formatter pipeline to prevent cross-contamination
                agent_pipeline = server._get_agent_pipeline(agent_id)
                # For model output with streaming formatter pipeline
                if source == "model" and agent_pipeline:
                    # Process chunk through streaming pipeline
                    # Pipeline buffers code blocks, passes through regular text
                    for output in agent_pipeline.process_chunk(text):
                        if output:
                            server.emit(AgentOutputEvent(
                                agent_id=agent_id,
                                source=source,
                                text=output,
                                mode=mode,
                            ))
                else:
                    # Non-model output: strip <hidden>...</hidden> content
                    # These are mid-turn prompts that may contain internal tags
                    filtered_text = re.sub(r'<hidden>.*?</hidden>', '', text, flags=re.DOTALL)

                    # Flush mode: flush the formatter pipeline to emit buffered content
                    # BEFORE tool events, ensuring text appears in correct order
                    if mode == "flush" and agent_pipeline:
                        for output in agent_pipeline.flush():
                            if output:
                                server.emit(AgentOutputEvent(
                                    agent_id=agent_id,
                                    source="model",
                                    text=output,
                                    mode="append",
                                ))

                    # For other modes, only emit if content remains after filtering
                    if filtered_text.strip():
                        server.emit(AgentOutputEvent(
                            agent_id=agent_id,
                            source=source,
                            text=filtered_text,
                            mode=mode,
                        ))

            def on_agent_status_changed(self, agent_id, status, error=None):
                if agent_id in server._agents:
                    server._agents[agent_id].status = status
                server.emit(AgentStatusChangedEvent(
                    agent_id=agent_id,
                    status=status,
                    error=error,
                ))

            def on_agent_completed(self, agent_id, completed_at, success,
                                   token_usage=None, turns_used=None):
                # Convert datetime to isoformat string if needed
                completed_at_str = completed_at
                if completed_at and hasattr(completed_at, 'isoformat'):
                    completed_at_str = completed_at.isoformat()
                elif completed_at:
                    completed_at_str = str(completed_at)

                if agent_id in server._agents:
                    server._agents[agent_id].completed_at = completed_at_str

                server.emit(AgentCompletedEvent(
                    agent_id=agent_id,
                    completed_at=completed_at_str,
                    success=success,
                    token_usage=token_usage,
                    turns_used=turns_used,
                ))

            def on_agent_turn_completed(self, agent_id, turn_number, prompt_tokens,
                                        output_tokens, total_tokens, duration_seconds,
                                        function_calls):
                # Flush any remaining buffered content from the agent's formatter pipeline
                agent_pipeline = server._get_agent_pipeline(agent_id)
                if agent_pipeline:
                    for output in agent_pipeline.flush():
                        if output:
                            server.emit(AgentOutputEvent(
                                agent_id=agent_id,
                                source="model",
                                text=output,
                                mode="append",
                            ))
                    # Reset pipeline for next turn
                    agent_pipeline.reset()

                if agent_id in server._agents:
                    server._agents[agent_id].turn_accounting.append({
                        'turn': turn_number,
                        'prompt': prompt_tokens,
                        'output': output_tokens,
                        'total': total_tokens,
                        'duration_seconds': duration_seconds,
                        'function_calls': function_calls,
                    })
                server.emit(TurnCompletedEvent(
                    agent_id=agent_id,
                    turn_number=turn_number,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    duration_seconds=duration_seconds,
                    function_calls=function_calls,
                ))

            def on_agent_context_updated(self, agent_id, total_tokens, prompt_tokens,
                                         output_tokens, turns, percent_used):
                if agent_id in server._agents:
                    server._agents[agent_id].context_usage = {
                        'total_tokens': total_tokens,
                        'prompt_tokens': prompt_tokens,
                        'output_tokens': output_tokens,
                        'turns': turns,
                        'percent_used': percent_used,
                    }
                context_limit = server._jaato.get_context_limit() if server._jaato else 0
                # Get GC config from agent state
                gc_threshold = None
                gc_strategy = None
                gc_target_percent = None
                gc_continuous_mode = False
                if agent_id in server._agents:
                    gc_threshold = server._agents[agent_id].gc_threshold
                    gc_strategy = server._agents[agent_id].gc_strategy
                    gc_target_percent = server._agents[agent_id].gc_target_percent
                    gc_continuous_mode = server._agents[agent_id].gc_continuous_mode
                server.emit(ContextUpdatedEvent(
                    agent_id=agent_id,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    context_limit=context_limit,
                    percent_used=percent_used,
                    tokens_remaining=max(0, context_limit - total_tokens),
                    turns=turns,
                    gc_threshold=gc_threshold,
                    gc_strategy=gc_strategy,
                    gc_target_percent=gc_target_percent,
                    gc_continuous_mode=gc_continuous_mode,
                ))

            def on_agent_gc_config(self, agent_id, threshold, strategy, target_percent=None, continuous_mode=False):
                # Store GC config in agent state
                if agent_id in server._agents:
                    server._agents[agent_id].gc_threshold = threshold
                    server._agents[agent_id].gc_strategy = strategy
                    server._agents[agent_id].gc_target_percent = target_percent
                    server._agents[agent_id].gc_continuous_mode = continuous_mode
                    # Emit ContextUpdatedEvent with GC config so client gets notified
                    agent = server._agents[agent_id]
                    server.emit(ContextUpdatedEvent(
                        agent_id=agent_id,
                        gc_threshold=threshold,
                        gc_strategy=strategy,
                        gc_target_percent=target_percent,
                        gc_continuous_mode=continuous_mode,
                        # Include current context usage if available
                        total_tokens=agent.context_usage.get('total_tokens', 0),
                        prompt_tokens=agent.context_usage.get('prompt_tokens', 0),
                        output_tokens=agent.context_usage.get('output_tokens', 0),
                        context_limit=agent.context_usage.get('context_limit', 0),
                        percent_used=agent.context_usage.get('percent_used', 0.0),
                        tokens_remaining=agent.context_usage.get('tokens_remaining', 0),
                        turns=agent.context_usage.get('turns', 0),
                    ))

            def on_agent_history_updated(self, agent_id, history):
                if agent_id in server._agents:
                    server._agents[agent_id].history = history

            def on_tool_call_start(self, agent_id, tool_name, tool_args, call_id=None):
                # Track current agent for permission/clarification routing
                server._current_tool_agent_id = agent_id

                # Flush any buffered model output before starting the tool
                # This ensures model text appears BEFORE the tool tree
                # Use agent-specific pipeline to prevent cross-contamination
                agent_pipeline = server._get_agent_pipeline(agent_id)
                if agent_pipeline:
                    for output in agent_pipeline.flush():
                        if output:
                            server.emit(AgentOutputEvent(
                                agent_id=agent_id,
                                source="model",
                                text=output,
                                mode="append",
                            ))
                    agent_pipeline.reset()

                # Extract "message" or similar intent arguments and emit as model text
                # This shows the model's intent before the tool block, not collapsed in it
                # We keep the arg in tool_args so the tool still receives it
                intent_arg_names = ("message", "summary", "intent", "rationale")
                intent_text = None
                if tool_args:
                    for arg_name in intent_arg_names:
                        if arg_name in tool_args:
                            val = tool_args[arg_name]
                            if val and isinstance(val, str) and val.strip():
                                intent_text = val.strip()
                                break  # Use first found intent arg

                if intent_text:
                    server.emit(AgentOutputEvent(
                        agent_id=agent_id,
                        source="model",
                        text=intent_text,
                        mode="write",
                    ))

                server.emit(ToolCallStartEvent(
                    agent_id=agent_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    call_id=call_id,
                ))

            def on_tool_call_end(self, agent_id, tool_name, success, duration_seconds,
                                 error_message=None, call_id=None, backgrounded=False,
                                 continuation_id=None, show_output=None):
                server.emit(ToolCallEndEvent(
                    agent_id=agent_id,
                    tool_name=tool_name,
                    call_id=call_id,
                    success=success,
                    duration_seconds=duration_seconds,
                    error_message=error_message,
                    backgrounded=backgrounded,
                    continuation_id=continuation_id,
                    show_output=show_output,
                ))

            def on_tool_output(self, agent_id, call_id, chunk):
                # Process tool output through formatter pipeline for syntax highlighting
                # and marker transformation (e.g., <notebook-cell> → <nb-row>)
                # Use agent-specific pipeline to prevent cross-contamination
                agent_pipeline = server._get_agent_pipeline(agent_id)
                if agent_pipeline:
                    formatted_parts = []
                    for output in agent_pipeline.process_chunk(chunk):
                        formatted_parts.append(output)
                    for output in agent_pipeline.flush():
                        formatted_parts.append(output)
                    agent_pipeline.reset()
                    chunk = "".join(formatted_parts)

                server.emit(ToolOutputEvent(
                    agent_id=agent_id,
                    call_id=call_id,
                    chunk=chunk,
                ))

            def on_agent_instruction_budget_updated(self, agent_id, budget_snapshot):
                server.emit(InstructionBudgetEvent(
                    agent_id=agent_id,
                    budget_snapshot=budget_snapshot,
                ))

            def on_turn_progress(self, agent_id, total_tokens, prompt_tokens,
                                 output_tokens, percent_used, pending_tool_calls):
                context_limit = server._jaato.get_context_limit() if server._jaato else 0
                server.emit(TurnProgressEvent(
                    agent_id=agent_id,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    context_limit=context_limit,
                    percent_used=percent_used,
                    tokens_remaining=max(0, context_limit - total_tokens),
                    pending_tool_calls=pending_tool_calls,
                ))

        logger.debug("  _setup_agent_hooks: class defined, creating instance...")
        hooks = ServerAgentHooks()
        logger.debug("  _setup_agent_hooks: calling jaato.set_ui_hooks...")
        self._jaato.set_ui_hooks(hooks)
        logger.debug("  _setup_agent_hooks: jaato.set_ui_hooks done")

        # Register with subagent plugin
        if self.registry:
            logger.debug("  _setup_agent_hooks: getting subagent plugin...")
            subagent_plugin = self.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_ui_hooks'):
                logger.debug("  _setup_agent_hooks: calling subagent.set_ui_hooks...")
                subagent_plugin.set_ui_hooks(hooks)
                logger.debug("  _setup_agent_hooks: subagent.set_ui_hooks done")
        logger.debug("  _setup_agent_hooks: completed")

    def _setup_permission_hooks(self) -> None:
        """Set up permission lifecycle hooks."""
        if not self.permission_plugin:
            return

        server = self

        def on_permission_requested(tool_name: str, request_id: str,
                                    tool_args: dict, response_options: list,
                                    call_id: Optional[str] = None):
            server._pending_permission_request_id = request_id
            server._waiting_for_channel_input = True

            # Convert response options to dicts
            options_dicts = []
            for opt in response_options:
                opt_dict = {
                    "key": getattr(opt, 'short', getattr(opt, 'key', str(opt))),
                    "label": getattr(opt, 'full', getattr(opt, 'label', str(opt))),
                    "action": getattr(opt, 'decision', getattr(opt, 'action', 'unknown')),
                }
                # Convert enum to string if needed
                if hasattr(opt_dict["action"], 'value'):
                    opt_dict["action"] = opt_dict["action"].value
                if hasattr(opt, 'description') and opt.description:
                    opt_dict["description"] = opt.description
                options_dicts.append(opt_dict)

            # Get formatted prompt from permission plugin
            prompt_lines = None
            format_hint = None
            warnings = None
            warning_level = None
            if hasattr(server.permission_plugin, 'get_formatted_prompt'):
                try:
                    result = server.permission_plugin.get_formatted_prompt(
                        tool_name, tool_args or {}, "ipc"
                    )
                    # Handle both old (4-tuple) and new (6-tuple) return formats
                    if len(result) >= 6:
                        prompt_lines, format_hint, language, raw_details, warnings, warning_level = result
                    else:
                        prompt_lines, format_hint, language, raw_details = result

                    # Use agent-specific pipeline to prevent cross-contamination
                    agent_pipeline = server._get_agent_pipeline(server._current_tool_agent_id)
                    if agent_pipeline:
                        # First, flush any buffered model output and emit it separately
                        # This prevents model text from leaking into the permission prompt
                        for output in agent_pipeline.flush():
                            if output:
                                server.emit(AgentOutputEvent(
                                    agent_id=server._current_tool_agent_id,
                                    source="model",
                                    text=output,
                                    mode="append",
                                ))
                        agent_pipeline.reset()

                        # Build permission content for unified output flow
                        content_parts = []

                        # When format_hint is "code", include code block first with syntax highlighting
                        if format_hint == "code" and language and raw_details:
                            code_block = f"```{language}\n{raw_details}\n```\n"
                            # Format through pipeline for syntax highlighting
                            formatted_code = []
                            for output in agent_pipeline.process_chunk(code_block):
                                formatted_code.append(output)
                            for output in agent_pipeline.flush():
                                formatted_code.append(output)
                            agent_pipeline.reset()
                            if formatted_code:
                                content_parts.append("".join(formatted_code))

                        # Add security warnings with special markers for client styling
                        if warnings:
                            # Use XML-style markers that client can parse and style separately
                            level_marker = warning_level or "warning"
                            warnings_block = f"<security-warning level=\"{level_marker}\">\n{warnings}\n</security-warning>\n"
                            content_parts.append(warnings_block)

                        # Format the permission prompt summary + options
                        if prompt_lines:
                            formatted_lines = []
                            for line in prompt_lines:
                                for output in agent_pipeline.process_chunk(line + "\n"):
                                    formatted_lines.extend(output.rstrip("\n").split("\n"))
                            for output in agent_pipeline.flush():
                                formatted_lines.extend(output.rstrip("\n").split("\n"))
                            agent_pipeline.reset()
                            content_parts.append("\n".join(formatted_lines))

                        # Emit content as AgentOutputEvent (flows through main output area)
                        if content_parts:
                            full_content = "\n".join(content_parts)
                            server.emit(AgentOutputEvent(
                                agent_id=server._current_tool_agent_id,
                                source="permission",
                                text=full_content,
                                mode="write",
                            ))

                except Exception:
                    pass  # Content formatting failed, tool tree will show minimal status

            # Check if edit option is available (indicates editable tool)
            has_edit = any(opt.get("action") == "edit" for opt in options_dicts)
            editable_metadata = None
            if has_edit and server.permission_plugin and hasattr(server.permission_plugin, '_get_tool_schema'):
                try:
                    schema = server.permission_plugin._get_tool_schema(tool_name)
                    if schema and schema.editable:
                        editable_metadata = {
                            "parameters": schema.editable.parameters if hasattr(schema.editable, 'parameters') else [],
                            "format": schema.editable.format if hasattr(schema.editable, 'format') else "yaml",
                        }
                except Exception:
                    pass

            # Emit control event to signal input mode (lightweight, no content)
            server.emit(PermissionInputModeEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                call_id=call_id,
                response_options=options_dicts,
                tool_args=tool_args if has_edit else None,
                editable_metadata=editable_metadata,
            ))

        def on_permission_resolved(tool_name: str, request_id: str,
                                   granted: bool, method: str):
            server._pending_permission_request_id = None
            server._waiting_for_channel_input = False

            # Resolution status is shown in the tool tree (e.g., "✓ [once]")
            # No need to emit separate output text

            server.emit(PermissionResolvedEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                granted=granted,
                method=method,
            ))

            # Emit updated permission status (a/t/i responses change the policy)
            server.emit_permission_status()

        self.permission_plugin.set_permission_hooks(
            on_requested=on_permission_requested,
            on_resolved=on_permission_resolved,
        )

    def emit_permission_status(self) -> None:
        """Emit current permission status for client toolbar updates."""
        if not self.permission_plugin:
            return
        status = self.permission_plugin.get_permission_status()
        self.emit(PermissionStatusEvent(
            effective_default=status.get("effective_default", "ask"),
            suspension_scope=status.get("suspension_scope"),
        ))

    def _setup_clarification_hooks(self) -> None:
        """Set up clarification lifecycle hooks."""
        if not self.registry:
            return

        clarification_plugin = self.registry.get_plugin("clarification")
        if not clarification_plugin or not hasattr(clarification_plugin, 'set_clarification_hooks'):
            return

        server = self

        def on_clarification_requested(tool_name: str, prompt_lines: list):
            request_id = f"clarify_{datetime.utcnow().timestamp()}"
            server._pending_clarification_request_id = request_id
            server._waiting_for_channel_input = True

            # Emit context content as AgentOutputEvent (flows through main output)
            if prompt_lines:
                content = "\n".join(prompt_lines)
                server.emit(AgentOutputEvent(
                    agent_id=server._current_tool_agent_id,
                    source="clarification",
                    text=content,
                    mode="write",
                ))

        def on_clarification_resolved(tool_name: str, qa_pairs: list):
            request_id = server._pending_clarification_request_id or ""
            server._pending_clarification_request_id = None
            server._waiting_for_channel_input = False
            # Convert qa_pairs from list of tuples to list of lists for JSON serialization
            qa_pairs_serializable = [[q, a] for q, a in qa_pairs] if qa_pairs else []
            server.emit(ClarificationResolvedEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                qa_pairs=qa_pairs_serializable,
            ))

        def on_question_displayed(tool_name: str, question_index: int,
                                  total_questions: int, question_lines: list):
            # Emit question content as AgentOutputEvent (flows through main output)
            if question_lines:
                content = "\n".join(question_lines)
                server.emit(AgentOutputEvent(
                    agent_id=server._current_tool_agent_id,
                    source="clarification",
                    text=content,
                    mode="write",
                ))

            # Emit control event to signal input mode (lightweight, no content)
            server.emit(ClarificationInputModeEvent(
                agent_id=server._current_tool_agent_id,
                request_id=server._pending_clarification_request_id or "",
                tool_name=tool_name,
                question_index=question_index,
                total_questions=total_questions,
            ))

        def on_question_answered(tool_name: str, question_index: int, answer_summary: str):
            # Question answered, waiting for next or resolution
            pass

        clarification_plugin.set_clarification_hooks(
            on_requested=on_clarification_requested,
            on_resolved=on_clarification_resolved,
            on_question_displayed=on_question_displayed,
            on_question_answered=on_question_answered,
        )

    def _setup_reference_selection_hooks(self) -> None:
        """Set up reference selection lifecycle hooks."""
        if not self.registry:
            return

        references_plugin = self.registry.get_plugin("references")
        if not references_plugin or not hasattr(references_plugin, 'set_selection_hooks'):
            return

        server = self

        def on_selection_requested(tool_name: str, prompt_lines: list):
            request_id = f"ref_selection_{datetime.utcnow().timestamp()}"
            server._pending_reference_selection_request_id = request_id
            server._waiting_for_channel_input = True
            server.emit(ReferenceSelectionRequestedEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                prompt_lines=prompt_lines,
            ))

        def on_selection_resolved(tool_name: str, selected_ids: list):
            request_id = server._pending_reference_selection_request_id or ""
            server._pending_reference_selection_request_id = None
            server._waiting_for_channel_input = False
            server.emit(ReferenceSelectionResolvedEvent(
                agent_id=server._current_tool_agent_id,
                request_id=request_id,
                tool_name=tool_name,
                selected_ids=selected_ids,
            ))

        references_plugin.set_selection_hooks(
            on_requested=on_selection_requested,
            on_resolved=on_selection_resolved,
        )

    def _setup_plan_hooks(self) -> None:
        """Set up plan update hooks."""
        if not self.todo_plugin:
            return

        server = self

        def _get_agent_id(agent_name: Optional[str]) -> str:
            """Get agent ID from agent name."""
            agent_id = "main" if agent_name is None else agent_name
            for aid, agent in server._agents.items():
                if agent.profile_name == agent_name:
                    agent_id = aid
                    break
            return agent_id

        def update_callback(plan_data: dict, agent_name: Optional[str] = None):
            """Emit PlanUpdatedEvent from plan data."""
            agent_id = _get_agent_id(agent_name)
            steps = []
            for step in plan_data.get('steps', []):
                step_data = {
                    'content': step.get('description', ''),
                    'status': step.get('status', 'pending'),
                    'active_form': step.get('active_form'),
                }
                # Include cross-agent dependency info for blocked steps
                if step.get('blocked_by'):
                    step_data['blocked_by'] = step['blocked_by']
                if step.get('depends_on'):
                    step_data['depends_on'] = step['depends_on']
                if step.get('received_outputs'):
                    step_data['received_outputs'] = step['received_outputs']
                steps.append(step_data)
            server.emit(PlanUpdatedEvent(
                agent_id=agent_id,
                plan_name=plan_data.get('title', 'Plan'),
                steps=steps,
            ))

        def clear_callback(agent_name: Optional[str] = None):
            """Emit PlanClearedEvent."""
            agent_id = _get_agent_id(agent_name)
            server.emit(PlanClearedEvent(agent_id=agent_id))

        def output_callback(source: str, text: str, mode: str):
            """Emit AgentOutputEvent for plan messages."""
            server.emit(AgentOutputEvent(
                agent_id="main",
                source=source,
                text=text,
                mode=mode,
            ))

        # Reuse LivePlanReporter from rich-client with event-emitting callbacks
        reporter = create_live_reporter(
            update_callback=update_callback,
            clear_callback=clear_callback,
            output_callback=output_callback,
        )

        if hasattr(self.todo_plugin, '_reporter'):
            self.todo_plugin._reporter = reporter

        # Also set for subagent plugin
        if self.registry:
            subagent_plugin = self.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_plan_reporter'):
                subagent_plugin.set_plan_reporter(reporter)

    def _setup_queue_channels(self) -> None:
        """Set up queue-based channels for permission/clarification."""
        server = self

        def get_cancel_token():
            if server._jaato:
                session = server._jaato.get_session()
                if session and hasattr(session, '_cancel_token'):
                    return session._cancel_token
            return None

        class CancelTokenProxy:
            @property
            def is_cancelled(self):
                token = get_cancel_token()
                return token.is_cancelled if token else False

        cancel_token_proxy = CancelTokenProxy()

        # Output callback for channels
        def output_callback(source: str, text: str, mode: str):
            server.emit(AgentOutputEvent(
                agent_id="main",
                source=source,
                text=text,
                mode=mode,
            ))

        def on_prompt_state_change(waiting: bool):
            server._waiting_for_channel_input = waiting

        # Set callbacks on clarification plugin
        if self.registry:
            clarification_plugin = self.registry.get_plugin("clarification")
            if clarification_plugin and hasattr(clarification_plugin, '_channel'):
                channel = clarification_plugin._channel
                if hasattr(channel, 'set_callbacks'):
                    channel.set_callbacks(
                        output_callback=output_callback,
                        input_queue=self._channel_input_queue,
                        prompt_callback=on_prompt_state_change,
                        cancel_token=cancel_token_proxy,
                    )

            # References plugin
            references_plugin = self.registry.get_plugin("references")
            if references_plugin and hasattr(references_plugin, '_channel'):
                channel = references_plugin._channel
                if hasattr(channel, 'set_callbacks'):
                    channel.set_callbacks(
                        output_callback=output_callback,
                        input_queue=self._channel_input_queue,
                        prompt_callback=on_prompt_state_change,
                    )

        # Permission plugin
        if self.permission_plugin and hasattr(self.permission_plugin, '_channel'):
            channel = self.permission_plugin._channel
            if channel and hasattr(channel, 'set_callbacks'):
                channel.set_callbacks(
                    output_callback=output_callback,
                    input_queue=self._channel_input_queue,
                    prompt_callback=on_prompt_state_change,
                    cancel_token=cancel_token_proxy,
                    edit_callback=self._create_edit_callback(),
                )

    def _create_edit_callback(self) -> Callable:
        """Create edit callback for permission plugin in server mode.

        In server mode, editing happens on the client side. The client opens
        the external editor and sends back the edited arguments via
        PermissionResponseRequest.edited_arguments. This callback retrieves
        those pre-stored edited arguments.

        Returns:
            Callback that returns client-provided edited arguments.
        """
        server = self

        def edit_callback(arguments: Dict[str, Any], editable: Any) -> Optional[Dict[str, Any]]:
            """Return edited arguments provided by the client."""
            edited = server._pending_edited_arguments
            server._pending_edited_arguments = None  # Consume
            return edited

        return edit_callback

    # =========================================================================
    # Client Request Handlers
    # =========================================================================

    def send_message(self, text: str, attachments: Optional[List[Dict]] = None) -> None:
        """Send a message to the model.

        Args:
            text: The message text.
            attachments: Optional list of attachments.
        """
        if not self._jaato:
            self.emit(ErrorEvent(
                error="Client not initialized",
                error_type="StateError",
            ))
            return

        if self._model_running:
            # Inject directly into the session's queue (USER source - high priority)
            if self._jaato:
                session = self._jaato.get_session()
                session.inject_prompt(text, source_id="user", source_type=SourceType.USER)
                self.emit(MidTurnPromptQueuedEvent(
                    text=text,
                    position_in_queue=0,
            ))
            return

        # Track input
        self._original_inputs.append({"text": text, "local": False})

        # Emit user message as output
        self.emit(AgentOutputEvent(
            agent_id="main",
            source="user",
            text=text,
            mode="write",
        ))

        # Signal main agent is active
        self.emit(AgentStatusChangedEvent(
            agent_id="main",
            status="active",
        ))

        # Expand file references
        expanded_prompt = self._input_handler.expand_file_references(text)

        # Start model in background
        self._start_model_thread(expanded_prompt)

    def _start_model_thread(self, prompt: str) -> None:
        """Start the model call in a background thread."""
        server = self

        # Set up callback for when injected prompts are processed
        # This allows UI to remove prompts from pending bar
        if self._jaato:
            session = self._jaato.get_session()
            session.set_prompt_injected_callback(
                lambda text: server.emit(MidTurnPromptInjectedEvent(text=text))
            )

            # Set up callback for when child messages need continuation
            # This triggers a new turn when subagent sends messages while parent is idle
            def continuation_callback(child_messages: str):
                # Only trigger if not already running a model call
                if not server._model_running and child_messages:
                    server._trace(f"CONTINUATION: Child messages drained ({len(child_messages)} chars), triggering new turn")
                    # Signal main agent is active
                    server.emit(AgentStatusChangedEvent(
                        agent_id="main",
                        status="active",
                    ))
                    # Start model thread with child messages as the prompt
                    server._start_model_thread(child_messages)

            session.set_continuation_callback(continuation_callback)

            # Set up callback for retry notifications
            # This notifies the client when API calls are being retried due to rate limits
            def retry_callback(message: str, attempt: int, max_attempts: int, delay: float) -> None:
                # Determine error type from message
                error_type = "rate_limit" if "rate-limit" in message.lower() else "transient"
                server.emit(RetryEvent(
                    message=message,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay=delay,
                    error_type=error_type,
                ))

            session.set_retry_callback(retry_callback)

            # Set up callback for when streaming is interrupted for mid-turn prompt
            def mid_turn_interrupt_callback(partial_chars: int, prompt_preview: str):
                server._trace(f"MID_TURN_INTERRUPT: partial={partial_chars}, preview={prompt_preview[:50]}...")
                server.emit(MidTurnInterruptEvent(
                    partial_response_chars=partial_chars,
                    user_prompt_preview=prompt_preview,
                ))

            session.set_mid_turn_interrupt_callback(mid_turn_interrupt_callback)

            # Note: instruction_budget_callback is set up in initialize() after configure_tools()

        def output_callback(source: str, text: str, mode: str) -> None:
            # Skip - output is routed through agent hooks
            pass

        def usage_update_callback(usage) -> None:
            if usage.total_tokens == 0:
                return
            if server._jaato:
                context_limit = server._jaato.get_context_limit()
                percent_used = (usage.total_tokens / context_limit * 100) if context_limit > 0 else 0
                # Get current turn count from accounting
                turn_accounting = server._jaato.get_turn_accounting()
                turns = len(turn_accounting)
                # Get GC config from main agent state
                gc_threshold = None
                gc_strategy = None
                gc_target_percent = None
                gc_continuous_mode = False
                if "main" in server._agents:
                    gc_threshold = server._agents["main"].gc_threshold
                    gc_strategy = server._agents["main"].gc_strategy
                    gc_target_percent = server._agents["main"].gc_target_percent
                    gc_continuous_mode = server._agents["main"].gc_continuous_mode
                server.emit(ContextUpdatedEvent(
                    agent_id="main",
                    total_tokens=usage.total_tokens,
                    prompt_tokens=usage.prompt_tokens,
                    output_tokens=usage.output_tokens,
                    context_limit=context_limit,
                    percent_used=percent_used,
                    tokens_remaining=max(0, context_limit - usage.total_tokens),
                    turns=turns,
                    gc_threshold=gc_threshold,
                    gc_strategy=gc_strategy,
                    gc_target_percent=gc_target_percent,
                    gc_continuous_mode=gc_continuous_mode,
                ))

        def gc_threshold_callback(percent_used: float, threshold: float) -> None:
            server.emit(SystemMessageEvent(
                message=f"Context usage ({percent_used:.1f}%) exceeds threshold ({threshold}%). GC will run after this turn.",
                style="warning",
            ))

        # Capture logging context for propagation into model thread
        from server.session_logging import (
            get_logging_context, set_logging_context, clear_logging_context,
        )
        _log_ctx = get_logging_context()

        def model_thread():
            # Propagate session logging context so plugin logger calls
            # are routed to per-session log files.
            if _log_ctx.get('session_id') and _log_ctx.get('workspace_path'):
                set_logging_context(
                    session_id=_log_ctx['session_id'],
                    client_id=_log_ctx.get('client_id'),
                    workspace_path=_log_ctx['workspace_path'],
                    session_env=_log_ctx.get('session_env'),
                )
            server._model_running = True
            try:
                # Run in workspace context so file operations use client's CWD
                # Also apply session env so provider/tools can access session-specific config
                with server._with_session_env(), server._in_workspace():
                    server._jaato.send_message(
                        prompt,
                        on_output=output_callback,
                        on_usage_update=usage_update_callback,
                        on_gc_threshold=gc_threshold_callback,
                    )

                    # Update context usage
                    if server._jaato:
                        usage = server._jaato.get_context_usage()
                        context_limit = server._jaato.get_context_limit()
                        # Get GC config from main agent state
                        gc_threshold = None
                        gc_strategy = None
                        gc_target_percent = None
                        gc_continuous_mode = False
                        if "main" in server._agents:
                            gc_threshold = server._agents["main"].gc_threshold
                            gc_strategy = server._agents["main"].gc_strategy
                            gc_target_percent = server._agents["main"].gc_target_percent
                            gc_continuous_mode = server._agents["main"].gc_continuous_mode
                        server.emit(ContextUpdatedEvent(
                            agent_id="main",
                            total_tokens=usage.get('total_tokens', 0),
                            prompt_tokens=usage.get('prompt_tokens', 0),
                            output_tokens=usage.get('output_tokens', 0),
                            context_limit=context_limit,
                            percent_used=usage.get('percent_used', 0),
                            tokens_remaining=usage.get('tokens_remaining', 0),
                            turns=usage.get('turns', 0),
                            gc_threshold=gc_threshold,
                            gc_strategy=gc_strategy,
                            gc_target_percent=gc_target_percent,
                            gc_continuous_mode=gc_continuous_mode,
                        ))

            except KeyboardInterrupt:
                server.emit(SystemMessageEvent(
                    message="Interrupted",
                    style="warning",
                ))
            except Exception as e:
                server.emit(ErrorEvent(
                    error=str(e),
                    error_type=type(e).__name__,
                ))
            finally:
                server._model_running = False
                server._model_thread = None
                # Emit "idle" if waiting for user input, "done" if truly finished
                status = "idle" if server._waiting_for_channel_input else "done"
                server.emit(AgentStatusChangedEvent(
                    agent_id="main",
                    status=status,
                ))
                clear_logging_context()

        self._model_thread = threading.Thread(target=model_thread, daemon=True)
        self._model_thread.start()

    def respond_to_permission(self, request_id: str, response: str,
                              edited_arguments: Optional[Dict[str, Any]] = None) -> None:
        """Respond to a permission request.

        Args:
            request_id: The permission request ID.
            response: The response (y, n, a, never, etc.).
            edited_arguments: Optional edited tool arguments (when response is "e"
                and the client handled editing locally).
        """
        if self._pending_permission_request_id != request_id:
            self.emit(ErrorEvent(
                error=f"Unknown permission request: {request_id}",
                error_type="StateError",
            ))
            return

        # Store edited arguments before putting response in queue so the
        # edit_callback can retrieve them synchronously
        if edited_arguments is not None:
            self._pending_edited_arguments = edited_arguments

        self._channel_input_queue.put(response)

    def respond_to_clarification(self, request_id: str, response: str) -> None:
        """Respond to a clarification question.

        Args:
            request_id: The clarification request ID.
            response: The user's answer.
        """
        if self._pending_clarification_request_id != request_id:
            self.emit(ErrorEvent(
                error=f"Unknown clarification request: {request_id}",
                error_type="StateError",
            ))
            return

        self._channel_input_queue.put(response)

    def respond_to_reference_selection(self, request_id: str, response: str) -> None:
        """Respond to a reference selection request.

        Args:
            request_id: The reference selection request ID.
            response: The user's selection (e.g., "1,3,4", "all", "none").
        """
        if self._pending_reference_selection_request_id != request_id:
            self.emit(ErrorEvent(
                error=f"Unknown reference selection request: {request_id}",
                error_type="StateError",
            ))
            return

        self._channel_input_queue.put(response)

    def has_pending_mid_turn_prompt(self) -> bool:
        """Check if there are pending mid-turn prompts.

        Returns:
            True if there are queued prompts.
        """
        return not self._mid_turn_prompt_queue.empty()

    def get_pending_mid_turn_prompt(self) -> Optional[str]:
        """Get the next pending mid-turn prompt if available.

        Returns:
            The prompt text, or None if no prompts are queued.
        """
        try:
            prompt = self._mid_turn_prompt_queue.get_nowait()
            # Emit event that the prompt is being injected
            self.emit(MidTurnPromptInjectedEvent(text=prompt))
            return prompt
        except queue.Empty:
            return None

    def clear_mid_turn_prompts(self) -> int:
        """Clear all pending mid-turn prompts.

        Returns:
            Number of prompts cleared.
        """
        count = 0
        while not self._mid_turn_prompt_queue.empty():
            try:
                self._mid_turn_prompt_queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count

    def _find_plugin_for_command(self, command: str) -> Any:
        """Find the plugin that provides a user command.

        Args:
            command: The command name to find.

        Returns:
            The plugin instance or None if not found.
        """
        if not self._jaato:
            return None

        runtime = self._jaato.get_runtime()
        registry = runtime.registry
        if not registry:
            return None

        # Search exposed plugins for the command
        for plugin_name in registry.list_exposed():
            plugin = registry.get_plugin(plugin_name)
            if plugin and hasattr(plugin, 'get_user_commands'):
                for cmd in plugin.get_user_commands():
                    if cmd.name == command:
                        return plugin

        # Also check permission plugin
        perm = runtime.permission_plugin
        if perm and hasattr(perm, 'get_user_commands'):
            for cmd in perm.get_user_commands():
                if cmd.name == command:
                    return perm

        return None

    def stop(self) -> bool:
        """Stop current operation.

        Returns:
            True if stop was initiated.
        """
        if self._jaato and self._jaato.is_processing:
            return self._jaato.stop()
        return False

    def execute_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        """Execute a command.

        Args:
            command: Command name (e.g., 'model', 'save', 'resume').
            args: Command arguments.

        Returns:
            Command result dict.
        """
        if not self._jaato:
            return {"error": "Client not initialized"}

        user_commands = self._jaato.get_user_commands()
        if command not in user_commands:
            return {"error": f"Unknown command: {command}"}

        cmd = user_commands[command]
        raw_args = " ".join(args)
        parsed_args = parse_command_args(cmd, raw_args)

        # Special handling for save command
        if command.lower() == "save":
            parsed_args["user_inputs"] = self._original_inputs.copy()

        # Find and configure plugin output callback for real-time output
        plugin = self._find_plugin_for_command(command)
        if plugin and hasattr(plugin, 'set_output_callback'):
            # Create callback that emits events
            def output_callback(source: str, text: str, mode: str) -> None:
                self.emit(AgentOutputEvent(
                    text=text,
                    source=source,
                    mode=mode,
                ))
            plugin.set_output_callback(output_callback)

        try:
            result, shared = self._jaato.execute_user_command(command, parsed_args)

            # After memory commands, push updated memory list for completion cache
            # (must run before HelpLines early return so memory list/help also refresh)
            if command.lower() == "memory":
                mem_plugin = self._find_plugin_for_command("memory")
                if mem_plugin and hasattr(mem_plugin, 'get_memory_metadata'):
                    self.emit(MemoryListEvent(memories=mem_plugin.get_memory_metadata()))

            # Handle HelpLines result - emit HelpTextEvent for pager display
            if isinstance(result, HelpLines):
                self.emit(HelpTextEvent(lines=result.lines))
                return {"result": "help displayed"}

            # Handle model change
            if command.lower() == "model" and isinstance(result, dict):
                if result.get("success") and result.get("current_model"):
                    self._model_name = result["current_model"]
                    self.emit(SystemMessageEvent(
                        message=f"Model changed to: {self._model_name}",
                        style="info",
                    ))

            # Handle permission status change
            if command.lower() == "permissions":
                self.emit_permission_status()

            # Handle auth completion - if auth was pending and user ran the matching auth command
            if self._auth_pending and self._auth_plugin_command and command.lower() == self._auth_plugin_command.lower():
                self._check_auth_completion()

            return result if isinstance(result, dict) else {"result": str(result)}

        except Exception as e:
            return {"error": str(e)}

        finally:
            # Clear output callback
            if plugin and hasattr(plugin, 'set_output_callback'):
                plugin.set_output_callback(None)

    def clear_history(self) -> None:
        """Clear conversation history."""
        if self._jaato:
            self._jaato.reset_session()
        self._original_inputs = []
        if "main" in self._agents:
            self._agents["main"].history = []
            self._agents["main"].turn_accounting = []
            self._agents["main"].context_usage = {}

        self.emit(SystemMessageEvent(
            message="History cleared",
            style="info",
        ))

    # =========================================================================
    # Getters
    # =========================================================================

    @property
    def is_processing(self) -> bool:
        """Check if model is currently processing."""
        return self._model_running

    @property
    def is_waiting_for_input(self) -> bool:
        """Check if waiting for permission/clarification input."""
        return self._waiting_for_channel_input

    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self._model_name

    @property
    def model_provider(self) -> str:
        """Get current model provider."""
        return self._model_provider

    def get_agents(self) -> Dict[str, AgentState]:
        """Get all tracked agents."""
        return self._agents.copy()

    def get_history(self, agent_id: str = "main") -> List[Any]:
        """Get conversation history for an agent."""
        if agent_id in self._agents:
            return self._agents[agent_id].history
        return []

    def get_turn_accounting(self, agent_id: str = "main") -> List[Dict]:
        """Get turn accounting for an agent."""
        if agent_id in self._agents:
            return self._agents[agent_id].turn_accounting
        return []

    def get_context_usage(self, agent_id: str = "main") -> Dict[str, Any]:
        """Get context usage for an agent."""
        if agent_id in self._agents:
            return self._agents[agent_id].context_usage
        return {}

    def get_available_commands(self) -> Dict[str, str]:
        """Get available commands with descriptions."""
        if not self._jaato:
            return {}
        user_commands = self._jaato.get_user_commands()
        return {name: cmd.description for name, cmd in user_commands.items()}

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools with status."""
        if not self.registry:
            return []
        return self.registry.get_tool_status()

    def get_tool_status(self) -> List[Dict[str, Any]]:
        """Get tool status for state snapshot.

        Returns list of {name, description, enabled, plugin}.
        """
        if not self.registry:
            return []
        # Use registry's tool status which includes enabled/disabled info
        return self.registry.get_tool_status()

    def get_available_models(self) -> List[str]:
        """Get available model names for completion.

        Returns list of model name strings.
        """
        if not self._jaato:
            return []
        # Get model completions for the "select" subcommand to get actual model names
        # (calling with [] returns subcommands like "list", "select" instead)
        try:
            completions = self._jaato.get_model_completions(["select"])
            return [c.value if hasattr(c, 'value') else str(c) for c in completions]
        except Exception:
            return []

    # =========================================================================
    # Cleanup
    # =========================================================================

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.registry:
            self.registry.unexpose_all()
        if self.permission_plugin:
            self.permission_plugin.shutdown()

    def _trace(self, msg: str) -> None:
        """Write trace message for debugging (goes to daemon log)."""
        logger.debug(msg)

    def _get_auth_plugin_for_provider(self, provider_name: str):
        """Get the authentication plugin for a provider, if available.

        Args:
            provider_name: Provider name (e.g., 'anthropic', 'google_genai')

        Returns:
            The auth plugin instance, or None if not available.
        """
        # Map provider names to their auth plugin names
        auth_plugin_map = {
            "anthropic": "anthropic_auth",
            "github_models": "github_auth",
        }

        plugin_name = auth_plugin_map.get(provider_name)
        if not plugin_name:
            return None

        return self.registry.get_plugin(plugin_name)

    def _check_auth_completion(self) -> None:
        """Check if auth has been completed and finish initialization if so."""
        if not self._auth_pending:
            return

        self._trace(f"[auth] Checking if auth is now complete... (workspace={self._workspace_path})")

        # Try to verify auth again (use session env and workspace context for credentials)
        try:
            with self._with_session_env(), self._in_workspace():
                import os
                self._trace(f"[auth] Current working directory: {os.getcwd()}")
                auth_ok = self._jaato.verify_auth(allow_interactive=False)
            if auth_ok:
                self._trace("[auth] Auth completed successfully, finishing initialization...")
                self._auth_pending = False
                self._auth_plugin_command = None

                # Complete the remaining initialization steps that were skipped
                self._emit_init_progress("Verifying authentication", "done", 4, 6)

                # Step 5: Configure tools (use session env and workspace context for plugin config)
                self._emit_init_progress("Configuring tools", "running", 5, 6)
                with self._with_session_env(), self._in_workspace():
                    self._jaato.configure_tools(self.registry, self.permission_plugin, self.ledger)

                    gc_result = load_gc_from_file()
                gc_threshold = None
                gc_strategy = None
                gc_target_percent = None
                gc_continuous_mode = False
                if gc_result:
                    gc_plugin, gc_config = gc_result
                    self._jaato.set_gc_plugin(gc_plugin, gc_config)
                    gc_threshold = gc_config.threshold_percent
                    gc_target_percent = gc_config.target_percent
                    gc_continuous_mode = gc_config.continuous_mode
                    gc_strategy = getattr(gc_plugin, 'name', 'gc')
                    if gc_strategy.startswith('gc_'):
                        gc_strategy = gc_strategy[3:]

                # Set up instruction budget callback and emit initial events
                session = self._jaato.get_session()
                if session:
                    server = self

                    def instruction_budget_callback(snapshot: dict):
                        server.emit(InstructionBudgetEvent(
                            agent_id=snapshot.get('agent_id', 'main'),
                            budget_snapshot=snapshot,
                        ))

                    session.set_instruction_budget_callback(instruction_budget_callback)

                    # Emit initial budget snapshot
                    if session.instruction_budget:
                        self.emit(InstructionBudgetEvent(
                            agent_id=session.agent_id,
                            budget_snapshot=session.instruction_budget.snapshot(),
                        ))

                self._emit_init_progress("Configuring tools", "done", 5, 6)

                # Step 6: Set up session
                self._emit_init_progress("Setting up session", "running", 6, 6)
                self._setup_session_plugin()
                self._setup_agent_hooks()
                self._setup_permission_hooks()
                self._setup_clarification_hooks()
                self._setup_reference_selection_hooks()
                self._setup_plan_hooks()
                self._setup_queue_channels()
                self._create_main_agent()
                if "main" in self._agents and gc_threshold is not None:
                    self._agents["main"].gc_threshold = gc_threshold
                    self._agents["main"].gc_strategy = gc_strategy
                    self._agents["main"].gc_target_percent = gc_target_percent
                    self._agents["main"].gc_continuous_mode = gc_continuous_mode

                # Emit initial context update so toolbar shows correct usage
                if self._jaato:
                    usage = self._jaato.get_context_usage()
                    self.emit(ContextUpdatedEvent(
                        agent_id="main",
                        total_tokens=usage.get('total_tokens', 0),
                        prompt_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('output_tokens', 0),
                        context_limit=usage.get('context_limit', 128000),
                        percent_used=usage.get('percent_used', 0.0),
                        tokens_remaining=usage.get('tokens_remaining', 128000),
                        turns=usage.get('turns', 0),
                        gc_threshold=gc_threshold,
                        gc_strategy=gc_strategy,
                        gc_target_percent=gc_target_percent,
                        gc_continuous_mode=gc_continuous_mode,
                    ))

                self._emit_init_progress("Setting up session", "done", 6, 6)

                self.emit(SystemMessageEvent(
                    message="Authentication successful. Session is now ready.",
                    style="success",
                ))
                self.emit(SystemMessageEvent(
                    message=f"Connected to {self._model_provider}/{self._model_name}",
                    style="info",
                ))

                # Notify session_manager to emit session info
                if self._on_auth_complete:
                    self._on_auth_complete()
            else:
                self._trace("[auth] Auth still pending")
        except Exception as e:
            self._trace(f"[auth] Auth check failed: {e}")
            # Emit error so user knows what happened
            self.emit(SystemMessageEvent(
                message=f"Auth verification failed: {e}",
                style="error",
            ))
