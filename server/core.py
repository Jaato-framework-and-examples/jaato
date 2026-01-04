"""JaatoServer - Core logic for multi-client support.

This module extracts the non-UI logic from RichClient into a reusable
server that can be driven by different frontends (TUI, WebSocket, HTTP).

The server emits events for all state changes, allowing clients to
subscribe and render appropriately.
"""

import contextlib
import logging
import os
import sys
import pathlib
import queue
import threading
import tempfile
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
from shared.plugins.session import create_plugin as create_session_plugin, load_session_config
from shared.plugins.base import parse_command_args
from shared.plugins.gc import load_gc_from_file

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
    PermissionRequestedEvent,
    PermissionResolvedEvent,
    ClarificationRequestedEvent,
    ClarificationQuestionEvent,
    ClarificationResolvedEvent,
    ReferenceSelectionRequestedEvent,
    ReferenceSelectionResolvedEvent,
    ReferenceSelectionResponseRequest,
    PlanUpdatedEvent,
    PlanClearedEvent,
    ContextUpdatedEvent,
    TurnCompletedEvent,
    SystemMessageEvent,
    InitProgressEvent,
    ErrorEvent,
    SessionInfoEvent,
    SessionDescriptionUpdatedEvent,
    SendMessageRequest,
    PermissionResponseRequest,
    ClarificationResponseRequest,
    StopRequest,
    CommandRequest,
    MidTurnPromptQueuedEvent,
    MidTurnPromptInjectedEvent,
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
        self._pending_clarification_request_id: Optional[str] = None
        self._pending_reference_selection_request_id: Optional[str] = None

        # Queue for mid-turn prompts (messages sent while model is running)
        self._mid_turn_prompt_queue: queue.Queue[str] = queue.Queue()

        # Background model thread
        self._model_thread: Optional[threading.Thread] = None
        self._model_running: bool = False

        # Model info
        self._model_provider: str = ""
        self._model_name: str = ""

        # Trace log path
        self._trace_path = os.path.join(tempfile.gettempdir(), "jaato_server_trace.log")

        # Terminal width for formatting (default 80)
        self._terminal_width: int = 80

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

    @contextlib.contextmanager
    def _in_workspace(self):
        """Context manager to temporarily change to the workspace directory.

        This is thread-safe - it saves and restores the current directory
        for the calling thread. Uses a lock to prevent race conditions
        when multiple threads try to change directory simultaneously.
        """
        if not self._workspace_path:
            yield
            return

        original_cwd = os.getcwd()
        try:
            os.chdir(self._workspace_path)
            logger.debug(f"Changed to workspace: {self._workspace_path}")
            yield
        finally:
            os.chdir(original_cwd)
            logger.debug(f"Restored to: {original_cwd}")

    def _update_plugin_workspace(self, path: Optional[str]) -> None:
        """Update workspace-aware plugins with the new workspace path.

        This notifies plugins like LSP and MCP that need to find config files
        relative to the client's working directory.
        """
        if not path or not hasattr(self, 'registry') or not self.registry:
            return

        # Update LSP plugin if registered
        lsp_plugin = self.registry.get_plugin('lsp')
        if lsp_plugin and hasattr(lsp_plugin, 'set_workspace_path'):
            lsp_plugin.set_workspace_path(path)
            logger.debug(f"Updated LSP plugin workspace_path to {path}")

        # Update MCP plugin if registered
        mcp_plugin = self.registry.get_plugin('mcp')
        if mcp_plugin and hasattr(mcp_plugin, 'set_workspace_path'):
            mcp_plugin.set_workspace_path(path)
            logger.debug(f"Updated MCP plugin workspace_path to {path}")

    # =========================================================================
    # Event Emission
    # =========================================================================

    def emit(self, event: Event) -> None:
        """Emit an event to all subscribed clients."""
        self._on_event(event)

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for clients."""
        self._on_event = callback

    def emit_current_state(
        self,
        emit_fn: Optional[EventCallback] = None,
        skip_session_info: bool = False
    ) -> None:
        """Emit current agent state to a specific client or all clients.

        This is useful when a client attaches to an existing session and needs
        to receive the current agent state that was emitted before they connected.

        Args:
            emit_fn: Optional callback to emit to a specific client.
                     If None, uses the default event callback (broadcast).
            skip_session_info: If True, skip emitting SessionInfoEvent (caller will send it).
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

        # Emit AgentCreatedEvent for all existing agents
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
        total_steps = 5

        # Step 1: Load configuration
        self._emit_init_progress("Loading configuration", "running", 1, total_steps)

        # Read session's env file without modifying global os.environ permanently
        # This allows each session to have its own config
        from dotenv import dotenv_values
        session_env = dotenv_values(self.env_file) if self.env_file else {}

        def get_config(key: str) -> Optional[str]:
            """Get config value from session env, falling back to global env."""
            return session_env.get(key) or os.environ.get(key)

        # Temporarily apply session env vars for provider initialization
        # Save original values to restore later
        original_env = {}
        for key, value in session_env.items():
            if value is not None:
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

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
            self._emit_init_progress("Connecting to model provider", "running", 2, total_steps)
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
        finally:
            # Restore original environment variables
            for key, orig_value in original_env.items():
                if orig_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = orig_value

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
            # Pass workspace_path and session_id to LSP and MCP plugins so they:
            # 1. Load config files from the client's workspace, not the server's cwd
            # 2. Include session_id in log messages for multi-session debugging
            "lsp": {
                "workspace_path": self._workspace_path,
                "session_id": self._session_id,
            },
            "mcp": {
                "workspace_path": self._workspace_path,
                "session_id": self._session_id,
            },
        }
        self.registry.expose_all(plugin_configs)
        self.todo_plugin = self.registry.get_plugin("todo")

        # Wire up artifact tracker with registry for cross-plugin access (LSP integration)
        artifact_tracker = self.registry.get_plugin("artifact_tracker")
        if artifact_tracker and hasattr(artifact_tracker, 'set_plugin_registry'):
            artifact_tracker.set_plugin_registry(self.registry)
            self._trace("artifact_tracker wired with registry (LSP integration enabled)")
        else:
            self._trace("artifact_tracker not found or missing set_plugin_registry - LSP integration disabled")

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

        # Step 4: Configure tools
        self._emit_init_progress("Configuring tools", "running", 4, total_steps)
        self._jaato.configure_tools(self.registry, self.permission_plugin, self.ledger)

        gc_result = load_gc_from_file()
        if gc_result:
            gc_plugin, gc_config = gc_result
            self._jaato.set_gc_plugin(gc_plugin, gc_config)
        self._emit_init_progress("Configuring tools", "done", 4, total_steps)

        # Step 5: Set up session
        self._emit_init_progress("Setting up session", "running", 5, total_steps)
        self._setup_session_plugin()
        self._setup_agent_hooks()
        self._setup_permission_hooks()
        self._setup_clarification_hooks()
        self._setup_reference_selection_hooks()
        self._setup_plan_hooks()
        self._setup_queue_channels()
        self._create_main_agent()
        self._emit_init_progress("Setting up session", "done", 5, total_steps)

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
                server.emit(AgentOutputEvent(
                    agent_id=agent_id,
                    source=source,
                    text=text,
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
                server.emit(ContextUpdatedEvent(
                    agent_id=agent_id,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    context_limit=context_limit,
                    percent_used=percent_used,
                    tokens_remaining=max(0, context_limit - total_tokens),
                    turns=turns,
                ))

            def on_agent_history_updated(self, agent_id, history):
                if agent_id in server._agents:
                    server._agents[agent_id].history = history

            def on_tool_call_start(self, agent_id, tool_name, tool_args, call_id=None):
                server.emit(ToolCallStartEvent(
                    agent_id=agent_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    call_id=call_id,
                ))

            def on_tool_call_end(self, agent_id, tool_name, success, duration_seconds,
                                 error_message=None, call_id=None):
                server.emit(ToolCallEndEvent(
                    agent_id=agent_id,
                    tool_name=tool_name,
                    call_id=call_id,
                    success=success,
                    duration_seconds=duration_seconds,
                    error_message=error_message,
                ))

            def on_tool_output(self, agent_id, call_id, chunk):
                server.emit(ToolOutputEvent(
                    agent_id=agent_id,
                    call_id=call_id,
                    chunk=chunk,
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
                                    tool_args: dict, response_options: list):
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

            # Get formatted prompt lines (with diff for file edits)
            prompt_lines = None
            format_hint = None
            if hasattr(server.permission_plugin, 'get_formatted_prompt'):
                try:
                    prompt_lines, format_hint = server.permission_plugin.get_formatted_prompt(
                        tool_name, tool_args or {}, "ipc"
                    )
                except Exception:
                    pass  # Fall back to client-side formatting

            server.emit(PermissionRequestedEvent(
                request_id=request_id,
                tool_name=tool_name,
                tool_args=tool_args or {},
                response_options=options_dicts,
                prompt_lines=prompt_lines,
                format_hint=format_hint,
            ))

        def on_permission_resolved(tool_name: str, request_id: str,
                                   granted: bool, method: str):
            server._pending_permission_request_id = None
            server._waiting_for_channel_input = False
            server.emit(PermissionResolvedEvent(
                request_id=request_id,
                tool_name=tool_name,
                granted=granted,
                method=method,
            ))

        self.permission_plugin.set_permission_hooks(
            on_requested=on_permission_requested,
            on_resolved=on_permission_resolved,
        )

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
            server.emit(ClarificationRequestedEvent(
                request_id=request_id,
                tool_name=tool_name,
                context_lines=prompt_lines,
            ))

        def on_clarification_resolved(tool_name: str):
            server._pending_clarification_request_id = None
            server._waiting_for_channel_input = False
            server.emit(ClarificationResolvedEvent(
                request_id=server._pending_clarification_request_id or "",
                tool_name=tool_name,
            ))

        def on_question_displayed(tool_name: str, question_index: int,
                                  total_questions: int, question_lines: list):
            server.emit(ClarificationQuestionEvent(
                request_id=server._pending_clarification_request_id or "",
                question_index=question_index,
                total_questions=total_questions,
                question_type="free_text",  # Default, could be enhanced
                question_text="\n".join(question_lines),
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
                request_id=request_id,
                tool_name=tool_name,
                prompt_lines=prompt_lines,
            ))

        def on_selection_resolved(tool_name: str, selected_ids: list):
            request_id = server._pending_reference_selection_request_id or ""
            server._pending_reference_selection_request_id = None
            server._waiting_for_channel_input = False
            server.emit(ReferenceSelectionResolvedEvent(
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
                steps.append({
                    'content': step.get('description', ''),
                    'status': step.get('status', 'pending'),
                    'active_form': step.get('active_form'),
                })
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
                )

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
            # Queue the message for mid-turn injection instead of returning an error
            self._mid_turn_prompt_queue.put(text)
            queue_size = self._mid_turn_prompt_queue.qsize()
            self.emit(MidTurnPromptQueuedEvent(
                text=text,
                position_in_queue=queue_size - 1,
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
                server.emit(ContextUpdatedEvent(
                    agent_id="main",
                    total_tokens=usage.total_tokens,
                    prompt_tokens=usage.prompt_tokens,
                    output_tokens=usage.output_tokens,
                    context_limit=context_limit,
                    percent_used=percent_used,
                    tokens_remaining=max(0, context_limit - usage.total_tokens),
                    turns=turns,
                ))

        def gc_threshold_callback(percent_used: float, threshold: float) -> None:
            server.emit(SystemMessageEvent(
                message=f"Context usage ({percent_used:.1f}%) exceeds threshold ({threshold}%). GC will run after this turn.",
                style="warning",
            ))

        def model_thread():
            server._model_running = True
            try:
                # Set up mid-turn prompt callback to check the server's queue
                session = server._jaato.get_session()
                session.set_mid_turn_prompt_callback(
                    lambda: server.get_pending_mid_turn_prompt()
                )

                # Run in workspace context so file operations use client's CWD
                with server._in_workspace():
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
                        server.emit(ContextUpdatedEvent(
                            agent_id="main",
                            total_tokens=usage.get('total_tokens', 0),
                            prompt_tokens=usage.get('prompt_tokens', 0),
                            output_tokens=usage.get('output_tokens', 0),
                            context_limit=context_limit,
                            percent_used=usage.get('percent_used', 0),
                            tokens_remaining=usage.get('tokens_remaining', 0),
                            turns=usage.get('turns', 0),
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
                # Clear mid-turn prompt callback
                if server._jaato:
                    session = server._jaato.get_session()
                    session.set_mid_turn_prompt_callback(None)

                server._model_running = False
                server._model_thread = None
                server.emit(AgentStatusChangedEvent(
                    agent_id="main",
                    status="done",
                ))

        self._model_thread = threading.Thread(target=model_thread, daemon=True)
        self._model_thread.start()

    def respond_to_permission(self, request_id: str, response: str) -> None:
        """Respond to a permission request.

        Args:
            request_id: The permission request ID.
            response: The response (y, n, a, never, etc.).
        """
        if self._pending_permission_request_id != request_id:
            self.emit(ErrorEvent(
                error=f"Unknown permission request: {request_id}",
                error_type="StateError",
            ))
            return

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

            # Handle model change
            if command.lower() == "model" and isinstance(result, dict):
                if result.get("success") and result.get("current_model"):
                    self._model_name = result["current_model"]
                    self.emit(SystemMessageEvent(
                        message=f"Model changed to: {self._model_name}",
                        style="info",
                    ))

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
        # Get model completions and extract just the names
        try:
            completions = self._jaato.get_model_completions([])
            return [c[0] if isinstance(c, tuple) else str(c) for c in completions]
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
        """Write trace message for debugging."""
        try:
            with open(self._trace_path, "a") as f:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass
