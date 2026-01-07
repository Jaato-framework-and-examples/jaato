"""JaatoSession - Per-agent conversation session.

Provides isolated conversation state for an agent (main or subagent),
while sharing resources from the parent JaatoRuntime.
"""

import json
import os
import queue
import re
import tempfile
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .ai_tool_runner import ToolExecutor
from .retry_utils import with_retry, RequestPacer, RetryCallback, RetryConfig
from .token_accounting import TokenLedger
from .plugins.base import UserCommand, OutputCallback
from .plugins.gc import GCConfig, GCPlugin, GCResult, GCTriggerReason
from .plugins.gc.utils import estimate_history_tokens
from .plugins.session import SessionPlugin, SessionConfig, SessionState, SessionInfo
from .plugins.model_provider.base import UsageUpdateCallback, GCThresholdCallback
from .plugins.model_provider.types import (
    Attachment,
    CancelledException,
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    ToolResult,
    ToolSchema,
)

if TYPE_CHECKING:
    from .jaato_runtime import JaatoRuntime
    from .plugins.model_provider.base import ModelProviderPlugin
    from .plugins.subagent.ui_hooks import AgentUIHooks
    from .plugins.telemetry import TelemetryPlugin

# Import framework instruction for tool result injection
from .jaato_runtime import _TASK_COMPLETION_INSTRUCTION

# Pattern to match @references in prompts
AT_REFERENCE_PATTERN = re.compile(r'@([\w./\-]+(?:\.\w+)?)')


class ActivityPhase(Enum):
    """Activity phases for tracking what a session is doing.

    Used to help parent agents/UIs understand whether a session is
    actively working vs genuinely idle.
    """
    IDLE = "idle"                        # Waiting for input, ready to receive messages
    WAITING_FOR_LLM = "waiting_for_llm"  # Request sent, awaiting cloud response
    STREAMING = "streaming"              # Receiving tokens from LLM
    EXECUTING_TOOL = "executing_tool"    # Running a tool


class JaatoSession:
    """Per-agent conversation session.

    A session represents an isolated conversation with its own:
    - Model selection
    - Tool configuration (can be a subset of runtime's tools)
    - Conversation history
    - System instructions
    - Turn accounting

    Sessions share the runtime's resources (registry, permissions, ledger)
    but maintain independent state.

    Usage:
        # Created via runtime.create_session()
        session = runtime.create_session(
            model="gemini-2.5-flash",
            tools=["cli", "web_search"],
            system_instructions="You are a research assistant."
        )

        # Use the session
        response = session.send_message("Search for Python tutorials")
        history = session.get_history()
    """

    def __init__(
        self,
        runtime: 'JaatoRuntime',
        model: str,
        provider_name: Optional[str] = None
    ):
        """Initialize a session.

        Note: Use runtime.create_session() instead of calling this directly.

        Args:
            runtime: Parent JaatoRuntime providing shared resources.
            model: Model name to use for this session.
            provider_name: Optional provider override for cross-provider sessions.
                          If specified, uses a different AI provider than the runtime default.
        """
        self._runtime = runtime
        self._model_name = model
        self._provider_name_override = provider_name

        # Provider for this session (created during configure())
        self._provider: Optional['ModelProviderPlugin'] = None

        # Tool configuration
        self._executor: Optional[ToolExecutor] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._system_instruction: Optional[str] = None
        self._tool_plugins: Optional[List[str]] = None  # Plugin names for this session

        # Per-turn token accounting
        self._turn_accounting: List[Dict[str, int]] = []

        # User commands for this session
        self._user_commands: Dict[str, UserCommand] = {}

        # Context garbage collection
        self._gc_plugin: Optional[GCPlugin] = None
        self._gc_config: Optional[GCConfig] = None
        self._gc_history: List[GCResult] = []

        # Session persistence
        self._session_plugin: Optional[SessionPlugin] = None
        self._session_config: Optional[SessionConfig] = None

        # Agent type context (for permission checks)
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None

        # UI hooks for agent lifecycle events
        self._ui_hooks: Optional['AgentUIHooks'] = None
        self._agent_id: str = "main"  # Unique ID for this agent

        # Retry notification callback (client-configurable)
        self._on_retry: Optional[RetryCallback] = None

        # Request pacing (proactive rate limiting)
        # Reads AI_REQUEST_INTERVAL from env (default: 0 = disabled)
        self._pacer = RequestPacer()

        # Cancellation support
        self._cancel_token: Optional[CancelToken] = None
        self._parent_cancel_token: Optional[CancelToken] = None  # For parent→child propagation
        self._is_running: bool = False
        self._use_streaming: bool = True  # Enable streaming by default if provider supports it

        # Activity phase tracking (for parent agents/UIs to understand what we're doing)
        self._activity_phase: ActivityPhase = ActivityPhase.IDLE
        self._phase_started_at: Optional[datetime] = None
        # Disable model notifications about cancellation by default - they cause
        # the model to hallucinate "interruptions" on subsequent turns
        self._notify_model_on_cancel: bool = False

        # Proactive GC tracking
        self._gc_threshold_crossed: bool = False  # Set when threshold crossed during streaming
        self._gc_threshold_callback: Optional[GCThresholdCallback] = None

        # Terminal width for formatting (used by enrichment notifications)
        self._terminal_width: int = 80

        # Injection queue for receiving prompts from parent/user mid-turn
        # This is the core mechanism for async agent communication
        self._injection_queue: queue.Queue[str] = queue.Queue()

        # Parent session for output forwarding (subagent -> parent visibility)
        # When set, all output events are forwarded to parent's injection queue
        self._parent_session: Optional['JaatoSession'] = None

        # Callback when a prompt is injected (processed from queue)
        # Used by server to emit MidTurnPromptInjectedEvent
        self._on_prompt_injected: Optional[Callable[[str], None]] = None

        # Current output callback for this turn (used by enrichment to route notifications)
        # Stored here so _enrich_tool_result_dict can pass it to registry.enrich_tool_result()
        # This ensures enrichment notifications go to the correct agent panel even when
        # multiple sessions share the same registry (e.g., subagents)
        self._current_output_callback: Optional['OutputCallback'] = None

        # Turn counter for telemetry
        self._turn_index: int = 0

    @property
    def _telemetry(self) -> 'TelemetryPlugin':
        """Get telemetry plugin from runtime."""
        return self._runtime.telemetry

    def set_terminal_width(self, width: int) -> None:
        """Set the terminal width for formatting.

        This affects enrichment notification formatting.

        Args:
            width: Terminal width in columns.
        """
        self._terminal_width = width

    def _get_trace_prefix(self) -> str:
        """Get the trace prefix including agent context."""
        if self._agent_type == "main":
            return "session:main"
        elif self._agent_name:
            return f"session:subagent:{self._agent_name}"
        else:
            return f"session:subagent:{self._agent_id}"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get("JAATO_TRACE_LOG")
        if not trace_path:
            trace_path = os.environ.get(
                "JAATO_PROVIDER_TRACE",
                os.path.join(tempfile.gettempdir(), "provider_trace.log")
            )
        # Empty string means disabled
        if trace_path == "":
            return
        try:
            prefix = self._get_trace_prefix()
            with open(trace_path, "a") as f:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] [{prefix}] {msg}\n")
                f.flush()
        except (IOError, OSError):
            pass

    @property
    def model_name(self) -> Optional[str]:
        """Get the model name for this session."""
        return self._model_name

    @property
    def runtime(self) -> 'JaatoRuntime':
        """Get the parent runtime."""
        return self._runtime

    @property
    def is_configured(self) -> bool:
        """Check if session is configured and ready."""
        return self._provider is not None

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None
    ) -> None:
        """Set the agent context for permission checks and trace identification.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent (e.g., profile name).
        """
        self._agent_type = agent_type
        self._agent_name = agent_name

        # Update executor permission context if already configured
        if self._executor and self._runtime.permission_plugin:
            context = {"agent_type": agent_type}
            if agent_name:
                context["agent_name"] = agent_name
            self._executor.set_permission_plugin(
                self._runtime.permission_plugin,
                context=context
            )

        # Propagate agent context to provider for trace identification
        if self._provider and hasattr(self._provider, 'set_agent_context'):
            self._provider.set_agent_context(
                agent_type=agent_type,
                agent_name=agent_name,
                agent_id=self._agent_id
            )

    def set_ui_hooks(
        self,
        hooks: 'AgentUIHooks',
        agent_id: str
    ) -> None:
        """Set UI hooks for agent lifecycle events.

        This enables rich terminal UIs to track tool execution and other
        lifecycle events for this session.

        Args:
            hooks: Implementation of AgentUIHooks protocol.
            agent_id: Unique identifier for this agent (e.g., "main", "subagent_1").
        """
        self._ui_hooks = hooks
        self._agent_id = agent_id

    def set_retry_callback(self, callback: Optional[RetryCallback]) -> None:
        """Set callback for retry notifications.

        Clients can use this to control how retry messages are delivered:
        - Simple interactive client: Don't set (uses console print)
        - Rich client: Set callback to route to queue/status bar/etc.

        Args:
            callback: Function called on each retry attempt.
                Signature: (message: str, attempt: int, max_attempts: int, delay: float) -> None
                Set to None to revert to console output.

        Example:
            # Route retries to a queue for non-disruptive display
            session.set_retry_callback(
                lambda msg, att, max_att, delay: status_queue.put(msg)
            )
        """
        self._on_retry = callback

    def set_parent_session(self, parent: Optional['JaatoSession']) -> None:
        """Set parent session for output forwarding.

        When set, all output events from this session are forwarded to the
        parent session's injection queue. This enables parent agents to
        monitor and react to their subagents' activities in real-time.

        The forwarding is one level only - each parent sees only its
        direct children, not grandchildren.

        Args:
            parent: The parent session to forward output to, or None to disable.

        Example:
            # In SubagentPlugin when creating a subagent session:
            subagent_session.set_parent_session(self._parent_session)
        """
        self._parent_session = parent

    def set_prompt_injected_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set callback for when a prompt is processed from the injection queue.

        This callback is invoked when a queued prompt is about to be processed
        (injected into the conversation). The server uses this to emit
        MidTurnPromptInjectedEvent to notify the client UI.

        Args:
            callback: Function called with the prompt text when injected.
        """
        self._on_prompt_injected = callback

    def inject_prompt(self, text: str) -> None:
        """Inject a prompt into this agent's queue.

        The prompt will be processed at the next yield point in the chat loop
        (e.g., after tool execution, after model response). This enables:
        - User mid-turn input to main agent
        - Parent agent sending messages to subagent
        - Subagent returning results to parent

        This method is thread-safe and can be called from any thread.

        Args:
            text: The prompt text to inject.

        Example:
            # Parent sending guidance to subagent
            subagent_session.inject_prompt("Focus on the authentication module")

            # Subagent returning result to parent
            parent_session.inject_prompt("[SUBAGENT agent_id=researcher COMPLETED]\\nFound 3 issues")
        """
        self._trace(f"INJECT_PROMPT: agent_id={self._agent_id}, queue_size_before={self._injection_queue.qsize()}, text={text[:50]}...")
        self._injection_queue.put(text)
        self._trace(f"INJECT_PROMPT: queue_size_after={self._injection_queue.qsize()}")

    def _forward_to_parent(self, event_type: str, content: str) -> None:
        """Forward an event to the parent session.

        Forwards all event types so the parent model can monitor subagent progress
        and react in real-time (e.g., provide guidance via send_to_subagent).

        These messages are NOT echoed in the parent's output panel - the check in
        _check_and_handle_mid_turn_prompt skips output for [SUBAGENT ...] messages.
        The subagent's output is visible in its own UI panel via UI hooks.

        Args:
            event_type: Type of event:
                - MODEL_OUTPUT: Text the subagent is generating
                - TOOL_CALL: Tool the subagent is calling
                - TOOL_OUTPUT: Output from subagent's tool execution
                - COMPLETED: Subagent finished its task
                - ERROR: Subagent encountered an error
                - CANCELLED: Subagent was cancelled
                - CLARIFICATION_REQUESTED: Subagent needs clarification from parent
                - PERMISSION_REQUESTED: Subagent needs permission approval from parent
            content: Event content/payload.
        """
        if self._parent_session:
            message = f"[SUBAGENT agent_id={self._agent_id} event={event_type}]\n{content}"
            self._parent_session.inject_prompt(message)

    # ==================== Cancellation Support ====================

    @property
    def is_running(self) -> bool:
        """Check if a message is currently being processed.

        Returns:
            True if send_message() is in progress, False otherwise.
        """
        return self._is_running

    @property
    def activity_phase(self) -> ActivityPhase:
        """Get the current activity phase.

        Returns:
            The current ActivityPhase (IDLE, WAITING_FOR_LLM, STREAMING, EXECUTING_TOOL).
        """
        return self._activity_phase

    @property
    def phase_started_at(self) -> Optional[datetime]:
        """Get when the current activity phase started.

        Returns:
            Datetime when the current phase began, or None if IDLE.
        """
        return self._phase_started_at

    @property
    def phase_duration_seconds(self) -> Optional[float]:
        """Get how long we've been in the current phase.

        Returns:
            Duration in seconds, or None if IDLE (no active phase).
        """
        if self._phase_started_at is None:
            return None
        return (datetime.now() - self._phase_started_at).total_seconds()

    def _set_activity_phase(self, phase: ActivityPhase) -> None:
        """Set the current activity phase (internal use).

        Args:
            phase: The new activity phase.
        """
        self._activity_phase = phase
        self._phase_started_at = datetime.now() if phase != ActivityPhase.IDLE else None

    def request_stop(self) -> bool:
        """Request cancellation of the current message processing.

        If a message is being processed, signals the cancel token to stop.
        The message loop will check this token and exit gracefully.

        Returns:
            True if a cancellation was requested (message was running),
            False if no message was running.

        Note:
            Cancellation is cooperative - it may not be immediate.
            The current streaming chunk will complete before stopping.
        """
        if self._cancel_token and self._is_running:
            self._cancel_token.cancel()
            return True
        return False

    def set_streaming_enabled(self, enabled: bool) -> None:
        """Enable or disable streaming mode.

        When enabled (default), the session uses streaming APIs for
        real-time output and better cancellation support.

        Args:
            enabled: True to use streaming, False for batched responses.
        """
        self._use_streaming = enabled

    def set_parent_cancel_token(self, token: CancelToken) -> None:
        """Set a parent cancel token for cancellation propagation.

        When set, this session will check both its own cancel token
        and the parent token. If the parent is cancelled, this session
        will also stop - enabling automatic parent→child propagation.

        Args:
            token: The parent session's cancel token.
        """
        self._parent_cancel_token = token

    def _is_cancelled(self) -> bool:
        """Check if this session or its parent has been cancelled.

        Returns:
            True if either this session's token or parent token is cancelled.
        """
        if self._cancel_token and self._cancel_token.is_cancelled:
            return True
        if self._parent_cancel_token and self._parent_cancel_token.is_cancelled:
            return True
        return False

    @property
    def supports_stop(self) -> bool:
        """Check if the current provider supports mid-turn cancellation.

        Stop capability requires both streaming support and provider
        implementation of cancellation handling.

        Returns:
            True if stop is supported, False otherwise.
        """
        if not self._provider:
            return False
        # Check if provider has supports_stop method and it returns True
        if hasattr(self._provider, 'supports_stop'):
            return self._provider.supports_stop()
        # Fallback: if streaming is supported, stop is supported
        if hasattr(self._provider, 'supports_streaming'):
            return self._provider.supports_streaming()
        return False

    def configure(
        self,
        tools: Optional[List[str]] = None,
        system_instructions: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Configure the session with tools and instructions.

        Args:
            tools: Optional list of plugin names to expose. If None, uses all
                   exposed plugins from the runtime's registry.
            system_instructions: Optional additional system instructions.
            plugin_configs: Optional per-plugin configuration overrides.
                           Plugins will be re-initialized with these configs.
        """
        # Store tool plugin names
        self._tool_plugins = tools

        # Re-initialize plugins with session-specific configs if provided
        if plugin_configs and self._runtime.registry:
            for plugin_name, config in plugin_configs.items():
                if tools is None or plugin_name in tools:
                    try:
                        # Inject agent_name into plugin config for trace logging
                        if self._agent_name and "agent_name" not in config:
                            config = {**config, "agent_name": self._agent_name}
                        # expose_tool with new config will re-initialize
                        self._runtime.registry.expose_tool(plugin_name, config)
                    except Exception as e:
                        print(f"Warning: Failed to configure plugin '{plugin_name}': {e}")

        # Create provider for this session (with optional provider override)
        self._provider = self._runtime.create_provider(
            self._model_name,
            provider_name=self._provider_name_override
        )

        # Propagate agent context to provider for trace identification
        if hasattr(self._provider, 'set_agent_context'):
            self._provider.set_agent_context(
                agent_type=self._agent_type,
                agent_name=self._agent_name,
                agent_id=self._agent_id
            )

        # Create executor
        self._executor = ToolExecutor(ledger=self._runtime.ledger)

        # Get tool schemas and executors from runtime
        self._tools = self._runtime.get_tool_schemas(tools)
        executors = self._runtime.get_executors(tools)

        # Register executors
        for name, fn in executors.items():
            self._executor.register(name, fn)

        # Set registry for auto-background support
        if self._runtime.registry:
            self._executor.set_registry(self._runtime.registry)

        # Set permission plugin with agent context
        if self._runtime.permission_plugin:
            context = {"agent_type": self._agent_type}
            if self._agent_name:
                context["agent_name"] = self._agent_name
            self._executor.set_permission_plugin(
                self._runtime.permission_plugin,
                context=context
            )

        # Set this session as parent for subagent plugin (for cancellation propagation)
        if self._runtime.registry:
            subagent_plugin = self._runtime.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_parent_session'):
                subagent_plugin.set_parent_session(self)

        # Build system instructions
        self._system_instruction = self._runtime.get_system_instructions(
            plugin_names=tools,
            additional=system_instructions
        )

        # Store user commands
        if self._runtime.registry:
            self._user_commands = {}
            for cmd in self._runtime.registry.get_exposed_user_commands():
                self._user_commands[cmd.name] = cmd

        # Register built-in model command
        self._register_model_command()

        # Register built-in telepathy tool (share_context)
        self._register_telepathy_tool()

        # Create provider session
        self._create_provider_session()

    def _create_provider_session(
        self,
        history: Optional[List[Message]] = None
    ) -> None:
        """Create or recreate the provider session.

        Args:
            history: Optional initial conversation history.
        """
        if not self._provider:
            return

        self._provider.create_session(
            system_instruction=self._system_instruction,
            tools=self._tools,
            history=history
        )

    def refresh_tools(self) -> None:
        """Refresh tools from the runtime.

        Call this after enabling/disabling tools in the registry to update
        the session's tool configuration. Preserves conversation history.
        """
        if not self._provider or not self._executor:
            return

        # Refresh runtime's cache first
        self._runtime.refresh_tool_cache()

        # Get updated tool schemas and executors from runtime
        self._tools = self._runtime.get_tool_schemas(self._tool_plugins)
        executors = self._runtime.get_executors(self._tool_plugins)

        # Clear and re-register executors
        self._executor.clear_executors()
        for name, fn in executors.items():
            self._executor.register(name, fn)

        # Re-register the model command executor
        self._executor.register("model", self._execute_model_command)

        # Re-register session plugin executors if available
        if self._session_plugin and hasattr(self._session_plugin, 'get_executors'):
            for name, fn in self._session_plugin.get_executors().items():
                self._executor.register(name, fn)

        # Add session plugin tool schemas if available
        if self._session_plugin and hasattr(self._session_plugin, 'get_tool_schemas'):
            session_schemas = self._session_plugin.get_tool_schemas()
            if session_schemas:
                self._tools = list(self._tools) if self._tools else []
                self._tools.extend(session_schemas)

        # Recreate provider session with updated tools (preserve history)
        history = self.get_history()
        self._create_provider_session(history)

    def _register_model_command(self) -> None:
        """Register the built-in model command for listing and switching models."""
        from .plugins.base import CommandParameter

        # Define the command with subcommand parameter
        model_cmd = UserCommand(
            name="model",
            description="Manage models: list, select <name>",
            share_with_model=False,
            parameters=[
                CommandParameter(
                    name="subcommand",
                    description="Subcommand: list, select",
                    required=False
                ),
                CommandParameter(
                    name="model_name",
                    description="Model name (for select)",
                    required=False
                )
            ]
        )

        # Register command
        self._user_commands["model"] = model_cmd

        # Register executor
        if self._executor:
            self._executor.register("model", self._execute_model_command)

    def _execute_model_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the model command.

        Subcommands:
            list   - Show available models and current model
            select - Switch to a different model

        Args:
            args: Command arguments with 'subcommand' and optionally 'model_name'.

        Returns:
            Dict with command result.
        """
        subcommand = args.get("subcommand", "").lower()
        model_name = args.get("model_name")

        # No subcommand - show help
        if not subcommand:
            return {
                "current_model": self._model_name,
                "subcommands": {
                    "list": "Show available models",
                    "select <name>": "Switch to a different model"
                }
            }

        # List subcommand
        if subcommand == "list":
            # Use session's provider if available (faster, no new API connection)
            if self._provider and hasattr(self._provider, 'list_models'):
                models = self._provider.list_models()
            else:
                models = self._runtime.list_available_models(
                    provider_name=self._provider_name_override
                )
            return {
                "current_model": self._model_name,
                "available_models": models
            }

        # Select subcommand
        if subcommand == "select":
            if not model_name:
                return {
                    "error": "Model name required",
                    "usage": "model select <name>",
                    "hint": "Use 'model list' to see available models"
                }

            available = self._runtime.list_available_models(
                provider_name=self._provider_name_override
            )
            if model_name not in available:
                return {
                    "error": f"Model '{model_name}' not found",
                    "available_models": available
                }

            # Preserve current history
            history = self.get_history()

            # Update model name
            old_model = self._model_name
            self._model_name = model_name

            # Create new provider for the new model (preserving provider override)
            self._provider = self._runtime.create_provider(
                model_name,
                provider_name=self._provider_name_override
            )

            # Propagate agent context to new provider for trace identification
            if hasattr(self._provider, 'set_agent_context'):
                self._provider.set_agent_context(
                    agent_type=self._agent_type,
                    agent_name=self._agent_name,
                    agent_id=self._agent_id
                )

            # Recreate session with existing history
            self._create_provider_session(history=history)

            return {
                "success": True,
                "previous_model": old_model,
                "current_model": model_name,
                "history_preserved": True,
                "message": f"Switched from {old_model} to {model_name}"
            }

        # Unknown subcommand
        return {
            "error": f"Unknown subcommand: {subcommand}",
            "valid_subcommands": ["list", "select"]
        }

    def _register_telepathy_tool(self) -> None:
        """Register the built-in share_context tool for agent communication.

        This tool allows any agent (main or subagent) to share structured
        context with its parent agent. It's a native session capability,
        not tied to any specific plugin.
        """
        # Only register if we have a parent session (subagents can share with parent)
        # Main agent can also use this to share with subagents via the subagent plugin
        share_context_schema = ToolSchema(
            name='share_context',
            description=(
                'Share context from your memory with your parent agent. '
                'Use this to transfer knowledge without the parent needing to '
                're-read files or re-execute tools.\n\n'
                'CRITICAL: Share the COMPLETE file content, not summaries or excerpts. '
                'The parent needs the full content to work with it. Never omit content '
                '"for brevity" - that defeats the purpose of this tool.\n\n'
                'IMPORTANT: Do NOT re-read files before sharing. Use your memory of files '
                'you have already read. Copy the full content from your context.\n\n'
                'Use this to:\n'
                '- Share complete file contents you have already read\n'
                '- Share your analysis or findings\n'
                '- Share relevant facts you have discovered'
            ),
            parameters={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "object",
                        "description": (
                            "Files to share from your memory. Keys are file paths, "
                            "values are the COMPLETE file content from your context. "
                            "Do NOT summarize or omit content - share the full text."
                        ),
                        "additionalProperties": {"type": "string"}
                    },
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Key findings, facts, or conclusions to share. "
                            "These should be insights from your analysis."
                        )
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "Free-form context, analysis, guidance, or explanation "
                            "to help the parent agent understand the shared context."
                        )
                    }
                },
                "required": []
            }
        )

        # Add tool schema to session tools
        if self._tools is None:
            self._tools = []
        self._tools = list(self._tools)
        self._tools.append(share_context_schema)

        # Register executor
        if self._executor:
            self._executor.register("share_context", self._execute_share_context)

    def _execute_share_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the share_context tool.

        Shares structured context with the parent agent.

        Args:
            args: Tool arguments containing:
                - files: Optional dict of path -> content from memory
                - findings: Optional list of key findings
                - notes: Optional free-form notes

        Returns:
            Status dict indicating success or error.
        """
        files = args.get('files', {})
        findings = args.get('findings', [])
        notes = args.get('notes', '')

        # Check if there's anything to share
        if not files and not findings and not notes:
            return {
                'success': False,
                'error': 'No context to share. Provide at least one of: files, findings, notes.'
            }

        # Check if we have a parent to share with
        if not self._parent_session:
            return {
                'success': False,
                'error': 'No parent session available. This agent may be the main agent.'
            }

        # Format the context
        formatted_context = self._format_shared_context(files, findings, notes)

        try:
            # Use same pattern as subagent communication: inject if busy, send if idle
            if self._parent_session.is_running:
                # Parent is busy - queue for mid-turn processing
                self._parent_session.inject_prompt(formatted_context)
                return {
                    'success': True,
                    'status': 'queued',
                    'message': 'Context queued for parent. Will be processed at next yield point.',
                    'shared': {
                        'files': list(files.keys()) if files else [],
                        'findings_count': len(findings) if findings else 0,
                        'has_notes': bool(notes)
                    }
                }

            # Parent is idle - this shouldn't normally happen (subagent runs while parent waits)
            # But handle it gracefully by injecting anyway
            self._parent_session.inject_prompt(formatted_context)
            return {
                'success': True,
                'status': 'sent',
                'message': 'Context sent to parent.',
                'shared': {
                    'files': list(files.keys()) if files else [],
                    'findings_count': len(findings) if findings else 0,
                    'has_notes': bool(notes)
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to share context: {str(e)}'
            }

    def _format_shared_context(
        self,
        files: Dict[str, str],
        findings: List[str],
        notes: str
    ) -> str:
        """Format shared context for injection into parent's conversation.

        Args:
            files: Dict of file path -> content from memory
            findings: List of key findings
            notes: Free-form notes

        Returns:
            Formatted string for injection with instructions.
        """
        parts = []

        # Add instruction prefix so the receiving agent knows to use this content
        if files:
            parts.append(
                "CONTEXT FROM SUBAGENT: The following files and findings are shared from the subagent's memory. "
                "DO NOT re-read these files - use the content provided below directly."
            )
            parts.append("")

        parts.append('<shared_context from_agent="subagent">')

        if files:
            parts.append('<files>')
            for path, content in files.items():
                parts.append(f'<file path="{path}">')
                parts.append(content)
                parts.append('</file>')
            parts.append('</files>')

        if findings:
            parts.append('<findings>')
            for finding in findings:
                parts.append(f'  - {finding}')
            parts.append('</findings>')

        if notes:
            parts.append('<notes>')
            parts.append(notes)
            parts.append('</notes>')

        parts.append('</shared_context>')
        return '\n'.join(parts)

    def get_model_completions(self, args: List[str]) -> List['CommandCompletion']:
        """Get completions for the model command.

        Args:
            args: Arguments typed so far.

        Returns:
            List of CommandCompletion objects.
        """
        from .plugins.base import CommandCompletion

        # No args yet - show subcommands
        if not args:
            return [
                CommandCompletion(value="list", description="Show available models"),
                CommandCompletion(value="select", description="Switch to a model"),
            ]

        subcommand = args[0].lower() if args else ""

        # Completing subcommand
        if len(args) == 1:
            subcommands = [
                ("list", "Show available models"),
                ("select", "Switch to a model"),
            ]
            return [
                CommandCompletion(value=cmd, description=desc)
                for cmd, desc in subcommands
                if cmd.startswith(subcommand)
            ]

        # Completing model name for 'select' subcommand
        if subcommand == "select" and len(args) >= 2:
            prefix = args[1] if len(args) > 1 else ""
            models = self._runtime.list_available_models(
                provider_name=self._provider_name_override
            )
            if prefix:
                models = [m for m in models if m.startswith(prefix)]
            return [CommandCompletion(value=m, description="") for m in sorted(models)]

        return []

    def send_message(
        self,
        message: str,
        on_output: Optional[OutputCallback] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_gc_threshold: Optional[GCThresholdCallback] = None
    ) -> str:
        """Send a message to the model.

        Args:
            message: The user's message text.
            on_output: Optional callback for real-time output.
                Signature: (source: str, text: str, mode: str) -> None
            on_usage_update: Optional callback for real-time token usage.
                Signature: (usage: TokenUsage) -> None
            on_gc_threshold: Optional callback when GC threshold is crossed.
                Signature: (percent_used: float, threshold: float) -> None

        Returns:
            The final model response text.

        Raises:
            RuntimeError: If session is not configured.
        """
        if not self._provider:
            raise RuntimeError("Session not configured. Call configure() first.")

        self._trace(f"SESSION_SEND_MESSAGE len={len(message)} streaming={self._use_streaming}")

        # Increment turn counter
        self._turn_index += 1

        # Wrap entire turn with telemetry span
        with self._telemetry.turn_span(
            session_id=self._agent_id,
            agent_type=self._agent_type,
            agent_name=self._agent_name,
            turn_index=self._turn_index,
        ) as turn_span:
            # Check and perform GC if needed (pre-send)
            if self._gc_plugin and self._gc_config and self._gc_config.check_before_send:
                self._maybe_collect_before_send()

            # Reset proactive GC tracking for this turn
            self._gc_threshold_crossed = False
            self._gc_threshold_callback = on_gc_threshold

            # Store output callback for this turn so enrichment can use it directly
            # This avoids the race condition where concurrent sessions overwrite
            # the shared registry callback
            self._current_output_callback = on_output

            # Wrap usage callback to check GC threshold
            wrapped_usage_callback = self._wrap_usage_callback_with_gc_check(on_usage_update)

            # Set output callback on registry BEFORE prompt enrichment
            # so enrichment notifications are visible to the user
            if self._runtime.registry and on_output:
                self._runtime.registry.set_output_callback(on_output, self._terminal_width)

            # Run prompt enrichment if registry is available
            processed_message = self._enrich_and_clean_prompt(message)

            try:
                response = self._run_chat_loop(processed_message, on_output, wrapped_usage_callback)
                turn_span.set_status_ok()
            except Exception as e:
                turn_span.record_exception(e)
                turn_span.set_status_error(str(e))
                raise

            # Record turn completion attributes
            turn_span.set_attribute("jaato.cancelled", self._is_cancelled())
            turn_span.set_attribute("jaato.streaming", self._use_streaming)

            # Proactive GC: if threshold was crossed during streaming, trigger GC now
            if self._gc_threshold_crossed and self._gc_plugin and self._gc_config:
                self._trace("PROACTIVE_GC: Threshold crossed during streaming, triggering post-turn GC")
                self._maybe_collect_after_turn()

            # Notify session plugin
            self._notify_session_turn_complete()

            return response

    def _wrap_usage_callback_with_gc_check(
        self,
        on_usage_update: Optional[UsageUpdateCallback]
    ) -> Optional[UsageUpdateCallback]:
        """Wrap usage callback to check GC threshold during streaming."""
        if not self._gc_plugin or not self._gc_config:
            return on_usage_update

        def wrapped_callback(usage: TokenUsage) -> None:
            # Check if threshold crossed
            if not self._gc_threshold_crossed and usage.total_tokens > 0:
                context_limit = self.get_context_limit()
                if context_limit > 0:
                    percent_used = (usage.total_tokens / context_limit) * 100
                    threshold = self._gc_config.threshold_percent if self._gc_config else 80.0

                    if percent_used >= threshold:
                        self._gc_threshold_crossed = True
                        self._trace(f"PROACTIVE_GC: Threshold crossed ({percent_used:.1f}% >= {threshold}%)")

                        # Notify via callback if provided
                        if self._gc_threshold_callback:
                            self._gc_threshold_callback(percent_used, threshold)

            # Call original callback if provided
            if on_usage_update:
                on_usage_update(usage)

        return wrapped_callback

    def _maybe_collect_after_turn(self) -> Optional[GCResult]:
        """Perform GC after turn if threshold was crossed during streaming."""
        if not self._gc_plugin or not self._gc_config:
            return None

        context_usage = self.get_context_usage()
        history = self.get_history()

        # Use THRESHOLD as the reason since it was triggered by threshold crossing
        new_history, result = self._gc_plugin.collect(
            history, context_usage, self._gc_config, GCTriggerReason.THRESHOLD
        )

        if result.success:
            if result.items_collected == 0:
                # GC ran but collected nothing - this is often surprising to users
                self._trace(
                    f"PROACTIVE_GC: WARNING - GC triggered but collected 0 items. "
                    f"Check preserve_recent_turns setting vs actual turn count. "
                    f"Details: {result.details}"
                )
            else:
                self._trace(
                    f"PROACTIVE_GC: Collected {result.items_collected} items, "
                    f"freed {result.tokens_freed} tokens"
                )
            self.reset_session(new_history)
            self._gc_history.append(result)

        return result

    def _enrich_and_clean_prompt(self, prompt: str) -> str:
        """Run prompt through enrichment pipeline and strip @references."""
        enriched_prompt = prompt

        # Run through plugin enrichment pipeline
        if self._runtime.registry:
            result = self._runtime.registry.enrich_prompt(prompt)
            enriched_prompt = result.prompt

        # Strip @references
        return AT_REFERENCE_PATTERN.sub(r'\1', enriched_prompt)

    def _run_chat_loop(
        self,
        message: str,
        on_output: Optional[OutputCallback],
        on_usage_update: Optional[UsageUpdateCallback] = None
    ) -> str:
        """Internal function calling loop with streaming and cancellation support.

        Args:
            message: The user's message text.
            on_output: Optional callback for real-time output.
            on_usage_update: Optional callback for real-time token usage updates.

        Returns:
            The final response text.
        """
        # Set output callback on executor
        if self._executor:
            self._executor.set_output_callback(on_output)

        # Set output callback on registry for enrichment notifications
        if self._runtime.registry and on_output:
            self._runtime.registry.set_output_callback(on_output, self._terminal_width)

        # Initialize cancellation support
        self._cancel_token = CancelToken()
        self._is_running = True
        cancellation_notified = False  # Track if we've already shown cancellation message

        # Track tokens and timing
        turn_start = datetime.now()
        turn_data = {
            'prompt': 0,
            'output': 0,
            'total': 0,
            'start_time': turn_start.isoformat(),
            'end_time': None,
            'duration_seconds': None,
            'function_calls': [],
        }
        response: Optional[ProviderResponse] = None

        # Wrap usage callback to also update turn_data during streaming
        # This ensures we capture token values even if streaming is cancelled
        # Always enabled for internal turn tracking, regardless of external callback
        def usage_callback_with_turn_tracking(usage: TokenUsage) -> None:
            if usage.total_tokens > 0:
                turn_data['prompt'] = usage.prompt_tokens
                turn_data['output'] = usage.output_tokens
                turn_data['total'] = usage.total_tokens
            if on_usage_update:
                on_usage_update(usage)

        wrapped_usage_callback = usage_callback_with_turn_tracking

        # Determine if we should use streaming
        use_streaming = (
            self._use_streaming and
            self._provider and
            hasattr(self._provider, 'supports_streaming') and
            self._provider.supports_streaming()
        )

        try:
            # Check for cancellation before starting (including parent)
            if self._is_cancelled():
                msg = "[Cancelled before start]"
                if on_output:
                    on_output("system", msg, "write")
                return msg

            # Proactive rate limiting: wait if needed before request
            self._pacer.pace()

            # Set activity phase: we're about to wait for LLM response
            self._set_activity_phase(ActivityPhase.WAITING_FOR_LLM)

            # Send message (streaming or batched)
            if use_streaming:
                # Track whether we've sent the first chunk (to use "write" vs "append")
                first_chunk_sent = False

                # Streaming callback that routes to on_output and forwards to parent
                def streaming_callback(chunk: str) -> None:
                    nonlocal first_chunk_sent
                    # Transition to STREAMING phase on first chunk
                    if not first_chunk_sent:
                        self._set_activity_phase(ActivityPhase.STREAMING)
                    if on_output:
                        # First chunk uses "write" to start block, subsequent use "append"
                        mode = "append" if first_chunk_sent else "write"
                        self._trace(f"SESSION_OUTPUT mode={mode} len={len(chunk)} preview={repr(chunk[:50])}")
                        on_output("model", chunk, mode)
                        first_chunk_sent = True
                    # Forward model output to parent for real-time visibility
                    self._forward_to_parent("MODEL_OUTPUT", chunk)

                self._trace(f"STREAMING on_usage_update={'set' if wrapped_usage_callback else 'None'}")
                response, _retry_stats = with_retry(
                    lambda: self._provider.send_message_streaming(
                        message,
                        on_chunk=streaming_callback,
                        cancel_token=self._cancel_token,
                        on_usage_update=wrapped_usage_callback
                        # Note: on_function_call is intentionally NOT used here.
                        # The SDK may deliver function calls before preceding text,
                        # which would cause tool trees to appear in wrong positions.
                        # Tool trees are displayed during parts processing instead.
                    ),
                    context="send_message_streaming",
                    on_retry=self._on_retry,
                    cancel_token=self._cancel_token
                )
            else:
                response, _retry_stats = with_retry(
                    lambda: self._provider.send_message(message),
                    context="send_message",
                    on_retry=self._on_retry,
                    cancel_token=self._cancel_token
                )
            self._record_token_usage(response)
            self._accumulate_turn_tokens(response, turn_data)
            self._trace(f"SESSION_STREAMING_COMPLETE parts_count={len(response.parts)} finish={response.finish_reason}")

            # Check for cancellation after initial message (including parent)
            if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                partial_text = response.get_text()
                cancel_msg = "[Generation cancelled]"
                if on_output and not cancellation_notified:
                    self._trace(f"CANCEL_NOTIFY: {cancel_msg} (after initial message)")
                    on_output("system", cancel_msg, "write")
                    cancellation_notified = True
                elif cancellation_notified:
                    self._trace(f"CANCEL_DUPLICATE: {cancel_msg} (after initial message) - already notified!")
                # Notify model of cancellation for context on next turn
                self._notify_model_of_cancellation(cancel_msg, partial_text)
                if partial_text:
                    return f"{partial_text}\n\n{cancel_msg}"
                return cancel_msg

            # Handle function calling loop - process parts in order to support interleaved text/tools
            accumulated_text: List[str] = []
            self._trace(f"SESSION_PARTS_PROCESSING parts_count={len(response.parts)}")

            def get_pending_function_calls() -> List[FunctionCall]:
                """Extract function calls from response.parts."""
                return [p.function_call for p in response.parts if p.function_call]

            def get_all_text() -> str:
                """Concatenate all text from response.parts."""
                texts = [p.text for p in response.parts if p.text]
                return ''.join(texts) if texts else ''

            pending_calls = get_pending_function_calls()

            # Handle edge case: finish_reason indicates tool use but no function calls present
            # This can happen when the model "narrates" intent to call a tool but doesn't emit the call
            tool_use_nudge_attempts = 0
            max_tool_use_nudge_attempts = 3
            while (not pending_calls and response.finish_reason == FinishReason.TOOL_USE
                   and tool_use_nudge_attempts < max_tool_use_nudge_attempts):
                tool_use_nudge_attempts += 1
                self._trace(f"TOOL_USE_WITHOUT_CALL: Model indicated tool use but no function call emitted (attempt {tool_use_nudge_attempts}/{max_tool_use_nudge_attempts})")
                # Inject a prompt to push the model to execute the intended tool call
                nudge_prompt = (
                    "<hidden>Your response indicated TOOL_USE but contained no function call. "
                    "Do NOT describe or re-read files. Execute the tool call you just mentioned directly. "
                    "Your next response MUST start with the function call, not text.</hidden>"
                )
                self._injection_queue.put(nudge_prompt)
                # Process the injected prompt to get a new response
                nudge_response = self._check_and_handle_mid_turn_prompt(
                    use_streaming, on_output, wrapped_usage_callback, turn_data
                )
                if nudge_response:
                    response = nudge_response
                    pending_calls = get_pending_function_calls()
                    self._trace(f"TOOL_USE_NUDGE_RESULT: Got response with {len(pending_calls)} function calls")
                else:
                    # No response from nudge, break to avoid infinite loop
                    self._trace("TOOL_USE_NUDGE_NO_RESPONSE: Nudge did not produce a response")
                    break

            if tool_use_nudge_attempts >= max_tool_use_nudge_attempts and not pending_calls:
                self._trace("TOOL_USE_NUDGE_EXHAUSTED: Max attempts reached, model still not calling tools")

            while pending_calls:
                # Check for cancellation before processing tools (including parent)
                if self._is_cancelled():
                    cancel_msg = "[Cancelled during tool execution]"
                    if on_output and not cancellation_notified:
                        self._trace(f"CANCEL_NOTIFY: {cancel_msg} (before processing tools)")
                        on_output("system", cancel_msg, "write")
                        cancellation_notified = True
                    elif cancellation_notified:
                        self._trace(f"CANCEL_DUPLICATE: {cancel_msg} (before processing tools) - already notified!")
                    # Notify model of cancellation for context on next turn
                    all_text = get_all_text()
                    self._notify_model_of_cancellation(cancel_msg, all_text)
                    if all_text:
                        return f"{all_text}\n\n{cancel_msg}"
                    return cancel_msg

                # Process parts in order - emit text, collect function calls into groups
                current_fc_group: List[FunctionCall] = []
                for idx, part in enumerate(response.parts):
                    # Enhanced trace: show empty text parts (which indicate unknown SDK parts)
                    text_info = "empty" if part.text == "" else bool(part.text) if part.text else None
                    fc_info = part.function_call.name if part.function_call else None
                    self._trace(f"SESSION_PART[{idx}] text={text_info} fc={fc_info}")
                    if part.text:
                        # Before emitting text, execute any pending function calls
                        if current_fc_group:
                            tool_results = self._execute_function_call_group(
                                current_fc_group, turn_data, on_output, cancellation_notified
                            )
                            if self._is_cancelled():
                                cancel_msg = "[Cancelled after tool execution]"
                                if on_output and not cancellation_notified:
                                    on_output("system", cancel_msg, "write")
                                self._notify_model_of_cancellation(cancel_msg)
                                return cancel_msg
                            # Send tool results and get continuation
                            response = self._send_tool_results_and_continue(
                                tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                            )
                            if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                                partial = get_all_text()
                                cancel_msg = "[Generation cancelled]"
                                if on_output and not cancellation_notified:
                                    on_output("system", cancel_msg, "write")
                                self._notify_model_of_cancellation(cancel_msg, partial)
                                return f"{partial}\n\n{cancel_msg}" if partial else cancel_msg

                            # Check for mid-turn prompts after interleaved tool execution
                            # Only inject if response doesn't have more function calls
                            # (otherwise we'd break the tool_use -> tool_result sequence)
                            response_has_fc = any(p.function_call for p in response.parts)
                            if not response_has_fc:
                                mid_turn_response = self._check_and_handle_mid_turn_prompt(
                                    use_streaming, on_output, wrapped_usage_callback, turn_data
                                )
                                if mid_turn_response:
                                    # Update response to the mid-turn prompt response
                                    response = mid_turn_response
                                    if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                                        partial = get_all_text()
                                        cancel_msg = "[Generation cancelled]"
                                        if on_output and not cancellation_notified:
                                            on_output("system", cancel_msg, "write")
                                        self._notify_model_of_cancellation(cancel_msg, partial)
                                        return f"{partial}\n\n{cancel_msg}" if partial else cancel_msg

                            current_fc_group = []

                        # Emit text (only in non-streaming mode)
                        if not use_streaming:
                            if on_output:
                                on_output("model", part.text, "write")
                            # Forward to parent for visibility
                            self._forward_to_parent("MODEL_OUTPUT", part.text)
                        accumulated_text.append(part.text)

                    elif part.function_call:
                        current_fc_group.append(part.function_call)

                # Execute remaining function calls at end of parts
                if current_fc_group:
                    tool_results = self._execute_function_call_group(
                        current_fc_group, turn_data, on_output, cancellation_notified
                    )
                    if self._is_cancelled():
                        cancel_msg = "[Cancelled after tool execution]"
                        if on_output and not cancellation_notified:
                            on_output("system", cancel_msg, "write")
                        self._notify_model_of_cancellation(cancel_msg)
                        return cancel_msg

                    # Send tool results and get next response
                    response = self._send_tool_results_and_continue(
                        tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                    )
                    if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                        partial = get_all_text()
                        cancel_msg = "[Generation cancelled]"
                        if on_output and not cancellation_notified:
                            on_output("system", cancel_msg, "write")
                        self._notify_model_of_cancellation(cancel_msg, partial)
                        return f"{partial}\n\n{cancel_msg}" if partial else cancel_msg

                    if response.finish_reason not in (FinishReason.STOP, FinishReason.UNKNOWN, FinishReason.TOOL_USE, FinishReason.CANCELLED):
                        import sys
                        print(f"[warning] Model stopped with finish_reason={response.finish_reason}", file=sys.stderr)
                        final_text = get_all_text()
                        if final_text:
                            return f"{final_text}\n\n[Model stopped: {response.finish_reason}]"
                        else:
                            return f"[Model stopped unexpectedly: {response.finish_reason}]"

                    # Check for mid-turn prompts at this natural pause point
                    # Only inject if response doesn't have more function calls
                    # (otherwise we'd break the tool_use -> tool_result sequence)
                    response_has_fc = any(p.function_call for p in response.parts)

                    # Handle edge case: TOOL_USE without function calls after tool results in main loop
                    main_loop_nudge_attempts = 0
                    while (not response_has_fc and response.finish_reason == FinishReason.TOOL_USE
                           and main_loop_nudge_attempts < max_tool_use_nudge_attempts):
                        main_loop_nudge_attempts += 1
                        self._trace(f"TOOL_USE_WITHOUT_CALL: In main loop after tool results, no function call (attempt {main_loop_nudge_attempts}/{max_tool_use_nudge_attempts})")
                        nudge_prompt = (
                            "<hidden>Your response indicated TOOL_USE but contained no function call. "
                            "Do NOT describe or re-read files. Execute the tool call you just mentioned directly. "
                            "Your next response MUST start with the function call, not text.</hidden>"
                        )
                        self._injection_queue.put(nudge_prompt)
                        nudge_response = self._check_and_handle_mid_turn_prompt(
                            use_streaming, on_output, wrapped_usage_callback, turn_data
                        )
                        if nudge_response:
                            response = nudge_response
                            response_has_fc = any(p.function_call for p in response.parts)
                            self._trace(f"TOOL_USE_NUDGE_RESULT: Got response, has_fc={response_has_fc}")
                        else:
                            self._trace("TOOL_USE_NUDGE_NO_RESPONSE: Nudge did not produce a response")
                            break

                    if not response_has_fc:
                        mid_turn_response = self._check_and_handle_mid_turn_prompt(
                            use_streaming, on_output, wrapped_usage_callback, turn_data
                        )
                        if mid_turn_response:
                            # The model responded to the injected prompt - use that response
                            response = mid_turn_response
                            if self._is_cancelled() or response.finish_reason == FinishReason.CANCELLED:
                                partial = get_all_text()
                                cancel_msg = "[Generation cancelled]"
                                if on_output and not cancellation_notified:
                                    on_output("system", cancel_msg, "write")
                                self._notify_model_of_cancellation(cancel_msg, partial)
                                return f"{partial}\n\n{cancel_msg}" if partial else cancel_msg

                # Check for more function calls in the new response
                pending_calls = get_pending_function_calls()

            # Collect any remaining text from the final response
            for part in response.parts:
                if part.text:
                    if not use_streaming:
                        if on_output:
                            on_output("model", part.text, "write")
                        # Forward to parent for visibility
                        self._forward_to_parent("MODEL_OUTPUT", part.text)
                    accumulated_text.append(part.text)

            # Final check for mid-turn prompts before completing the turn
            # This handles prompts that arrived while the model was generating its final response
            self._trace(f"FINAL_MID_TURN_CHECK: Starting drain loop, queue_size={self._injection_queue.qsize()}")
            while True:
                mid_turn_response = self._check_and_handle_mid_turn_prompt(
                    use_streaming, on_output, wrapped_usage_callback, turn_data
                )
                if not mid_turn_response:
                    break

                # Process the response to the injected prompt
                if self._is_cancelled() or mid_turn_response.finish_reason == FinishReason.CANCELLED:
                    cancel_msg = "[Generation cancelled]"
                    if on_output and not cancellation_notified:
                        on_output("system", cancel_msg, "write")
                    final_text = ''.join(accumulated_text) if accumulated_text else ''
                    self._notify_model_of_cancellation(cancel_msg, final_text)
                    return f"{final_text}\n\n{cancel_msg}" if final_text else cancel_msg

                # Collect text from the mid-turn response
                for part in mid_turn_response.parts:
                    if part.text:
                        # Note: In streaming mode, text was already emitted by the callback
                        if not use_streaming:
                            if on_output:
                                on_output("model", part.text, "write")
                            # Forward to parent for visibility
                            self._forward_to_parent("MODEL_OUTPUT", part.text)
                        accumulated_text.append(part.text)

                # Check if the mid-turn response triggered more function calls
                mid_turn_fc = [p.function_call for p in mid_turn_response.parts if p.function_call]
                if mid_turn_fc:
                    # Process these function calls and continue
                    tool_results = self._execute_function_call_group(
                        mid_turn_fc, turn_data, on_output, cancellation_notified
                    )
                    if self._is_cancelled():
                        cancel_msg = "[Cancelled after tool execution]"
                        if on_output and not cancellation_notified:
                            on_output("system", cancel_msg, "write")
                        self._notify_model_of_cancellation(cancel_msg)
                        return cancel_msg

                    response = self._send_tool_results_and_continue(
                        tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                    )

                    # Continue the main loop with this response if it has function calls
                    pending_calls = [p.function_call for p in response.parts if p.function_call]

                    # Handle edge case: TOOL_USE without function calls after tool results
                    inner_nudge_attempts = 0
                    while (not pending_calls and response.finish_reason == FinishReason.TOOL_USE
                           and inner_nudge_attempts < max_tool_use_nudge_attempts):
                        inner_nudge_attempts += 1
                        self._trace(f"TOOL_USE_WITHOUT_CALL: After tool results, no function call (attempt {inner_nudge_attempts}/{max_tool_use_nudge_attempts})")
                        nudge_prompt = (
                            "<hidden>Your response indicated TOOL_USE but contained no function call. "
                            "Do NOT describe or re-read files. Execute the tool call you just mentioned directly. "
                            "Your next response MUST start with the function call, not text.</hidden>"
                        )
                        self._injection_queue.put(nudge_prompt)
                        nudge_response = self._check_and_handle_mid_turn_prompt(
                            use_streaming, on_output, wrapped_usage_callback, turn_data
                        )
                        if nudge_response:
                            response = nudge_response
                            pending_calls = [p.function_call for p in response.parts if p.function_call]
                            self._trace(f"TOOL_USE_NUDGE_RESULT: Got response with {len(pending_calls)} function calls")
                        else:
                            self._trace("TOOL_USE_NUDGE_NO_RESPONSE: Nudge did not produce a response")
                            break

                    while pending_calls:
                        # Re-enter the main processing loop for tool calls
                        current_fc_group = []
                        for part in response.parts:
                            if part.text:
                                if not use_streaming:
                                    if on_output:
                                        on_output("model", part.text, "write")
                                    # Forward to parent for visibility
                                    self._forward_to_parent("MODEL_OUTPUT", part.text)
                                accumulated_text.append(part.text)
                            elif part.function_call:
                                current_fc_group.append(part.function_call)

                        if current_fc_group:
                            tool_results = self._execute_function_call_group(
                                current_fc_group, turn_data, on_output, cancellation_notified
                            )
                            if self._is_cancelled():
                                cancel_msg = "[Cancelled after tool execution]"
                                if on_output and not cancellation_notified:
                                    on_output("system", cancel_msg, "write")
                                self._notify_model_of_cancellation(cancel_msg)
                                return cancel_msg

                            response = self._send_tool_results_and_continue(
                                tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                            )

                        pending_calls = [p.function_call for p in response.parts if p.function_call]

                        # Handle edge case: TOOL_USE without function calls in nested loop
                        nested_nudge_attempts = 0
                        while (not pending_calls and response.finish_reason == FinishReason.TOOL_USE
                               and nested_nudge_attempts < max_tool_use_nudge_attempts):
                            nested_nudge_attempts += 1
                            self._trace(f"TOOL_USE_WITHOUT_CALL: In nested loop, no function call (attempt {nested_nudge_attempts}/{max_tool_use_nudge_attempts})")
                            nudge_prompt = (
                                "<hidden>Your response indicated TOOL_USE but contained no function call. "
                                "Do NOT describe or re-read files. Execute the tool call you just mentioned directly. "
                                "Your next response MUST start with the function call, not text.</hidden>"
                            )
                            self._injection_queue.put(nudge_prompt)
                            nudge_response = self._check_and_handle_mid_turn_prompt(
                                use_streaming, on_output, wrapped_usage_callback, turn_data
                            )
                            if nudge_response:
                                response = nudge_response
                                pending_calls = [p.function_call for p in response.parts if p.function_call]
                                self._trace(f"TOOL_USE_NUDGE_RESULT: Got response with {len(pending_calls)} function calls")
                            else:
                                self._trace("TOOL_USE_NUDGE_NO_RESPONSE: Nudge did not produce a response")
                                break

                    # Collect final text from nested response
                    for part in response.parts:
                        if part.text:
                            if not use_streaming:
                                if on_output:
                                    on_output("model", part.text, "write")
                                # Forward to parent for visibility
                                self._forward_to_parent("MODEL_OUTPUT", part.text)
                            accumulated_text.append(part.text)

                # Explicitly continue to check for more queued prompts
                # This ensures we drain the queue even after processing function calls
                continue

            # Safety check: process any prompts that might have been added during the final iteration
            # This handles the race condition where prompts arrive just as the drain loop exits
            final_queue_size = self._injection_queue.qsize()
            if final_queue_size > 0:
                self._trace(f"FINAL_MID_TURN_CHECK: Queue not empty after drain loop! size={final_queue_size}, processing remaining")
                # Process remaining prompts (with a limit to prevent infinite loops)
                safety_iterations = 0
                max_safety_iterations = 10  # Prevent livelock
                while safety_iterations < max_safety_iterations:
                    safety_iterations += 1
                    remaining_response = self._check_and_handle_mid_turn_prompt(
                        use_streaming, on_output, wrapped_usage_callback, turn_data
                    )
                    if not remaining_response:
                        break
                    # Collect text from the response
                    for part in remaining_response.parts:
                        if part.text:
                            if not use_streaming:
                                if on_output:
                                    on_output("model", part.text, "write")
                                self._forward_to_parent("MODEL_OUTPUT", part.text)
                            accumulated_text.append(part.text)
                    # Note: We don't process function calls here to avoid complexity
                    # Any function calls in these responses will be logged but not executed
                    if any(p.function_call for p in remaining_response.parts):
                        self._trace("FINAL_MID_TURN_CHECK: Safety loop response had function calls (not processed)")

            # Forward completion to parent
            final_response = ''.join(accumulated_text) if accumulated_text else ''
            self._forward_to_parent("COMPLETED", final_response)

            return final_response

        except CancelledException:
            # Handle explicit cancellation exception
            # Note: Don't send on_output here - the explicit checks above already do
            self._forward_to_parent("CANCELLED", "Generation cancelled")
            return "[Generation cancelled]"

        except Exception as exc:
            # Route provider errors through output callback before re-raising
            # This ensures errors appear in the UI (queue channel) instead of raw console
            exc_name = type(exc).__name__
            exc_module = type(exc).__module__

            # Check if this is a known provider error (from model_provider plugins)
            is_provider_error = 'model_provider' in exc_module or exc_name in (
                # Anthropic errors
                'AnthropicProviderError', 'APIKeyNotFoundError', 'APIKeyInvalidError',
                'RateLimitError', 'ContextLimitError', 'ModelNotFoundError',
                'OverloadedError', 'UsageLimitError',
                # GitHub Models errors
                'GitHubModelsError', 'TokenNotFoundError', 'TokenInvalidError',
                'TokenPermissionError', 'ModelsDisabledError',
                # Google GenAI errors
                'JaatoAuthError', 'CredentialsNotFoundError', 'CredentialsInvalidError',
                'CredentialsPermissionError', 'ProjectConfigurationError',
            )

            if is_provider_error and on_output:
                # Format error message nicely for the UI
                error_msg = f"[Error] {exc_name}: {str(exc)}"
                on_output("error", error_msg, "write")
                self._trace(f"PROVIDER_ERROR routed to callback: {exc_name}")

            # Forward error to parent for visibility
            self._forward_to_parent("ERROR", f"{exc_name}: {str(exc)}")

            # Re-raise so caller can also handle if needed
            raise

        finally:
            # Record turn end time
            turn_end = datetime.now()
            turn_data['end_time'] = turn_end.isoformat()
            turn_data['duration_seconds'] = (turn_end - turn_start).total_seconds()

            if turn_data['total'] > 0:
                self._turn_accounting.append(turn_data)

            # Clean up cancellation state and activity phase
            self._is_running = False
            self._cancel_token = None
            self._set_activity_phase(ActivityPhase.IDLE)

            # Notify parent that this subagent is now idle
            # This is a hidden status update so the parent knows we're ready for input
            self._forward_to_parent(
                "STATUS",
                f"<hidden>Subagent {self._agent_id} is now idle and ready for input.</hidden>"
            )

    def _execute_function_call_group(
        self,
        function_calls: List[FunctionCall],
        turn_data: Dict[str, Any],
        on_output: Optional[OutputCallback],
        cancellation_notified: bool
    ) -> List[ToolResult]:
        """Execute a group of function calls and return their results."""
        # Set activity phase: we're executing tools
        self._set_activity_phase(ActivityPhase.EXECUTING_TOOL)

        tool_results: List[ToolResult] = []

        for fc in function_calls:
            # Check for cancellation before each tool (including parent)
            if self._is_cancelled():
                break

            name = fc.name
            args = fc.args

            # Forward tool call to parent for visibility
            self._forward_to_parent("TOOL_CALL", f"{name}({json.dumps(args)})")

            # Emit hook: tool starting
            if self._ui_hooks:
                # Signal UI to flush any pending output before displaying tool tree
                # This ensures text appears before tool trees in async/buffered UIs
                if on_output:
                    self._trace(f"SESSION_OUTPUT_FLUSH before tool {name}")
                    on_output("system", "", "flush")
                self._trace(f"SESSION_TOOL_START name={name} call_id={fc.id}")
                self._ui_hooks.on_tool_call_start(
                    agent_id=self._agent_id,
                    tool_name=name,
                    tool_args=args,
                    call_id=fc.id
                )

            fc_start = datetime.now()

            # Determine plugin type for telemetry
            plugin_type = "unknown"
            if self._runtime.registry:
                plugin = self._runtime.registry.get_plugin_for_tool(name)
                if plugin:
                    plugin_type = getattr(plugin, 'plugin_type', type(plugin).__name__)

            # Wrap tool execution with telemetry span
            with self._telemetry.tool_span(
                tool_name=name,
                call_id=fc.id or "",
                plugin_type=plugin_type,
            ) as tool_span:
                if self._executor:
                    # Set up tool output callback for streaming output during execution
                    # Captures _name for use in the closure
                    def tool_output_callback(chunk: str, _call_id=fc.id, _name=name) -> None:
                        if self._ui_hooks and _call_id:
                            self._ui_hooks.on_tool_output(
                                agent_id=self._agent_id,
                                call_id=_call_id,
                                chunk=chunk
                            )
                        # Forward tool output to parent for visibility
                        self._forward_to_parent("TOOL_OUTPUT", f"[{_name}] {chunk}")
                    self._executor.set_tool_output_callback(tool_output_callback)

                    executor_result = self._executor.execute(name, args)

                    # Clear the callback after execution
                    self._executor.set_tool_output_callback(None)
                else:
                    executor_result = (False, {"error": f"No executor registered for {name}"})

                fc_end = datetime.now()

                # Determine success and error message from executor result
                fc_success = True
                fc_error_message = None
                if isinstance(executor_result, tuple) and len(executor_result) == 2:
                    fc_success = executor_result[0]
                    if not fc_success and isinstance(executor_result[1], dict):
                        fc_error_message = executor_result[1].get('error')

                # Record tool execution attributes in telemetry
                fc_duration = (fc_end - fc_start).total_seconds()
                tool_span.set_attribute("jaato.tool.success", fc_success)
                tool_span.set_attribute("jaato.tool.duration_seconds", fc_duration)
                if fc_error_message:
                    tool_span.set_attribute("jaato.tool.error", fc_error_message)
                    tool_span.set_status_error(fc_error_message)
                else:
                    tool_span.set_status_ok()

            # Emit hook: tool ended
            if self._ui_hooks:
                self._ui_hooks.on_tool_call_end(
                    agent_id=self._agent_id,
                    tool_name=name,
                    success=fc_success,
                    duration_seconds=fc_duration,
                    error_message=fc_error_message,
                    call_id=fc.id
                )

            # Record function call timing
            turn_data['function_calls'].append({
                'name': name,
                'start_time': fc_start.isoformat(),
                'end_time': fc_end.isoformat(),
                'duration_seconds': fc_duration,
            })

            # Build ToolResult
            tool_result = self._build_tool_result(fc, executor_result)
            tool_results.append(tool_result)

        return tool_results

    def _send_tool_results_and_continue(
        self,
        tool_results: List[ToolResult],
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any]
    ) -> ProviderResponse:
        """Send tool results back to the model and get the continuation response."""
        # with_retry is already imported at module level from .retry_utils

        # Inject task completion spur into last tool result
        if tool_results:
            last = tool_results[-1]
            result_text = str(last.result) if last.result is not None else ""
            spurred_result = f"{result_text}\n\n<hidden>{_TASK_COMPLETION_INSTRUCTION}</hidden>"
            tool_results = tool_results[:-1] + [
                ToolResult(
                    call_id=last.call_id,
                    name=last.name,
                    result=spurred_result,
                    is_error=last.is_error,
                    attachments=last.attachments
                )
            ]

        # Proactive rate limiting
        self._pacer.pace()

        # Set activity phase: we're waiting for LLM response again
        self._set_activity_phase(ActivityPhase.WAITING_FOR_LLM)

        try:
            return self._do_send_tool_results(
                tool_results, use_streaming, on_output, wrapped_usage_callback, turn_data
            )
        except Exception as e:
            # Check if this is a token limit error
            error_str = str(e).lower()
            if 'token' in error_str and ('exceed' in error_str or 'limit' in error_str or 'maximum' in error_str):
                self._trace(f"TOKEN_LIMIT_ERROR: Tool results too large, retrying with truncated results")
                # Replace oversized results with error messages
                truncated_results = self._truncate_oversized_results(tool_results)
                return self._do_send_tool_results(
                    truncated_results, use_streaming, on_output, wrapped_usage_callback, turn_data
                )
            raise

    def _truncate_oversized_results(self, tool_results: List[ToolResult]) -> List[ToolResult]:
        """Replace oversized tool results with error messages guiding the model to retry."""
        truncated = []
        for tr in tool_results:
            result_str = str(tr.result) if tr.result is not None else ""
            # Estimate tokens (~4 chars per token) - if over 100K tokens, truncate
            estimated_tokens = len(result_str) // 4
            if estimated_tokens > 100000:
                error_msg = (
                    f"[ERROR: Result too large ({estimated_tokens:,} estimated tokens). "
                    f"This exceeds the context window. "
                    f"Use chunked reading with offset/limit parameters, or process in smaller pieces.]"
                    f"\n\n<hidden>{_TASK_COMPLETION_INSTRUCTION}</hidden>"
                )
                truncated.append(ToolResult(
                    call_id=tr.call_id,
                    name=tr.name,
                    result=error_msg,
                    is_error=True,
                    attachments=None  # Drop attachments too
                ))
            else:
                truncated.append(tr)
        return truncated

    def _do_send_tool_results(
        self,
        tool_results: List[ToolResult],
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any]
    ) -> ProviderResponse:
        """Actually send tool results to the provider."""
        if use_streaming:
            # Track first chunk to use "write" for new block, "append" for continuation
            first_chunk_after_tools = [False]  # Use list to allow mutation in closure

            def streaming_callback(chunk: str) -> None:
                if on_output:
                    # First chunk after tool results starts a new block
                    mode = "append" if first_chunk_after_tools[0] else "write"
                    self._trace(f"SESSION_TOOL_RESULT_OUTPUT mode={mode} len={len(chunk)} preview={repr(chunk[:50])}")
                    on_output("model", chunk, mode)
                    first_chunk_after_tools[0] = True

            response, _retry_stats = with_retry(
                lambda: self._provider.send_tool_results_streaming(
                    tool_results,
                    on_chunk=streaming_callback,
                    cancel_token=self._cancel_token,
                    on_usage_update=wrapped_usage_callback
                    # Note: on_function_call is intentionally NOT used here.
                    # See comment in send_message for explanation.
                ),
                context="send_tool_results_streaming",
                on_retry=self._on_retry,
                cancel_token=self._cancel_token
            )
        else:
            response, _retry_stats = with_retry(
                lambda: self._provider.send_tool_results(tool_results),
                context="send_tool_results",
                on_retry=self._on_retry,
                cancel_token=self._cancel_token
            )

        self._record_token_usage(response)
        self._accumulate_turn_tokens(response, turn_data)

        return response

    def _check_and_handle_mid_turn_prompt(
        self,
        use_streaming: bool,
        on_output: Optional[OutputCallback],
        wrapped_usage_callback: Optional[UsageUpdateCallback],
        turn_data: Dict[str, Any]
    ) -> Optional[ProviderResponse]:
        """Check for and handle a pending mid-turn prompt.

        This is called at natural pause points during message processing
        (e.g., after tool execution, after receiving model response).

        Prompts can come from:
        - User input (via inject_prompt from server)
        - Parent agent (via inject_prompt for guidance)
        - Child agents (via inject_prompt for results/output)

        If a mid-turn prompt is pending, this method:
        1. Emits the prompt as user output
        2. Sends it to the model as a new user message
        3. Returns the model's response

        Args:
            use_streaming: Whether to use streaming for the model call.
            on_output: Callback for output events.
            wrapped_usage_callback: Callback for usage updates.
            turn_data: Current turn's data for token tracking.

        Returns:
            The model's response if a prompt was handled, None otherwise.
        """
        # Process queue items, skipping subagent messages until we find a real prompt
        # or the queue is empty. Subagent messages are for logging/visibility only.
        while True:
            # Check internal injection queue
            try:
                prompt = self._injection_queue.get_nowait()
            except queue.Empty:
                self._trace("MID_TURN_PROMPT: Queue empty, returning None")
                return None

            self._trace(f"MID_TURN_PROMPT: Handling injected prompt: {prompt[:100]}...")

            # Notify that prompt is being injected (for UI to remove from pending bar)
            if self._on_prompt_injected:
                self._on_prompt_injected(prompt)

            # Check if this is a subagent message (for logging/visibility, not model prompts)
            # Subagent messages are forwarded for tracing but the model shouldn't respond to them
            # The parent model gets subagent results via spawn_subagent tool response
            is_subagent_message = prompt.startswith("<subagent_event ") or prompt.startswith("[SUBAGENT ")
            self._trace(f"MID_TURN_PROMPT: is_subagent_message={is_subagent_message}, on_output={on_output is not None}, _parent_session={self._parent_session is not None}")

            if is_subagent_message:
                # Subagent messages are for logging/visibility only - don't send to model
                # This prevents duplicate model responses to each forwarded MODEL_OUTPUT chunk
                self._trace(f"MID_TURN_PROMPT: Skipping model call for subagent message (logged only)")
                continue  # Check for more messages in the queue

            # Found a real prompt to process - break out of skip loop
            break

        # Emit the prompt as user/parent output so UI shows it
        if on_output:
            # Use "parent" source if this is a subagent (has parent session),
            # otherwise "user" for main agent receiving user input
            source = "parent" if self._parent_session else "user"
            self._trace(f"MID_TURN_PROMPT: Emitting with source={source}")
            on_output(source, prompt, "write")

        # Proactive rate limiting
        self._pacer.pace()

        # Send the prompt to the model
        if use_streaming:
            first_chunk_sent = [False]

            def streaming_callback(chunk: str) -> None:
                if on_output:
                    mode = "append" if first_chunk_sent[0] else "write"
                    self._trace(f"MID_TURN_RESPONSE mode={mode} len={len(chunk)}")
                    on_output("model", chunk, mode)
                    first_chunk_sent[0] = True

            response, _retry_stats = with_retry(
                lambda: self._provider.send_message_streaming(
                    prompt,
                    on_chunk=streaming_callback,
                    cancel_token=self._cancel_token,
                    on_usage_update=wrapped_usage_callback
                ),
                context="mid_turn_prompt_streaming",
                on_retry=self._on_retry,
                cancel_token=self._cancel_token
            )
        else:
            response, _retry_stats = with_retry(
                lambda: self._provider.send_message(prompt),
                context="mid_turn_prompt",
                on_retry=self._on_retry,
                cancel_token=self._cancel_token
            )

            # Emit response text if not streaming
            if on_output and response.get_text():
                on_output("model", response.get_text(), "write")

        self._record_token_usage(response)
        self._accumulate_turn_tokens(response, turn_data)

        return response

    def _build_tool_result(
        self,
        fc: FunctionCall,
        executor_result: Any
    ) -> ToolResult:
        """Build a ToolResult from executor output."""
        # Executor returns (ok, result_dict) tuple
        if isinstance(executor_result, tuple) and len(executor_result) == 2:
            ok, result_data = executor_result
        else:
            ok = True
            result_data = executor_result

        # Check for multimodal result
        attachments: Optional[List[Attachment]] = None
        if isinstance(result_data, dict) and result_data.get('_multimodal'):
            attachments = self._extract_multimodal_attachments(result_data)
            result_data = {k: v for k, v in result_data.items()
                          if not k.startswith('_multimodal') and k not in ('image_data',)}

        # Build result dict
        if isinstance(result_data, dict):
            result_dict = result_data
        else:
            result_dict = {"result": result_data}

        # Run tool result enrichment (e.g., template extraction)
        if ok and self._runtime.registry:
            result_dict = self._enrich_tool_result_dict(fc.name, result_dict)

        return ToolResult(
            call_id=fc.id,
            name=fc.name,
            result=result_dict,
            is_error=not ok,
            attachments=attachments
        )

    def _enrich_tool_result_dict(
        self,
        tool_name: str,
        result_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run tool result enrichment on tool results.

        Two enrichment modes:
        1. For file-writing tools (writeNewFile, updateFile): Pass the full JSON
           result so enrichers can extract file paths and run diagnostics.
        2. For other tools with large text fields: Enrich individual text fields.

        Note: Passes the session's output callback to enrich_tool_result() so that
        enrichment notifications are routed to the correct agent panel. This is
        critical for concurrent sessions (e.g., subagents running in parallel with
        the parent) that share the same registry.

        Args:
            tool_name: Name of the tool that produced the result.
            result_dict: The result dictionary to enrich.

        Returns:
            Enriched result dictionary.
        """
        enriched_dict = result_dict.copy()

        # Tools that write files - pass full JSON for LSP diagnostics enrichment
        file_writing_tools = {'writeNewFile', 'updateFile', 'lsp_rename_symbol', 'lsp_apply_code_action'}

        if tool_name in file_writing_tools:
            # Pass full result as JSON so LSP can extract file paths
            import json
            result_json = json.dumps(result_dict)
            # Pass session's callback to route notifications to correct agent panel
            enrichment = self._runtime.registry.enrich_tool_result(
                tool_name,
                result_json,
                output_callback=self._current_output_callback,
                terminal_width=self._terminal_width
            )
            if enrichment.result != result_json:
                try:
                    enriched_dict = json.loads(enrichment.result)
                except json.JSONDecodeError:
                    # If enrichment broke JSON, keep original and append as text
                    enriched_dict['_lsp_diagnostics'] = enrichment.result
            return enriched_dict

        # For other tools: enrich large text fields
        text_fields = ('result', 'content', 'stdout', 'output', 'text', 'data')
        min_length = 100

        for field in text_fields:
            if field in enriched_dict:
                value = enriched_dict[field]
                if isinstance(value, str) and len(value) >= min_length:
                    # Pass session's callback to route notifications to correct agent panel
                    enrichment = self._runtime.registry.enrich_tool_result(
                        tool_name,
                        value,
                        output_callback=self._current_output_callback,
                        terminal_width=self._terminal_width
                    )
                    if enrichment.result != value:
                        enriched_dict[field] = enrichment.result

        return enriched_dict

    def _extract_multimodal_attachments(
        self,
        result: Dict[str, Any]
    ) -> Optional[List[Attachment]]:
        """Extract multimodal attachments from a result dict."""
        multimodal_type = result.get('_multimodal_type', 'image')

        if multimodal_type == 'image':
            image_data = result.get('image_data')
            if not image_data:
                return None

            mime_type = result.get('mime_type', 'image/png')
            display_name = result.get('display_name', 'image')

            return [Attachment(
                mime_type=mime_type,
                data=image_data,
                display_name=display_name
            )]

        return None

    def _accumulate_turn_tokens(
        self,
        response: ProviderResponse,
        turn_tokens: Dict[str, int]
    ) -> None:
        """Update token counts from provider response.

        Note: We REPLACE (not sum) because each API response's prompt_tokens
        already includes ALL previous history. The final API call in a turn
        has the complete context usage.

        However, we only replace if values are non-zero, to preserve good values
        when streaming is cancelled mid-turn (which may return zero tokens).
        """
        if response.usage.total_tokens > 0:
            turn_tokens['prompt'] = response.usage.prompt_tokens
            turn_tokens['output'] = response.usage.output_tokens
            turn_tokens['total'] = response.usage.total_tokens

    def _record_token_usage(self, response: ProviderResponse) -> None:
        """Record token usage to ledger if available."""
        if not self._runtime.ledger:
            return

        self._runtime.ledger._record('response', {
            'prompt_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.output_tokens,
            'total_tokens': response.usage.total_tokens,
        })

    def get_history(self) -> List[Message]:
        """Get current conversation history."""
        if not self._provider:
            return []
        return self._provider.get_history()

    def get_turn_accounting(self) -> List[Dict[str, Any]]:
        """Get token usage and timing per turn."""
        return list(self._turn_accounting)

    def get_context_limit(self) -> int:
        """Get the context window limit for the current model."""
        if not self._provider:
            return 1_048_576
        return self._provider.get_context_limit()

    def get_context_usage(self) -> Dict[str, Any]:
        """Get context window usage statistics.

        Note: Each turn's prompt_tokens includes ALL previous history,
        so we use the LAST turn's values (not sum) for context usage.

        When turn_accounting is empty (e.g., after GC reset), we estimate
        tokens from the actual history to avoid incorrectly reporting 0%.
        """
        turn_accounting = self.get_turn_accounting()

        if turn_accounting:
            # Use the last turn's values - prompt includes full history
            last_turn = turn_accounting[-1]
            # The current context is the last turn's prompt + its output
            # (which will become part of the next turn's prompt)
            total_prompt = last_turn.get('prompt', 0)
            total_output = last_turn.get('output', 0)
            total_tokens = last_turn.get('total', 0)
        else:
            # No turn accounting available - estimate from history
            # This happens after GC resets the session with compressed history
            history = self.get_history()
            if history:
                estimated_tokens = estimate_history_tokens(history)
                total_prompt = estimated_tokens
                total_output = 0
                total_tokens = estimated_tokens
            else:
                total_prompt = 0
                total_output = 0
                total_tokens = 0

        context_limit = self.get_context_limit()
        percent_used = (total_tokens / context_limit * 100) if context_limit > 0 else 0
        tokens_remaining = max(0, context_limit - total_tokens)

        return {
            'model': self._model_name or 'unknown',
            'context_limit': context_limit,
            'total_tokens': total_tokens,
            'prompt_tokens': total_prompt,
            'output_tokens': total_output,
            'turns': len(turn_accounting),
            'percent_used': percent_used,
            'tokens_remaining': tokens_remaining,
        }

    def reset_session(self, history: Optional[List[Message]] = None) -> None:
        """Reset the chat session.

        Args:
            history: Optional initial history for the new session.
        """
        self._turn_accounting = []
        self._create_provider_session(history)

    def get_turn_boundaries(self) -> List[int]:
        """Get indices where each turn starts in the history."""
        history = self.get_history()
        boundaries = []

        for i, msg in enumerate(history):
            if msg.role == Role.USER and msg.parts and msg.parts[0].text:
                boundaries.append(i)

        return boundaries

    def revert_to_turn(self, turn_id: int) -> Dict[str, Any]:
        """Revert the conversation to a specific turn."""
        boundaries = self.get_turn_boundaries()
        total_turns = len(boundaries)

        if turn_id < 1:
            raise ValueError(f"Turn ID must be >= 1, got {turn_id}")

        if turn_id > total_turns:
            raise ValueError(f"Turn {turn_id} does not exist. Current session has {total_turns} turn(s).")

        if turn_id == total_turns:
            return {
                'success': True,
                'turns_removed': 0,
                'new_turn_count': total_turns,
                'message': f"Already at turn {turn_id}, no changes made."
            }

        history = self.get_history()

        if turn_id < total_turns:
            truncate_at = boundaries[turn_id]
        else:
            truncate_at = len(history)

        truncated_history = list(history[:truncate_at])
        turns_removed = total_turns - turn_id

        if turn_id <= len(self._turn_accounting):
            self._turn_accounting = self._turn_accounting[:turn_id]

        self._create_provider_session(truncated_history)

        if self._session_plugin and hasattr(self._session_plugin, 'set_turn_count'):
            self._session_plugin.set_turn_count(turn_id)

        return {
            'success': True,
            'turns_removed': turns_removed,
            'new_turn_count': turn_id,
            'message': f"Reverted to turn {turn_id} (removed {turns_removed} turn(s))."
        }

    def get_user_commands(self) -> Dict[str, UserCommand]:
        """Get available user commands."""
        return dict(self._user_commands)

    def execute_user_command(
        self,
        command_name: str,
        args: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, bool]:
        """Execute a user command."""
        if command_name not in self._user_commands:
            raise ValueError(f"Unknown user command: {command_name}")

        if not self._executor:
            raise RuntimeError("Executor not configured.")

        cmd = self._user_commands[command_name]
        args = args or {}

        _ok, result = self._executor.execute(command_name, args)

        if cmd.share_with_model and self._provider:
            self._inject_command_into_history(command_name, args, result)

        return result, cmd.share_with_model

    def _inject_command_into_history(
        self,
        command_name: str,
        args: Dict[str, Any],
        result: Any
    ) -> None:
        """Inject a user command execution into conversation history."""
        current_history = self.get_history()

        user_message = Message(
            role=Role.USER,
            parts=[Part.from_text(f"[User executed command: {command_name}]")]
        )

        result_dict = result if isinstance(result, dict) else {"result": result}
        model_message = Message(
            role=Role.MODEL,
            parts=[Part.from_function_response(ToolResult(
                call_id="",
                name=command_name,
                result=result_dict
            ))]
        )

        new_history = list(current_history) + [user_message, model_message]
        self._create_provider_session(new_history)

    def _notify_model_of_cancellation(self, cancel_msg: str, partial_text: str = '') -> None:
        """Inject cancellation notice into history so model has context.

        This adds a user message noting the cancellation, so on the next turn
        the model understands why the previous response was cut short.

        NOTE: This feature is disabled by default (_notify_model_on_cancel=False)
        because it causes the model to hallucinate "interruptions" on subsequent
        turns, even when the cancellation was internal or expected.

        Args:
            cancel_msg: The cancellation message shown to user.
            partial_text: Any partial response text before cancellation.
        """
        # Skip notification if disabled (default) - prevents model hallucinations
        if not self._notify_model_on_cancel:
            self._trace(f"CANCEL_NOTIFY_SKIP: notifications disabled")
            return

        if not self._provider:
            return

        current_history = self.get_history()

        # Create a note for the model about what happened
        if partial_text:
            note = f"[System: Your previous response was cancelled by the user after: \"{partial_text[:100]}{'...' if len(partial_text) > 100 else ''}\"]"
        else:
            note = "[System: Your previous response was cancelled by the user before any output was generated.]"

        user_message = Message(
            role=Role.USER,
            parts=[Part.from_text(note)]
        )

        new_history = list(current_history) + [user_message]
        self._create_provider_session(new_history)

    def generate(self, prompt: str) -> str:
        """Simple generation without tools."""
        if not self._provider:
            raise RuntimeError("Session not configured.")

        response = self._provider.generate(prompt)
        return response.get_text() or ''

    def send_message_with_parts(
        self,
        parts: List[Part],
        on_output: OutputCallback
    ) -> str:
        """Send a message with custom Part objects."""
        if not self._provider:
            raise RuntimeError("Session not configured.")

        return self._run_chat_loop_with_parts(parts, on_output)

    def _run_chat_loop_with_parts(
        self,
        parts: List[Part],
        on_output: OutputCallback
    ) -> str:
        """Internal function calling loop for multi-part messages."""
        if self._executor:
            self._executor.set_output_callback(on_output)

        turn_start = datetime.now()
        turn_data = {
            'prompt': 0,
            'output': 0,
            'total': 0,
            'start_time': turn_start.isoformat(),
            'end_time': None,
            'duration_seconds': None,
            'function_calls': [],
        }
        response: Optional[ProviderResponse] = None

        try:
            # Proactive rate limiting: wait if needed before request
            self._pacer.pace()

            response, _retry_stats = with_retry(
                lambda: self._provider.send_message_with_parts(parts),
                context="send_message_with_parts",
                on_retry=self._on_retry
            )
            self._record_token_usage(response)
            self._accumulate_turn_tokens(response, turn_data)

            from .plugins.model_provider.types import FinishReason
            if response.finish_reason not in (FinishReason.STOP, FinishReason.UNKNOWN, FinishReason.TOOL_USE):
                import sys
                print(f"[warning] Model stopped with finish_reason={response.finish_reason}", file=sys.stderr)
                if response.text:
                    return f"{response.text}\n\n[Model stopped: {response.finish_reason}]"
                else:
                    return f"[Model stopped unexpectedly: {response.finish_reason}]"

            function_calls = list(response.function_calls) if response.function_calls else []
            while function_calls:
                if response.text and on_output:
                    on_output("model", response.text, "write")

                tool_results: List[ToolResult] = []

                for fc in function_calls:
                    name = fc.name
                    args = fc.args

                    # Emit hook: tool starting
                    if self._ui_hooks:
                        self._ui_hooks.on_tool_call_start(
                            agent_id=self._agent_id,
                            tool_name=name,
                            tool_args=args,
                            call_id=fc.id
                        )

                    fc_start = datetime.now()
                    if self._executor:
                        # Set up tool output callback for streaming output during execution
                        if self._ui_hooks and fc.id:
                            def tool_output_callback(chunk: str, _call_id=fc.id) -> None:
                                self._ui_hooks.on_tool_output(
                                    agent_id=self._agent_id,
                                    call_id=_call_id,
                                    chunk=chunk
                                )
                            self._executor.set_tool_output_callback(tool_output_callback)

                        executor_result = self._executor.execute(name, args)

                        # Clear the callback after execution
                        self._executor.set_tool_output_callback(None)
                    else:
                        executor_result = (False, {"error": f"No executor registered for {name}"})
                    fc_end = datetime.now()

                    # Determine success and error message from executor result
                    fc_success = True
                    fc_error_message = None
                    if isinstance(executor_result, tuple) and len(executor_result) == 2:
                        fc_success = executor_result[0]
                        # Extract error message if tool failed
                        if not fc_success and isinstance(executor_result[1], dict):
                            fc_error_message = executor_result[1].get('error')

                    # Emit hook: tool ended
                    fc_duration = (fc_end - fc_start).total_seconds()
                    if self._ui_hooks:
                        self._ui_hooks.on_tool_call_end(
                            agent_id=self._agent_id,
                            tool_name=name,
                            success=fc_success,
                            duration_seconds=fc_duration,
                            error_message=fc_error_message,
                            call_id=fc.id
                        )

                    turn_data['function_calls'].append({
                        'name': name,
                        'start_time': fc_start.isoformat(),
                        'end_time': fc_end.isoformat(),
                        'duration_seconds': fc_duration,
                    })

                    tool_result = self._build_tool_result(fc, executor_result)
                    tool_results.append(tool_result)

                # Send tool results back (with retry for rate limits)
                self._pacer.pace()  # Proactive rate limiting
                response, _retry_stats = with_retry(
                    lambda: self._provider.send_tool_results(tool_results),
                    context="send_tool_results",
                    on_retry=self._on_retry
                )
                self._record_token_usage(response)
                self._accumulate_turn_tokens(response, turn_data)
                function_calls = list(response.function_calls) if response.function_calls else []

            if response.text and on_output:
                on_output("model", response.text, "write")

            return response.text or ''

        except Exception as exc:
            # Route provider errors through output callback before re-raising
            exc_name = type(exc).__name__
            exc_module = type(exc).__module__

            is_provider_error = 'model_provider' in exc_module or exc_name in (
                'AnthropicProviderError', 'APIKeyNotFoundError', 'APIKeyInvalidError',
                'RateLimitError', 'ContextLimitError', 'ModelNotFoundError',
                'OverloadedError', 'UsageLimitError',
                'GitHubModelsError', 'TokenNotFoundError', 'TokenInvalidError',
                'TokenPermissionError', 'ModelsDisabledError',
                'JaatoAuthError', 'CredentialsNotFoundError', 'CredentialsInvalidError',
                'CredentialsPermissionError', 'ProjectConfigurationError',
            )

            if is_provider_error and on_output:
                error_msg = f"[Error] {exc_name}: {str(exc)}"
                on_output("error", error_msg, "write")
                self._trace(f"PROVIDER_ERROR routed to callback: {exc_name}")

            raise

        finally:
            turn_end = datetime.now()
            turn_data['end_time'] = turn_end.isoformat()
            turn_data['duration_seconds'] = (turn_end - turn_start).total_seconds()

            if turn_data['total'] > 0:
                self._turn_accounting.append(turn_data)

    # ==================== Context Garbage Collection ====================

    def set_gc_plugin(
        self,
        plugin: GCPlugin,
        config: Optional[GCConfig] = None
    ) -> None:
        """Set the GC plugin for context management."""
        self._gc_plugin = plugin
        self._gc_config = config or GCConfig()

    def remove_gc_plugin(self) -> None:
        """Remove the GC plugin."""
        if self._gc_plugin:
            self._gc_plugin.shutdown()
        self._gc_plugin = None
        self._gc_config = None

    def manual_gc(self) -> GCResult:
        """Manually trigger garbage collection."""
        if not self._gc_plugin:
            raise RuntimeError("No GC plugin configured.")
        if not self._gc_config:
            self._gc_config = GCConfig()

        history = self.get_history()
        context_usage = self.get_context_usage()
        self._trace(
            f"MANUAL_GC: triggering manual GC (usage={context_usage.get('percent_used', 0):.1f}%)"
        )

        new_history, result = self._gc_plugin.collect(
            history, context_usage, self._gc_config, GCTriggerReason.MANUAL
        )

        if result.success:
            if result.items_collected == 0:
                self._trace(
                    f"MANUAL_GC: WARNING - GC ran but collected 0 items. "
                    f"Check preserve_recent_turns setting vs actual turn count. "
                    f"Details: {result.details}"
                )
            else:
                self._trace(
                    f"MANUAL_GC: collected {result.items_collected} items, "
                    f"freed {result.tokens_freed} tokens"
                )
            self.reset_session(new_history)
            self._gc_history.append(result)

        return result

    def get_gc_history(self) -> List[GCResult]:
        """Get history of GC operations."""
        return list(self._gc_history)

    def _maybe_collect_before_send(self) -> Optional[GCResult]:
        """Check and perform GC if needed before sending."""
        if not self._gc_plugin or not self._gc_config:
            return None

        context_usage = self.get_context_usage()
        should_gc, reason = self._gc_plugin.should_collect(context_usage, self._gc_config)

        if should_gc and reason:
            self._trace(
                f"GC_BEFORE_SEND: triggering GC (reason={reason.value}, "
                f"usage={context_usage.get('percent_used', 0):.1f}%)"
            )
            history = self.get_history()
            new_history, result = self._gc_plugin.collect(
                history, context_usage, self._gc_config, reason
            )

            if result.success:
                if result.items_collected == 0:
                    # GC ran but collected nothing - this is often surprising to users
                    self._trace(
                        f"GC_BEFORE_SEND: WARNING - GC triggered but collected 0 items. "
                        f"Check preserve_recent_turns setting vs actual turn count. "
                        f"Details: {result.details}"
                    )
                else:
                    self._trace(
                        f"GC_BEFORE_SEND: collected {result.items_collected} items, "
                        f"freed {result.tokens_freed} tokens"
                    )
                self.reset_session(new_history)
                self._gc_history.append(result)

            return result

        return None

    # ==================== Session Persistence ====================

    def set_session_plugin(
        self,
        plugin: SessionPlugin,
        config: Optional[SessionConfig] = None
    ) -> None:
        """Set the session plugin for persistence."""
        self._session_plugin = plugin
        self._session_config = config or SessionConfig()

        if hasattr(plugin, 'set_session'):
            plugin.set_session(self)

        if hasattr(plugin, 'get_user_commands'):
            for cmd in plugin.get_user_commands():
                self._user_commands[cmd.name] = cmd

        if hasattr(plugin, 'get_executors') and self._executor:
            for name, fn in plugin.get_executors().items():
                self._executor.register(name, fn)

        if hasattr(plugin, 'get_tool_schemas'):
            session_schemas = plugin.get_tool_schemas()
            if session_schemas:
                current_tools = list(self._tools) if self._tools else []
                current_tools.extend(session_schemas)
                self._tools = current_tools
                history = self.get_history() if self._provider else None
                self._create_provider_session(history)

        if self._session_config.auto_resume_last:
            state = self._session_plugin.on_session_start(self._session_config)
            if state:
                self._restore_session_state(state)

    def remove_session_plugin(self) -> None:
        """Remove the session plugin."""
        if self._session_plugin:
            self._session_plugin.shutdown()
        self._session_plugin = None
        self._session_config = None

    def save_session(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None
    ) -> str:
        """Save the current session."""
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured.")

        state = self._get_session_state(session_id, user_inputs)
        self._session_plugin.save(state)

        if hasattr(self._session_plugin, 'set_current_session_id'):
            self._session_plugin.set_current_session_id(state.session_id)

        return state.session_id

    def resume_session(self, session_id: str) -> SessionState:
        """Resume a previously saved session."""
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured.")

        state = self._session_plugin.load(session_id)
        self._restore_session_state(state)
        return state

    def list_sessions(self) -> List[SessionInfo]:
        """List all available sessions."""
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured.")
        return self._session_plugin.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session."""
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured.")
        return self._session_plugin.delete(session_id)

    def _get_session_state(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None
    ) -> SessionState:
        """Build a SessionState from current state."""
        if not session_id:
            if (self._session_plugin and
                    hasattr(self._session_plugin, 'get_current_session_id')):
                session_id = self._session_plugin.get_current_session_id()
            if not session_id:
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        now = datetime.now()
        turn_accounting = self.get_turn_accounting()

        description = None
        if self._session_plugin and hasattr(self._session_plugin, '_session_description'):
            description = self._session_plugin._session_description

        return SessionState(
            session_id=session_id,
            history=self.get_history(),
            created_at=now,
            updated_at=now,
            turn_count=len(turn_accounting),
            turn_accounting=turn_accounting,
            user_inputs=user_inputs or [],
            project=self._runtime.project,
            location=self._runtime.location,
            model=self._model_name,
            description=description,
        )

    def _restore_session_state(self, state: SessionState) -> None:
        """Restore session state from a SessionState."""
        self.reset_session(state.history)
        self._turn_accounting = list(state.turn_accounting)

    def _notify_session_turn_complete(self) -> None:
        """Notify session plugin that a turn completed."""
        if not self._session_plugin or not self._session_config:
            return

        state = self._get_session_state()

        if hasattr(self._session_plugin, 'increment_turn_count'):
            self._session_plugin.increment_turn_count()

        self._session_plugin.on_turn_complete(state, self._session_config)

    def close_session(self) -> None:
        """Close the current session."""
        if self._session_plugin and self._session_config:
            state = self._get_session_state()
            self._session_plugin.on_session_end(state, self._session_config)


__all__ = ['JaatoSession']
