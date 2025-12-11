"""JaatoClient - Core client for the jaato framework.

Provides a unified interface for interacting with AI models. This is a
facade that wraps JaatoRuntime (shared resources) and JaatoSession
(per-agent conversation state).

For simple use cases, JaatoClient provides a convenient all-in-one API.
For advanced use cases (like subagents), access the underlying runtime
via get_runtime() to create additional sessions.
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

from .jaato_runtime import JaatoRuntime
from .jaato_session import JaatoSession
from .token_accounting import TokenLedger
from .plugins.base import UserCommand, OutputCallback
from .plugins.gc import GCConfig, GCPlugin, GCResult
from .plugins.session import SessionPlugin, SessionConfig, SessionState, SessionInfo
from .plugins.model_provider.types import (
    Message,
    Part,
    ToolSchema,
)

# Default provider name
DEFAULT_PROVIDER = "google_genai"

if TYPE_CHECKING:
    from .plugins.registry import PluginRegistry
    from .plugins.permission import PermissionPlugin
    from .plugins.subagent.ui_hooks import AgentUIHooks


class JaatoClient:
    """Core client for jaato framework - facade wrapping Runtime and Session.

    This client provides a unified interface for:
    - Connecting to AI models via ModelProviderPlugin abstraction
    - Configuring tools from plugin registry or custom declarations
    - Multi-turn conversations with provider-managed history
    - History access and reset for flexibility

    Internally, JaatoClient manages:
    - JaatoRuntime: Shared resources (provider config, registry, permissions)
    - JaatoSession: Per-conversation state (history, tools, model)

    For advanced use cases like subagents, use get_runtime() to access the
    shared runtime and create additional sessions.

    Usage:
        # Basic setup (unchanged from before)
        client = JaatoClient()
        client.connect(project_id, location, model_name)
        client.configure_tools(registry, permission_plugin, ledger)

        # Multi-turn conversation
        def on_output(source: str, text: str, mode: str):
            print(f"[{source}]: {text}")

        response = client.send_message("Hello!", on_output=on_output)
        response = client.send_message("Tell me more", on_output=on_output)

        # Access or reset history
        history = client.get_history()
        client.reset_session()

        # Advanced: Access runtime for subagent creation
        runtime = client.get_runtime()
        sub_session = runtime.create_session(model="gemini-2.5-flash")
    """

    def __init__(self, provider_name: str = DEFAULT_PROVIDER):
        """Initialize JaatoClient with specified provider.

        Args:
            provider_name: Name of the model provider to use (default: 'google_genai').
        """
        self._runtime: Optional[JaatoRuntime] = None
        self._session: Optional[JaatoSession] = None
        self._provider_name: str = provider_name

        # Store model name for session creation
        self._model_name: Optional[str] = None

        # Store for backwards compatibility properties
        self._project: Optional[str] = None
        self._location: Optional[str] = None

        # UI hooks for agent lifecycle events
        self._ui_hooks: Optional['AgentUIHooks'] = None
        self._agent_id: str = "main"

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to the model provider."""
        return self._runtime is not None and self._runtime.is_connected

    @property
    def model_name(self) -> Optional[str]:
        """Get the configured model name."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the model provider name."""
        return self._provider_name

    def get_runtime(self) -> JaatoRuntime:
        """Get the underlying JaatoRuntime.

        Use this to access shared resources and create additional sessions
        (e.g., for subagents).

        Returns:
            The JaatoRuntime instance.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._runtime:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._runtime

    def get_session(self) -> JaatoSession:
        """Get the underlying JaatoSession.

        Returns:
            The main JaatoSession instance.

        Raises:
            RuntimeError: If client is not configured.
        """
        if not self._session:
            raise RuntimeError("Client not configured. Call configure_tools() first.")
        return self._session

    def set_ui_hooks(self, hooks: 'AgentUIHooks') -> None:
        """Set UI hooks for agent lifecycle events.

        This enables rich terminal UIs (like rich-client) to track the main agent's
        lifecycle, output, and accounting data.

        Args:
            hooks: Implementation of AgentUIHooks protocol.
        """
        self._ui_hooks = hooks

        # Notify about main agent creation
        if self._ui_hooks:
            self._ui_hooks.on_agent_created(
                agent_id=self._agent_id,
                agent_name="main",
                agent_type="main",
                profile_name=None,
                parent_agent_id=None,
                icon_lines=None,  # Uses default main icon
                created_at=datetime.now()
            )
            self._ui_hooks.on_agent_status_changed(
                agent_id=self._agent_id,
                status="active"
            )

    def list_available_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models from the provider.

        Args:
            prefix: Optional name prefix to filter by (e.g., "gemini").

        Returns:
            List of model names from the catalog.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._runtime:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._runtime.list_available_models(prefix=prefix)

    def connect(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:
        """Connect to the AI model provider.

        For AI Studio (API key): Only model is required.
        For Vertex AI: project, location, and model are all required.

        Args:
            project: Cloud project ID (required for Vertex AI).
            location: Provider region (required for Vertex AI).
            model: Model name (e.g., 'gemini-2.5-flash').
        """
        if not model:
            raise ValueError("model is required")

        # Create runtime and connect
        self._runtime = JaatoRuntime(provider_name=self._provider_name)
        self._runtime.connect(project, location)

        # Store for reference and session creation
        self._model_name = model
        self._project = project
        self._location = location

    def configure_tools(
        self,
        registry: 'PluginRegistry',
        permission_plugin: Optional['PermissionPlugin'] = None,
        ledger: Optional[TokenLedger] = None
    ) -> None:
        """Configure tools from plugin registry.

        Args:
            registry: PluginRegistry with exposed plugins.
            permission_plugin: Optional permission plugin for access control.
            ledger: Optional token ledger for accounting.
        """
        if not self._runtime:
            raise RuntimeError("Client not connected. Call connect() first.")

        # Configure runtime with plugins
        self._runtime.configure_plugins(registry, permission_plugin, ledger)

        # Create main session
        self._session = self._runtime.create_session(model=self._model_name)

    def configure_custom_tools(
        self,
        tools: List[ToolSchema],
        executors: Dict[str, Callable[[Dict[str, Any]], Any]],
        ledger: Optional[TokenLedger] = None,
        system_instruction: Optional[str] = None
    ) -> None:
        """Configure tools directly without plugin registry.

        Use this for specialized scenarios where custom tool declarations
        are needed (e.g., training pipelines with specific function calling).

        Args:
            tools: List of ToolSchema objects defining available tools.
            executors: Dict mapping tool names to executor functions.
            ledger: Optional token ledger for accounting.
            system_instruction: Optional system instruction text.
        """
        if not self._runtime:
            raise RuntimeError("Client not connected. Call connect() first.")

        # For custom tools, we need to create session directly
        # without going through the registry
        from .ai_tool_runner import ToolExecutor

        # Create provider directly
        provider = self._runtime.create_provider(self._model_name)

        # Create session manually
        self._session = JaatoSession(self._runtime, self._model_name)
        self._session._provider = provider
        self._session._tools = tools
        self._session._system_instruction = system_instruction

        # Create executor
        executor = ToolExecutor(ledger=ledger)
        for name, fn in executors.items():
            executor.register(name, fn)
        self._session._executor = executor

        # Create provider session
        provider.create_session(
            system_instruction=system_instruction,
            tools=tools,
            history=None
        )

    def send_message(
        self,
        message: str,
        on_output: Optional[OutputCallback] = None
    ) -> str:
        """Send a message to the model.

        Args:
            message: The user's message text.
            on_output: Optional callback for real-time output.
                Signature: (source: str, text: str, mode: str) -> None

        Returns:
            The final model response text.

        Raises:
            RuntimeError: If client is not connected or not configured.
        """
        if not self._session:
            raise RuntimeError("Tools not configured. Call configure_tools() first.")

        # Wrap output callback to route through UI hooks
        def wrapped_output_callback(source: str, text: str, mode: str) -> None:
            # Call UI hooks if present
            if self._ui_hooks:
                self._ui_hooks.on_agent_output(
                    agent_id=self._agent_id,
                    source=source,
                    text=text,
                    mode=mode
                )
            # Call user's callback if provided
            if on_output:
                on_output(source, text, mode)

        response = self._session.send_message(message, wrapped_output_callback)

        # After turn completes, update UI hooks with accounting data
        if self._ui_hooks:
            # Get turn accounting for the latest turn
            turn_accounting = self._session.get_turn_accounting()
            if turn_accounting:
                last_turn = turn_accounting[-1]
                self._ui_hooks.on_agent_turn_completed(
                    agent_id=self._agent_id,
                    turn_number=len(turn_accounting) - 1,
                    prompt_tokens=last_turn.get('prompt', 0),
                    output_tokens=last_turn.get('output', 0),
                    total_tokens=last_turn.get('total', 0),
                    duration_seconds=last_turn.get('duration_seconds', 0),
                    function_calls=last_turn.get('function_calls', [])
                )

            # Update context usage
            usage = self._session.get_context_usage()
            self._ui_hooks.on_agent_context_updated(
                agent_id=self._agent_id,
                total_tokens=usage.get('total_tokens', 0),
                prompt_tokens=usage.get('prompt_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                turns=usage.get('turns', 0),
                percent_used=usage.get('percent_used', 0)
            )

            # Update history
            history = self._session.get_history()
            self._ui_hooks.on_agent_history_updated(
                agent_id=self._agent_id,
                history=history
            )

        return response

    def get_history(self) -> List[Message]:
        """Get current conversation history.

        Returns:
            List of Message objects representing the conversation.
        """
        if not self._session:
            return []
        return self._session.get_history()

    def get_turn_accounting(self) -> List[Dict[str, Any]]:
        """Get token usage and timing per turn.

        Returns:
            List of dicts with token counts and timing.
        """
        if not self._session:
            return []
        return self._session.get_turn_accounting()

    def get_context_limit(self) -> int:
        """Get the context window limit for the current model.

        Returns:
            The context window size in tokens.
        """
        if not self._session:
            return 1_048_576
        return self._session.get_context_limit()

    def get_context_usage(self) -> Dict[str, Any]:
        """Get context window usage statistics.

        Returns:
            Dict with context usage information.
        """
        if not self._session:
            return {
                'model': self._model_name or 'unknown',
                'context_limit': 1_048_576,
                'total_tokens': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'turns': 0,
                'percent_used': 0,
                'tokens_remaining': 1_048_576,
            }
        return self._session.get_context_usage()

    def reset_session(self, history: Optional[List[Message]] = None) -> None:
        """Reset the chat session, optionally with modified history.

        Args:
            history: Optional initial history for the new session.
        """
        if self._session:
            self._session.reset_session(history)

    def get_turn_boundaries(self) -> List[int]:
        """Get indices where each turn starts in the history.

        Returns:
            List of history indices where each turn starts.
        """
        if not self._session:
            return []
        return self._session.get_turn_boundaries()

    def revert_to_turn(self, turn_id: int) -> Dict[str, Any]:
        """Revert the conversation to a specific turn.

        Args:
            turn_id: 1-based turn number to revert to.

        Returns:
            Dict with reversion status.

        Raises:
            ValueError: If turn_id is invalid.
        """
        if not self._session:
            raise RuntimeError("Session not configured.")
        return self._session.revert_to_turn(turn_id)

    def get_user_commands(self) -> Dict[str, UserCommand]:
        """Get available user commands.

        Returns:
            Dict mapping command names to UserCommand objects.
        """
        if not self._session:
            return {}
        return self._session.get_user_commands()

    def execute_user_command(
        self,
        command_name: str,
        args: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, bool]:
        """Execute a user command.

        Args:
            command_name: Name of the command to execute.
            args: Optional arguments dict.

        Returns:
            Tuple of (result, shared_with_model).

        Raises:
            ValueError: If the command is not found.
            RuntimeError: If executor is not configured.
        """
        if not self._session:
            raise RuntimeError("Session not configured.")
        return self._session.execute_user_command(command_name, args)

    def generate(
        self,
        prompt: str,
        ledger: Optional[TokenLedger] = None
    ) -> str:
        """Simple generation without tools.

        Args:
            prompt: The prompt text.
            ledger: Optional token ledger (currently unused).

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._session:
            raise RuntimeError("Client not configured.")
        return self._session.generate(prompt)

    def send_message_with_parts(
        self,
        parts: List[Part],
        on_output: OutputCallback
    ) -> str:
        """Send a message with custom Part objects.

        Args:
            parts: List of Part objects forming the user's message.
            on_output: Callback for real-time output.

        Returns:
            The final model response text.

        Raises:
            RuntimeError: If client is not connected or not configured.
        """
        if not self._session:
            raise RuntimeError("Session not configured.")
        return self._session.send_message_with_parts(parts, on_output)

    # ==================== Context Garbage Collection ====================

    def set_gc_plugin(
        self,
        plugin: GCPlugin,
        config: Optional[GCConfig] = None
    ) -> None:
        """Set the GC plugin for context management.

        Args:
            plugin: A plugin implementing the GCPlugin protocol.
            config: Optional GC configuration.
        """
        if self._session:
            self._session.set_gc_plugin(plugin, config)

    def remove_gc_plugin(self) -> None:
        """Remove the GC plugin."""
        if self._session:
            self._session.remove_gc_plugin()

    def manual_gc(self) -> GCResult:
        """Manually trigger garbage collection.

        Returns:
            GCResult with details about what was collected.

        Raises:
            RuntimeError: If no GC plugin is configured.
        """
        if not self._session:
            raise RuntimeError("Session not configured.")
        return self._session.manual_gc()

    def get_gc_history(self) -> List[GCResult]:
        """Get history of GC operations.

        Returns:
            List of GCResult objects from previous collections.
        """
        if not self._session:
            return []
        return self._session.get_gc_history()

    # ==================== Session Persistence ====================

    def set_session_plugin(
        self,
        plugin: SessionPlugin,
        config: Optional[SessionConfig] = None
    ) -> None:
        """Set the session plugin for persistence.

        Args:
            plugin: A plugin implementing the SessionPlugin protocol.
            config: Optional session configuration.
        """
        if self._session:
            self._session.set_session_plugin(plugin, config)

    def remove_session_plugin(self) -> None:
        """Remove the session plugin."""
        if self._session:
            self._session.remove_session_plugin()

    def save_session(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None
    ) -> str:
        """Save the current session.

        Args:
            session_id: Optional session ID.
            user_inputs: Optional list of user input strings.

        Returns:
            The session ID that was saved.

        Raises:
            RuntimeError: If no session plugin is configured.
        """
        if not self._session:
            raise RuntimeError("Session not configured.")
        return self._session.save_session(session_id, user_inputs)

    def resume_session(self, session_id: str) -> SessionState:
        """Resume a previously saved session.

        Args:
            session_id: The session ID to resume.

        Returns:
            The loaded SessionState.

        Raises:
            RuntimeError: If no session plugin is configured.
        """
        if not self._session:
            raise RuntimeError("Session not configured.")
        return self._session.resume_session(session_id)

    def list_sessions(self) -> List[SessionInfo]:
        """List all available sessions.

        Returns:
            List of SessionInfo objects.

        Raises:
            RuntimeError: If no session plugin is configured.
        """
        if not self._session:
            raise RuntimeError("Session not configured.")
        return self._session.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted, False if session didn't exist.

        Raises:
            RuntimeError: If no session plugin is configured.
        """
        if not self._session:
            raise RuntimeError("Session not configured.")
        return self._session.delete_session(session_id)

    def close_session(self) -> None:
        """Close the current session, triggering auto-save if configured."""
        if self._session:
            self._session.close_session()

    # ==================== Backwards Compatibility ====================
    # These attributes are kept for backwards compatibility with code
    # that accesses internal state directly. Prefer using the public API.

    @property
    def _ledger(self) -> Optional[TokenLedger]:
        """Internal: Get ledger (for backwards compatibility)."""
        if self._runtime:
            return self._runtime.ledger
        return None

    @property
    def _registry(self) -> Optional['PluginRegistry']:
        """Internal: Get registry (for backwards compatibility)."""
        if self._runtime:
            return self._runtime.registry
        return None

    @property
    def _executor(self) -> Optional[Any]:
        """Internal: Get executor (for backwards compatibility)."""
        if self._session:
            return self._session._executor
        return None

    @property
    def _tools(self) -> Optional[List[ToolSchema]]:
        """Internal: Get tools (for backwards compatibility)."""
        if self._session:
            return self._session._tools
        return None

    @property
    def _system_instruction(self) -> Optional[str]:
        """Internal: Get system instruction (for backwards compatibility)."""
        if self._session:
            return self._session._system_instruction
        return None

    @property
    def _turn_accounting(self) -> List[Dict[str, Any]]:
        """Internal: Get turn accounting (for backwards compatibility)."""
        if self._session:
            return self._session._turn_accounting
        return []

    @property
    def _user_commands(self) -> Dict[str, UserCommand]:
        """Internal: Get user commands (for backwards compatibility)."""
        if self._session:
            return self._session._user_commands
        return {}

    @property
    def _gc_plugin(self) -> Optional[GCPlugin]:
        """Internal: Get GC plugin (for backwards compatibility)."""
        if self._session:
            return self._session._gc_plugin
        return None

    @property
    def _gc_config(self) -> Optional[GCConfig]:
        """Internal: Get GC config (for backwards compatibility)."""
        if self._session:
            return self._session._gc_config
        return None

    @property
    def _session_plugin(self) -> Optional[SessionPlugin]:
        """Internal: Get session plugin (for backwards compatibility)."""
        if self._session:
            return self._session._session_plugin
        return None

    @property
    def _session_config(self) -> Optional[SessionConfig]:
        """Internal: Get session config (for backwards compatibility)."""
        if self._session:
            return self._session._session_config
        return None

    @property
    def _provider(self) -> Optional[Any]:
        """Internal: Get provider (for backwards compatibility)."""
        if self._session:
            return self._session._provider
        return None


__all__ = ['JaatoClient', 'DEFAULT_PROVIDER']
