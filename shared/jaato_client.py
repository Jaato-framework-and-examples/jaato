"""JaatoClient - Core client for the jaato framework.

Provides a unified interface for interacting with AI models via the
ModelProviderPlugin abstraction, supporting multiple providers
(Google GenAI, Anthropic, etc.).
Uses the provider's session management for multi-turn conversations.
"""

import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .ai_tool_runner import ToolExecutor
from .token_accounting import TokenLedger
from .plugins.base import UserCommand, PromptEnrichmentResult, OutputCallback
from .plugins.gc import GCConfig, GCPlugin, GCResult, GCTriggerReason
from .plugins.session import SessionPlugin, SessionConfig, SessionState, SessionInfo
from .plugins.model_provider.types import (
    Attachment,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    ToolResult,
    ToolSchema,
)
from .plugins.model_provider.base import ModelProviderPlugin, ProviderConfig
from .plugins.model_provider import load_provider

# Pattern to match @references in prompts (e.g., @file.png, @path/to/file.txt)
# Matches @ followed by a path-like string (no spaces, common file chars)
AT_REFERENCE_PATTERN = re.compile(r'@([\w./\-]+(?:\.\w+)?)')

# Default provider name (can be overridden)
DEFAULT_PROVIDER = "google_genai"

if TYPE_CHECKING:
    from .plugins.registry import PluginRegistry
    from .plugins.permission import PermissionPlugin


class JaatoClient:
    """Core client for jaato framework with provider-managed conversation history.

    This client provides a unified interface for:
    - Connecting to AI models via ModelProviderPlugin abstraction
    - Configuring tools from plugin registry or custom declarations
    - Multi-turn conversations with provider-managed history
    - History access and reset for flexibility

    The provider manages conversation history internally. Use
    get_history() to access it and reset_session() to modify it.

    Usage:
        # Basic setup
        client = JaatoClient()
        client.connect(project_id, location, model_name)
        client.configure_tools(registry, permission_plugin, ledger)

        # Multi-turn conversation (provider manages history)
        # Output callback receives (source, text, mode) for real-time display
        def on_output(source: str, text: str, mode: str):
            print(f"[{source}]: {text}")

        response = client.send_message("Hello!", on_output=on_output)
        response = client.send_message("Tell me more", on_output=on_output)

        # Access or reset history when needed
        history = client.get_history()
        client.reset_session()  # Clear history
        client.reset_session(modified_history)  # Reset with custom history
    """

    def __init__(self, provider_name: str = DEFAULT_PROVIDER):
        """Initialize JaatoClient with specified provider.

        Args:
            provider_name: Name of the model provider to use (default: 'google_genai').
        """
        # Model provider (abstracts SDK interactions)
        self._provider: Optional[ModelProviderPlugin] = None
        self._provider_name: str = provider_name

        # Connection info (for reference)
        self._model_name: Optional[str] = None
        self._project: Optional[str] = None
        self._location: Optional[str] = None

        # Tool configuration
        self._executor: Optional[ToolExecutor] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._system_instruction: Optional[str] = None
        self._ledger: Optional[TokenLedger] = None

        # Per-turn token accounting
        self._turn_accounting: List[Dict[str, int]] = []

        # User commands: name -> UserCommand mapping
        self._user_commands: Dict[str, UserCommand] = {}

        # Context garbage collection
        self._gc_plugin: Optional[GCPlugin] = None
        self._gc_config: Optional[GCConfig] = None
        self._gc_history: List[GCResult] = []

        # Session persistence
        self._session_plugin: Optional[SessionPlugin] = None
        self._session_config: Optional[SessionConfig] = None

        # Plugin registry for prompt enrichment
        self._registry: Optional['PluginRegistry'] = None

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to the model provider."""
        return self._provider is not None and self._provider.is_connected

    @property
    def model_name(self) -> Optional[str]:
        """Get the configured model name."""
        return self._model_name

    def list_available_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models from the provider.

        Note: This returns the model catalog from the provider.
        Availability may vary by region or configuration.

        Args:
            prefix: Optional name prefix to filter by (e.g., "gemini").
                    Defaults to None (all models).

        Returns:
            List of model names from the catalog.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._provider:
            raise RuntimeError("Client not connected. Call connect() first.")

        return self._provider.list_models(prefix=prefix)

    def connect(self, project: str, location: str, model: str) -> None:
        """Connect to the AI model provider.

        Args:
            project: Cloud project ID (e.g., GCP project).
            location: Provider region (e.g., 'us-central1', 'global').
            model: Model name (e.g., 'gemini-2.0-flash').
        """
        # Load and initialize the provider
        config = ProviderConfig(project=project, location=location)
        self._provider = load_provider(self._provider_name, config)
        self._provider.connect(model)

        # Store for reference
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
        self._ledger = ledger
        self._executor = ToolExecutor(ledger=ledger)
        self._registry = registry  # Store for prompt enrichment

        # Pass connection info and permission plugin to subagent plugin if it's exposed
        self._configure_subagent_plugin(registry, permission_plugin)

        # Register executors from plugins
        for name, fn in registry.get_exposed_executors().items():
            self._executor.register(name, fn)

        # Pass registry to executor for auto-background support
        self._executor.set_registry(registry)

        # Configure background plugin if exposed
        self._configure_background_plugin(registry)

        # Set permission plugin for enforcement
        if permission_plugin:
            # Give permission plugin access to registry for plugin lookups
            # This enables format_permission_request() calls for custom diff display
            permission_plugin.set_registry(registry)
            # Pass agent_type context so permission prompts can identify the requester
            self._executor.set_permission_plugin(
                permission_plugin,
                context={"agent_type": "main"}
            )
            # Also register askPermission tool
            for name, fn in permission_plugin.get_executors().items():
                self._executor.register(name, fn)
            # Whitelist auto-approved tools from plugins
            auto_approved = registry.get_auto_approved_tools()
            if auto_approved:
                permission_plugin.add_whitelist_tools(auto_approved)

        # Build tool schemas list (provider-agnostic)
        all_schemas = registry.get_exposed_tool_schemas()
        if permission_plugin:
            all_schemas.extend(permission_plugin.get_tool_schemas())
        self._tools = all_schemas if all_schemas else None

        # Collect system instructions
        parts = []
        registry_instructions = registry.get_system_instructions()
        if registry_instructions:
            parts.append(registry_instructions)
        if permission_plugin:
            perm_instructions = permission_plugin.get_system_instructions()
            if perm_instructions:
                parts.append(perm_instructions)
        self._system_instruction = "\n\n".join(parts) if parts else None

        # Store user commands for execute_user_command()
        self._user_commands = {}
        for cmd in registry.get_exposed_user_commands():
            self._user_commands[cmd.name] = cmd

        # Create session with provider
        self._create_session()

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
        self._ledger = ledger
        self._executor = ToolExecutor(ledger=ledger)

        for name, fn in executors.items():
            self._executor.register(name, fn)

        self._tools = tools if tools else None
        self._system_instruction = system_instruction

        # Create session with provider
        self._create_session()

    def _configure_subagent_plugin(
        self,
        registry: 'PluginRegistry',
        permission_plugin: Optional['PermissionPlugin'] = None
    ) -> None:
        """Pass connection info, parent plugins, and permission plugin to subagent plugin.

        This allows subagents to inherit the parent's connection settings
        (project, location, model), plugin configuration, and permission
        enforcement without requiring explicit inline_config.

        Args:
            registry: PluginRegistry to check for subagent plugin.
            permission_plugin: Optional permission plugin to share with subagents.
        """
        # Check if subagent plugin is exposed
        try:
            subagent_plugin = registry.get_plugin('subagent')
            if not subagent_plugin:
                return

            # Pass connection info
            if self._project and self._location and self._model_name:
                if hasattr(subagent_plugin, 'set_connection'):
                    subagent_plugin.set_connection(self._project, self._location, self._model_name)

            # Pass parent's exposed plugins for inheritance
            if hasattr(subagent_plugin, 'set_parent_plugins'):
                exposed = registry.list_exposed()
                # Exclude subagent itself to prevent recursion
                parent_plugins = [p for p in exposed if p != 'subagent']
                subagent_plugin.set_parent_plugins(parent_plugins)

            # Pass permission plugin so subagents can use it with their own context
            if permission_plugin and hasattr(subagent_plugin, 'set_permission_plugin'):
                subagent_plugin.set_permission_plugin(permission_plugin)

        except (KeyError, AttributeError):
            # Subagent plugin not exposed or not available
            pass

    def _configure_background_plugin(self, registry: 'PluginRegistry') -> None:
        """Pass registry to background plugin for capability discovery.

        This allows the background plugin to discover which other plugins
        implement BackgroundCapable and can have their tools backgrounded.

        Args:
            registry: PluginRegistry to check for background plugin.
        """
        try:
            background_plugin = registry.get_plugin('background')
            if background_plugin and hasattr(background_plugin, 'set_registry'):
                background_plugin.set_registry(registry)
        except (KeyError, AttributeError):
            # Background plugin not exposed or not available
            pass

    def _create_session(self, history: Optional[List[Message]] = None) -> None:
        """Create or recreate the provider session.

        Args:
            history: Optional initial conversation history (provider-agnostic Messages).
        """
        if not self._provider or not self._provider.is_connected:
            return  # Can't create session without connection

        self._provider.create_session(
            system_instruction=self._system_instruction,
            tools=self._tools,
            history=history
        )

    def send_message(
        self,
        message: str,
        on_output: OutputCallback
    ) -> str:
        """Send a message to the model.

        The provider manages conversation history internally. Use get_history()
        to access it and reset_session() to modify it.

        If a GC plugin is configured with check_before_send=True, this will
        automatically check and perform garbage collection if needed before
        sending the message.

        If plugins are configured that subscribe to prompt enrichment, the
        message will be passed through them before sending. After enrichment,
        any @references are stripped from the message.

        Args:
            message: The user's message text.
            on_output: Callback for real-time output from model and plugins.
                Signature: (source: str, text: str, mode: str) -> None
                - source: "model" for model responses, plugin name for plugins
                - text: The output text
                - mode: "write" for new block, "append" to continue

        Returns:
            The final model response text (after all function calls resolved).

        Raises:
            RuntimeError: If client is not connected or not configured.
        """
        if not self._provider or not self._provider.is_connected:
            raise RuntimeError("Client not connected. Call connect() first.")
        if not self._tools and not self._system_instruction:
            raise RuntimeError("Tools not configured. Call configure_tools() first.")

        # Check and perform GC if needed before sending
        if self._gc_plugin and self._gc_config and self._gc_config.check_before_send:
            self._maybe_collect_before_send()

        # Run prompt enrichment pipeline if registry is configured
        processed_message = self._enrich_and_clean_prompt(message)

        response = self._run_chat_loop(processed_message, on_output)

        # Notify session plugin that turn completed
        self._notify_session_turn_complete()

        return response

    def _enrich_and_clean_prompt(self, prompt: str) -> str:
        """Run prompt through enrichment pipeline and strip @references.

        Args:
            prompt: The user's original prompt text.

        Returns:
            The processed prompt ready to send to the model.
        """
        enriched_prompt = prompt

        # Run through plugin enrichment pipeline (all registered plugins)
        if self._registry:
            result = self._registry.enrich_prompt(prompt)
            enriched_prompt = result.prompt
            # Metadata is available in result.metadata if needed for debugging

        # Strip @references from the prompt (framework responsibility)
        cleaned_prompt = self._strip_at_references(enriched_prompt)

        return cleaned_prompt

    def _strip_at_references(self, prompt: str) -> str:
        """Remove @ prefix from references in the prompt.

        Converts @filename.png to filename.png, preserving the rest of the text.

        Args:
            prompt: Prompt text possibly containing @references.

        Returns:
            Prompt with @ prefixes removed from references.
        """
        return AT_REFERENCE_PATTERN.sub(r'\1', prompt)

    def _run_chat_loop(
        self,
        message: str,
        on_output: OutputCallback
    ) -> str:
        """Internal function calling loop using provider.send_message().

        Args:
            message: The user's message text.
            on_output: Callback for real-time output.
                Invoked with (source, text, mode) each time the model produces
                text during the function calling loop.

        Returns:
            The final response text (after all function calls resolved).
        """
        # Set output callback on executor so plugins can emit output
        if self._executor:
            self._executor.set_output_callback(on_output)

        # Track tokens and timing for this turn
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
            response = self._provider.send_message(message)
            self._record_provider_token_usage(response)
            self._accumulate_provider_turn_tokens(response, turn_data)

            # Handle function calling loop
            function_calls = list(response.function_calls) if response.function_calls else []
            while function_calls:
                # Emit any text produced alongside function calls
                if response.text:
                    on_output("model", response.text, "write")

                tool_results: List[ToolResult] = []

                for fc in function_calls:
                    # Execute the function with timing
                    name = fc.name
                    args = fc.args

                    fc_start = datetime.now()
                    if self._executor:
                        executor_result = self._executor.execute(name, args)
                    else:
                        executor_result = (False, {"error": f"No executor registered for {name}"})
                    fc_end = datetime.now()

                    # Record function call timing
                    turn_data['function_calls'].append({
                        'name': name,
                        'start_time': fc_start.isoformat(),
                        'end_time': fc_end.isoformat(),
                        'duration_seconds': (fc_end - fc_start).total_seconds(),
                    })

                    # Build ToolResult from executor result
                    tool_result = self._build_tool_result(fc, executor_result)
                    tool_results.append(tool_result)

                # Send tool results back to model via provider
                response = self._provider.send_tool_results(tool_results)
                self._record_provider_token_usage(response)
                self._accumulate_provider_turn_tokens(response, turn_data)

                # Check finish_reason for abnormal termination
                from .plugins.model_provider.types import FinishReason
                if response.finish_reason not in (FinishReason.STOP, FinishReason.UNKNOWN, FinishReason.TOOL_USE):
                    # Non-normal finish reason - model stopped unexpectedly
                    import sys
                    print(f"[warning] Model stopped with finish_reason={response.finish_reason}", file=sys.stderr)
                    if response.text:
                        return f"{response.text}\n\n[Model stopped: {response.finish_reason}]"
                    else:
                        return f"[Model stopped unexpectedly: {response.finish_reason}]"

                # Re-cache function_calls for next iteration
                function_calls = list(response.function_calls) if response.function_calls else []

            # Return the final response text
            return response.text or ''

        finally:
            # Record turn end time and duration
            turn_end = datetime.now()
            turn_data['end_time'] = turn_end.isoformat()
            turn_data['duration_seconds'] = (turn_end - turn_start).total_seconds()

            # Always store turn accounting, even on errors
            if turn_data['total'] > 0:
                self._turn_accounting.append(turn_data)

    def _build_tool_result(
        self,
        fc: FunctionCall,
        executor_result: Any
    ) -> ToolResult:
        """Build a ToolResult from executor output, handling multimodal content.

        If the result contains multimodal data (indicated by '_multimodal': True),
        this extracts attachments for the provider to handle.

        Args:
            fc: The FunctionCall that was executed.
            executor_result: The executor result (tuple: (ok, result_dict)).

        Returns:
            ToolResult ready for the provider.
        """
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
            # Clean up internal multimodal flags from result
            result_data = {k: v for k, v in result_data.items()
                          if not k.startswith('_multimodal') and k not in ('image_data',)}

        # Build the result dict
        if isinstance(result_data, dict):
            result_dict = result_data
        else:
            result_dict = {"result": result_data}

        return ToolResult(
            call_id=fc.id,
            name=fc.name,
            result=result_dict,
            is_error=not ok,
            attachments=attachments
        )

    def _extract_multimodal_attachments(
        self,
        result: Dict[str, Any]
    ) -> Optional[List[Attachment]]:
        """Extract multimodal attachments from a result dict.

        Args:
            result: Dict with '_multimodal': True and image/file data.

        Returns:
            List of Attachment objects, or None if extraction fails.
        """
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

    def _accumulate_provider_turn_tokens(
        self,
        response: ProviderResponse,
        turn_tokens: Dict[str, int]
    ) -> None:
        """Accumulate token counts from provider response into turn totals."""
        turn_tokens['prompt'] += response.usage.prompt_tokens
        turn_tokens['output'] += response.usage.output_tokens
        turn_tokens['total'] += response.usage.total_tokens

    def _record_provider_token_usage(self, response: ProviderResponse) -> None:
        """Record token usage from provider response to ledger if available."""
        if not self._ledger:
            return

        self._ledger._record('response', {
            'prompt_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.output_tokens,
            'total_tokens': response.usage.total_tokens,
        })

    def get_history(self) -> List[Message]:
        """Get current conversation history.

        Returns:
            List of Message objects representing the conversation.
        """
        if not self._provider:
            return []
        return self._provider.get_history()

    def get_turn_accounting(self) -> List[Dict[str, Any]]:
        """Get token usage and timing per turn.

        Each entry corresponds to one send_message() call and contains
        aggregated tokens across all API calls in that turn (including
        function calling loops), plus timing information.

        Returns:
            List of dicts with:
            - 'prompt': Prompt token count
            - 'output': Output token count
            - 'total': Total token count
            - 'start_time': ISO format timestamp when turn started
            - 'end_time': ISO format timestamp when turn ended
            - 'duration_seconds': Total turn duration in seconds
            - 'function_calls': List of function call timing dicts, each with:
                - 'name': Function name
                - 'start_time': ISO format timestamp
                - 'end_time': ISO format timestamp
                - 'duration_seconds': Duration in seconds
        """
        return list(self._turn_accounting)

    def get_context_limit(self) -> int:
        """Get the context window limit for the current model.

        Returns:
            The context window size in tokens.
        """
        if not self._provider:
            return 1_048_576  # Default fallback

        return self._provider.get_context_limit()

    def get_context_usage(self) -> Dict[str, Any]:
        """Get context window usage statistics.

        Returns:
            Dict containing:
            - model: The model name
            - context_limit: Maximum context window size
            - total_tokens: Total tokens used in session
            - prompt_tokens: Total prompt/input tokens
            - output_tokens: Total output/completion tokens
            - turns: Number of conversation turns
            - percent_used: Percentage of context window used
            - tokens_remaining: Tokens remaining in context window
        """
        turn_accounting = self.get_turn_accounting()

        total_prompt = sum(t['prompt'] for t in turn_accounting)
        total_output = sum(t['output'] for t in turn_accounting)
        total_tokens = sum(t['total'] for t in turn_accounting)

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
        """Reset the chat session, optionally with modified history.

        Use this to:
        - Clear conversation history: reset_session()
        - Start with custom history: reset_session(modified_history)

        Args:
            history: Optional initial history for the new session (provider-agnostic Messages).
        """
        self._turn_accounting = []
        self._create_session(history)

    def get_turn_boundaries(self) -> List[int]:
        """Get indices where each turn starts in the history.

        A turn starts with a user text message (not a function response).
        Returns 1-based turn numbers mapped to 0-based history indices.

        Returns:
            List of history indices where each turn starts.
            Index i contains the history index where turn (i+1) starts.
        """
        history = self.get_history()
        boundaries = []

        for i, msg in enumerate(history):
            # A turn starts with a user message that has text (not function response)
            if msg.role == Role.USER and msg.parts and msg.parts[0].text:
                boundaries.append(i)

        return boundaries

    def revert_to_turn(self, turn_id: int) -> Dict[str, Any]:
        """Revert the conversation to a specific turn.

        Removes all history after the specified turn, keeping the turn
        and everything before it.

        Args:
            turn_id: 1-based turn number to revert to.

        Returns:
            Dict with:
            - success: True if reverted successfully
            - turns_removed: Number of turns removed
            - new_turn_count: Current turn count after reversion
            - message: Human-readable status message

        Raises:
            ValueError: If turn_id is invalid.
        """
        boundaries = self.get_turn_boundaries()
        total_turns = len(boundaries)

        if turn_id < 1:
            raise ValueError(f"Turn ID must be >= 1, got {turn_id}")

        if turn_id > total_turns:
            raise ValueError(f"Turn {turn_id} does not exist. Current session has {total_turns} turn(s).")

        if turn_id == total_turns:
            # Already at the requested turn, nothing to do
            return {
                'success': True,
                'turns_removed': 0,
                'new_turn_count': total_turns,
                'message': f"Already at turn {turn_id}, no changes made."
            }

        # Find where to truncate: keep everything up to (but not including) the next turn
        history = self.get_history()

        if turn_id < total_turns:
            # Truncate at the start of the next turn
            truncate_at = boundaries[turn_id]  # boundaries[turn_id] is where turn (turn_id+1) starts
        else:
            truncate_at = len(history)

        truncated_history = list(history[:truncate_at])
        turns_removed = total_turns - turn_id

        # Truncate turn accounting to match
        if turn_id <= len(self._turn_accounting):
            self._turn_accounting = self._turn_accounting[:turn_id]

        # Reset session with truncated history
        self._create_session(truncated_history)

        # Reset session plugin's turn count if it has one
        if self._session_plugin and hasattr(self._session_plugin, 'set_turn_count'):
            self._session_plugin.set_turn_count(turn_id)

        return {
            'success': True,
            'turns_removed': turns_removed,
            'new_turn_count': turn_id,
            'message': f"Reverted to turn {turn_id} (removed {turns_removed} turn(s))."
        }

    def get_user_commands(self) -> Dict[str, UserCommand]:
        """Get available user commands.

        Returns:
            Dict mapping command names to UserCommand objects.
        """
        return dict(self._user_commands)

    def execute_user_command(
        self,
        command_name: str,
        args: Optional[Dict[str, Any]] = None
    ) -> tuple[Any, bool]:
        """Execute a user command and optionally share with model.

        User commands are plugin-provided commands that can be invoked
        directly by the user. Each command declares whether its output
        should be shared with the model via the share_with_model flag.

        Args:
            command_name: Name of the command to execute.
            args: Optional arguments dict for the command.

        Returns:
            Tuple of (result, shared_with_model):
            - result: The command's return value
            - shared_with_model: True if the result was added to conversation history

        Raises:
            ValueError: If the command is not found.
            RuntimeError: If executor is not configured.
        """
        if command_name not in self._user_commands:
            raise ValueError(f"Unknown user command: {command_name}")

        if not self._executor:
            raise RuntimeError("Executor not configured. Call configure_tools() first.")

        cmd = self._user_commands[command_name]
        args = args or {}

        # Execute the command - executor returns (success_flag, result) tuple
        _ok, result = self._executor.execute(command_name, args)

        # If share_with_model is True, add to conversation history
        if cmd.share_with_model and self._provider:
            # Add as a user message with function call and response
            # This way the model sees what command was executed and what it returned
            self._inject_command_into_history(command_name, args, result)

        return result, cmd.share_with_model

    def _inject_command_into_history(
        self,
        command_name: str,
        args: Dict[str, Any],
        result: Any
    ) -> None:
        """Inject a user command execution into conversation history.

        This adds the command and its result to history so the model
        can see what the user executed and use the information.

        Args:
            command_name: Name of the executed command.
            args: Arguments passed to the command.
            result: The command's return value.
        """
        # Get current history
        current_history = self.get_history()

        # Create a user message indicating the command was run
        user_message = Message(
            role=Role.USER,
            parts=[Part.from_text(f"[User executed command: {command_name}]")]
        )

        # Create a model message with the function response
        # This simulates the model having called the function
        result_dict = result if isinstance(result, dict) else {"result": result}
        model_message = Message(
            role=Role.MODEL,
            parts=[Part.from_function_response(ToolResult(
                call_id="",
                name=command_name,
                result=result_dict
            ))]
        )

        # Recreate session with updated history
        new_history = list(current_history) + [user_message, model_message]
        self._create_session(new_history)

    def generate(
        self,
        prompt: str,
        ledger: Optional[TokenLedger] = None
    ) -> str:
        """Simple generation without tools.

        Use this for basic prompts that don't need function calling.

        Args:
            prompt: The prompt text.
            ledger: Optional token ledger for accounting (currently unused with provider).

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._provider or not self._provider.is_connected:
            raise RuntimeError("Client not connected. Call connect() first.")

        response = self._provider.generate(prompt)

        # Record token usage if ledger provided
        if ledger:
            ledger._record('response', {
                'prompt_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.total_tokens,
            })

        return response.text or ''

    def send_message_with_parts(
        self,
        parts: List[Part],
        on_output: OutputCallback
    ) -> str:
        """Send a message with custom Part objects.

        Similar to send_message but allows sending multi-modal content
        (images, etc.) via Part objects.

        Args:
            parts: List of Part objects forming the user's message.
            on_output: Callback for real-time output from model and plugins.
                Signature: (source: str, text: str, mode: str) -> None

        Returns:
            The final model response text (after all function calls resolved).

        Raises:
            RuntimeError: If client is not connected or not configured.
        """
        if not self._provider or not self._provider.is_connected:
            raise RuntimeError("Client not connected. Call connect() first.")
        if not self._tools and not self._system_instruction:
            raise RuntimeError("Tools not configured. Call configure_tools() first.")

        return self._run_chat_loop_with_parts(parts, on_output)

    def _run_chat_loop_with_parts(
        self,
        parts: List[Part],
        on_output: OutputCallback
    ) -> str:
        """Internal function calling loop for multi-part messages.

        Args:
            parts: List of Part objects forming the user's message.
            on_output: Callback for real-time output.
                Invoked with (source, text, mode) each time the model produces
                text during the function calling loop.

        Returns:
            The final response text (after all function calls resolved).
        """
        # Set output callback on executor so plugins can emit output
        if self._executor:
            self._executor.set_output_callback(on_output)

        # Track tokens and timing for this turn
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
            # Send parts via provider
            response = self._provider.send_message_with_parts(parts)
            self._record_provider_token_usage(response)
            self._accumulate_provider_turn_tokens(response, turn_data)

            # Check finish_reason for abnormal termination
            from .plugins.model_provider.types import FinishReason
            if response.finish_reason not in (FinishReason.STOP, FinishReason.UNKNOWN, FinishReason.TOOL_USE):
                # Non-normal finish reason - model stopped unexpectedly
                import sys
                print(f"[warning] Model stopped with finish_reason={response.finish_reason}", file=sys.stderr)
                if response.text:
                    return f"{response.text}\n\n[Model stopped: {response.finish_reason}]"
                else:
                    return f"[Model stopped unexpectedly: {response.finish_reason}]"

            # Handle function calling loop
            function_calls = list(response.function_calls) if response.function_calls else []
            while function_calls:
                # Emit any text produced alongside function calls
                if response.text:
                    on_output("model", response.text, "write")

                tool_results: List[ToolResult] = []

                for fc in function_calls:
                    name = fc.name
                    args = fc.args

                    fc_start = datetime.now()
                    if self._executor:
                        executor_result = self._executor.execute(name, args)
                    else:
                        executor_result = (False, {"error": f"No executor registered for {name}"})
                    fc_end = datetime.now()

                    # Record function call timing
                    turn_data['function_calls'].append({
                        'name': name,
                        'start_time': fc_start.isoformat(),
                        'end_time': fc_end.isoformat(),
                        'duration_seconds': (fc_end - fc_start).total_seconds(),
                    })

                    # Build ToolResult from executor result
                    tool_result = self._build_tool_result(fc, executor_result)
                    tool_results.append(tool_result)

                # Send tool results back to model via provider
                response = self._provider.send_tool_results(tool_results)
                self._record_provider_token_usage(response)
                self._accumulate_provider_turn_tokens(response, turn_data)
                # Re-cache function_calls for next iteration
                function_calls = list(response.function_calls) if response.function_calls else []

            # Return the final response text
            return response.text or ''

        finally:
            # Record turn end time and duration
            turn_end = datetime.now()
            turn_data['end_time'] = turn_end.isoformat()
            turn_data['duration_seconds'] = (turn_end - turn_start).total_seconds()

            # Store turn accounting
            if turn_data['total'] > 0:
                self._turn_accounting.append(turn_data)

    # ==================== Context Garbage Collection ====================

    def set_gc_plugin(
        self,
        plugin: GCPlugin,
        config: Optional[GCConfig] = None
    ) -> None:
        """Set the GC plugin for context management.

        The GC plugin will be used to manage conversation history and
        prevent context window overflow. Different plugins implement
        different strategies (truncation, summarization, hybrid).

        Args:
            plugin: A plugin implementing the GCPlugin protocol.
            config: Optional GC configuration. Uses defaults if not provided.

        Example:
            from shared.plugins.gc_truncate import create_plugin

            gc_plugin = create_plugin()
            gc_plugin.initialize({"preserve_recent_turns": 10})
            client.set_gc_plugin(gc_plugin, GCConfig(threshold_percent=75.0))
        """
        self._gc_plugin = plugin
        self._gc_config = config or GCConfig()

    def remove_gc_plugin(self) -> None:
        """Remove the GC plugin, disabling automatic garbage collection."""
        if self._gc_plugin:
            self._gc_plugin.shutdown()
        self._gc_plugin = None
        self._gc_config = None

    def manual_gc(self) -> GCResult:
        """Manually trigger garbage collection.

        Forces a GC operation regardless of current context usage.
        Useful for proactive cleanup or testing.

        Returns:
            GCResult with details about what was collected.

        Raises:
            RuntimeError: If no GC plugin is configured.
        """
        if not self._gc_plugin:
            raise RuntimeError("No GC plugin configured. Call set_gc_plugin() first.")
        if not self._gc_config:
            self._gc_config = GCConfig()

        history = self.get_history()
        context_usage = self.get_context_usage()

        new_history, result = self._gc_plugin.collect(
            history, context_usage, self._gc_config, GCTriggerReason.MANUAL
        )

        if result.success:
            self.reset_session(new_history)
            self._gc_history.append(result)

        return result

    def get_gc_history(self) -> List[GCResult]:
        """Get history of GC operations performed in this session.

        Returns:
            List of GCResult objects from previous collections.
        """
        return list(self._gc_history)

    def _maybe_collect_before_send(self) -> Optional[GCResult]:
        """Check and perform GC if needed before sending a message.

        Called automatically by send_message() if GC is configured
        with check_before_send=True.

        Returns:
            GCResult if GC was performed, None otherwise.
        """
        if not self._gc_plugin or not self._gc_config:
            return None

        context_usage = self.get_context_usage()
        should_gc, reason = self._gc_plugin.should_collect(context_usage, self._gc_config)

        if should_gc and reason:
            history = self.get_history()
            new_history, result = self._gc_plugin.collect(
                history, context_usage, self._gc_config, reason
            )

            if result.success:
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
        """Set the session plugin for persistence.

        The session plugin handles saving and loading conversation history,
        allowing users to resume sessions across client restarts.

        Args:
            plugin: A plugin implementing the SessionPlugin protocol.
            config: Optional session configuration. Uses defaults if not provided.

        Example:
            from shared.plugins.session import create_plugin, SessionConfig

            session_plugin = create_plugin()
            session_plugin.initialize({'storage_path': '.jaato/sessions'})
            client.set_session_plugin(session_plugin, SessionConfig())
        """
        self._session_plugin = plugin
        self._session_config = config or SessionConfig()

        # Give plugin a reference to this client for user command execution
        if hasattr(plugin, 'set_client'):
            plugin.set_client(self)

        # Register session plugin's user commands and executors
        if hasattr(plugin, 'get_user_commands'):
            for cmd in plugin.get_user_commands():
                self._user_commands[cmd.name] = cmd

        if hasattr(plugin, 'get_executors') and self._executor:
            for name, fn in plugin.get_executors().items():
                self._executor.register(name, fn)

        # Add session plugin's tool schemas to the tools list
        if hasattr(plugin, 'get_tool_schemas'):
            session_schemas = plugin.get_tool_schemas()
            if session_schemas:
                # Add session schemas to current tools
                current_tools = list(self._tools) if self._tools else []
                current_tools.extend(session_schemas)
                self._tools = current_tools

                # Recreate session with updated tools
                history = self.get_history() if self._provider else None
                self._create_session(history)

        # Check for auto-resume
        if self._session_config.auto_resume_last:
            state = self._session_plugin.on_session_start(self._session_config)
            if state:
                self._restore_session_state(state)

    def remove_session_plugin(self) -> None:
        """Remove the session plugin, disabling session persistence."""
        if self._session_plugin:
            self._session_plugin.shutdown()
        self._session_plugin = None
        self._session_config = None

    def save_session(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None
    ) -> str:
        """Save the current session.

        Args:
            session_id: Optional session ID. If not provided, generates one
                       from the current timestamp.
            user_inputs: Optional list of user input strings for readline
                        history restoration on resume.

        Returns:
            The session ID that was saved.

        Raises:
            RuntimeError: If no session plugin is configured.
        """
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured. Call set_session_plugin() first.")

        state = self._get_session_state(session_id, user_inputs)
        self._session_plugin.save(state)

        # Update the plugin's current session tracking
        if hasattr(self._session_plugin, 'set_current_session_id'):
            self._session_plugin.set_current_session_id(state.session_id)

        return state.session_id

    def resume_session(self, session_id: str) -> SessionState:
        """Resume a previously saved session.

        Loads the session's history and restores it to the current client.

        Args:
            session_id: The session ID to resume.

        Returns:
            The loaded SessionState.

        Raises:
            RuntimeError: If no session plugin is configured.
            FileNotFoundError: If the session doesn't exist.
        """
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured. Call set_session_plugin() first.")

        state = self._session_plugin.load(session_id)
        self._restore_session_state(state)
        return state

    def list_sessions(self) -> List[SessionInfo]:
        """List all available sessions.

        Returns:
            List of SessionInfo objects, sorted by updated_at descending.

        Raises:
            RuntimeError: If no session plugin is configured.
        """
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured. Call set_session_plugin() first.")

        return self._session_plugin.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Delete a saved session.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted, False if session didn't exist.

        Raises:
            RuntimeError: If no session plugin is configured.
        """
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured. Call set_session_plugin() first.")

        return self._session_plugin.delete(session_id)

    def _get_session_state(
        self,
        session_id: Optional[str] = None,
        user_inputs: Optional[List[str]] = None
    ) -> SessionState:
        """Build a SessionState from the current client state.

        Args:
            session_id: Optional session ID. Generates one if not provided.
            user_inputs: Optional list of user input strings for history.

        Returns:
            SessionState with current history and metadata.
        """
        # Generate session ID if not provided
        if not session_id:
            # Check if plugin has a current session ID
            if (self._session_plugin and
                    hasattr(self._session_plugin, 'get_current_session_id')):
                session_id = self._session_plugin.get_current_session_id()
            if not session_id:
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        now = datetime.now()
        turn_accounting = self.get_turn_accounting()

        # Get description from session plugin if available
        description = None
        if self._session_plugin and hasattr(self._session_plugin, '_session_description'):
            description = self._session_plugin._session_description

        return SessionState(
            session_id=session_id,
            history=self.get_history(),
            created_at=now,  # Will be overwritten if loading existing
            updated_at=now,
            turn_count=len(turn_accounting),
            turn_accounting=turn_accounting,
            user_inputs=user_inputs or [],
            project=self._project,
            location=self._location,
            model=self._model_name,
            description=description,
        )

    def _restore_session_state(self, state: SessionState) -> None:
        """Restore client state from a SessionState.

        Args:
            state: The SessionState to restore.
        """
        # Restore history
        self.reset_session(state.history)

        # Restore turn accounting
        self._turn_accounting = list(state.turn_accounting)

    def _notify_session_turn_complete(self) -> None:
        """Notify session plugin that a turn completed.

        Called after each send_message() completes.
        """
        if not self._session_plugin or not self._session_config:
            return

        state = self._get_session_state()

        # Increment turn count in plugin for prompt enrichment tracking
        if hasattr(self._session_plugin, 'increment_turn_count'):
            self._session_plugin.increment_turn_count()

        # Call plugin hook
        self._session_plugin.on_turn_complete(state, self._session_config)

    def close_session(self) -> None:
        """Close the current session, triggering auto-save if configured.

        Call this before exiting to ensure session is saved.
        """
        if self._session_plugin and self._session_config:
            state = self._get_session_state()
            self._session_plugin.on_session_end(state, self._session_config)


__all__ = ['JaatoClient', 'MODEL_CONTEXT_LIMITS', 'DEFAULT_CONTEXT_LIMIT']
