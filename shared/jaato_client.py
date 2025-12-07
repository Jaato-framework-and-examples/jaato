"""JaatoClient - Core client for the jaato framework.

Provides a unified interface for interacting with Vertex AI models,
with support for tool execution via plugins or custom declarations.
Uses the SDK chat API for multi-turn conversation management.
"""

import re
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from google import genai
from google.genai import types

from .ai_tool_runner import ToolExecutor
from .token_accounting import TokenLedger
from .plugins.base import UserCommand, PromptEnrichmentResult
from .plugins.gc import GCConfig, GCPlugin, GCResult, GCTriggerReason
from .plugins.session import SessionPlugin, SessionConfig, SessionState, SessionInfo

# Pattern to match @references in prompts (e.g., @file.png, @path/to/file.txt)
# Matches @ followed by a path-like string (no spaces, common file chars)
AT_REFERENCE_PATTERN = re.compile(r'@([\w./\-]+(?:\.\w+)?)')

# Context window limits for known Gemini models (total tokens)
# These are approximate limits; actual limits may vary by API version
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # Gemini 2.5 models
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-pro-preview-05-06": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-preview-04-17": 1_048_576,
    # Gemini 2.0 models
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-exp": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    # Gemini 1.5 models
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-pro-latest": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-flash-latest": 1_048_576,
    # Gemini 1.0 models (legacy)
    "gemini-1.0-pro": 32_760,
    "gemini-pro": 32_760,
}

# Default context limit for unknown models
DEFAULT_CONTEXT_LIMIT = 1_048_576

if TYPE_CHECKING:
    from .plugins.registry import PluginRegistry
    from .plugins.permission import PermissionPlugin


class JaatoClient:
    """Core client for jaato framework with SDK-managed conversation history.

    This client provides a unified interface for:
    - Connecting to Vertex AI
    - Configuring tools from plugin registry or custom declarations
    - Multi-turn conversations with SDK-managed history
    - History access and reset for flexibility

    The SDK chat API manages conversation history internally. Use
    get_history() to access it and reset_session() to modify it.

    Usage:
        # Basic setup
        client = JaatoClient()
        client.connect(project_id, location, model_name)
        client.configure_tools(registry, permission_plugin, ledger)

        # Multi-turn conversation (SDK manages history)
        response = client.send_message("Hello!")
        response = client.send_message("Tell me more")

        # Access or reset history when needed
        history = client.get_history()
        client.reset_session()  # Clear history
        client.reset_session(modified_history)  # Reset with custom history
    """

    def __init__(self):
        """Initialize JaatoClient (not yet connected)."""
        # Connection state
        self._client: Optional[genai.Client] = None
        self._model_name: Optional[str] = None
        self._project: Optional[str] = None
        self._location: Optional[str] = None

        # Chat session (SDK-managed)
        self._chat = None  # genai Chat object

        # Tool configuration
        self._executor: Optional[ToolExecutor] = None
        self._tool_decl: Optional[types.Tool] = None
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
        """Check if client is connected to Vertex AI."""
        return self._client is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get the configured model name."""
        return self._model_name

    def list_available_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models from Vertex AI.

        Note: This returns the model catalog, not region-specific availability.
        Some models may not be available in all regions. Use location='global'
        when connecting for widest model access, or check Google's documentation
        for region-specific availability.

        Args:
            prefix: Optional name prefix to filter by (e.g., "gemini").
                    Defaults to None (all models).

        Returns:
            List of model names from the catalog.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")

        models = []
        for model in self._client.models.list():
            # Filter by prefix if specified
            if prefix and not model.name.startswith(prefix):
                continue
            models.append(model.name)

        return models

    def connect(self, project: str, location: str, model: str) -> None:
        """Connect to Vertex AI.

        Args:
            project: GCP project ID.
            location: Vertex AI region (e.g., 'us-central1', 'global').
            model: Model name (e.g., 'gemini-2.0-flash').
        """
        self._client = genai.Client(vertexai=True, project=project, location=location)
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

        # Build tool declarations
        all_decls = registry.get_exposed_declarations()
        if permission_plugin:
            all_decls.extend(permission_plugin.get_function_declarations())
        self._tool_decl = types.Tool(function_declarations=all_decls) if all_decls else None

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

        # Create chat session with configured tools
        self._create_chat()

    def configure_custom_tools(
        self,
        declarations: List[types.FunctionDeclaration],
        executors: Dict[str, Callable[[Dict[str, Any]], Any]],
        ledger: Optional[TokenLedger] = None,
        system_instruction: Optional[str] = None
    ) -> None:
        """Configure tools directly without plugin registry.

        Use this for specialized scenarios where custom tool declarations
        are needed (e.g., training pipelines with specific function calling).

        Args:
            declarations: List of FunctionDeclaration objects.
            executors: Dict mapping tool names to executor functions.
            ledger: Optional token ledger for accounting.
            system_instruction: Optional system instruction text.
        """
        self._ledger = ledger
        self._executor = ToolExecutor(ledger=ledger)

        for name, fn in executors.items():
            self._executor.register(name, fn)

        self._tool_decl = types.Tool(function_declarations=declarations) if declarations else None
        self._system_instruction = system_instruction

        # Create chat session with configured tools
        self._create_chat()

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

    def _create_chat(self, history: Optional[List[types.Content]] = None) -> None:
        """Create or recreate the chat session.

        Args:
            history: Optional initial conversation history.
        """
        if not self._client or not self._model_name:
            return  # Can't create chat without connection

        config = types.GenerateContentConfig(
            system_instruction=self._system_instruction,
            tools=[self._tool_decl] if self._tool_decl else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        self._chat = self._client.chats.create(
            model=self._model_name,
            config=config,
            history=history
        )

    def send_message(self, message: str) -> str:
        """Send a message to the model.

        The SDK manages conversation history internally. Use get_history()
        to access it and reset_session() to modify it.

        If a GC plugin is configured with check_before_send=True, this will
        automatically check and perform garbage collection if needed before
        sending the message.

        If plugins are configured that subscribe to prompt enrichment, the
        message will be passed through them before sending. After enrichment,
        any @references are stripped from the message.

        Args:
            message: The user's message text.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If client is not connected or not configured.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Client not connected. Call connect() first.")
        if not self._chat:
            raise RuntimeError("Tools not configured. Call configure_tools() first.")

        # Check and perform GC if needed before sending
        if self._gc_plugin and self._gc_config and self._gc_config.check_before_send:
            self._maybe_collect_before_send()

        # Run prompt enrichment pipeline if registry is configured
        processed_message = self._enrich_and_clean_prompt(message)

        response = self._run_chat_loop(processed_message)

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

    def _run_chat_loop(self, message: str) -> str:
        """Internal function calling loop using chat.send_message().

        Args:
            message: The user's message text.

        Returns:
            The final response text after all function calls are resolved.
        """
        # Track tokens for this turn
        turn_tokens = {'prompt': 0, 'output': 0, 'total': 0}
        response = None

        try:
            response = self._chat.send_message(message)
            self._record_token_usage(response)
            self._accumulate_turn_tokens(response, turn_tokens)

            # Handle function calling loop
            while response.function_calls:
                func_responses = []

                for fc in response.function_calls:
                    # Execute the function
                    name = fc.name
                    args = dict(fc.args) if fc.args else {}

                    if self._executor:
                        result = self._executor.execute(name, args)
                    else:
                        result = {"error": f"No executor registered for {name}"}

                    # Build function response part(s)
                    parts = self._build_function_response_parts(name, result)
                    func_responses.extend(parts)

                # Send function responses back to model
                # Chat API accepts Parts directly (not Content objects)
                response = self._chat.send_message(func_responses)
                self._record_token_usage(response)
                self._accumulate_turn_tokens(response, turn_tokens)

            return response.text if response.text else ''

        finally:
            # Always store turn accounting, even on errors
            if turn_tokens['total'] > 0:
                self._turn_accounting.append(turn_tokens)

    def _build_function_response_parts(
        self,
        name: str,
        result: Any
    ) -> List[types.Part]:
        """Build function response parts, handling multimodal content.

        If the result contains multimodal data (indicated by '_multimodal': True),
        this builds a multimodal function response with image data.

        Args:
            name: The function name.
            result: The function result (tuple from executor: (ok, result_dict)).

        Returns:
            List of Part objects for the function response.
        """
        # Executor returns (ok, result_dict) tuple
        if isinstance(result, tuple) and len(result) == 2:
            _ok, result_data = result
        else:
            result_data = result

        # Check for multimodal result
        if isinstance(result_data, dict) and result_data.get('_multimodal'):
            return self._build_multimodal_function_response(name, result_data)

        # Standard text/JSON response
        response_dict = result_data if isinstance(result_data, dict) else {"result": result_data}
        return [types.Part.from_function_response(name=name, response=response_dict)]

    def _build_multimodal_function_response(
        self,
        name: str,
        result: Dict[str, Any]
    ) -> List[types.Part]:
        """Build a multimodal function response with image data.

        For Gemini 3 Pro+, this creates a function response that includes
        inline image data that the model can "see".

        Args:
            name: The function name.
            result: Dict with '_multimodal': True and image data.

        Returns:
            List of Part objects including function response and image data.
        """
        multimodal_type = result.get('_multimodal_type', 'image')

        if multimodal_type == 'image':
            image_data = result.get('image_data')
            mime_type = result.get('mime_type', 'image/png')
            display_name = result.get('display_name', 'image')

            if not image_data:
                # Fallback to text response if no image data
                return [types.Part.from_function_response(
                    name=name,
                    response={'error': 'No image data available'}
                )]

            # Build multimodal response
            # The function response references the image by display_name
            # and the image is included as a separate inline data part
            try:
                # Create the structured response with reference to image
                response_dict = {
                    'status': 'success',
                    'image': {'$ref': display_name},
                    'file_path': result.get('file_path', ''),
                    'size_bytes': result.get('size_bytes', len(image_data)),
                }

                # For Gemini 3+, we can include multimodal parts in the function response
                # The SDK's FunctionResponsePart supports file_data for multimodal content
                # However, the exact API may vary - try the standard approach first

                # Approach 1: Include image as inline_data Part alongside function response
                # This is a workaround that may work with Gemini 2.x as well
                parts = [
                    types.Part.from_function_response(
                        name=name,
                        response=response_dict
                    ),
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type=mime_type
                    )
                ]
                return parts

            except Exception as e:
                # Fallback: return text description if multimodal fails
                return [types.Part.from_function_response(
                    name=name,
                    response={
                        'error': f'Failed to build multimodal response: {e}',
                        'file_path': result.get('file_path', ''),
                    }
                )]

        # Unknown multimodal type
        return [types.Part.from_function_response(
            name=name,
            response={'error': f'Unknown multimodal type: {multimodal_type}'}
        )]

    def _accumulate_turn_tokens(self, response, turn_tokens: Dict[str, int]) -> None:
        """Accumulate token counts from response into turn totals."""
        usage = getattr(response, 'usage_metadata', None)
        if usage:
            turn_tokens['prompt'] += getattr(usage, 'prompt_token_count', 0) or 0
            turn_tokens['output'] += getattr(usage, 'candidates_token_count', 0) or 0
            turn_tokens['total'] += getattr(usage, 'total_token_count', 0) or 0

    def _record_token_usage(self, response) -> None:
        """Record token usage from response to ledger if available."""
        if not self._ledger:
            return

        usage = getattr(response, 'usage_metadata', None)
        if usage:
            self._ledger._record('response', {
                'prompt_tokens': getattr(usage, 'prompt_token_count', None),
                'output_tokens': getattr(usage, 'candidates_token_count', None),
                'total_tokens': getattr(usage, 'total_token_count', None),
            })

    def get_history(self) -> List[types.Content]:
        """Get current conversation history from the SDK.

        Returns:
            List of Content objects representing the conversation.
        """
        if not self._chat:
            return []
        return list(self._chat.get_history())

    def get_turn_accounting(self) -> List[Dict[str, int]]:
        """Get token usage per turn.

        Each entry corresponds to one send_message() call and contains
        aggregated tokens across all API calls in that turn (including
        function calling loops).

        Returns:
            List of dicts with 'prompt', 'output', 'total' token counts.
        """
        return list(self._turn_accounting)

    def get_context_limit(self) -> int:
        """Get the context window limit for the current model.

        Returns:
            The context window size in tokens.
        """
        if not self._model_name:
            return DEFAULT_CONTEXT_LIMIT

        # Try exact match first
        if self._model_name in MODEL_CONTEXT_LIMITS:
            return MODEL_CONTEXT_LIMITS[self._model_name]

        # Try prefix matching for versioned model names
        for model_prefix, limit in MODEL_CONTEXT_LIMITS.items():
            if self._model_name.startswith(model_prefix):
                return limit

        return DEFAULT_CONTEXT_LIMIT

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

    def reset_session(self, history: Optional[List[types.Content]] = None) -> None:
        """Reset the chat session, optionally with modified history.

        Use this to:
        - Clear conversation history: reset_session()
        - Start with custom history: reset_session(modified_history)

        Args:
            history: Optional initial history for the new session.
        """
        self._turn_accounting = []
        self._create_chat(history)

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
        if cmd.share_with_model and self._chat:
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
        user_content = types.Content(
            role='user',
            parts=[types.Part(text=f"[User executed command: {command_name}]")]
        )

        # Create a model message with the function call and response
        # This simulates the model having called the function
        model_content = types.Content(
            role='model',
            parts=[types.Part.from_function_response(
                name=command_name,
                response=result if isinstance(result, dict) else {"result": result}
            )]
        )

        # Recreate chat with updated history
        new_history = list(current_history) + [user_content, model_content]
        self._create_chat(new_history)

    def generate(
        self,
        prompt: str,
        ledger: Optional[TokenLedger] = None
    ) -> str:
        """Simple generation without tools.

        Use this for basic prompts that don't need function calling.

        Args:
            prompt: The prompt text.
            ledger: Optional token ledger for accounting.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If client is not connected.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Client not connected. Call connect() first.")

        if ledger:
            response = ledger.generate_with_accounting(self._client, self._model_name, prompt)
        else:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt
            )

        return getattr(response, 'text', '') or ''

    def send_message_with_parts(self, parts: List[types.Part]) -> str:
        """Send a message with custom Part objects.

        Similar to send_message but allows sending multi-modal content
        (images, etc.) via Part objects.

        Args:
            parts: List of Part objects forming the user's message.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If client is not connected or not configured.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Client not connected. Call connect() first.")
        if not self._chat:
            raise RuntimeError("Tools not configured. Call configure_tools() first.")

        return self._run_chat_loop_with_parts(parts)

    def _run_chat_loop_with_parts(self, parts: List[types.Part]) -> str:
        """Internal function calling loop for multi-part messages.

        Args:
            parts: List of Part objects forming the user's message.

        Returns:
            The final response text after all function calls are resolved.
        """
        # Track tokens for this turn
        turn_tokens = {'prompt': 0, 'output': 0, 'total': 0}

        # Send parts as Content object
        user_content = types.Content(role='user', parts=parts)
        response = self._chat.send_message(user_content)
        self._record_token_usage(response)
        self._accumulate_turn_tokens(response, turn_tokens)

        # Handle function calling loop (same as _run_chat_loop)
        while response.function_calls:
            func_responses = []

            for fc in response.function_calls:
                name = fc.name
                args = dict(fc.args) if fc.args else {}

                if self._executor:
                    result = self._executor.execute(name, args)
                else:
                    result = {"error": f"No executor registered for {name}"}

                # Build function response part(s), handling multimodal
                response_parts = self._build_function_response_parts(name, result)
                func_responses.extend(response_parts)

            # Chat API accepts Parts directly (not Content objects)
            response = self._chat.send_message(func_responses)
            self._record_token_usage(response)
            self._accumulate_turn_tokens(response, turn_tokens)

        # Store turn accounting
        self._turn_accounting.append(turn_tokens)

        return response.text if response.text else ''

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

        # Add session plugin's function declarations to chat tools
        if hasattr(plugin, 'get_function_declarations'):
            session_decls = plugin.get_function_declarations()
            if session_decls:
                # Rebuild tool declarations including session plugin's
                current_decls = []
                if self._tool_decl and self._tool_decl.function_declarations:
                    current_decls = list(self._tool_decl.function_declarations)
                current_decls.extend(session_decls)
                self._tool_decl = types.Tool(function_declarations=current_decls)

                # Recreate chat with updated tools
                history = self.get_history() if self._chat else None
                self._create_chat(history)

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

    def save_session(self, session_id: Optional[str] = None) -> str:
        """Save the current session.

        Args:
            session_id: Optional session ID. If not provided, generates one
                       from the current timestamp.

        Returns:
            The session ID that was saved.

        Raises:
            RuntimeError: If no session plugin is configured.
        """
        if not self._session_plugin:
            raise RuntimeError("No session plugin configured. Call set_session_plugin() first.")

        state = self._get_session_state(session_id)
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

    def _get_session_state(self, session_id: Optional[str] = None) -> SessionState:
        """Build a SessionState from the current client state.

        Args:
            session_id: Optional session ID. Generates one if not provided.

        Returns:
            SessionState with current history and metadata.
        """
        from datetime import datetime

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

        return SessionState(
            session_id=session_id,
            history=self.get_history(),
            created_at=now,  # Will be overwritten if loading existing
            updated_at=now,
            turn_count=len(turn_accounting),
            turn_accounting=turn_accounting,
            project=self._project,
            location=self._location,
            model=self._model_name,
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
