"""JaatoClient - Core client for the jaato framework.

Provides a unified interface for interacting with Vertex AI models,
with support for tool execution via plugins or custom declarations.
Uses the SDK chat API for multi-turn conversation management.
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from google import genai
from google.genai import types

from .ai_tool_runner import ToolExecutor
from .token_accounting import TokenLedger

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

        # Chat session (SDK-managed)
        self._chat = None  # genai Chat object

        # Tool configuration
        self._executor: Optional[ToolExecutor] = None
        self._tool_decl: Optional[types.Tool] = None
        self._system_instruction: Optional[str] = None
        self._ledger: Optional[TokenLedger] = None

        # Per-turn token accounting
        self._turn_accounting: List[Dict[str, int]] = []

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to Vertex AI."""
        return self._client is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get the configured model name."""
        return self._model_name

    def connect(self, project: str, location: str, model: str) -> None:
        """Connect to Vertex AI.

        Args:
            project: GCP project ID.
            location: Vertex AI region (e.g., 'us-central1', 'global').
            model: Model name (e.g., 'gemini-2.0-flash').
        """
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model_name = model

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

        # Register executors from plugins
        for name, fn in registry.get_exposed_executors().items():
            self._executor.register(name, fn)

        # Set permission plugin for enforcement
        if permission_plugin:
            self._executor.set_permission_plugin(permission_plugin)
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

        return self._run_chat_loop(message)

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

                    # Build function response part
                    func_responses.append(types.Part.from_function_response(
                        name=name,
                        response=result if isinstance(result, dict) else {"result": result}
                    ))

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

                func_responses.append(types.Part.from_function_response(
                    name=name,
                    response=result if isinstance(result, dict) else {"result": result}
                ))

            # Chat API accepts Parts directly (not Content objects)
            response = self._chat.send_message(func_responses)
            self._record_token_usage(response)
            self._accumulate_turn_tokens(response, turn_tokens)

        # Store turn accounting
        self._turn_accounting.append(turn_tokens)

        return response.text if response.text else ''


__all__ = ['JaatoClient']
