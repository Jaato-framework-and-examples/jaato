"""Base protocol for Model Provider plugins.

This module defines the interface that all model provider plugins must implement.
Model providers encapsulate all SDK-specific logic for interacting with AI models
(Google GenAI, Anthropic, OpenAI, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, runtime_checkable

from .types import (
    Message,
    Part,
    ProviderResponse,
    ToolResult,
    ToolSchema,
    TokenUsage,
)


# Output callback type for real-time streaming
# Parameters: (source: str, text: str, mode: str)
#   source: "model" for model output, plugin name for plugin output
#   text: The output text
#   mode: "write" for new block, "append" to continue
OutputCallback = Callable[[str, str, str], None]


# Authentication method type for Google GenAI provider
GoogleAuthMethod = Literal["auto", "api_key", "service_account_file", "adc", "impersonation"]


@dataclass
class ProviderConfig:
    """Configuration for model provider initialization.

    Providers may use different subsets of these fields depending on
    their authentication requirements.

    Attributes:
        project: Cloud project ID (GCP, AWS, etc.).
        location: Region/location for the service.
        api_key: API key for authentication (if applicable).
        credentials_path: Path to credentials file (if applicable).
        use_vertex_ai: If True, use Vertex AI endpoint (requires project/location).
            If False, use Google AI Studio endpoint (requires api_key).
            Default is True for backwards compatibility.
        auth_method: Authentication method to use. Options:
            - "auto": Automatically detect from available credentials (default)
            - "api_key": Use API key (Google AI Studio)
            - "service_account_file": Use service account JSON file
            - "adc": Use Application Default Credentials
            - "impersonation": Use service account impersonation
        target_service_account: Target service account email for impersonation.
            Required when auth_method is "impersonation".
        credentials: Pre-built credentials object (advanced usage).
            When provided, this takes precedence over other auth methods.
        extra: Provider-specific additional configuration.
    """
    project: Optional[str] = None
    location: Optional[str] = None
    api_key: Optional[str] = None
    credentials_path: Optional[str] = None
    use_vertex_ai: bool = True
    auth_method: GoogleAuthMethod = "auto"
    target_service_account: Optional[str] = None
    credentials: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelProviderPlugin(Protocol):
    """Protocol for Model Provider plugins.

    Model providers encapsulate all interactions with a specific AI SDK:
    - Connection and authentication
    - Chat session management
    - Message sending and function calling
    - Token counting and context management
    - History serialization for persistence

    This follows the same pattern as GCPlugin and SessionPlugin.

    Example implementation:
        class GoogleGenAIProvider:
            @property
            def name(self) -> str:
                return "google_genai"

            def initialize(self, config: ProviderConfig) -> None:
                self._client = genai.Client(
                    vertexai=True,
                    project=config.project,
                    location=config.location
                )

            def connect(self, model: str) -> None:
                self._model_name = model

            def send_message(self, message: str) -> ProviderResponse:
                response = self._chat.send_message(message)
                return self._convert_response(response)
    """

    @property
    def name(self) -> str:
        """Unique identifier for this provider (e.g., 'google_genai', 'anthropic')."""
        ...

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with configuration.

        This is called once when the provider is first set up.
        Establishes the SDK client connection.

        Args:
            config: Provider configuration with auth details.
        """
        ...

    def shutdown(self) -> None:
        """Clean up any resources held by the provider."""
        ...

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use for this provider.

        Args:
            model: Model name/ID (e.g., 'gemini-2.5-flash', 'claude-sonnet-4-5-20250929').
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Check if the provider is connected and ready."""
        ...

    @property
    def model_name(self) -> Optional[str]:
        """Get the currently configured model name."""
        ...

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models from this provider.

        Args:
            prefix: Optional filter prefix (e.g., 'gemini', 'claude').

        Returns:
            List of model names/IDs.
        """
        ...

    # ==================== Session Management ====================

    def create_session(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        history: Optional[List[Message]] = None
    ) -> None:
        """Create or reset the chat session.

        Args:
            system_instruction: System prompt for the model.
            tools: List of available tools/functions.
            history: Previous conversation history to restore.
        """
        ...

    def get_history(self) -> List[Message]:
        """Get the current conversation history.

        Returns:
            List of messages in provider-agnostic format.
        """
        ...

    # ==================== Messaging ====================

    def generate(self, prompt: str) -> ProviderResponse:
        """Simple one-shot generation without session context.

        Use this for basic prompts that don't need conversation history
        or function calling.

        Args:
            prompt: The prompt text.

        Returns:
            ProviderResponse with the model's response.
        """
        ...

    def send_message(
        self,
        message: str,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send a user message and get a response.

        Does NOT automatically execute function calls - that's the
        responsibility of JaatoClient's orchestration loop.

        Args:
            message: The user's message text.
            response_schema: Optional JSON Schema to constrain the model's
                response format. When provided, the model will return JSON
                conforming to this schema, and the response's structured_output
                field will contain the parsed result. Not all providers support
                this - check supports_structured_output() first.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        ...

    def send_message_with_parts(
        self,
        parts: List[Part],
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send a message with multiple parts (text, images, etc.).

        Use this for multimodal input where the user message contains
        more than just text.

        Args:
            parts: List of Part objects forming the message.
            response_schema: Optional JSON Schema to constrain the response.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        ...

    def send_tool_results(
        self,
        results: List[ToolResult],
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send tool execution results back to the model.

        Called after executing function calls to continue the conversation.

        Args:
            results: List of tool execution results.
            response_schema: Optional JSON Schema to constrain the model's
                response format. See send_message() for details.

        Returns:
            ProviderResponse with the model's next response.
        """
        ...

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Count tokens for the given content.

        Args:
            content: Text to count tokens for.

        Returns:
            Token count.
        """
        ...

    def get_context_limit(self) -> int:
        """Get the context window size for the current model.

        Returns:
            Maximum tokens the model can handle.
        """
        ...

    def get_token_usage(self) -> TokenUsage:
        """Get token usage from the last response.

        Returns:
            TokenUsage with prompt/output/total counts.
        """
        ...

    # ==================== Serialization ====================
    # Used by SessionPlugin for persistence

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize conversation history to a string.

        Used by SessionPlugin to persist conversations.
        The format should be provider-independent (e.g., JSON).

        Args:
            history: List of messages to serialize.

        Returns:
            Serialized string representation.
        """
        ...

    def deserialize_history(self, data: str) -> List[Message]:
        """Deserialize conversation history from a string.

        Args:
            data: Previously serialized history string.

        Returns:
            List of Message objects.
        """
        ...

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """Check if this provider supports structured output (response_schema).

        When True, the provider can accept response_schema in send_message()
        and send_tool_results() to constrain the model's output to valid JSON
        matching the provided schema.

        Returns:
            True if structured output is supported.
        """
        ...

    # ==================== Optional Extensions ====================
    # These methods have sensible defaults but can be overridden

    # def supports_streaming(self) -> bool:
    #     """Check if this provider supports streaming responses.
    #
    #     Returns:
    #         True if streaming is supported.
    #     """
    #     ...
    #
    # def send_message_streaming(
    #     self,
    #     message: str,
    #     on_token: Callable[[str], None]
    # ) -> ProviderResponse:
    #     """Send a message with streaming response.
    #
    #     Args:
    #         message: The user's message.
    #         on_token: Callback for each token as it arrives.
    #
    #     Returns:
    #         Final ProviderResponse after streaming completes.
    #     """
    #     ...
