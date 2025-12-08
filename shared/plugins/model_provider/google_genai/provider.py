"""Google GenAI (Vertex AI / Gemini) model provider implementation.

This provider encapsulates all interactions with the Google GenAI SDK,
including Vertex AI authentication, chat session management, and
function calling.
"""

from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from ..base import ModelProviderPlugin, ProviderConfig
from ..types import (
    Message,
    ProviderResponse,
    Role,
    ToolResult,
    ToolSchema,
    TokenUsage,
    Part,
)
from .converters import (
    history_from_sdk,
    history_to_sdk,
    message_from_sdk,
    response_from_sdk,
    tool_results_to_sdk_parts,
    tool_schemas_to_sdk_tool,
    serialize_history,
    deserialize_history,
)


# Context window limits for known Gemini models (total tokens)
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

DEFAULT_CONTEXT_LIMIT = 1_048_576


class GoogleGenAIProvider:
    """Google GenAI / Vertex AI model provider.

    This provider supports:
    - Vertex AI authentication via service account or ADC
    - Gemini model family (1.5, 2.0, 2.5)
    - Multi-turn chat with SDK-managed history
    - Function calling with manual control
    - Token counting and context management

    Usage:
        provider = GoogleGenAIProvider()
        provider.initialize(ProviderConfig(
            project='my-project',
            location='us-central1'
        ))
        provider.connect('gemini-2.5-flash')
        provider.create_session(
            system_instruction="You are a helpful assistant.",
            tools=[ToolSchema(name='greet', description='Say hello', parameters={})]
        )
        response = provider.send_message("Hello!")
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[genai.Client] = None
        self._model_name: Optional[str] = None
        self._project: Optional[str] = None
        self._location: Optional[str] = None
        self._chat = None  # genai Chat object

        # Current session configuration
        self._system_instruction: Optional[str] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._last_usage: TokenUsage = TokenUsage()

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "google_genai"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with Vertex AI credentials.

        Args:
            config: Configuration with project and location.
        """
        if config is None:
            config = ProviderConfig()

        self._project = config.project
        self._location = config.location

        # Create the client
        # Vertex AI mode uses ADC or GOOGLE_APPLICATION_CREDENTIALS
        self._client = genai.Client(
            vertexai=True,
            project=config.project,
            location=config.location
        )

    def shutdown(self) -> None:
        """Clean up resources."""
        self._chat = None
        self._client = None
        self._model_name = None

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use.

        Args:
            model: Model name (e.g., 'gemini-2.5-flash').
        """
        self._model_name = model

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models.

        Args:
            prefix: Optional filter prefix (e.g., 'gemini').

        Returns:
            List of model names.
        """
        if not self._client:
            return []

        models = []
        for model in self._client.models.list():
            if prefix and not model.name.startswith(prefix):
                continue
            models.append(model.name)

        return models

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
            tools: List of available tools.
            history: Previous conversation history to restore.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not initialized. Call initialize() and connect() first.")

        self._system_instruction = system_instruction
        self._tools = tools

        # Convert tools to SDK format
        sdk_tool = tool_schemas_to_sdk_tool(tools) if tools else None

        # Build config
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[sdk_tool] if sdk_tool else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        # Convert history to SDK format
        sdk_history = history_to_sdk(history) if history else None

        # Create the chat session
        self._chat = self._client.chats.create(
            model=self._model_name,
            config=config,
            history=sdk_history
        )

    def get_history(self) -> List[Message]:
        """Get the current conversation history.

        Returns:
            List of messages in internal format.
        """
        if not self._chat:
            return []

        sdk_history = list(self._chat.get_history())
        return history_from_sdk(sdk_history)

    # ==================== Messaging ====================

    def send_message(self, message: str) -> ProviderResponse:
        """Send a user message and get a response.

        Args:
            message: The user's message text.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        if not self._chat:
            raise RuntimeError("No chat session. Call create_session() first.")

        response = self._chat.send_message(message)
        provider_response = response_from_sdk(response)
        self._last_usage = provider_response.usage

        return provider_response

    def send_tool_results(self, results: List[ToolResult]) -> ProviderResponse:
        """Send tool execution results back to the model.

        Args:
            results: List of tool execution results.

        Returns:
            ProviderResponse with the model's next response.
        """
        if not self._chat:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Convert results to SDK parts
        sdk_parts = tool_results_to_sdk_parts(results)

        # Send to model
        response = self._chat.send_message(sdk_parts)
        provider_response = response_from_sdk(response)
        self._last_usage = provider_response.usage

        return provider_response

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Count tokens for the given content.

        Args:
            content: Text to count tokens for.

        Returns:
            Token count.
        """
        if not self._client or not self._model_name:
            return 0

        try:
            result = self._client.models.count_tokens(
                model=self._model_name,
                contents=content
            )
            return result.total_tokens
        except Exception:
            # Fallback: rough estimate (4 chars per token)
            return len(content) // 4

    def get_context_limit(self) -> int:
        """Get the context window size for the current model.

        Returns:
            Maximum tokens the model can handle.
        """
        if not self._model_name:
            return DEFAULT_CONTEXT_LIMIT

        # Try exact match first
        if self._model_name in MODEL_CONTEXT_LIMITS:
            return MODEL_CONTEXT_LIMITS[self._model_name]

        # Try prefix match
        for model_prefix, limit in MODEL_CONTEXT_LIMITS.items():
            if self._model_name.startswith(model_prefix):
                return limit

        return DEFAULT_CONTEXT_LIMIT

    def get_token_usage(self) -> TokenUsage:
        """Get token usage from the last response.

        Returns:
            TokenUsage with prompt/output/total counts.
        """
        return self._last_usage

    # ==================== Serialization ====================

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize conversation history to a JSON string.

        Args:
            history: List of messages to serialize.

        Returns:
            JSON string representation.
        """
        return serialize_history(history)

    def deserialize_history(self, data: str) -> List[Message]:
        """Deserialize conversation history from a JSON string.

        Args:
            data: Previously serialized history string.

        Returns:
            List of Message objects.
        """
        return deserialize_history(data)


def create_provider() -> GoogleGenAIProvider:
    """Factory function for plugin discovery."""
    return GoogleGenAIProvider()
