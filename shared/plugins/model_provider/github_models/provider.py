"""GitHub Models provider implementation.

This provider enables access to AI models through the GitHub Models API,
supporting multiple models from different providers (OpenAI, Anthropic, Google)
through a unified interface.

Authentication methods:
- Personal Access Token (PAT) with `models: read` scope
- Fine-grained PAT (recommended for enterprise SSO)
- GitHub App token with `models: read` permission

Enterprise features:
- Organization-attributed billing via org-scoped endpoint
- Enterprise policy compliance
- SSO support (fine-grained PATs auto-authorized)
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletionsResponseFormatJSON,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential

from ..base import ModelProviderPlugin, ProviderConfig
from ..types import (
    Message,
    Part,
    ProviderResponse,
    Role,
    ToolResult,
    ToolSchema,
    TokenUsage,
)
from .converters import (
    history_from_sdk,
    history_to_sdk,
    response_from_sdk,
    serialize_history,
    deserialize_history,
    tool_results_to_sdk,
    tool_schemas_to_sdk,
)
from .env import (
    DEFAULT_ENDPOINT,
    resolve_auth_method,
    resolve_endpoint,
    resolve_enterprise,
    resolve_organization,
    resolve_token,
    get_checked_credential_locations,
)
from .errors import (
    ModelNotFoundError,
    ModelsDisabledError,
    RateLimitError,
    TokenInvalidError,
    TokenNotFoundError,
    TokenPermissionError,
)


# Context window limits for known models (total tokens)
# These are approximate - actual limits may vary
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # OpenAI models
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    "openai/gpt-4-turbo": 128_000,
    "openai/gpt-4": 8_192,
    "openai/gpt-3.5-turbo": 16_385,
    "openai/o1-preview": 128_000,
    "openai/o1-mini": 128_000,
    # Anthropic models
    "anthropic/claude-3.5-sonnet": 200_000,
    "anthropic/claude-3-opus": 200_000,
    "anthropic/claude-3-sonnet": 200_000,
    "anthropic/claude-3-haiku": 200_000,
    # Google models
    "google/gemini-1.5-pro": 1_048_576,
    "google/gemini-1.5-flash": 1_048_576,
    # Meta models
    "meta/llama-3.1-405b-instruct": 128_000,
    "meta/llama-3.1-70b-instruct": 128_000,
    "meta/llama-3.1-8b-instruct": 128_000,
    # Mistral models
    "mistral/mistral-large": 32_000,
    "mistral/mistral-small": 32_000,
}

DEFAULT_CONTEXT_LIMIT = 128_000


@dataclass
class GitHubModelsConfig:
    """Configuration for GitHub Models provider.

    Attributes:
        token: GitHub token (PAT or App token) with models:read permission.
        organization: Organization for billing attribution (optional).
        enterprise: Enterprise name for context (optional).
        endpoint: API endpoint URL (defaults to GitHub Models endpoint).
    """
    token: Optional[str] = None
    organization: Optional[str] = None
    enterprise: Optional[str] = None
    endpoint: str = DEFAULT_ENDPOINT


class GitHubModelsProvider:
    """GitHub Models provider.

    This provider supports:
    - Multiple AI models (GPT, Claude, Gemini, Llama, Mistral, etc.)
    - Organization-attributed billing
    - Enterprise policy compliance
    - Function calling with manual control
    - Token counting (estimated)

    Usage:
        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(
            api_key='ghp_your_token',  # Or set GITHUB_TOKEN env var
            extra={
                'organization': 'your-org',  # Optional, for billing
            }
        ))
        provider.connect('openai/gpt-4o')
        response = provider.send_message("Hello!")

    Environment variables:
        GITHUB_TOKEN: Personal access token or app token
        JAATO_GITHUB_ORGANIZATION: Organization for billing attribution
        JAATO_GITHUB_ENTERPRISE: Enterprise name (for context/debugging)
        JAATO_GITHUB_ENDPOINT: Override the API endpoint URL
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[ChatCompletionsClient] = None
        self._model_name: Optional[str] = None

        # Configuration
        self._token: Optional[str] = None
        self._organization: Optional[str] = None
        self._enterprise: Optional[str] = None
        self._endpoint: str = DEFAULT_ENDPOINT

        # Session state
        self._system_instruction: Optional[str] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._history: List[Message] = []
        self._last_usage: TokenUsage = TokenUsage()

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "github_models"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with credentials.

        Supports GitHub PAT and App tokens. Uses fail-fast validation
        to catch configuration errors early.

        Args:
            config: Configuration with authentication details.
                - api_key: GitHub token (PAT or App token)
                - extra['organization']: Organization for billing (optional)
                - extra['enterprise']: Enterprise name (optional)
                - extra['endpoint']: Override API endpoint (optional)

        Raises:
            TokenNotFoundError: No token found.
            TokenInvalidError: Token is invalid or rejected.
            ModelsDisabledError: GitHub Models is disabled for the org/enterprise.
        """
        if config is None:
            config = ProviderConfig()

        # Resolve configuration
        self._token = config.api_key or resolve_token()
        self._organization = config.extra.get('organization') or resolve_organization()
        self._enterprise = config.extra.get('enterprise') or resolve_enterprise()
        self._endpoint = config.extra.get('endpoint') or resolve_endpoint()

        # Validate token
        if not self._token:
            raise TokenNotFoundError(
                auth_method=resolve_auth_method(),
                checked_locations=get_checked_credential_locations(resolve_auth_method()),
            )

        # Create the client
        self._client = self._create_client()

        # Verify connectivity
        self._verify_connectivity()

    def _create_client(self) -> ChatCompletionsClient:
        """Create the ChatCompletionsClient.

        Returns:
            Initialized ChatCompletionsClient.
        """
        return ChatCompletionsClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._token),
        )

    def _verify_connectivity(self) -> None:
        """Verify connectivity by making a lightweight API call.

        Note: The azure-ai-inference SDK doesn't have a list_models endpoint,
        so we skip connectivity verification at init time. Errors will be
        caught on first send_message call.
        """
        # The SDK doesn't provide a lightweight connectivity check
        # Verification happens on first actual API call
        pass

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None
        self._history = []

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use.

        Args:
            model: Model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet').
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
        """List known models.

        Note: GitHub Models API doesn't provide a list endpoint through
        azure-ai-inference, so this returns a static list of known models.

        Args:
            prefix: Optional filter prefix (e.g., 'openai/', 'anthropic/').

        Returns:
            List of model IDs.
        """
        models = list(MODEL_CONTEXT_LIMITS.keys())
        if prefix:
            models = [m for m in models if m.startswith(prefix)]
        return sorted(models)

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
        self._history = list(history) if history else []

    def get_history(self) -> List[Message]:
        """Get the current conversation history.

        Returns:
            List of messages in internal format.
        """
        return list(self._history)

    # ==================== Messaging ====================

    def generate(self, prompt: str) -> ProviderResponse:
        """Simple one-shot generation without session context.

        Args:
            prompt: The prompt text.

        Returns:
            ProviderResponse with the model's response.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not connected. Call connect() first.")

        messages = [UserMessage(content=prompt)]

        try:
            response = self._client.complete(
                model=self._model_name,
                messages=messages,
            )
            provider_response = response_from_sdk(response)
            self._last_usage = provider_response.usage
            return provider_response
        except Exception as e:
            self._handle_api_error(e)
            raise

    def send_message(
        self,
        message: str,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send a user message and get a response.

        Args:
            message: The user's message text.
            response_schema: Optional JSON Schema to constrain the response.
                Note: GitHub Models has limited structured output support.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Build messages list
        messages = self._build_messages()
        messages.append(UserMessage(content=message))

        # Add user message to history
        self._history.append(Message.from_text(Role.USER, message))

        # Build completion kwargs
        kwargs = self._build_completion_kwargs(response_schema)

        try:
            response = self._client.complete(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )
            provider_response = response_from_sdk(response)
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            # Parse structured output if schema was requested
            if response_schema and provider_response.text:
                try:
                    provider_response.structured_output = json.loads(provider_response.text)
                except json.JSONDecodeError:
                    pass

            return provider_response
        except Exception as e:
            self._handle_api_error(e)
            raise

    def send_message_with_parts(
        self,
        parts: List[Part],
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send a message with multiple parts.

        Note: Multimodal support depends on the underlying model.

        Args:
            parts: List of Part objects forming the message.
            response_schema: Optional JSON Schema to constrain the response.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        # For now, extract text content only
        # Full multimodal support would require model-specific handling
        text_parts = [p.text for p in parts if p.text]
        combined_text = "".join(text_parts)

        return self.send_message(combined_text, response_schema)

    def send_tool_results(
        self,
        results: List[ToolResult],
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send tool execution results back to the model.

        Args:
            results: List of tool execution results.
            response_schema: Optional JSON Schema to constrain the response.

        Returns:
            ProviderResponse with the model's next response.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add tool results to history
        for result in results:
            self._history.append(Message(
                role=Role.TOOL,
                parts=[Part(function_response=result)],
            ))

        # Build messages including tool results
        messages = self._build_messages()

        # Build completion kwargs
        kwargs = self._build_completion_kwargs(response_schema)

        try:
            response = self._client.complete(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )
            provider_response = response_from_sdk(response)
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            # Parse structured output if schema was requested
            if response_schema and provider_response.text:
                try:
                    provider_response.structured_output = json.loads(provider_response.text)
                except json.JSONDecodeError:
                    pass

            return provider_response
        except Exception as e:
            self._handle_api_error(e)
            raise

    def _build_messages(self) -> List:
        """Build the messages list for the API call."""
        messages = []

        # Add system instruction if present
        if self._system_instruction:
            messages.append(SystemMessage(content=self._system_instruction))

        # Convert history to SDK format
        messages.extend(history_to_sdk(self._history))

        return messages

    def _build_completion_kwargs(self, response_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build kwargs for the complete() call."""
        kwargs = {}

        # Add tools if configured
        if self._tools:
            sdk_tools = tool_schemas_to_sdk(self._tools)
            if sdk_tools:
                kwargs['tools'] = sdk_tools

        # Add response format for structured output
        if response_schema:
            kwargs['response_format'] = ChatCompletionsResponseFormatJSON()

        return kwargs

    def _add_response_to_history(self, response: ProviderResponse) -> None:
        """Add the model's response to history."""
        parts = []

        if response.text:
            parts.append(Part(text=response.text))

        for fc in response.function_calls:
            parts.append(Part(function_call=fc))

        if parts:
            self._history.append(Message(role=Role.MODEL, parts=parts))

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors and convert to appropriate exceptions."""
        error_str = str(error).lower()

        # Check for disabled error
        if "disabled" in error_str and "models" in error_str:
            raise ModelsDisabledError(
                organization=self._organization,
                enterprise=self._enterprise,
                original_error=str(error),
            ) from error

        # Check for auth errors
        if "401" in error_str or "unauthorized" in error_str:
            if "disabled" in error_str:
                raise ModelsDisabledError(
                    organization=self._organization,
                    enterprise=self._enterprise,
                    original_error=str(error),
                ) from error
            raise TokenInvalidError(
                reason="Token rejected by API",
                token_prefix=self._token[:10] if self._token else None,
                original_error=str(error),
            ) from error

        # Check for permission errors
        if "403" in error_str or "forbidden" in error_str:
            raise TokenPermissionError(
                organization=self._organization,
                enterprise=self._enterprise,
                original_error=str(error),
            ) from error

        # Check for rate limit errors
        if "429" in error_str or "rate limit" in error_str:
            raise RateLimitError(
                original_error=str(error),
            ) from error

        # Check for model not found
        if "404" in error_str or "not found" in error_str:
            raise ModelNotFoundError(
                model=self._model_name or "unknown",
                available_models=self.list_models(),
                original_error=str(error),
            ) from error

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Count tokens for the given content.

        Note: This is an estimate as GitHub Models API doesn't provide
        a token counting endpoint. Uses ~4 chars per token heuristic.

        Args:
            content: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Rough estimate: ~4 characters per token
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Get the context window size for the current model.

        Returns:
            Maximum tokens the model can handle.
        """
        if not self._model_name:
            return DEFAULT_CONTEXT_LIMIT

        # Try exact match
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

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """Check if structured output is supported.

        Note: Structured output support varies by model.
        OpenAI models generally support it, others may not.

        Returns:
            True for OpenAI models, False otherwise.
        """
        if not self._model_name:
            return False
        return self._model_name.startswith("openai/")

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


def create_provider() -> GitHubModelsProvider:
    """Factory function for plugin discovery."""
    return GitHubModelsProvider()
