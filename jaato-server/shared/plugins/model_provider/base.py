"""Base protocol for Model Provider plugins.

This module defines the interface that all model provider plugins must implement.
Model providers encapsulate all SDK-specific logic for interacting with AI models
(Google GenAI, Anthropic, OpenAI, etc.).

Providers are stateless with respect to conversation history. The session owns
the canonical message list and passes it to ``complete()`` on each call.
Providers hold only connection/auth state set by ``initialize()`` and
``connect()``.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, runtime_checkable

from jaato_sdk.plugins.model_provider.types import (
    CancelledException,
    CancelToken,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    ThinkingConfig,
    ToolResult,
    ToolSchema,
    TokenUsage,
    TurnResult,
)


# Output callback type for real-time streaming
# Parameters: (source: str, text: str, mode: str)
#   source: "model" for model output, plugin name for plugin output
#   text: The output text
#   mode: "write" for new block, "append" to continue
OutputCallback = Callable[[str, str, str], None]

# Streaming callback type for token-level streaming
# Parameters: (chunk: str) - the text chunk received from the model
StreamingCallback = Callable[[str], None]

# Thinking callback type for extended thinking content
# Parameters: (thinking: str) - accumulated thinking content from the model
# Called BEFORE text streaming begins, when thinking is complete
ThinkingCallback = Callable[[str], None]

# Usage update callback for real-time token accounting
# Parameters: (usage: TokenUsage) - current token usage from streaming
UsageUpdateCallback = Callable[[TokenUsage], None]

# GC threshold callback for proactive garbage collection notifications
# Parameters: (percent_used: float, threshold: float) - current and threshold percentages
# Called when context usage crosses configured threshold during streaming
GCThresholdCallback = Callable[[float, float], None]

# Function call detected callback for streaming
# Parameters: (function_call: FunctionCall) - the function call detected mid-stream
# Called when a function call is detected during streaming, BEFORE any subsequent
# text chunks are emitted. This allows the caller to insert tool tree markers
# at the correct position between text blocks.
FunctionCallDetectedCallback = Callable[[FunctionCall], None]


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
    - Stateless completion via ``complete()``
    - Token counting and context management
    - History serialization for persistence

    Providers are stateless with respect to conversation history. The session
    owns the canonical message list and passes it to ``complete()`` on each
    call. Providers hold only connection/auth state set during lifecycle
    methods.

    Example implementation:
        class MyProvider:
            @property
            def name(self) -> str:
                return "my_provider"

            def initialize(self, config: ProviderConfig) -> None:
                self._client = SomeSDK(api_key=config.api_key)

            def connect(self, model: str) -> None:
                self._model_name = model

            def complete(self, messages, system_instruction=None, tools=None, **kw):
                response = self._client.chat(messages=messages, model=self._model_name)
                return TurnResult.from_provider_response(convert_response(response))
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

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Verify that authentication is configured and optionally trigger interactive login.

        This should be called BEFORE sending messages to ensure
        credentials are available. For providers that support interactive login
        (like Anthropic OAuth), this can trigger the login flow.

        Args:
            allow_interactive: If True and auth is not configured, attempt
                interactive login (e.g., browser-based OAuth). If False,
                only check if credentials exist without prompting.
            on_message: Optional callback for status messages during interactive
                login (e.g., "Opening browser...", "Waiting for auth...").

        Returns:
            True if authentication is configured and valid.
            False if authentication failed or was not completed.

        Raises:
            APIKeyNotFoundError: If allow_interactive=False and no credentials found.
        """
        ...

    def shutdown(self) -> None:
        """Clean up any resources held by the provider."""
        ...

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used.

        Called after ``initialize()`` to describe which auth method or
        credential file was resolved. Displayed in the "Connected to"
        message so users can see where credentials came from — critical
        for diagnosing fallback credential issues across workspaces.

        Returns:
            Human-readable string, e.g. ``"API key from ~/.jaato/zhipuai_auth.json"``,
            ``"PKCE OAuth"``, ``"ADC"``. Empty string if unknown.
        """
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

    # ==================== Stateless Completion ====================

    def complete(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        cancel_token: Optional[CancelToken] = None,
        on_chunk: Optional[StreamingCallback] = None,
        on_usage_update: Optional['UsageUpdateCallback'] = None,
        on_function_call: Optional['FunctionCallDetectedCallback'] = None,
        on_thinking: Optional['ThinkingCallback'] = None,
    ) -> TurnResult:
        """Stateless completion: convert messages to provider format, call API, return response.

        This method does NOT modify any internal state. The caller (session)
        is responsible for maintaining the message list.

        When on_chunk is provided, the response is streamed token-by-token.
        When on_chunk is None, the response is returned in batch mode.

        Providers MUST:

        * Return ``TurnResult.from_provider_response(r)`` on success.
        * Return ``TurnResult.from_exception(exc)`` for **non-transient**
          errors (auth failures, context limits, safety blocks).
        * **Raise** transient errors (rate limits, overload) so the
          ``with_retry`` layer can retry transparently.

        Args:
            messages: Full conversation history in provider-agnostic Message format.
            system_instruction: System prompt text.
            tools: Available tool schemas.
            response_schema: Optional JSON Schema for structured output.
            cancel_token: Optional cancellation signal.
            on_chunk: If provided, enables streaming mode.
            on_usage_update: Real-time token usage callback (streaming).
            on_function_call: Callback when function call detected mid-stream.
            on_thinking: Callback for extended thinking content.

        Returns:
            A ``TurnResult`` classifying the outcome.
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

        When True, the provider can accept response_schema in complete()
        to constrain the model's output to valid JSON matching the provided
        schema.

        Returns:
            True if structured output is supported.
        """
        ...

    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses.

        When True, the provider supports the ``on_chunk`` callback in
        ``complete()`` for real-time token delivery.

        Returns:
            True if streaming is supported, False otherwise.
        """
        ...

    def supports_stop(self) -> bool:
        """Check if this provider supports mid-turn cancellation (stop).

        Stop capability requires streaming support since cancellation is
        implemented by breaking out of the streaming loop.

        Returns:
            True if stop/cancel is supported, False otherwise.
        """
        ...

    # ==================== Agent Context ====================
    # Optional agent identification for tracing

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main"
    ) -> None:
        """Set agent context for trace identification.

        This allows traces to identify which agent (main or subagent)
        produced the trace, useful for debugging multi-agent scenarios.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent (e.g., profile name).
            agent_id: Unique identifier for the agent instance.
        """
        ...

    # ==================== Thinking Mode ====================
    # Optional extended thinking/reasoning support

    def supports_thinking(self) -> bool:
        """Check if this provider/model supports extended thinking.

        Thinking mode enables extended reasoning capabilities:
        - Anthropic: Extended thinking with visible reasoning traces
        - Google Gemini: Thinking mode (Gemini 2.0+)

        Returns:
            True if thinking mode is supported, False otherwise.
        """
        ...

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Set the thinking/reasoning mode configuration.

        Dynamically enables or disables extended thinking for subsequent
        API calls. Takes effect immediately for the next complete() call.

        Args:
            config: ThinkingConfig with enabled flag and budget.
                - enabled: Whether to use thinking mode
                - budget: Token budget for thinking (provider-specific)

        Note:
            If the provider/model doesn't support thinking, this is a no-op.
            Check supports_thinking() to verify capability first.
        """
        ...

    # ==================== Error Classification for Retry ====================
    # Optional methods for provider-specific error handling in retry logic

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry purposes.

        Each provider knows its own error types and can provide precise
        classification. If not implemented, the global fallback in
        retry_utils.classify_error() is used.

        Args:
            exc: The exception to classify.

        Returns:
            Dict with keys:
                - transient: True if error is transient (should retry)
                - rate_limit: True if error is a rate limit (429)
                - infra: True if error is infrastructure (503, 500)
            Or None to use global fallback classification.

        Example implementation:
            def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
                if isinstance(exc, MyRateLimitError):
                    return {"transient": True, "rate_limit": True, "infra": False}
                if isinstance(exc, MyServerError):
                    return {"transient": True, "rate_limit": False, "infra": True}
                return None  # Use fallback
        """
        ...

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract retry-after hint from an exception.

        Many APIs include a Retry-After header or equivalent in their
        rate limit responses. Providers can extract this for better backoff.

        Args:
            exc: The exception to extract retry-after from.

        Returns:
            Suggested delay in seconds, or None if not available.

        Example implementation:
            def get_retry_after(self, exc: Exception) -> Optional[float]:
                if hasattr(exc, 'retry_after'):
                    return float(exc.retry_after)
                return None
        """
        ...
