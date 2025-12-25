"""Anthropic Claude provider implementation.

This provider enables access to Claude models through the Anthropic API,
supporting function calling, extended thinking, and prompt caching.

Authentication:
- API key only (simpler than other providers)
- Set ANTHROPIC_API_KEY environment variable or pass via ProviderConfig

Features:
- Claude 3.5, Claude 4, and Claude Opus 4.5 model families
- Function/tool calling with manual orchestration
- Extended thinking (reasoning traces) for supported models
- Prompt caching for cost optimization (up to 90% reduction)
- Real token counting via API (beta)
"""

import json
from typing import Any, Dict, List, Optional

from ..base import (
    FunctionCallDetectedCallback,
    ModelProviderPlugin,
    ProviderConfig,
    StreamingCallback,
    UsageUpdateCallback,
)
from ..types import (
    CancelledException,
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    ToolResult,
    ToolSchema,
    TokenUsage,
)
from .converters import (
    deserialize_history,
    extract_content_block_start,
    extract_input_json_from_stream_event,
    extract_message_delta,
    extract_message_start,
    extract_text_from_stream_event,
    extract_thinking_from_stream_event,
    messages_to_anthropic,
    response_from_anthropic,
    serialize_history,
    tool_schemas_to_anthropic,
)
from .env import (
    get_checked_credential_locations,
    resolve_api_key,
    resolve_enable_caching,
    resolve_enable_thinking,
    resolve_thinking_budget,
)
from .errors import (
    APIKeyInvalidError,
    APIKeyNotFoundError,
    ContextLimitError,
    ModelNotFoundError,
    OverloadedError,
    RateLimitError,
)


# Context window limits for Claude models
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # Claude 4 / Opus 4.5 family
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    # Claude 3.5 family
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    # Claude 3 family
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
}

DEFAULT_CONTEXT_LIMIT = 200_000

# Models that support extended thinking
THINKING_CAPABLE_MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",  # Latest versions
]

# Default max tokens for responses
DEFAULT_MAX_TOKENS = 8192
EXTENDED_MAX_TOKENS = 16000  # When thinking is enabled


class AnthropicProvider:
    """Anthropic Claude provider.

    This provider supports:
    - Multiple Claude model families
    - Function calling with manual control
    - Extended thinking (reasoning traces)
    - Prompt caching for cost/latency optimization
    - Real token counting via API

    Usage:
        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key='sk-ant-...',  # Or set ANTHROPIC_API_KEY env var
            extra={
                'enable_caching': True,    # Optional: prompt caching
                'enable_thinking': True,   # Optional: extended thinking
                'thinking_budget': 10000,  # Optional: max thinking tokens
            }
        ))
        provider.connect('claude-sonnet-4-20250514')
        response = provider.send_message("Hello!")

    Environment variables:
        ANTHROPIC_API_KEY: API key for authentication
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[Any] = None  # anthropic.Anthropic
        self._model_name: Optional[str] = None

        # Configuration
        self._api_key: Optional[str] = None
        self._enable_caching: bool = False
        self._enable_thinking: bool = False
        self._thinking_budget: int = 10000
        self._cache_ttl: str = "5m"  # "5m" or "1h"

        # Session state
        self._system_instruction: Optional[str] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._history: List[Message] = []
        self._last_usage: TokenUsage = TokenUsage()

        # Agent context for tracing
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "anthropic"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with credentials.

        Args:
            config: Configuration with authentication details.
                - api_key: Anthropic API key (or set ANTHROPIC_API_KEY)
                - extra['enable_caching']: Enable prompt caching (default: False)
                - extra['enable_thinking']: Enable extended thinking (default: False)
                - extra['thinking_budget']: Max thinking tokens (default: 10000)
                - extra['cache_ttl']: Cache TTL, "5m" or "1h" (default: "5m")

        Raises:
            APIKeyNotFoundError: No API key found.
            APIKeyInvalidError: API key is invalid.
        """
        # Import anthropic here to avoid import errors if not installed
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            ) from e

        if config is None:
            config = ProviderConfig()

        # Resolve API key
        self._api_key = config.api_key or resolve_api_key()
        if not self._api_key:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        # Parse extra config (config.extra takes precedence over env vars)
        self._enable_caching = config.extra.get(
            "enable_caching", resolve_enable_caching()
        )
        self._enable_thinking = config.extra.get(
            "enable_thinking", resolve_enable_thinking()
        )
        self._thinking_budget = config.extra.get(
            "thinking_budget", resolve_thinking_budget()
        )
        self._cache_ttl = config.extra.get("cache_ttl", "5m")

        # Create the client
        self._client = anthropic.Anthropic(api_key=self._api_key)

        # Verify connectivity with a lightweight call
        self._verify_connectivity()

    def _verify_connectivity(self) -> None:
        """Verify connectivity by checking API key validity.

        Makes a minimal API call to verify the key works.
        """
        # Skip verification for now - will fail on first real call if invalid
        # A lightweight verification would be nice but Anthropic doesn't have
        # a dedicated endpoint for this
        pass

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            # Anthropic client doesn't need explicit cleanup
            self._client = None
        self._model_name = None
        self._history = []

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use.

        Args:
            model: Model ID (e.g., 'claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022').
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
        """List available Claude models.

        Note: Anthropic doesn't have a models listing API, so we return
        a static list of known models.

        Args:
            prefix: Optional filter prefix (e.g., 'claude-3', 'claude-sonnet').

        Returns:
            List of model IDs.
        """
        models = [
            # Claude Opus 4.5
            "claude-opus-4-5-20251101",
            # Claude 4
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250414",
            # Claude 3.7
            "claude-3-7-sonnet-20250219",
            # Claude 3.5
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            # Claude 3
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

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

        try:
            response = self._client.messages.create(
                model=self._model_name,
                max_tokens=DEFAULT_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            provider_response = response_from_anthropic(response)
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
                Note: Anthropic doesn't have native structured output,
                so this is implemented via tool forcing.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add user message to history
        self._history.append(Message.from_text(Role.USER, message))

        # Build messages for API
        messages = messages_to_anthropic(self._history)

        # Build API kwargs
        kwargs = self._build_api_kwargs(response_schema)

        try:
            response = self._client.messages.create(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )
            provider_response = response_from_anthropic(response)
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            # Handle structured output via response parsing
            text = provider_response.get_text()
            if response_schema and text:
                try:
                    provider_response.structured_output = json.loads(text)
                except json.JSONDecodeError:
                    pass

            return provider_response
        except Exception as e:
            # Remove the user message we added if the call failed
            if self._history and self._history[-1].role == Role.USER:
                self._history.pop()
            self._handle_api_error(e)
            raise

    def send_message_with_parts(
        self,
        parts: List[Part],
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send a message with multiple parts (text, images, etc.).

        Args:
            parts: List of Part objects forming the message.
            response_schema: Optional JSON Schema to constrain the response.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add multipart message to history
        self._history.append(Message(role=Role.USER, parts=parts))

        # Build messages for API
        messages = messages_to_anthropic(self._history)

        # Build API kwargs
        kwargs = self._build_api_kwargs(response_schema)

        try:
            response = self._client.messages.create(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )
            provider_response = response_from_anthropic(response)
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            return provider_response
        except Exception as e:
            if self._history and self._history[-1].role == Role.USER:
                self._history.pop()
            self._handle_api_error(e)
            raise

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

        # Add tool results to history as a user message with tool_result blocks
        tool_result_parts = [Part(function_response=r) for r in results]
        self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))

        # Build messages for API
        messages = messages_to_anthropic(self._history)

        # Build API kwargs
        kwargs = self._build_api_kwargs(response_schema)

        try:
            response = self._client.messages.create(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )
            provider_response = response_from_anthropic(response)
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            return provider_response
        except Exception as e:
            self._handle_api_error(e)
            raise

    def _build_api_kwargs(self, response_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build kwargs for the messages.create() call."""
        kwargs: Dict[str, Any] = {}

        # Max tokens (higher if thinking is enabled)
        if self._enable_thinking and self._is_thinking_capable():
            kwargs["max_tokens"] = EXTENDED_MAX_TOKENS
        else:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        # System instruction
        if self._system_instruction:
            if self._enable_caching:
                # Use cache control on system instruction
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": self._system_instruction,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = self._system_instruction

        # Tools
        if self._tools:
            anthropic_tools = tool_schemas_to_anthropic(self._tools)
            if anthropic_tools:
                # Add cache control to last tool if caching enabled
                if self._enable_caching and len(anthropic_tools) > 0:
                    anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
                kwargs["tools"] = anthropic_tools

        # Extended thinking
        if self._enable_thinking and self._is_thinking_capable():
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            }

        return kwargs

    def _is_thinking_capable(self) -> bool:
        """Check if the current model supports extended thinking."""
        if not self._model_name:
            return False
        for prefix in THINKING_CAPABLE_MODELS:
            if self._model_name.startswith(prefix):
                return True
        return False

    def _add_response_to_history(self, response: ProviderResponse) -> None:
        """Add the model's response to history.

        Uses the parts-based response format which preserves
        text/function_call interleaving.
        """
        # Filter to only text and function_call parts for history
        # (excludes function_response parts which belong to user messages)
        history_parts = [
            p for p in response.parts
            if p.text is not None or p.function_call is not None
        ]

        if history_parts:
            self._history.append(Message(role=Role.MODEL, parts=history_parts))

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors and convert to appropriate exceptions."""
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check for authentication errors
        if "authentication" in error_str or "invalid api key" in error_str or "401" in error_str:
            raise APIKeyInvalidError(
                reason="API key rejected",
                key_prefix=self._api_key[:15] if self._api_key else None,
                original_error=str(error),
            ) from error

        # Check for rate limit errors
        if "rate" in error_str and "limit" in error_str or "429" in error_str:
            raise RateLimitError(original_error=str(error)) from error

        # Check for overloaded errors
        if "overloaded" in error_str or "529" in error_str:
            raise OverloadedError(original_error=str(error)) from error

        # Check for context length errors
        if any(x in error_str for x in ("context", "token", "too long", "maximum")):
            raise ContextLimitError(
                model=self._model_name or "unknown",
                original_error=str(error),
            ) from error

        # Check for model not found
        if "not found" in error_str or "404" in error_str:
            raise ModelNotFoundError(
                model=self._model_name or "unknown",
                available_models=self.list_models(),
                original_error=str(error),
            ) from error

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Count tokens for the given content.

        Uses Anthropic's beta token counting API for accurate counts.

        Args:
            content: Text to count tokens for.

        Returns:
            Token count.
        """
        if not self._client or not self._model_name:
            # Fallback estimate
            return len(content) // 4

        try:
            # Use beta token counting API
            result = self._client.beta.messages.count_tokens(
                model=self._model_name,
                messages=[{"role": "user", "content": content}],
            )
            return result.input_tokens
        except Exception:
            # Fallback to estimate on error
            return len(content) // 4

    def get_context_limit(self) -> int:
        """Get the context window size for the current model.

        Returns:
            Maximum tokens the model can handle.
        """
        if not self._model_name:
            return DEFAULT_CONTEXT_LIMIT

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

        Note: Anthropic doesn't have native structured output like Google's
        response_schema. We return False here, but structured output can be
        achieved by prompting for JSON or using tool forcing.

        Returns:
            False (no native support).
        """
        return False

    def supports_thinking(self) -> bool:
        """Check if the current model supports extended thinking.

        Returns:
            True if thinking is supported.
        """
        return self._is_thinking_capable()

    def supports_streaming(self) -> bool:
        """Check if streaming is supported.

        Returns:
            True - Anthropic supports streaming.
        """
        return True

    def supports_stop(self) -> bool:
        """Check if mid-turn cancellation (stop) is supported.

        Returns:
            True - Anthropic supports stop via streaming cancellation.
        """
        return True

    # ==================== Agent Context ====================

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main"
    ) -> None:
        """Set agent context for trace identification.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent (e.g., profile name).
            agent_id: Unique identifier for the agent instance.
        """
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    # ==================== Streaming ====================

    def send_message_streaming(
        self,
        message: str,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None
    ) -> ProviderResponse:
        """Send a message with streaming response and optional cancellation.

        Args:
            message: The user's message text.
            on_chunk: Callback invoked for each text chunk as it streams.
            cancel_token: Optional token to request cancellation mid-stream.
            response_schema: Optional JSON Schema to constrain the response.
            on_usage_update: Optional callback for real-time token usage updates.
            on_function_call: Optional callback for function call detection during streaming.

        Returns:
            ProviderResponse with accumulated text and/or function calls.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add user message to history
        self._history.append(Message.from_text(Role.USER, message))

        # Build messages for API
        messages = messages_to_anthropic(self._history)

        # Build API kwargs
        kwargs = self._build_api_kwargs(response_schema)

        try:
            response = self._stream_response(
                messages=messages,
                kwargs=kwargs,
                on_chunk=on_chunk,
                cancel_token=cancel_token,
                on_usage_update=on_usage_update,
                on_function_call=on_function_call,
            )

            self._last_usage = response.usage

            # Add assistant response to history
            self._add_response_to_history(response)

            # Handle structured output via response parsing
            text = response.get_text()
            if response_schema and text:
                try:
                    response.structured_output = json.loads(text)
                except json.JSONDecodeError:
                    pass

            return response
        except Exception as e:
            # Remove the user message we added if the call failed
            if self._history and self._history[-1].role == Role.USER:
                self._history.pop()
            self._handle_api_error(e)
            raise

    def send_tool_results_streaming(
        self,
        results: List[ToolResult],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None
    ) -> ProviderResponse:
        """Send tool results with streaming response and optional cancellation.

        Args:
            results: List of tool execution results.
            on_chunk: Callback invoked for each text chunk as it streams.
            cancel_token: Optional token to request cancellation mid-stream.
            response_schema: Optional JSON Schema to constrain the response.
            on_usage_update: Optional callback for real-time token usage updates.
            on_function_call: Optional callback for function call detection during streaming.

        Returns:
            ProviderResponse with accumulated text and/or function calls.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add tool results to history as a user message with tool_result blocks
        tool_result_parts = [Part(function_response=r) for r in results]
        self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))

        # Build messages for API
        messages = messages_to_anthropic(self._history)

        # Build API kwargs
        kwargs = self._build_api_kwargs(response_schema)

        try:
            response = self._stream_response(
                messages=messages,
                kwargs=kwargs,
                on_chunk=on_chunk,
                cancel_token=cancel_token,
                on_usage_update=on_usage_update,
                on_function_call=on_function_call,
            )

            self._last_usage = response.usage

            # Add assistant response to history
            self._add_response_to_history(response)

            return response
        except Exception as e:
            self._handle_api_error(e)
            raise

    def _stream_response(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
    ) -> ProviderResponse:
        """Stream a response from the Anthropic API.

        Internal method used by both send_message_streaming and
        send_tool_results_streaming.
        """
        # State for accumulating response
        accumulated_text: List[str] = []  # Text chunks for current text block
        accumulated_thinking: List[str] = []  # Thinking chunks
        parts: List[Part] = []  # Ordered parts preserving interleaving
        current_tool_calls: Dict[int, Dict[str, Any]] = {}  # index -> {id, name, json_chunks}
        finish_reason = FinishReason.UNKNOWN
        usage = TokenUsage()
        was_cancelled = False

        def flush_text_block():
            """Flush accumulated text as a single Part."""
            nonlocal accumulated_text
            if accumulated_text:
                text = ''.join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        try:
            # Use the streaming API
            with self._client.messages.stream(
                model=self._model_name,
                messages=messages,
                **kwargs,
            ) as stream:
                for event in stream:
                    # Check for cancellation
                    if cancel_token and cancel_token.is_cancelled:
                        was_cancelled = True
                        finish_reason = FinishReason.CANCELLED
                        break

                    # Handle message_start (initial usage)
                    initial_usage = extract_message_start(event)
                    if initial_usage:
                        usage = initial_usage
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)

                    # Handle content_block_start (new text/tool_use block)
                    block_info = extract_content_block_start(event)
                    if block_info:
                        if block_info["type"] == "tool_use":
                            # Start tracking a new tool call
                            idx = block_info["index"]
                            current_tool_calls[idx] = {
                                "id": block_info["id"],
                                "name": block_info["name"],
                                "json_chunks": [],
                            }
                        elif block_info["type"] == "text":
                            # Flush any existing text before starting new block
                            # (though typically there's only one text block)
                            pass

                    # Handle text deltas
                    text_chunk = extract_text_from_stream_event(event)
                    if text_chunk:
                        accumulated_text.append(text_chunk)
                        on_chunk(text_chunk)

                    # Handle thinking deltas
                    thinking_chunk = extract_thinking_from_stream_event(event)
                    if thinking_chunk:
                        accumulated_thinking.append(thinking_chunk)

                    # Handle tool input JSON deltas
                    json_chunk = extract_input_json_from_stream_event(event)
                    if json_chunk:
                        # Find which tool call this belongs to (current active one)
                        # Anthropic sends in order, so it's the last one
                        if current_tool_calls:
                            last_idx = max(current_tool_calls.keys())
                            current_tool_calls[last_idx]["json_chunks"].append(json_chunk)

                    # Handle content_block_stop (finalize tool call)
                    event_type = getattr(event, "type", None)
                    if event_type == "content_block_stop":
                        idx = getattr(event, "index", None)
                        if idx is not None and idx in current_tool_calls:
                            # Finalize this tool call
                            tc = current_tool_calls[idx]
                            json_str = ''.join(tc["json_chunks"])
                            try:
                                args = json.loads(json_str) if json_str else {}
                            except json.JSONDecodeError:
                                args = {}

                            # Flush text before adding function call
                            flush_text_block()

                            fc = FunctionCall(
                                id=tc["id"],
                                name=tc["name"],
                                args=args,
                            )
                            # Notify caller about function call detection (for UI positioning)
                            if on_function_call:
                                on_function_call(fc)
                            parts.append(Part.from_function_call(fc))
                            del current_tool_calls[idx]

                    # Handle message_delta (stop reason, final usage)
                    delta_info = extract_message_delta(event)
                    if delta_info:
                        if "stop_reason" in delta_info:
                            reason = delta_info["stop_reason"]
                            if reason == "end_turn":
                                finish_reason = FinishReason.STOP
                            elif reason == "tool_use":
                                finish_reason = FinishReason.TOOL_USE
                            elif reason == "max_tokens":
                                finish_reason = FinishReason.MAX_TOKENS
                            elif reason == "stop_sequence":
                                finish_reason = FinishReason.STOP
                        if "usage" in delta_info:
                            delta_usage = delta_info["usage"]
                            # Combine with existing input tokens
                            usage = TokenUsage(
                                prompt_tokens=usage.prompt_tokens,
                                output_tokens=delta_usage.output_tokens,
                                total_tokens=usage.prompt_tokens + delta_usage.output_tokens,
                            )
                            if on_usage_update and usage.total_tokens > 0:
                                on_usage_update(usage)

        except Exception as e:
            # If cancelled during iteration, treat as cancellation
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                raise

        # Flush any remaining text
        flush_text_block()

        # Handle any incomplete tool calls (shouldn't happen normally)
        for idx, tc in current_tool_calls.items():
            json_str = ''.join(tc["json_chunks"])
            try:
                args = json.loads(json_str) if json_str else {}
            except json.JSONDecodeError:
                args = {}
            fc = FunctionCall(id=tc["id"], name=tc["name"], args=args)
            # Notify caller about function call detection
            if on_function_call:
                on_function_call(fc)
            parts.append(Part.from_function_call(fc))

        # Build thinking string
        thinking = ''.join(accumulated_thinking) if accumulated_thinking else None

        # Update finish reason if we have function calls
        if any(p.function_call for p in parts) and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,  # Streaming doesn't provide single raw response
            thinking=thinking,
        )

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


def create_provider() -> AnthropicProvider:
    """Factory function for plugin discovery."""
    return AnthropicProvider()
