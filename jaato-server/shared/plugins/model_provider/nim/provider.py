"""NVIDIA NIM model provider implementation.

This provider enables access to AI models through NVIDIA NIM (Inference
Microservices), supporting both NVIDIA's hosted API (build.nvidia.com)
and self-hosted NIM containers.

NIM exposes an OpenAI-compatible chat completions API, so this provider
uses the ``openai`` Python SDK as its transport layer.

Supported models include Llama, Mistral, Nemotron, DeepSeek-R1, and
other models available in the NIM catalog.

Authentication:
- Hosted API: JAATO_NIM_API_KEY (nvapi-... key from build.nvidia.com)
- Self-hosted: No API key required (NIM containers run without auth)

Environment variables:
    JAATO_NIM_API_KEY: API key for hosted NIM
    JAATO_NIM_BASE_URL: Endpoint (default: https://integrate.api.nvidia.com/v1)
    JAATO_NIM_MODEL: Default model name
    JAATO_NIM_CONTEXT_LENGTH: Override context window size
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ._lazy import get_openai_client_class, get_openai_module

if TYPE_CHECKING:
    from openai import OpenAI

from ..base import (
    FunctionCallDetectedCallback,
    ModelProviderPlugin,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
)
from jaato_sdk.plugins.model_provider.types import (
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
    ThinkingConfig,
    TurnResult,
)
from .converters import (
    clear_tool_name_mapping,
    get_original_tool_name,
    history_to_openai,
    map_finish_reason,
    response_from_openai,
    tool_schemas_to_openai,
)
from .env import (
    DEFAULT_BASE_URL,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_model,
    is_self_hosted,
    get_checked_credential_locations,
)
from .errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)

# Models known to expose reasoning/thinking content via `reasoning_content`.
REASONING_CAPABLE_MODELS = [
    "deepseek/deepseek-r1",
    "deepseek-r1",
]


class NIMProvider:
    """NVIDIA NIM model provider.

    Provides access to NIM's model catalog via the OpenAI-compatible
    chat completions API.  Supports both NVIDIA's hosted API and
    self-hosted NIM containers.

    Providers are stateless with respect to conversation history. The
    session owns the canonical message list and passes it to
    ``complete()`` on each call.  Providers hold only connection/auth
    state set by ``initialize()`` and ``connect()``.

    Lifecycle:
        1. ``__init__()`` — create instance (no connections yet)
        2. ``initialize(config)`` — resolve credentials, create OpenAI client
        3. ``connect(model)`` — set the active model
        4. ``complete(messages, ...)`` — stateless completion
        5. ``shutdown()`` — release resources

    Usage:
        provider = NIMProvider()
        provider.initialize(ProviderConfig(api_key='nvapi-...'))
        provider.connect('meta/llama-3.1-70b-instruct')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[OpenAI] = None
        self._model_name: Optional[str] = None

        # Configuration
        self._api_key: Optional[str] = None
        self._base_url: str = DEFAULT_BASE_URL

        # Per-call accounting (NOT conversation state)
        self._last_usage: TokenUsage = TokenUsage()
        self._context_length: int = 0

        # Thinking/reasoning configuration
        self._enable_thinking: bool = True

        # Agent context for trace identification
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main"
    ) -> None:
        """Set agent context for trace identification.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent.
            agent_id: Unique identifier for the agent instance.
        """
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        from shared.trace import provider_trace
        if self._agent_type == "main":
            prefix = "nim:main"
        elif self._agent_name:
            prefix = f"nim:subagent:{self._agent_name}"
        else:
            prefix = f"nim:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "nim"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with credentials.

        Creates an OpenAI client configured for the NIM endpoint.
        For hosted NIM (integrate.api.nvidia.com), an API key is required.
        For self-hosted NIM containers, the key is optional.

        Args:
            config: Configuration with authentication details.
                - api_key: NIM API key (nvapi-...)
                - extra['base_url']: Override API endpoint
                - extra['context_length']: Override context window size

        Raises:
            APIKeyNotFoundError: No API key found and endpoint is not self-hosted.
        """
        if config is None:
            config = ProviderConfig()

        # Resolve configuration
        self._api_key = config.api_key or resolve_api_key()
        self._base_url = config.extra.get("base_url") or resolve_base_url()

        context_length_extra = config.extra.get("context_length")
        if context_length_extra:
            self._context_length = int(context_length_extra)
        else:
            self._context_length = resolve_context_length()

        # Validate API key (required for hosted, optional for self-hosted)
        if not self._api_key and not is_self_hosted(self._base_url):
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(),
            )

        # Create client
        self._client = self._create_client()
        self._trace(f"[INIT] client created, base_url={self._base_url}")

    def _create_client(self) -> "OpenAI":
        """Create the OpenAI client configured for NIM.

        Returns:
            Initialized OpenAI client.
        """
        client_class = get_openai_client_class()
        # For self-hosted without key, use a dummy key (OpenAI SDK requires one)
        api_key = self._api_key or "not-needed"
        return client_class(
            base_url=self._base_url,
            api_key=api_key,
        )

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None
    ) -> bool:
        """Verify that authentication is configured.

        Must work before ``initialize()`` — checks for the API key
        in environment variables and stored credentials. Does not
        access ``self._client`` (not yet initialized).

        Args:
            allow_interactive: Ignored (NIM uses API keys only).
            on_message: Optional callback for status messages.

        Returns:
            True if authentication is configured or endpoint is self-hosted.

        Raises:
            APIKeyNotFoundError: If no key found and not self-hosted.
        """
        import os
        from .env import ENV_NIM_API_KEY

        base_url = resolve_base_url()

        # Check env var first (highest priority)
        env_key = os.environ.get(ENV_NIM_API_KEY)
        if env_key:
            if on_message:
                on_message("Found NIM API key (environment variable)")
            return True

        # Check stored credentials
        api_key = resolve_api_key()  # also checks stored credentials
        if api_key:
            if on_message:
                on_message("Found NIM API key (stored credentials)")
            return True

        if is_self_hosted(base_url):
            if on_message:
                on_message(f"Self-hosted NIM endpoint ({base_url}), no API key required")
            return True

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        return False

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used.

        Differentiates between env var, stored credentials, and self-hosted.

        Returns:
            Human-readable auth description.
        """
        import os
        from .env import ENV_NIM_API_KEY

        if is_self_hosted(self._base_url):
            return f"Self-hosted NIM ({self._base_url})"

        if os.environ.get(ENV_NIM_API_KEY):
            return f"NIM API key ({ENV_NIM_API_KEY})"

        try:
            from .auth import get_credential_file_path
            cred_path = get_credential_file_path()
            if cred_path:
                return f"NIM API key ({cred_path})"
        except ImportError:
            pass

        return "NIM API key"

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use.

        Model validation is deferred to the first API call.

        Args:
            model: Model ID (e.g., 'meta/llama-3.1-70b-instruct').
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

        NIM's hosted API does not provide a standard list-models endpoint
        via the OpenAI SDK, so this returns an empty list. Users should
        consult build.nvidia.com for available models.

        Args:
            prefix: Optional prefix filter (unused).

        Returns:
            Empty list (NIM catalog must be consulted directly).
        """
        return []

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
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> TurnResult:
        """Stateless completion: convert messages to OpenAI format, call API, return response.

        The caller (session) is responsible for maintaining the message
        list and passing it in full each call.  This method does not hold
        any conversation state.

        Returns ``TurnResult.from_provider_response(r)`` on success and
        **raises** transient errors for ``with_retry``.

        Args:
            messages: Full conversation history in provider-agnostic Message
                format.  Must already include the latest user message or tool
                results — the provider does not append anything.
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

        Raises:
            RuntimeError: If provider is not initialized/connected.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not connected. Call initialize() and connect() first.")

        # Clear tool name mapping (sanitized ↔ original) on each call
        clear_tool_name_mapping()

        # Build OpenAI-format messages from explicit parameters
        openai_messages: List[Dict[str, Any]] = []
        if system_instruction:
            openai_messages.append({"role": "system", "content": system_instruction})
        openai_messages.extend(history_to_openai(list(messages)))

        # Build kwargs
        kwargs: Dict[str, Any] = {}
        if tools:
            openai_tools = tool_schemas_to_openai(tools)
            if openai_tools:
                kwargs["tools"] = openai_tools
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            if on_chunk:
                # Streaming mode
                provider_response = self._stream_response(
                    messages=openai_messages,
                    kwargs=kwargs,
                    on_chunk=on_chunk,
                    cancel_token=cancel_token,
                    on_usage_update=on_usage_update,
                    on_thinking=on_thinking,
                    trace_prefix="COMPLETE_STREAM",
                )
            else:
                # Batch mode
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=openai_messages,
                    **kwargs,
                )
                provider_response = response_from_openai(response)

            # Per-call accounting (NOT conversation state)
            self._last_usage = provider_response.usage

            # Parse structured output if schema was requested
            text = provider_response.get_text()
            if response_schema and text:
                try:
                    provider_response.structured_output = json.loads(text)
                except json.JSONDecodeError:
                    pass

            return TurnResult.from_provider_response(provider_response)
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
        on_thinking: Optional[ThinkingCallback] = None,
        trace_prefix: str = "STREAM",
    ) -> ProviderResponse:
        """Core streaming loop shared by send_message and send_tool_results.

        Accumulates text chunks, tool call deltas, reasoning content, and
        usage information from the streaming response. Handles cancellation
        via cancel_token.

        Args:
            messages: OpenAI-format message list.
            kwargs: Additional kwargs for chat.completions.create().
            on_chunk: Callback for each text chunk.
            cancel_token: Optional cancellation token.
            on_usage_update: Optional usage callback.
            on_thinking: Optional callback for reasoning/thinking chunks.
            trace_prefix: Prefix for trace logging.

        Returns:
            ProviderResponse with accumulated response.
        """
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        accumulated_text: List[str] = []
        accumulated_thinking: List[str] = []
        parts: List[Part] = []
        finish_reason = FinishReason.UNKNOWN
        function_calls: List[FunctionCall] = []
        usage = TokenUsage()
        was_cancelled = False

        # Track tool call accumulation (streaming sends tool calls in pieces)
        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        def flush_text_block():
            """Flush accumulated text as a single Part."""
            nonlocal accumulated_text
            if accumulated_text:
                text = "".join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        def flush_tool_calls():
            """Flush accumulated tool calls as Parts."""
            nonlocal tool_call_accumulators
            for idx in sorted(tool_call_accumulators.keys()):
                tc = tool_call_accumulators[idx]
                func_name = tc.get("function", {}).get("name")
                if func_name:
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    tool_id = tc.get("id")
                    original_name = get_original_tool_name(func_name)
                    if not tool_id:
                        self._trace(f"ERROR: Missing tool call ID for {func_name}")
                    fc = FunctionCall(
                        id=tool_id,
                        name=original_name,
                        args=args,
                    )
                    parts.append(Part.from_function_call(fc))
                    function_calls.append(fc)
            tool_call_accumulators.clear()

        try:
            self._trace(f"{trace_prefix}_START")
            chunk_count = 0
            response_stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )

            for chunk in response_stream:
                # Check for cancellation
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"{trace_prefix}_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                if not chunk.choices:
                    # Final chunk may have only usage
                    if chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                        )
                        self._trace(f"{trace_prefix}_USAGE prompt={usage.prompt_tokens} output={usage.output_tokens}")
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)
                    continue

                for choice in chunk.choices:
                    delta = choice.delta
                    if not delta:
                        if choice.finish_reason:
                            finish_reason = map_finish_reason(choice.finish_reason)
                        continue

                    # Extract reasoning/thinking (e.g. DeepSeek-R1 on NIM)
                    if self._enable_thinking:
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning and isinstance(reasoning, str):
                            self._trace(f"{trace_prefix}_THINKING len={len(reasoning)}")
                            accumulated_thinking.append(reasoning)
                            if on_thinking:
                                on_thinking(reasoning)

                    # Accumulate text
                    if delta.content:
                        chunk_count += 1
                        accumulated_text.append(delta.content)
                        on_chunk(delta.content)

                    # Accumulate tool calls (they come in pieces)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_accumulators:
                                self._trace(f"TOOL_CALL_START idx={idx} id={tc_delta.id!r} name={getattr(tc_delta.function, 'name', '')!r}")
                                tool_call_accumulators[idx] = {
                                    "id": tc_delta.id,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            acc = tool_call_accumulators[idx]
                            if tc_delta.id:
                                acc["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    acc["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    acc["function"]["arguments"] += tc_delta.function.arguments

                    # Extract finish reason
                    if choice.finish_reason:
                        finish_reason = map_finish_reason(choice.finish_reason)

                # Extract usage from chunk (some providers include it on each chunk)
                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                    )
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            self._trace(f"{trace_prefix}_END chunks={chunk_count} finish_reason={finish_reason}")

        except Exception as e:
            self._trace(f"{trace_prefix}_ERROR {type(e).__name__}: {e}")
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                raise

        # Flush remaining text and tool calls
        flush_text_block()
        flush_tool_calls()

        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        thinking = "".join(accumulated_thinking) if accumulated_thinking else None

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=thinking,
        )

    # ==================== Error Handling ====================

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors and convert to provider-specific exceptions.

        Maps OpenAI SDK exception types to jaato error types for
        consistent error handling across the framework.

        Args:
            error: The original exception from the OpenAI SDK.
        """
        openai = get_openai_module()

        if isinstance(error, openai.AuthenticationError):
            raise AuthenticationError(
                original_error=str(error),
            ) from error

        if isinstance(error, openai.RateLimitError):
            retry_after = None
            # Try to extract retry-after from response headers
            response = getattr(error, "response", None)
            if response:
                retry_header = getattr(response.headers, "get", lambda *a: None)("retry-after")
                if retry_header:
                    try:
                        retry_after = float(retry_header)
                    except ValueError:
                        pass
            raise RateLimitError(
                retry_after=retry_after,
                original_error=str(error),
            ) from error

        if isinstance(error, openai.NotFoundError):
            raise ModelNotFoundError(
                model=self._model_name or "unknown",
                original_error=str(error),
            ) from error

        if isinstance(error, openai.APIConnectionError):
            raise InfrastructureError(
                status_code=0,
                original_error=str(error),
            ) from error

        if isinstance(error, openai.InternalServerError):
            status_code = getattr(error, "status_code", 500)
            raise InfrastructureError(
                status_code=status_code,
                original_error=str(error),
            ) from error

        if isinstance(error, openai.APIStatusError):
            status_code = getattr(error, "status_code", 0)
            error_str = str(error).lower()

            # Context limit errors
            if any(x in error_str for x in ("context_length", "too large", "max size", "tokens_limit")):
                max_tokens = None
                match = re.search(r'max (?:size|tokens)[:\s]+(\d+)', error_str)
                if match:
                    max_tokens = int(match.group(1))
                raise ContextLimitError(
                    model=self._model_name or "unknown",
                    max_tokens=max_tokens,
                    original_error=str(error),
                ) from error

            # 5xx infrastructure errors
            if 500 <= status_code < 600:
                raise InfrastructureError(
                    status_code=status_code,
                    original_error=str(error),
                ) from error

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Count tokens for the given content.

        Uses a heuristic estimate (~4 chars per token) since NIM does
        not provide a token counting endpoint.

        Args:
            content: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Get the context window size for the current model.

        Returns the value from JAATO_NIM_CONTEXT_LENGTH or config,
        since NIM hosts many models with varying limits and does not
        expose this information through the API.

        Returns:
            Maximum tokens the model can handle.
        """
        return self._context_length

    def get_token_usage(self) -> TokenUsage:
        """Get token usage from the last response.

        Returns:
            TokenUsage with prompt/output/total counts.
        """
        return self._last_usage

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """Check if structured output (json_object response format) is supported.

        Returns:
            True — NIM's OpenAI-compatible API supports response_format.
        """
        return True

    def supports_streaming(self) -> bool:
        """Check if streaming is supported.

        Returns:
            True — NIM supports streaming via the OpenAI-compatible API.
        """
        return True

    def supports_stop(self) -> bool:
        """Check if mid-turn cancellation (stop) is supported.

        Returns:
            True — streaming responses can be cancelled via cancel_token.
        """
        return True

    def supports_thinking(self) -> bool:
        """Check if reasoning/thinking content is supported.

        Returns True for models known to expose ``reasoning_content``
        (e.g. DeepSeek-R1). Other models return False.

        Returns:
            True if the current model exposes reasoning content.
        """
        return self._is_reasoning_capable()

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Set thinking configuration.

        For reasoning-capable models this enables/disables extraction of
        ``reasoning_content`` from responses.

        Args:
            config: ThinkingConfig with enabled flag and budget.
        """
        self._enable_thinking = config.enabled

    def _is_reasoning_capable(self) -> bool:
        """Check if the current model exposes reasoning content."""
        if not self._model_name:
            return False
        name_lower = self._model_name.lower()
        for prefix in REASONING_CAPABLE_MODELS:
            if name_lower.startswith(prefix) or name_lower.endswith(prefix):
                return True
        return False

    # ==================== Error Classification for Retry ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry purposes.

        Args:
            exc: The exception to classify.

        Returns:
            Classification dict or None to use global fallback.
        """
        if isinstance(exc, RateLimitError):
            return {"transient": True, "rate_limit": True, "infra": False}

        if isinstance(exc, InfrastructureError):
            return {"transient": True, "rate_limit": False, "infra": True}

        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract retry-after hint from an exception.

        Args:
            exc: The exception to extract retry-after from.

        Returns:
            Suggested delay in seconds, or None if not available.
        """
        if isinstance(exc, RateLimitError) and exc.retry_after:
            return float(exc.retry_after)

        return None


def create_provider() -> NIMProvider:
    """Factory function for plugin discovery.

    Returns:
        A new NIMProvider instance.
    """
    return NIMProvider()
