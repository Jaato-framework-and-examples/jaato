"""Zhipu AI provider using the OpenAI-compatible API.

This provider accesses Z.AI's GLM models via the OpenAI-compatible endpoint
(``/api/paas/v4``) instead of the Anthropic-compatible one used by the
``zhipuai`` provider.  The OpenAI endpoint supports prompt caching, which
can reduce latency and cost for agentic workloads with large system prompts
and tool schemas.

Available models are the same GLM family (GLM-5, GLM-4.7, GLM-4.6, GLM-4.5, etc.).

Model discovery:
    The provider queries Z.AI's ``GET /models`` endpoint natively — unlike the
    Anthropic-compat provider which derives the URL, this endpoint is native to
    the OpenAI API surface.

Usage:
    provider = ZhipuAIOpenAIProvider()
    provider.initialize(ProviderConfig(api_key="your-key"))
    provider.connect('glm-5')
    response = provider.complete(messages=[...])

Environment variables:
    ZHIPUAI_API_KEY: Z.AI API key (shared with zhipuai provider)
    ZHIPUAI_OPENAI_BASE_URL: API base URL (default: https://api.z.ai/api/paas/v4)
    ZHIPUAI_OPENAI_MODEL: Default model to use
    ZHIPUAI_OPENAI_CONTEXT_LENGTH: Override context length for models
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Reuse NIM's OpenAI lazy imports — identical SDK.
from ..nim._lazy import get_openai_client_class, get_openai_module

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
# Reuse NIM's OpenAI converters — identical format.
from ..nim.converters import (
    clear_tool_name_mapping,
    get_original_tool_name,
    history_to_openai,
    map_finish_reason,
    response_from_openai,
    tool_schemas_to_openai,
)
from .env import (
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_LENGTH,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_enable_thinking,
    resolve_model,
    resolve_thinking_budget,
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


logger = logging.getLogger(__name__)


def _extract_cache_tokens(usage_obj) -> Optional[int]:
    """Extract cached_tokens from an OpenAI usage object.

    OpenAI-compatible APIs report cached tokens under
    ``usage.prompt_tokens_details.cached_tokens``.  This function
    safely navigates that structure.

    Args:
        usage_obj: The ``usage`` attribute from a ChatCompletion or
            streaming chunk.

    Returns:
        Number of cached prompt tokens, or None if not reported.
    """
    if not usage_obj:
        return None
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if not details:
        return None
    cached = getattr(details, "cached_tokens", None)
    if cached is not None and isinstance(cached, int) and cached > 0:
        return cached
    return None


# Known GLM models with their context window sizes in tokens.
# Used as metadata fallback when the dynamic /models endpoint is unavailable.
MODEL_CONTEXT_LIMITS = {
    # GLM-5 family — 200K context, 128K output
    "glm-5": 204800,
    "glm-5-turbo": 204800,
    # GLM-4.7 family — 200K context
    "glm-4.7": 204800,
    "glm-4.7-flash": 204800,
    "glm-4.7-flashx": 204800,
    # GLM-4.6 family — 200K context
    "glm-4.6": 204800,
    # GLM-4.5 family — 128K context
    "glm-4.5": 131072,
    "glm-4.5-air": 131072,
    "glm-4.5-airx": 131072,
    "glm-4.5-flash": 131072,
    "glm-4.5-x": 131072,
}

KNOWN_MODELS = sorted(MODEL_CONTEXT_LIMITS.keys())

# GLM models that support extended thinking (chain-of-thought reasoning).
THINKING_CAPABLE_MODELS = [
    "glm-5",
    "glm-4.7",
]


class ZhipuAIOpenAIProvider:
    """Zhipu AI provider using the OpenAI-compatible API.

    Accesses Z.AI's GLM models via the OpenAI-compatible endpoint
    (``/api/paas/v4``) using the ``openai`` Python SDK.  This endpoint
    supports prompt caching, which the Anthropic-compatible endpoint
    does not.

    Providers are stateless with respect to conversation history.  The
    session owns the canonical message list and passes it to
    ``complete()`` on each call.

    Lifecycle:
        1. ``__init__()`` — create instance (no connections yet)
        2. ``initialize(config)`` — resolve credentials, create OpenAI client
        3. ``connect(model)`` — set the active model
        4. ``complete(messages, ...)`` — stateless completion
        5. ``shutdown()`` — release resources
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[OpenAI] = None
        self._model_name: Optional[str] = None

        # Configuration
        self._api_key: Optional[str] = None
        self._base_url: str = DEFAULT_BASE_URL
        self._auth_info: str = ""

        # Per-call accounting (NOT conversation state)
        self._last_usage: TokenUsage = TokenUsage()
        self._context_length_override: Optional[int] = None

        # Thinking/reasoning configuration
        self._enable_thinking: bool = False
        self._thinking_budget: int = 10000

        # Agent context for trace identification
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main",
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
            prefix = "zhipuai-openai:main"
        elif self._agent_name:
            prefix = f"zhipuai-openai:subagent:{self._agent_name}"
        else:
            prefix = f"zhipuai-openai:subagent:{self._agent_id}"
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "zhipuai-openai"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with credentials.

        Creates an OpenAI client configured for Z.AI's OpenAI-compatible
        endpoint.

        Args:
            config: Configuration with authentication details.
                - api_key: Z.AI API key (overrides ZHIPUAI_API_KEY)
                - extra['base_url']: Override ZHIPUAI_OPENAI_BASE_URL
                - extra['context_length']: Override context length
                - extra['enable_thinking']: Enable extended thinking
                - extra['thinking_budget']: Max thinking tokens

        Raises:
            APIKeyNotFoundError: If no API key is found.
        """
        self._trace("[INIT] Starting initialization")

        if config is None:
            config = ProviderConfig()

        # Resolve API key
        if config.api_key:
            self._api_key = config.api_key
            self._auth_info = "API key (config)"
        else:
            self._api_key = resolve_api_key()
            if self._api_key:
                self._auth_info = "API key (env/stored)"

        if not self._api_key:
            self._trace("[INIT] No API key found")
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(),
            )

        self._trace(f"[INIT] API key resolved (len={len(self._api_key)})")

        # Resolve base URL
        self._base_url = config.extra.get("base_url") or resolve_base_url()
        self._base_url = self._base_url.rstrip("/")
        self._trace(f"[INIT] base_url={self._base_url}")

        # Optional context length override
        context_extra = config.extra.get("context_length")
        if context_extra:
            self._context_length_override = int(context_extra)
        else:
            self._context_length_override = resolve_context_length()
        if self._context_length_override:
            self._trace(f"[INIT] context_length_override={self._context_length_override}")

        # Extended thinking configuration
        self._enable_thinking = config.extra.get(
            "enable_thinking", resolve_enable_thinking()
        )
        self._thinking_budget = config.extra.get(
            "thinking_budget", resolve_thinking_budget()
        )

        # Create the client
        self._client = self._create_client()
        self._trace("[INIT] Initialization complete")

    def _create_client(self) -> "OpenAI":
        """Create the OpenAI client configured for Z.AI.

        Returns:
            Initialized OpenAI client.
        """
        client_class = get_openai_client_class()
        return client_class(
            base_url=self._base_url,
            api_key=self._api_key,
        )

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
    ) -> bool:
        """Verify that authentication is configured.

        Must work before ``initialize()`` — checks for the API key
        in environment variables and stored credentials.  Does not
        access ``self._client`` (not yet initialized).

        Args:
            allow_interactive: Ignored (Z.AI uses API keys only).
            on_message: Optional callback for status messages.

        Returns:
            True if an API key is available.
        """
        self._trace("[AUTH] Verifying credentials")
        api_key = resolve_api_key()
        if api_key:
            self._trace("[AUTH] API key found")
            if on_message:
                on_message("Found Zhipu AI API key")
            return True

        self._trace("[AUTH] No credentials found")
        if on_message:
            on_message("No Zhipu AI credentials found")
        return False

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()
        self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used.

        Returns:
            Human-readable auth description.
        """
        return self._auth_info or "Z.AI API key"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the model to use.

        Model validation is deferred to the first API call.

        Args:
            model: Model name (e.g., 'glm-5', 'glm-4.7').
            skip_model_test: Accepted for protocol compatibility.
        """
        self._model_name = model
        context_limit = self.get_context_limit()
        self._trace(f"[CONNECT] model={model} context_limit={context_limit}")
        logger.info(f"Connected to Zhipu AI (OpenAI) model: {model}")

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available GLM models.

        Queries Z.AI's ``GET /models`` endpoint (native to the OpenAI API
        surface).  Falls back to the static KNOWN_MODELS list on failure.

        Args:
            prefix: Optional filter prefix.

        Returns:
            Sorted list of model names.
        """
        models = self._fetch_remote_models()
        if not models:
            models = KNOWN_MODELS.copy()

        if prefix:
            models = [m for m in models if m.startswith(prefix)]

        return sorted(models)

    def _fetch_remote_models(self) -> List[str]:
        """Fetch model list from Z.AI using ``fetch_zhipuai_models()``.

        Resolves the API key from the provider instance or environment.

        Returns:
            List of model ID strings, or an empty list on failure.
        """
        api_key = self._api_key
        if not api_key:
            api_key = resolve_api_key()
        if not api_key:
            self._trace("[_fetch_remote_models] No API key available, skipping")
            return []

        from ..zhipuai.provider import fetch_zhipuai_models, ZHIPUAI_MODELS_URL
        self._trace(f"[_fetch_remote_models] GET {ZHIPUAI_MODELS_URL}")
        models = fetch_zhipuai_models(api_key)
        self._trace(f"[_fetch_remote_models] Got {len(models)} models")
        return models

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
        """Stateless completion via Z.AI's OpenAI-compatible API.

        The caller (session) is responsible for maintaining the message
        list and passing it in full each call.

        Args:
            messages: Full conversation history in provider-agnostic format.
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

        # Clear tool name mapping (sanitized <-> original) on each call
        clear_tool_name_mapping()

        # Build OpenAI-format messages
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
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=openai_messages,
                    **kwargs,
                )
                provider_response = response_from_openai(response)

                # Enrich usage with cached token info from prompt_tokens_details
                if response and response.usage:
                    cached = _extract_cache_tokens(response.usage)
                    if cached is not None:
                        provider_response.usage.cache_read_tokens = cached

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
        """Core streaming loop.

        Accumulates text chunks, tool call deltas, reasoning content, and
        usage information from the streaming response.  Handles cancellation
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
                            cache_read_tokens=_extract_cache_tokens(chunk.usage),
                        )
                        self._trace(
                            f"{trace_prefix}_USAGE prompt={usage.prompt_tokens} "
                            f"output={usage.output_tokens} cached={usage.cache_read_tokens}"
                        )
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)
                    continue

                for choice in chunk.choices:
                    delta = choice.delta
                    if not delta:
                        if choice.finish_reason:
                            finish_reason = map_finish_reason(choice.finish_reason)
                        continue

                    # Extract reasoning/thinking content
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
                                self._trace(
                                    f"TOOL_CALL_START idx={idx} id={tc_delta.id!r} "
                                    f"name={getattr(tc_delta.function, 'name', '')!r}"
                                )
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

                # Extract usage from chunk
                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                        cache_read_tokens=_extract_cache_tokens(chunk.usage),
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
                known_models=KNOWN_MODELS,
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

        Uses a heuristic estimate (~4 chars per token) since Z.AI does
        not provide a token counting endpoint via the OpenAI API.

        Args:
            content: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Get context window size for the current model.

        Returns context_length_override if set, otherwise looks up the
        per-model limit from MODEL_CONTEXT_LIMITS.

        Returns:
            Maximum tokens the model can handle.
        """
        if self._context_length_override:
            return self._context_length_override
        return MODEL_CONTEXT_LIMITS.get(self._model_name, DEFAULT_CONTEXT_LENGTH)

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
            True — Z.AI's OpenAI-compatible API supports response_format.
        """
        return True

    def supports_streaming(self) -> bool:
        """Check if streaming is supported.

        Returns:
            True — Z.AI supports streaming via the OpenAI-compatible API.
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

        Returns True for GLM-5 and GLM-4.7 (non-flash) which have native
        chain-of-thought reasoning.

        Returns:
            True if the current model supports extended thinking.
        """
        return self._is_thinking_capable()

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Set thinking configuration.

        Args:
            config: ThinkingConfig with enabled flag and budget.
        """
        self._enable_thinking = config.enabled
        if config.budget_tokens is not None:
            self._thinking_budget = config.budget_tokens

    def _is_thinking_capable(self) -> bool:
        """Check if the current model supports extended thinking.

        GLM-5 and GLM-4.7 (non-flash) have native chain-of-thought
        reasoning capability.
        """
        if not self._model_name:
            return False
        name_lower = self._model_name.lower()
        if name_lower.startswith("glm-5"):
            return True
        return name_lower.startswith("glm-4.7") and "flash" not in name_lower

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

    # ==================== Static Auth Helpers ====================

    @staticmethod
    def login(
        on_message=None,
        on_input=None,
    ) -> bool:
        """Interactive login for Zhipu AI.

        Delegates to the shared zhipuai auth module since the API key
        is the same for both endpoints.

        Args:
            on_message: Optional callback for status messages.
            on_input: Optional callback for user input.

        Returns:
            True if login successful.
        """
        from ..zhipuai.auth import login_interactive
        result = login_interactive(on_message=on_message, on_input=on_input)
        return result is not None

    @staticmethod
    def logout(on_message=None) -> None:
        """Clear stored credentials.

        Args:
            on_message: Optional callback for status messages.
        """
        from ..zhipuai.auth import logout
        logout(on_message=on_message)

    @staticmethod
    def auth_status(on_message=None) -> bool:
        """Check authentication status.

        Args:
            on_message: Optional callback for status messages.

        Returns:
            True if valid credentials are stored.
        """
        from ..zhipuai.auth import status as _status
        return _status(on_message=on_message)


def create_provider() -> ZhipuAIOpenAIProvider:
    """Factory function for plugin discovery."""
    return ZhipuAIOpenAIProvider()
