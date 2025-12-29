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

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Lazy imports - SDK is only loaded when actually used
from ._lazy import (
    get_chat_client_class,
    get_models,
    get_azure_key_credential,
    get_response_format_json,
)

if TYPE_CHECKING:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )
    from azure.core.credentials import AzureKeyCredential

from ..base import ModelProviderPlugin, ProviderConfig, StreamingCallback, UsageUpdateCallback
from ..types import (
    CancelledException,
    CancelToken,
    FinishReason,
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
    ContextLimitError,
    ModelNotFoundError,
    ModelsDisabledError,
    RateLimitError,
    TokenInvalidError,
    TokenNotFoundError,
    TokenPermissionError,
)

# GitHub Models catalog API endpoint (models.github.ai as of May 2025)
CATALOG_API_ENDPOINT = "https://models.github.ai/catalog/models"


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

    # Cache TTL in seconds (5 minutes)
    _MODELS_CACHE_TTL = 300

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

        # Models cache: (timestamp, models_list)
        self._models_cache: Optional[Tuple[float, List[str]]] = None

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
            agent_name: Optional name for the agent (e.g., profile name).
            agent_id: Unique identifier for the agent instance.
        """
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _get_trace_prefix(self) -> str:
        """Get the trace prefix including agent context."""
        if self._agent_type == "main":
            return "github_models:main"
        elif self._agent_name:
            return f"github_models:subagent:{self._agent_name}"
        else:
            return f"github_models:subagent:{self._agent_id}"

    def _trace(self, msg: str) -> None:
        """Write trace message to file for debugging streaming interactions."""
        import tempfile
        trace_path = os.environ.get(
            "JAATO_PROVIDER_TRACE",
            os.path.join(tempfile.gettempdir(), "provider_trace.log")
        )
        if not trace_path:
            return
        import datetime
        try:
            prefix = self._get_trace_prefix()
            with open(trace_path, "a") as f:
                ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                f.write(f"[{ts}] [{prefix}] {msg}\n")
                f.flush()
        except Exception:
            pass  # Don't let tracing errors break the provider

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

    def _create_client(self) -> "ChatCompletionsClient":
        """Create the ChatCompletionsClient.

        Returns:
            Initialized ChatCompletionsClient.
        """
        return get_chat_client_class()(
            endpoint=self._endpoint,
            credential=get_azure_key_credential()(self._token),
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
        """List available models from the GitHub Models catalog API.

        Fetches models from the GitHub Models marketplace API and caches
        the results for _MODELS_CACHE_TTL seconds.

        Args:
            prefix: Optional filter prefix (e.g., 'openai/', 'anthropic/').

        Returns:
            List of model IDs (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet').
        """
        if not self._token:
            # Fallback to static list if not initialized
            models = list(MODEL_CONTEXT_LIMITS.keys())
            if prefix:
                models = [m for m in models if m.startswith(prefix)]
            return sorted(models)

        # Check cache validity
        now = time.time()
        if self._models_cache:
            cache_time, cached_models = self._models_cache
            if now - cache_time < self._MODELS_CACHE_TTL:
                # Cache is valid
                if prefix:
                    return sorted([m for m in cached_models if m.startswith(prefix)])
                return sorted(cached_models)

        # Fetch from GitHub Models catalog API
        models = self._fetch_models_from_api()

        if models:
            # Update cache
            self._models_cache = (now, models)
        else:
            # API failed, fallback to static list
            models = list(MODEL_CONTEXT_LIMITS.keys())

        if prefix:
            models = [m for m in models if m.startswith(prefix)]
        return sorted(models)

    def _fetch_models_from_api(self) -> List[str]:
        """Fetch available models from the GitHub Models catalog API.

        Returns:
            List of model IDs, or empty list on failure.
        """
        try:
            req = urllib.request.Request(
                CATALOG_API_ENDPOINT,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {self._token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            # Extract model IDs from the response
            # Response format: list of model objects with 'id' field (e.g., "openai/gpt-4.1")
            models = []
            if isinstance(data, list):
                for model in data:
                    # 'id' is the primary field for model identifier
                    model_id = model.get('id') or model.get('name') or model.get('model_id')
                    if model_id:
                        models.append(model_id)
            elif isinstance(data, dict):
                # Response might be paginated with a 'models' key
                model_list = data.get('models', data.get('items', []))
                for model in model_list:
                    model_id = model.get('id') or model.get('name') or model.get('model_id')
                    if model_id:
                        models.append(model_id)

            return models
        except urllib.error.HTTPError:
            # API error, return empty to trigger fallback
            return []
        except Exception:
            # Network or parsing error
            return []

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

        messages = [get_models().UserMessage(content=prompt)]

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
        messages.append(get_models().UserMessage(content=message))

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
            text = provider_response.get_text()
            if response_schema and text:
                try:
                    provider_response.structured_output = json.loads(text)
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
            text = provider_response.get_text()
            if response_schema and text:
                try:
                    provider_response.structured_output = json.loads(text)
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
            messages.append(get_models().SystemMessage(content=self._system_instruction))

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
        response_format_json = get_response_format_json()
        if response_schema and response_format_json is not None:
            kwargs['response_format'] = response_format_json()

        return kwargs

    def _add_response_to_history(self, response: ProviderResponse) -> None:
        """Add the model's response to history."""
        if response.parts:
            self._history.append(Message(role=Role.MODEL, parts=response.parts))

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

        # Check for context/token limit errors
        if any(x in error_str for x in ("tokens_limit_reached", "too large", "max size", "context_length")):
            # Try to extract max tokens from error message
            import re
            max_tokens = None
            match = re.search(r'max (?:size|tokens)[:\s]+(\d+)', error_str)
            if match:
                max_tokens = int(match.group(1))
            raise ContextLimitError(
                model=self._model_name or "unknown",
                max_tokens=max_tokens,
                original_error=str(error),
            ) from error

        # Check for model not found (various formats)
        if any(x in error_str for x in ("404", "not found", "unknown_model", "unknown model")):
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

    def supports_streaming(self) -> bool:
        """Check if streaming is supported.

        Returns:
            True - GitHub Models supports streaming via Azure AI Inference SDK.
        """
        return True

    def supports_stop(self) -> bool:
        """Check if mid-turn cancellation (stop) is supported.

        Returns:
            True - GitHub Models supports stop via streaming cancellation.
        """
        return True

    # ==================== Streaming ====================

    def send_message_streaming(
        self,
        message: str,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None
    ) -> ProviderResponse:
        """Send a message with streaming response and optional cancellation.

        Args:
            message: The user's message text.
            on_chunk: Callback invoked for each text chunk as it streams.
            cancel_token: Optional token to request cancellation mid-stream.
            response_schema: Optional JSON Schema to constrain the response.
            on_usage_update: Optional callback for real-time token usage updates.

        Returns:
            ProviderResponse with accumulated text and/or function calls.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Build messages list
        messages = self._build_messages()
        messages.append(get_models().UserMessage(content=message))

        # Add user message to history
        self._history.append(Message.from_text(Role.USER, message))

        # Build completion kwargs
        kwargs = self._build_completion_kwargs(response_schema)
        kwargs['stream'] = True

        # Accumulate response with parts preserving order
        accumulated_text = []  # Text chunks for current text block
        parts = []  # Ordered parts preserving text/function_call interleaving
        finish_reason = FinishReason.UNKNOWN
        function_calls = []  # Also keep flat list for backwards compatibility
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
            self._trace(f"STREAM_START message_len={len(message)}")
            self._trace(f"STREAM_INPUT>>>\n{message}\n<<<STREAM_INPUT")
            chunk_count = 0
            response_stream = self._client.complete(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )

            for chunk in response_stream:
                # Check for cancellation
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"STREAM_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                # Extract content from choices
                if hasattr(chunk, 'choices') and chunk.choices:
                    for choice in chunk.choices:
                        # Extract text delta
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta
                            if hasattr(delta, 'content') and delta.content:
                                chunk_count += 1
                                self._trace(f"STREAM_CHUNK[{chunk_count}] len={len(delta.content)} text={repr(delta.content)}")
                                accumulated_text.append(delta.content)
                                on_chunk(delta.content)

                            # Extract tool calls
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                from .converters import extract_function_calls_from_stream_delta
                                new_calls = extract_function_calls_from_stream_delta(delta.tool_calls)
                                for fc in new_calls:
                                    self._trace(f"STREAM_FUNC_CALL name={fc.name}")
                                    # Flush any pending text before adding function call
                                    flush_text_block()
                                    # Add function call as a part
                                    parts.append(Part.from_function_call(fc))
                                function_calls.extend(new_calls)

                        # Extract finish reason
                        if hasattr(choice, 'finish_reason') and choice.finish_reason:
                            finish_reason = self._map_finish_reason(choice.finish_reason)

                # Extract usage from final chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=getattr(chunk.usage, 'prompt_tokens', 0) or 0,
                        output_tokens=getattr(chunk.usage, 'completion_tokens', 0) or 0,
                        total_tokens=getattr(chunk.usage, 'total_tokens', 0) or 0
                    )
                    self._trace(f"STREAM_USAGE prompt={usage.prompt_tokens} output={usage.output_tokens} total={usage.total_tokens}")
                    # Notify about usage update for real-time accounting
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            # Get accumulated output before flushing
            all_text = ''.join(accumulated_text)
            self._trace(f"STREAM_END chunks={chunk_count} finish_reason={finish_reason} output_len={len(all_text)}")
            self._trace(f"STREAM_OUTPUT>>>\n{all_text}\n<<<STREAM_OUTPUT")

        except Exception as e:
            self._trace(f"STREAM_ERROR {type(e).__name__}: {e}")
            # If cancelled during iteration, treat as cancellation
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                self._handle_api_error(e)
                raise

        # Flush any remaining text as final part
        flush_text_block()

        # If we have function calls, update finish reason
        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        provider_response = ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None
        )

        self._last_usage = usage

        # Add assistant response to history
        self._add_response_to_history(provider_response)

        # Parse structured output if schema was requested
        final_text = provider_response.get_text()
        if response_schema and final_text and not was_cancelled:
            try:
                provider_response.structured_output = json.loads(final_text)
            except json.JSONDecodeError:
                pass

        return provider_response

    def send_tool_results_streaming(
        self,
        results: List[ToolResult],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None
    ) -> ProviderResponse:
        """Send tool results with streaming response and optional cancellation.

        Args:
            results: List of tool execution results.
            on_chunk: Callback invoked for each text chunk as it streams.
            cancel_token: Optional token to request cancellation mid-stream.
            response_schema: Optional JSON Schema to constrain the response.
            on_usage_update: Optional callback for real-time token usage updates.

        Returns:
            ProviderResponse with accumulated text and/or function calls.
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
        kwargs['stream'] = True

        # Accumulate response with parts preserving order
        accumulated_text = []  # Text chunks for current text block
        parts = []  # Ordered parts preserving text/function_call interleaving
        finish_reason = FinishReason.UNKNOWN
        function_calls = []  # Also keep flat list for backwards compatibility
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
            tool_names = [r.name for r in results]
            self._trace(f"STREAM_TOOL_RESULTS_START tools={tool_names}")
            # Log tool results as input
            tool_results_summary = []
            for r in results:
                result_str = str(r.result) if r.result is not None else "None"
                tool_results_summary.append(f"{r.name}: {result_str}")
            self._trace(f"STREAM_TOOL_INPUT>>>\n" + "\n".join(tool_results_summary) + "\n<<<STREAM_TOOL_INPUT")
            chunk_count = 0
            response_stream = self._client.complete(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )

            for chunk in response_stream:
                # Check for cancellation
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"STREAM_TOOL_RESULTS_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                # Extract content from choices
                if hasattr(chunk, 'choices') and chunk.choices:
                    for choice in chunk.choices:
                        # Extract text delta
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta
                            if hasattr(delta, 'content') and delta.content:
                                chunk_count += 1
                                self._trace(f"STREAM_TOOL_CHUNK[{chunk_count}] len={len(delta.content)} text={repr(delta.content)}")
                                accumulated_text.append(delta.content)
                                on_chunk(delta.content)

                            # Extract tool calls
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                from .converters import extract_function_calls_from_stream_delta
                                new_calls = extract_function_calls_from_stream_delta(delta.tool_calls)
                                for fc in new_calls:
                                    self._trace(f"STREAM_TOOL_FUNC_CALL name={fc.name}")
                                    # Flush any pending text before adding function call
                                    flush_text_block()
                                    # Add function call as a part
                                    parts.append(Part.from_function_call(fc))
                                function_calls.extend(new_calls)

                        # Extract finish reason
                        if hasattr(choice, 'finish_reason') and choice.finish_reason:
                            finish_reason = self._map_finish_reason(choice.finish_reason)

                # Extract usage from final chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=getattr(chunk.usage, 'prompt_tokens', 0) or 0,
                        output_tokens=getattr(chunk.usage, 'completion_tokens', 0) or 0,
                        total_tokens=getattr(chunk.usage, 'total_tokens', 0) or 0
                    )
                    self._trace(f"STREAM_TOOL_USAGE prompt={usage.prompt_tokens} output={usage.output_tokens} total={usage.total_tokens}")
                    # Notify about usage update for real-time accounting
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            # Get accumulated output before flushing
            all_text = ''.join(accumulated_text)
            self._trace(f"STREAM_TOOL_RESULTS_END chunks={chunk_count} finish_reason={finish_reason} output_len={len(all_text)}")
            self._trace(f"STREAM_TOOL_OUTPUT>>>\n{all_text}\n<<<STREAM_TOOL_OUTPUT")

        except Exception as e:
            self._trace(f"STREAM_TOOL_RESULTS_ERROR {type(e).__name__}: {e}")
            # If cancelled during iteration, treat as cancellation
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                self._handle_api_error(e)
                raise

        # Flush any remaining text as final part
        flush_text_block()

        # If we have function calls, update finish reason
        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        provider_response = ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None
        )

        self._last_usage = usage

        # Add assistant response to history
        self._add_response_to_history(provider_response)

        # Parse structured output if schema was requested
        final_text = provider_response.get_text()
        if response_schema and final_text and not was_cancelled:
            try:
                provider_response.structured_output = json.loads(final_text)
            except json.JSONDecodeError:
                pass

        return provider_response

    def _map_finish_reason(self, reason: str) -> FinishReason:
        """Map SDK finish reason string to internal FinishReason."""
        if not reason:
            return FinishReason.UNKNOWN

        reason_lower = reason.lower()
        if reason_lower == 'stop':
            return FinishReason.STOP
        elif reason_lower in ('length', 'max_tokens'):
            return FinishReason.MAX_TOKENS
        elif reason_lower in ('tool_calls', 'function_call'):
            return FinishReason.TOOL_USE
        elif reason_lower == 'content_filter':
            return FinishReason.SAFETY

        return FinishReason.UNKNOWN

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
