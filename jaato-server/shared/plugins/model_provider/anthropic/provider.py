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
import os
import re
from typing import Any, Dict, List, Optional

from ..base import (
    FunctionCallDetectedCallback,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
)
from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    ThinkingConfig,
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
    get_original_tool_name,
    messages_to_anthropic,
    response_from_anthropic,
    serialize_history,
    tool_schemas_to_anthropic,
    validate_tool_use_pairing,
)
from .env import (
    get_checked_credential_locations,
    resolve_api_key,
    resolve_enable_thinking,
    resolve_oauth_token,
    resolve_thinking_budget,
)
from .errors import (
    APIKeyInvalidError,
    APIKeyNotFoundError,
    ContextLimitError,
    ModelNotFoundError,
    OverloadedError,
    RateLimitError,
    UsageLimitError,
)
from .oauth import (
    get_valid_access_token,
    load_tokens,
    login as oauth_login,
    refresh_tokens,
    save_tokens,
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

# Claude Code identity - required for OAuth tokens (server-side validation)
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


class AnthropicProvider:
    """Stateless Anthropic Claude provider.

    This provider uses a stateless design: the caller (session) owns the
    conversation history and passes it to ``complete()`` on every call.
    The provider does not maintain internal message state.

    Features:
    - Multiple Claude model families
    - Function calling with manual control
    - Extended thinking (reasoning traces)
    - Prompt caching via ``cache_anthropic`` plugin (up to 90% cost reduction)
    - Real token counting via API
    - Streaming with cancellation support

    Usage:
        provider = AnthropicProvider()
        provider.initialize(ProviderConfig(
            api_key='sk-ant-...',  # Or set ANTHROPIC_API_KEY env var
            extra={
                'enable_thinking': True,   # Optional: extended thinking
                'thinking_budget': 10000,  # Optional: max thinking tokens
            }
        ))
        provider.connect('claude-sonnet-4-20250514')
        response = provider.complete(
            messages=[Message.from_text(Role.USER, "Hello!")],
            system_instruction="You are helpful.",
        )

    Environment variables:
        ANTHROPIC_API_KEY: API key for authentication
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[Any] = None  # anthropic.Anthropic
        self._model_name: Optional[str] = None

        # Configuration
        self._api_key: Optional[str] = None
        self._enable_thinking: bool = False
        self._thinking_budget: int = 10000

        # Per-call accounting (updated after each complete() call)
        self._last_usage: TokenUsage = TokenUsage()

        # Cache plugin (optional, for delegated cache control)
        self._cache_plugin: Optional[Any] = None  # CachePlugin protocol

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
                - extra['enable_thinking']: Enable extended thinking (default: False)
                - extra['thinking_budget']: Max thinking tokens (default: 10000)

            Note: Cache configuration (enable_caching, cache_ttl, etc.) is now
            handled by the ``cache_anthropic`` plugin via ``CachePlugin.initialize()``.

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

        # Set workspace path from config.extra if provided
        # This ensures token resolution can find workspace-specific OAuth tokens
        # even when JAATO_WORKSPACE_ROOT env var isn't set (e.g., subagent spawning)
        workspace_path = config.extra.get('workspace_path')
        if workspace_path and not os.environ.get('JAATO_WORKSPACE_ROOT'):
            os.environ['JAATO_WORKSPACE_ROOT'] = workspace_path

        # Resolve credentials in priority order:
        # 1. PKCE OAuth tokens (from interactive login, stored in config dir)
        # 2. OAuth token from env var (sk-ant-oat01-... from claude setup-token)
        # 3. API key (sk-ant-api03-... from console.anthropic.com)
        self._api_key = config.api_key or resolve_api_key()
        self._oauth_token = config.extra.get("oauth_token") or resolve_oauth_token()
        self._pkce_access_token: Optional[str] = None
        self._use_pkce = False
        self._auth_info: str = ""

        # Try PKCE OAuth first (interactive login tokens)
        try:
            self._pkce_access_token = get_valid_access_token()
            if self._pkce_access_token:
                self._use_pkce = True
                self._auth_info = "PKCE OAuth"
        except Exception:
            # PKCE token refresh failed, will try other methods
            self._pkce_access_token = None

        # Track which credential source was resolved
        if not self._auth_info:
            if self._oauth_token:
                if config.extra.get("oauth_token"):
                    self._auth_info = "OAuth token (config)"
                else:
                    self._auth_info = "OAuth token (ANTHROPIC_AUTH_TOKEN)"
            elif self._api_key:
                if config.api_key:
                    self._auth_info = "API key (config)"
                else:
                    self._auth_info = "API key (ANTHROPIC_API_KEY)"

        if not self._pkce_access_token and not self._oauth_token and not self._api_key:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        # Parse extra config (config.extra takes precedence over env vars)
        self._enable_thinking = config.extra.get(
            "enable_thinking", resolve_enable_thinking()
        )
        self._thinking_budget = config.extra.get(
            "thinking_budget", resolve_thinking_budget()
        )

        # Create the client based on auth method
        # Priority: PKCE OAuth > env var OAuth > API key
        self._client = self._create_client()

        # Verify connectivity with a lightweight call
        self._verify_connectivity()

    def _create_http_client(self) -> Optional[Any]:
        """Create a custom httpx client if proxy or SSL configuration is needed.

        Returns an httpx.Client configured with corporate CA certificates,
        Kerberos/SPNEGO proxy auth, and standard proxy env vars — all handled
        centrally by ``get_httpx_client()``.

        Returns None if no custom configuration is needed, letting the
        Anthropic SDK create its own default client.
        """
        from shared.ssl_helper import active_cert_bundle
        from shared.http.proxy import (
            get_httpx_client,
            get_proxy_url,
            is_kerberos_proxy_enabled,
        )

        ca_bundle = active_cert_bundle()
        kerberos_enabled = is_kerberos_proxy_enabled()
        proxy_url = get_proxy_url()

        if not ca_bundle and not kerberos_enabled and not proxy_url:
            return None  # Let SDK create its own client with default settings

        return get_httpx_client()

    def _create_client(self):
        """Create Anthropic client with appropriate auth method.

        Configures proxy and SSL settings when corporate CA certificates
        (REQUESTS_CA_BUNDLE / SSL_CERT_FILE) or Kerberos proxy authentication
        (JAATO_KERBEROS_PROXY) are detected. Otherwise lets the Anthropic SDK
        create its own default httpx client.
        """
        import anthropic

        # Build custom httpx client for proxy/SSL if needed
        http_client = self._create_http_client()
        client_kwargs: Dict[str, Any] = {}
        if http_client:
            client_kwargs["http_client"] = http_client

        # Headers for OAuth authentication
        # Must match Claude Code CLI headers for OAuth tokens to work
        oauth_headers = {
            "anthropic-beta": (
                "oauth-2025-04-20,"
                "interleaved-thinking-2025-05-14,"
                "claude-code-20250219"
            ),
            "user-agent": "claude-cli/2.1.2 (external, cli)",
        }

        # Priority: PKCE OAuth > env var OAuth > API key
        if self._use_pkce and self._pkce_access_token:
            # PKCE OAuth - uses access token from interactive login
            return anthropic.Anthropic(
                auth_token=self._pkce_access_token,
                default_headers=oauth_headers,
                **client_kwargs,
            )
        elif self._oauth_token:
            # Env var OAuth - uses token from claude setup-token
            return anthropic.Anthropic(
                auth_token=self._oauth_token,
                default_headers=oauth_headers,
                **client_kwargs,
            )
        else:
            # API key - standard authentication
            return anthropic.Anthropic(api_key=self._api_key, **client_kwargs)

    def _refresh_pkce_token_if_needed(self) -> None:
        """Refresh PKCE access token if expired."""
        if not self._use_pkce:
            return

        tokens = load_tokens()
        if tokens and tokens.is_expired:
            try:
                new_tokens = refresh_tokens(tokens.refresh_token)
                save_tokens(new_tokens)
                self._pkce_access_token = new_tokens.access_token
                # Recreate client with new token
                self._client = self._create_client()
            except Exception as e:
                # Token refresh failed - fall back to other auth methods
                self._use_pkce = False
                self._pkce_access_token = None
                if self._oauth_token or self._api_key:
                    self._client = self._create_client()
                else:
                    raise RuntimeError(f"OAuth token refresh failed: {e}")

    def _verify_connectivity(self) -> None:
        """Verify connectivity by checking API key validity.

        Makes a minimal API call to verify the key works.
        """
        # Skip verification for now - will fail on first real call if invalid
        # A lightweight verification would be nice but Anthropic doesn't have
        # a dedicated endpoint for this
        pass

    @staticmethod
    def login(on_message=None) -> None:
        """Run interactive OAuth login flow.

        Opens browser to authenticate with Claude Pro/Max subscription.
        Stores tokens for future use.

        Args:
            on_message: Optional callback for status messages.
        """
        oauth_login(on_message=on_message)
        msg = "Successfully authenticated with Claude Pro/Max subscription."
        if on_message:
            on_message(msg)
        else:
            print(msg)

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None
    ) -> bool:
        """Verify that authentication is configured and optionally trigger interactive login.

        This can be called BEFORE initialize() to ensure credentials are available.
        For Anthropic, this checks for PKCE OAuth tokens, OAuth env tokens, or API keys.

        Args:
            allow_interactive: If True and auth is not configured, attempt
                interactive OAuth login (opens browser).
            on_message: Optional callback for status messages during login.

        Returns:
            True if authentication is configured and valid.
            False if authentication failed or was cancelled.

        Raises:
            APIKeyNotFoundError: If allow_interactive=False and no credentials found.
        """
        from typing import Callable

        # Check existing credentials in priority order
        # 1. PKCE OAuth tokens (from interactive login)
        try:
            pkce_token = get_valid_access_token()
            if pkce_token:
                if on_message:
                    on_message("Found valid PKCE OAuth token")
                return True
        except Exception:
            # Token refresh failed, will try other methods
            pass

        # 2. OAuth token from env var
        oauth_token = resolve_oauth_token()
        if oauth_token:
            if on_message:
                on_message("Found OAuth token from environment")
            return True

        # 3. API key
        api_key = resolve_api_key()
        if api_key:
            if on_message:
                on_message("Found API key")
            return True

        # No credentials found
        if on_message:
            on_message("No credentials found.")

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        # Return False to signal interactive login is needed
        # The caller (e.g., server) should use the anthropic_auth plugin for login
        return False

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            # Anthropic client doesn't need explicit cleanup
            self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        return self._auth_info

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use and verify it responds.

        Args:
            model: Model ID (e.g., 'claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022').

        Raises:
            ModelNotFoundError: Model doesn't exist or is not accessible.
            APIKeyInvalidError: Authentication failed.
        """
        self._model_name = model

        # Verify model can actually respond
        self._verify_model_responds()

    def _verify_model_responds(self) -> None:
        """Verify the model can actually respond.

        Sends a minimal test message to catch issues like:
        - Invalid model name
        - Authentication issues
        - Model access restrictions
        """
        if not self._client:
            return  # Will fail later with clear error

        try:
            # Send minimal request to verify model responds
            self._client.messages.create(
                model=self._model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
        except Exception as e:
            # Use our error handler to provide helpful messages
            self._handle_api_error(e)

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

    def _is_using_oauth(self) -> bool:
        """Check if OAuth authentication is being used."""
        return self._use_pkce or bool(self._oauth_token)

    def _build_system_blocks_from(
        self, system_instruction: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """Build system instruction content blocks in Anthropic API format.

        Used by ``complete()`` to convert the system prompt into the
        Anthropic API's content-block list format.

        Handles OAuth identity prepending.  Does NOT apply cache_control --
        that is handled by the cache plugin.

        Args:
            system_instruction: The system prompt text, or None.

        Returns:
            List of system content blocks, or None if no system instruction.
        """
        if self._is_using_oauth():
            combined_system = CLAUDE_CODE_IDENTITY
            if system_instruction:
                combined_system = f"{CLAUDE_CODE_IDENTITY}\n\n{system_instruction}"
            return [{"type": "text", "text": combined_system}]
        elif system_instruction:
            return [{"type": "text", "text": system_instruction}]
        return None

    def _build_tool_list_from(
        self, tools: Optional[List[ToolSchema]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Build tool definitions in Anthropic API format.

        Used by ``complete()`` to convert tool schemas into the
        Anthropic API's tool definition format.

        Sorts by name for cache stability.  Does NOT apply cache_control --
        that is handled by the cache plugin.

        Args:
            tools: List of tool schemas, or None.

        Returns:
            Sorted list of tool dicts, or None if no tools.
        """
        if not tools:
            return None
        anthropic_tools = tool_schemas_to_anthropic(tools)
        if not anthropic_tools:
            return None
        # Sort by name for consistent ordering (improves cache hits)
        return sorted(anthropic_tools, key=lambda t: t["name"])

    def _is_thinking_capable(self) -> bool:
        """Check if the current model supports extended thinking."""
        if not self._model_name:
            return False
        for prefix in THINKING_CAPABLE_MODELS:
            if self._model_name.startswith(prefix):
                return True
        return False

    def _compute_history_cache_breakpoint_from(
        self, messages: List[Message]
    ) -> int:
        """Compute the optimal history index for cache breakpoint BP3.

        Operates on the given message list. Used by ``complete()`` to
        determine where to place cache_control annotations in the
        conversation history.

        Delegates to the attached ``CachePlugin`` for budget-aware placement.
        Without a plugin, returns -1 (no history caching).

        Args:
            messages: The conversation history to search for breakpoint.

        Returns:
            Message index for cache_control, or -1 to skip history caching.
        """
        if not self._cache_plugin:
            return -1

        # The plugin's prepare_request already computed the breakpoint.
        # Use its internal result if available (-2 = budget-based).
        bp = getattr(self._cache_plugin, '_budget_bp3_message_id', None)
        if bp is not None:
            idx = self._resolve_message_id_to_index_in(messages, message_id=bp)
            if idx >= 0:
                return idx

        return -1

    @staticmethod
    def _resolve_message_id_to_index_in(
        messages: List[Message], message_id: str
    ) -> int:
        """Find the index of a message by its ID in the given list.

        Searches backward since the target is typically near the end
        of the stable prefix (before recent ephemeral turns).

        Args:
            messages: The message list to search.
            message_id: The message ID to find.

        Returns:
            Index in the list, or -1 if not found.
        """
        for i in range(len(messages) - 1, -1, -1):
            if getattr(messages[i], 'id', None) == message_id:
                return i
        return -1

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors and convert to appropriate exceptions.

        Detects SSL/TLS errors (common with corporate proxies) and provides
        actionable guidance via shared.ssl_helper.
        """
        import ssl as _ssl

        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check for SSL/TLS errors (corporate proxy TLS inspection, missing CA certs)
        ssl_keywords = ("ssl", "handshake_failure", "certificate_verify_failed", "sslv3")
        is_ssl_error = (
            isinstance(error, _ssl.SSLError)
            or any(kw in error_str for kw in ssl_keywords)
        )
        if is_ssl_error:
            from shared.ssl_helper import log_ssl_guidance
            log_ssl_guidance("Anthropic API", error)

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

        # Check for usage limit errors (API spending/quota limits)
        if "usage limit" in error_str or "api usage" in error_str:
            # Try to extract reset date from error message
            reset_date = None
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', str(error))
            if date_match:
                reset_date = date_match.group(1)
            raise UsageLimitError(
                reset_date=reset_date,
                original_error=str(error),
            ) from error

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

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Set the thinking/reasoning mode configuration.

        Dynamically enables or disables extended thinking for subsequent
        API calls.

        Args:
            config: ThinkingConfig with enabled flag and budget.
        """
        self._enable_thinking = config.enabled
        self._thinking_budget = config.budget

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

    # ==================== Agent Context & Tracing ====================

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
            return "anthropic:main"
        elif self._agent_name:
            return f"anthropic:subagent:{self._agent_name}"
        else:
            return f"anthropic:subagent:{self._agent_id}"

    def _trace(self, msg: str) -> None:
        """Write trace message for debugging provider interactions.

        No-op by default. Subclasses (e.g., ZhipuAIProvider) override
        this to write to the provider trace log.
        """
        pass

    # ==================== Cache Plugin Delegation ====================

    def set_cache_plugin(self, plugin: Any) -> None:
        """Attach a cache control plugin for delegated breakpoint placement.

        When set, the provider delegates cache annotation decisions
        (breakpoint placement, threshold checks) to this plugin instead
        of using provider-internal logic.  This decouples cache strategy
        from provider implementation, allowing ZhipuAIProvider and
        OllamaProvider to inherit from AnthropicProvider without
        inheriting the wrong cache logic.

        Args:
            plugin: A CachePlugin instance (duck-typed).
        """
        self._cache_plugin = plugin

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
    ) -> ProviderResponse:
        """Stateless completion: convert messages to provider format, call API, return response.

        The caller (session) is responsible for maintaining the message list
        and passing it in full each call. This method does not maintain any
        internal conversation state.

        When ``on_chunk`` is provided, the response is streamed token-by-token
        via ``_stream_response()``. When ``on_chunk`` is None, the response
        is returned in batch mode via ``messages.create()``.

        Args:
            messages: Full conversation history in provider-agnostic Message
                format. Must already include the latest user message or tool
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
            ProviderResponse with text, function calls, and usage.

        Raises:
            RuntimeError: If provider is not initialized/connected.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not connected. Call initialize() and connect() first.")

        # Validate and repair message history (defensive against cancellation artifacts)
        validated = validate_tool_use_pairing(list(messages))

        # Build API kwargs from explicit parameters (NOT instance state)
        kwargs: Dict[str, Any] = {}

        # Max tokens (higher if thinking is enabled)
        if self._enable_thinking and self._is_thinking_capable():
            kwargs["max_tokens"] = EXTENDED_MAX_TOKENS
        else:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        # System instruction (parameterized)
        system_blocks = self._build_system_blocks_from(system_instruction)
        if system_blocks:
            kwargs["system"] = system_blocks

        # Tools (parameterized)
        tool_list = self._build_tool_list_from(tools)
        if tool_list is not None:
            kwargs["tools"] = tool_list

        # Delegate cache annotations to plugin if attached
        if self._cache_plugin:
            cache_result = self._cache_plugin.prepare_request(
                system=kwargs.get("system"),
                tools=kwargs.get("tools", []),
                messages=[],  # Messages are handled separately via cache_breakpoint_index
            )
            if cache_result.get("system") is not None:
                kwargs["system"] = cache_result["system"]
            if cache_result.get("tools"):
                kwargs["tools"] = cache_result["tools"]

        # Extended thinking
        if self._enable_thinking and self._is_thinking_capable():
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            }

        # Compute history cache breakpoint from the passed messages
        history_breakpoint = self._compute_history_cache_breakpoint_from(validated)

        # Convert to Anthropic API format
        api_messages = messages_to_anthropic(
            validated, cache_breakpoint_index=history_breakpoint
        )

        try:
            if on_chunk:
                # Streaming mode
                provider_response = self._stream_response(
                    messages=api_messages,
                    kwargs=kwargs,
                    on_chunk=on_chunk,
                    cancel_token=cancel_token,
                    on_usage_update=on_usage_update,
                    on_function_call=on_function_call,
                    on_thinking=on_thinking,
                )
            else:
                # Batch mode
                response = self._client.messages.create(
                    model=self._model_name,
                    messages=api_messages,
                    **kwargs,
                )
                provider_response = response_from_anthropic(response)

            # Update last_usage (this is per-call accounting, not conversation state)
            self._last_usage = provider_response.usage

            # Handle structured output via response parsing
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

    # ==================== Streaming ====================

    def _stream_response(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> ProviderResponse:
        """Stream a response from the Anthropic API.

        Internal method used by ``complete()`` when ``on_chunk`` is provided.
        Accumulates text, thinking, and function call parts from the stream
        events, invoking callbacks as chunks arrive.
        """
        # State for accumulating response
        accumulated_text: List[str] = []  # Text chunks for current text block
        accumulated_thinking: List[str] = []  # Thinking chunks
        thinking_emitted = False  # Whether thinking was emitted via callback
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
            self._trace(f"STREAM_START msg_count={len(messages)}")
            chunk_count = 0
            with self._client.messages.stream(
                model=self._model_name,
                messages=messages,
                **kwargs,
            ) as stream:
                for event in stream:
                    # Check for cancellation
                    if cancel_token and cancel_token.is_cancelled:
                        self._trace(f"STREAM_CANCELLED after {chunk_count} chunks")
                        was_cancelled = True
                        finish_reason = FinishReason.CANCELLED
                        break

                    # Handle message_start (initial usage)
                    initial_usage = extract_message_start(event)
                    if initial_usage:
                        usage = initial_usage
                        self._trace(f"STREAM_MSG_START prompt={usage.prompt_tokens} cache_creation={usage.cache_creation_tokens} cache_read={usage.cache_read_tokens}")
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)

                    # Handle content_block_start (new text/tool_use block)
                    block_info = extract_content_block_start(event)
                    if block_info:
                        if block_info["type"] == "tool_use":
                            # Emit thinking before tool calls if not yet emitted
                            # (handles thinking → tool_use without text)
                            if not thinking_emitted and accumulated_thinking and on_thinking:
                                thinking_text = ''.join(accumulated_thinking)
                                on_thinking(thinking_text)
                                thinking_emitted = True
                            # Start tracking a new tool call
                            idx = block_info["index"]
                            self._trace(f"STREAM_TOOL_START idx={idx} name={block_info['name']}")
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
                        # Emit accumulated thinking before first text chunk
                        # (model thinks first, then speaks)
                        if not thinking_emitted and accumulated_thinking and on_thinking:
                            thinking_text = ''.join(accumulated_thinking)
                            on_thinking(thinking_text)
                            thinking_emitted = True
                        chunk_count += 1
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

                            # Restore original tool name if it was sanitized
                            original_name = get_original_tool_name(tc["name"])
                            fc = FunctionCall(
                                id=tc["id"],
                                name=original_name,
                                args=args,
                            )
                            self._trace(f"STREAM_FUNC_CALL name={fc.name}")
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
                            # Update output tokens; preserve all fields set by message_start
                            # (cache_read_tokens, cache_creation_tokens, thinking_tokens, etc.)
                            usage.output_tokens = delta_usage.output_tokens
                            usage.total_tokens = usage.prompt_tokens + delta_usage.output_tokens
                            self._trace(f"STREAM_USAGE prompt={usage.prompt_tokens} output={usage.output_tokens} total={usage.total_tokens}")
                            if on_usage_update and usage.total_tokens > 0:
                                on_usage_update(usage)

            self._trace(f"STREAM_END chunks={chunk_count} finish_reason={finish_reason}")

        except Exception as e:
            self._trace(f"STREAM_ERROR {type(e).__name__}: {e}")
            # If cancelled during iteration, treat as cancellation
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                raise

        # Flush any remaining text
        flush_text_block()

        # Handle incomplete tool calls only if NOT cancelled
        # When cancelled, incomplete tool calls would create unpaired tool_use blocks
        if not was_cancelled:
            for idx, tc in current_tool_calls.items():
                json_str = ''.join(tc["json_chunks"])
                try:
                    args = json.loads(json_str) if json_str else {}
                except json.JSONDecodeError:
                    args = {}
                # Restore original tool name if it was sanitized
                original_name = get_original_tool_name(tc["name"])
                fc = FunctionCall(id=tc["id"], name=original_name, args=args)
                # Notify caller about function call detection
                if on_function_call:
                    on_function_call(fc)
                parts.append(Part.from_function_call(fc))

        # Build thinking string
        thinking = ''.join(accumulated_thinking) if accumulated_thinking else None

        # Estimate thinking tokens from accumulated text (streaming doesn't provide
        # separate thinking token counts in message_delta)
        if thinking and usage.thinking_tokens is None:
            usage.thinking_tokens = max(1, len(thinking) // 4)

        # When cancelled, filter out function_call parts to prevent unpaired tool_use blocks
        # These would cause API errors on next call since there won't be tool_results
        if was_cancelled:
            parts = [p for p in parts if p.function_call is None]

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

    # ==================== Error Classification for Retry ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry purposes.

        Anthropic SDK has specific error types for rate limits and overload.

        Args:
            exc: The exception to classify.

        Returns:
            Classification dict or None to use fallback.
        """
        from .errors import RateLimitError, OverloadedError

        if isinstance(exc, RateLimitError):
            return {"transient": True, "rate_limit": True, "infra": False}
        if isinstance(exc, OverloadedError):
            return {"transient": True, "rate_limit": False, "infra": True}

        # Fall back to global classification
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract retry-after hint from an exception.

        Anthropic's RateLimitError includes retry_after attribute.

        Args:
            exc: The exception to extract retry-after from.

        Returns:
            Suggested delay in seconds, or None if not available.
        """
        from .errors import RateLimitError

        if isinstance(exc, RateLimitError) and exc.retry_after:
            return float(exc.retry_after)

        return None


def create_provider() -> AnthropicProvider:
    """Factory function for plugin discovery."""
    return AnthropicProvider()
