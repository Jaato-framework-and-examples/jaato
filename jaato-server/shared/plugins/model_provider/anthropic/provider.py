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
    ThinkingConfig,
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
    resolve_enable_caching,
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

# Cache configuration
# Anthropic requires minimum token thresholds for caching to be effective
# Claude 3.5 Sonnet: 1024 tokens minimum
# Other models (Haiku, Opus): 2048 tokens minimum
CACHE_MIN_TOKENS_SONNET = 1024
CACHE_MIN_TOKENS_OTHER = 2048

# Maximum cache breakpoints allowed per request
MAX_CACHE_BREAKPOINTS = 4

# Default number of recent turns to exclude from history caching
# (newer messages are less stable and may not benefit from caching)
DEFAULT_CACHE_EXCLUDE_RECENT_TURNS = 2


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

        # Cache optimization settings
        self._cache_history: bool = True  # Cache historical messages
        self._cache_exclude_recent_turns: int = DEFAULT_CACHE_EXCLUDE_RECENT_TURNS
        self._cache_min_tokens: bool = True  # Enforce minimum token threshold

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

        # Try PKCE OAuth first (interactive login tokens)
        try:
            self._pkce_access_token = get_valid_access_token()
            if self._pkce_access_token:
                self._use_pkce = True
        except Exception:
            # PKCE token refresh failed, will try other methods
            self._pkce_access_token = None

        if not self._pkce_access_token and not self._oauth_token and not self._api_key:
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

        # Cache optimization settings
        # cache_history: Whether to add a cache breakpoint in message history
        self._cache_history = config.extra.get("cache_history", True)
        # cache_exclude_recent_turns: Number of recent turns to exclude from caching
        # (these are less stable and may reduce cache hit rate)
        self._cache_exclude_recent_turns = config.extra.get(
            "cache_exclude_recent_turns", DEFAULT_CACHE_EXCLUDE_RECENT_TURNS
        )
        # cache_min_tokens: Whether to enforce minimum token threshold for caching
        # (Anthropic ignores cache markers on content smaller than threshold)
        self._cache_min_tokens = config.extra.get("cache_min_tokens", True)

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
        self._history = []

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

        # Validate and repair history (in case of prior cancellation/errors)
        self._history = validate_tool_use_pairing(self._history)

        # Compute history cache breakpoint (Cache breakpoint #3)
        history_breakpoint = self._compute_history_cache_breakpoint()

        # Build messages for API with optional cache breakpoint
        messages = messages_to_anthropic(self._history, cache_breakpoint_index=history_breakpoint)

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

        # Validate and repair history (in case of prior cancellation/errors)
        self._history = validate_tool_use_pairing(self._history)

        # Compute history cache breakpoint (Cache breakpoint #3)
        history_breakpoint = self._compute_history_cache_breakpoint()

        # Build messages for API with optional cache breakpoint
        messages = messages_to_anthropic(self._history, cache_breakpoint_index=history_breakpoint)

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

        # Validate and repair history (in case of prior cancellation/errors)
        self._history = validate_tool_use_pairing(self._history)

        # Compute history cache breakpoint (Cache breakpoint #3)
        history_breakpoint = self._compute_history_cache_breakpoint()

        # Build messages for API with optional cache breakpoint
        messages = messages_to_anthropic(self._history, cache_breakpoint_index=history_breakpoint)

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
            # Rollback the tool results message we added
            if self._history and self._history[-1].role == Role.TOOL:
                self._history.pop()
            self._handle_api_error(e)
            raise

    def _is_using_oauth(self) -> bool:
        """Check if OAuth authentication is being used."""
        return self._use_pkce or bool(self._oauth_token)

    def _build_api_kwargs(self, response_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build kwargs for the messages.create() call.

        Cache optimization strategy (up to 4 breakpoints):
        1. System instruction (most stable)
        2. Tool definitions (stable within session, sorted for consistency)
        3. Historical messages (computed breakpoint in older turns)
        4. Reserved for future use (large document injection, etc.)

        Note: Breakpoint 3 (history) is applied in send_message() via
        messages_to_anthropic(cache_breakpoint_index=...) since it needs
        access to the converted message list.
        """
        kwargs: Dict[str, Any] = {}

        # Max tokens (higher if thinking is enabled)
        if self._enable_thinking and self._is_thinking_capable():
            kwargs["max_tokens"] = EXTENDED_MAX_TOKENS
        else:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        # System instruction (Cache breakpoint #1)
        # When using OAuth tokens, prepend Claude Code identity (server-side validation)
        if self._is_using_oauth():
            # OAuth requires Claude Code identity
            # Combine identity + instruction into single block for better caching
            combined_system = CLAUDE_CODE_IDENTITY
            if self._system_instruction:
                combined_system = f"{CLAUDE_CODE_IDENTITY}\n\n{self._system_instruction}"

            # Only add cache_control if content meets threshold
            system_block: Dict[str, Any] = {
                "type": "text",
                "text": combined_system,
            }
            if self._enable_caching and self._should_cache_content(combined_system):
                system_block["cache_control"] = {"type": "ephemeral"}
            kwargs["system"] = [system_block]

        elif self._system_instruction:
            system_block = {
                "type": "text",
                "text": self._system_instruction,
            }
            # Only add cache_control if caching enabled and meets threshold
            if self._enable_caching and self._should_cache_content(self._system_instruction):
                system_block["cache_control"] = {"type": "ephemeral"}
            kwargs["system"] = [system_block]

        # Tools (Cache breakpoint #2)
        if self._tools:
            anthropic_tools = tool_schemas_to_anthropic(self._tools)
            if anthropic_tools:
                # Sort tools by name for consistent ordering (improves cache hits)
                # Tool order changes would otherwise invalidate the cache
                anthropic_tools = sorted(anthropic_tools, key=lambda t: t["name"])

                # Add cache control to last tool if caching enabled
                if self._enable_caching and len(anthropic_tools) > 0:
                    # Estimate combined size of all tools for threshold check
                    tools_json = json.dumps(anthropic_tools)
                    if self._should_cache_content(tools_json):
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

    def _get_cache_min_tokens(self) -> int:
        """Get minimum token threshold for caching based on model.

        Anthropic requires minimum token thresholds:
        - Claude 3.5 Sonnet: 1024 tokens
        - Other models: 2048 tokens
        """
        if not self._model_name:
            return CACHE_MIN_TOKENS_OTHER
        if "sonnet" in self._model_name.lower() and "3-5" in self._model_name:
            return CACHE_MIN_TOKENS_SONNET
        return CACHE_MIN_TOKENS_OTHER

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content.

        Uses a rough heuristic of ~4 characters per token.
        For accurate counts, use count_tokens() but that requires an API call.
        """
        return len(content) // 4

    def _should_cache_content(self, content: str) -> bool:
        """Check if content meets minimum token threshold for caching.

        Args:
            content: Text content to check.

        Returns:
            True if content is large enough to benefit from caching.
        """
        if not self._cache_min_tokens:
            return True  # Skip threshold check if disabled
        min_tokens = self._get_cache_min_tokens()
        estimated = self._estimate_tokens(content)
        return estimated >= min_tokens

    def _compute_history_cache_breakpoint(self) -> int:
        """Compute the optimal history index for a cache breakpoint.

        Returns the index of the message that should receive cache_control,
        or -1 if history caching should be disabled.

        Strategy:
        - Skip if caching disabled or history too short
        - Find the last assistant message before the "exclude recent" window
        - This creates a stable cache key for older conversation context
        """
        if not self._enable_caching or not self._cache_history:
            return -1

        if not self._history:
            return -1

        # Count turns (user-assistant pairs)
        # We want to cache up to but not including the last N turns
        exclude_count = self._cache_exclude_recent_turns
        if exclude_count <= 0:
            # Cache everything except the very last message (current turn)
            exclude_count = 1

        # Find candidate: last assistant message before the exclusion window
        # Walk backward to find the Nth user message from the end
        user_count = 0
        for i in range(len(self._history) - 1, -1, -1):
            if self._history[i].role == Role.USER:
                user_count += 1
                if user_count >= exclude_count:
                    # Found the boundary - cache everything before this user message
                    # Find the last assistant message before this point
                    for j in range(i - 1, -1, -1):
                        if self._history[j].role == Role.MODEL:
                            return j
                    # No assistant message found before this point
                    return -1
        return -1

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

    # ==================== Streaming ====================

    def send_message_streaming(
        self,
        message: str,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None
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

        # Validate and repair history (in case of prior cancellation/errors)
        self._history = validate_tool_use_pairing(self._history)

        # Compute history cache breakpoint (Cache breakpoint #3)
        history_breakpoint = self._compute_history_cache_breakpoint()

        # Build messages for API with optional cache breakpoint
        messages = messages_to_anthropic(self._history, cache_breakpoint_index=history_breakpoint)

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
                on_thinking=on_thinking,
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
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None
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

        # Validate and repair history (in case of prior cancellation/errors)
        self._history = validate_tool_use_pairing(self._history)

        # Compute history cache breakpoint (Cache breakpoint #3)
        history_breakpoint = self._compute_history_cache_breakpoint()

        # Build messages for API with optional cache breakpoint
        messages = messages_to_anthropic(self._history, cache_breakpoint_index=history_breakpoint)

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
                on_thinking=on_thinking,
            )

            self._last_usage = response.usage

            # Add assistant response to history
            self._add_response_to_history(response)

            return response
        except Exception as e:
            # Rollback the tool results message we added
            if self._history and self._history[-1].role == Role.TOOL:
                self._history.pop()
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
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> ProviderResponse:
        """Stream a response from the Anthropic API.

        Internal method used by both send_message_streaming and
        send_tool_results_streaming.
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
                            # Combine with existing input tokens
                            usage = TokenUsage(
                                prompt_tokens=usage.prompt_tokens,
                                output_tokens=delta_usage.output_tokens,
                                total_tokens=usage.prompt_tokens + delta_usage.output_tokens,
                            )
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
