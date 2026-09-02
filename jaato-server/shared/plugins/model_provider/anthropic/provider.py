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
import logging
import os
import re
from typing import Any, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)

from ..base import (
    MODALITY_TEXT,
    ModalityCapabilityMixin,
    FunctionCallDetectedCallback,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
    resolve_context_window,
    resolve_modalities,
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
    TurnResult,
    parse_tool_call_arguments,
    StreamInterruptedError,
    require_terminated_stream,
    resolve_tool_use_finish,
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
    try_load_tokens_with_reason,
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

# INPUT modalities per Claude model family.  All shipping Claude API
# models (3.x, 4.x) accept image input alongside text; Claude 2.x was
# text-only and isn't listed.  Prefix-matched like MODEL_CONTEXT_LIMITS;
# a model absent here resolves to the text-only floor in modalities()
# (never a false image claim).
MODEL_INPUT_MODALITIES: Dict[str, FrozenSet[str]] = {
    "claude-opus-4": frozenset({"text", "image", "file"}),
    "claude-sonnet-4": frozenset({"text", "image", "file"}),
    "claude-haiku-4": frozenset({"text", "image", "file"}),
    # claude-3 prefix covers 3.5/3.7 (PDF-capable); 3.0 is EOL.
    "claude-3": frozenset({"text", "image", "file"}),
}

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


class AnthropicProvider(ModalityCapabilityMixin):
    """Stateless Anthropic Claude provider.

    This provider uses a stateless design: the caller (session) owns the
    conversation history and passes it to ``complete()`` on every call.
    The provider does not maintain internal message state.

    Subclasses override the ``_provider_display_name`` /
    ``_provider_console_url`` class attributes so vendor-targeted error
    messages from ``_handle_api_error`` name the actual provider
    (e.g. ZhipuAIProvider sets ``"Zhipu AI"``).  Without an override
    the defaults are correct for Anthropic.

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

    # Vendor identity for error messages produced by
    # ``_handle_api_error``.  Subclasses (ZhipuAIProvider, etc.) override
    # these class attributes; ``_handle_api_error`` reads them via
    # ``self`` so dynamic dispatch picks the right vendor for the
    # subclass instance.
    _provider_display_name: str = "Anthropic API"
    _provider_console_url: str = "https://console.anthropic.com/"

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[Any] = None  # anthropic.Anthropic
        self._model_name: Optional[str] = None

        # Per-profile context-window override (framework_overrides.context_length).
        # When set, wins over the per-model MODEL_CONTEXT_LIMITS table in
        # get_context_limit() — the escape hatch for a model not yet in the
        # table.  None = use the table; unknown model + no override → raise
        # (no hardcoded fallback, per project rule).
        self._context_length_knob: Optional[int] = None
        # INPUT-modality assertion (framework_overrides.modalities) —
        # the escape hatch layered over MODEL_INPUT_MODALITIES.
        self._modalities_knob: Optional[List[str]] = None

        # Configuration
        self._api_key: Optional[str] = None
        self._enable_thinking: bool = False
        self._thinking_budget: int = 10000

        # Sampling parameters (None = let the API apply its server-side
        # default — Anthropic Messages API defaults to temperature=1.0).
        # Wired through to ``messages.create()`` only when set on a
        # profile, mirroring openrouter's namespaced api_params layer.
        self._temperature: Optional[float] = None
        self._top_p: Optional[float] = None
        self._top_k: Optional[int] = None
        # Profile-level override of the framework's hard-coded
        # DEFAULT_MAX_TOKENS / EXTENDED_MAX_TOKENS choice.  ``None``
        # keeps the existing thinking-aware default selection.
        self._max_tokens_override: Optional[int] = None

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

        # Stash the config so post-init helpers (token refresh, status
        # reporting) can reuse the workspace_path / config_root that
        # the runtime injected.
        self._config = config

        # Pull workspace_path / config_root from config.extra (injected
        # by JaatoRuntime.create_provider).  Threading them explicitly
        # makes credential lookup independent of the
        # ``JAATO_WORKSPACE_ROOT`` / ``JAATO_CONFIG_ROOT`` env vars,
        # which are unreliable for headless reactor-spawned sessions
        # running in fresh threads outside any active ``_in_workspace``
        # context.
        workspace_path = config.extra.get('workspace_path')
        config_root = config.extra.get('config_root')

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
            self._pkce_access_token = get_valid_access_token(
                workspace_path=workspace_path, config_root=config_root,
            )
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

        # Parse extra config — namespaced into the same four layers as
        # the openrouter provider (server 0.6.23+):
        #
        #   plugin_configs.anthropic:
        #     <top-level>           # auth / identity (api_key, oauth_token)
        #     api_params:           # Anthropic Messages API request body
        #                           # (temperature, top_p, top_k, max_tokens,
        #                           #  enable_thinking, thinking_budget)
        #     framework_overrides:  # rare escape hatches (none defined for
        #                           # anthropic today; reserved for future use)
        #
        # The ``routing`` layer is omitted — Anthropic's API has no gateway
        # routing extension equivalent to OpenRouter's ``provider`` field.
        #
        # Backward compatibility: every key is also read from the legacy
        # flat position with a one-time deprecation warning per key.  Flat
        # support will be removed in a future release.
        api_params = config.extra.get("api_params") or {}
        framework_overrides = config.extra.get("framework_overrides") or {}

        def _knob(
            key: str, *, layer: Dict[str, Any], default: Any = None,
        ) -> Any:
            """Read a config knob from its nested layer first, falling
            back to the legacy flat ``config.extra[key]`` position with
            a deprecation warning when only the flat form is present."""
            if key in layer:
                return layer[key]
            if key in config.extra:
                logger.warning(
                    "Anthropic profile uses legacy flat config key %r — "
                    "move under the appropriate nested layer "
                    "(api_params / framework_overrides) per the 0.6.24+ "
                    "namespacing.  Flat-key support will be removed in a "
                    "future release.",
                    key,
                )
                return config.extra[key]
            return default

        # Thinking knobs (kept on api_params since they translate to wire
        # fields — Anthropic's ``thinking`` request body extension).
        self._enable_thinking = _knob(
            "enable_thinking", layer=api_params, default=resolve_enable_thinking(),
        )
        self._thinking_budget = _knob(
            "thinking_budget", layer=api_params, default=resolve_thinking_budget(),
        )

        # Context-window override (framework_overrides.context_length) via the
        # shared precedence helper.  Anthropic has no live capacity endpoint and
        # no env var for this, so detect_capacity/env are absent — the knob is
        # the only override tier, layered over the per-model MODEL_CONTEXT_LIMITS
        # table consulted in get_context_limit().
        self._context_length_knob = resolve_context_window(
            detect_capacity=None,
            profile_value=_knob("context_length", layer=framework_overrides),
            env_value=None,
        )

        # INPUT-modality assertion (framework_overrides.modalities) — the
        # escape hatch for a model not in MODEL_INPUT_MODALITIES, or to
        # correct it.  Layered over the table in modalities().
        modalities_override = _knob("modalities", layer=framework_overrides)
        if modalities_override is not None:
            if not isinstance(modalities_override, (list, tuple)) or not all(
                isinstance(m, str) for m in modalities_override
            ):
                raise TypeError(
                    "Anthropic 'modalities' config must be a list of "
                    f"strings (e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

        # Sampling parameters.  ``None`` means "omit from the request and
        # let Anthropic apply its server-side default" (temperature=1.0).
        # Profiles wanting determinism set ``api_params.temperature: 0.0``.
        temp_extra = _knob("temperature", layer=api_params)
        if temp_extra is not None:
            self._temperature = float(temp_extra)
        top_p_extra = _knob("top_p", layer=api_params)
        if top_p_extra is not None:
            self._top_p = float(top_p_extra)
        top_k_extra = _knob("top_k", layer=api_params)
        if top_k_extra is not None:
            self._top_k = int(top_k_extra)
        max_tokens_extra = _knob("max_tokens", layer=api_params)
        if max_tokens_extra is not None:
            self._max_tokens_override = int(max_tokens_extra)

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

        # Pull the workspace_path / config_root the runtime injected
        # so refresh writes back to the same credential file we
        # originally loaded from.
        extra = getattr(self._config, 'extra', None) or {}
        ws_path = extra.get('workspace_path')
        cr = extra.get('config_root')

        tokens = load_tokens(workspace_path=ws_path, config_root=cr)
        if tokens and tokens.is_expired:
            try:
                new_tokens = refresh_tokens(tokens.refresh_token)
                save_tokens(new_tokens, workspace_path=ws_path, config_root=cr)
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
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify that authentication is configured and optionally trigger interactive login.

        This can be called BEFORE initialize() to ensure credentials are available.

        ``config`` is accepted for protocol compatibility but unused: Anthropic
        resolves credentials from environment, OAuth, and PKCE storage rather
        than from the profile's ``plugin_configs``.
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
        #
        # Before trying a refresh, inspect the on-disk token file so we can
        # tell "no tokens yet" from "file exists but cannot be parsed".
        # A broken token file used to look identical to "never logged in",
        # which is the exact "provider error not being surfaced" bug the
        # branch name calls out.
        # Pull workspace_path / config_root from config.extra so
        # verify_auth surfaces credentials from the same path the
        # runtime configured the provider with.  Read the ``config``
        # PARAMETER (not ``self._config``): per the base contract
        # verify_auth runs BEFORE initialize(), so ``self._config`` is
        # unset here — reading it raised AttributeError on the
        # in-process runtime path, which does not initialize() first.
        extra = getattr(config, 'extra', None) or {}
        ws_path = extra.get('workspace_path')
        cr = extra.get('config_root')

        tokens, load_error = try_load_tokens_with_reason(
            workspace_path=ws_path, config_root=cr,
        )
        if tokens:
            try:
                pkce_token = get_valid_access_token(
                    workspace_path=ws_path, config_root=cr,
                )
                if pkce_token:
                    if on_message:
                        on_message("Found valid PKCE OAuth token")
                    return True
            except Exception as refresh_err:
                # Token refresh failed — surface it so users see the
                # actual refresh error rather than falling through to a
                # misleading "no credentials" message.
                if on_message:
                    on_message(
                        f"Stored PKCE OAuth tokens could not be refreshed: "
                        f"{refresh_err.__class__.__name__}: {refresh_err}"
                    )
        elif load_error:
            # File exists but could not be loaded (corrupt JSON, missing
            # field, permission error).  Surface the real reason.
            if on_message:
                on_message(
                    f"Anthropic OAuth token file found but could not be loaded: "
                    f"{load_error}"
                )
                on_message(
                    "Run 'anthropic-auth login' to re-authenticate, or set "
                    "ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY."
                )

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
        if on_message and not load_error:
            # Only emit the generic "No credentials found" when we didn't
            # already surface a specific load error above; otherwise we'd
            # contradict ourselves.
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

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the model to use and optionally verify it responds.

        Args:
            model: Model ID (e.g., 'claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022').
            skip_model_test: If True, skip the network call to verify the model
                responds.  The model will be validated on the first real
                message instead.

        Raises:
            ModelNotFoundError: Model doesn't exist or is not accessible.
            APIKeyInvalidError: Authentication failed.
        """
        self._model_name = model

        if not skip_model_test:
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
            log_ssl_guidance(self._provider_display_name, error)

        # Check for authentication errors
        if "authentication" in error_str or "invalid api key" in error_str or "401" in error_str:
            raise APIKeyInvalidError(
                reason="API key rejected",
                key_prefix=self._api_key[:15] if self._api_key else None,
                original_error=str(error),
                provider_name=self._provider_display_name,
            ) from error

        # Check for rate limit errors
        if "rate" in error_str and "limit" in error_str or "429" in error_str:
            raise RateLimitError(
                original_error=str(error),
                provider_name=self._provider_display_name,
            ) from error

        # Check for usage limit errors (API spending/quota limits)
        if "usage limit" in error_str or "api usage" in error_str:
            # Try to extract a reset date or full timestamp from the
            # error message.  Providers vary in format — Zhipu emits
            # ``2026-05-15 18:06:38``; Anthropic emits a bare date.  We
            # capture the optional ``HH:MM:SS`` suffix when present so
            # the user gets the full reset time rather than just the day.
            reset_date = None
            date_match = re.search(
                r'(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?)', str(error)
            )
            if date_match:
                reset_date = date_match.group(1)
            raise UsageLimitError(
                reset_date=reset_date,
                original_error=str(error),
                provider_name=self._provider_display_name,
                console_url=self._provider_console_url,
            ) from error

        # Check for overloaded errors
        if "overloaded" in error_str or "529" in error_str:
            raise OverloadedError(
                original_error=str(error),
                provider_name=self._provider_display_name,
            ) from error

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

        Resolution precedence (no hardcoded fallback, per project rule):
        1. ``framework_overrides.context_length`` knob (``_context_length_knob``)
           — the escape hatch for a model not yet in the table.
        2. ``MODEL_CONTEXT_LIMITS`` prefix match — the documented per-model
           window (the authoritative source for closed Claude models).
        3. else raise — an unknown model with no override is a configuration
           error, surfaced loudly rather than papered over with a guess.

        Returns:
            Maximum tokens the model can handle.

        Raises:
            ValueError: when neither the knob nor the table yields a value.
        """
        if self._context_length_knob:
            return self._context_length_knob

        if self._model_name:
            for model_prefix, limit in MODEL_CONTEXT_LIMITS.items():
                if self._model_name.startswith(model_prefix):
                    return limit

        raise ValueError(
            f"Anthropic provider: no known context window for model "
            f"{self._model_name!r}, and no override is set.  Add the model to "
            f"MODEL_CONTEXT_LIMITS, or set framework_overrides.context_length in "
            f"the profile.  No hardcoded fallback exists per the project's "
            f"no-fallback rule."
        )

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities the active Claude model accepts.

        Precedence mirrors get_context_limit() (no live capacity
        endpoint, so detect is absent): framework_overrides.modalities
        knob -> MODEL_INPUT_MODALITIES prefix match -> text-only floor
        (every Claude model accepts text; image stays unconfirmed, so
        the content gate / vision-tier validation treats it text-only).
        """
        resolved = resolve_modalities(
            profile_value=self._modalities_knob,
            table_value=self._lookup_input_modalities(model),
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    def _lookup_input_modalities(
        self, model: Optional[str] = None
    ) -> Optional[FrozenSet[str]]:
        """Table-declared input modalities for ``model`` (or active)."""
        model = model or self._model_name
        if model:
            for prefix, mods in MODEL_INPUT_MODALITIES.items():
                if model.startswith(prefix):
                    return mods
        return None

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
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Stateless completion: convert messages to provider format, call API, return response.

        The caller (session) is responsible for maintaining the message list
        and passing it in full each call. This method does not maintain any
        internal conversation state.

        When ``on_chunk`` is provided, the response is streamed token-by-token
        via ``_stream_response()``. When ``on_chunk`` is None, the response
        is returned in batch mode via ``messages.create()``.

        Returns ``TurnResult.from_provider_response(r)`` on success,
        ``TurnResult.from_exception(exc)`` for non-transient errors, and
        **raises** transient errors (rate limits, overload) for ``with_retry``.

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
            tool_choice: Per-call lifecycle tool-choice hint (the session
                passes it generically — see ``ModelProvider.complete``
                contract in ``base.py``).  AnthropicProvider has no
                ``force_tool_choice_for_lifecycle``-style wire quirk, so
                it ACCEPTS and IGNORES this kwarg (the contract's "no-op"
                half).  Present purely for signature parity so the
                session can pass ``tool_choice`` to every provider
                without per-provider branching — without it, the
                keyword-only signature raised ``TypeError`` and broke any
                forced-completion stage (host_validator, build_descriptor)
                on z.ai / GLM-5, which routes through this adapter.

        Returns:
            A ``TurnResult`` classifying the outcome.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not connected. Call initialize() and connect() first.")

        # Validate and repair message history (defensive against cancellation artifacts)
        validated = validate_tool_use_pairing(list(messages))

        # Build API kwargs from explicit parameters (NOT instance state)
        kwargs: Dict[str, Any] = {}

        # Max tokens.  Profile override (api_params.max_tokens) wins;
        # otherwise pick the framework default based on whether thinking
        # is enabled (extended needs more output room for the trace +
        # the answer).
        if self._max_tokens_override is not None:
            kwargs["max_tokens"] = self._max_tokens_override
        elif self._enable_thinking and self._is_thinking_capable():
            kwargs["max_tokens"] = EXTENDED_MAX_TOKENS
        else:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        # Sampling parameters (api_params.{temperature, top_p, top_k}).
        # Only sent when the profile set them — omitting them lets
        # Anthropic apply its server-side defaults (temperature=1.0).
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if self._top_k is not None:
            kwargs["top_k"] = self._top_k

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

        # Diagnostic: when JAATO_DUMP_PROVIDER_REQUEST is set, dump the
        # full request payload (tools, system, messages) and the resulting
        # response (text + function_calls + finish_reason) so we can diff
        # what the model receives and produces across framework changes.
        # The marker tokens PROVIDER_REQUEST_DUMP / PROVIDER_RESPONSE_DUMP
        # make this greppable in mixed daemon logs.
        _dump_enabled = os.environ.get("JAATO_DUMP_PROVIDER_REQUEST", "").lower() in ("1", "true", "yes", "on")  # env: debug — log full request/response dumps (greppable PROVIDER_*_DUMP markers)
        if _dump_enabled:
            try:
                tools_in_kwargs = kwargs.get("tools") or []
                tool_names_in_request = [t.get("name") for t in tools_in_kwargs if isinstance(t, dict)]
                logger.info(
                    "PROVIDER_REQUEST_DUMP model=%s tool_count=%d tool_names=%s",
                    self._model_name,
                    len(tools_in_kwargs),
                    tool_names_in_request,
                )
                logger.info("PROVIDER_REQUEST_DUMP system=%s", json.dumps(kwargs.get("system")))
                logger.info("PROVIDER_REQUEST_DUMP tools=%s", json.dumps(tools_in_kwargs))
                logger.info("PROVIDER_REQUEST_DUMP messages=%s", json.dumps(api_messages))
            except Exception as _dump_err:
                logger.warning("PROVIDER_REQUEST_DUMP failed: %s", _dump_err)

        provider_response = None
        complete_exception: Optional[Exception] = None
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
        except Exception as e:
            complete_exception = e

        if _dump_enabled:
            try:
                if complete_exception is not None:
                    logger.info(
                        "PROVIDER_RESPONSE_DUMP outcome=exception exc_type=%s exc_msg=%s",
                        type(complete_exception).__name__,
                        str(complete_exception),
                    )
                elif provider_response is not None:
                    fcalls = [
                        {"name": fc.name, "args": fc.args}
                        for fc in provider_response.get_function_calls()
                    ]
                    logger.info(
                        "PROVIDER_RESPONSE_DUMP outcome=ok finish_reason=%s text_len=%d function_calls=%s",
                        getattr(provider_response, "finish_reason", None),
                        len(provider_response.get_text() or ""),
                        fcalls,
                    )
                    logger.info("PROVIDER_RESPONSE_DUMP text=%s", json.dumps(provider_response.get_text()))
                else:
                    logger.info("PROVIDER_RESPONSE_DUMP outcome=unreachable")
            except Exception as _dump_err:
                logger.warning("PROVIDER_RESPONSE_DUMP failed: %s", _dump_err)

        if complete_exception is not None:
            # An interrupted stream is not an API error to be mapped --
            # it is already the diagnosis, and already classified as
            # retryable (#687).  Checked BEFORE ``_handle_api_error``
            # because that mapper matches on message TEXT (a model id
            # containing "401" would become APIKeyInvalidError), and
            # re-raised rather than fallen through, because
            # ``TurnResult.from_exception`` below would make ``fn()``
            # return NORMALLY -- no retry, terminal error at the caller,
            # the same swallow that cost the SDK network errors below
            # their retries.  Subclasses that override the mapper
            # (ollama, zhipuai) inherit this, so it is the only place
            # the check has to exist.
            if isinstance(complete_exception, StreamInterruptedError):
                raise complete_exception
            try:
                self._handle_api_error(complete_exception)
            except Exception:
                raise
            # _handle_api_error converts to domain errors. Transient ones
            # (RateLimitError, OverloadedError) must propagate for with_retry.
            from .errors import RateLimitError as _RL, OverloadedError as _OL
            if isinstance(complete_exception, (_RL, _OL)):
                raise complete_exception
            # PR #177 (2026-05-21): anthropic SDK network-layer errors
            # (APIConnectionError, APITimeoutError) must ALSO propagate
            # to with_retry — they're classified as transient by
            # ANTHROPIC_TRANSIENT_CLASSES post-PR-175.  Without this
            # re-raise the SDK error gets swallowed into
            # ``TurnResult.from_exception`` below, ``with_retry`` sees
            # fn() return normally, no retry fires, the caller
            # surfaces it as MODEL_THREAD_TERMINAL_ERROR.  Surfaced by
            # kb-orchestrator v152-retry-11 (Finding C, 2026-05-21):
            # streaming chunk read raised APIConnectionError ~2s after
            # the SDK's internal retry succeeded with 200 OK — that
            # mid-stream disconnect never reached the classifier.
            #
            # Mirrors the _RL/_OL pattern above.  Defensive inner
            # try/except so older anthropic SDK versions (or test envs
            # without the SDK) don't break the existing _RL/_OL path.
            try:
                import anthropic as _anthropic_sdk
                if isinstance(complete_exception, (
                    _anthropic_sdk.APIConnectionError,
                    _anthropic_sdk.APITimeoutError,
                )):
                    raise complete_exception
            except (ImportError, AttributeError):
                pass
            return TurnResult.from_exception(complete_exception)

        # Update last_usage (this is per-call accounting, not conversation state)
        self._last_usage = provider_response.usage

        # Handle structured output via response parsing
        text = provider_response.get_text()
        if response_schema and text:
            try:
                provider_response.structured_output = json.loads(text)
            except json.JSONDecodeError:
                pass

        return TurnResult.from_provider_response(provider_response)

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
        # Whether Anthropic ever said the message ended.  ``message_stop``
        # is the spec's terminal event; ``message_delta.stop_reason`` is
        # accepted too because several Anthropic-compatible endpoints
        # (Z.AI, Ollama) send the reason and close without a separate
        # ``message_stop``.  Absent both, the stream was cut (#687).
        terminal_seen = False
        usage = TokenUsage()
        was_cancelled = False

        def flush_text_block():
            """Flush accumulated text as a single Part."""
            nonlocal accumulated_text
            if accumulated_text:
                text = ''.join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        chunk_count = 0
        try:
            # Use the streaming API
            self._trace(f"STREAM_START msg_count={len(messages)}")
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
                    # The spec's terminal event.  Recorded with ``|=``
                    # rather than a branch of its own because this loop
                    # is already the most complex function in the file
                    # and the ratchet holds it at its frozen size.
                    terminal_seen |= event_type == "message_stop"
                    if event_type == "content_block_stop":
                        idx = getattr(event, "index", None)
                        if idx is not None and idx in current_tool_calls:
                            # Finalize this tool call
                            tc = current_tool_calls[idx]
                            json_str = ''.join(tc["json_chunks"])
                            # Unreadable arguments stay unreadable (#750):
                            # the accumulated ``input_json_delta`` chunks
                            # may be a severed object, and an empty dict
                            # would present that as a zero-argument call.
                            args, unreadable_args = parse_tool_call_arguments(
                                json_str
                            )

                            # Flush text before adding function call
                            flush_text_block()

                            from shared.tool_id_map import id_to_name
                            fc = FunctionCall(
                                id=tc["id"],
                                name=id_to_name(tc["name"]),
                                args=args,
                                unreadable_args=unreadable_args,
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
                            terminal_seen = True
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
                            # message_delta usage is cumulative per Anthropic spec.
                            # Update prompt_tokens if the delta provides them
                            # (Z.AI sends input_tokens here; Anthropic may too
                            # when web search increases input mid-stream).
                            if delta_usage.prompt_tokens > 0:
                                usage.prompt_tokens = delta_usage.prompt_tokens
                            if delta_usage.cache_read_tokens is not None:
                                usage.cache_read_tokens = delta_usage.cache_read_tokens
                            if delta_usage.cache_creation_tokens is not None:
                                usage.cache_creation_tokens = delta_usage.cache_creation_tokens
                            usage.output_tokens = delta_usage.output_tokens
                            usage.total_tokens = usage.prompt_tokens + usage.output_tokens
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
            elif isinstance(e, ValueError):
                # Malformed SSE/JSON from the provider (e.g. Anthropic-
                # compatible APIs returning broken streaming chunks).
                # Recover gracefully: discard incomplete tool calls and
                # inject an error message so the model can self-correct.
                self._trace("STREAM_MALFORMED_RECOVERY discarding incomplete tool calls")
                current_tool_calls.clear()
                error_notice = (
                    "\n\n[Model returned malformed streaming data. "
                    "Your last response was cut short — the tool calls "
                    "you attempted were lost. Please try again.]"
                )
                accumulated_text.append(error_notice)
                on_chunk(error_notice)
                finish_reason = FinishReason.STOP
                # This branch has already accounted for the turn's end:
                # it discarded the incomplete calls and told the model
                # the response was cut short.  ``require_terminated_stream``
                # must not raise on top of that recovery (#687).
                terminal_seen = True
            else:
                raise

        # Flush any remaining text
        flush_text_block()

        # Handle incomplete tool calls only if NOT cancelled
        # When cancelled, incomplete tool calls would create unpaired tool_use blocks
        if not was_cancelled:
            for idx, tc in current_tool_calls.items():
                json_str = ''.join(tc["json_chunks"])
                # These are the calls whose ``content_block_stop`` never
                # arrived, so a severed ``input`` is the expected case
                # here rather than the exotic one (#750).
                args, unreadable_args = parse_tool_call_arguments(json_str)
                from shared.tool_id_map import id_to_name
                fc = FunctionCall(
                    id=tc["id"],
                    name=id_to_name(tc["name"]),
                    args=args,
                    unreadable_args=unreadable_args,
                )
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

        # TOOL_USE fills in an unreported or merely-``stop`` finish; it
        # must not displace a terminal one.  A turn that hit the output
        # cap mid-``arguments`` carries fragments, not a request — see
        # ``resolve_tool_use_finish`` and issue #745.
        finish_reason = resolve_tool_use_finish(
            finish_reason,
            has_function_calls=(
                any(p.function_call for p in parts) and not was_cancelled
            ),
        )

        # A stream that stopped arriving is not a turn that finished
        # (#687).  Raises rather than returning the fragment.
        return require_terminated_stream(
            ProviderResponse(
                parts=parts,
                usage=usage,
                finish_reason=finish_reason,
                raw=None,  # Streaming doesn't provide single raw response
                thinking=thinking,
            ),
            terminal_seen=terminal_seen,
            was_cancelled=was_cancelled,
            provider=self.name,
            model=self._model_name,
            chunks=chunk_count,
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
