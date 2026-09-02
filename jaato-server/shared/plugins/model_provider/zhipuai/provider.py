"""Zhipu AI (Z.AI) provider implementation.

This provider enables access to Zhipu AI's GLM models via the Anthropic-compatible
API endpoint, primarily targeting GLM Coding Plan subscribers.

Zhipu AI offers the GLM family of models including:
- GLM-5: Flagship MoE model with agentic engineering focus (200K context)
- GLM-4.7: Flagship with native chain-of-thought reasoning (200K context)
- GLM-4.7-Flash/Flashx: Fast inference variants (200K context)
- GLM-4.6: Previous flagship, strong coding (200K context)
- GLM-4.5/Air/Flash: Balanced and lightweight models (128K context)

Model discovery:
    The provider supports dynamic model listing via Z.AI's OpenAI-compatible
    ``GET /models`` endpoint (``/api/paas/v4/models``).  When an API key is
    available, ``list_models()`` queries this endpoint so that newly released
    models (e.g. GLM-5) appear automatically.  A static ``MODEL_CONTEXT_LIMITS``
    dict provides fallback metadata (context window sizes) for known models.

Usage:
    provider = ZhipuAIProvider()
    provider.initialize(ProviderConfig(api_key="your-key"))
    provider.connect('glm-5')
    response = provider.complete(messages=[...])

Environment variables:
    ZHIPUAI_API_KEY: Zhipu AI API key
    ZHIPUAI_BASE_URL: API base URL (default: https://api.z.ai/api/anthropic)
    ZHIPUAI_MODEL: Default model to use
    ZHIPUAI_CONTEXT_LENGTH: Override context length for models
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from ..anthropic.provider import AnthropicProvider
from ..base import (
    ProviderConfig,
    resolve_context_window,
    resolve_modalities,
    MODALITY_TEXT,
)
from .env import (
    DEFAULT_ZHIPUAI_BASE_URL,
    DEFAULT_ZHIPUAI_MODEL,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_enable_thinking,
    resolve_model,
    resolve_thinking_budget,
)
from .auth import (
    get_stored_api_key,
    get_stored_base_url,
    login_interactive,
    logout,
    status as auth_status,
    try_load_credentials_with_reason,
)


logger = logging.getLogger(__name__)


# Known GLM models with their context window sizes in tokens.
# Used as metadata fallback when the dynamic /models endpoint is unavailable.
# Source: models.dev, Roo Code, lm-deluge, ekai-gateway, moai-adk, Z.AI docs.
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
    # GLM-4 generation — 128K context (legacy / largely deprecated; not in the
    # live /models catalog, retained for backward compat).  The whole GLM-4
    # generation, including the 4V vision variant, is 128K.
    # Source: Z.AI docs; models.dev; Zhipu GLM-4.6V 128K announcement.
    "glm-4": 131072,
    "glm-4v": 131072,
}

# No blanket default for unknown models — context is resolved from the override
# (framework_overrides.context_length / ZHIPUAI_CONTEXT_LENGTH) then the table
# (exact then longest-prefix), else get_context_limit raises (project
# no-fallback rule).  The Z.AI /models endpoint is id-only (no context field,
# verified live), so there is no auto-detect tier.

KNOWN_MODELS = sorted(MODEL_CONTEXT_LIMITS.keys())

# GLM input-modality table (longest-prefix match).  The 4V / 4.5V family accepts
# images; the text/coding GLMs are text-only.  This replaces the inherited
# Anthropic ``MODEL_INPUT_MODALITIES`` table, whose ``claude-*`` prefixes never
# match a GLM model name — so the inherited modalities() override was inert and
# images were silently gated off even for the vision models.
GLM_INPUT_MODALITIES = {
    "glm-4.5v": frozenset({"text", "image"}),
    "glm-4v": frozenset({"text", "image"}),
}


# GLM models that support extended thinking (chain-of-thought reasoning).
# GLM-5 and GLM-4.7 (non-flash) have native chain-of-thought capability.
THINKING_CAPABLE_MODELS = [
    "glm-5",
    "glm-4.7",
]

# ── OpenAI-compatible models endpoint for dynamic discovery ───────────
# Z.AI exposes GET /models on the OpenAI-compatible API surface at
# open.bigmodel.cn (their main platform), NOT on the api.z.ai domain
# used for the Anthropic-compatible chat endpoint.
ZHIPUAI_MODELS_URL = "https://open.bigmodel.cn/api/paas/v4/models"


def fetch_zhipuai_models(api_key: str) -> List[str]:
    """Fetch available Z.AI models using the OpenAI-compatible endpoint.

    Low-level function used internally by the provider's ``list_models()``.
    For cross-provider workspace-aware model listing, use
    ``shared.plugins.model_provider.list_provider_models()`` instead.

    Args:
        api_key: Z.AI API key (Bearer token).

    Returns:
        Sorted list of model ID strings, or empty list on failure.
    """
    try:
        from shared.http.proxy import get_httpx_client

        client = get_httpx_client()
        resp = client.get(
            ZHIPUAI_MODELS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        model_ids = [m["id"] for m in data.get("data", []) if "id" in m]
        return sorted(model_ids)
    except Exception:
        return []


class ZhipuAIAPIKeyNotFoundError(Exception):
    """Zhipu AI API key not found."""

    def __init__(self):
        super().__init__(
            "Zhipu AI API key not found.\n"
            "Set ZHIPUAI_API_KEY environment variable or pass via config.\n"
            "Get your key at: https://open.bigmodel.cn/"
        )


class ZhipuAIConnectionError(Exception):
    """Failed to connect to Zhipu AI API."""

    def __init__(self, message: str = ""):
        detail = f": {message}" if message else ""
        super().__init__(
            f"Cannot connect to Zhipu AI API{detail}\n"
            f"Check your API key and network connection."
        )


class ZhipuAIRateLimitError(Exception):
    """Rate limit exceeded for Z.AI API.

    Classified as transient by the retry system so ``with_retry()``
    retries the call with exponential backoff.
    """

    def __init__(self, message: str = "", retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(message or "Zhipu AI rate limit exceeded. Please wait and try again.")


class ZhipuAIProvider(AnthropicProvider):
    """Zhipu AI provider using Anthropic-compatible API.

    Stateless provider: the session calls ``complete()`` for every API
    interaction. No conversation state is held inside the provider.

    This provider inherits from AnthropicProvider and overrides only
    what's necessary for Zhipu AI's API:
    - Custom base_url pointing to Z.AI's Anthropic-compatible endpoint
    - API key authentication via ZHIPUAI_API_KEY
    - Dynamic model discovery via the OpenAI-compatible ``GET /models``
      endpoint, with a static fallback for offline/unconfigured use
    - Caching disabled (may not be supported)
    - Extended thinking for GLM-5 and GLM-4.7 (native chain-of-thought)

    All completion handling, streaming, and converters are inherited from
    AnthropicProvider since Zhipu AI uses the same API format.
    """

    # Vendor identity overrides for error messages.  The base class's
    # ``_handle_api_error`` raises domain errors that read these via
    # ``self`` — so when ``ZhipuAIProvider._handle_api_error`` falls
    # through to ``super()`` (for usage-limit, overloaded, context-limit
    # etc.) the error's user-facing text names "Zhipu AI" rather than
    # the inherited "Anthropic API".
    _provider_display_name: str = "Zhipu AI"
    _provider_console_url: str = "https://open.bigmodel.cn/usercenter/apikeys"

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._base_url: str = DEFAULT_ZHIPUAI_BASE_URL
        self._context_length_override: Optional[int] = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "zhipuai"

    def _get_trace_prefix(self) -> str:
        """Get the trace prefix including agent context."""
        if self._agent_type == "main":
            return "zhipuai:main"
        elif self._agent_name:
            return f"zhipuai:subagent:{self._agent_name}"
        else:
            return f"zhipuai:subagent:{self._agent_id}"

    def _trace(self, msg: str) -> None:
        """Write trace message to provider trace log for debugging."""
        from shared.trace import provider_trace
        prefix = self._get_trace_prefix()
        provider_trace(prefix, msg)

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider.

        Args:
            config: Optional configuration.
                - api_key: Zhipu AI API key (overrides ZHIPUAI_API_KEY)
                - extra['base_url']: Override ZHIPUAI_BASE_URL
                - extra['context_length']: Override context length
                - extra['enable_thinking']: Enable extended thinking (default: False)
                - extra['thinking_budget']: Max thinking tokens (default: 10000)

        Raises:
            ZhipuAIAPIKeyNotFoundError: If no API key is found.
            ImportError: If anthropic package is not installed.
        """
        self._trace("[INIT] Starting initialization")

        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            ) from e

        if config is None:
            config = ProviderConfig()

        # Pull the workspace_path / config_root that the runtime injected
        # into ``config.extra``.  Threading them through the auth
        # resolver makes credential lookup independent of the
        # ``JAATO_CONFIG_ROOT`` env var, which is unreliable for
        # headless reactor-spawned sessions running in fresh threads
        # outside any active ``_in_workspace`` context.
        _ws_path = config.extra.get('workspace_path') if config.extra else None
        _config_root = config.extra.get('config_root') if config.extra else None

        # Resolve API key from config, environment, or stored credentials.
        # Track which source was used for the "Connected to" message.
        self._auth_info: str = ""
        if config.api_key:
            self._api_key = config.api_key
            self._auth_info = "API key (config)"
        elif resolve_api_key():
            self._api_key = resolve_api_key()
            self._auth_info = "API key (env ZHIPUAI_API_KEY)"
        elif get_stored_api_key(workspace_path=_ws_path, config_root=_config_root):
            self._api_key = get_stored_api_key(
                workspace_path=_ws_path, config_root=_config_root,
            )
            from .auth import get_credential_file_path
            cred_path = get_credential_file_path(
                workspace_path=_ws_path, config_root=_config_root,
            )
            self._auth_info = f"API key from {cred_path}" if cred_path else "API key (stored)"
        else:
            self._api_key = None

        if not self._api_key:
            self._trace("[INIT] No API key found")
            raise ZhipuAIAPIKeyNotFoundError()

        self._trace(f"[INIT] API key resolved (len={len(self._api_key)})")

        # Parse extra config — namespaced into the same layers as the
        # anthropic provider (server 0.6.24+):
        #
        #   plugin_configs.zhipuai:
        #     <top-level>           # auth / identity (api_key)
        #     api_params:           # Anthropic-compatible request body
        #                           # (temperature, top_p, top_k, max_tokens,
        #                           #  enable_thinking, thinking_budget)
        #     framework_overrides:  # rare escape hatches (base_url, context_length)
        #
        # Backward compatibility: every key is also read from the legacy
        # flat position with a one-time deprecation warning per key.
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
                    "Zhipuai profile uses legacy flat config key %r — "
                    "move under the appropriate nested layer "
                    "(api_params / framework_overrides) per the 0.6.24+ "
                    "namespacing.  Flat-key support will be removed in a "
                    "future release.",
                    key,
                )
                return config.extra[key]
            return default

        # Resolve base URL from config, environment, or stored credentials.
        # (framework_overrides — base URL is a deployment escape hatch,
        # not a per-request knob.)
        self._base_url = (
            _knob("base_url", layer=framework_overrides)
            or resolve_base_url()
        )
        # Check stored base_url only if using default (not overridden)
        if self._base_url == DEFAULT_ZHIPUAI_BASE_URL:
            stored_base_url = get_stored_base_url(
                workspace_path=_ws_path, config_root=_config_root,
            )
            if stored_base_url:
                self._base_url = stored_base_url

        # Ensure base URL doesn't have trailing slash
        self._base_url = self._base_url.rstrip("/")
        self._trace(f"[INIT] base_url={self._base_url}")

        # Context-window override (framework_overrides.context_length / env) via
        # the shared precedence helper.  GLM model windows are documented in
        # MODEL_CONTEXT_LIMITS (consulted in get_context_limit); the Z.AI
        # /models endpoint carries no context field (verified), so there is no
        # auto-detect tier.  The override, when set, wins over the table.
        self._context_length_override = resolve_context_window(
            detect_capacity=None,
            profile_value=_knob("context_length", layer=framework_overrides),
            env_value=resolve_context_length(),
        )
        if self._context_length_override:
            self._trace(f"[INIT] context_length_override={self._context_length_override}")

        # Extended thinking: configurable for GLM-4.7+ which has native
        # CoT reasoning (api_params layer — these translate to wire fields).
        self._enable_thinking = _knob(
            "enable_thinking", layer=api_params, default=resolve_enable_thinking(),
        )
        self._thinking_budget = _knob(
            "thinking_budget", layer=api_params, default=resolve_thinking_budget(),
        )

        # Sampling parameters (api_params layer).  ``None`` means "omit
        # from the request and let GLM apply its server-side default"
        # (Anthropic-compat → temperature=1.0).  These reach
        # ``messages.create()`` via the inherited ``complete()`` method
        # in AnthropicProvider, which reads ``self._temperature`` etc.
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

        # Zhipu AI doesn't use OAuth/PKCE - set to disabled
        self._use_pkce = False
        self._pkce_access_token = None
        self._oauth_token = None

        # Create the client
        self._trace("[INIT] Creating client")
        self._client = self._create_client()
        self._trace("[INIT] Initialization complete")

    def _create_client(self) -> Any:
        """Create Anthropic client pointing to Zhipu AI server.

        Uses the parent's _create_http_client() to configure proxy and SSL
        settings (corporate CA certificates, Kerberos auth, standard proxy
        env vars) so connections work behind corporate proxies.
        """
        import anthropic

        self._trace(f"[_create_client] Creating Anthropic client with base_url={self._base_url}")

        # Build custom httpx client for proxy/SSL if needed
        http_client = self._create_http_client()
        client_kwargs: Dict[str, Any] = {
            "base_url": self._base_url,
            "api_key": self._api_key,
        }
        if http_client:
            client_kwargs["http_client"] = http_client

        client = anthropic.Anthropic(**client_kwargs)
        self._trace("[_create_client] Client created successfully")
        return client

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify Zhipu AI API key is available.

        ``config`` is accepted for protocol compatibility but unused — Z.AI
        reads its credentials from the environment / stored auth file.

        This can be called BEFORE initialize() to check that credentials
        exist. Checks environment variable and stored credentials.

        When the stored credential file exists but cannot be loaded
        (corrupt JSON, permission error, missing ``api_key`` field), the
        failure reason is surfaced via ``on_message`` instead of being
        swallowed as a generic "No credentials found".  Without this,
        a broken auth file produces the same message as a missing one,
        hiding the real problem from the user.

        Args:
            allow_interactive: Ignored (no interactive auth for Zhipu AI).
            on_message: Optional callback for status messages.

        Returns:
            True if an API key is available.
        """
        self._trace("[AUTH] Verifying credentials")
        env_key = resolve_api_key()
        if env_key:
            self._trace("[AUTH] API key found in environment")
            if on_message:
                on_message("Found Zhipu AI API key (env ZHIPUAI_API_KEY)")
            return True

        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            self._trace("[AUTH] API key loaded from stored credentials")
            if on_message:
                on_message("Found Zhipu AI API key (stored credentials)")
            return True

        if load_error:
            # File exists but could not be parsed — surface the reason so
            # users can distinguish "never logged in" from "auth file is
            # broken / unreadable".
            self._trace(f"[AUTH] Stored credentials unusable: {load_error}")
            if on_message:
                on_message(
                    f"Zhipu AI credentials file found but could not be loaded: "
                    f"{load_error}"
                )
                on_message(
                    "Run 'zhipuai-auth key <your_api_key>' to re-authenticate, "
                    "or set ZHIPUAI_API_KEY."
                )
            return False

        self._trace("[AUTH] No credentials found")
        if on_message:
            on_message("No Zhipu AI credentials found")
        return False

    def connect(self, model_name: str, *, skip_model_test: bool = False) -> None:
        """Connect to a specific model.

        Args:
            model_name: Model name (e.g., 'glm-5', 'glm-4.7', 'glm-4.7-flash').
            skip_model_test: Accepted for protocol compatibility; this provider
                defers validation to the first API call.
        """
        # For Zhipu AI, we don't have a model listing API via the Anthropic endpoint,
        # so we just accept the model name and let the API validate it
        self._model_name = model_name
        context_limit = self.get_context_limit()
        self._trace(f"[CONNECT] model={model_name} context_limit={context_limit}")
        logger.info(f"Connected to Zhipu AI model: {model_name}")

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available GLM models.

        Attempts dynamic discovery via Z.AI's OpenAI-compatible
        ``GET /models`` endpoint.  Falls back to the static
        ``KNOWN_MODELS`` list when the API call fails (network
        errors, missing credentials, etc.).

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

        Resolves the API key from the provider instance, environment
        variables, or stored credentials.

        Returns:
            List of model ID strings, or an empty list on failure.
        """
        api_key = getattr(self, "_api_key", None)
        if not api_key:
            api_key = resolve_api_key() or get_stored_api_key()
        if not api_key:
            self._trace("[_fetch_remote_models] No API key available, skipping")
            return []

        self._trace(f"[_fetch_remote_models] GET {ZHIPUAI_MODELS_URL}")
        models = fetch_zhipuai_models(api_key)
        self._trace(f"[_fetch_remote_models] Got {len(models)} models")
        return models

    def modalities(self, model: Optional[str] = None):
        """INPUT modalities — GLM-4V / 4.5V accept images; other GLMs text-only.

        Overrides the inherited Anthropic ``modalities()`` (whose ``claude-*``
        prefix table never matches a GLM name, so it was inert and silently
        gated images off for the vision models).  Precedence: profile
        ``modalities`` knob → GLM table → text floor.
        """
        model = (model or self._model_name or "").lower()
        table = None
        for prefix, mods in GLM_INPUT_MODALITIES.items():
            if model.startswith(prefix):
                table = mods
                break
        resolved = resolve_modalities(
            profile_value=self._modalities_knob,
            table_value=table,
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    def get_context_limit(self) -> int:
        """Get context window size.

        Resolution precedence (no hardcoded fallback, per project rule):
        1. ``context_length_override`` (framework_overrides knob / env) — wins.
        2. ``MODEL_CONTEXT_LIMITS`` exact match.
        3. ``MODEL_CONTEXT_LIMITS`` LONGEST-prefix match — so a dated variant
           (e.g. ``glm-4.7-20250601``) resolves to its family (``glm-4.7``)
           rather than a shorter, wrong prefix (``glm-4``).
        4. else raise — unknown model with no override is a configuration error.

        Raises:
            ValueError: when neither the override nor the table yields a value.
        """
        if self._context_length_override:
            return self._context_length_override
        model = self._model_name
        if model:
            if model in MODEL_CONTEXT_LIMITS:
                return MODEL_CONTEXT_LIMITS[model]
            # Longest-prefix match (GLM model names nest: glm-4 is a prefix of
            # glm-4.5/4.7 — exact match above handles those; here we take the
            # most specific prefix so dated variants land on the right family).
            prefixes = [p for p in MODEL_CONTEXT_LIMITS if model.startswith(p)]
            if prefixes:
                return MODEL_CONTEXT_LIMITS[max(prefixes, key=len)]
        raise ValueError(
            f"ZhipuAI provider: no known context window for model {model!r}, and "
            f"no override is set.  Add the model to MODEL_CONTEXT_LIMITS, or set "
            f"framework_overrides.context_length / ZHIPUAI_CONTEXT_LENGTH.  No "
            f"hardcoded fallback exists per the project's no-fallback rule."
        )

    def _is_thinking_capable(self) -> bool:
        """Check if the current model supports extended thinking.

        GLM-5 and GLM-4.7 (non-flash) have native chain-of-thought
        reasoning capability.  Flash and other GLM variants do not.
        """
        if not self._model_name:
            return False
        name_lower = self._model_name.lower()
        # GLM-5 always supports thinking
        if name_lower.startswith("glm-5"):
            return True
        # GLM-4.7 supports thinking, but flash variants do not
        return name_lower.startswith("glm-4.7") and "flash" not in name_lower

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors with Zhipu AI-specific interpretation.

        Overrides parent to provide more helpful error messages for
        Zhipu AI-specific issues.
        """
        error_str = str(error).lower()
        self._trace(f"[API_ERROR] {type(error).__name__}: {error}")

        # Check for authentication errors
        if "401" in error_str or "unauthorized" in error_str or "invalid api key" in error_str:
            self._trace("[API_ERROR] Authentication failure (401/unauthorized)")
            raise ZhipuAIConnectionError(
                "Invalid API key. Check your ZHIPUAI_API_KEY.\n"
                f"Original error: {error}"
            ) from error

        # Check for rate limiting
        if "429" in error_str or "rate limit" in error_str:
            self._trace("[API_ERROR] Rate limit exceeded (429)")
            raise ZhipuAIRateLimitError(
                f"Zhipu AI rate limit exceeded. Please wait and try again.\n"
                f"Original error: {error}"
            ) from error

        # Check for model not found
        if "404" in error_str and "model" in error_str:
            self._trace(f"[API_ERROR] Model not found: {self._model_name}")
            raise RuntimeError(
                f"Model '{self._model_name}' not found on Zhipu AI.\n"
                f"Available models: {', '.join(KNOWN_MODELS)}\n"
                f"Original error: {error}"
            ) from error

        # For other errors, use parent's handling
        super()._handle_api_error(error)

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry purposes.

        ``ZhipuAIRateLimitError`` is transient and should be retried
        with exponential backoff.
        """
        if isinstance(exc, ZhipuAIRateLimitError):
            return {"transient": True, "rate_limit": True, "infra": False}
        return None  # Fall back to global classification

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract retry-after hint from an exception."""
        if isinstance(exc, ZhipuAIRateLimitError) and exc.retry_after:
            return float(exc.retry_after)
        return None

    @staticmethod
    def login(
        on_message: Optional[Callable[[str], None]] = None,
        on_input: Optional[Callable[[str], str]] = None,
    ) -> bool:
        """Interactive login for Zhipu AI.

        Prompts user for API key and validates it.

        Args:
            on_message: Optional callback for status messages.
            on_input: Optional callback for user input. If None, uses builtin input().

        Returns:
            True if login successful, False otherwise.
        """
        result = login_interactive(on_message=on_message, on_input=on_input)
        return result is not None

    @staticmethod
    def logout(on_message: Optional[Callable[[str], None]] = None) -> None:
        """Clear stored credentials.

        Args:
            on_message: Optional callback for status messages.
        """
        logout(on_message=on_message)

    @staticmethod
    def auth_status(on_message: Optional[Callable[[str], None]] = None) -> bool:
        """Check authentication status.

        Args:
            on_message: Optional callback for status messages.

        Returns:
            True if valid credentials are stored.
        """
        return auth_status(on_message=on_message)


def create_provider() -> ZhipuAIProvider:
    """Factory function for plugin discovery."""
    return ZhipuAIProvider()
