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
    response = provider.send_message("Hello!")

Environment variables:
    ZHIPUAI_API_KEY: Zhipu AI API key
    ZHIPUAI_BASE_URL: API base URL (default: https://api.z.ai/api/anthropic)
    ZHIPUAI_MODEL: Default model to use
    ZHIPUAI_CONTEXT_LENGTH: Override context length for models
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from ..anthropic.provider import AnthropicProvider
from ..base import ProviderConfig
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
)


logger = logging.getLogger(__name__)


# Known GLM models with their context window sizes in tokens.
# Used as metadata fallback when the dynamic /models endpoint is unavailable.
# Source: models.dev, Roo Code, lm-deluge, ekai-gateway, moai-adk, Z.AI docs.
MODEL_CONTEXT_LIMITS = {
    # GLM-5 — 200K context, 128K output
    "glm-5": 204800,
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

# Fallback for unknown models
DEFAULT_CONTEXT_LIMIT = 131072

KNOWN_MODELS = sorted(MODEL_CONTEXT_LIMITS.keys())


# GLM models that support extended thinking (chain-of-thought reasoning).
# GLM-5 and GLM-4.7 (non-flash) have native chain-of-thought capability.
THINKING_CAPABLE_MODELS = [
    "glm-5",
    "glm-4.7",
]

# ── OpenAI-compatible models endpoint for dynamic discovery ───────────
# Z.AI exposes GET /models on the OpenAI-compatible API surface.  The
# Anthropic-compatible endpoint we use for chat does NOT have this, so
# we derive the OpenAI base URL from the configured Anthropic base URL.
_ANTHROPIC_TO_OPENAI_PATH = {
    "/api/anthropic": "/api/paas/v4",
    "/api/coding/anthropic": "/api/coding/paas/v4",
}


def _openai_models_url(anthropic_base_url: str) -> str:
    """Derive the OpenAI-compatible ``/models`` URL from the Anthropic base.

    Handles both the pay-per-token (``/api/paas/v4``) and coding-plan
    (``/api/coding/paas/v4``) variants.

    Args:
        anthropic_base_url: The Anthropic-compat base URL
            (e.g. ``https://api.z.ai/api/anthropic``).

    Returns:
        Full URL for the ``GET /models`` endpoint.
    """
    base = anthropic_base_url.rstrip("/")
    for suffix, replacement in _ANTHROPIC_TO_OPENAI_PATH.items():
        if base.endswith(suffix):
            return base[: -len(suffix)] + replacement + "/models"
    # Best-effort: assume sibling /paas/v4 next to whatever path is set
    return base.rsplit("/", 1)[0] + "/paas/v4/models"


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


class ZhipuAIProvider(AnthropicProvider):
    """Zhipu AI provider using Anthropic-compatible API.

    This provider inherits from AnthropicProvider and overrides only
    what's necessary for Zhipu AI's API:
    - Custom base_url pointing to Z.AI's Anthropic-compatible endpoint
    - API key authentication via ZHIPUAI_API_KEY
    - Dynamic model discovery via the OpenAI-compatible ``GET /models``
      endpoint, with a static fallback for offline/unconfigured use
    - Caching disabled (may not be supported)
    - Extended thinking for GLM-5 and GLM-4.7 (native chain-of-thought)

    All message handling, streaming, and converters are inherited from
    AnthropicProvider since Zhipu AI uses the same API format.
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._base_url: str = DEFAULT_ZHIPUAI_BASE_URL
        self._context_length_override: Optional[int] = None

        # Caching may not be supported by Zhipu AI's Anthropic-compatible API
        self._enable_caching = False

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

        # Resolve API key from config, environment, or stored credentials
        self._api_key = (
            config.api_key
            or resolve_api_key()
            or get_stored_api_key()
        )
        if not self._api_key:
            self._trace("[INIT] No API key found")
            raise ZhipuAIAPIKeyNotFoundError()

        self._trace(f"[INIT] API key resolved (len={len(self._api_key)})")

        # Resolve base URL from config, environment, or stored credentials
        self._base_url = (
            config.extra.get("base_url")
            or resolve_base_url()
        )
        # Check stored base_url only if using default (not overridden)
        if self._base_url == DEFAULT_ZHIPUAI_BASE_URL:
            stored_base_url = get_stored_base_url()
            if stored_base_url:
                self._base_url = stored_base_url

        # Ensure base URL doesn't have trailing slash
        self._base_url = self._base_url.rstrip("/")
        self._trace(f"[INIT] base_url={self._base_url}")

        # Optional context length override
        self._context_length_override = (
            config.extra.get("context_length") or resolve_context_length()
        )
        if self._context_length_override:
            self._trace(f"[INIT] context_length_override={self._context_length_override}")

        # Caching may not be supported by Zhipu AI's Anthropic-compatible API
        self._enable_caching = False

        # Extended thinking: configurable for GLM-4.7 which has native CoT reasoning
        self._enable_thinking = config.extra.get(
            "enable_thinking", resolve_enable_thinking()
        )
        self._thinking_budget = config.extra.get(
            "thinking_budget", resolve_thinking_budget()
        )

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
        on_message=None
    ) -> bool:
        """Verify Zhipu AI API key is available.

        This can be called BEFORE initialize() to check that credentials
        exist. Checks environment variable and stored credentials.

        Args:
            allow_interactive: Ignored (no interactive auth for Zhipu AI).
            on_message: Optional callback for status messages.

        Returns:
            True if an API key is available.
        """
        self._trace("[AUTH] Verifying credentials")
        api_key = resolve_api_key() or get_stored_api_key()
        if api_key:
            self._trace("[AUTH] API key found")
            if on_message:
                on_message("Found Zhipu AI API key")
            return True

        self._trace("[AUTH] No credentials found")
        if on_message:
            on_message("No Zhipu AI credentials found")
        return False

    def connect(self, model_name: str) -> None:
        """Connect to a specific model.

        Args:
            model_name: Model name (e.g., 'glm-5', 'glm-4.7', 'glm-4.7-flash').
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
        """Fetch model list from Z.AI's OpenAI-compatible ``GET /models`` endpoint.

        Derives the correct URL from the configured Anthropic base URL so
        that both the pay-per-token and coding-plan endpoints are handled.
        Uses the project's corporate-ready httpx client (``shared.http``)
        for proxy, Kerberos, and custom CA-cert support.

        Returns:
            List of model ID strings, or an empty list on failure.
        """
        api_key = getattr(self, "_api_key", None)
        if not api_key:
            # Try environment / stored credentials so listing works
            # even on an uninitialized provider instance.
            api_key = resolve_api_key() or get_stored_api_key()
        if not api_key:
            self._trace("[_fetch_remote_models] No API key available, skipping")
            return []

        base_url = getattr(self, "_base_url", DEFAULT_ZHIPUAI_BASE_URL)
        url = _openai_models_url(base_url)
        self._trace(f"[_fetch_remote_models] GET {url}")

        try:
            from shared.http.proxy import get_httpx_client

            client = get_httpx_client()
            resp = client.get(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            model_ids = [m["id"] for m in data.get("data", []) if "id" in m]
            self._trace(f"[_fetch_remote_models] Got {len(model_ids)} models: {model_ids}")
            return model_ids
        except Exception as exc:
            self._trace(f"[_fetch_remote_models] Failed: {exc}")
            logger.debug("Failed to fetch Z.AI model list: %s", exc)
            return []

    def get_context_limit(self) -> int:
        """Get context window size.

        Returns context_length_override if set, otherwise looks up the
        per-model limit from MODEL_CONTEXT_LIMITS.
        """
        if self._context_length_override:
            return self._context_length_override
        return MODEL_CONTEXT_LIMITS.get(self._model_name, DEFAULT_CONTEXT_LIMIT)

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
            raise RuntimeError(
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
