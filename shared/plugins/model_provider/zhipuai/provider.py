"""Zhipu AI (Z.AI) provider implementation.

This provider enables access to Zhipu AI's GLM models via the Anthropic-compatible
API endpoint, primarily targeting GLM Coding Plan subscribers.

Zhipu AI offers the GLM family of models including:
- GLM-4.7: Latest model with native chain-of-thought reasoning
- GLM-4.7-Flash: Fast inference variant
- GLM-4: General purpose model
- GLM-4V: Vision-enabled multimodal model

Usage:
    provider = ZhipuAIProvider()
    provider.initialize(ProviderConfig(api_key="your-key"))
    provider.connect('glm-4.7')
    response = provider.send_message("Hello!")

Environment variables:
    ZHIPUAI_API_KEY: Zhipu AI API key
    ZHIPUAI_BASE_URL: API base URL (default: https://api.z.ai/api/anthropic/v1)
    ZHIPUAI_MODEL: Default model to use
    ZHIPUAI_CONTEXT_LENGTH: Override context length for models
"""

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


# Default context limit for Zhipu AI models
# GLM-4.7 supports 128K context
DEFAULT_CONTEXT_LIMIT = 131072


# Known GLM models available via the Anthropic-compatible API
KNOWN_MODELS = [
    "glm-4.7",
    "glm-4.7-flash",
    "glm-4",
    "glm-4v",
    "glm-4-assistant",
]


# GLM models that support extended thinking (chain-of-thought reasoning)
# GLM-4.7 has native chain-of-thought reasoning capability
THINKING_CAPABLE_MODELS = [
    "glm-4.7",
]


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
    - Model listing for GLM models
    - Caching disabled (may not be supported)
    - Extended thinking for GLM-4.7 (native chain-of-thought reasoning)

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
            raise ZhipuAIAPIKeyNotFoundError()

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

        # Optional context length override
        self._context_length_override = (
            config.extra.get("context_length") or resolve_context_length()
        )

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
        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Create Anthropic client pointing to Zhipu AI server."""
        import anthropic

        return anthropic.Anthropic(
            base_url=self._base_url,
            api_key=self._api_key,
        )

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None
    ) -> bool:
        """Verify Zhipu AI API key is valid.

        Sends a minimal request to verify the API key works.

        Args:
            allow_interactive: Ignored (no interactive auth for Zhipu AI).
            on_message: Optional callback for status messages.

        Returns:
            True if API key is valid.
        """
        try:
            # Send a minimal request to verify auth
            self._client.messages.create(
                model=DEFAULT_ZHIPUAI_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            if on_message:
                on_message(f"Connected to Zhipu AI at {self._base_url}")
            return True
        except Exception as e:
            if on_message:
                on_message(f"Cannot connect to Zhipu AI: {e}")
            return False

    def connect(self, model_name: str) -> None:
        """Connect to a specific model.

        Args:
            model_name: Model name (e.g., 'glm-4.7', 'glm-4.7-flash').
        """
        # For Zhipu AI, we don't have a model listing API via the Anthropic endpoint,
        # so we just accept the model name and let the API validate it
        self._model_name = model_name
        logger.info(f"Connected to Zhipu AI model: {model_name}")

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available GLM models.

        Returns known GLM models. Since the Anthropic-compatible API
        doesn't provide a model listing endpoint, we return a static list.

        Args:
            prefix: Optional filter prefix.

        Returns:
            List of model names.
        """
        models = KNOWN_MODELS.copy()

        if prefix:
            models = [m for m in models if m.startswith(prefix)]

        return sorted(models)

    def get_context_limit(self) -> int:
        """Get context window size.

        Returns context_length_override if set, otherwise the default.
        GLM-4.7 supports 128K context.
        """
        if self._context_length_override:
            return self._context_length_override
        return DEFAULT_CONTEXT_LIMIT

    def _is_thinking_capable(self) -> bool:
        """Check if the current model supports extended thinking.

        GLM-4.7 has native chain-of-thought reasoning capability.
        Flash and other GLM variants do not support it.
        """
        if not self._model_name:
            return False
        name_lower = self._model_name.lower()
        # GLM-4.7 supports thinking, but flash variants do not
        return name_lower.startswith("glm-4.7") and "flash" not in name_lower

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors with Zhipu AI-specific interpretation.

        Overrides parent to provide more helpful error messages for
        Zhipu AI-specific issues.
        """
        error_str = str(error).lower()

        # Check for authentication errors
        if "401" in error_str or "unauthorized" in error_str or "invalid api key" in error_str:
            raise ZhipuAIConnectionError(
                "Invalid API key. Check your ZHIPUAI_API_KEY.\n"
                f"Original error: {error}"
            ) from error

        # Check for rate limiting
        if "429" in error_str or "rate limit" in error_str:
            raise RuntimeError(
                f"Zhipu AI rate limit exceeded. Please wait and try again.\n"
                f"Original error: {error}"
            ) from error

        # Check for model not found
        if "404" in error_str and "model" in error_str:
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
