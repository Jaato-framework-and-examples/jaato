"""Ollama provider implementation.

This provider enables access to local models via Ollama's Anthropic-compatible
Messages API (available in Ollama v0.14.0+).

Ollama acts as a local server that can run various open-source models
(Llama, Qwen, Mistral, etc.) and exposes them via an Anthropic-compatible API.

Usage:
    provider = OllamaProvider()
    provider.initialize()  # No API key needed
    provider.connect('qwen3:32b')
    response = provider.send_message("Hello!")

Environment variables:
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL: Default model to use
    OLLAMA_CONTEXT_LENGTH: Override context length for models
"""

import logging
from typing import Any, Dict, List, Optional

import requests

from ..anthropic.provider import AnthropicProvider
from ..base import ProviderConfig
from .env import (
    DEFAULT_OLLAMA_HOST,
    resolve_context_length,
    resolve_host,
    resolve_model,
)


logger = logging.getLogger(__name__)


# Default context limit for Ollama models
# Most models support at least 8K, many support 32K+
DEFAULT_CONTEXT_LIMIT = 32768


class OllamaConnectionError(Exception):
    """Failed to connect to Ollama server."""

    def __init__(self, host: str, message: str = ""):
        self.host = host
        detail = f": {message}" if message else ""
        super().__init__(
            f"Cannot connect to Ollama server at {host}{detail}\n"
            f"Make sure Ollama is running: ollama serve"
        )


class OllamaModelNotFoundError(Exception):
    """Requested model not found in Ollama."""

    def __init__(self, model: str, available: List[str] = None):
        self.model = model
        self.available = available or []
        if self.available:
            avail_str = ", ".join(self.available[:5])
            if len(self.available) > 5:
                avail_str += f", ... ({len(self.available)} total)"
            super().__init__(
                f"Model '{model}' not found in Ollama.\n"
                f"Available models: {avail_str}\n"
                f"Pull it with: ollama pull {model}"
            )
        else:
            super().__init__(
                f"Model '{model}' not found in Ollama.\n"
                f"Pull it with: ollama pull {model}"
            )


class OllamaProvider(AnthropicProvider):
    """Ollama provider using Anthropic-compatible API.

    This provider inherits from AnthropicProvider and overrides only
    what's necessary for Ollama's local server:
    - No API key required
    - Custom base_url pointing to Ollama
    - Model listing via Ollama's native API
    - Caching and thinking disabled (not supported)

    All message handling, streaming, and converters are inherited from
    AnthropicProvider since Ollama uses the same API format.
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._host: str = DEFAULT_OLLAMA_HOST
        self._context_length_override: Optional[int] = None

        # Ollama doesn't support these features
        self._enable_caching = False
        self._enable_thinking = False

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "ollama"

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider.

        No API key is required - Ollama runs locally.

        Args:
            config: Optional configuration.
                - extra['host']: Override OLLAMA_HOST
                - extra['context_length']: Override context length
        """
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            ) from e

        if config is None:
            config = ProviderConfig()

        # Resolve host from config or environment
        self._host = config.extra.get("host") or resolve_host()

        # Ensure host doesn't have trailing slash
        self._host = self._host.rstrip("/")

        # Optional context length override
        self._context_length_override = (
            config.extra.get("context_length") or resolve_context_length()
        )

        # Ollama doesn't support caching or thinking - force disable
        self._enable_caching = False
        self._enable_thinking = False

        # Ollama doesn't use OAuth/PKCE - set to disabled
        self._use_pkce = False
        self._pkce_access_token = None
        self._oauth_token = None
        self._api_key = "ollama"  # Dummy value, Ollama ignores it

        # Create the client
        self._client = self._create_client()

        # Verify Ollama is running
        self._verify_connectivity()

    def _create_client(self) -> Any:
        """Create Anthropic client pointing to Ollama server."""
        import anthropic

        # Ollama requires the /v1 suffix for Anthropic API compatibility
        base_url = f"{self._host}/v1"

        return anthropic.Anthropic(
            base_url=base_url,
            # Ollama ignores the API key but the SDK requires one
            api_key="ollama",
        )

    def _verify_connectivity(self) -> None:
        """Verify Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self._host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError(self._host)
        except requests.exceptions.Timeout:
            raise OllamaConnectionError(self._host, "Connection timed out")
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(self._host, str(e))

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None
    ) -> bool:
        """Verify Ollama server is accessible.

        For Ollama, this checks if the server is running rather than
        checking API credentials.

        Args:
            allow_interactive: Ignored (no interactive auth for Ollama).
            on_message: Optional callback for status messages.

        Returns:
            True if Ollama server is accessible.
        """
        try:
            response = requests.get(f"{self._host}/api/tags", timeout=5)
            response.raise_for_status()
            if on_message:
                on_message(f"Connected to Ollama at {self._host}")
            return True
        except requests.exceptions.RequestException:
            if on_message:
                on_message(f"Cannot connect to Ollama at {self._host}")
            return False

    def connect(self, model_name: str) -> None:
        """Connect to a specific model.

        Args:
            model_name: Model name (e.g., 'qwen3:32b', 'llama3.3:70b').

        Raises:
            OllamaModelNotFoundError: Model not available in Ollama.
        """
        # Verify model exists in Ollama
        available = self._get_local_models()
        # Check both exact match and with default tag
        if model_name not in available:
            # Try with :latest tag
            if f"{model_name}:latest" not in available:
                raise OllamaModelNotFoundError(model_name, available)

        self._model_name = model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models available in Ollama.

        Args:
            prefix: Optional filter prefix.

        Returns:
            List of model names.
        """
        models = self._get_local_models()

        if prefix:
            models = [m for m in models if m.startswith(prefix)]

        return sorted(models)

    def _get_local_models(self) -> List[str]:
        """Get list of models available in Ollama."""
        try:
            response = requests.get(f"{self._host}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

    def get_context_limit(self) -> int:
        """Get context window size.

        Returns context_length_override if set, otherwise a default.
        Ollama models vary widely in context size.
        """
        if self._context_length_override:
            return self._context_length_override
        return DEFAULT_CONTEXT_LIMIT

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors with Ollama-specific interpretation.

        Overrides parent to avoid misinterpreting Ollama errors.
        For example, a 404 from Ollama likely means the Anthropic API
        endpoint isn't available, not that the model wasn't found.
        """
        error_str = str(error).lower()

        # Check for Ollama-specific memory errors
        if "system memory" in error_str or "not enough memory" in error_str:
            raise RuntimeError(
                f"Ollama: Not enough memory to load model '{self._model_name}'. "
                f"Try a smaller model or increase available memory.\n"
                f"Original error: {error}"
            ) from error

        # Check for 404 - likely means Anthropic API not supported
        if "404" in error_str or "page not found" in error_str:
            raise RuntimeError(
                f"Ollama returned 404. This may indicate:\n"
                f"  1. Ollama version < 0.14.0 (Anthropic API requires 0.14.0+)\n"
                f"  2. The Anthropic API endpoint is not enabled\n"
                f"Check your Ollama version: curl {self._host}/api/version\n"
                f"Original error: {error}"
            ) from error

        # For other errors, use parent's handling
        super()._handle_api_error(error)

    @staticmethod
    def login(on_message=None) -> None:
        """Not applicable for Ollama (no authentication required)."""
        if on_message:
            on_message("Ollama doesn't require authentication - it runs locally.")


def create_provider() -> OllamaProvider:
    """Factory function for plugin discovery."""
    return OllamaProvider()
