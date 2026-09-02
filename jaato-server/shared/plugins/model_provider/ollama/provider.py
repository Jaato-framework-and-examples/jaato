"""Ollama provider implementation.

This provider enables access to local models via Ollama's Anthropic-compatible
Messages API (available in Ollama v0.14.0+).

Ollama acts as a local server that can run various open-source models
(Llama, Qwen, Mistral, etc.) and exposes them via an Anthropic-compatible API.

Usage:
    provider = OllamaProvider()
    provider.initialize()  # No API key needed
    provider.connect('qwen3:32b')
    response = provider.complete(messages=[...])

Environment variables:
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL: Default model to use
    OLLAMA_CONTEXT_LENGTH: Override context length for models
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..anthropic.provider import AnthropicProvider
from ..base import (
    ProviderConfig,
    resolve_context_window,
    resolve_modalities,
    MODALITY_TEXT,
)
from .env import (
    DEFAULT_OLLAMA_HOST,
    resolve_context_length,
    resolve_host,
    resolve_model,
)


logger = logging.getLogger(__name__)



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
    - Thinking disabled (not supported by local models)
    - No cache plugin matches 'ollama', so no cache annotations are applied

    The session calls ``complete()`` for all API interactions; stateless
    completion, streaming, and converters are inherited from
    AnthropicProvider since Ollama uses the same API format.
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._host: str = DEFAULT_OLLAMA_HOST
        self._context_length_override: Optional[int] = None
        self._context_length_knob: Optional[int] = None

        # Ollama doesn't support thinking.
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

        # Context window: resolved with tier-1 auto-detect PRIMARY at
        # connect() (the model is known then — Ollama's context length is
        # per-model).  Stash the manual knob and seed a provisional from
        # knob -> env so get_context_limit() is usable before connect();
        # connect() re-resolves with detect (POST /api/show) winning.
        self._context_length_knob = config.extra.get("context_length")
        self._context_length_override = resolve_context_window(
            profile_value=self._context_length_knob,
            env_value=resolve_context_length(),
        )

        # Sampling parameters (api_params layer).  They live NESTED at
        # plugin_configs.ollama.api_params.{temperature,top_p,top_k,max_tokens}
        # → config.extra["api_params"][...].  ``None`` means "omit from the
        # request and let Ollama apply its server-side default"; a profile
        # wanting determinism sets ``api_params.temperature: 0.0`` (which is
        # falsy, hence the ``is not None`` guards).  These reach
        # ``messages.create()`` via the inherited AnthropicProvider.complete(),
        # which emits self._temperature / _top_p / _top_k / _max_tokens_override.
        api_params = config.extra.get("api_params") or {}
        if not isinstance(api_params, dict):
            raise TypeError(
                "Ollama 'api_params' config must be a dict of Anthropic "
                f"Messages API fields, got {type(api_params).__name__}"
            )
        temp_extra = api_params.get("temperature")
        if temp_extra is not None:
            self._temperature = float(temp_extra)
        top_p_extra = api_params.get("top_p")
        if top_p_extra is not None:
            self._top_p = float(top_p_extra)
        top_k_extra = api_params.get("top_k")
        if top_k_extra is not None:
            self._top_k = int(top_k_extra)
        max_tokens_extra = api_params.get("max_tokens")
        if max_tokens_extra is not None:
            self._max_tokens_override = int(max_tokens_extra)

        # Ollama doesn't support thinking.
        self._enable_thinking = False

        # Ollama doesn't use OAuth/PKCE - set to disabled
        self._use_pkce = False
        self._pkce_access_token = None
        self._oauth_token = None
        self._api_key = "ollama"  # Dummy value, Ollama ignores it
        self._auth_info = f"local ({self._host})"

        # Create the client
        self._client = self._create_client()

        # Verify Ollama is running
        self._verify_connectivity()

    def _create_client(self) -> Any:
        """Create Anthropic client pointing to Ollama server."""
        import anthropic

        # Anthropic SDK appends /v1/messages to base_url, so use host directly
        # Ollama serves Anthropic API at /v1/messages
        return anthropic.Anthropic(
            base_url=self._host,
            # Ollama ignores the API key but the SDK requires one
            api_key="ollama",
        )

    def _verify_connectivity(self) -> None:
        """Verify Ollama server is running and accessible."""
        try:
            response = httpx.get(f"{self._host}/api/tags", timeout=5)
            response.raise_for_status()
        except httpx.ConnectError:
            raise OllamaConnectionError(self._host)
        except httpx.TimeoutException:
            raise OllamaConnectionError(self._host, "Connection timed out")
        except httpx.HTTPError as e:
            raise OllamaConnectionError(self._host, str(e))

    def _verify_model_responds(self) -> None:
        """Verify the model can actually respond.

        Sends a minimal test message to catch issues like:
        - Insufficient memory to load model
        - Model file corruption
        - Other runtime errors
        """
        try:
            # Send minimal request to verify model loads and responds
            self._client.messages.create(
                model=self._model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
        except Exception as e:
            # Use our error handler to provide helpful messages
            self._handle_api_error(e)

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
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
            response = httpx.get(f"{self._host}/api/tags", timeout=5)
            response.raise_for_status()
            if on_message:
                on_message(f"Connected to Ollama at {self._host}")
            return True
        except httpx.HTTPError:
            if on_message:
                on_message(f"Cannot connect to Ollama at {self._host}")
            return False

    def connect(self, model_name: str, *, skip_model_test: bool = False) -> None:
        """Connect to a specific model.

        Args:
            model_name: Model name (e.g., 'qwen3:32b', 'llama3.3:70b').
            skip_model_test: If True, skip the network call to verify the model
                responds.  The model will be validated on the first real
                message instead.

        Raises:
            OllamaModelNotFoundError: Model not available in Ollama.
            RuntimeError: Model cannot be loaded (memory, etc.).
        """
        # Verify model exists in Ollama (local check, not a network call)
        available = self._get_local_models()
        # Check both exact match and with default tag
        if model_name not in available:
            # Try with :latest tag
            if f"{model_name}:latest" not in available:
                raise OllamaModelNotFoundError(model_name, available)

        self._model_name = model_name

        # Tier-1 context-window auto-detect now that the model is known
        # (detect PRIMARY -> manual knob -> env).  POST /api/show reports
        # the model's real context length; a stale per-profile value can no
        # longer silently under-declare it.
        self._context_length_override = resolve_context_window(
            detect_capacity=self._detect_context_capacity,
            profile_value=self._context_length_knob,
            env_value=resolve_context_length(),
        )
        if not self._context_length_override:
            raise ValueError(
                "Ollama provider: context_length could not be resolved.  "
                "POST /api/show did not report the model's context length "
                "(older Ollama, or unreachable), and no manual override is "
                "set.  Set plugin_configs.ollama.context_length in the "
                "profile, or OLLAMA_CONTEXT_LENGTH in the environment.  No "
                "hardcoded fallback exists per the project's no-fallback rule."
            )

        if not skip_model_test:
            # Verify model can actually respond (catches memory issues, etc.)
            self._verify_model_responds()

    def modalities(self, model: Optional[str] = None):
        """INPUT modalities, detected from ``POST /api/show`` ``capabilities``.

        Ollama reports ``"vision"`` in a model's ``capabilities`` list for
        vision models (llava, llama3.2-vision, qwen2.5-vl, …).  This overrides
        the inherited Anthropic ``claude-*`` prefix table, which never matches
        an Ollama model name (so the inherited override was inert — images were
        silently gated off even for vision models).  Precedence: live detect →
        profile ``modalities`` knob → text floor.
        """
        model = model or self._model_name
        resolved = resolve_modalities(
            detect=lambda: self._detect_modalities(model),
            profile_value=self._modalities_knob,
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    def _detect_modalities(self, model: Optional[str]):
        """Read ``capabilities`` from ``POST /api/show`` (cached per model).

        Returns ``None`` on any HTTP failure so resolution degrades to the knob
        / text floor (transient failures are not cached).
        """
        if not model:
            return None
        cache = self.__dict__.setdefault("_modality_cache", {})
        if model in cache:
            return cache[model]
        try:
            response = httpx.post(
                f"{self._host}/api/show",
                json={"model": model},
                timeout=10,
            )
            response.raise_for_status()
            caps = response.json().get("capabilities") or []
        except httpx.HTTPError:
            return None
        mods = {"text"}
        if "vision" in caps:
            mods.add("image")
        cache[model] = mods
        return mods

    def _detect_context_capacity(self) -> Optional[int]:
        """Tier-1 context-window auto-detection hook for Ollama.

        Reads the model's context length from ``POST /api/show``, whose
        ``model_info`` dict carries an architecture-prefixed
        ``<arch>.context_length`` entry (e.g. ``qwen3.context_length``).
        Returns ``None`` when the server is unreachable or the field is
        absent, so resolution degrades to the manual knob / env var.
        """
        if not self._model_name:
            return None
        try:
            response = httpx.post(
                f"{self._host}/api/show",
                json={"model": self._model_name},
                timeout=10,
            )
            response.raise_for_status()
            info = response.json().get("model_info") or {}
        except httpx.HTTPError:
            return None
        for key, val in info.items():
            if key.endswith(".context_length") and val:
                return int(val)
        return None

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
            response = httpx.get(f"{self._host}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except httpx.HTTPError as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

    def get_context_limit(self) -> int:
        """Get context window size.

        Resolved at connect() via detect (POST /api/show) -> manual knob
        -> env (see resolve_context_window).  Returns 0 ("unknown") only
        before connect(); connect() raises if nothing resolves, so a
        connected provider always reports a real window (no hardcoded
        default).
        """
        return self._context_length_override or 0

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
