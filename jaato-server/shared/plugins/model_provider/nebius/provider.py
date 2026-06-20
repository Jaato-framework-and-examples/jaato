"""Nebius Token Factory model provider implementation.

Nebius exposes an OpenAI-compatible chat completions API, so this provider is a
thin subclass of :class:`OpenAICompatProvider` (the shared OpenAI-compat
machinery — streaming loop, completion skeleton incl. ``api_params`` /
``extra_body`` / ``tool_choice`` forwarding, error mapping, capability
boilerplate).

Nebius-specific concerns live here: identity, the error-class parameterization,
credential resolution, and the **catalog bootstrap** — context window and input
modalities are auto-detected from the ``GET /v1/models`` catalog (RichModel
entries carry ``context_length`` + ``architecture.modality``) at ``connect()``
once the active model is known, with profile-knob / env override fallbacks and
a fail-loud "not configured" error (no hardcoded fallback).

Authentication (API key, Bearer token):
- JAATO_NEBIUS_API_KEY (jaato namespace) or NEBIUS_API_KEY (vendor's own
  documented variable), or stored credentials via nebius-auth.

Environment variables:
    JAATO_NEBIUS_API_KEY / NEBIUS_API_KEY: API key
    JAATO_NEBIUS_BASE_URL: Endpoint (default: https://api.tokenfactory.nebius.com/v1)
    JAATO_NEBIUS_MODEL: Default model name
    JAATO_NEBIUS_CONTEXT_LENGTH: Override the catalog-detected context window
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .._openai_compat.base import OpenAICompatProvider
from ..base import (
    MODALITY_TEXT,
    ProviderConfig,
    resolve_context_window,
    resolve_modalities,
)
from .env import (
    DEFAULT_BASE_URL,
    ENV_NEBIUS_CONTEXT_LENGTH,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    is_self_hosted,
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

# Models known to expose reasoning/thinking content via ``reasoning_content``.
REASONING_CAPABLE_MODELS = [
    "deepseek/deepseek-r1",
    "deepseek-r1",
]


class NebiusProvider(OpenAICompatProvider):
    """Nebius Token Factory model provider.

    Access to the Nebius Token Factory catalog via the OpenAI-compatible
    serverless inference API.  Context window and input modalities are
    bootstrapped from ``GET /v1/models`` at connect time.  All transport
    machinery is inherited from :class:`OpenAICompatProvider`.

    Usage:
        provider = NebiusProvider()
        provider.initialize(ProviderConfig(api_key='<your-key>'))
        provider.connect('meta-llama/Llama-3.3-70B-Instruct')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    # Parameterize the base's shared error mapping with Nebius's taxonomy.
    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError

    REASONING_CAPABLE_MODELS = REASONING_CAPABLE_MODELS

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._base_url = DEFAULT_BASE_URL
        # Manual override tiers (the catalog auto-detect is PRIMARY).
        self._context_length_knob: Optional[int] = None
        self._modalities_knob: Optional[List[str]] = None
        # Cached ``GET /v1/models`` catalog (RichModel entries carry
        # context_length + architecture.modality).  Fetched once, lazily.
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "nebius"

    # ==================== Credential / context hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve the API key + base URL; validate (required for hosted).

        Pulls workspace_path / config_root from ``config.extra`` so credential
        lookup resolves under the session's explicit config_root rather than the
        unreliable ``JAATO_CONFIG_ROOT`` env var for headless sessions.
        """
        _ws_path = config.extra.get('workspace_path') if config.extra else None
        _config_root = config.extra.get('config_root') if config.extra else None

        self._api_key = config.api_key or resolve_api_key(
            workspace_path=_ws_path, config_root=_config_root,
        )
        self._base_url = config.extra.get("base_url") or resolve_base_url()

        # Required for the hosted service; optional only when fronted by a local
        # proxy (see is_self_hosted).
        if not self._api_key and not is_self_hosted(self._base_url):
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(),
            )

    def _resolve_context(self, config: ProviderConfig) -> None:
        """Defer the context window to ``connect()`` (catalog bootstrap).

        Overrides the base's resolve-at-init: Nebius's PRIMARY tier is the
        ``GET /v1/models`` catalog, consulted lazily once the active model is
        known, so init stays a cheap, network-light credential check.  Only the
        manual-override knobs are stashed here.
        """
        self._context_length_knob = (
            config.extra.get("context_length") or resolve_context_length()
        )
        modalities_override = config.extra.get("modalities")
        if modalities_override is not None:
            if not isinstance(modalities_override, (list, tuple)) or not all(
                isinstance(m, str) for m in modalities_override
            ):
                raise TypeError(
                    "Nebius 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active model and bootstrap its context window.

        Context resolution is catalog auto-detect PRIMARY: the per-model
        ``context_length`` from ``GET /v1/models`` (RichModel) wins, then a
        manual override (profile knob / ``JAATO_NEBIUS_CONTEXT_LENGTH``), else a
        fail-fast error (no hardcoded default).  The catalog lookup is a
        cacheable HTTP GET, orthogonal to ``skip_model_test``, so it always
        runs.  Real model validation happens on the first chat call.
        """
        self._model_name = model
        self._context_length = resolve_context_window(
            detect_capacity=lambda: self._lookup_context_length(model),
            profile_value=self._context_length_knob,
        ) or 0
        if not self._context_length:
            raise ValueError(
                "Nebius provider: context_length could not be resolved.  The "
                f"model {model!r} is absent from GET /v1/models (so no per-model "
                "context_length was found), and no manual override is set.  Set "
                "plugin_configs.nebius.context_length in the profile, or "
                f"{ENV_NEBIUS_CONTEXT_LENGTH} in the environment.  No hardcoded "
                "fallback exists per the project's no-fallback rule."
            )
        self._trace(
            f"[CONNECT] model={model} context_length={self._context_length}"
        )

    # ==================== Catalog (bootstrap metadata) ====================

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and cache the Nebius ``GET /v1/models`` catalog.

        RichModel entries carry ``context_length`` and ``architecture.modality``
        (an ``"input->output"`` string).  Returns the cached list on subsequent
        calls; an empty list is returned and *not* cached on failure, so the
        next call retries.
        """
        if self._catalog_cache is not None:
            return self._catalog_cache

        import httpx

        url = f"{self._base_url.rstrip('/')}/models"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = httpx.get(url, headers=headers, timeout=15)
            response.raise_for_status()
        except Exception as exc:
            self._trace(f"[CATALOG] fetch failed: {type(exc).__name__}: {exc}")
            return []

        try:
            data = response.json()
        except ValueError as exc:
            self._trace(f"[CATALOG] invalid JSON: {exc}")
            return []

        catalog = data.get("data") if isinstance(data, dict) else None
        if not isinstance(catalog, list):
            self._trace("[CATALOG] response missing 'data' list")
            return []

        self._catalog_cache = catalog
        return catalog

    def _lookup_context_length(self, model: str) -> Optional[int]:
        """Return the catalog-reported context length for ``model``."""
        for entry in self._fetch_catalog():
            if entry.get("id") == model:
                ctx = entry.get("context_length")
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
        return None

    def _lookup_modalities(self, model: str) -> Optional[List[str]]:
        """Return the catalog-reported INPUT modalities for ``model``.

        Parses ``architecture.modality`` — Nebius uses the OpenRouter-style
        ``"input->output"`` form.  The INPUT side (left of ``->``) is split on
        ``+``/``,`` and the known modality tokens (text / image / audio / video)
        extracted.  Returns ``None`` when the model is absent or the field is
        missing, so resolution falls through to the manual knob / text floor.
        """
        for entry in self._fetch_catalog():
            if entry.get("id") != model:
                continue
            arch = entry.get("architecture")
            if not isinstance(arch, dict):
                return None
            modality = arch.get("modality")
            if not isinstance(modality, str) or not modality.strip():
                return None
            input_side = modality.split("->", 1)[0]
            tokens = re.split(r"[+,/\s]+", input_side.strip().lower())
            known = {"text", "image", "audio", "video"}
            mods = [t for t in tokens if t in known]
            return mods or None
        return None

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities ``model`` (default: the active model) accepts.

        Catalog auto-detect PRIMARY (``architecture.modality``) → manual
        ``modalities`` knob → text-only floor.
        """
        model = model or self._model_name
        if not model:
            return {MODALITY_TEXT}
        resolved = resolve_modalities(
            detect=lambda: self._lookup_modalities(model),
            profile_value=self._modalities_knob,
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models from Nebius's ``GET /v1/models`` catalog.

        Returns model IDs sorted alphabetically; an empty list on network
        failure (callers surface that as a clear error rather than a fake
        catalog).
        """
        ids = [entry["id"] for entry in self._fetch_catalog() if entry.get("id")]
        if prefix:
            ids = [m for m in ids if m.startswith(prefix)]
        return sorted(ids)

    # ==================== Auth introspection ====================

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify that authentication is configured.

        Must work before ``initialize()`` — checks for the API key without ever
        touching ``self._client``.  A profile-supplied ``api_key`` (the daemon
        expands ``pass://`` secrets into the verify-time ``ProviderConfig``)
        takes effect during this pre-init gate.  A stored credential file that
        exists but cannot be loaded surfaces its load error via ``on_message``.

        Raises:
            APIKeyNotFoundError: If no key found and not self-hosted (unless
                ``allow_interactive``).
        """
        import os
        from .auth import try_load_credentials_with_reason
        from .env import ENV_NEBIUS_API_KEY

        base_url = resolve_base_url()

        profile_key: Optional[str] = None
        if config is not None:
            profile_key = config.api_key or (
                config.extra.get("api_key") if config.extra else None
            )
        if profile_key:
            if on_message:
                on_message("Found Nebius API key (profile config)")
            return True

        env_key = os.environ.get(ENV_NEBIUS_API_KEY)
        if env_key:
            if on_message:
                on_message("Found Nebius API key (environment variable)")
            return True

        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            if on_message:
                on_message("Found Nebius API key (stored credentials)")
            return True

        if load_error:
            if on_message:
                on_message(
                    f"Nebius credentials file found but could not be loaded: "
                    f"{load_error}"
                )
                on_message(
                    "Run 'nebius-auth key <your_api_key>' to re-authenticate, "
                    f"or set {ENV_NEBIUS_API_KEY}."
                )
            if not allow_interactive:
                raise APIKeyNotFoundError(
                    checked_locations=get_checked_credential_locations()
                )
            return False

        if is_self_hosted(base_url):
            if on_message:
                on_message(f"Self-hosted Nebius endpoint ({base_url}), no API key required")
            return True

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        return False

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        import os
        from .env import ENV_NEBIUS_API_KEY

        if is_self_hosted(self._base_url):
            return f"Self-hosted Nebius ({self._base_url})"

        if os.environ.get(ENV_NEBIUS_API_KEY):
            return f"Nebius API key ({ENV_NEBIUS_API_KEY})"

        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, 'extra', None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get('workspace_path'),
                config_root=extra.get('config_root'),
            )
            if cred_path:
                return f"Nebius API key ({cred_path})"
        except ImportError:
            pass

        return "Nebius API key"


def create_provider() -> NebiusProvider:
    """Factory function for plugin discovery.

    Returns:
        A new NebiusProvider instance.
    """
    return NebiusProvider()
