"""OVHcloud AI Endpoints model provider implementation.

OVHcloud AI Endpoints exposes an OpenAI-compatible chat completions API
through a single unified gateway, so this provider is a thin subclass of
:class:`OpenAICompatProvider` (the shared OpenAI-compat machinery —
streaming loop, completion skeleton incl. ``api_params`` / ``extra_body`` /
``tool_choice`` forwarding, error mapping, capability boilerplate).

OVHcloud-specific concerns live here: identity, the error-class
parameterization, credential resolution (incl. the explicit opt-in to the
keyless rate-limited free tier), and the **catalog bootstrap** — the context
window and input modalities are auto-detected from the ``GET /v1/models``
catalog at ``connect()`` once the active model is known, with profile-knob /
env override fallbacks and a fail-loud "not configured" error (no hardcoded
fallback).  OVHcloud's catalog metadata is not pinned to one schema, so the
context lookup tolerates the common key spellings (``context_length``,
``max_model_len``, ``max_context_length``) and degrades to the manual tiers
when none is present.

Authentication (API key, Bearer token):
- JAATO_OVHCLOUD_API_KEY (jaato namespace) or OVH_AI_ENDPOINTS_ACCESS_TOKEN
  (the vendor's own documented variable), or stored credentials via
  ovhcloud-auth.  JAATO_OVHCLOUD_ALLOW_ANONYMOUS=true opts into the keyless
  rate-limited free tier.

Environment variables:
    JAATO_OVHCLOUD_API_KEY / OVH_AI_ENDPOINTS_ACCESS_TOKEN: API key
    JAATO_OVHCLOUD_BASE_URL: Endpoint (default: https://oai.endpoints.kepler.ai.cloud.ovh.net/v1)
    JAATO_OVHCLOUD_MODEL: Default model name
    JAATO_OVHCLOUD_CONTEXT_LENGTH: Override the catalog-detected context window
    JAATO_OVHCLOUD_ALLOW_ANONYMOUS: Opt into the keyless free tier
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
    ENV_OVHCLOUD_CONTEXT_LENGTH,
    resolve_api_key,
    resolve_allow_anonymous,
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
# Matched case-insensitively by prefix/suffix against the active model ID
# (OVHcloud IDs are case-sensitive on the wire but compared lowercased here).
REASONING_CAPABLE_MODELS = [
    "deepseek-r1",
    "gpt-oss",
    "qwq",
]

# Catalog keys OVHcloud (or a fronting proxy) may use for a model's context
# window.  Checked in order; the first positive int wins.
_CONTEXT_LENGTH_KEYS = ("context_length", "max_model_len", "max_context_length")


class OVHcloudProvider(OpenAICompatProvider):
    """OVHcloud AI Endpoints model provider.

    Access to the OVHcloud AI Endpoints catalog (Llama, Mistral, Qwen,
    gpt-oss, DeepSeek, ...) via the OpenAI-compatible unified gateway.
    Context window and input modalities are bootstrapped from
    ``GET /v1/models`` at connect time when the catalog reports them.
    All transport machinery is inherited from :class:`OpenAICompatProvider`.

    Usage:
        provider = OVHcloudProvider()
        provider.initialize(ProviderConfig(api_key='<your-key>'))
        provider.connect('gpt-oss-120b')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    # Parameterize the base's shared error mapping with OVHcloud's taxonomy.
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
        # Explicit opt-in to OVHcloud's keyless rate-limited free tier.
        self._allow_anonymous: bool = False
        # Cached ``GET /v1/models`` catalog.  Fetched once, lazily.
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "ovhcloud"

    # ==================== Credential / context hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve the API key + base URL; validate (required unless opted out).

        Pulls workspace_path / config_root from ``config.extra`` so credential
        lookup resolves under the session's explicit config_root rather than the
        unreliable ``JAATO_CONFIG_ROOT`` env var for headless sessions.

        A missing key is an error unless the endpoint is a local proxy
        (:func:`is_self_hosted`) or the keyless free tier was explicitly
        opted into (``allow_anonymous`` knob / ``JAATO_OVHCLOUD_ALLOW_ANONYMOUS``)
        — never a silent fallback.
        """
        _ws_path = config.extra.get('workspace_path') if config.extra else None
        _config_root = config.extra.get('config_root') if config.extra else None

        self._api_key = config.api_key or resolve_api_key(
            workspace_path=_ws_path, config_root=_config_root,
        )
        self._base_url = config.extra.get("base_url") or resolve_base_url()
        self._allow_anonymous = bool(
            config.extra.get("allow_anonymous")
            if config.extra.get("allow_anonymous") is not None
            else resolve_allow_anonymous()
        )

        if (
            not self._api_key
            and not self._allow_anonymous
            and not is_self_hosted(self._base_url)
        ):
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config),
            )

    def _resolve_context(self, config: ProviderConfig) -> None:
        """Defer the context window to ``connect()`` (catalog bootstrap).

        Overrides the base's resolve-at-init: the PRIMARY tier is the
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
                    "OVHcloud 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active model and bootstrap its context window.

        Context resolution is catalog auto-detect PRIMARY: a per-model
        context length from ``GET /v1/models`` wins (any of the common key
        spellings — OVHcloud's catalog schema is not pinned), then a manual
        override (profile knob / ``JAATO_OVHCLOUD_CONTEXT_LENGTH``), else a
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
                "OVHcloud provider: context_length could not be resolved.  "
                f"The model {model!r} is absent from GET /v1/models (or the "
                "catalog entry reports no context length), and no manual "
                "override is set.  Set plugin_configs.ovhcloud.context_length "
                f"in the profile, or {ENV_OVHCLOUD_CONTEXT_LENGTH} in the "
                "environment (the per-model context size is listed at "
                "https://endpoints.ai.cloud.ovh.net/catalog).  No hardcoded "
                "fallback exists per the project's no-fallback rule."
            )
        self._trace(
            f"[CONNECT] model={model} context_length={self._context_length}"
        )

    # ==================== Catalog (bootstrap metadata) ====================

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and cache the OVHcloud ``GET /v1/models`` catalog.

        Returns the cached list on subsequent calls; an empty list is
        returned and *not* cached on failure, so the next call retries.

        The GET is **anonymous** — deliberately no ``Authorization`` header.
        OVHcloud's ``/v1/models`` is a *public* catalog (the same fixed model
        list + pricing regardless of account; there is no OVHcloud private-
        model / fine-tune surface like Nebius's account-scoped catalog), and
        it serves that catalog with ``200`` to unauthenticated callers.
        Attaching a Bearer token buys nothing and can only hurt: a token that
        is valid for chat but not entitled on the models endpoint answers the
        keyed catalog GET with ``401`` ("token is not allowed to perform the
        required actions on any tenant"), which would silently break the
        context-window auto-detect even though the public catalog holds the
        answer.  Verified live 2026-07: anon → 200 (22 models, each with a
        ``context_length``); the same token keyed → 401.  So the catalog is
        always fetched without the key.
        """
        if self._catalog_cache is not None:
            return self._catalog_cache

        import httpx

        url = f"{self._base_url.rstrip('/')}/models"
        try:
            response = httpx.get(url, timeout=15)
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
        """Return the catalog-reported context length for ``model``.

        Tolerates the common key spellings (:data:`_CONTEXT_LENGTH_KEYS`)
        since OVHcloud's OpenAI-compat catalog schema is not pinned; returns
        ``None`` when the model is absent or no key is present, so resolution
        falls through to the manual override tiers.
        """
        for entry in self._fetch_catalog():
            if entry.get("id") != model:
                continue
            for key in _CONTEXT_LENGTH_KEYS:
                ctx = entry.get(key)
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
            return None
        return None

    def _lookup_modalities(self, model: str) -> Optional[List[str]]:
        """Return the catalog-reported INPUT modalities for ``model``.

        Parses ``architecture.modality`` when present (the OpenRouter-style
        ``"input->output"`` form some gateways report).  The INPUT side (left
        of ``->``) is split on ``+``/``,`` and the known modality tokens
        (text / image / audio / video) extracted.  Returns ``None`` when the
        model is absent or the field is missing, so resolution falls through
        to the manual knob / text floor.
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

        Catalog auto-detect PRIMARY (``architecture.modality``, when the
        catalog reports it) → manual ``modalities`` knob → text-only floor.
        Vision models on OVHcloud (e.g. ``Qwen2.5-VL-72B-Instruct``,
        ``llava-next-mistral-7b``) that the catalog doesn't classify are
        asserted via ``plugin_configs.ovhcloud.modalities: ["text","image"]``.
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
        """List available models from OVHcloud's ``GET /v1/models`` catalog.

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
        The keyless free tier (``allow_anonymous`` knob /
        ``JAATO_OVHCLOUD_ALLOW_ANONYMOUS``) passes the gate with a notice.

        Raises:
            APIKeyNotFoundError: If no key found, anonymous access is not
                opted into, and the endpoint is not self-hosted (unless
                ``allow_interactive``).
        """
        import os
        from .auth import try_load_credentials_with_reason
        from .env import ENV_OVHCLOUD_API_KEY

        base_url = resolve_base_url()

        profile_key: Optional[str] = None
        allow_anonymous_knob: Optional[Any] = None
        if config is not None:
            profile_key = config.api_key or (
                config.extra.get("api_key") if config.extra else None
            )
            if config.extra:
                allow_anonymous_knob = config.extra.get("allow_anonymous")
        if profile_key:
            if on_message:
                on_message("Found OVHcloud API key (profile config)")
            return True

        env_key = os.environ.get(ENV_OVHCLOUD_API_KEY)
        if env_key:
            if on_message:
                on_message("Found OVHcloud API key (environment variable)")
            return True

        from .env import ENV_OVHCLOUD_API_KEY_VENDOR
        vendor_key = os.environ.get(ENV_OVHCLOUD_API_KEY_VENDOR)
        if vendor_key:
            if on_message:
                on_message(
                    "Found OVHcloud API key "
                    f"({ENV_OVHCLOUD_API_KEY_VENDOR} environment variable)"
                )
            return True

        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            if on_message:
                on_message("Found OVHcloud API key (stored credentials)")
            return True

        if load_error:
            if on_message:
                on_message(
                    f"OVHcloud credentials file found but could not be loaded: "
                    f"{load_error}"
                )
                on_message(
                    "Run 'ovhcloud-auth key <your_api_key>' to re-authenticate, "
                    f"or set {ENV_OVHCLOUD_API_KEY}."
                )
            if not allow_interactive:
                raise APIKeyNotFoundError(
                    checked_locations=get_checked_credential_locations(config=config)
                )
            return False

        anonymous = (
            bool(allow_anonymous_knob)
            if allow_anonymous_knob is not None
            else resolve_allow_anonymous()
        )
        if anonymous:
            if on_message:
                on_message(
                    "No OVHcloud API key; using the anonymous free tier "
                    "(heavily rate-limited)"
                )
            return True

        if is_self_hosted(base_url):
            if on_message:
                on_message(f"Self-hosted OVHcloud proxy ({base_url}), no API key required")
            return True

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config)
            )

        return False

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        import os
        from .env import ENV_OVHCLOUD_API_KEY, ENV_OVHCLOUD_API_KEY_VENDOR

        if not self._api_key and self._allow_anonymous:
            return "OVHcloud anonymous free tier (rate-limited)"

        if is_self_hosted(self._base_url):
            return f"Self-hosted OVHcloud proxy ({self._base_url})"

        if os.environ.get(ENV_OVHCLOUD_API_KEY):
            return f"OVHcloud API key ({ENV_OVHCLOUD_API_KEY})"

        if os.environ.get(ENV_OVHCLOUD_API_KEY_VENDOR):
            return f"OVHcloud API key ({ENV_OVHCLOUD_API_KEY_VENDOR})"

        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, 'extra', None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get('workspace_path'),
                config_root=extra.get('config_root'),
            )
            if cred_path:
                return f"OVHcloud API key ({cred_path})"
        except ImportError:
            pass

        return "OVHcloud API key"


def create_provider() -> OVHcloudProvider:
    """Factory function for plugin discovery.

    Returns:
        A new OVHcloudProvider instance.
    """
    return OVHcloudProvider()
