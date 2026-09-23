"""Doubleword model provider implementation.

Doubleword (https://doubleword.ai) is a hosted serverless inference service
for open models (DeepSeek, Qwen, GLM, Kimi, gpt-oss, Nemotron, ...) that
prices by **delivery window** on one OpenAI-compatible API: the same
``/chat/completions`` endpoint serves the realtime tier and, via the
``service_tier: "flex"`` request-body field, the discounted async tier
(queued work guaranteed to start within ~1 minute — minutes-level latency
instead of seconds, at a fraction of realtime pricing).  This provider is a
thin subclass of :class:`OpenAICompatProvider` (the shared OpenAI-compat
machinery — streaming loop, completion skeleton incl. ``api_params`` /
``extra_body`` / ``tool_choice`` forwarding, error mapping, capability
boilerplate).

Doubleword-specific concerns live here: identity, the error-class
parameterization, credential resolution, the ``service_tier`` knob (the
only ``api_params`` addition over the shared allowlist, with a
``JAATO_DOUBLEWORD_SERVICE_TIER`` env fallback), and the **catalog
bootstrap** — the context window and input modalities are looked up in the
``GET /v1/models`` catalog at ``connect()`` once the active model is known,
falling through to the profile-knob / env override tiers and finally a
fail-loud "not configured" error (no hardcoded fallback).  The catalog GET
is **authenticated** (Doubleword's ``/models`` is account-scoped — it lists
the models your API key can access).

.. important::
   **The catalog tier is dormant against the live API.**  Verified
   2026-07-19: ``GET /v1/models`` serves bare OpenAI-shaped entries —
   all 25 listed models carry only ``{id, object, created, owned_by}``,
   with no context-length or modality field.  So
   ``plugin_configs.doubleword.context_length`` (or
   ``JAATO_DOUBLEWORD_CONTEXT_LENGTH``) is effectively **required**, and
   vision models must be asserted through the ``modalities`` knob.  The
   lookups below are kept — they are written to tolerate the schema
   Doubleword *might* serve (``context_length`` / ``max_model_len`` /
   ``max_context_length``) and return ``None`` cleanly when, as today,
   none is present — so enriching the catalog upstream would make the
   manual knob redundant with no code change here.

Doubleword's deeper-discount batch tier (JSONL file upload + ``/batches``
jobs, 24h SLA) is a different interaction shape and out of scope here; a
follow-up may add background-polling / batch-job support.

Authentication (API key, Bearer token):
- JAATO_DOUBLEWORD_API_KEY, or stored credentials via doubleword-auth.
  Keys are generated at https://app.doubleword.ai/api-keys.

Environment variables:
    JAATO_DOUBLEWORD_API_KEY: API key
    JAATO_DOUBLEWORD_BASE_URL: Endpoint (default: https://api.doubleword.ai/v1)
    JAATO_DOUBLEWORD_MODEL: Default model name
    JAATO_DOUBLEWORD_CONTEXT_LENGTH: Context window (required in practice — the
        catalog reports no per-model window; see the note above)
    JAATO_DOUBLEWORD_SERVICE_TIER: Inference tier ("flex" = discounted async,
        "priority" = realtime); the profile knob
        plugin_configs.doubleword.api_params.service_tier wins when both are set
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
    ENV_DOUBLEWORD_CONTEXT_LENGTH,
    ENV_DOUBLEWORD_API_KEY,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_service_tier,
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
# Matched case-insensitively by prefix/suffix against the active model ID.
# Doubleword IDs are vendor-prefixed (e.g. ``deepseek-ai/DeepSeek-V4-Pro``),
# so vendor prefixes cover whole reasoning-capable families.
REASONING_CAPABLE_MODELS = [
    "deepseek",       # deepseek-ai/DeepSeek-V4-*
    "qwen",           # Qwen/Qwen3.5-* (thinking variants)
    "gpt-oss",        # gpt-oss-20b/120b (bare id)
    "openai/gpt-oss",  # vendor-prefixed spelling
    "zai-org",        # zai-org/GLM-5.*
    "moonshotai",     # moonshotai/Kimi-K2.*
    "nvidia",         # nvidia/Nemotron-3-*
]

# Catalog keys Doubleword (or a fronting proxy) may use for a model's context
# window.  Checked in order; the first positive int wins.
_CONTEXT_LENGTH_KEYS = ("context_length", "max_model_len", "max_context_length")


class DoublewordProvider(OpenAICompatProvider):
    """Doubleword model provider.

    Access to the Doubleword catalog (DeepSeek, Qwen, GLM, Kimi, gpt-oss,
    Nemotron, ...) via the OpenAI-compatible inference API, including the
    discounted flex (async) tier via ``api_params.service_tier``.  Context
    window and input modalities are bootstrapped from ``GET /v1/models`` at
    connect time.  All transport machinery is inherited from
    :class:`OpenAICompatProvider`.

    Usage:
        provider = DoublewordProvider()
        provider.initialize(ProviderConfig(api_key='<your-key>'))
        provider.connect('deepseek-ai/DeepSeek-V4-Pro')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    # Parameterize the base's shared error mapping with Doubleword's taxonomy.
    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError

    REASONING_CAPABLE_MODELS = REASONING_CAPABLE_MODELS

    # Doubleword extends the shared allowlist with ``service_tier`` — the
    # tier selector is an ordinary Chat Completions body field there
    # ("flex" = discounted async, "priority" = realtime).  Values are
    # forwarded verbatim (Doubleword validates server-side), so future
    # tier names work without a provider release.
    _FORWARDED_API_PARAMS = (
        OpenAICompatProvider._FORWARDED_API_PARAMS | frozenset({"service_tier"})
    )

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._base_url = DEFAULT_BASE_URL
        # Manual override tiers (the catalog auto-detect is PRIMARY).
        self._context_length_knob: Optional[int] = None
        self._modalities_knob: Optional[List[str]] = None
        # Cached ``GET /v1/models`` catalog.  Fetched once, lazily.
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "doubleword"

    # ==================== Credential / context hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve the API key + base URL; validate (required unless local).

        Pulls workspace_path / config_root from ``config.extra`` so credential
        lookup resolves under the session's explicit config_root rather than the
        unreliable ``JAATO_CONFIG_ROOT`` env var for headless sessions.

        A missing key is an error unless the endpoint is a local proxy
        (:func:`is_self_hosted`) — never a silent fallback.
        """
        _ws_path = config.extra.get('workspace_path') if config.extra else None
        _config_root = config.extra.get('config_root') if config.extra else None

        self._api_key = config.api_key or resolve_api_key(
            workspace_path=_ws_path, config_root=_config_root,
        )
        self._base_url = config.extra.get("base_url") or resolve_base_url()

        if not self._api_key and not is_self_hosted(self._base_url):
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config),
            )

    def _read_api_params(self, config: ProviderConfig) -> None:
        """Read ``api_params`` + apply the ``service_tier`` env fallback.

        The base reads the allowlisted profile ``api_params`` (which for
        this provider includes ``service_tier``).  When the profile does
        not pin a tier, ``JAATO_DOUBLEWORD_SERVICE_TIER`` fills it in, so
        a session can be flipped onto the discounted flex tier without
        editing profiles.  Profile knob wins over env when both are set.
        """
        super()._read_api_params(config)
        if "service_tier" not in self._api_params:
            env_tier = resolve_service_tier()
            if env_tier:
                self._api_params["service_tier"] = env_tier
        tier = self._api_params.get("service_tier")
        if tier is not None and not isinstance(tier, str):
            raise TypeError(
                "doubleword 'service_tier' must be a string (e.g. \"flex\" "
                f"or \"priority\"), got {type(tier).__name__}"
            )
        if tier:
            self._trace(f"[INIT] service_tier={tier}")

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
                    "Doubleword 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active model and bootstrap its context window.

        Context resolution is catalog auto-detect PRIMARY: a per-model
        context length from ``GET /v1/models`` wins (any of the common key
        spellings — Doubleword's catalog schema is not pinned), then a
        manual override (profile knob / ``JAATO_DOUBLEWORD_CONTEXT_LENGTH``),
        else a fail-fast error (no hardcoded default).  The catalog lookup is
        a cacheable HTTP GET, orthogonal to ``skip_model_test``, so it always
        runs.  Real model validation happens on the first chat call.
        """
        self._model_name = model
        self._context_length = resolve_context_window(
            detect_capacity=lambda: self._lookup_context_length(model),
            profile_value=self._context_length_knob,
        ) or 0
        if not self._context_length:
            raise ValueError(
                "Doubleword provider: context_length could not be resolved.  "
                f"The model {model!r} is absent from GET /v1/models (or the "
                "catalog entry reports no context length), and no manual "
                "override is set.  Set plugin_configs.doubleword.context_length "
                f"in the profile, or {ENV_DOUBLEWORD_CONTEXT_LENGTH} in the "
                "environment (the per-model context size is listed at "
                "https://doubleword.ai/models).  No hardcoded fallback exists "
                "per the project's no-fallback rule."
            )
        self._trace(
            f"[CONNECT] model={model} context_length={self._context_length}"
        )

    # ==================== Catalog (bootstrap metadata) ====================

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and cache the Doubleword ``GET /v1/models`` catalog.

        Returns the cached list on subsequent calls; an empty list is
        returned and *not* cached on failure, so the next call retries.

        The GET is **authenticated** (Bearer key): Doubleword's ``/models``
        endpoint requires auth and the listing is account-scoped — it
        returns the models *your* API key can access (401 without a key).
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
        """Return the catalog-reported context length for ``model``.

        Tolerates the common key spellings (:data:`_CONTEXT_LENGTH_KEYS`)
        since Doubleword's OpenAI-compat catalog schema is not pinned;
        returns ``None`` when the model is absent or no key is present, so
        resolution falls through to the manual override tiers.

        **Returns ``None`` for every live model today** — Doubleword's
        catalog carries only ``{id, object, created, owned_by}`` (verified
        2026-07-19), so the manual tier always wins.  Kept for the case
        where the vendor enriches the listing.  See
        ``test_live_catalog_shape_falls_through_to_manual_tier``.
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

        **Returns ``None`` for every live model today** — the catalog carries
        no ``architecture`` block (verified 2026-07-19), so vision models
        must be asserted via ``plugin_configs.doubleword.modalities``.
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
        Vision models on Doubleword (e.g. ``Qwen/Qwen3-VL-30B-A3B-Instruct``)
        that the catalog doesn't classify are asserted via
        ``plugin_configs.doubleword.modalities: ["text","image"]``.
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
        """List available models from Doubleword's ``GET /v1/models`` catalog.

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
            APIKeyNotFoundError: If no key found and the endpoint is not
                self-hosted (unless ``allow_interactive``).
        """
        import os
        from .auth import try_load_credentials_with_reason

        base_url = resolve_base_url()

        profile_key: Optional[str] = None
        if config is not None:
            profile_key = config.api_key or (
                config.extra.get("api_key") if config.extra else None
            )
        if profile_key:
            if on_message:
                on_message("Found Doubleword API key (profile config)")
            return True

        env_key = os.environ.get(ENV_DOUBLEWORD_API_KEY)
        if env_key:
            if on_message:
                on_message("Found Doubleword API key (environment variable)")
            return True

        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            if on_message:
                on_message("Found Doubleword API key (stored credentials)")
            return True

        if load_error:
            if on_message:
                on_message(
                    f"Doubleword credentials file found but could not be loaded: "
                    f"{load_error}"
                )
                on_message(
                    "Run 'doubleword-auth key <your_api_key>' to re-authenticate, "
                    f"or set {ENV_DOUBLEWORD_API_KEY}."
                )
            if not allow_interactive:
                raise APIKeyNotFoundError(
                    checked_locations=get_checked_credential_locations(config=config)
                )
            return False

        if is_self_hosted(base_url):
            if on_message:
                on_message(f"Self-hosted Doubleword proxy ({base_url}), no API key required")
            return True

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config)
            )

        return False

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        import os

        if is_self_hosted(self._base_url):
            return f"Self-hosted Doubleword proxy ({self._base_url})"

        if os.environ.get(ENV_DOUBLEWORD_API_KEY):
            return f"Doubleword API key ({ENV_DOUBLEWORD_API_KEY})"

        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, 'extra', None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get('workspace_path'),
                config_root=extra.get('config_root'),
            )
            if cred_path:
                return f"Doubleword API key ({cred_path})"
        except ImportError:
            pass

        return "Doubleword API key"


def create_provider() -> DoublewordProvider:
    """Factory function for plugin discovery.

    Returns:
        A new DoublewordProvider instance.
    """
    return DoublewordProvider()
