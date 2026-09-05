"""Helmcode model provider implementation.

Helmcode (https://helmcode.com) is private AI inference for European
teams: open-weight models (GLM, DeepSeek, Qwen, Gemma) served from
hardware Helmcode operates in the EU, with zero prompt retention, behind
one OpenAI-compatible API at ``https://api.helmcode.com/v1``.  Volume on
those models is covered by a flat monthly plan rather than metered per
token.  The same endpoint and the same key also reach nine resold
frontier models from Anthropic, OpenAI and Google, which are a separate
offer: they run on the provider's own US infrastructure and are billed
per token from prepaid credit.  Only the ``model`` id distinguishes the
two, which is why :class:`CreditsExhaustedError` exists — see below.

This provider is a thin subclass of :class:`OpenAICompatProvider` (the
shared OpenAI-compat machinery — streaming loop, completion skeleton
incl. ``api_params`` / ``extra_body`` / ``tool_choice`` forwarding, error
mapping, capability boilerplate).

Helmcode-specific concerns live here: identity, the error-class
parameterization, credential resolution, the ``402 credits_exhausted``
mapping, and the **catalog bootstrap** — the context window and input
modalities are looked up in the ``GET /v1/models`` catalog at
``connect()`` once the active model is known, falling through to the
profile-knob / env override tiers and finally a fail-loud "not
configured" error (no hardcoded fallback).  The catalog GET is
**authenticated**: Helmcode's ``/v1/models`` answers ``401`` to an
unkeyed caller (verified live 2026-09-05), and the listing is
account-scoped — it carries the resold frontier models alongside
Helmcode's own, and reflects the entitlements of the key (GLM 5.3 is a
per-key add-on).

.. note::
   **Whether the catalog reports a context window is unverified.**  The
   endpoint requires a key, so its entry schema could not be inspected
   without an account.  The lookups below are therefore written the same
   way as the rest of the OpenAI-compat family: they tolerate the schema
   Helmcode *may* serve (``context_length`` / ``max_model_len`` /
   ``max_context_length``) and return ``None`` cleanly when none is
   present, so resolution falls through to
   ``plugin_configs.helmcode.context_length`` /
   ``JAATO_HELMCODE_CONTEXT_LENGTH`` and, failing that, to a loud error
   naming both.  If the catalog does report a window, the manual knob is
   redundant with no code change here.  The published per-model windows
   are at https://helmcode.com/docs/models — as of 2026-09: 1M for
   ``glm-5.3`` and ``deepseek-v4-flash``, 256K for ``qwen3.6`` and
   ``gemma4``.  They are deliberately NOT hardcoded (project
   no-fallback rule): a served window is an operational fact that moves
   (``glm-5.2`` served 500K of a 1M model), and a stale constant would
   silently mis-budget GC.

Helmcode's non-chat surfaces — embeddings (``qwen3-embedding``), the
``/v1/rerank`` endpoint, speech (``whisper-large-v3``, ``kokoro``) and
the ``/v1/search`` web-search tool — are outside a model provider's
contract, which covers chat completions only.  They are reachable with
the same key and base URL from a plugin that wants them.

Authentication (API key, Bearer token):
- JAATO_HELMCODE_API_KEY (jaato namespace) or HELMCODE_API_KEY (the
  vendor's own documented variable), or stored credentials via
  helmcode-auth.  Keys are issued per workspace from the Helmcode
  dashboard.

Environment variables:
    JAATO_HELMCODE_API_KEY / HELMCODE_API_KEY: API key
    JAATO_HELMCODE_BASE_URL: Endpoint (default: https://api.helmcode.com/v1)
    JAATO_HELMCODE_MODEL: Default model name
    JAATO_HELMCODE_CONTEXT_LENGTH: Override the catalog-detected context window
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
    ENV_HELMCODE_API_KEY,
    ENV_HELMCODE_API_KEY_VENDOR,
    ENV_HELMCODE_CONTEXT_LENGTH,
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
    CreditsExhaustedError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)

# Models known to expose reasoning/thinking content via ``reasoning_content``.
# Matched case-insensitively by prefix/suffix against the active model ID.
# Helmcode ids are bare (not vendor-prefixed), and every language model in
# its own catalogue is documented as reasoning-capable, so each is listed
# by its id prefix rather than by a vendor family.
REASONING_CAPABLE_MODELS = [
    "glm-",              # glm-5.3 (reasoning: High and Max)
    "deepseek-",         # deepseek-v4-flash
    "qwen3",             # qwen3.6
    "gemma",             # gemma4
]

# Catalog keys Helmcode (or a fronting proxy) may use for a model's context
# window.  Checked in order; the first positive int wins.
_CONTEXT_LENGTH_KEYS = ("context_length", "max_model_len", "max_context_length")

# Substring that marks Helmcode's ``402`` as the credit-exhausted refusal
# rather than some other payment-required condition.  Matched
# case-insensitively against the error text.  The documented code is
# ``credits_exhausted``; the shorter stem is what is matched so a
# reworded body ("prepaid credit balance is 0") is still recognised
# rather than silently demoted to an unmapped error.
_CREDITS_EXHAUSTED_MARKERS = ("credit",)


class HelmcodeProvider(OpenAICompatProvider):
    """Helmcode model provider.

    Access to the Helmcode catalogue — the open-weight models it runs in
    the EU (``glm-5.3``, ``deepseek-v4-flash``, ``qwen3.6``, ``gemma4``)
    and the resold frontier models (``claude-*``, ``gpt-5.6-*``,
    ``gemini-*``) — via the OpenAI-compatible API.  Context window and
    input modalities are bootstrapped from ``GET /v1/models`` at connect
    time.  All transport machinery is inherited from
    :class:`OpenAICompatProvider`.

    Usage:
        provider = HelmcodeProvider()
        provider.initialize(ProviderConfig(api_key='<your-key>'))
        provider.connect('qwen3.6')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    # Parameterize the base's shared error mapping with Helmcode's taxonomy.
    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError
    # Not part of the base's ``_ERR_*`` contract — Helmcode-only, raised
    # by the ``_handle_api_error`` override below.
    _ERR_CREDITS_EXHAUSTED = CreditsExhaustedError

    REASONING_CAPABLE_MODELS = REASONING_CAPABLE_MODELS

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
        return "helmcode"

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
                    "Helmcode 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active model and bootstrap its context window.

        Context resolution is catalog auto-detect PRIMARY: a per-model
        context length from ``GET /v1/models`` wins (any of the common key
        spellings — Helmcode's catalog schema is not pinned), then a manual
        override (profile knob / ``JAATO_HELMCODE_CONTEXT_LENGTH``), else a
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
                "Helmcode provider: context_length could not be resolved.  "
                f"The model {model!r} is absent from GET /v1/models (or the "
                "catalog entry reports no context length), and no manual "
                "override is set.  Set plugin_configs.helmcode.context_length "
                f"in the profile, or {ENV_HELMCODE_CONTEXT_LENGTH} in the "
                "environment (the per-model context size is listed at "
                "https://helmcode.com/docs/models).  No hardcoded fallback "
                "exists per the project's no-fallback rule."
            )
        self._trace(
            f"[CONNECT] model={model} context_length={self._context_length}"
        )

    # ==================== Error mapping ====================

    def _handle_api_error(self, error: Exception) -> None:
        """Map SDK exceptions, adding Helmcode's ``402 credits_exhausted``.

        The base maps the OpenAI-compat taxonomy (auth / rate limit / not
        found / context / 5xx) and leaves a ``402`` unmapped, so it would
        surface as a raw ``APIStatusError``.  Helmcode raises exactly one
        documented ``402``: prepaid credit ran out, which only the resold
        frontier models can hit.  It is caught here FIRST — before
        delegating — and turned into :class:`CreditsExhaustedError`,
        which :meth:`classify_error` deliberately leaves non-transient so
        the turn fails fast with a message naming the remedy (top up, or
        switch to a plan-covered model) instead of burning the retry
        budget on a refusal that cannot change without a human.
        """
        status_code = getattr(error, "status_code", None)
        if status_code == 402:
            error_str = str(error).lower()
            if any(m in error_str for m in _CREDITS_EXHAUSTED_MARKERS):
                raise self._ERR_CREDITS_EXHAUSTED(
                    model=self._model_name,
                    original_error=str(error),
                ) from error

        super()._handle_api_error(error)

    # ==================== Catalog (bootstrap metadata) ====================

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and cache the Helmcode ``GET /v1/models`` catalog.

        Returns the cached list on subsequent calls; an empty list is
        returned and *not* cached on failure, so the next call retries.

        The GET is **authenticated** (Bearer key): Helmcode's ``/models``
        endpoint requires auth — an unkeyed request answers ``401``
        ``{"error": {"type": "auth_error"}}`` (verified live 2026-09-05)
        — and the listing is account-scoped, reflecting the models the
        key can actually reach (GLM 5.3 is a per-key add-on, and the
        resold frontier models appear alongside Helmcode's own).
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
        since Helmcode's OpenAI-compat catalog schema is not pinned;
        returns ``None`` when the model is absent or no key is present, so
        resolution falls through to the manual override tiers.
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
        Helmcode's vision models (``qwen3.6`` and ``gemma4`` take image
        input; ``glm-5.3`` and ``deepseek-v4-flash`` are text only) are
        asserted via ``plugin_configs.helmcode.modalities: ["text","image"]``
        when the catalog does not classify them.
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
        """List available models from Helmcode's ``GET /v1/models`` catalog.

        Returns model IDs sorted alphabetically; an empty list on network
        failure (callers surface that as a clear error rather than a fake
        catalog).  The listing is account-scoped, so it reflects what this
        key can reach — including the resold frontier models.
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
                on_message("Found Helmcode API key (profile config)")
            return True

        env_key = os.environ.get(ENV_HELMCODE_API_KEY)
        if env_key:
            if on_message:
                on_message("Found Helmcode API key (environment variable)")
            return True

        vendor_key = os.environ.get(ENV_HELMCODE_API_KEY_VENDOR)
        if vendor_key:
            if on_message:
                on_message(
                    "Found Helmcode API key "
                    f"({ENV_HELMCODE_API_KEY_VENDOR} environment variable)"
                )
            return True

        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            if on_message:
                on_message("Found Helmcode API key (stored credentials)")
            return True

        if load_error:
            if on_message:
                on_message(
                    f"Helmcode credentials file found but could not be loaded: "
                    f"{load_error}"
                )
                on_message(
                    "Run 'helmcode-auth key <your_api_key>' to re-authenticate, "
                    f"or set {ENV_HELMCODE_API_KEY}."
                )
            if not allow_interactive:
                raise APIKeyNotFoundError(
                    checked_locations=get_checked_credential_locations(config=config)
                )
            return False

        if is_self_hosted(base_url):
            if on_message:
                on_message(f"Self-hosted Helmcode proxy ({base_url}), no API key required")
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
            return f"Self-hosted Helmcode proxy ({self._base_url})"

        if os.environ.get(ENV_HELMCODE_API_KEY):
            return f"Helmcode API key ({ENV_HELMCODE_API_KEY})"

        if os.environ.get(ENV_HELMCODE_API_KEY_VENDOR):
            return f"Helmcode API key ({ENV_HELMCODE_API_KEY_VENDOR})"

        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, 'extra', None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get('workspace_path'),
                config_root=extra.get('config_root'),
            )
            if cred_path:
                return f"Helmcode API key ({cred_path})"
        except ImportError:
            pass

        return "Helmcode API key"


def create_provider() -> HelmcodeProvider:
    """Factory function for plugin discovery.

    Returns:
        A new HelmcodeProvider instance.
    """
    return HelmcodeProvider()
