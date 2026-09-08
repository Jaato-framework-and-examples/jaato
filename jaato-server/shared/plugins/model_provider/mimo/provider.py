"""Xiaomi MiMo model provider implementation.

MiMo (https://mimo.mi.com) is Xiaomi's hosted inference API for the MiMo
model family, OpenAI-compatible at ``https://api.xiaomimimo.com/v1``.
This provider is a thin subclass of :class:`OpenAICompatProvider` (the
shared streaming loop, completion skeleton, error mapping and capability
boilerplate); MiMo's own dialect lives in the hooks it overrides:

- **Reasoning replay is mandatory.**  In thinking mode an assistant
  message that carries ``tool_calls`` must also carry the
  ``reasoning_content`` the model produced, or the next request is
  refused with ``400 Param Incorrect`` (vendor docs, corroborated by
  XiaomiMiMo/MiMo#44).  ``replay_reasoning = True`` opts into the
  framework seam (docs/design/minimax-kimi-mimo-providers.md §3).
- **Thinking toggle** is the non-standard body field
  ``thinking: {"type": "enabled" | "disabled"}`` (default enabled on the
  vendor side).  The profile convention ``api_params.enable_thinking``
  maps onto it; ``thinking_level`` / ``thinking_budget`` have no MiMo
  equivalent on chat completions and are rejected with a clear error.
  In thinking mode the vendor forces ``temperature=1.0`` / ``top_p=0.95``
  whatever is sent.
- **``tool_choice`` accepts only ``"auto"``**; anything else is folded to
  ``auto`` with a WARNING (never a silent drop, never a 400).
- **``max_completion_tokens``** is the output cap (``max_tokens`` is
  renamed at apply time).
- **``finish_reason: repetition_truncation``** is a truncation and maps
  to ``MAX_TOKENS``.
- **Catalog** — ``GET /v1/models`` is authenticated and returns bare
  ``{id, object, owned_by}`` entries, so the context window comes from
  :data:`MODEL_CONTEXT_LIMITS` (longest-prefix match) beneath the
  ``plugin_configs.mimo.context_length`` / ``JAATO_MIMO_CONTEXT_LENGTH``
  override, fail-loud for an id neither names; the catalog lookup is
  kept in front so an enriched listing takes over with no code change.
- **Errors** — ``402`` insufficient balance and ``403`` region / key
  restriction are not transient; ``421`` is the vendor's content-filter
  status.

Caching is automatic on the vendor side (``prompt_tokens_details.
cached_tokens`` is read by the base); nothing is emitted.

Environment variables:
    JAATO_MIMO_API_KEY (then MIMO_API_KEY): API key
    JAATO_MIMO_BASE_URL: Endpoint (default: https://api.xiaomimimo.com/v1;
        Token Plan hosts are https://token-plan-{cn,sgp,ams}.xiaomimimo.com/v1)
    JAATO_MIMO_MODEL: Default model name
    JAATO_MIMO_CONTEXT_LENGTH: Context-window override
"""

from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List, Optional, Set

from .._openai_compat.base import OpenAICompatProvider
from .._openai_compat.converters import map_finish_reason
from .._openai_compat._lazy import get_openai_module
from ..base import (
    MODALITY_TEXT,
    ProviderConfig,
    resolve_context_window,
    resolve_modalities,
)
from jaato_sdk.plugins.model_provider.types import FinishReason
from .env import (
    DEFAULT_BASE_URL,
    ENV_MIMO_API_KEY,
    ENV_MIMO_CONTEXT_LENGTH,
    ENV_MIMO_VENDOR_API_KEY,
    get_checked_credential_locations,
    is_self_hosted,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
)
from .errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    ContentFilteredError,
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    QuotaExhaustedError,
    RateLimitError,
    RegionDeniedError,
)

logger = logging.getLogger(__name__)

# Context windows of the current public models (vendor docs, 2026-09).
# Longest-prefix, case-insensitive match against the model id, so
# ``mimo-v2.5-pro`` must not be shadowed by ``mimo-v2.5``.  The V2 series
# (``mimo-v2-flash`` / ``-pro`` / ``-omni``) was deprecated on 2026-06-30
# and is deliberately absent: the vendor auto-routes those ids to V2.5,
# but a table entry would assert a window for a model that no longer
# exists.
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "mimo-v2.5-pro": 1_048_576,
    "mimo-v2.5": 1_048_576,
}

# Input modalities of the current models.  ``mimo-v2.5`` is omnimodal on
# the vendor side (image, video, audio); this wire carries images only,
# and declaring ``audio`` without an ``input_audio`` path would trip the
# capability cross-rule — audio is a follow-up once the OpenAI
# ``input_audio`` block is probed against the endpoint.
MODEL_INPUT_MODALITIES: Dict[str, FrozenSet[str]] = {
    "mimo-v2.5-pro": frozenset({"text"}),
    "mimo-v2.5": frozenset({"text", "image"}),
}

# Every current MiMo model exposes ``reasoning_content``.
REASONING_CAPABLE_MODELS = ["mimo"]

# Catalog keys a fronting proxy might use for the window.  Checked in
# order; the live catalog reports none of them today.
_CONTEXT_LENGTH_KEYS = ("context_length", "max_model_len", "max_context_length")


def _longest_prefix(table: Dict[str, Any], model: str) -> Optional[Any]:
    """Value of the longest ``table`` key that prefixes ``model`` (case-insensitive)."""
    name = model.lower()
    best = None
    for prefix, value in table.items():
        if name.startswith(prefix.lower()) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, value)
    return best[1] if best else None


class MiMoProvider(OpenAICompatProvider):
    """Xiaomi MiMo provider over the OpenAI-compatible API.

    Usage:
        provider = MiMoProvider()
        provider.initialize(ProviderConfig(api_key='<your-key>'))
        provider.connect('mimo-v2.5-pro')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError

    REASONING_CAPABLE_MODELS = REASONING_CAPABLE_MODELS

    # The seam this provider exists to exercise: without it the second
    # request of every tool loop is a 400.
    replay_reasoning = True

    _THINKING_KNOBS = frozenset({"enable_thinking", "thinking_level", "thinking_budget"})
    _MAX_TOKENS_WIRE_NAME = "max_completion_tokens"

    # MiMo documents ``response_format: {type: json_object}`` (no
    # json_schema) on both models; ``frequency_penalty`` /
    # ``presence_penalty`` / ``seed`` are not in its parameter list.
    _FORWARDED_API_PARAMS = (
        (OpenAICompatProvider._FORWARDED_API_PARAMS
         - frozenset({"frequency_penalty", "presence_penalty", "seed"}))
        | frozenset({"response_format"})
    )

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._base_url = DEFAULT_BASE_URL
        self._context_length_knob: Optional[int] = None
        self._modalities_knob: Optional[List[str]] = None
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None
        self._auth_source: str = ""

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "mimo"

    # ==================== Credential / context hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve the API key + base URL; a missing key is an error unless
        the endpoint is a local proxy (never a silent fallback)."""
        extra = config.extra or {}
        self._api_key = config.api_key or resolve_api_key(
            workspace_path=extra.get("workspace_path"),
            config_root=extra.get("config_root"),
        )
        self._base_url = (extra.get("base_url") or resolve_base_url()).rstrip("/")
        if not self._api_key and not is_self_hosted(self._base_url):
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config),
            )

    def _apply_thinking_knobs(self, api_params: Dict[str, Any]) -> None:
        """``enable_thinking`` → the ``thinking`` body field; the other two
        framework knobs have no MiMo equivalent and are refused by name."""
        unsupported = sorted(k for k in ("thinking_level", "thinking_budget") if k in api_params)
        if unsupported:
            raise ValueError(
                f"mimo api_params: {unsupported} have no equivalent on MiMo's chat "
                "completions API (thinking is a plain on/off toggle) — use "
                "enable_thinking: true|false instead."
            )
        if "enable_thinking" in api_params:
            self._enable_thinking = bool(api_params["enable_thinking"])

    def _thinking_request_fields(self) -> Dict[str, Any]:
        """The vendor's toggle, sent explicitly on every call."""
        return {"thinking": {"type": "enabled" if self._enable_thinking else "disabled"}}

    def _tool_choice_vocabulary(self, model: Optional[str]) -> Optional[FrozenSet[str]]:
        """MiMo accepts ``tool_choice: "auto"`` and nothing else."""
        return frozenset({"auto"})

    def _map_finish_reason(self, reason: Optional[str]) -> FinishReason:
        """``repetition_truncation`` is a truncation, not an unknown label."""
        if reason == "repetition_truncation":
            return FinishReason.MAX_TOKENS
        return map_finish_reason(reason)

    def _resolve_context(self, config: ProviderConfig) -> None:
        """Stash the override knobs; the window is resolved at ``connect()``."""
        self._context_length_knob = (
            config.extra.get("context_length") or resolve_context_length()
        )
        modalities_override = config.extra.get("modalities")
        if modalities_override is not None:
            if not isinstance(modalities_override, (list, tuple)) or not all(
                isinstance(m, str) for m in modalities_override
            ):
                raise TypeError(
                    "mimo 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got {type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active model and resolve its context window.

        Catalog (the server's own answer, dormant today) → the manual
        override → :data:`MODEL_CONTEXT_LIMITS` → fail-loud.  The table
        is this provider's guess, so an operator's knob outranks it; only
        the catalog, when it ever reports a window, outranks the knob.
        The catalog GET is cacheable and orthogonal to ``skip_model_test``;
        model validation happens on the first chat call.
        """
        self._model_name = model
        self._context_length = (
            resolve_context_window(
                detect_capacity=lambda: self._lookup_context_length(model),
                profile_value=self._context_length_knob,
            )
            or _longest_prefix(MODEL_CONTEXT_LIMITS, model)
            or 0
        )
        if not self._context_length:
            raise ValueError(
                "MiMo provider: context_length could not be resolved.  "
                f"The model {model!r} is not in the built-in table "
                f"({sorted(MODEL_CONTEXT_LIMITS)}) and GET /v1/models reports no "
                "window.  Set plugin_configs.mimo.context_length in the profile "
                f"or {ENV_MIMO_CONTEXT_LENGTH} in the environment (both current "
                "models are 1,048,576).  No hardcoded fallback exists per the "
                "project's no-fallback rule."
            )
        self._trace(f"[CONNECT] model={model} context_length={self._context_length}")

    # ==================== Catalog ====================

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and cache the authenticated ``GET /v1/models`` listing.

        An empty list is returned and *not* cached on failure, so the next
        call retries.
        """
        if self._catalog_cache is not None:
            return self._catalog_cache

        import httpx

        url = f"{self._base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            response = httpx.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            self._trace(f"[CATALOG] fetch failed: {type(exc).__name__}: {exc}")
            return []
        catalog = data.get("data") if isinstance(data, dict) else None
        if not isinstance(catalog, list):
            self._trace("[CATALOG] response missing 'data' list")
            return []
        self._catalog_cache = catalog
        return catalog

    def _catalog_entry(self, model: str) -> Optional[Dict[str, Any]]:
        for entry in self._fetch_catalog():
            if entry.get("id") == model:
                return entry
        return None

    def _lookup_context_length(self, model: str) -> Optional[int]:
        """Catalog-reported window, or None (every live entry is bare today)."""
        entry = self._catalog_entry(model)
        if not entry:
            return None
        for key in _CONTEXT_LENGTH_KEYS:
            value = entry.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities: the built-in table → ``modalities`` knob → text."""
        model = model or self._model_name
        if not model:
            return {MODALITY_TEXT}
        resolved = resolve_modalities(
            detect=lambda: _longest_prefix(MODEL_INPUT_MODALITIES, model),
            profile_value=self._modalities_knob,
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """Model ids from the catalog (empty on network failure)."""
        ids = [entry["id"] for entry in self._fetch_catalog() if entry.get("id")]
        if prefix:
            ids = [m for m in ids if m.startswith(prefix)]
        return sorted(ids)

    # ==================== Errors ====================

    def _handle_api_error(self, error: Exception) -> None:
        """MiMo's extra statuses, then the shared mapping.

        ``402`` (balance), ``403`` (region or key restriction) and ``421``
        (content filter) are not transient and must not be retried; the
        base would fold the first two into a generic status error and
        ``with_retry`` would spend its whole budget on them.
        """
        openai = get_openai_module()
        if isinstance(error, openai.APIStatusError) and not isinstance(
            error, (openai.AuthenticationError, openai.RateLimitError, openai.NotFoundError),
        ):
            status = getattr(error, "status_code", 0)
            if status == 402:
                raise QuotaExhaustedError(original_error=str(error)) from error
            if status == 403:
                raise RegionDeniedError(original_error=str(error)) from error
            if status == 421:
                raise ContentFilteredError(original_error=str(error)) from error
        super()._handle_api_error(error)

    # ==================== Auth introspection ====================

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify that authentication is configured, before ``initialize()``.

        A profile-supplied ``api_key`` (``pass://`` expanded by the daemon)
        counts; a stored credential file that exists but cannot be loaded
        surfaces its load error via ``on_message``.
        """
        import os
        from .auth import try_load_credentials_with_reason

        profile_key = None
        if config is not None:
            profile_key = config.api_key or (config.extra.get("api_key") if config.extra else None)
        if profile_key:
            if on_message:
                on_message("Found MiMo API key (profile config)")
            return True
        for var in (ENV_MIMO_API_KEY, ENV_MIMO_VENDOR_API_KEY):
            if os.environ.get(var):
                if on_message:
                    on_message(f"Found MiMo API key ({var})")
                return True

        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            if on_message:
                on_message("Found MiMo API key (stored credentials)")
            return True
        if load_error and on_message:
            on_message(f"MiMo credentials file found but could not be loaded: {load_error}")
            on_message(f"Run 'mimo-auth key <your_api_key>' to re-authenticate, or set {ENV_MIMO_API_KEY}.")
        if not load_error and is_self_hosted(resolve_base_url()):
            if on_message:
                on_message(f"Self-hosted MiMo proxy ({resolve_base_url()}), no API key required")
            return True
        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config))
        return False

    def get_auth_info(self) -> str:
        """Short description of the credential source used."""
        import os

        if is_self_hosted(self._base_url):
            return f"Self-hosted MiMo proxy ({self._base_url})"
        for var in (ENV_MIMO_API_KEY, ENV_MIMO_VENDOR_API_KEY):
            if os.environ.get(var):
                return f"MiMo API key ({var})"
        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, "extra", None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get("workspace_path"),
                config_root=extra.get("config_root"),
            )
            if cred_path:
                return f"MiMo API key ({cred_path})"
        except ImportError:
            pass
        return "MiMo API key"


def create_provider() -> MiMoProvider:
    """Factory function for plugin discovery."""
    return MiMoProvider()
