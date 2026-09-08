"""Moonshot AI Kimi model provider implementation.

Kimi (https://platform.kimi.ai) is Moonshot AI's hosted API, OpenAI-
compatible at ``https://api.moonshot.ai/v1`` (China: ``api.moonshot.cn``).
This provider is a thin subclass of :class:`OpenAICompatProvider`; Kimi's
own dialect lives in the hooks it overrides:

- **The catalog is the best of the three vendors.**  ``GET /v1/models``
  reports ``context_length``, ``supports_image_in``, ``supports_video_in``
  and ``supports_reasoning`` per model, so the context window and input
  modalities are catalog-PRIMARY at ``connect()`` (the nebius shape),
  beneath the ``context_length`` / ``modalities`` knobs, fail-loud.  There
  is no built-in table: every pre-K2.6 id was retired on 2026-08-31 and
  answers ``404 resource_not_found_error``.
- **Reasoning replay.**  K3 "requires the complete assistant message
  returned by the API to be passed back as-is, including
  ``reasoning_content`` and ``tool_calls``"; K2.7-code and K2.6 with
  ``keep: "all"`` the same.  ``replay_reasoning = True``.
- **Two thinking dialects, one framework knob set.**  ``kimi-k3`` takes
  a top-level ``reasoning_effort`` (``low`` / ``high`` / ``max``) and has
  no ``thinking`` object; ``kimi-k2.6`` takes ``thinking: {type, keep}``;
  ``kimi-k2.7-code`` accepts only ``{type: enabled, keep: all}``.
  ``thinking_level`` → effort, ``enable_thinking`` → type,
  ``thinking_keep`` → keep; ``thinking_budget`` has no equivalent.
- **Sampling parameters are a 400.**  ``temperature``, ``top_p``, ``n``
  and the penalties are absent from the request schema (fixed per
  model), so the forwarded allow-list is the narrow set the spec names.
- **``tool_choice``** ``required`` / named only on K3 (and a named choice
  is incompatible with thinking); K2.x accept ``auto`` / ``none``.
- **``strict`` defaults to true** on Kimi's tool definitions; the
  framework's schemas are not authored for strict mode, so every tool is
  stamped ``strict: false`` unless ``api_params.strict_tools`` asks.
- **Cache accounting** is reported as top-level ``usage.cached_tokens``,
  not under ``prompt_tokens_details``.
- **``429`` splits three ways** by ``error.type``: engine overloaded and
  rate limited are transient; ``exceeded_current_quota_error`` (balance)
  is not and must stop the retry loop.

Environment variables:
    JAATO_KIMI_API_KEY (then MOONSHOT_API_KEY): API key
    JAATO_KIMI_BASE_URL: Endpoint (default: https://api.moonshot.ai/v1;
        China https://api.moonshot.cn/v1; Kimi Code plan
        https://api.kimi.com/coding/v1 with its own model ids)
    JAATO_KIMI_MODEL: Default model name
    JAATO_KIMI_CONTEXT_LENGTH: Context-window override
"""

from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List, Optional, Set

from .._openai_compat.base import OpenAICompatProvider
from .._openai_compat._lazy import get_openai_module
from .._openai_compat.converters import tool_schemas_to_openai
from ..base import (
    MODALITY_TEXT,
    ProviderConfig,
    resolve_context_window,
    resolve_modalities,
)
from jaato_sdk.plugins.model_provider.types import ThinkingConfig, ToolSchema
from .env import (
    DEFAULT_BASE_URL,
    ENV_KIMI_API_KEY,
    ENV_KIMI_CONTEXT_LENGTH,
    ENV_KIMI_VENDOR_API_KEY,
    get_checked_credential_locations,
    is_self_hosted,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
)
from .errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    QuotaExhaustedError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Every current Kimi model exposes ``reasoning_content``.
REASONING_CAPABLE_MODELS = ["kimi-"]

# K3's ``reasoning_effort`` vocabulary (no ``medium``).
REASONING_EFFORTS = frozenset({"low", "high", "max"})

# ``error.type`` values Kimi sends with HTTP 429.  Only the quota one is
# non-transient: the account cannot pay, and retrying cannot change that.
_QUOTA_ERROR_TYPES = frozenset({"exceeded_current_quota_error"})

# 400 messages that mean "too much input" on this wire (docs).
_CONTEXT_PHRASES = ("input token length too long", "exceeds the model specification")


def _family(model: Optional[str]) -> str:
    """``"k3"`` / ``"k2.7-code"`` / ``"k2.6"`` / ``""`` for an unknown id.

    The Kimi Code plan's ids (``k3``, ``k3-256k``, ``kimi-for-coding``)
    are matched by the same prefixes.
    """
    name = (model or "").lower()
    if name.startswith(("kimi-k3", "k3")):
        return "k3"
    if name.startswith("kimi-k2.7"):
        return "k2.7-code"
    if name.startswith("kimi-k2.6"):
        return "k2.6"
    return ""


def _configured_key_source(config: Optional[ProviderConfig]) -> Optional[str]:
    """Where a key is configured ahead of the stored file: the profile
    (``api_key`` top-level or in ``extra``), else the first env var set."""
    import os

    if config is not None and (config.api_key or (config.extra or {}).get("api_key")):
        return "profile config"
    for var in (ENV_KIMI_API_KEY, ENV_KIMI_VENDOR_API_KEY):
        if os.environ.get(var):
            return var
    return None


class KimiProvider(OpenAICompatProvider):
    """Moonshot AI Kimi provider over the OpenAI-compatible API.

    Usage:
        provider = KimiProvider()
        provider.initialize(ProviderConfig(api_key='<your-key>'))
        provider.connect('kimi-k3')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError

    REASONING_CAPABLE_MODELS = REASONING_CAPABLE_MODELS

    replay_reasoning = True

    _THINKING_KNOBS = frozenset({
        "enable_thinking", "thinking_level", "thinking_budget", "thinking_keep",
    })
    _MAX_TOKENS_WIRE_NAME = "max_completion_tokens"

    # The request schema names no sampling parameter — sending one is a
    # 400 — so the allow-list is exactly what the spec accepts.
    # ``strict_tools`` is consumed here (it shapes the tool definitions),
    # never forwarded.
    _FORWARDED_API_PARAMS = frozenset({
        "max_tokens", "tool_choice", "stop", "response_format",
        "prompt_cache_key",
    }) | OpenAICompatProvider._FORWARDED_API_PARAMS.intersection({"modalities", "audio"})

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._base_url = DEFAULT_BASE_URL
        self._context_length_knob: Optional[int] = None
        self._modalities_knob: Optional[List[str]] = None
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None
        # Thinking knobs: None = not set by the profile = the vendor default.
        self._enable_thinking_knob: Optional[bool] = None
        self._thinking_level: Optional[str] = None
        self._thinking_keep: Optional[str] = None
        self._strict_tools: bool = False

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "kimi"

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

    def _read_api_params(self, config: ProviderConfig) -> None:
        """The base's allow-listed read, plus ``strict_tools`` (a
        tool-definition knob, not a body field)."""
        api_params = config.extra.get("api_params") or {}
        if isinstance(api_params, dict):
            self._strict_tools = bool(api_params.pop("strict_tools", False))
        super()._read_api_params(config)

    def _apply_thinking_knobs(self, api_params: Dict[str, Any]) -> None:
        """Stash the framework thinking knobs; they become wire fields per
        model family in :meth:`_thinking_request_fields`."""
        if "thinking_budget" in api_params:
            raise ValueError(
                "kimi api_params: thinking_budget has no equivalent on Kimi — "
                "K3 takes thinking_level (low|high|max → reasoning_effort), "
                "K2.6 takes enable_thinking / thinking_keep."
            )
        level = api_params.get("thinking_level")
        if level is not None:
            if str(level).lower() not in REASONING_EFFORTS:
                raise ValueError(
                    f"kimi api_params: thinking_level must be one of "
                    f"{sorted(REASONING_EFFORTS)} (K3's reasoning_effort), got {level!r}"
                )
            self._thinking_level = str(level).lower()
        if "enable_thinking" in api_params:
            self._enable_thinking_knob = bool(api_params["enable_thinking"])
            self._enable_thinking = self._enable_thinking_knob
        keep = api_params.get("thinking_keep")
        if keep is not None:
            if keep not in ("all",):
                raise ValueError(
                    f"kimi api_params: thinking_keep must be \"all\" (or unset for "
                    f"the vendor default), got {keep!r}"
                )
            self._thinking_keep = keep

    def _thinking_request_fields(self) -> Dict[str, Any]:
        """The family's thinking fields; nothing for an id outside the three.

        K3 has no ``thinking`` object and cannot be turned off, so only
        ``reasoning_effort`` is sent (when the profile set a level).
        K2.7-code accepts one shape.  K2.6 gets ``type`` from
        ``enable_thinking`` and ``keep`` only when the profile set it —
        an unset knob is the vendor default, not ``null`` spelled out.
        """
        family = _family(self._model_name)
        if family == "k3":
            if self._enable_thinking_knob is False:
                logger.info("kimi: enable_thinking=false is ignored on %s — K3 "
                            "reasoning cannot be disabled", self._model_name)
            return {"reasoning_effort": self._thinking_level} if self._thinking_level else {}
        if family == "k2.7-code":
            return {"thinking": {"type": "enabled", "keep": "all"}}
        if family == "k2.6":
            thinking: Dict[str, Any] = {}
            if self._enable_thinking_knob is not None:
                thinking["type"] = "enabled" if self._enable_thinking_knob else "disabled"
            if self._thinking_keep is not None:
                thinking["keep"] = self._thinking_keep
            return {"thinking": thinking} if thinking else {}
        return {}

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Runtime toggle.  On K3 a change of effort mid-session breaks the
        prefix cache (docs), and reasoning cannot be disabled at all."""
        super().set_thinking_config(config)
        self._enable_thinking_knob = config.enabled
        if _family(self._model_name) == "k3":
            logger.info("kimi: thinking config changed mid-session on %s; K3 "
                        "reasoning stays on, and a changed effort restarts the "
                        "prefix cache", self._model_name)

    def supports_thinking(self) -> bool:
        """The catalog's ``supports_reasoning`` when listed, else the prefix."""
        entry = self._catalog_entry(self._model_name) if self._model_name else None
        if entry and isinstance(entry.get("supports_reasoning"), bool):
            return entry["supports_reasoning"]
        return self._is_reasoning_capable()

    def _tool_choice_vocabulary(self, model: Optional[str]) -> Optional[FrozenSet[str]]:
        """K3 takes the full set; K2.x ``auto`` / ``none`` only."""
        if _family(model) == "k3":
            return None
        return frozenset({"auto", "none"})

    def _wire_tools(self, tools: List[ToolSchema]) -> Optional[List[Dict[str, Any]]]:
        """Stamp ``strict`` explicitly: Kimi defaults it to true, and the
        framework's schemas are not written for strict mode."""
        wire = tool_schemas_to_openai(tools)
        for tool in wire or []:
            function = tool.get("function")
            if isinstance(function, dict):
                function["strict"] = self._strict_tools
        return wire

    @staticmethod
    def _extract_cache_tokens(usage: Any) -> Optional[int]:
        """Kimi reports the hit count as top-level ``usage.cached_tokens``;
        fall back to the OpenAI-shaped location for a proxy that rewrites."""
        cached = getattr(usage, "cached_tokens", None)
        if isinstance(cached, int) and cached:
            return cached
        return OpenAICompatProvider._extract_cache_tokens(usage)

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
                    "kimi 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got {type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active model and bootstrap its context window from the
        catalog (PRIMARY) → the manual override → fail-loud."""
        self._model_name = model
        self._context_length = resolve_context_window(
            detect_capacity=lambda: self._lookup_context_length(model),
            profile_value=self._context_length_knob,
        ) or 0
        if not self._context_length:
            raise ValueError(
                "Kimi provider: context_length could not be resolved.  "
                f"The model {model!r} is absent from GET /v1/models (every id "
                "before kimi-k2.6 was retired on 2026-08-31 and answers 404) "
                "and no manual override is set.  Set "
                "plugin_configs.kimi.context_length in the profile or "
                f"{ENV_KIMI_CONTEXT_LENGTH} in the environment (K3: 1048576; "
                "K2.7-code / K2.6: 262144).  No hardcoded fallback exists per "
                "the project's no-fallback rule."
            )
        self._trace(f"[CONNECT] model={model} context_length={self._context_length}")

    # ==================== Catalog ====================

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and cache ``GET /v1/models`` (authenticated).

        An empty list is returned and *not* cached on failure.
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
        """The catalog's ``context_length`` for ``model``, or None."""
        entry = self._catalog_entry(model)
        value = entry.get("context_length") if entry else None
        return value if isinstance(value, int) and value > 0 else None

    def _lookup_modalities(self, model: str) -> Optional[List[str]]:
        """``supports_image_in`` → image.  Video is reported too but no wire
        in the tree carries it, so it is not declared (cross-rule)."""
        entry = self._catalog_entry(model)
        if not entry:
            return None
        mods = ["text"]
        if entry.get("supports_image_in") is True:
            mods.append("image")
        return mods

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities: catalog → ``modalities`` knob → text."""
        model = model or self._model_name
        if not model:
            return {MODALITY_TEXT}
        resolved = resolve_modalities(
            detect=lambda: self._lookup_modalities(model),
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

    @staticmethod
    def _error_type(error: Exception) -> str:
        """``error.type`` from an OpenAI SDK status error's parsed body."""
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            inner = body.get("error") if isinstance(body.get("error"), dict) else body
            return str(inner.get("type") or "")
        return ""

    def _handle_api_error(self, error: Exception) -> None:
        """Kimi's three-way 429 and its context-limit phrasings, then the
        shared mapping."""
        openai = get_openai_module()
        if isinstance(error, openai.RateLimitError) and self._error_type(error) in _QUOTA_ERROR_TYPES:
            raise QuotaExhaustedError(original_error=str(error)) from error
        if isinstance(error, openai.BadRequestError):
            text = str(error).lower()
            if any(phrase in text for phrase in _CONTEXT_PHRASES):
                raise ContextLimitError(
                    model=self._model_name or "unknown", original_error=str(error),
                ) from error
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
        from .auth import try_load_credentials_with_reason

        say = on_message or (lambda _m: None)
        source = _configured_key_source(config)
        if source:
            say(f"Found Kimi API key ({source})")
            return True
        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            say("Found Kimi API key (stored credentials)")
            return True
        if load_error:
            say(f"Kimi credentials file found but could not be loaded: {load_error}")
            say(f"Run 'kimi-auth key <your_api_key>' to re-authenticate, or set {ENV_KIMI_API_KEY}.")
        elif is_self_hosted(resolve_base_url()):
            say(f"Self-hosted Kimi proxy ({resolve_base_url()}), no API key required")
            return True
        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config))
        return False

    def get_auth_info(self) -> str:
        """Short description of the credential source used."""
        import os

        if is_self_hosted(self._base_url):
            return f"Self-hosted Kimi proxy ({self._base_url})"
        for var in (ENV_KIMI_API_KEY, ENV_KIMI_VENDOR_API_KEY):
            if os.environ.get(var):
                return f"Kimi API key ({var})"
        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, "extra", None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get("workspace_path"),
                config_root=extra.get("config_root"),
            )
            if cred_path:
                return f"Kimi API key ({cred_path})"
        except ImportError:
            pass
        return "Kimi API key"


def create_provider() -> KimiProvider:
    """Factory function for plugin discovery."""
    return KimiProvider()
