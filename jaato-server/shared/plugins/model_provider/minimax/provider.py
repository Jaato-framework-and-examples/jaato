"""MiniMax model provider implementation.

MiniMax (https://platform.minimax.io) is a hosted API for the MiniMax
model family, OpenAI-compatible at ``https://api.minimax.io/v1`` (China:
``https://api.minimax.cn/v1``; keys are region-bound).  This provider is a
thin subclass of :class:`OpenAICompatProvider`; MiniMax's dialect lives in
the hooks it overrides:

- **``reasoning_split: true`` on every call.**  Without it the reasoning
  arrives *inside* ``content`` as ``<think>…</think>`` — which would
  pollute the agent's output and be replayed as text.  With it the
  stream delivers ``delta.reasoning_details[]`` (and, on some paths,
  ``reasoning_content``), so :meth:`_reasoning_from_delta` reads both.
  A ``<think>`` block that still leaks into the text is stripped and
  moved to the reasoning channel.
- **Reasoning replay.**  The vendor's rule is that "the complete model
  response must be appended to the conversation history"; with the
  split, ``reasoning_details`` is echoed alongside ``reasoning_content``
  (:meth:`_reasoning_replay_fields`).  Dropping it measured an 87 → 64
  drop on Tau².
- **Thinking toggle only on M3** (``thinking: {type: adaptive |
  disabled}``, default on over ``/v1``); the M2.x family is always on,
  so ``enable_thinking: false`` there is logged and ignored.
  ``thinking_level`` / ``thinking_budget`` have no chat-completions
  equivalent.
- **``tool_choice``** ``none`` / ``auto`` only.
- **``max_completion_tokens``** always sent: the vendor's default when
  omitted is small and truncates tool-call JSON, so the model's
  recommended cap is the fallback when the profile sets none.
- **Catalog** — ``GET /v1/models`` is bare, so the window comes from
  :data:`MODEL_CONTEXT_LIMITS` beneath the ``context_length`` knob.
- **Errors** — MiniMax appends its numeric code to the message: ``2056``
  (Token Plan window exhausted, names the reset time), ``1008``
  (insufficient balance) and the sensitive-content codes ``1026`` /
  ``1027`` are not transient.

Caching is automatic (≥ 512 tokens; ``prompt_tokens_details.
cached_tokens`` is read by the base); nothing is emitted.

Environment variables:
    JAATO_MINIMAX_API_KEY (then MINIMAX_API_KEY): API key, or a Token
        Plan subscription key (sk-cp-...), which works on /v1 unchanged
    JAATO_MINIMAX_BASE_URL: Endpoint (default: https://api.minimax.io/v1)
    JAATO_MINIMAX_MODEL: Default model name
    JAATO_MINIMAX_CONTEXT_LENGTH: Context-window override
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Set

from .._openai_compat.base import OpenAICompatProvider
from .._openai_compat._lazy import get_openai_module
from ..base import (
    MODALITY_TEXT,
    ProviderConfig,
    resolve_context_window,
    resolve_modalities,
)
from jaato_sdk.plugins.model_provider.types import Part, TurnResult
from .env import (
    DEFAULT_BASE_URL,
    ENV_MINIMAX_API_KEY,
    ENV_MINIMAX_CONTEXT_LENGTH,
    ENV_MINIMAX_VENDOR_API_KEY,
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
)

logger = logging.getLogger(__name__)

# Context windows of the current and legacy public models (vendor docs,
# 2026-09).  Longest-prefix, case-insensitive match (ids are spelled
# ``MiniMax-M3``).
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "minimax-m3": 1_000_000,
    "minimax-m2.7": 204_800,
    "minimax-m2.5": 204_800,
    "minimax-m2.1": 204_800,
    "minimax-m2": 204_800,
}

# The vendor's recommended ``max_completion_tokens`` per family — sent
# when the profile sets no cap, because the vendor's own default is small
# enough to truncate a tool call's JSON.
RECOMMENDED_MAX_OUTPUT: Dict[str, int] = {
    "minimax-m3": 131_072,
    "minimax-m2": 65_536,
}

# Input modalities: M3 accepts images (and video, which no wire in the
# tree carries); the M2 family is text-only.
MODEL_INPUT_MODALITIES: Dict[str, FrozenSet[str]] = {
    "minimax-m3": frozenset({"text", "image"}),
    "minimax-m2": frozenset({"text"}),
}

REASONING_CAPABLE_MODELS = ["minimax-m"]

_CONTEXT_LENGTH_KEYS = ("context_length", "max_model_len", "max_context_length")

# A ``<think>`` block the vendor still leaves in ``content`` on some paths.
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)

# ``... resets at 2026-09-08T12:00:00Z`` in a 2056 message.
_RESETS_AT_RE = re.compile(r"resets? at\s+([0-9T:\-+.Z]+)", re.IGNORECASE)

_QUOTA_CODES = ("2056", "1008")
_CONTENT_FILTER_CODES = ("1026", "1027")


def _longest_prefix(table: Dict[str, Any], model: str) -> Optional[Any]:
    """Value of the longest ``table`` key that prefixes ``model`` (case-insensitive)."""
    name = model.lower()
    best = None
    for prefix, value in table.items():
        if name.startswith(prefix.lower()) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, value)
    return best[1] if best else None


def _is_m3(model: Optional[str]) -> bool:
    return (model or "").lower().startswith("minimax-m3")


def _configured_key_source(config: Optional[ProviderConfig]) -> Optional[str]:
    """Where a key is configured ahead of the stored file: the profile
    (``api_key`` top-level or in ``extra``), else the first env var set."""
    import os

    if config is not None and (config.api_key or (config.extra or {}).get("api_key")):
        return "profile config"
    for var in (ENV_MINIMAX_API_KEY, ENV_MINIMAX_VENDOR_API_KEY):
        if os.environ.get(var):
            return var
    return None


def _has_code(error: Exception, codes) -> bool:
    text = str(error)
    return any(code in text for code in codes)


class MiniMaxProvider(OpenAICompatProvider):
    """MiniMax provider over the OpenAI-compatible API.

    Usage:
        provider = MiniMaxProvider()
        provider.initialize(ProviderConfig(api_key='<your-key>'))
        provider.connect('MiniMax-M3')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError

    REASONING_CAPABLE_MODELS = REASONING_CAPABLE_MODELS

    replay_reasoning = True

    _THINKING_KNOBS = frozenset({"enable_thinking", "thinking_level", "thinking_budget"})
    _MAX_TOKENS_WIRE_NAME = "max_completion_tokens"

    # ``presence_penalty`` / ``frequency_penalty`` are unsupported (docs);
    # ``seed`` is undocumented; ``service_tier: priority`` is 1.5x price.
    # ``response_format`` is silently ignored upstream for M2.x, so it is
    # not allow-listed: a profile asking for JSON mode hears it from us.
    _FORWARDED_API_PARAMS = (
        (OpenAICompatProvider._FORWARDED_API_PARAMS
         - frozenset({"frequency_penalty", "presence_penalty", "seed"}))
        | frozenset({"service_tier"})
    )

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        super().__init__()
        self._base_url = DEFAULT_BASE_URL
        self._context_length_knob: Optional[int] = None
        self._modalities_knob: Optional[List[str]] = None
        self._catalog_cache: Optional[List[Dict[str, Any]]] = None
        self._enable_thinking_knob: Optional[bool] = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "minimax"

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
        """``enable_thinking`` → M3's ``thinking`` field; the other two
        framework knobs have no MiniMax equivalent and are refused."""
        unsupported = sorted(k for k in ("thinking_level", "thinking_budget") if k in api_params)
        if unsupported:
            raise ValueError(
                f"minimax api_params: {unsupported} have no equivalent on MiniMax's "
                "chat completions API (M3 has an adaptive|disabled toggle, M2.x is "
                "always on) — use enable_thinking: true|false instead."
            )
        if "enable_thinking" in api_params:
            self._enable_thinking_knob = bool(api_params["enable_thinking"])
            self._enable_thinking = self._enable_thinking_knob

    def _thinking_request_fields(self) -> Dict[str, Any]:
        """``reasoning_split`` always; the M3 toggle when the profile set it."""
        fields: Dict[str, Any] = {"reasoning_split": True}
        if self._enable_thinking_knob is None:
            return fields
        if _is_m3(self._model_name):
            fields["thinking"] = {"type": "adaptive" if self._enable_thinking_knob else "disabled"}
        elif not self._enable_thinking_knob:
            logger.info("minimax: enable_thinking=false is ignored on %s — the M2 "
                        "family's reasoning cannot be disabled", self._model_name)
        return fields

    def _reasoning_from_delta(self, delta: Any) -> Optional[str]:
        """``reasoning_content`` or the joined ``reasoning_details[].text``."""
        reasoning = super()._reasoning_from_delta(delta)
        if reasoning:
            return reasoning
        details = getattr(delta, "reasoning_details", None)
        if not isinstance(details, list):
            return None
        texts = []
        for item in details:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)
        return "".join(texts) or None

    def _reasoning_replay_fields(self, text: str) -> Dict[str, Any]:
        """Both spellings, since the split response carries both."""
        return {
            "reasoning_content": text,
            "reasoning_details": [{
                "type": "reasoning.text",
                "id": "reasoning-text-1",
                "format": "MiniMax-response-v1",
                "index": 0,
                "text": text,
            }],
        }

    def _tool_choice_vocabulary(self, model: Optional[str]) -> Optional[FrozenSet[str]]:
        """MiniMax documents ``none`` and ``auto``."""
        return frozenset({"auto", "none"})

    def _apply_api_params(self, kwargs: Dict[str, Any], tool_choice: Optional[Any]) -> None:
        """The base's forwarding, then the recommended output cap when the
        profile set none (the vendor's own default truncates tool JSON)."""
        super()._apply_api_params(kwargs, tool_choice)
        recommended = _longest_prefix(RECOMMENDED_MAX_OUTPUT, self._model_name or "")
        if recommended:
            kwargs.setdefault("max_completion_tokens", recommended)

    def complete(self, messages, system_instruction=None, tools=None, **kwargs) -> TurnResult:
        """The base's completion, then any ``<think>`` block that leaked
        into the text is moved to the reasoning channel."""
        result = super().complete(messages, system_instruction, tools, **kwargs)
        response = getattr(result, "response", None)
        if response is not None:
            self._strip_think_tags(response)
        return result

    def _strip_think_tags(self, response) -> None:
        """Move ``<think>…</think>`` out of text parts, in place.

        Recovered reasoning fills ``thinking`` (and the replayed thought
        part) only when the split channel delivered none — a turn that
        already has reasoning is not given a second copy.
        """
        recovered: List[str] = []
        for part in response.parts:
            if part.text and "<think>" in part.text:
                recovered.extend(m.strip() for m in _THINK_RE.findall(part.text))
                part.text = _THINK_RE.sub("", part.text)
        if recovered and not response.thinking:
            response.thinking = "\n".join(recovered)
            response.parts.insert(0, Part.from_thought(response.thinking))

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
                    "minimax 'modalities' config must be a list of strings "
                    f"(e.g. [\"text\", \"image\"]), got {type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Set the active model and resolve its context window.

        Catalog (the server's own answer, dormant today) → the manual
        override → :data:`MODEL_CONTEXT_LIMITS` → fail-loud.  The table
        is this provider's guess, so an operator's knob outranks it.
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
                "MiniMax provider: context_length could not be resolved.  "
                f"The model {model!r} is not in the built-in table "
                f"({sorted(MODEL_CONTEXT_LIMITS)}) and GET /v1/models reports no "
                "window.  Set plugin_configs.minimax.context_length in the profile "
                f"or {ENV_MINIMAX_CONTEXT_LENGTH} in the environment (M3: 1000000; "
                "M2.x: 204800).  No hardcoded fallback exists per the project's "
                "no-fallback rule."
            )
        self._trace(f"[CONNECT] model={model} context_length={self._context_length}")

    # ==================== Catalog ====================

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Fetch and cache ``GET /v1/models`` (Bearer).  Empty and not
        cached on failure."""
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

    def _lookup_context_length(self, model: str) -> Optional[int]:
        """Catalog-reported window, or None (every live entry is bare today)."""
        for entry in self._fetch_catalog():
            if entry.get("id") != model:
                continue
            for key in _CONTEXT_LENGTH_KEYS:
                value = entry.get(key)
                if isinstance(value, int) and value > 0:
                    return value
            return None
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
        """MiniMax's numbered codes, then the shared mapping.

        ``2056`` (Token Plan window) and ``1008`` (balance) mean the
        account cannot pay: not transient, whatever HTTP status carried
        them — a 429 for 2056 would otherwise be retried until the backoff
        budget was spent, hours before the window resets.  ``1026`` /
        ``1027`` are the content filter.
        """
        openai = get_openai_module()
        if isinstance(error, openai.APIStatusError):
            if _has_code(error, _QUOTA_CODES):
                match = _RESETS_AT_RE.search(str(error))
                raise QuotaExhaustedError(
                    original_error=str(error),
                    resets_at=match.group(1) if match else None,
                ) from error
            if _has_code(error, _CONTENT_FILTER_CODES):
                raise ContentFilteredError(original_error=str(error)) from error
        super()._handle_api_error(error)

    # ==================== Auth introspection ====================

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify that authentication is configured, before ``initialize()``."""
        from .auth import try_load_credentials_with_reason

        say = on_message or (lambda _m: None)
        source = _configured_key_source(config)
        if source:
            say(f"Found MiniMax API key ({source})")
            return True
        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            say("Found MiniMax API key (stored credentials)")
            return True
        if load_error:
            say(f"MiniMax credentials file found but could not be loaded: {load_error}")
            say(f"Run 'minimax-auth key <your_api_key>' to re-authenticate, or set {ENV_MINIMAX_API_KEY}.")
        elif is_self_hosted(resolve_base_url()):
            say(f"Self-hosted MiniMax proxy ({resolve_base_url()}), no API key required")
            return True
        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(config=config))
        return False

    def get_auth_info(self) -> str:
        """Short description of the credential source used."""
        import os

        if is_self_hosted(self._base_url):
            return f"Self-hosted MiniMax proxy ({self._base_url})"
        for var in (ENV_MINIMAX_API_KEY, ENV_MINIMAX_VENDOR_API_KEY):
            if os.environ.get(var):
                return f"MiniMax API key ({var})"
        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, "extra", None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get("workspace_path"),
                config_root=extra.get("config_root"),
            )
            if cred_path:
                return f"MiniMax API key ({cred_path})"
        except ImportError:
            pass
        return "MiniMax API key"


def create_provider() -> MiniMaxProvider:
    """Factory function for plugin discovery."""
    return MiniMaxProvider()
