"""LM Studio provider — OpenAI-compatible chat + native model load-control.

A thin subclass of :class:`OpenAICompatLocalHostProvider` — the shared
self-hosted machinery (streaming/completion, error mapping, the connectivity
probe, auth helpers).  LM Studio's specifics, expressed as overrides:

- **connect-time context**: the window is discovered live at ``connect()``
  (``/api/v0`` ``max_context_length`` and the loaded instance's live
  ``context_length`` via ``/api/v1/models``), not at init — so ``_resolve_context``
  only stashes the manual override and ``get_context_limit`` resolves
  discovered → override → fail-loud.
- **load-control**: when a ``load`` dict is supplied, ``connect()`` POSTs
  ``/api/v1/models/load`` — but LM Studio's ``/load`` is **not idempotent**, so
  it first looks for a matching loaded instance and reuses it (avoids pinning
  duplicate VRAM).
- ``_ERR_MIDSTREAM`` stays ``None`` — LM Studio's error handler has no
  mid-stream / pre-flight split; ``_ERR_LOAD`` is ``LMStudioLoadError``.

Auth is optional (local server); a bearer is sent when ``LMSTUDIO_API_TOKEN``
(or ``plugin_configs.lmstudio.api_token``) is set, for "Require API Token".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .._openai_compat.local_host import OpenAICompatLocalHostProvider
from ..base import ProviderConfig, resolve_context_window
from .env import (
    DEFAULT_HOST,
    resolve_api_token,
    resolve_context_length,
    resolve_host,
)
from .errors import (
    LMStudioAuthenticationError,
    LMStudioConnectionError,
    LMStudioLoadError,
    LMStudioModelNotFoundError,
)

logger = logging.getLogger(__name__)


class LMStudioProvider(OpenAICompatLocalHostProvider):
    """LM Studio provider talking to its OpenAI-compatible /v1 endpoint + the
    native /api model-control surface.  Transport / error / auth machinery is
    inherited from :class:`OpenAICompatLocalHostProvider`."""

    # Parameterize the shared local-host error mapping with LM Studio's taxonomy.
    _ERR_AUTHENTICATION = LMStudioAuthenticationError
    _ERR_MODEL_NOT_FOUND = LMStudioModelNotFoundError
    _ERR_CONNECTION = LMStudioConnectionError
    # _ERR_MIDSTREAM stays None — no mid-stream / pre-flight split.
    _ERR_LOAD = LMStudioLoadError

    def __init__(self):
        super().__init__()
        self._host = DEFAULT_HOST
        # Context tiers: explicit override (profile/env) vs live-discovered.
        self._context_length_override: Optional[int] = None
        self._discovered_context_length: Optional[int] = None
        # Load body passthrough.  None = passive mode (no /load at connect()).
        self._load_config: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        """Provider identifier — used as the key in ``plugin_configs``."""
        return "lmstudio"

    # ==================== Credential / context hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve host + optional bearer token + the optional load body."""
        self._host = (config.extra.get("host") or resolve_host()).rstrip("/")
        self._api_token = config.extra.get("api_token") or resolve_api_token()
        self._base_url = f"{self._host}/v1"
        self._api_key = self._api_token   # base _create_client substitutes a placeholder

        load = config.extra.get("load")
        if load is not None and not isinstance(load, dict):
            raise ValueError(
                f"lmstudio load config must be a dict, got {type(load).__name__}"
            )
        self._load_config = load

        self._auth_info = (
            f"local ({self._host}, bearer)" if self._api_token
            else f"local ({self._host})"
        )

    def _resolve_context(self, config: ProviderConfig) -> None:
        """Defer the window to connect() (live discovery); stash the override.

        Overrides the base's resolve-at-init: LM Studio's PRIMARY tier is the
        live ``/api/v0,v1`` state, only known once the model is connected (and
        possibly just ``/load``-ed), so init only captures the manual override.
        """
        ctx = config.extra.get("context_length")
        self._context_length_override = int(ctx) if ctx else resolve_context_length()

    def _probe_url(self) -> str:
        return f"{self._host}/api/v0/models"

    def _resolve_api_token(self) -> Optional[str]:
        return resolve_api_token()

    def get_context_limit(self) -> int:
        """Discovered (tier-1, live) → explicit override (tier-2/3) → 0."""
        return resolve_context_window(
            detect_capacity=lambda: self._discovered_context_length,
            profile_value=self._context_length_override,
        ) or 0

    # ==================== Connection (catalog + load + context) ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Select the model: validate, optionally load, then discover its window.

        Raises:
            LMStudioModelNotFoundError: Model is not present in LM Studio.
            LMStudioLoadError: ``/api/v1/models/load`` returned a non-2xx.
            ValueError: Context window unresolved (no discovered + no override).
        """
        if not skip_model_test:
            catalog = self._fetch_catalog()
            if catalog and model not in {entry["id"] for entry in catalog}:
                raise LMStudioModelNotFoundError(
                    model, available=[entry["id"] for entry in catalog],
                )
        self._model_name = model

        if self._load_config is not None:
            self._load_model(model, self._load_config)

        # Tier-1 auto-detect: the live configured/max context window.
        self._refresh_discovered_context(model)
        if not self._discovered_context_length and not self._context_length_override:
            raise ValueError(
                "LM Studio provider: context_length could not be resolved.  "
                "GET /api/v0/models did not report max_context_length (and no "
                "instance is loaded), and no manual override is set.  Set "
                "plugin_configs.lmstudio.context_length in the profile, or "
                "LMSTUDIO_CONTEXT_LENGTH in the environment.  No hardcoded "
                "fallback exists per the project's no-fallback rule."
            )

        logger.info(
            "Connected to LM Studio model: %s (context=%d, load_applied=%s)",
            model, self.get_context_limit(), bool(self._load_config),
        )

    def _load_model(self, model: str, load_params: Dict[str, Any]) -> None:
        """Ensure an instance of ``model`` is loaded with the supplied params.

        LM Studio's ``POST /api/v1/models/load`` is **not idempotent** — every
        call spins up a fresh in-memory instance.  So first look for a loaded
        instance whose config matches every explicitly-requested key and reuse
        it; only ``/load`` when none matches.
        """
        desired_config = {k: v for k, v in load_params.items() if k != "model"}

        existing = self._find_matching_loaded_instance(model, desired_config)
        if existing is not None:
            instance_id, _ = existing
            self._trace(
                f"[LOAD] reusing existing instance '{instance_id}' for {model} "
                f"(matches {len(desired_config)} requested config keys)"
            )
            return

        body = {**load_params, "model": model}
        self._trace(
            f"[LOAD] POST {self._host}/api/v1/models/load "
            f"body_keys={sorted(body.keys())} (no matching instance)"
        )
        try:
            response = httpx.post(
                f"{self._host}/api/v1/models/load",
                json=body, headers=self._auth_headers(),
                timeout=600.0,   # disk read + KV-cache init + GPU transfer
            )
        except httpx.HTTPError as exc:
            raise LMStudioConnectionError(self._host, f"load failed: {exc}") from exc
        if response.status_code >= 400:
            raise LMStudioLoadError(
                model=model, status_code=response.status_code,
                body=response.text, load_config=load_params,
            )
        self._trace(f"[LOAD] status={response.status_code}")

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Query ``GET /api/v0/models`` → the raw ``data`` array (empty on error)."""
        try:
            response = httpx.get(
                f"{self._host}/api/v0/models",
                headers=self._auth_headers(), timeout=10,
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except httpx.HTTPError as exc:
            logger.warning("Failed to list LM Studio models: %s", exc)
            return []

    def _fetch_v1_models(self) -> List[Dict[str, Any]]:
        """Query ``GET /api/v1/models`` → the ``models`` array.

        Richer than v0: each entry carries ``key`` (= v0 ``id``),
        ``max_context_length``, and a ``loaded_instances`` array with each
        running instance's live config — used for load-reuse + context discovery.
        """
        try:
            response = httpx.get(
                f"{self._host}/api/v1/models",
                headers=self._auth_headers(), timeout=10,
            )
            response.raise_for_status()
            return response.json().get("models", [])
        except httpx.HTTPError as exc:
            logger.warning("Failed to list LM Studio models (v1): %s", exc)
            return []

    def _find_matching_loaded_instance(
        self, model: str, desired_config: Dict[str, Any],
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Find a loaded instance of ``model`` matching every desired key.

        Keys not in ``desired_config`` are ignored (LM Studio's own defaults
        mustn't trigger spurious reloads).  Returns ``(instance_id, config)`` or
        ``None`` (also on a v1 fetch failure → caller falls through to /load).
        """
        for entry in self._fetch_v1_models():
            if entry.get("key") != model:
                continue
            for inst in entry.get("loaded_instances", []) or []:
                inst_config = inst.get("config", {}) or {}
                if all(inst_config.get(k) == v for k, v in desired_config.items()):
                    return inst.get("id", ""), inst_config
            return None
        return None

    def _refresh_discovered_context(self, model: str) -> None:
        """Set ``_discovered_context_length`` from live LM Studio state.

        Priority: the loaded instance's live ``config.context_length`` → the
        model entry's ``max_context_length`` (no instance loaded) → leave unset
        (connect() then relies on the manual override or fails fast).
        """
        for entry in self._fetch_v1_models():
            if entry.get("key") != model:
                continue
            for inst in entry.get("loaded_instances", []) or []:
                ctx = (inst.get("config") or {}).get("context_length")
                if isinstance(ctx, int) and ctx > 0:
                    self._discovered_context_length = ctx
                    return
            max_ctx = entry.get("max_context_length")
            if isinstance(max_ctx, int) and max_ctx > 0:
                self._discovered_context_length = max_ctx
            return

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models available in LM Studio via ``/api/v0/models``."""
        names = [entry["id"] for entry in self._fetch_catalog()]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return sorted(names)

    @staticmethod
    def login(on_message=None) -> None:
        """No-op — LM Studio runs locally, no login."""
        if on_message:
            on_message(
                "LM Studio runs locally; no login required. Load models via the "
                "LM Studio UI or `lms load`, or set plugin_configs.lmstudio.load "
                "to auto-load at connect()."
            )


def create_provider() -> LMStudioProvider:
    """Factory function consumed by the provider discovery machinery."""
    return LMStudioProvider()
