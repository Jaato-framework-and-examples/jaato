"""Triton provider — OpenAI-compatible chat + KServe v2 model lifecycle.

Triton serves models via two cooperating HTTP surfaces:

- Chat:    ``POST /v1/chat/completions``   on the OpenAI frontend (port 9000)
- Load:    ``POST /v2/repository/models/<name>/load``   on tritonserver (port 8000)
- Index:   ``POST /v2/repository/index``   on tritonserver
- Health:  ``GET  /v2/health/live``        on tritonserver

A thin subclass of :class:`OpenAICompatLocalHostProvider` — the shared
self-hosted machinery (streaming/completion, error mapping, auth helpers).
Triton's wrinkles, all expressed as overrides: the **dual URL** (chat vs
control), the ``surface=`` kwarg on its ConnectionError (runtime errors carry
``openai``, the connectivity probe carries ``control`` — the probe hits the
control surface, which load/unload depend on), and the active/passive
``connect()`` (repository-index validation + an optional load step).

Authentication: Triton has no native bearer mechanism; when ``TRITON_API_TOKEN``
(or ``plugin_configs.triton.api_token``) is set it is sent as a bearer token on
both surfaces for fronting proxies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from .._openai_compat.local_host import OpenAICompatLocalHostProvider
from ..base import ProviderConfig
from .env import (
    ENV_CONTEXT_LENGTH,
    ENV_HOST,
    ENV_OPENAI_URL,
    _origin_for_host,
    DEFAULT_CONTROL_PORT,
    DEFAULT_OPENAI_PORT,
    resolve_api_token,
    resolve_context_length,
    resolve_urls,
)
from .errors import (
    TritonAuthenticationError,
    TritonConnectionError,
    TritonLoadError,
    TritonMidStreamError,
    TritonModelNotFoundError,
)

logger = logging.getLogger(__name__)


class TritonProvider(OpenAICompatLocalHostProvider):
    """Triton provider driving the OpenAI chat frontend + the KServe v2 control
    surface.  All transport / error / auth machinery is inherited from
    :class:`OpenAICompatLocalHostProvider`."""

    # Parameterize the shared local-host error mapping with Triton's taxonomy.
    _ERR_AUTHENTICATION = TritonAuthenticationError
    _ERR_MODEL_NOT_FOUND = TritonModelNotFoundError
    _ERR_CONNECTION = TritonConnectionError
    _ERR_MIDSTREAM = TritonMidStreamError
    _ERR_LOAD = TritonLoadError

    def __init__(self):
        super().__init__()
        self._openai_url: str = ""
        self._control_url: str = ""
        # Load body passthrough.  None = passive mode (no /load at connect()).
        self._load_config: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        """Provider identifier — used as the key in ``plugin_configs``."""
        return "triton"

    # ============ Connection-identity overrides (dual-URL + surface=) ============

    def _connection_host(self) -> str:
        """Runtime errors carry the chat (OpenAI) URL."""
        return self._openai_url

    def _new_connection_error(self, message: Optional[str] = None):
        if message is None:
            return TritonConnectionError(self._openai_url, surface="openai")
        return TritonConnectionError(self._openai_url, message, surface="openai")

    def _probe_url(self) -> str:
        return f"{self._control_url}/v2/health/live"

    def _verify_connectivity(self, probe_url: str) -> None:
        """Probe the CONTROL surface (load/unload depend on it; surface=control).

        Overrides the base because the probe targets a different URL + surface
        than runtime errors: ``self._control_url`` with ``surface="control"``.
        """
        try:
            httpx.get(
                probe_url, headers=self._auth_headers(), timeout=5,
            ).raise_for_status()
        except httpx.ConnectError:
            raise TritonConnectionError(self._control_url, surface="control")
        except httpx.TimeoutException:
            raise TritonConnectionError(
                self._control_url, "Connection timed out", surface="control")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise TritonAuthenticationError(original_error=str(exc))
            raise TritonConnectionError(
                self._control_url, str(exc), surface="control")
        except httpx.HTTPError as exc:
            raise TritonConnectionError(
                self._control_url, str(exc), surface="control")

    def _resolve_api_token(self) -> Optional[str]:
        return resolve_api_token()

    # ==================== Credential / context hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve the two URLs + optional token + the optional load body."""
        self._resolve_urls_from_config(config)
        self._api_token = config.extra.get("api_token") or resolve_api_token()
        self._base_url = f"{self._openai_url}/v1"
        self._api_key = self._api_token   # base _create_client substitutes a placeholder

        load = config.extra.get("load")
        if load is not None and not isinstance(load, dict):
            raise ValueError(
                f"triton load config must be a dict, got {type(load).__name__}"
            )
        self._load_config = load

        self._auth_info = (
            f"openai={self._openai_url}, control={self._control_url}, bearer"
            if self._api_token
            else f"openai={self._openai_url}, control={self._control_url}"
        )

    def _resolve_urls_from_config(self, config: ProviderConfig) -> None:
        """Pick OpenAI + control URLs from config.extra, falling back to env.

        Precedence: explicit ``openai_url`` / ``control_url`` > the ``host``
        shorthand (with the canonical Triton ports) > env vars.  No localhost
        fallback (fail-loud per the no-fallback rule).
        """
        env_openai, env_control = resolve_urls()

        explicit_openai = config.extra.get("openai_url")
        explicit_control = config.extra.get("control_url")
        host_shorthand = config.extra.get("host")

        if explicit_openai:
            self._openai_url = explicit_openai.rstrip("/")
        elif host_shorthand:
            self._openai_url = _origin_for_host(host_shorthand, DEFAULT_OPENAI_PORT)
        elif env_openai:
            self._openai_url = env_openai
        else:
            raise ValueError(
                f"Triton provider: OpenAI frontend URL is not configured.  "
                f"Set {ENV_OPENAI_URL} (or {ENV_HOST} for a shorthand using the "
                f"canonical port {DEFAULT_OPENAI_PORT}) in the environment, or "
                f"plugin_configs.triton.openai_url / plugin_configs.triton.host "
                f"in the profile.  No hardcoded localhost fallback exists per "
                f"the project's no-fallback rule."
            )

        if explicit_control:
            self._control_url = explicit_control.rstrip("/")
        elif host_shorthand:
            self._control_url = _origin_for_host(host_shorthand, DEFAULT_CONTROL_PORT)
        elif env_control:
            self._control_url = env_control
        else:
            raise ValueError(
                f"Triton provider: control (KServe v2) URL is not configured.  "
                f"Set TRITON_CONTROL_URL (or {ENV_HOST} for a shorthand using "
                f"the canonical port {DEFAULT_CONTROL_PORT}) in the environment, "
                f"or plugin_configs.triton.control_url / plugin_configs.triton."
                f"host in the profile.  No hardcoded localhost fallback exists "
                f"per the project's no-fallback rule."
            )

    def _detect_context(self, config: ProviderConfig) -> Optional[int]:
        """Operator-supplied only — Triton's model config carries no standard
        context-length field (backend-specific: TRT-LLM max_seq_len, etc.)."""
        ctx = config.extra.get("context_length")
        if ctx:
            return int(ctx)
        return resolve_context_length()

    def _context_error_message(self) -> str:
        return (
            f"Triton provider: context_length is not configured.  Triton's "
            f"model config does not carry a standard 'context length' field "
            f"(backend-specific — TRT-LLM has max_seq_len, vLLM has "
            f"max_model_len, etc.), so the framework cannot auto-discover it.  "
            f"Set {ENV_CONTEXT_LENGTH} in the environment, or "
            f"plugin_configs.triton.context_length in the profile.  No "
            f"hardcoded fallback exists per the project's no-fallback rule."
        )

    def get_auth_info(self) -> str:
        """Short human-readable description of auth state."""
        return self._auth_info or f"Triton ({self._openai_url})"

    # ==================== Connection (catalog + load) ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Select the model: validate against the repository, then optionally load.

        Raises:
            TritonModelNotFoundError: Model is not in the repository.
            TritonLoadError: ``/load`` returned a non-2xx.
        """
        if not skip_model_test:
            catalog = self._fetch_repository_index()
            if catalog and model not in {entry["name"] for entry in catalog}:
                raise TritonModelNotFoundError(
                    model, available=[entry["name"] for entry in catalog],
                )
        self._model_name = model
        if self._load_config is not None:
            self._load_model(model, self._load_config)
        logger.info(
            "Connected to Triton model: %s (context=%d, load_applied=%s)",
            model, self.get_context_limit(), bool(self._load_config),
        )

    def _load_model(self, model: str, load_params: Dict[str, Any]) -> None:
        """POST the load body to ``/v2/repository/models/<model>/load``.

        Triton's load is idempotent at the protocol level (the model is already
        on disk; Triton tracks load state), so — unlike LM Studio — we don't
        pre-query for a matching loaded instance.  ``{}`` means "load with the
        stored config".
        """
        url = f"{self._control_url}/v2/repository/models/{model}/load"
        self._trace(f"[LOAD] POST {url} body_keys={sorted(load_params.keys())}")
        try:
            response = httpx.post(
                url, json=load_params or {}, headers=self._auth_headers(),
                timeout=600.0,   # engine deserialization + KV-cache init + GPU xfer
            )
        except httpx.HTTPError as exc:
            raise TritonConnectionError(
                self._control_url, f"load failed: {exc}", surface="control",
            ) from exc
        if response.status_code >= 400:
            raise TritonLoadError(
                model=model, status_code=response.status_code,
                body=response.text, load_config=load_params,
            )
        self._trace(f"[LOAD] status={response.status_code}")

    def _fetch_repository_index(self) -> List[Dict[str, Any]]:
        """Query ``POST /v2/repository/index`` → the model list (a top-level
        array, distinct from OpenAI's ``/v1/models`` shape).  Empty on error."""
        try:
            response = httpx.post(
                f"{self._control_url}/v2/repository/index",
                json={}, headers=self._auth_headers(), timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, list) else []
        except httpx.HTTPError as exc:
            logger.warning("Failed to list Triton models: %s", exc)
            return []

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models in Triton's repository (loaded or not) via the index."""
        names = [entry["name"] for entry in self._fetch_repository_index()]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return sorted(names)

    @staticmethod
    def login(on_message=None) -> None:
        """No-op — Triton does not use interactive auth."""
        if on_message:
            on_message(
                "Triton has no built-in auth. If your deployment is fronted by "
                "an auth proxy, set TRITON_API_TOKEN."
            )


def create_provider() -> TritonProvider:
    """Factory function consumed by the provider discovery machinery."""
    return TritonProvider()
