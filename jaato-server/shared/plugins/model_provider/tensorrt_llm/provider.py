"""TensorRT-LLM provider using ``trtllm-serve``'s OpenAI-compatible API.

``trtllm-serve`` is NVIDIA's HTTP front-end for TensorRT-LLM engines, with a
plain OpenAI surface (``POST /v1/chat/completions``, ``GET /v1/models``,
``GET /health``).  This provider is a thin subclass of
:class:`OpenAICompatLocalHostProvider` — the shared self-hosted machinery
(streaming/completion, error mapping, the connectivity probe, auth helpers).
Only trtllm specifics live here: identity, the error-class parameterization,
host/token + context resolution, the ``/health`` probe URL, and the
single-engine catalog validation at ``connect()``.

The provider is **passive**: it talks to a ``trtllm-serve`` instance the
operator has already launched; engine build + per-engine knobs live at the
server-launch boundary.  Auth is optional — when ``TENSORRT_LLM_API_TOKEN``
(or ``plugin_configs.tensorrt_llm.api_token``) is set it is sent as a bearer
token for fronting auth proxies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from .._openai_compat.local_host import OpenAICompatLocalHostProvider
from ..base import ProviderConfig, resolve_context_window
from .env import (
    ENV_CONTEXT_LENGTH,
    ENV_HOST,
    resolve_api_token,
    resolve_context_length,
    resolve_host,
)
from .errors import (
    TensorRTLLMAuthenticationError,
    TensorRTLLMConnectionError,
    TensorRTLLMMidStreamError,
    TensorRTLLMModelNotFoundError,
)

logger = logging.getLogger(__name__)


class TensorRTLLMProvider(OpenAICompatLocalHostProvider):
    """TensorRT-LLM provider talking to ``trtllm-serve``'s /v1 endpoint.

    All transport / error / connectivity machinery is inherited from
    :class:`OpenAICompatLocalHostProvider`.

    Lifecycle: ``initialize()`` resolves host/token/context, creates the client,
    and probes ``/health``; ``connect()`` validates the requested model against
    the single-engine ``/v1/models`` catalog.
    """

    # Parameterize the shared local-host error mapping with trtllm's taxonomy.
    _ERR_AUTHENTICATION = TensorRTLLMAuthenticationError
    _ERR_MODEL_NOT_FOUND = TensorRTLLMModelNotFoundError
    _ERR_CONNECTION = TensorRTLLMConnectionError
    _ERR_MIDSTREAM = TensorRTLLMMidStreamError
    # _ERR_LOAD stays None — trtllm-serve has no in-band load endpoint.

    @property
    def name(self) -> str:
        """Provider identifier — used as the key in ``plugin_configs``."""
        return "tensorrt_llm"

    # ==================== Credential / context hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve host + optional bearer token; no fail-loud localhost default."""
        host = config.extra.get("host") or resolve_host()
        if not host:
            raise ValueError(
                f"TensorRT-LLM provider: host is not configured.  Set "
                f"{ENV_HOST} in the environment, or "
                f"plugin_configs.tensorrt_llm.host in the profile.  No "
                f"hardcoded localhost fallback exists per the project's "
                f"no-fallback rule."
            )
        self._host = host.rstrip("/")
        self._api_token = config.extra.get("api_token") or resolve_api_token()
        self._base_url = f"{self._host}/v1"
        self._api_key = self._api_token   # base _create_client substitutes a placeholder
        self._auth_info = (
            f"local ({self._host}, bearer)" if self._api_token
            else f"local ({self._host})"
        )

    def _read_api_params(self, config: ProviderConfig) -> None:
        """Base api_params + the trtllm ``max_tokens`` knob.

        ``max_tokens`` caps the per-request output budget (forwarded as the
        OpenAI field).  trtllm exposes it as a top-level profile knob
        (``plugin_configs.tensorrt_llm.max_tokens``); fold it into the forwarded
        api_params so the inherited ``complete()`` sends it.  Required for large
        cascade prompts where max_seq_len minus prompt would leave the
        default-unbounded generation room to exhaust the KV-cache (mid-stream
        ``peer closed connection``).
        """
        super()._read_api_params(config)
        max_tokens = config.extra.get("max_tokens")
        if max_tokens is not None:
            self._api_params["max_tokens"] = int(max_tokens)

    def _detect_context(self, config: ProviderConfig) -> Optional[int]:
        """No auto-detect tier — trtllm-serve's /v1/models omits max_seq_len."""
        return resolve_context_window(
            detect_capacity=None,
            profile_value=config.extra.get("context_length"),
            env_value=resolve_context_length(),
        )

    def _context_error_message(self) -> str:
        return (
            f"TensorRT-LLM provider: context_length is not configured.  "
            f"trtllm-serve's GET /v1/models does not surface the engine's "
            f"max_seq_len, so the framework cannot auto-discover it.  Set "
            f"{ENV_CONTEXT_LENGTH} in the environment, or "
            f"plugin_configs.tensorrt_llm.context_length in the profile.  No "
            f"hardcoded fallback exists per the project's no-fallback rule."
        )

    def _resolve_api_token(self) -> Optional[str]:
        return resolve_api_token()

    def _probe_url(self) -> str:
        return f"{self._host}/health"

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Select the model, validating it against the single-engine catalog.

        A ``trtllm-serve`` process exposes exactly one engine, so ``/v1/models``
        carries one entry — the name passed to ``--model``.

        Raises:
            TensorRTLLMModelNotFoundError: Server is hosting a different engine.
        """
        if not skip_model_test:
            catalog = self._fetch_catalog()
            if catalog and model not in {entry["id"] for entry in catalog}:
                raise TensorRTLLMModelNotFoundError(
                    model, available=[entry["id"] for entry in catalog],
                )
        self._model_name = model
        logger.info(
            "Connected to trtllm-serve model: %s (context=%d)",
            model, self.get_context_limit(),
        )

    def _fetch_catalog(self) -> List[Dict[str, Any]]:
        """Query ``GET /v1/models`` → the raw ``data`` array (empty on error)."""
        try:
            response = httpx.get(
                f"{self._host}/v1/models", headers=self._auth_headers(), timeout=10,
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except httpx.HTTPError as exc:
            logger.warning("Failed to list trtllm-serve models: %s", exc)
            return []

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List models served by this trtllm-serve instance (usually one)."""
        names = [entry["id"] for entry in self._fetch_catalog()]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return sorted(names)

    @staticmethod
    def login(on_message=None) -> None:
        """No-op — TensorRT-LLM does not use interactive auth."""
        if on_message:
            on_message(
                "trtllm-serve has no built-in auth. If your deployment is "
                "fronted by an auth proxy, set TENSORRT_LLM_API_TOKEN."
            )


def create_provider() -> TensorRTLLMProvider:
    """Factory function consumed by the provider discovery machinery."""
    return TensorRTLLMProvider()
