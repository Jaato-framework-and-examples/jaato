"""Shared base for self-hosted OpenAI-compatible providers (local-host family).

``OpenAICompatLocalHostProvider`` sits between :class:`OpenAICompatProvider`
(the transport machinery) and the providers that talk to a locally-launched
OpenAI-compatible server the operator runs themselves — vllm, lmstudio,
tensorrt_llm, triton.  Those share:

- a **credential / connection model**: a host URL + optional bearer token, and a
  liveness PROBE at the end of ``initialize()`` (``GET /health`` or equivalent);
- an **error taxonomy** distinct from the hosted nim-family —
  ``*ConnectionError`` / ``*ModelNotFoundError`` / ``*AuthenticationError`` +
  ``*MidStreamError`` / ``*LoadError``.

This base owns those: a parameterized ``_handle_api_error`` / ``classify_error``
/ ``get_retry_after``, the ``_verify_connectivity`` probe, ``_auth_headers``,
``verify_auth``, ``get_auth_info``.  Subclasses supply the env module (host /
token resolution → ``_resolve_credentials``), the probe URL, and context
resolution.

Error classes are named via class attrs:

- ``_ERR_AUTHENTICATION`` / ``_ERR_MODEL_NOT_FOUND`` / ``_ERR_CONNECTION`` — required.
- ``_ERR_MIDSTREAM`` — ``None`` disables the mid-stream / pre-flight split
  (e.g. lmstudio).
- ``_ERR_LOAD`` — ``None`` unless the provider has a model-load step
  (lmstudio / triton), in which case it is classified non-retryable.

Two small hooks handle the non-uniform ``*ConnectionError`` constructors:
``_connection_host()`` (the host string carried in the error; triton uses its
chat URL) and ``_new_connection_error()`` (triton injects ``surface=``).
"""

from __future__ import annotations

from typing import Dict, Optional

import httpx

from ._lazy import get_openai_module
from .base import OpenAICompatProvider
from ..base import ProviderConfig


class OpenAICompatLocalHostProvider(OpenAICompatProvider):
    """Base for self-hosted OpenAI-compatible providers.  See module docstring."""

    # Local-host error taxonomy (subclasses set these; _ERR_AUTHENTICATION and
    # _ERR_MODEL_NOT_FOUND are inherited names from OpenAICompatProvider).
    _ERR_CONNECTION: type
    _ERR_MIDSTREAM = None        # None -> no mid-stream / pre-flight split
    _ERR_LOAD = None             # None -> no load-error classify branch

    def __init__(self) -> None:
        super().__init__()
        self._host: str = ""
        self._api_token: Optional[str] = None
        self._auth_info: str = ""

    # ---- connection-identity hooks (triton overrides for its dual-URL) ----

    def _connection_host(self) -> str:
        """Host string carried in connection errors (default: ``self._host``)."""
        return self._host

    def _new_connection_error(self, message: Optional[str] = None):
        """Construct the provider's ConnectionError (override for odd ctors)."""
        if message is None:
            return self._ERR_CONNECTION(self._connection_host())
        return self._ERR_CONNECTION(self._connection_host(), message)

    def _resolve_api_token(self) -> Optional[str]:
        """Hook: the provider's env-resolved bearer token (for verify_auth)."""
        return None

    def _probe_url(self) -> str:
        """Hook: the liveness-probe URL hit at the end of ``initialize()``."""
        raise NotImplementedError

    # ---- shared auth + connectivity ----

    def _auth_headers(self) -> Dict[str, str]:
        """Auth headers for direct ``httpx`` calls (probe / catalog)."""
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
        return headers

    def _verify_connectivity(self, probe_url: str) -> None:
        """Confirm the server is reachable; raise the provider's conn/auth error."""
        try:
            httpx.get(
                probe_url, headers=self._auth_headers(), timeout=5,
            ).raise_for_status()
        except httpx.ConnectError:
            raise self._new_connection_error()
        except httpx.TimeoutException:
            raise self._new_connection_error("Connection timed out")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise self._ERR_AUTHENTICATION(original_error=str(exc))
            raise self._new_connection_error(str(exc))
        except httpx.HTTPError as exc:
            raise self._new_connection_error(str(exc))

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Base init (credentials, api_params, context, client) + liveness probe."""
        super().initialize(config)
        self._verify_connectivity(self._probe_url())

    def verify_auth(
        self, allow_interactive: bool = False, on_message=None, config=None,
    ) -> bool:
        """Credentials-only check — self-hosted servers usually have no auth.

        Reports an optional bearer token for operator visibility; reachability
        is validated at ``initialize()``.  Always returns True (nothing to
        gate before connecting).
        """
        token = config.extra.get("api_token") if (config and config.extra) else None
        if not token:
            token = self._resolve_api_token()
        if on_message:
            if token:
                masked = f"{token[:4]}…{token[-4:]}" if len(token) > 8 else "***"
                on_message(f"{self.name} bearer token configured ({masked})")
            else:
                on_message(
                    f"{self.name}: no authentication required "
                    "(reachability validated at session start)")
        return True

    def get_auth_info(self) -> str:
        """Short description of auth state (set during credential resolution)."""
        return self._auth_info or f"{self.name} (local)"

    # ---- error mapping (local-host taxonomy) ----

    def _handle_api_error(self, error: Exception) -> None:
        """Map OpenAI SDK exceptions to the local-host error taxonomy.

        When ``_ERR_MIDSTREAM`` is set, distinguishes mid-stream connection
        drops (engine error after HTTP 200 was committed — phrasings like
        "peer closed connection" / "incomplete chunked") from pre-flight
        failures, so the reliability layer + the operator get the right
        fix-tree (server-side engine config vs network/host).
        """
        openai = get_openai_module()
        if isinstance(error, openai.AuthenticationError):
            raise self._ERR_AUTHENTICATION(original_error=str(error)) from error
        if isinstance(error, openai.NotFoundError):
            raise self._ERR_MODEL_NOT_FOUND(
                model=self._model_name or "unknown") from error
        if isinstance(error, openai.APIConnectionError):
            msg = str(error).lower()
            mid = (
                "peer closed connection" in msg
                or "incomplete chunked" in msg
                or "remoteprotocolerror" in msg
            )
            if mid and self._ERR_MIDSTREAM is not None:
                raise self._ERR_MIDSTREAM(
                    self._connection_host(), original_error=str(error)) from error
            raise self._new_connection_error(str(error)) from error
        # Other APIStatusError subclasses fall through — caller sees original.

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Connection = transient/infra; load = permanent.  Others -> global."""
        if isinstance(exc, self._ERR_CONNECTION):
            return {"transient": True, "rate_limit": False, "infra": True}
        if self._ERR_LOAD is not None and isinstance(exc, self._ERR_LOAD):
            return {"transient": False, "rate_limit": False, "infra": False}
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Self-hosted servers don't emit retry-after hints."""
        return None
