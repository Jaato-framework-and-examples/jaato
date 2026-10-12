"""The HTTP half of ``OpenRouterProvider.decide()``.

OpenRouter serves decision models (``typesafe/jev-1.13``) on
``POST /api/alpha/decisions``, beside chat completions on
``/api/v1/chat/completions``, with the same Bearer key and attribution
headers.  The body and response are the vendor's (see
``jaato_sdk.plugins.model_provider.decisions``); this module only
locates the endpoint and binds OpenRouter's error types to the shared
transport (``model_provider/_decisions_http.py``), which sends one
request under the provider's deadlines, honours a cancel token and maps
HTTP statuses so ``with_retry`` classifies them:

======  ===========================================  ==========
status  error                                        transient
======  ===========================================  ==========
401     ``AuthenticationError``                      no
404     ``ModelNotFoundError``                       no
429     ``RateLimitError`` (``Retry-After`` read)    yes
5xx     ``InfrastructureError`` (``529`` overloaded) yes
other   ``DecisionRequestRejectedError`` (``422``)   no
======  ===========================================  ==========

``/api/alpha/`` is not a stable route.  Nothing here assumes more of the
response than the parser in the SDK module checks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from jaato_sdk.plugins.model_provider.types import CancelToken

from .. import _decisions_http as _http
from .._decisions_http import DecisionHttpErrors

from .errors import (
    AuthenticationError,
    DecisionRequestRejectedError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)

#: Path of the decision endpoint relative to OpenRouter's ``/api`` root.
DECISIONS_PATH = "/alpha/decisions"


def decisions_url(base_url: str, override: Optional[str] = None) -> str:
    """The decision endpoint for a chat ``base_url``.

    ``https://openrouter.ai/api/v1`` gives
    ``https://openrouter.ai/api/alpha/decisions``: the version segment is
    replaced, because the endpoint sits beside ``v1``, not inside it.  A
    ``base_url`` with no ``/v1`` suffix (a proxy) gets the path appended.
    ``override`` (``framework_overrides.decisions_url``) wins outright.
    """
    if override:
        return override
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root + DECISIONS_PATH


_ERRORS = DecisionHttpErrors(
    authentication=lambda text: AuthenticationError(original_error=text),
    not_found=lambda model, text: ModelNotFoundError(model=model, original_error=text),
    rate_limit=lambda retry_after, text: RateLimitError(
        retry_after=retry_after, original_error=text),
    infrastructure=lambda status, text: InfrastructureError(
        status_code=status, original_error=text),
    rejected=lambda status, model, body: DecisionRequestRejectedError(
        status_code=status, model=model, body=body),
)


def raise_for_decision_status(response: Any, model: str) -> None:
    """Raise the provider error for a non-2xx decision response."""
    _http.raise_for_decision_status(response, model, _ERRORS)


def post_decision(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    *,
    model: str,
    connect_timeout: float,
    request_timeout: float,
    cancel_token: Optional[CancelToken] = None,
) -> Dict[str, Any]:
    """POST one decision request with OpenRouter's error types.

    The transport (worker thread, cancel token, status mapping) is
    ``model_provider/_decisions_http.py``, shared with ``laya``.
    """
    return _http.post_decision(
        url, headers, body, model=model, errors=_ERRORS,
        connect_timeout=connect_timeout, request_timeout=request_timeout,
        cancel_token=cancel_token,
    )
