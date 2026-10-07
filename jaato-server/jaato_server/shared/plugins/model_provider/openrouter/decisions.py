"""The HTTP half of ``OpenRouterProvider.decide()``.

OpenRouter serves decision models (``typesafe/jev-1.13``) on
``POST /api/alpha/decisions``, beside chat completions on
``/api/v1/chat/completions``, with the same Bearer key and attribution
headers.  The body and response are the vendor's (see
``jaato_sdk.plugins.model_provider.decisions``); this module only
locates the endpoint, sends one request under the provider's deadlines,
honours a cancel token, and maps HTTP statuses onto the provider's
error types so ``with_retry`` classifies them:

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

import concurrent.futures
from typing import Any, Dict, Optional

from jaato_sdk.plugins.model_provider.types import CancelledException, CancelToken

from .errors import (
    AuthenticationError,
    DecisionRequestRejectedError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)

#: Path of the decision endpoint relative to OpenRouter's ``/api`` root.
DECISIONS_PATH = "/alpha/decisions"

#: How often a waiting call looks at its cancel token, seconds.
CANCEL_POLL_SECONDS = 0.05


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


def _retry_after(response: Any) -> Optional[float]:
    raw = response.headers.get("retry-after")
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def raise_for_decision_status(response: Any, model: str) -> None:
    """Raise the provider error for a non-2xx decision response."""
    status = response.status_code
    if status < 400:
        return
    text = response.text
    if status == 401:
        raise AuthenticationError(original_error=text)
    if status == 404:
        raise ModelNotFoundError(model=model, original_error=text)
    if status == 429:
        raise RateLimitError(retry_after=_retry_after(response), original_error=text)
    if status >= 500:
        raise InfrastructureError(status_code=status, original_error=text)
    raise DecisionRequestRejectedError(status_code=status, model=model, body=text)


def _send(client: Any, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Any:
    import httpx

    try:
        return client.post(url, headers=headers, json=body)
    except httpx.TransportError as exc:   # connect, read, write, timeouts
        raise InfrastructureError(status_code=0, original_error=str(exc)) from exc


def _wait(future: Any, client: Any, cancel_token: Optional[CancelToken]) -> Any:
    while True:
        try:
            return future.result(timeout=CANCEL_POLL_SECONDS)
        except concurrent.futures.TimeoutError:
            if cancel_token is not None and cancel_token.is_cancelled:
                client.close()   # aborts the in-flight request
                raise CancelledException("decision request cancelled")


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
    """POST one decision request and return the parsed JSON body.

    The request runs on a worker thread so the caller can stop waiting
    the moment ``cancel_token`` trips; the client is then closed, which
    aborts the request.  A ``0`` deadline means none, as on the chat path.
    """
    import httpx

    if cancel_token is not None:
        cancel_token.raise_if_cancelled()
    timeout = httpx.Timeout(request_timeout or None, connect=connect_timeout or None)
    client = httpx.Client(timeout=timeout)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_send, client, url, headers, body)
        response = _wait(future, client, cancel_token)
        raise_for_decision_status(response, model)
        try:
            return response.json()
        except ValueError as exc:
            raise InfrastructureError(
                status_code=response.status_code,
                original_error=f"decision response is not JSON: {response.text[:500]}",
            ) from exc
    finally:
        pool.shutdown(wait=False)
        client.close()
