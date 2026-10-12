"""One decision request over HTTP, for every provider that serves ``decide()``.

A decision endpoint takes the vendor's body (see
``jaato_sdk.plugins.model_provider.decisions``) and answers with one JSON
document: TypeSafe's ``/v1/systemone``, OpenRouter's
``/api/alpha/decisions`` and ``laya-serve``'s ``/v1/systemone`` all speak
the same shape.  This module sends one request under the caller's
deadlines, honours a cancel token, and maps HTTP statuses onto the
CALLING provider's error types, so ``with_retry`` classifies them with
that provider's ``classify_error``:

======  =================================  ==========
status  error (``DecisionHttpErrors``)     transient
======  =================================  ==========
401     ``authentication``                 no
404     ``not_found``                      no
429     ``rate_limit`` (``Retry-After``)   yes
5xx     ``infrastructure`` (``529``)       yes
other   ``rejected`` (``422``)             no
======  =================================  ==========

A transport failure (connect refused, a timeout) is ``infrastructure``
with status ``0``.  A body that is not JSON is ``infrastructure`` with
the status the server sent.  Nothing here assumes more of the response
than the SDK parser checks.

It is a plain module, imported rather than inherited, for the reason
``_media_deltas.py`` is: the providers that serve decisions share no base
class.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from jaato_sdk.plugins.model_provider.types import CancelledException, CancelToken

#: How often a waiting call looks at its cancel token, seconds.
CANCEL_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class DecisionHttpErrors:
    """The exceptions one provider raises for each failure class.

    Attributes:
        authentication: ``(text) -> exc``, for a ``401``.
        not_found: ``(model, text) -> exc``, for a ``404``.
        rate_limit: ``(retry_after, text) -> exc``, for a ``429``.
        infrastructure: ``(status, text) -> exc``, for ``5xx``, a
            transport failure (status ``0``) or a body that is not JSON.
        rejected: ``(status, model, body) -> exc``, for any other ``4xx``.
    """

    authentication: Callable[[str], Exception]
    not_found: Callable[[str, str], Exception]
    rate_limit: Callable[[Optional[float], str], Exception]
    infrastructure: Callable[[int, str], Exception]
    rejected: Callable[[int, str, str], Exception]


def _retry_after(response: Any) -> Optional[float]:
    raw = response.headers.get("retry-after")
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def raise_for_decision_status(
    response: Any, model: str, errors: DecisionHttpErrors,
) -> None:
    """Raise the provider's error for a non-2xx decision response."""
    status = response.status_code
    if status < 400:
        return
    text = response.text
    if status == 401:
        raise errors.authentication(text)
    if status == 404:
        raise errors.not_found(model, text)
    if status == 429:
        raise errors.rate_limit(_retry_after(response), text)
    if status >= 500:
        raise errors.infrastructure(status, text)
    raise errors.rejected(status, model, text)


def _send(client: Any, url: str, headers: Dict[str, str], body: Dict[str, Any],
          errors: DecisionHttpErrors) -> Any:
    import httpx

    try:
        return client.post(url, headers=headers, json=body)
    except httpx.TransportError as exc:   # connect, read, write, timeouts
        raise errors.infrastructure(0, f"{url}: {exc}") from exc


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
    errors: DecisionHttpErrors,
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
        future = pool.submit(_send, client, url, headers, body, errors)
        response = _wait(future, client, cancel_token)
        raise_for_decision_status(response, model, errors)
        try:
            return response.json()
        except ValueError as exc:
            raise errors.infrastructure(
                response.status_code,
                f"decision response is not JSON: {response.text[:500]}",
            ) from exc
    finally:
        pool.shutdown(wait=False)
        client.close()
