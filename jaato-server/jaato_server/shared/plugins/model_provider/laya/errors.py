"""Errors raised by the Laya decision provider.

The HTTP statuses map onto these through
``model_provider/_decisions_http.py``.  ``LayaProvider.classify_error``
reads ``RateLimitError`` and ``InfrastructureError`` as transient, so
``with_retry`` retries a ``429``, a ``5xx`` and an unreachable server,
and nothing else.
"""

from typing import Optional


class LayaError(Exception):
    """Base class for Laya provider errors."""


class AuthenticationError(LayaError):
    """``laya-serve`` refused the bearer token (``401``)."""

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        super().__init__(
            "laya-serve refused the request (HTTP 401).  The server was "
            "started with LAYA_API_KEY; set plugin_configs.laya.api_key or "
            "JAATO_LAYA_API_KEY to the same value."
            + (f"\nResponse: {original_error[:500]}" if original_error else "")
        )


class ModelNotFoundError(LayaError):
    """The server has no such endpoint or checkpoint (``404``)."""

    def __init__(self, model: str, original_error: Optional[str] = None):
        self.model = model
        self.original_error = original_error
        super().__init__(
            f"laya-serve answered 404 for {model!r}: no /v1/systemone at the "
            "configured host, or no such checkpoint."
            + (f"\nResponse: {original_error[:500]}" if original_error else "")
        )


class RateLimitError(LayaError):
    """``429`` from the server or a proxy in front of it.  Transient."""

    def __init__(self, retry_after: Optional[float] = None,
                 original_error: Optional[str] = None):
        self.retry_after = retry_after
        self.original_error = original_error
        super().__init__(
            "laya-serve is rate limiting (HTTP 429)."
            + (f" Retry after {retry_after:g}s." if retry_after else "")
        )


class InfrastructureError(LayaError):
    """A ``5xx``, an unreachable server (status ``0``) or a body that is
    not JSON.  Transient."""

    def __init__(self, status_code: int, original_error: Optional[str] = None):
        self.status_code = status_code
        self.original_error = original_error
        what = ("could not be reached" if status_code == 0
                else f"failed (HTTP {status_code})")
        super().__init__(
            f"laya-serve {what}."
            + (f"\n{original_error[:1000]}" if original_error else "")
        )


class DecisionRequestRejectedError(LayaError):
    """The server refused the request (a ``4xx`` other than 401/404/429).

    ``422`` is a malformed question, the body naming it.  Not transient.
    """

    def __init__(self, status_code: int, model: str, body: Optional[str] = None):
        self.status_code = status_code
        self.model = model
        self.body = body
        super().__init__(
            f"laya-serve refused the request for {model!r} (HTTP {status_code})."
            + (f"\nResponse: {body[:2000]}" if body else "")
        )


class LayaNotInstalledError(LayaError):
    """``mode: local`` and the ``laya`` package is not importable."""

    def __init__(self, cause: str):
        super().__init__(
            "plugin_configs.laya.mode is 'local' but the laya package is not "
            f"importable ({cause}).  Install it in the runner's environment "
            "(pip install laya), or run laya-serve and use mode 'http'."
        )
