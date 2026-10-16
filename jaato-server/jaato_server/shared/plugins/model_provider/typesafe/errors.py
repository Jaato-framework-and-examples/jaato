"""Errors raised by the TypeSafe decision provider.

HTTP statuses map onto these through ``model_provider/_decisions_http.py``.
``TypeSafeProvider.classify_error`` reads ``RateLimitError`` and
``InfrastructureError`` as transient, so ``with_retry`` retries a ``429``,
a ``5xx`` (``529`` included) and an unreachable host, and nothing else.
"""

from typing import Optional

KEYS_URL = "https://console.typesafe.ai/keys"


class TypeSafeError(Exception):
    """Base class for TypeSafe provider errors."""


class CredentialsNotFoundError(TypeSafeError):
    """No key in any of the four places the provider looks."""

    def __init__(self) -> None:
        super().__init__(
            "No TypeSafe API key found.  Looked in, in order: "
            "plugin_configs.typesafe.api_key, JAATO_TYPESAFE_API_KEY, "
            "TYPESAFE_API_KEY, and the key stored by 'typesafe-auth key'.  "
            f"Keys are issued at {KEYS_URL}."
        )


class AuthenticationError(TypeSafeError):
    """TypeSafe refused the key (``401``)."""

    def __init__(self, original_error: Optional[str] = None,
                 source: Optional[str] = None):
        self.original_error = original_error
        self.source = source
        super().__init__(
            "TypeSafe refused the API key (HTTP 401)"
            + (f"; it came from {source}" if source else "")
            + f".  Check it at {KEYS_URL}, or store a new one with "
            "'typesafe-auth key <key>'."
            + (f"\nResponse: {original_error[:500]}" if original_error else "")
        )


class ModelNotFoundError(TypeSafeError):
    """No such model or endpoint (``404``)."""

    def __init__(self, model: str, original_error: Optional[str] = None):
        self.model = model
        self.original_error = original_error
        super().__init__(
            f"TypeSafe answered 404 for {model!r}: no such model, or no "
            "/v1/systemone at the configured base_url."
            + (f"\nResponse: {original_error[:500]}" if original_error else "")
        )


class RateLimitError(TypeSafeError):
    """``429``.  Transient."""

    def __init__(self, retry_after: Optional[float] = None,
                 original_error: Optional[str] = None):
        self.retry_after = retry_after
        self.original_error = original_error
        super().__init__(
            "TypeSafe is rate limiting (HTTP 429)."
            + (f" Retry after {retry_after:g}s." if retry_after else "")
        )


class InfrastructureError(TypeSafeError):
    """A ``5xx``, an unreachable host (status ``0``) or a body that is not
    JSON.  Transient."""

    def __init__(self, status_code: int, original_error: Optional[str] = None):
        self.status_code = status_code
        self.original_error = original_error
        what = ("could not be reached" if status_code == 0
                else f"failed (HTTP {status_code})")
        super().__init__(
            f"The TypeSafe API {what}."
            + (f"\n{original_error[:1000]}" if original_error else "")
        )


class DecisionRequestRejectedError(TypeSafeError):
    """TypeSafe refused the request (a ``4xx`` other than 401/404/429).

    ``422`` is a malformed question and ``400`` an unreferenced asset, the
    body naming it.  Not transient.
    """

    def __init__(self, status_code: int, model: str, body: Optional[str] = None):
        self.status_code = status_code
        self.model = model
        self.body = body
        super().__init__(
            f"TypeSafe refused the request for {model!r} (HTTP {status_code})."
            + (f"\nResponse: {body[:2000]}" if body else "")
        )
