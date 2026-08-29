"""A connection error must not poison the provider for the session's life.

``openai.APIConnectionError`` can leave the underlying ``httpx`` transport
permanently unusable.  ``with_retry`` retries the CALL, not the client, so
without a rebuild every backoff re-runs against the same dead object: the
ladder exhausts, and the session keeps failing after the network recovers
while still telling the operator "This is a transient error. The request
will be automatically retried."

Measured (jaato #705): a session died at 19:57 and still failed at 20:01
while ``curl`` reached OpenRouter in 30ms, a direct provider call in
another process succeeded, and a NEW session in the SAME daemon completed
normally.  The only difference was a freshly built client.  A second
session was lost the same way an hour later, which is why this is fixed
rather than documented.
"""

import openai
import pytest

from shared.plugins.model_provider.openrouter.provider import OpenRouterProvider
from shared.plugins.model_provider._openai_compat.base import OpenAICompatProvider


class _DeadClient:
    """Stands in for a client whose transport has already failed."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _UnclosableClient(_DeadClient):
    """A transport so broken that even closing it raises."""

    def close(self) -> None:
        raise OSError("transport already torn down")


def _providers():
    """One instance of each independent implementation.

    They are separate class hierarchies with their own ``_create_client``
    and ``_handle_api_error``; ``OpenAICompatProvider`` is the base for
    nim / nebius / ovhcloud / lmstudio / vllm / tensorrt_llm / zhipuai /
    doubleword, so a fix in only one of the two leaves most providers
    poisoned.
    """
    return [OpenRouterProvider(), OpenAICompatProvider()]


@pytest.mark.parametrize("provider", _providers(), ids=lambda p: type(p).__name__)
def test_connection_error_replaces_the_client(provider) -> None:
    """The dead client must be gone before the error propagates."""
    dead = _DeadClient()
    fresh = object()
    provider._client = dead
    provider._create_client = lambda: fresh          # type: ignore[method-assign]

    with pytest.raises(Exception) as caught:
        provider._handle_api_error(
            openai.APIConnectionError(request=None)  # type: ignore[arg-type]
        )

    assert provider._client is fresh, (
        "the poisoned client survived: every remaining retry would run "
        "against the same dead transport and be guaranteed to fail"
    )
    assert dead.closed, "the old client was leaked rather than closed"
    assert caught.value is not None, "the error must still reach the caller"


@pytest.mark.parametrize("provider", _providers(), ids=lambda p: type(p).__name__)
def test_rebuild_failure_does_not_mask_the_original_error(provider) -> None:
    """A failed rebuild must not replace the caller's error with its own.

    The connection error is what the caller needs in order to classify and
    retry.  Swallowing it in favour of a rebuild failure would turn a
    recognised transient into an unrecognised one.
    """
    def _explode():
        raise RuntimeError("cannot build a client right now")

    provider._client = _UnclosableClient()
    provider._create_client = _explode               # type: ignore[method-assign]

    with pytest.raises(Exception) as caught:
        provider._handle_api_error(
            openai.APIConnectionError(request=None)  # type: ignore[arg-type]
        )

    assert "cannot build a client right now" not in str(caught.value), (
        "the rebuild failure masked the connection error the caller must see"
    )


@pytest.mark.parametrize("provider", _providers(), ids=lambda p: type(p).__name__)
def test_non_connection_errors_keep_their_client(provider) -> None:
    """Only connection errors poison a transport.

    Rebuilding on every failure would throw away a healthy connection pool
    on an ordinary 500 or a rate limit, turning a cheap retry into a new
    TLS handshake each time.
    """
    keep = _DeadClient()
    provider._client = keep
    provider._create_client = lambda: object()       # type: ignore[method-assign]

    with pytest.raises(Exception):
        provider._handle_api_error(
            openai.InternalServerError(
                "boom", response=None, body=None      # type: ignore[arg-type]
            )
        )

    assert provider._client is keep, (
        "a server-side 500 does not break the transport; rebuilding would "
        "discard a healthy connection pool"
    )
    assert not keep.closed
