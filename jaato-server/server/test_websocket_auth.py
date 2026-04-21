"""Tests for WebSocket bearer-token authentication.

Mainline jaato historically accepted every WS connection unconditionally
(``set_client_user`` existed as a hook for the jaato-premium SSO
extension but mainline never enforced anything). When the daemon is
exposed beyond a trusted LAN this is unsafe — anyone reaching the port
can open sessions and execute tools.

These tests pin down the bearer-token gate added to
``JaatoWSServer._handle_client``: token presented in either an
``Authorization: Bearer ...`` header (Python clients, curl, proxies) or
a ``?token=...`` query parameter (browsers, which can't set custom
headers from the WebSocket constructor) is validated against a
configured digest with :func:`hmac.compare_digest`. No token, wrong
token, or wrong scheme → reject.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from server.websocket import HAS_WEBSOCKETS, JaatoWSServer

pytestmark = pytest.mark.skipif(
    not HAS_WEBSOCKETS, reason="websockets package not installed"
)


def _fake_ws(*, headers: Optional[dict] = None, path: str = "/") -> MagicMock:
    """Build the minimum surface ``_check_ws_token`` reads from.

    The real ``websockets.ServerConnection`` exposes ``request.headers``
    (a ``Headers`` mapping) and ``request.path`` (URI path with query
    string). Mocking it directly avoids spinning up a server socket.
    """
    request = MagicMock()
    request.headers = headers or {}
    request.path = path
    ws = MagicMock()
    ws.request = request
    return ws


def _server(token: Optional[str]) -> JaatoWSServer:
    """Construct a server instance without binding to a port.

    ``__init__`` is the only thing under test; ``start()`` is never
    called so no socket is opened.
    """
    return JaatoWSServer(host="127.0.0.1", port=0, required_token=token)


# ---------------------------------------------------------------------------
# Auth disabled (legacy behaviour)
# ---------------------------------------------------------------------------

class TestAuthDisabled:

    def test_no_token_configured_accepts_any_connection(self):
        srv = _server(token=None)
        ws = _fake_ws()  # no auth at all
        assert srv._check_ws_token(ws) is True

    def test_no_token_configured_ignores_bogus_header(self):
        srv = _server(token=None)
        ws = _fake_ws(headers={"Authorization": "Bearer whatever"})
        assert srv._check_ws_token(ws) is True


# ---------------------------------------------------------------------------
# Authorization header path
# ---------------------------------------------------------------------------

class TestBearerHeader:

    def test_correct_token_in_header_accepts(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(headers={"Authorization": "Bearer secret-xyz"})
        assert srv._check_ws_token(ws) is True

    def test_wrong_token_in_header_rejects(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(headers={"Authorization": "Bearer wrong-token"})
        assert srv._check_ws_token(ws) is False

    def test_missing_authorization_header_rejects(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(headers={})
        assert srv._check_ws_token(ws) is False

    def test_non_bearer_scheme_rejects(self):
        """``Basic ...`` and ``Token ...`` are valid Authorization
        schemes for other systems; we only honour ``Bearer``."""
        srv = _server(token="secret-xyz")
        ws = _fake_ws(headers={"Authorization": "Basic c2VjcmV0LXh5eg=="})
        assert srv._check_ws_token(ws) is False

    def test_bearer_prefix_with_empty_token_rejects(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(headers={"Authorization": "Bearer "})
        assert srv._check_ws_token(ws) is False

    def test_bearer_strips_surrounding_whitespace(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(headers={"Authorization": "Bearer   secret-xyz   "})
        assert srv._check_ws_token(ws) is True


# ---------------------------------------------------------------------------
# Query parameter path (browser-compat)
# ---------------------------------------------------------------------------

class TestQueryParam:

    def test_correct_token_in_query_accepts(self):
        """Browsers cannot set the Authorization header on
        ``new WebSocket(url, ...)`` — they pass the token via the URL."""
        srv = _server(token="secret-xyz")
        ws = _fake_ws(path="/?token=secret-xyz")
        assert srv._check_ws_token(ws) is True

    def test_wrong_token_in_query_rejects(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(path="/?token=wrong")
        assert srv._check_ws_token(ws) is False

    def test_query_token_with_other_params(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(path="/?foo=bar&token=secret-xyz&baz=1")
        assert srv._check_ws_token(ws) is True

    def test_empty_query_token_rejects(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(path="/?token=")
        assert srv._check_ws_token(ws) is False

    def test_path_without_query_rejects(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(path="/some/path")
        assert srv._check_ws_token(ws) is False


# ---------------------------------------------------------------------------
# Header takes precedence over query
# ---------------------------------------------------------------------------

class TestHeaderPrecedence:

    def test_header_wins_when_both_present_and_header_correct(self):
        srv = _server(token="secret-xyz")
        ws = _fake_ws(
            headers={"Authorization": "Bearer secret-xyz"},
            path="/?token=garbage",
        )
        assert srv._check_ws_token(ws) is True

    def test_header_wins_when_both_present_and_header_wrong(self):
        """Header check happens first and short-circuits — even when a
        valid token sits in the query string. This keeps the precedence
        rule trivially predictable."""
        srv = _server(token="secret-xyz")
        ws = _fake_ws(
            headers={"Authorization": "Bearer wrong"},
            path="/?token=secret-xyz",
        )
        assert srv._check_ws_token(ws) is False


# ---------------------------------------------------------------------------
# Digest storage
# ---------------------------------------------------------------------------

class TestTokenStorage:

    def test_plaintext_token_not_retained_on_instance(self):
        """The constructor should hash the token and discard the
        plaintext, so a memory dump or a debug-print of the instance
        doesn't leak the secret."""
        token = "highly-secret"
        srv = _server(token=token)
        # Walk every instance attribute looking for the plaintext.
        for name, value in vars(srv).items():
            assert token not in repr(value), (
                f"plaintext token leaked into JaatoWSServer.{name!r}"
            )
