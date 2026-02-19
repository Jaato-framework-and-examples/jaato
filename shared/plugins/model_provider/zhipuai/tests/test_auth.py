"""Tests for Zhipu AI API key validation."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ..auth import validate_api_key
from ..env import DEFAULT_ZHIPUAI_BASE_URL

# All tests mock _create_validation_client so validate_api_key() uses
# a controlled httpx client rather than making real network requests.
CLIENT_PATCH = "shared.plugins.model_provider.zhipuai.auth._create_validation_client"


def _mock_client(status_code=200):
    """Return a mock httpx client that returns the given status code."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code

    client = MagicMock()
    client.post.return_value = mock_response
    return client


class TestValidateApiKey:
    """Tests for validate_api_key()."""

    def test_uses_proxy_aware_client(self):
        """Must use _create_validation_client() for proxy/CA support."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()

            validate_api_key("test-key")

            mock_create.assert_called_once()
            mock_create.return_value.post.assert_called_once()

    def test_url_includes_v1_prefix(self):
        """Validation request must hit /v1/messages, not /messages."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()

            validate_api_key("test-key")

            call_args = mock_create.return_value.post.call_args
            assert call_args[0][0] == f"{DEFAULT_ZHIPUAI_BASE_URL}/v1/messages"

    def test_custom_base_url(self):
        """Should use custom base URL when provided."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()

            validate_api_key("test-key", base_url="https://custom.api.com")

            call_args = mock_create.return_value.post.call_args
            assert call_args[0][0] == "https://custom.api.com/v1/messages"

    def test_strips_trailing_slash_from_base_url(self):
        """Should strip trailing slash before appending /v1/messages."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()

            validate_api_key("test-key", base_url="https://custom.api.com/")

            call_args = mock_create.return_value.post.call_args
            assert call_args[0][0] == "https://custom.api.com/v1/messages"

    def test_success_returns_true(self):
        """HTTP 200 should return (True, '')."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(200)

            valid, detail = validate_api_key("test-key")
            assert valid is True
            assert detail == ""

    def test_401_returns_auth_error(self):
        """HTTP 401 should return (False, 'authentication_error')."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(401)

            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail == "authentication_error"

    def test_403_returns_auth_error(self):
        """HTTP 403 should return (False, 'authentication_error')."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(403)

            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail == "authentication_error"

    def test_400_returns_true(self):
        """HTTP 400 (bad request) means key was accepted; return True."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(400)

            valid, detail = validate_api_key("test-key")
            assert valid is True

    def test_network_error_returns_detail(self):
        """Connection error should return network_error detail."""
        with patch(CLIENT_PATCH) as mock_create:
            client = MagicMock()
            client.post.side_effect = httpx.ConnectError("Name or service not known")
            mock_create.return_value = client

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("network_error")
            assert "Name or service not known" in detail

    def test_timeout_returns_network_error(self):
        """Timeout should return network_error detail."""
        with patch(CLIENT_PATCH) as mock_create:
            client = MagicMock()
            client.post.side_effect = httpx.TimeoutException("timed out")
            mock_create.return_value = client

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("network_error")

    def test_sends_correct_headers(self):
        """Should send x-api-key and anthropic-version headers."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()

            validate_api_key("my-secret-key")

            call_args = mock_create.return_value.post.call_args
            headers = call_args[1]["headers"]
            assert headers["x-api-key"] == "my-secret-key"
            assert headers["anthropic-version"] == "2023-06-01"

    def test_ssl_error_returns_network_error(self):
        """SSL handshake failure should return network_error detail."""
        with patch(CLIENT_PATCH) as mock_create:
            client = MagicMock()
            client.post.side_effect = httpx.ConnectError(
                "[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] sslv3 alert handshake failure"
            )
            mock_create.return_value = client

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("network_error")
            assert "SSL" in detail
