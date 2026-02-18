"""Tests for Zhipu AI API key validation."""

import json
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from ..auth import validate_api_key
from ..env import DEFAULT_ZHIPUAI_BASE_URL


class TestValidateApiKey:
    """Tests for validate_api_key()."""

    def test_url_includes_v1_prefix(self):
        """Validation request must hit /v1/messages, not /messages."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            validate_api_key("test-key")

            req = mock_urlopen.call_args[0][0]
            assert req.full_url == f"{DEFAULT_ZHIPUAI_BASE_URL}/v1/messages"

    def test_custom_base_url(self):
        """Should use custom base URL when provided."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            validate_api_key("test-key", base_url="https://custom.api.com")

            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "https://custom.api.com/v1/messages"

    def test_strips_trailing_slash_from_base_url(self):
        """Should strip trailing slash before appending /v1/messages."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            validate_api_key("test-key", base_url="https://custom.api.com/")

            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "https://custom.api.com/v1/messages"

    def test_success_returns_true(self):
        """HTTP 200 should return (True, '')."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            valid, detail = validate_api_key("test-key")
            assert valid is True
            assert detail == ""

    def test_401_returns_auth_error(self):
        """HTTP 401 should return (False, 'authentication_error')."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="", code=401, msg="Unauthorized", hdrs={}, fp=None
            )

            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail == "authentication_error"

    def test_403_returns_auth_error(self):
        """HTTP 403 should return (False, 'authentication_error')."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="", code=403, msg="Forbidden", hdrs={}, fp=None
            )

            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail == "authentication_error"

    def test_400_returns_true(self):
        """HTTP 400 (bad request) means key was accepted; return True."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="", code=400, msg="Bad Request", hdrs={}, fp=None
            )

            valid, detail = validate_api_key("test-key")
            assert valid is True

    def test_network_error_returns_detail(self):
        """URLError (DNS/connection) should return network_error detail."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError(
                reason="Name or service not known"
            )

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("network_error")
            assert "Name or service not known" in detail

    def test_timeout_returns_network_error(self):
        """Timeout should return network_error detail."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("timed out")

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("network_error")

    def test_sends_correct_headers(self):
        """Should send x-api-key and anthropic-version headers."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            validate_api_key("my-secret-key")

            req = mock_urlopen.call_args[0][0]
            assert req.get_header("X-api-key") == "my-secret-key"
            assert req.get_header("Anthropic-version") == "2023-06-01"
