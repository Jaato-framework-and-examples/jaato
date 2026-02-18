"""Tests for Zhipu AI API key validation."""

import json
from unittest.mock import MagicMock, patch, call
import urllib.error

import pytest

from ..auth import validate_api_key
from ..env import DEFAULT_ZHIPUAI_BASE_URL

# All tests mock get_url_opener so validate_api_key() uses the proxy-aware
# opener from shared.http rather than bare urllib.request.urlopen().
OPENER_PATCH = "shared.plugins.model_provider.zhipuai.auth.get_url_opener"


def _mock_opener_ok():
    """Return a mock opener whose .open() returns a 200 response."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    mock_opener = MagicMock()
    mock_opener.open.return_value = mock_resp
    return mock_opener


class TestValidateApiKey:
    """Tests for validate_api_key()."""

    def test_uses_proxy_aware_opener(self):
        """Must use get_url_opener() from shared.http, not bare urlopen."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_get_opener.return_value = _mock_opener_ok()

            validate_api_key("test-key")

            # get_url_opener should be called with the test URL
            expected_url = f"{DEFAULT_ZHIPUAI_BASE_URL}/v1/messages"
            mock_get_opener.assert_called_once_with(expected_url)
            mock_get_opener.return_value.open.assert_called_once()

    def test_url_includes_v1_prefix(self):
        """Validation request must hit /v1/messages, not /messages."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_get_opener.return_value = _mock_opener_ok()

            validate_api_key("test-key")

            req = mock_get_opener.return_value.open.call_args[0][0]
            assert req.full_url == f"{DEFAULT_ZHIPUAI_BASE_URL}/v1/messages"

    def test_custom_base_url(self):
        """Should use custom base URL when provided."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_get_opener.return_value = _mock_opener_ok()

            validate_api_key("test-key", base_url="https://custom.api.com")

            req = mock_get_opener.return_value.open.call_args[0][0]
            assert req.full_url == "https://custom.api.com/v1/messages"

    def test_strips_trailing_slash_from_base_url(self):
        """Should strip trailing slash before appending /v1/messages."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_get_opener.return_value = _mock_opener_ok()

            validate_api_key("test-key", base_url="https://custom.api.com/")

            req = mock_get_opener.return_value.open.call_args[0][0]
            assert req.full_url == "https://custom.api.com/v1/messages"

    def test_success_returns_true(self):
        """HTTP 200 should return (True, '')."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_get_opener.return_value = _mock_opener_ok()

            valid, detail = validate_api_key("test-key")
            assert valid is True
            assert detail == ""

    def test_401_returns_auth_error(self):
        """HTTP 401 should return (False, 'authentication_error')."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_opener = MagicMock()
            mock_opener.open.side_effect = urllib.error.HTTPError(
                url="", code=401, msg="Unauthorized", hdrs={}, fp=None
            )
            mock_get_opener.return_value = mock_opener

            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail == "authentication_error"

    def test_403_returns_auth_error(self):
        """HTTP 403 should return (False, 'authentication_error')."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_opener = MagicMock()
            mock_opener.open.side_effect = urllib.error.HTTPError(
                url="", code=403, msg="Forbidden", hdrs={}, fp=None
            )
            mock_get_opener.return_value = mock_opener

            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail == "authentication_error"

    def test_400_returns_true(self):
        """HTTP 400 (bad request) means key was accepted; return True."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_opener = MagicMock()
            mock_opener.open.side_effect = urllib.error.HTTPError(
                url="", code=400, msg="Bad Request", hdrs={}, fp=None
            )
            mock_get_opener.return_value = mock_opener

            valid, detail = validate_api_key("test-key")
            assert valid is True

    def test_network_error_returns_detail(self):
        """URLError (DNS/connection) should return network_error detail."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_opener = MagicMock()
            mock_opener.open.side_effect = urllib.error.URLError(
                reason="Name or service not known"
            )
            mock_get_opener.return_value = mock_opener

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("network_error")
            assert "Name or service not known" in detail

    def test_timeout_returns_network_error(self):
        """Timeout should return network_error detail."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_opener = MagicMock()
            mock_opener.open.side_effect = TimeoutError("timed out")
            mock_get_opener.return_value = mock_opener

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("network_error")

    def test_sends_correct_headers(self):
        """Should send x-api-key and anthropic-version headers."""
        with patch(OPENER_PATCH) as mock_get_opener:
            mock_get_opener.return_value = _mock_opener_ok()

            validate_api_key("my-secret-key")

            req = mock_get_opener.return_value.open.call_args[0][0]
            assert req.get_header("X-api-key") == "my-secret-key"
            assert req.get_header("Anthropic-version") == "2023-06-01"
