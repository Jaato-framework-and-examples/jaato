"""Tests for Zhipu AI API key validation."""

import json
from unittest.mock import MagicMock, patch

import httpx
from pathlib import Path
import pytest

from ..auth import validate_api_key, try_load_credentials_with_reason
from ..env import DEFAULT_ZHIPUAI_BASE_URL

# All tests mock _create_validation_client so validate_api_key() uses
# a controlled httpx client rather than making real network requests.
CLIENT_PATCH = "shared.plugins.model_provider.zhipuai.auth._create_validation_client"


def _mock_client(status_code=200, body_text: str = ""):
    """Return a mock httpx client that returns the given status code and body."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.text = body_text

    client = MagicMock()
    client.post.return_value = mock_response
    return client


@pytest.fixture(autouse=True)
def _isolate_home_credentials(monkeypatch, tmp_path):
    """Keep the developer's real ~/.jaato credentials out of these tests.

    Credential resolution falls through to a HOME tier, so "no key
    configured" tests found a REAL stored key on a machine where the
    developer has actually authenticated -- one asserted a
    ZhipuAIAPIKeyNotFoundError that never came.  Clean CI has no such file,
    which is why it passed there while failing locally.
    """
    empty_home = tmp_path / "home"
    (empty_home / ".jaato").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty_home))
    # BOTH tiers leak, not just home: the project tier resolves to
    # ``<cwd>/.jaato/`` and pytest runs from the repo root, which carries real
    # stored credentials.  Move cwd somewhere empty and clear the workspace
    # env so neither tier can reach them.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)


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
        """HTTP 401 should return (False, 'authentication_error: ...')."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(401, body_text='{"error":"invalid key"}')

            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail.startswith("authentication_error")
            # Status and body should appear in detail so the user can see
            # what the provider said.
            assert "401" in detail
            assert "invalid key" in detail

    def test_403_returns_auth_error(self):
        """HTTP 403 should return (False, 'authentication_error: ...')."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(403)

            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail.startswith("authentication_error")
            assert "403" in detail

    def test_400_returns_true(self):
        """HTTP 400 (bad request) means key was accepted; return True."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(400)

            valid, detail = validate_api_key("test-key")
            assert valid is True

    def test_429_returns_rate_limit_not_success(self):
        """HTTP 429 means quota/rate limit exceeded — the key may be valid,
        but the user's account is over quota.  Previously this was masked
        as "key accepted" which hid the provider's real reason for
        failure (the user's own hunch about quota-exceeded errors being
        swallowed).  Now it returns a distinct ``rate_limit`` detail so
        callers can surface the state instead of saving the key blindly.
        """
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(
                429, body_text='{"error":"quota_exceeded"}'
            )

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("rate_limit")
            assert "429" in detail
            assert "quota" in detail

    def test_402_returns_payment_required(self):
        """HTTP 402 means billing/subscription is inactive."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(402, body_text="Payment Required")

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("payment_required")
            assert "402" in detail

    def test_500_returns_server_error(self):
        """HTTP 5xx means the provider is temporarily unavailable."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(503, body_text="overloaded")

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("server_error")
            assert "503" in detail
            assert "overloaded" in detail

    def test_unexpected_status_returns_http_error(self):
        """Any other non-2xx status should surface verbatim for debugging."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(418, body_text="I'm a teapot")

            valid, detail = validate_api_key("test-key")
            assert valid is False
            assert detail.startswith("http_error")
            assert "418" in detail

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


class TestTryLoadCredentialsWithReason:
    """Tests for ``try_load_credentials_with_reason``.

    Previously ``load_credentials`` collapsed every failure to ``None``,
    making a corrupt credential file indistinguishable from a missing one
    at the provider level.  The new helper returns a reason string so
    ``verify_auth`` can surface the real cause instead of reporting
    "No Zhipu AI credentials found" when the file actually exists.
    """

    def test_file_missing_returns_none_and_no_reason(self, tmp_path):
        """Missing file is not an error; reason is None."""
        # tmp_path has no .jaato/zhipuai_auth.json
        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None
        assert reason is None

    def test_valid_file_loads_credentials(self, tmp_path):
        """Well-formed file loads successfully with no reason."""
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        auth_file = jaato_dir / "zhipuai_auth.json"
        auth_file.write_text(json.dumps({
            "api_key": "abc.xyz",
            "created_at": 1234567890,
        }))

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is not None
        assert creds.api_key == "abc.xyz"
        assert reason is None

    def test_corrupt_json_surfaces_reason(self, tmp_path):
        """Corrupt JSON returns a reason mentioning the parse failure."""
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        auth_file = jaato_dir / "zhipuai_auth.json"
        auth_file.write_text("{not valid json")

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None
        assert reason is not None
        assert "invalid JSON" in reason
        assert str(auth_file) in reason

    def test_missing_api_key_field_surfaces_reason(self, tmp_path):
        """File without ``api_key`` field reports a malformed-credentials reason."""
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        auth_file = jaato_dir / "zhipuai_auth.json"
        auth_file.write_text(json.dumps({"created_at": 1234567890}))

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None
        assert reason is not None
        assert "malformed" in reason or "missing" in reason
