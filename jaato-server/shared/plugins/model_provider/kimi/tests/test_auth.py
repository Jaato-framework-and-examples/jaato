"""Tests for Moonshot AI Kimi API key validation and credential loading.

Mirrors the doubleword auth tests: a quota / billing / region / server
response surfaces a structured detail code instead of being accepted as
"key valid", and a corrupt credential file surfaces a reason instead of
looking identical to a missing one.
"""

import json
from unittest.mock import MagicMock, patch

import httpx

from ..auth import (
    KimiCredentials,
    try_load_credentials_with_reason,
    validate_api_key,
)
from ..env import DEFAULT_BASE_URL

CLIENT_PATCH = "shared.plugins.model_provider.kimi.auth._create_validation_client"


def _mock_client(status_code=200, body_text: str = ""):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.text = body_text
    client = MagicMock()
    client.get.return_value = mock_response
    return client


class TestValidateApiKey:
    def test_success_returns_true(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(200)
            assert validate_api_key("sk-test") == (True, "")

    def test_401_returns_auth_error(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(401, '{"error":"bad key"}')
            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail.startswith("authentication_error") and "401" in detail

    def test_403_is_forbidden_not_success(self):
        """A region / key restriction must not save the key as if it worked."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(403, "not available in region")
            valid, detail = validate_api_key("sk-test")
            assert valid is False
            assert detail.startswith("forbidden") and "region" in detail

    def test_429_returns_rate_limit_not_success(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(429, '{"error":"quota"}')
            valid, detail = validate_api_key("sk-test")
            assert valid is False
            assert detail.startswith("rate_limit") and "429" in detail

    def test_402_returns_payment_required(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(402, "insufficient balance")
            valid, detail = validate_api_key("sk-test")
            assert valid is False
            assert detail.startswith("payment_required")

    def test_5xx_returns_server_error(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(503, "overloaded")
            valid, detail = validate_api_key("sk-test")
            assert valid is False
            assert detail.startswith("server_error") and "503" in detail

    def test_unexpected_status_returns_http_error(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(418, "teapot")
            valid, detail = validate_api_key("sk-test")
            assert valid is False
            assert detail.startswith("http_error")

    def test_network_error_returns_detail(self):
        with patch(CLIENT_PATCH) as mock_create:
            client = MagicMock()
            client.get.side_effect = httpx.ConnectError("refused")
            mock_create.return_value = client
            valid, detail = validate_api_key("sk-test")
            assert valid is False
            assert detail.startswith("network_error")

    def test_uses_bearer_auth_and_the_probe_path(self):
        with patch(CLIENT_PATCH) as mock_create:
            client = _mock_client(200)
            mock_create.return_value = client
            validate_api_key("sk-test")
            args, kwargs = client.get.call_args
            assert args[0] == DEFAULT_BASE_URL + "/users/me/balance"
            assert kwargs["headers"]["Authorization"] == "Bearer sk-test"

    def test_custom_base_url_is_probed(self):
        with patch(CLIENT_PATCH) as mock_create:
            client = _mock_client(200)
            mock_create.return_value = client
            validate_api_key("sk-test", base_url="http://localhost:9/v1/")
            assert client.get.call_args[0][0] == "http://localhost:9/v1/users/me/balance"


class TestCredentialLoading:
    def test_file_missing_returns_none_and_no_reason(self, tmp_path):
        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None and reason is None

    def test_home_tier_is_read_when_the_project_tier_is_empty(self, tmp_path, fake_home):
        home_file = fake_home / ".jaato" / "kimi_auth.json"
        home_file.write_text(json.dumps({"api_key": "sk-home", "created_at": 1}))
        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is not None and creds.api_key == "sk-home" and reason is None

    def test_valid_file_loads_credentials(self, tmp_path):
        (tmp_path / ".jaato").mkdir()
        (tmp_path / ".jaato" / "kimi_auth.json").write_text(
            json.dumps({"api_key": "sk-proj", "created_at": 1, "base_url": "http://x/v1"}))
        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds == KimiCredentials(api_key="sk-proj", created_at=1, base_url="http://x/v1")
        assert reason is None

    def test_corrupt_json_surfaces_reason(self, tmp_path):
        (tmp_path / ".jaato").mkdir()
        (tmp_path / ".jaato" / "kimi_auth.json").write_text("{not json")
        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None and "invalid JSON" in reason

    def test_missing_api_key_field_surfaces_reason(self, tmp_path):
        (tmp_path / ".jaato").mkdir()
        (tmp_path / ".jaato" / "kimi_auth.json").write_text(json.dumps({"created_at": 1}))
        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None and "malformed" in reason

    def test_repr_hides_the_key(self):
        assert "sk-secret" not in repr(KimiCredentials(api_key="sk-secret-1234567890", created_at=1))
