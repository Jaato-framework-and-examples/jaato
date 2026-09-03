"""Tests for OpenRouter API key validation and credential loading.

Mirrors the NIM auth tests: verifies that quota / billing / server-error
responses surface structured detail codes instead of being silently
treated as "key valid", and that a corrupt credential file surfaces the
specific failure reason instead of looking identical to a missing one.
"""

import json
from unittest.mock import MagicMock, patch

import httpx

from ..auth import (
    OpenRouterCredentials,
    try_load_credentials_with_reason,
    validate_api_key,
)

CLIENT_PATCH = "shared.plugins.model_provider.openrouter.auth._create_validation_client"


def _mock_client(status_code=200, body_text: str = ""):
    """Return a mock httpx client that returns the given status and body."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.text = body_text

    client = MagicMock()
    client.get.return_value = mock_response
    return client


class TestValidateApiKey:
    """Tests for validate_api_key()."""

    def test_success_returns_true(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(200)
            valid, detail = validate_api_key("sk-or-test")
            assert valid is True
            assert detail == ""

    def test_401_returns_auth_error(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(
                401, body_text='{"error":"invalid key"}'
            )
            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail.startswith("authentication_error")
            assert "401" in detail

    def test_403_returns_auth_error(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(403)
            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail.startswith("authentication_error")

    def test_429_returns_rate_limit_not_success(self):
        """HTTP 429 must NOT be silently treated as 'key valid'."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(
                429, body_text='{"error":"rate_limit_exceeded"}'
            )
            valid, detail = validate_api_key("sk-or-test")
            assert valid is False
            assert detail.startswith("rate_limit")
            assert "429" in detail

    def test_402_returns_payment_required(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(402, body_text="Payment Required")
            valid, detail = validate_api_key("sk-or-test")
            assert valid is False
            assert detail.startswith("payment_required")

    def test_500_returns_server_error(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(503, body_text="overloaded")
            valid, detail = validate_api_key("sk-or-test")
            assert valid is False
            assert detail.startswith("server_error")

    def test_unexpected_status_returns_http_error(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(418, body_text="I'm a teapot")
            valid, detail = validate_api_key("sk-or-test")
            assert valid is False
            assert detail.startswith("http_error")

    def test_network_error_returns_detail(self):
        with patch(CLIENT_PATCH) as mock_create:
            client = MagicMock()
            client.get.side_effect = httpx.ConnectError("connection refused")
            mock_create.return_value = client
            valid, detail = validate_api_key("sk-or-test")
            assert valid is False
            assert detail.startswith("network_error")
            assert "connection refused" in detail

    def test_uses_bearer_auth_header(self):
        """OpenRouter uses Authorization: Bearer."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()
            validate_api_key("sk-or-secret")
            call_args = mock_create.return_value.get.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer sk-or-secret"

    def test_hits_key_introspection_endpoint(self):
        """Validation must call GET /key, not /chat/completions."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()
            validate_api_key("sk-or-secret", "https://openrouter.ai/api/v1")
            call_args = mock_create.return_value.get.call_args
            url = call_args[0][0]
            assert url.endswith("/key")


class TestTryLoadCredentialsWithReason:
    """Tests for try_load_credentials_with_reason."""

    def test_file_missing_returns_none_and_no_reason(self, tmp_path):
        """Missing file is not an error; reason is None.

        Depends on the ``HOME`` isolation in ``jaato-server/conftest.py``:
        the loader consults the project tier and then ``~/.jaato/``, so
        an empty ``tmp_path`` workspace alone does not make the file
        missing.  Without that isolation this assertion failed on any
        machine where the developer had authenticated — and pytest
        rendered the loaded credential into the failure message, which
        put a live key into scrollback and CI logs (#721).  The
        companion below is what proves the home tier is still read.
        """
        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None
        assert reason is None

    def test_home_tier_is_read_when_the_project_tier_is_empty(
        self, tmp_path, fake_home,
    ):
        """``~/.jaato/openrouter_auth.json`` answers when the workspace has none.

        The companion to the test above, and the reason that one means
        anything: "returns None" is only evidence of the missing-file
        path if the home tier would otherwise have answered.  Asserting
        both pins the resolution order the docstring on
        ``_get_token_storage_path`` claims — project first, then home.
        """
        home_file = fake_home / ".jaato" / "openrouter_auth.json"
        home_file.write_text(json.dumps({
            "api_key": "sk-or-home-tier",
            "created_at": 1234567890,
        }))

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is not None
        assert creds.api_key == "sk-or-home-tier"
        assert reason is None

    def test_valid_file_loads_credentials(self, tmp_path):
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        auth_file = jaato_dir / "openrouter_auth.json"
        auth_file.write_text(json.dumps({
            "api_key": "sk-or-abc",
            "created_at": 1234567890,
        }))

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is not None
        assert creds.api_key == "sk-or-abc"
        assert reason is None

    def test_corrupt_json_surfaces_reason(self, tmp_path):
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        auth_file = jaato_dir / "openrouter_auth.json"
        auth_file.write_text("{not valid json")

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None
        assert reason is not None
        assert "invalid JSON" in reason
        assert str(auth_file) in reason

    def test_missing_api_key_field_surfaces_reason(self, tmp_path):
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        auth_file = jaato_dir / "openrouter_auth.json"
        auth_file.write_text(json.dumps({"created_at": 1234567890}))

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None
        assert reason is not None
        assert "malformed" in reason or "missing" in reason
