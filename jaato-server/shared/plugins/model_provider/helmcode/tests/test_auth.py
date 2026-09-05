"""Tests for Helmcode API key validation and credential loading.

Parallels the Nebius/OVHcloud/Doubleword auth tests: verifies that rate
limit / billing / server-error responses surface structured detail codes
instead of being silently accepted as "key valid", and that a corrupt
credential file surfaces a reason instead of looking identical to a
missing one.

Helmcode validates against ``GET /v1/models`` rather than a chat
completion: the catalogue requires auth (an unkeyed request is answered
``401 auth_error``), so a 200 proves the key while costing no tokens and
naming no probe model that could rotate out of the catalogue.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ..auth import (
    HelmcodeCredentials,
    try_load_credentials_with_reason,
    validate_api_key,
)
from ..env import DEFAULT_BASE_URL

CLIENT_PATCH = "shared.plugins.model_provider.helmcode.auth._create_validation_client"


def _mock_client(status_code=200, body_text: str = ""):
    """Return a mock httpx client that returns the given status code and body."""
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
            valid, detail = validate_api_key("sk-test-test")
            assert valid is True
            assert detail == ""

    def test_403_returns_true(self):
        """HTTP 403 means the key authenticated but is not entitled to list
        the catalogue — Helmcode rejects a bad key with 401 before any
        entitlement check, so the key itself is proven valid."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(403, body_text="forbidden")
            valid, detail = validate_api_key("sk-test-test")
            assert valid is True

    def test_401_returns_auth_error(self):
        """The live shape: an unkeyed/bad-key request is answered 401 with an
        ``auth_error`` envelope (verified 2026-09-05)."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(
                401,
                body_text='{"error":{"message":"Authentication Error, No api '
                          'key passed in.","type":"auth_error","code":"401"}}',
            )
            valid, detail = validate_api_key("bad-key")
            assert valid is False
            assert detail.startswith("authentication_error")
            assert "401" in detail

    def test_404_is_not_success(self):
        """Unlike the chat-probe providers, a 404 here is NOT proof of a good
        key: there is no probe model to miss, so a 404 means the endpoint
        itself is wrong (a mis-set base_url) and must not save the key."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(404, body_text="not found")
            valid, detail = validate_api_key("sk-test-test")
            assert valid is False
            assert detail.startswith("http_error")
            assert "404" in detail

    def test_429_returns_rate_limit_not_success(self):
        """HTTP 429 must NOT be silently treated as "key valid" — a
        rate-limited response must not save the key as if it worked."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(
                429, body_text='{"error":"rate_limit_exceeded"}'
            )
            valid, detail = validate_api_key("sk-test-test")
            assert valid is False
            assert detail.startswith("rate_limit")
            assert "429" in detail

    def test_402_returns_payment_required(self):
        """HTTP 402 = prepaid credit exhausted (Helmcode's documented 402,
        raised only by the resold frontier models)."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(
                402, body_text="credits_exhausted"
            )
            valid, detail = validate_api_key("sk-test-test")
            assert valid is False
            assert detail.startswith("payment_required")
            assert "402" in detail

    def test_500_returns_server_error(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(503, body_text="overloaded")
            valid, detail = validate_api_key("sk-test-test")
            assert valid is False
            assert detail.startswith("server_error")
            assert "503" in detail

    def test_unexpected_status_returns_http_error(self):
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client(418, body_text="I'm a teapot")
            valid, detail = validate_api_key("sk-test-test")
            assert valid is False
            assert detail.startswith("http_error")
            assert "418" in detail

    def test_network_error_returns_detail(self):
        with patch(CLIENT_PATCH) as mock_create:
            client = MagicMock()
            client.get.side_effect = httpx.ConnectError("connection refused")
            mock_create.return_value = client
            valid, detail = validate_api_key("sk-test-test")
            assert valid is False
            assert detail.startswith("network_error")
            assert "connection refused" in detail

    def test_uses_bearer_auth_header(self):
        """Validates that Helmcode uses Authorization: Bearer."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()
            validate_api_key("sk-test-secret")
            call_args = mock_create.return_value.get.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer sk-test-secret"

    def test_gets_the_models_endpoint_on_the_default_gateway(self):
        """A GET to /models, not a chat completion: no tokens spent, and no
        probe model that could rotate out of the catalogue."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()
            validate_api_key("sk-test-secret")
            client = mock_create.return_value
            client.post.assert_not_called()
            assert client.get.call_args[0][0] == f"{DEFAULT_BASE_URL}/models"

    def test_custom_base_url_is_honored(self):
        """An On-premise / proxied deployment validates against its own URL."""
        with patch(CLIENT_PATCH) as mock_create:
            mock_create.return_value = _mock_client()
            validate_api_key("sk-test", base_url="https://helmcode.corp.example/v1")
            call_args = mock_create.return_value.get.call_args
            assert call_args[0][0] == "https://helmcode.corp.example/v1/models"


class TestTryLoadCredentialsWithReason:
    """Tests for ``try_load_credentials_with_reason``.

    Surfaces the specific failure reason when a credential file exists
    but cannot be loaded — broken file no longer masquerades as
    missing.
    """

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
        """``~/.jaato/helmcode_auth.json`` answers when the workspace has none.

        The companion to the test above, and the reason that one means
        anything: "returns None" is only evidence of the missing-file
        path if the home tier would otherwise have answered.  Asserting
        both pins the resolution order the docstring on
        ``_get_token_storage_path`` claims — project first, then home.
        """
        home_file = fake_home / ".jaato" / "helmcode_auth.json"
        home_file.write_text(json.dumps({
            "api_key": "sk-home-tier",
            "created_at": 1234567890,
        }))

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is not None
        assert creds.api_key == "sk-home-tier"
        assert reason is None

    def test_valid_file_loads_credentials(self, tmp_path):
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        auth_file = jaato_dir / "helmcode_auth.json"
        auth_file.write_text(json.dumps({
            "api_key": "sk-test-abc",
            "created_at": 1234567890,
        }))

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is not None
        assert creds.api_key == "sk-test-abc"
        assert reason is None

    def test_corrupt_json_surfaces_reason(self, tmp_path):
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        auth_file = jaato_dir / "helmcode_auth.json"
        auth_file.write_text("{not valid json")

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None
        assert reason is not None
        assert "invalid JSON" in reason
        assert str(auth_file) in reason

    def test_missing_api_key_field_surfaces_reason(self, tmp_path):
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        auth_file = jaato_dir / "helmcode_auth.json"
        auth_file.write_text(json.dumps({"created_at": 1234567890}))

        creds, reason = try_load_credentials_with_reason(workspace_path=str(tmp_path))
        assert creds is None
        assert reason is not None
        assert "malformed" in reason or "missing" in reason


class TestCredentialRepr:
    """The stored key must never render (#721).

    ``shared/tests/test_credential_hygiene.py`` discovers secret-bearing
    dataclasses by AST scan and asserts this fleet-wide; this is the
    local guard so a change here fails next to the code it breaks.
    """

    def test_repr_redacts_the_key(self):
        creds = HelmcodeCredentials(api_key="sk-live-do-not-print", created_at=0.0)
        for rendering in (repr(creds), str(creds), f"{creds}", repr([creds])):
            assert "sk-live-do-not-print" not in rendering

    def test_to_dict_still_carries_the_real_key(self):
        """Redaction guards display, never storage: the credential file is
        written from ``to_dict()``, and a redacted value there would lock
        the user out with a key that looks present and fails to
        authenticate."""
        creds = HelmcodeCredentials(api_key="sk-live-do-not-print", created_at=0.0)
        assert creds.to_dict()["api_key"] == "sk-live-do-not-print"
