"""Tests for secret URI resolution in service_connector auth.

Regression: ``AuthConfig.*_env`` fields historically held literal env
var names.  Users who store credentials via pass/vault/etc. write the
secret URI directly into those fields (e.g.
``token_env: pass://jaato-knowledge-manager/github-token``).  Before
the fix, ``auth.py`` called ``get_session_env("pass://...")`` which
treated the URI as an env var name and returned ``None``, so every
call_service request failed with "Bearer token not found in env var:
pass://..." — even though jaato-premium's pass:// resolver was
registered and available.

The fix centralises credential lookup in ``_resolve_credential`` which
detects URIs and routes them through the ``SecretResolver`` registry
(``subagent.config._resolve_secret_uri``), falling back to env-var
lookup for plain names.  These tests lock in that behaviour.
"""

from unittest.mock import patch

import pytest

from ..auth import AuthError, AuthManager, _resolve_credential
from ..types import AuthConfig, AuthType, ParameterLocation


class StubResolver:
    """Fake SecretResolver returning a canned value per scheme."""

    def __init__(self, schemes, value):
        self._schemes = frozenset(schemes)
        self._value = value
        self.calls = []

    @property
    def schemes(self):
        return self._schemes

    def resolve(self, scheme, path, key):
        self.calls.append((scheme, path, key))
        return self._value


@pytest.fixture
def pass_resolver():
    """Register a stub pass:// resolver for the duration of the test."""
    from shared.plugins.subagent import config as subagent_config

    stub = StubResolver(("pass",), "resolved-secret-value")
    # Bypass the entry-point discovery — inject directly into the
    # cache.  This matches what jaato-premium does at startup.
    original = subagent_config._resolvers
    subagent_config._resolvers = {"pass": stub}
    try:
        yield stub
    finally:
        subagent_config._resolvers = original


class TestResolveCredential:

    def test_env_var_lookup_legacy_behavior(self):
        with patch.dict("os.environ", {"MY_TOKEN": "abc123"}, clear=False):
            value, provenance = _resolve_credential("MY_TOKEN")
        assert value == "abc123"
        assert "env var MY_TOKEN" == provenance

    def test_uri_routed_to_resolver(self, pass_resolver):
        value, provenance = _resolve_credential("pass://secrets/github-token")
        assert value == "resolved-secret-value"
        assert provenance == "secret URI pass://secrets/github-token"
        assert pass_resolver.calls == [("pass", "secrets/github-token", None)]

    def test_uri_with_key_fragment(self, pass_resolver):
        _resolve_credential("pass://app/creds#api_key")
        assert pass_resolver.calls[-1] == ("pass", "app/creds", "api_key")

    def test_uri_without_registered_resolver_returns_none(self):
        """When no resolver matches the scheme, the URI is NOT a valid
        credential — return None so the caller emits a clear error
        instead of using the URI string as the token."""
        value, provenance = _resolve_credential("vault://no/resolver")
        assert value is None
        assert provenance == "secret URI vault://no/resolver"

    def test_none_input(self):
        value, provenance = _resolve_credential(None)
        assert value is None
        assert provenance == "(unset)"

    def test_empty_string_input(self):
        value, provenance = _resolve_credential("")
        assert value is None
        assert provenance == "(unset)"


class TestBearerAuthWithSecretURI:
    """Full integration through AuthManager.get_auth_headers."""

    def test_bearer_token_from_pass_uri(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.BEARER,
            value_env="pass://jaato-knowledge-manager/github-token",
        )
        manager = AuthManager()
        headers, _ = manager.get_auth_headers(auth, service_name="github")
        assert headers["Authorization"] == "Bearer resolved-secret-value"

    def test_bearer_token_from_env_var(self):
        auth = AuthConfig(type=AuthType.BEARER, value_env="GH_TOKEN")
        with patch.dict("os.environ", {"GH_TOKEN": "ghp_abc"}, clear=False):
            manager = AuthManager()
            headers, _ = manager.get_auth_headers(auth, service_name="github")
        assert headers["Authorization"] == "Bearer ghp_abc"

    def test_bearer_token_missing_uri_reports_provenance(self):
        """Error message names the URI so the user sees *where* the
        credential should have come from (without exposing the
        credential itself)."""
        auth = AuthConfig(
            type=AuthType.BEARER,
            value_env="vault://no-such-resolver/key",
        )
        manager = AuthManager()
        with pytest.raises(AuthError) as exc_info:
            manager.get_auth_headers(auth, service_name="svc")
        assert "vault://no-such-resolver/key" in str(exc_info.value)


class TestApiKeyAuthWithSecretURI:

    def test_api_key_header_from_pass_uri(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.API_KEY,
            key_location=ParameterLocation.HEADER,
            key_name="X-API-Key",
            value_env="pass://svc/api-key",
        )
        manager = AuthManager()
        headers, query = manager.get_auth_headers(auth, service_name="svc")
        assert headers["X-API-Key"] == "resolved-secret-value"
        assert query == {}

    def test_api_key_query_from_pass_uri(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.API_KEY,
            key_location=ParameterLocation.QUERY,
            key_name="api_key",
            value_env="pass://svc/api-key",
        )
        manager = AuthManager()
        headers, query = manager.get_auth_headers(auth, service_name="svc")
        assert query == {"api_key": "resolved-secret-value"}
        assert headers == {}


class TestBasicAuthWithSecretURI:

    def test_both_credentials_from_uris(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.BASIC,
            username_env="pass://svc/username",
            password_env="pass://svc/password",
        )
        manager = AuthManager()
        headers, _ = manager.get_auth_headers(auth, service_name="svc")
        # Basic auth base64-encodes user:pass; we just verify the header
        # was produced without AuthError (both fields resolved).
        assert headers["Authorization"].startswith("Basic ")

    def test_mixed_uri_and_env(self, pass_resolver):
        """Username via env, password via URI — should both work."""
        auth = AuthConfig(
            type=AuthType.BASIC,
            username_env="MY_USER",
            password_env="pass://svc/password",
        )
        with patch.dict("os.environ", {"MY_USER": "alice"}, clear=False):
            manager = AuthManager()
            headers, _ = manager.get_auth_headers(auth, service_name="svc")
        assert headers["Authorization"].startswith("Basic ")
