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

from ..auth import (
    METHOD_API_KEY,
    METHOD_BASIC_PASSWORD,
    METHOD_BASIC_USERNAME,
    METHOD_BEARER,
    METHOD_OAUTH2_CLIENT_ID,
    METHOD_OAUTH2_CLIENT_SECRET,
    AuthAttempt,
    AuthError,
    AuthManager,
    AuthSource,
    _make_attempt,
    _make_source,
    _resolve_credential,
    _rotation_hint_for_uri,
)
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
            value, source = _resolve_credential("MY_TOKEN", METHOD_BEARER)
        assert value == "abc123"
        assert source.kind == "env"
        assert source.env_var == "MY_TOKEN"
        assert source.uri is None
        assert source.method == METHOD_BEARER
        assert source.provenance == "env var MY_TOKEN"

    def test_uri_routed_to_resolver(self, pass_resolver):
        value, source = _resolve_credential(
            "pass://secrets/github-token", METHOD_BEARER
        )
        assert value == "resolved-secret-value"
        assert source.kind == "uri"
        assert source.uri == "pass://secrets/github-token"
        assert source.env_var is None
        assert source.provenance == "secret URI pass://secrets/github-token"
        assert pass_resolver.calls == [("pass", "secrets/github-token", None)]

    def test_uri_with_key_fragment(self, pass_resolver):
        _resolve_credential("pass://app/creds#api_key", METHOD_BEARER)
        assert pass_resolver.calls[-1] == ("pass", "app/creds", "api_key")

    def test_uri_without_registered_resolver_returns_none(self):
        """When no resolver matches the scheme, the URI is NOT a valid
        credential — return None so the caller emits a clear error
        instead of using the URI string as the token."""
        value, source = _resolve_credential("vault://no/resolver", METHOD_BEARER)
        assert value is None
        assert source.kind == "uri"
        assert source.uri == "vault://no/resolver"
        assert source.provenance == "secret URI vault://no/resolver"

    def test_none_input(self):
        value, source = _resolve_credential(None, METHOD_BEARER)
        assert value is None
        assert source.kind == "unset"
        assert source.provenance == "(unset)"

    def test_empty_string_input(self):
        value, source = _resolve_credential("", METHOD_BEARER)
        assert value is None
        assert source.kind == "unset"
        assert source.provenance == "(unset)"


class TestRotationHint:
    """The hint maps known URI schemes to actionable shell commands.
    Unknown schemes return None — never guess at the command."""

    def test_pass_scheme_known(self):
        assert _rotation_hint_for_uri("pass://jaato/github-token") == (
            "pass edit jaato/github-token"
        )

    def test_pass_scheme_with_nested_path(self):
        assert _rotation_hint_for_uri("pass://team/svc/api-key") == (
            "pass edit team/svc/api-key"
        )

    def test_unknown_scheme_returns_none(self):
        assert _rotation_hint_for_uri("vault://secret/db#password") is None

    def test_non_uri_returns_none(self):
        assert _rotation_hint_for_uri("GITHUB_TOKEN") is None


class TestMakeSource:
    """_make_source classifies the *_env raw value into (env, uri, unset)."""

    def test_env_var_kind(self):
        source = _make_source("MY_VAR", METHOD_BEARER)
        assert source.kind == "env"
        assert source.env_var == "MY_VAR"
        assert source.uri is None
        assert source.rotation_hint is None

    def test_uri_kind_with_known_rotation(self):
        source = _make_source("pass://x/y", METHOD_BEARER)
        assert source.kind == "uri"
        assert source.uri == "pass://x/y"
        assert source.env_var is None
        assert source.rotation_hint == "pass edit x/y"

    def test_uri_kind_with_unknown_rotation(self):
        source = _make_source("vault://x/y", METHOD_BEARER)
        assert source.kind == "uri"
        assert source.rotation_hint is None

    def test_unset(self):
        source = _make_source("", METHOD_BEARER)
        assert source.kind == "unset"
        source = _make_source(None, METHOD_BEARER)
        assert source.kind == "unset"


class TestMakeAttempt:
    """token_len and token_prefix only populate for secret-like methods."""

    def test_bearer_resolved_captures_len_and_prefix(self):
        source = _make_source("GITHUB_TOKEN", METHOD_BEARER)
        attempt = _make_attempt("ghp_abcdef1234", source)
        assert attempt.resolved is True
        assert attempt.token_len == 14
        assert attempt.token_prefix == "ghp_"

    def test_basic_username_does_not_capture_prefix(self):
        """Username is an identifier, not a secret — never leak its prefix."""
        source = _make_source("MY_USER", METHOD_BASIC_USERNAME)
        attempt = _make_attempt("alice", source)
        assert attempt.resolved is True
        assert attempt.token_len is None
        assert attempt.token_prefix is None

    def test_basic_password_captures_prefix(self):
        source = _make_source("MY_PASS", METHOD_BASIC_PASSWORD)
        attempt = _make_attempt("hunter2", source)
        assert attempt.token_len == 7
        assert attempt.token_prefix == "hunt"

    def test_oauth2_client_id_does_not_capture_prefix(self):
        source = _make_source("CLIENT_ID", METHOD_OAUTH2_CLIENT_ID)
        attempt = _make_attempt("public-id-123", source)
        assert attempt.token_prefix is None
        assert attempt.token_len is None

    def test_oauth2_client_secret_captures_prefix(self):
        source = _make_source("CLIENT_SECRET", METHOD_OAUTH2_CLIENT_SECRET)
        attempt = _make_attempt("sk_live_xyz", source)
        assert attempt.token_prefix == "sk_l"

    def test_unresolved_attempt_has_no_token_metadata(self):
        source = _make_source("MISSING_VAR", METHOD_BEARER)
        attempt = _make_attempt(None, source)
        assert attempt.resolved is False
        assert attempt.token_len is None
        assert attempt.token_prefix is None


class TestBearerAuthWithSecretURI:
    """Full integration through AuthManager.get_auth_headers."""

    def test_bearer_token_from_pass_uri(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.BEARER,
            value_env="pass://jaato-knowledge-manager/github-token",
        )
        manager = AuthManager()
        headers, _query, attempts = manager.get_auth_headers(
            auth, service_name="github"
        )
        assert headers["Authorization"] == "Bearer resolved-secret-value"
        # One bearer attempt with URI provenance + rotation hint.
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt.resolved is True
        assert attempt.source.method == METHOD_BEARER
        assert attempt.source.kind == "uri"
        assert attempt.source.uri == "pass://jaato-knowledge-manager/github-token"
        assert attempt.source.rotation_hint == (
            "pass edit jaato-knowledge-manager/github-token"
        )
        # Length and prefix populated for bearer (secret-like).
        assert attempt.token_len == len("resolved-secret-value")
        assert attempt.token_prefix == "reso"

    def test_bearer_token_from_env_var(self):
        auth = AuthConfig(type=AuthType.BEARER, value_env="GH_TOKEN")
        with patch.dict("os.environ", {"GH_TOKEN": "ghp_abc"}, clear=False):
            manager = AuthManager()
            headers, _query, attempts = manager.get_auth_headers(
                auth, service_name="github"
            )
        assert headers["Authorization"] == "Bearer ghp_abc"
        assert len(attempts) == 1
        assert attempts[0].source.kind == "env"
        assert attempts[0].source.env_var == "GH_TOKEN"
        assert attempts[0].source.rotation_hint is None
        assert attempts[0].token_prefix == "ghp_"

    def test_bearer_token_missing_uri_reports_provenance(self):
        """Error message names the URI so the user sees *where* the
        credential should have come from (without exposing the
        credential itself).  AuthError also carries the partial
        attempts list for the auth_context builder."""
        auth = AuthConfig(
            type=AuthType.BEARER,
            value_env="vault://no-such-resolver/key",
        )
        manager = AuthManager()
        with pytest.raises(AuthError) as exc_info:
            manager.get_auth_headers(auth, service_name="svc")
        assert "vault://no-such-resolver/key" in str(exc_info.value)
        # Attempts list is attached to the exception for diagnosis.
        assert len(exc_info.value.attempts) == 1
        attempt = exc_info.value.attempts[0]
        assert attempt.resolved is False
        assert attempt.source.uri == "vault://no-such-resolver/key"


class TestApiKeyAuthWithSecretURI:

    def test_api_key_header_from_pass_uri(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.API_KEY,
            key_location=ParameterLocation.HEADER,
            key_name="X-API-Key",
            value_env="pass://svc/api-key",
        )
        manager = AuthManager()
        headers, query, attempts = manager.get_auth_headers(
            auth, service_name="svc"
        )
        assert headers["X-API-Key"] == "resolved-secret-value"
        assert query == {}
        assert len(attempts) == 1
        assert attempts[0].source.method == METHOD_API_KEY

    def test_api_key_query_from_pass_uri(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.API_KEY,
            key_location=ParameterLocation.QUERY,
            key_name="api_key",
            value_env="pass://svc/api-key",
        )
        manager = AuthManager()
        headers, query, _attempts = manager.get_auth_headers(
            auth, service_name="svc"
        )
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
        headers, _query, attempts = manager.get_auth_headers(
            auth, service_name="svc"
        )
        # Basic auth base64-encodes user:pass; we just verify the header
        # was produced without AuthError (both fields resolved).
        assert headers["Authorization"].startswith("Basic ")
        # Two attempts: username (no prefix), password (with prefix).
        assert len(attempts) == 2
        methods = [a.source.method for a in attempts]
        assert methods == [METHOD_BASIC_USERNAME, METHOD_BASIC_PASSWORD]
        assert attempts[0].token_prefix is None  # username never leaks prefix
        assert attempts[1].token_prefix is not None  # password is secret-like

    def test_mixed_uri_and_env(self, pass_resolver):
        """Username via env, password via URI — should both work."""
        auth = AuthConfig(
            type=AuthType.BASIC,
            username_env="MY_USER",
            password_env="pass://svc/password",
        )
        with patch.dict("os.environ", {"MY_USER": "alice"}, clear=False):
            manager = AuthManager()
            headers, _query, attempts = manager.get_auth_headers(
                auth, service_name="svc"
            )
        assert headers["Authorization"].startswith("Basic ")
        assert attempts[0].source.kind == "env"
        assert attempts[1].source.kind == "uri"

    def test_missing_password_attaches_partial_attempts(self, pass_resolver):
        """When username resolves but password fails, the AuthError
        carries both attempts (the username success + password failure)
        so the auth_context can name exactly what went wrong."""
        auth = AuthConfig(
            type=AuthType.BASIC,
            username_env="MY_USER",
            password_env="vault://no-resolver/pw",
        )
        with patch.dict("os.environ", {"MY_USER": "alice"}, clear=False):
            manager = AuthManager()
            with pytest.raises(AuthError) as exc_info:
                manager.get_auth_headers(auth, service_name="svc")
        assert len(exc_info.value.attempts) == 2
        assert exc_info.value.attempts[0].resolved is True   # username
        assert exc_info.value.attempts[1].resolved is False  # password


class TestCheckCredentialsWithSecretURI:
    """Regression: ``check_credentials`` (called by
    ``configure_service_auth``) used to call ``get_session_env(URI)``
    directly, which always returned ``None``, making URI-backed
    credentials look "missing" even when the resolver was registered
    and would succeed.  ``configure_service_auth`` then reported a
    misleading auth status to the agent."""

    def test_uri_resolved_credential_reported_present(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.BEARER,
            value_env="pass://jaato-knowledge-manager/github-token",
        )
        manager = AuthManager()
        result = manager.check_credentials(auth)
        assert result["env_vars_required"] == [
            "pass://jaato-knowledge-manager/github-token"
        ]
        assert result["env_vars_present"] == [
            "pass://jaato-knowledge-manager/github-token"
        ]
        assert result["env_vars_missing"] == []

    def test_uri_without_resolver_reported_missing(self):
        """No resolver for the scheme → still reported as missing
        (same as if the env var didn't exist).  The required list
        preserves the URI string so the user sees what they asked for."""
        auth = AuthConfig(
            type=AuthType.BEARER,
            value_env="vault://no-such-resolver/key",
        )
        manager = AuthManager()
        result = manager.check_credentials(auth)
        assert result["env_vars_present"] == []
        assert result["env_vars_missing"] == [
            "vault://no-such-resolver/key"
        ]

    def test_env_var_path_unchanged(self):
        """Plain env var names still work — backwards compat."""
        auth = AuthConfig(type=AuthType.BEARER, value_env="GH_TOKEN")
        with patch.dict("os.environ", {"GH_TOKEN": "ghp_abc"}, clear=False):
            manager = AuthManager()
            result = manager.check_credentials(auth)
        assert result["env_vars_present"] == ["GH_TOKEN"]
        assert result["env_vars_missing"] == []

    def test_basic_auth_mixed_sources(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.BASIC,
            username_env="MY_USER",
            password_env="pass://svc/password",
        )
        with patch.dict("os.environ", {"MY_USER": "alice"}, clear=False):
            manager = AuthManager()
            result = manager.check_credentials(auth)
        assert set(result["env_vars_present"]) == {
            "MY_USER", "pass://svc/password"
        }
        assert result["env_vars_missing"] == []


class TestBuildAuthContext:
    """Plugin-level diagnostic builder.  Verifies that the new
    structured fields (resolved_from, token_prefix, rotation_hint)
    flow into the auth_context that 401/403 responses surface to the
    model — and that the hint string includes the rotation command
    verbatim when the URI scheme is recognised."""

    def _service(self, auth_config):
        from ..plugin import ServiceConnectorPlugin  # local import to avoid cycle
        from ..types import ServiceConfig
        # Build a minimal ServiceConfig — only fields read by
        # _build_auth_context (auth + name).
        return ServiceConfig(name="my-svc", base_url="https://x", auth=auth_config)

    def _build(self, service, resolved, attempts):
        from ..plugin import ServiceConnectorPlugin
        return ServiceConnectorPlugin._build_auth_context(
            service, resolved=resolved, attempts=attempts
        )

    def test_bearer_pass_uri_401_includes_rotation_command(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.BEARER,
            value_env="pass://jaato/github-token",
        )
        manager = AuthManager()
        _h, _q, attempts = manager.get_auth_headers(auth, service_name="github")
        ctx = self._build(self._service(auth), resolved=True, attempts=attempts)

        assert ctx["auth_type"] == "bearer"
        assert ctx["service_name"] == "my-svc"
        assert ctx["credentials_resolved"] is True
        # Per-credential structured entry.
        assert len(ctx["credentials"]) == 1
        cred = ctx["credentials"][0]
        assert cred["method"] == METHOD_BEARER
        assert cred["resolved_from"] == "pass://jaato/github-token"
        assert cred["rotation_hint"] == "pass edit jaato/github-token"
        assert cred["token_prefix"] == "reso"
        assert cred["token_len"] == len("resolved-secret-value")
        # Hint names the rotation command verbatim.
        assert "pass edit jaato/github-token" in ctx["hint"]
        assert "pass://jaato/github-token" in ctx["hint"]
        assert "stale" in ctx["hint"]

    def test_bearer_env_var_401_omits_rotation_command(self):
        """Env-var-backed credentials don't get a rotation command —
        we don't know how the user sets their env, so the hint stays
        generic but still names the env var."""
        auth = AuthConfig(type=AuthType.BEARER, value_env="GH_TOKEN")
        with patch.dict("os.environ", {"GH_TOKEN": "ghp_abc"}, clear=False):
            manager = AuthManager()
            _h, _q, attempts = manager.get_auth_headers(auth, service_name="x")
        ctx = self._build(self._service(auth), resolved=True, attempts=attempts)

        assert ctx["credentials"][0]["resolved_from"] == "env://GH_TOKEN"
        assert "rotation_hint" not in ctx["credentials"][0]
        assert "env var GH_TOKEN" in ctx["hint"]
        # No "Rotate it with" sentence when no rotation command exists.
        assert "Rotate it with" not in ctx["hint"]

    def test_unresolved_credential_names_failing_source(self):
        """When credentials never resolve, the hint names the failing
        source so the user knows whether to set an env var or check a
        secret backend."""
        auth = AuthConfig(
            type=AuthType.BEARER,
            value_env="vault://no-resolver/key",
        )
        manager = AuthManager()
        with pytest.raises(AuthError) as exc_info:
            manager.get_auth_headers(auth, service_name="x")
        ctx = self._build(
            self._service(auth),
            resolved=False,
            attempts=exc_info.value.attempts,
        )

        assert ctx["credentials_resolved"] is False
        assert "vault://no-resolver/key" in ctx["hint"]
        assert "no resolver is registered" in ctx["hint"]

    def test_basic_auth_emits_two_credentials(self, pass_resolver):
        auth = AuthConfig(
            type=AuthType.BASIC,
            username_env="MY_USER",
            password_env="pass://svc/password",
        )
        with patch.dict("os.environ", {"MY_USER": "alice"}, clear=False):
            manager = AuthManager()
            _h, _q, attempts = manager.get_auth_headers(auth, service_name="x")
        ctx = self._build(self._service(auth), resolved=True, attempts=attempts)

        assert len(ctx["credentials"]) == 2
        username, password = ctx["credentials"]
        assert username["method"] == METHOD_BASIC_USERNAME
        assert "token_prefix" not in username  # never leak username prefix
        assert password["method"] == METHOD_BASIC_PASSWORD
        assert password["token_prefix"] is not None
        assert password["rotation_hint"] == "pass edit svc/password"
        # Basic auth has two credentials but only one is secret-like
        # (the password) — the hint can still pinpoint it because
        # rotating the password is by far the likeliest fix on a 401.
        assert "pass edit svc/password" in ctx["hint"]
        assert "pass://svc/password" in ctx["hint"]

    def test_legacy_call_without_attempts_still_works(self):
        """Backwards-compat: callers that don't pass attempts get the
        old-shape auth_context (env_var fallback, generic hint) so
        in-flight code doesn't break."""
        auth = AuthConfig(type=AuthType.BEARER, value_env="GH_TOKEN")
        ctx = self._build(self._service(auth), resolved=True, attempts=None)

        assert ctx["credentials_resolved"] is True
        # Falls back to {env_var: ...} style entries.
        assert ctx["credentials"][0]["env_var"] == "GH_TOKEN"
        # Hint is the generic "expired or revoked" string.
        assert "expired or revoked" in ctx["hint"]
