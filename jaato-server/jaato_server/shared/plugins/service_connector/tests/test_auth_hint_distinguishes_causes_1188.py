"""#1188: the auth-failure hint distinguishes three causes.

``call_service``'s resolution-failure hint used to name jaato-premium as the
likely-missing resolver for EVERY failure.  That advice is right for exactly
one of three distinct causes and misleading for the other two:

  (a) a secret URI whose scheme has NO registered resolver — genuinely
      missing resolver, install the package;
  (b) a literal env-var NAME (never matched the secret-URI pattern, so the
      resolver chain was never entered) that is unset — the reporter's own
      case, whose fix is "set the variable / write a pass:// URI", NOT
      "install a resolver";
  (c) a secret URI with a registered resolver that returned an empty value —
      wrong entry / missing key, not a missing resolver.

These tests drive both ``resolution_failure_hint`` directly and the real
``_build_auth_context`` path (via ``AuthManager.get_auth_headers`` raising an
``AuthError`` carrying attempts), asserting each cause produces its own
wording and that the wordings do not bleed into each other.
"""

import pytest

from ..auth import (
    AuthManager,
    AuthSource,
    AuthError,
    METHOD_BEARER,
    resolution_failure_hint,
)
from ..types import AuthConfig, AuthType


class StubResolver:
    """Fake SecretResolver returning a canned value per scheme."""

    def __init__(self, schemes, value):
        self._schemes = frozenset(schemes)
        self._value = value

    @property
    def schemes(self):
        return self._schemes

    def resolve(self, scheme, path, key):
        return self._value


@pytest.fixture
def no_resolvers():
    """Force an empty resolver registry for the duration of a test."""
    from jaato_server.shared.plugins.subagent import config as subagent_config

    original = subagent_config._resolvers
    subagent_config._resolvers = {}
    try:
        yield
    finally:
        subagent_config._resolvers = original


@pytest.fixture
def pass_resolver_returns(request):
    """Register a stub ``pass://`` resolver returning ``request.param``."""
    from jaato_server.shared.plugins.subagent import config as subagent_config

    stub = StubResolver(("pass",), request.param)
    original = subagent_config._resolvers
    subagent_config._resolvers = {"pass": stub}
    try:
        yield stub
    finally:
        subagent_config._resolvers = original


def _service(auth):
    from ..plugin import ServiceConnectorPlugin  # noqa: F401 (import cycle guard)
    from ..types import ServiceConfig
    return ServiceConfig(name="svc", base_url="https://x", auth=auth)


def _build(auth, attempts):
    from ..plugin import ServiceConnectorPlugin
    return ServiceConnectorPlugin._build_auth_context(
        _service(auth), resolved=False, attempts=attempts
    )


# --------------------------------------------------------------------------
# resolution_failure_hint — unit, one branch per case
# --------------------------------------------------------------------------

class TestResolutionFailureHint:
    def test_env_name_case_is_not_a_missing_resolver(self):
        """(b) A literal env-var name says so, and NEVER blames a resolver."""
        source = AuthSource(
            method=METHOD_BEARER, raw_field="TYPESAFE_API_KEY",
            kind="env", env_var="TYPESAFE_API_KEY",
        )
        hint = resolution_failure_hint(source)
        assert "TYPESAFE_API_KEY" in hint
        assert "literal environment-variable NAME" in hint
        assert "configuration issue" in hint
        # The whole point: this case must not send the user to install a
        # resolver (it may say a resolver was not *consulted*, which is the
        # opposite advice).
        assert "no resolver is registered" not in hint
        assert "jaato-premium" not in hint

    def test_uri_no_resolver_case_names_missing_resolver(self, no_resolvers):
        """(a) A secret URI whose scheme has no resolver: install it."""
        source = AuthSource(
            method=METHOD_BEARER, raw_field="vault://secret/key",
            kind="uri", uri="vault://secret/key",
        )
        hint = resolution_failure_hint(source)
        assert "vault://secret/key" in hint
        assert "no resolver is registered" in hint
        assert "`vault://` scheme" in hint

    def test_pass_uri_no_resolver_names_jaato_premium(self, no_resolvers):
        source = AuthSource(
            method=METHOD_BEARER, raw_field="pass://jaato/tok",
            kind="uri", uri="pass://jaato/tok",
        )
        hint = resolution_failure_hint(source)
        assert "no resolver is registered" in hint
        assert "jaato-premium" in hint

    @pytest.mark.parametrize("pass_resolver_returns", [""], indirect=True)
    def test_uri_with_resolver_returning_empty_is_case_c(
        self, pass_resolver_returns
    ):
        """(c) Resolver registered but empty: fix the secret, not the config."""
        source = AuthSource(
            method=METHOD_BEARER, raw_field="pass://jaato/tok",
            kind="uri", uri="pass://jaato/tok",
        )
        hint = resolution_failure_hint(source)
        assert "IS registered" in hint
        assert "returned an empty value" in hint
        # It must NOT read as a missing resolver.
        assert "no resolver is registered" not in hint

    def test_none_source_falls_back_to_generic_env_message(self):
        hint = resolution_failure_hint(None)
        assert "not set in this session" in hint

    def test_the_three_uri_and_env_wordings_are_distinct(self, no_resolvers):
        env = resolution_failure_hint(AuthSource(
            method=METHOD_BEARER, raw_field="X", kind="env", env_var="X"))
        uri_missing = resolution_failure_hint(AuthSource(
            method=METHOD_BEARER, raw_field="vault://a/b",
            kind="uri", uri="vault://a/b"))
        assert env != uri_missing


# --------------------------------------------------------------------------
# End-to-end through _build_auth_context, the path 401/AuthError surfaces
# --------------------------------------------------------------------------

class TestBuildAuthContextDistinguishes:
    def test_env_var_case_reaches_the_hint(self, no_resolvers):
        """Reporter's case: value_env is a literal name, env unset."""
        auth = AuthConfig(type=AuthType.BEARER, value_env="TYPESAFE_API_KEY")
        manager = AuthManager()
        with pytest.raises(AuthError) as exc:
            manager.get_auth_headers(auth, service_name="x")
        ctx = _build(auth, exc.value.attempts)
        assert ctx["credentials_resolved"] is False
        assert "literal environment-variable NAME" in ctx["hint"]
        assert "TYPESAFE_API_KEY" in ctx["hint"]
        assert "no resolver is registered" not in ctx["hint"]
        assert "jaato-premium" not in ctx["hint"]

    def test_uri_no_resolver_case_reaches_the_hint(self, no_resolvers):
        auth = AuthConfig(type=AuthType.BEARER, value_env="vault://no-res/key")
        manager = AuthManager()
        with pytest.raises(AuthError) as exc:
            manager.get_auth_headers(auth, service_name="x")
        ctx = _build(auth, exc.value.attempts)
        assert "no resolver is registered" in ctx["hint"]
        assert "vault://no-res/key" in ctx["hint"]

    @pytest.mark.parametrize("pass_resolver_returns", [""], indirect=True)
    def test_uri_resolver_empty_case_reaches_the_hint(
        self, pass_resolver_returns
    ):
        auth = AuthConfig(type=AuthType.BEARER, value_env="pass://jaato/tok")
        manager = AuthManager()
        with pytest.raises(AuthError) as exc:
            manager.get_auth_headers(auth, service_name="x")
        ctx = _build(auth, exc.value.attempts)
        assert "IS registered" in ctx["hint"]
        assert "returned an empty value" in ctx["hint"]
