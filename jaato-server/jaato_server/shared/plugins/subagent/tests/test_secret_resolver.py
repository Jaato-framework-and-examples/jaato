"""Tests for secret URI resolution in expand_variables / _expand_string."""

import os
import threading
import time
from typing import FrozenSet, Optional
from unittest.mock import patch

import pytest

from ..config import (
    SecretResolutionError,
    SecretResolver,
    _expand_string,
    _resolve_secret_uri,
    _SECRET_URI_RE,
    _discover_secret_resolvers,
    looks_like_unresolved_secret_uri,
    looks_like_malformed_secret_uri,
    expand_variables,
    reset_secret_resolvers,
)


# ---------------------------------------------------------------------------
# Helpers — fake resolvers for testing
# ---------------------------------------------------------------------------

class FakeVaultResolver:
    """Test resolver for vault:// URIs."""

    @property
    def schemes(self) -> FrozenSet[str]:
        return frozenset({"vault"})

    def resolve(self, scheme: str, path: str, key: Optional[str] = None) -> str:
        if path == "secret/myapp" and key == "db_password":
            return "s3cret_from_vault"
        if path == "secret/myapp" and key is None:
            return '{"db_password": "s3cret"}'
        raise SecretResolutionError(
            f"{scheme}://{path}{'#' + key if key else ''}",
            "not found",
        )


class FakeMultiSchemeResolver:
    """Test resolver handling multiple schemes."""

    @property
    def schemes(self) -> FrozenSet[str]:
        return frozenset({"awssm", "gcpsm"})

    def resolve(self, scheme: str, path: str, key: Optional[str] = None) -> str:
        return f"resolved:{scheme}:{path}:{key}"


class FailingResolver:
    """Test resolver that always raises."""

    @property
    def schemes(self) -> FrozenSet[str]:
        return frozenset({"failscheme"})

    def resolve(self, scheme: str, path: str, key: Optional[str] = None) -> str:
        raise RuntimeError("backend unreachable")


# ---------------------------------------------------------------------------
# URI regex tests
# ---------------------------------------------------------------------------

class TestSecretURIRegex:
    """Tests for _SECRET_URI_RE pattern matching."""

    def test_simple_uri(self):
        m = _SECRET_URI_RE.match("vault://secret/myapp")
        assert m
        assert m.group("scheme") == "vault"
        assert m.group("path") == "secret/myapp"
        assert m.group("key") is None

    def test_uri_with_key(self):
        m = _SECRET_URI_RE.match("vault://secret/myapp#db_password")
        assert m
        assert m.group("scheme") == "vault"
        assert m.group("path") == "secret/myapp"
        assert m.group("key") == "db_password"

    def test_awssm_uri(self):
        m = _SECRET_URI_RE.match("awssm://prod/myapp/db_password")
        assert m
        assert m.group("scheme") == "awssm"
        assert m.group("path") == "prod/myapp/db_password"

    def test_sops_uri(self):
        m = _SECRET_URI_RE.match("sops://secrets.yaml#db_password")
        assert m
        assert m.group("scheme") == "sops"
        assert m.group("path") == "secrets.yaml"
        assert m.group("key") == "db_password"

    def test_keyring_uri(self):
        m = _SECRET_URI_RE.match("keyring://myapp/db_password")
        assert m
        assert m.group("scheme") == "keyring"
        assert m.group("path") == "myapp/db_password"

    def test_not_a_uri(self):
        assert _SECRET_URI_RE.match("just a string") is None
        assert _SECRET_URI_RE.match("localhost:5432") is None
        assert _SECRET_URI_RE.match("${VAR}") is None

    def test_http_not_matched_as_secret(self):
        """http/https are valid schemes but won't have resolvers — that's fine."""
        m = _SECRET_URI_RE.match("https://example.com/path")
        assert m  # regex matches, but no resolver will be registered
        assert m.group("scheme") == "https"

    def test_scheme_with_digits_and_hyphens(self):
        m = _SECRET_URI_RE.match("my-vault2://path/to/secret")
        assert m
        assert m.group("scheme") == "my-vault2"


# ---------------------------------------------------------------------------
# _resolve_secret_uri tests
# ---------------------------------------------------------------------------

class TestResolveSecretURI:
    """Tests for _resolve_secret_uri dispatch."""

    def setup_method(self):
        reset_secret_resolvers()

    def teardown_method(self):
        reset_secret_resolvers()

    def test_non_uri_passthrough(self):
        """Non-URI strings pass through unchanged."""
        assert _resolve_secret_uri("plain value") == "plain value"
        assert _resolve_secret_uri("localhost") == "localhost"
        assert _resolve_secret_uri("") == ""

    def test_no_resolver_registered(self):
        """URI with no matching resolver passes through unchanged."""
        # Force empty registry
        from .. import config as config_module
        config_module._resolvers = {}
        assert _resolve_secret_uri("vault://secret/x") == "vault://secret/x"

    def test_resolver_dispatched(self):
        """Registered resolver is called for matching scheme."""
        from .. import config as config_module
        config_module._resolvers = {"vault": FakeVaultResolver()}
        result = _resolve_secret_uri("vault://secret/myapp#db_password")
        assert result == "s3cret_from_vault"

    def test_resolver_not_found_raises(self):
        """SecretResolutionError from resolver propagates."""
        from .. import config as config_module
        config_module._resolvers = {"vault": FakeVaultResolver()}
        with pytest.raises(SecretResolutionError, match="not found"):
            _resolve_secret_uri("vault://secret/nonexistent#missing")

    def test_resolver_exception_wrapped(self):
        """Non-SecretResolutionError exceptions are wrapped."""
        from .. import config as config_module
        config_module._resolvers = {"failscheme": FailingResolver()}
        with pytest.raises(SecretResolutionError, match="backend unreachable"):
            _resolve_secret_uri("failscheme://anything")

    def test_multi_scheme_resolver(self):
        """A resolver handling multiple schemes works for each."""
        from .. import config as config_module
        resolver = FakeMultiSchemeResolver()
        config_module._resolvers = {s: resolver for s in resolver.schemes}

        assert _resolve_secret_uri("awssm://prod/key") == "resolved:awssm:prod/key:None"
        assert _resolve_secret_uri("gcpsm://proj/secret#field") == "resolved:gcpsm:proj/secret:field"

    # -- Server 0.6.57+: skip-conditions ------------------------------------

    def test_skip_unresolved_variable_substitution(self):
        """${VAR} placeholders short-circuit before regex match.

        Empirical regression — handoff_test cascade in 7:4 emitted
        ``http://127.0.0.1:${ANTIFRAUDE_PORT}`` before the env-file
        substitution layer resolved the placeholder.  The secret-URI
        regex matched the ``http://`` prefix and routed the literal
        through the (empty) resolver registry, returning the unsubstituted
        URL with the ``${VAR}`` still embedded.  Skip cleanly when a
        ``${`` is present.
        """
        # No resolvers — but skip happens before resolver lookup anyway.
        assert _resolve_secret_uri("http://localhost:${PORT}") == "http://localhost:${PORT}"
        assert _resolve_secret_uri("vault://${ENV}/secret") == "vault://${ENV}/secret"
        assert _resolve_secret_uri("${WHOLE_URL}") == "${WHOLE_URL}"

    def test_skip_network_schemes(self):
        """Standard network schemes are literal URLs, not secret references.

        Even with a (non-existent) ``http`` resolver, the dispatch must
        skip — these schemes carry HTTP traffic, not credentials.
        """
        from .. import config as config_module
        # Even pretend there were an http resolver — should still skip.
        config_module._resolvers = {"http": FakeMultiSchemeResolver()}

        assert _resolve_secret_uri("http://example.com/path") == "http://example.com/path"
        assert _resolve_secret_uri("https://example.com/path") == "https://example.com/path"
        assert _resolve_secret_uri("ws://localhost:8080/socket") == "ws://localhost:8080/socket"
        assert _resolve_secret_uri("wss://example.com/socket") == "wss://example.com/socket"
        assert _resolve_secret_uri("ftp://files.example.com/data") == "ftp://files.example.com/data"
        assert _resolve_secret_uri("ftps://files.example.com/data") == "ftps://files.example.com/data"

    def test_secret_schemes_still_resolve_with_resolver(self):
        """The skip is targeted — non-network schemes (vault, awssm, etc.) still dispatch."""
        from .. import config as config_module
        config_module._resolvers = {"vault": FakeVaultResolver()}

        # Vault still works
        assert (
            _resolve_secret_uri("vault://secret/myapp#db_password")
            == "s3cret_from_vault"
        )


# ---------------------------------------------------------------------------
# Integration with _expand_string and expand_variables
# ---------------------------------------------------------------------------

class TestExpandStringWithSecrets:
    """Tests that _expand_string chains variable expansion → secret resolution."""

    def setup_method(self):
        reset_secret_resolvers()

    def teardown_method(self):
        reset_secret_resolvers()

    def test_literal_secret_uri_resolved(self):
        """A config value that is a secret URI is resolved."""
        from .. import config as config_module
        config_module._resolvers = {"vault": FakeVaultResolver()}

        result = _expand_string("vault://secret/myapp#db_password", {})
        assert result == "s3cret_from_vault"

    def test_variable_expands_to_secret_uri(self):
        """${VAR} that resolves to a secret URI triggers resolution."""
        from .. import config as config_module
        config_module._resolvers = {"vault": FakeVaultResolver()}

        with patch.dict(os.environ, {"DB_SECRET": "vault://secret/myapp#db_password"}):
            result = _expand_string("${DB_SECRET}", {})
            assert result == "s3cret_from_vault"

    def test_embedded_uri_not_resolved(self):
        """Secret URIs embedded in larger strings are NOT resolved."""
        from .. import config as config_module
        config_module._resolvers = {"vault": FakeVaultResolver()}

        # URI embedded in a prefix — should stay as-is
        result = _expand_string("prefix vault://secret/myapp#db_password", {})
        assert "vault://" in result  # Not resolved

    def test_no_resolvers_installed_passthrough(self):
        """Without resolvers, URI-like strings pass through unchanged."""
        from .. import config as config_module
        config_module._resolvers = {}

        result = _expand_string("vault://secret/myapp#db_password", {})
        assert result == "vault://secret/myapp#db_password"

    def test_expand_variables_dict_with_secrets(self):
        """expand_variables resolves secrets in dict values."""
        from .. import config as config_module
        config_module._resolvers = {"vault": FakeVaultResolver()}

        input_dict = {
            "host": "localhost",
            "password": "vault://secret/myapp#db_password",
        }
        result = expand_variables(input_dict)
        assert result["host"] == "localhost"
        assert result["password"] == "s3cret_from_vault"

    def test_expand_variables_list_with_secrets(self):
        """expand_variables resolves secrets in list items."""
        from .. import config as config_module
        config_module._resolvers = {"vault": FakeVaultResolver()}

        input_list = ["vault://secret/myapp#db_password", "plain"]
        result = expand_variables(input_list)
        assert result[0] == "s3cret_from_vault"
        assert result[1] == "plain"


# ---------------------------------------------------------------------------
# Entry point discovery tests
# ---------------------------------------------------------------------------

class TestDiscoverSecretResolvers:
    """Tests for _discover_secret_resolvers entry point loading."""

    def setup_method(self):
        reset_secret_resolvers()

    def teardown_method(self):
        reset_secret_resolvers()

    def test_no_entry_points_returns_empty(self):
        """No premium package installed → empty registry."""
        resolvers = _discover_secret_resolvers()
        # In test environment, no entry points should be registered
        assert isinstance(resolvers, dict)

    def test_caching(self):
        """Second call returns cached result without re-scanning."""
        r1 = _discover_secret_resolvers()
        r2 = _discover_secret_resolvers()
        assert r1 is r2  # Same dict object


# ---------------------------------------------------------------------------
# SecretResolver protocol conformance
# ---------------------------------------------------------------------------

class TestSecretResolverProtocol:
    """Verify that test helpers satisfy the SecretResolver protocol."""

    def test_fake_vault_is_resolver(self):
        assert isinstance(FakeVaultResolver(), SecretResolver)

    def test_fake_multi_is_resolver(self):
        assert isinstance(FakeMultiSchemeResolver(), SecretResolver)

    def test_failing_is_resolver(self):
        assert isinstance(FailingResolver(), SecretResolver)


class TestLooksLikeUnresolvedSecretURI:
    """looks_like_unresolved_secret_uri — the provider-boundary fail-loud gate.

    True only for a non-network ``scheme://`` (an unresolved secret-URI that
    passed through because no resolver is registered).  Resolved secrets are
    plain strings (False); network URLs / ${VAR} / non-URIs are False.
    """

    def test_true_for_secret_scheme_uris(self):
        assert looks_like_unresolved_secret_uri("pass://jaato/nebius/api-key")
        assert looks_like_unresolved_secret_uri("vault://secret/x#k")
        assert looks_like_unresolved_secret_uri("awssm://prod/db")
        assert looks_like_unresolved_secret_uri("my-vault2://path/to/secret")

    def test_false_for_plain_strings(self):
        # A resolved credential is a plain string.
        assert not looks_like_unresolved_secret_uri("nbk-a-real-looking-key")
        assert not looks_like_unresolved_secret_uri("sk-ant-api03-xyz")
        assert not looks_like_unresolved_secret_uri("")

    def test_false_for_network_schemes(self):
        # http/ws are literal URLs, not secret indirections (e.g. self-hosted
        # base_url) — must not trip the credential gate.
        assert not looks_like_unresolved_secret_uri("https://api.example.com/v1")
        assert not looks_like_unresolved_secret_uri("ws://localhost:8080")

    def test_false_for_pending_var_substitution(self):
        assert not looks_like_unresolved_secret_uri("pass://${ENV}/key")
        assert not looks_like_unresolved_secret_uri("${WHOLE_KEY}")

    def test_false_for_non_str(self):
        assert not looks_like_unresolved_secret_uri(None)
        assert not looks_like_unresolved_secret_uri(12345)


class TestLooksLikeMalformedSecretURI:
    """looks_like_malformed_secret_uri — catches the ``//``-dropped typo.

    A single-colon ``pass:x`` (meant as ``pass://x``) is invisible to the
    resolver machinery (regex miss → passed through literally) and would leak
    to the provider as a bearer token, producing the confusing upstream 401 the
    ``//`` gate exists to prevent.  This detector flags it — but ONLY when the
    scheme is an actively registered resolver, so there are no false positives
    on hosts without that resolver and no hardcoded scheme list.
    """

    def setup_method(self):
        from .. import config as config_module
        reset_secret_resolvers()
        # Register a fake ``pass`` resolver so the scheme is "known".
        config_module._resolvers = {"pass": FakeVaultResolver()}

    def teardown_method(self):
        reset_secret_resolvers()

    def test_flags_single_colon_for_registered_scheme(self):
        # The exact user typo: pass:... instead of pass://...
        assert looks_like_malformed_secret_uri(
            "pass:jaato/openrouter/api-key") == "pass"

    def test_none_for_wellformed_double_slash(self):
        # Well-formed ``scheme://`` is the OTHER predicate's job, not malformed.
        assert looks_like_malformed_secret_uri(
            "pass://jaato/openrouter/api-key") is None

    def test_none_for_unregistered_scheme(self):
        # 'vault' single-colon but no vault resolver registered here → a plain
        # ``word:word`` value must be left untouched (no false positive).
        assert looks_like_malformed_secret_uri("vault:secret/x") is None

    def test_none_for_network_scheme(self):
        assert looks_like_malformed_secret_uri("http:example.com") is None
        assert looks_like_malformed_secret_uri("https://api.example.com/v1") is None

    def test_none_for_placeholder_and_plain(self):
        assert looks_like_malformed_secret_uri("pass:${ENV}/key") is None
        assert looks_like_malformed_secret_uri("sk-or-abcdef") is None
        # A vendor/model id with a ``:tag`` suffix must not be mistaken for a
        # scheme (the '/' and '.' aren't valid scheme chars).
        assert looks_like_malformed_secret_uri("google/gemini-2.5-flash:free") is None

    def test_none_for_non_str(self):
        assert looks_like_malformed_secret_uri(None) is None
        assert looks_like_malformed_secret_uri(12345) is None


# ---------------------------------------------------------------------------
# Concurrent first use
# ---------------------------------------------------------------------------

class TestConcurrentDiscovery:
    """The registry must never be observable half-built.

    Discovery is slow -- ``entry_points()`` scans installed distributions and
    ``ep.load()`` imports jaato-premium.  The cache used to be published
    EMPTY at the top of the function and filled afterwards, so a second
    thread arriving during that window took the ``is not None`` fast path and
    received an empty registry: it reported "(available: none)" and passed a
    literal ``pass://`` URI through to a provider as its api_key.

    Reported from a cold daemon whose first two sessions started 1ms apart:
    2 occurrences for 2 cold starts, 0 for 3 warm ones.  The distinguishing
    condition is concurrency at FIRST USE, not elapsed time -- once the
    registry is populated it is never empty again, which is why a warm daemon
    never shows it.
    """

    def setup_method(self):
        reset_secret_resolvers()

    def teardown_method(self):
        reset_secret_resolvers()

    def _slow_discovery(self, barrier, delay=0.2):
        """Stand in for the real entry-point scan, with its latency."""
        def _fake_uncached():
            barrier.wait(timeout=5)   # both threads are now inside
            time.sleep(delay)         # ...the window the racer used to hit
            return {"pass": FakeVaultResolver()}
        return _fake_uncached

    def test_second_caller_never_sees_an_empty_registry(self):
        """A caller arriving mid-discovery waits, and gets the full registry.

        Fails against the pre-fix code: the second thread returned ``{}``.
        """
        from .. import config as config_module

        started = threading.Barrier(2)
        seen = []

        def _slow():
            started.wait(timeout=5)
            time.sleep(0.2)
            return {"pass": FakeVaultResolver()}

        def _first():
            seen.append(("first", dict(_discover_secret_resolvers())))

        def _second():
            # Arrive while the first thread is inside discovery.
            started.wait(timeout=5)
            seen.append(("second", dict(_discover_secret_resolvers())))

        with patch.object(config_module,
                          "_discover_secret_resolvers_uncached",
                          _slow):
            t1 = threading.Thread(target=_first)
            t2 = threading.Thread(target=_second)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        assert len(seen) == 2, f"a thread did not finish: {seen}"
        for who, registry in seen:
            assert "pass" in registry, (
                f"the {who} caller saw a registry without 'pass' "
                f"(available: {', '.join(sorted(registry)) or 'none'}) -- "
                f"a partially-built registry was published"
            )

    def test_discovery_runs_once_under_concurrency(self):
        """Discovery runs once no matter how many callers arrive at once.

        NOT a guard for the race above -- this passes against the pre-fix
        code too, because publishing early also stopped the second caller
        re-scanning.  It guards the caching itself.
        """
        from .. import config as config_module

        calls = []

        def _counting():
            calls.append(1)
            time.sleep(0.05)
            return {"pass": FakeVaultResolver()}

        with patch.object(config_module,
                          "_discover_secret_resolvers_uncached",
                          _counting):
            threads = [threading.Thread(target=_discover_secret_resolvers)
                       for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert len(calls) == 1, f"discovery ran {len(calls)} times, want 1"

    def test_empty_result_is_still_cached(self):
        """"Discovered, and there are none" must not re-scan every call.

        The fix distinguishes absent (``None``) from empty (``{}``); an empty
        result is a real answer and stays cached.

        Also passes pre-fix; it pins the absent-vs-empty distinction the fix
        now depends on, so a later simplification cannot quietly drop it.
        """
        from .. import config as config_module

        calls = []

        def _finds_nothing():
            calls.append(1)
            return {}

        with patch.object(config_module,
                          "_discover_secret_resolvers_uncached",
                          _finds_nothing):
            assert _discover_secret_resolvers() == {}
            assert _discover_secret_resolvers() == {}

        assert len(calls) == 1, "an empty registry was re-discovered"
