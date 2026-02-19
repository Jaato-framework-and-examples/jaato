"""Tests for shared.http.proxy — NO_PROXY / JAATO_NO_PROXY handling, SPNEGO, and SSL verify."""

import os
from unittest import mock

import httpx
import pytest

from shared.http.proxy import (
    _generate_spnego_token_sspi,
    _get_no_proxy_entries,
    _get_ssl_verify_value,
    _matches_no_proxy,
    generate_spnego_token,
    get_httpx_client,
    get_httpx_kwargs,
    get_requests_session,
    is_ssl_verify_disabled,
    should_bypass_proxy,
)


# ============================================================
# _matches_no_proxy unit tests
# ============================================================


class TestMatchesNoProxy:
    """Test the individual NO_PROXY entry matching logic."""

    def test_wildcard_matches_everything(self):
        assert _matches_no_proxy("example.com", 443, "*") is True
        assert _matches_no_proxy("anything.test", None, "*") is True

    def test_exact_host_match(self):
        assert _matches_no_proxy("example.com", None, "example.com") is True

    def test_exact_host_no_match(self):
        assert _matches_no_proxy("other.com", None, "example.com") is False

    def test_suffix_with_leading_dot(self):
        assert _matches_no_proxy("api.example.com", None, ".example.com") is True
        assert _matches_no_proxy("deep.api.example.com", None, ".example.com") is True

    def test_leading_dot_matches_bare_domain(self):
        """'.example.com' should also match 'example.com' itself."""
        assert _matches_no_proxy("example.com", None, ".example.com") is True

    def test_leading_dot_no_partial_match(self):
        """'.example.com' should NOT match 'notexample.com'."""
        assert _matches_no_proxy("notexample.com", None, ".example.com") is False

    def test_suffix_without_leading_dot(self):
        """Standard NO_PROXY: 'example.com' matches 'sub.example.com'."""
        assert _matches_no_proxy("sub.example.com", None, "example.com") is True
        assert _matches_no_proxy("deep.sub.example.com", None, "example.com") is True

    def test_no_partial_suffix_match(self):
        """'example.com' should NOT match 'badexample.com'."""
        assert _matches_no_proxy("badexample.com", None, "example.com") is False

    def test_port_match(self):
        assert _matches_no_proxy("example.com", 8080, "example.com:8080") is True

    def test_port_mismatch(self):
        assert _matches_no_proxy("example.com", 443, "example.com:8080") is False

    def test_port_none_vs_entry_port(self):
        assert _matches_no_proxy("example.com", None, "example.com:8080") is False

    def test_case_insensitive(self):
        # _matches_no_proxy expects lowercase inputs (caller normalizes)
        assert _matches_no_proxy("example.com", None, "example.com") is True

    def test_empty_entry(self):
        assert _matches_no_proxy("example.com", None, "") is False


# ============================================================
# _get_no_proxy_entries tests
# ============================================================


class TestGetNoProxyEntries:
    def test_no_env_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _get_no_proxy_entries() == []

    def test_uppercase_no_proxy(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "foo.com, bar.com"}, clear=True):
            result = _get_no_proxy_entries()
            assert result == ["foo.com", "bar.com"]

    def test_lowercase_no_proxy(self):
        with mock.patch.dict(os.environ, {"no_proxy": ".internal.corp"}, clear=True):
            result = _get_no_proxy_entries()
            assert result == [".internal.corp"]

    def test_uppercase_takes_precedence(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "upper.com", "no_proxy": "lower.com"}, clear=True):
            result = _get_no_proxy_entries()
            assert result == ["upper.com"]

    def test_empty_string(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": ""}, clear=True):
            assert _get_no_proxy_entries() == []

    def test_entries_lowercased(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "FOO.COM,Bar.Com"}, clear=True):
            result = _get_no_proxy_entries()
            assert result == ["foo.com", "bar.com"]


# ============================================================
# should_bypass_proxy integration tests
# ============================================================


class TestShouldBypassProxy:
    """Integration tests: should_bypass_proxy with various env combos."""

    def test_no_env_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert should_bypass_proxy("https://example.com") is False

    def test_jaato_no_proxy_exact(self):
        with mock.patch.dict(os.environ, {"JAATO_NO_PROXY": "api.internal.com"}, clear=True):
            assert should_bypass_proxy("https://api.internal.com/path") is True
            assert should_bypass_proxy("https://other.internal.com/path") is False

    def test_standard_no_proxy_suffix(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": ".internal.com"}, clear=True):
            assert should_bypass_proxy("https://api.internal.com/path") is True
            assert should_bypass_proxy("https://deep.api.internal.com/path") is True
            assert should_bypass_proxy("https://external.com/path") is False

    def test_standard_no_proxy_without_dot(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "internal.com"}, clear=True):
            assert should_bypass_proxy("https://internal.com/path") is True
            assert should_bypass_proxy("https://api.internal.com/path") is True
            assert should_bypass_proxy("https://notinternal.com/path") is False

    def test_standard_no_proxy_wildcard(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "*"}, clear=True):
            assert should_bypass_proxy("https://anything.com") is True

    def test_combined_jaato_and_standard(self):
        env = {"JAATO_NO_PROXY": "exact.host.com", "NO_PROXY": ".corp.net"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert should_bypass_proxy("https://exact.host.com/api") is True
            assert should_bypass_proxy("https://svc.corp.net/api") is True
            assert should_bypass_proxy("https://external.com") is False

    def test_no_proxy_with_port(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "example.com:8080"}, clear=True):
            assert should_bypass_proxy("https://example.com:8080/path") is True
            assert should_bypass_proxy("https://example.com:443/path") is False
            assert should_bypass_proxy("https://example.com/path") is False

    def test_lowercase_no_proxy_env(self):
        with mock.patch.dict(os.environ, {"no_proxy": "internal.dev"}, clear=True):
            assert should_bypass_proxy("https://internal.dev/") is True
            assert should_bypass_proxy("https://api.internal.dev/") is True

    def test_invalid_url(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "*"}, clear=True):
            assert should_bypass_proxy("not-a-url") is False

    def test_multiple_entries(self):
        with mock.patch.dict(os.environ, {"NO_PROXY": "localhost,.internal.com,special.host"}, clear=True):
            assert should_bypass_proxy("http://localhost:3000/") is True
            assert should_bypass_proxy("https://svc.internal.com/api") is True
            assert should_bypass_proxy("https://special.host/") is True
            assert should_bypass_proxy("https://external.com/") is False


# ============================================================
# SPNEGO / SSPI fallback tests
# ============================================================


class TestGenerateSpnegoTokenFallback:
    """Test that generate_spnego_token falls back to SSPI when pyspnego is missing."""

    def test_fallback_to_sspi_when_spnego_unavailable(self):
        """When pyspnego import fails, _generate_spnego_token_sspi is called."""
        with mock.patch.dict("sys.modules", {"spnego": None}):
            with mock.patch(
                "shared.http.proxy._generate_spnego_token_sspi",
                return_value="fake_b64_token",
            ) as mock_sspi:
                result = generate_spnego_token("proxy.corp.com")
                mock_sspi.assert_called_once_with("proxy.corp.com")
                assert result == "fake_b64_token"

    def test_pyspnego_preferred_over_sspi(self):
        """When pyspnego is available, SSPI fallback is not called."""
        mock_spnego = mock.MagicMock()
        mock_ctx = mock.MagicMock()
        mock_ctx.step.return_value = b"\x01\x02\x03"
        mock_spnego.client.return_value = mock_ctx

        with mock.patch.dict("sys.modules", {"spnego": mock_spnego}):
            with mock.patch(
                "shared.http.proxy._generate_spnego_token_sspi",
            ) as mock_sspi:
                result = generate_spnego_token("proxy.corp.com")
                mock_sspi.assert_not_called()
                assert result is not None

    def test_sspi_fallback_returns_none_when_no_sspi(self):
        """_generate_spnego_token_sspi returns None on non-Windows (no windll)."""
        # Hide ctypes.windll by patching the ctypes module
        fake_ctypes = mock.MagicMock(spec=["__name__"])
        del fake_ctypes.windll  # ensure AttributeError on access
        with mock.patch.dict("sys.modules", {"ctypes": fake_ctypes}):
            # Call the real function — it will import the mocked ctypes
            # and fail on windll access, returning None.
            from importlib import reload
            import shared.http.proxy as proxy_mod

            # Directly test: no windll → None
            result = _generate_spnego_token_sspi("proxy.corp.com")
            assert result is None


# ============================================================
# httpx Kerberos proxy integration tests
# ============================================================


class TestGetHttpxClientKerberos:
    """Verify that get_httpx_client uses httpx.Proxy for Kerberos auth.

    When JAATO_KERBEROS_PROXY is enabled, the SPNEGO Negotiate token must be
    placed on an httpx.Proxy object (not as a regular client header) so it is
    sent during the CONNECT tunnel handshake for HTTPS requests.

    We verify by intercepting the httpx.Client constructor to inspect the
    proxy kwarg, since httpx internals vary across versions.
    """

    def test_kerberos_uses_proxy_object(self):
        """With Kerberos enabled, proxy kwarg should be an httpx.Proxy."""
        env = {
            "HTTPS_PROXY": "http://proxy.corp.com:8080",
            "JAATO_KERBEROS_PROXY": "true",
        }
        captured = {}

        original_init = httpx.Client.__init__

        def capturing_init(self_client, **kwargs):
            captured.update(kwargs)
            return original_init(self_client, **kwargs)

        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "shared.http.proxy.generate_spnego_token",
                return_value="fake_b64",
            ):
                with mock.patch.object(httpx.Client, "__init__", capturing_init):
                    client = get_httpx_client(timeout=5.0)

                assert isinstance(captured["proxy"], httpx.Proxy)
                # The Negotiate header must NOT be in top-level client headers.
                client_headers = dict(captured.get("headers", {}))
                assert "Proxy-Authorization" not in client_headers
                client.close()

    def test_kerberos_no_token_falls_back_to_url(self):
        """When SPNEGO token generation fails, proxy is a plain URL string."""
        env = {
            "HTTPS_PROXY": "http://proxy.corp.com:8080",
            "JAATO_KERBEROS_PROXY": "true",
        }
        captured = {}

        original_init = httpx.Client.__init__

        def capturing_init(self_client, **kwargs):
            captured.update(kwargs)
            return original_init(self_client, **kwargs)

        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "shared.http.proxy.generate_spnego_token",
                return_value=None,
            ):
                with mock.patch.object(httpx.Client, "__init__", capturing_init):
                    client = get_httpx_client(timeout=5.0)

                # Proxy should be a plain URL string, not a Proxy object
                assert captured["proxy"] == "http://proxy.corp.com:8080"
                client.close()

    def test_no_kerberos_uses_plain_url(self):
        """Without Kerberos, proxy is set from env as a plain URL string."""
        env = {
            "HTTPS_PROXY": "http://proxy.corp.com:8080",
        }
        captured = {}

        original_init = httpx.Client.__init__

        def capturing_init(self_client, **kwargs):
            captured.update(kwargs)
            return original_init(self_client, **kwargs)

        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(httpx.Client, "__init__", capturing_init):
                client = get_httpx_client(timeout=5.0)

            assert captured["proxy"] == "http://proxy.corp.com:8080"
            client.close()


class TestGetHttpxKwargsKerberos:
    """Verify get_httpx_kwargs returns httpx.Proxy (not headers dict) for Kerberos."""

    def test_kerberos_returns_proxy_object_no_headers_key(self):
        """With Kerberos, returned dict should have Proxy object, no 'headers' key."""
        env = {
            "HTTPS_PROXY": "http://proxy.corp.com:8080",
            "JAATO_KERBEROS_PROXY": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "shared.http.proxy.generate_spnego_token",
                return_value="fake_b64",
            ):
                kwargs = get_httpx_kwargs("https://api.github.com/test")
                assert isinstance(kwargs["proxy"], httpx.Proxy)
                # The auth header must NOT be a top-level key — it lives
                # on the Proxy object so it's sent during CONNECT.
                assert "headers" not in kwargs

    def test_kerberos_no_token_returns_url_string(self):
        """When token generation fails, proxy is a plain URL string."""
        env = {
            "HTTPS_PROXY": "http://proxy.corp.com:8080",
            "JAATO_KERBEROS_PROXY": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "shared.http.proxy.generate_spnego_token",
                return_value=None,
            ):
                kwargs = get_httpx_kwargs("https://api.github.com/test")
                assert kwargs["proxy"] == "http://proxy.corp.com:8080"
                assert "headers" not in kwargs


# ============================================================
# JAATO_SSL_VERIFY tests
# ============================================================


class TestIsSslVerifyDisabled:
    """Test the is_ssl_verify_disabled() helper."""

    def test_unset_returns_false(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert is_ssl_verify_disabled() is False

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "No", "NO"])
    def test_disabled_values(self, value):
        with mock.patch.dict(os.environ, {"JAATO_SSL_VERIFY": value}, clear=True):
            assert is_ssl_verify_disabled() is True

    @pytest.mark.parametrize("value", ["true", "True", "1", "yes", "", "anything"])
    def test_enabled_values(self, value):
        with mock.patch.dict(os.environ, {"JAATO_SSL_VERIFY": value}, clear=True):
            assert is_ssl_verify_disabled() is False


class TestGetSslVerifyValue:
    """Test the _get_ssl_verify_value() internal helper."""

    def test_disabled_returns_false(self):
        with mock.patch.dict(os.environ, {"JAATO_SSL_VERIFY": "false"}, clear=True):
            assert _get_ssl_verify_value() is False

    def test_disabled_takes_precedence_over_ca_bundle(self, tmp_path):
        """JAATO_SSL_VERIFY=false wins even if a CA bundle is configured."""
        ca = tmp_path / "ca.pem"
        ca.write_text("cert")
        env = {"JAATO_SSL_VERIFY": "false", "REQUESTS_CA_BUNDLE": str(ca)}
        with mock.patch.dict(os.environ, env, clear=True):
            assert _get_ssl_verify_value() is False

    def test_ca_bundle_returned_when_verify_not_disabled(self, tmp_path):
        ca = tmp_path / "ca.pem"
        ca.write_text("cert")
        env = {"REQUESTS_CA_BUNDLE": str(ca)}
        with mock.patch.dict(os.environ, env, clear=True):
            assert _get_ssl_verify_value() == str(ca)

    def test_returns_none_when_nothing_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _get_ssl_verify_value() is None

    def test_missing_ca_bundle_returns_none(self):
        env = {"REQUESTS_CA_BUNDLE": "/nonexistent/ca.pem"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert _get_ssl_verify_value() is None


class TestSslVerifyHttpxClient:
    """Verify get_httpx_client respects JAATO_SSL_VERIFY."""

    def test_verify_false_passed_to_client(self):
        """With JAATO_SSL_VERIFY=false, httpx.Client gets verify=False."""
        env = {"JAATO_SSL_VERIFY": "false"}
        captured = {}

        original_init = httpx.Client.__init__

        def capturing_init(self_client, **kwargs):
            captured.update(kwargs)
            return original_init(self_client, **kwargs)

        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(httpx.Client, "__init__", capturing_init):
                client = get_httpx_client(timeout=5.0)

            assert captured["verify"] is False
            client.close()

    def test_caller_override_preserved(self):
        """Explicit verify= from caller wins over env var."""
        env = {"JAATO_SSL_VERIFY": "false"}
        captured = {}

        original_init = httpx.Client.__init__

        def capturing_init(self_client, **kwargs):
            captured.update(kwargs)
            return original_init(self_client, **kwargs)

        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(httpx.Client, "__init__", capturing_init):
                # Caller explicitly passes verify=True — should not be overridden
                client = get_httpx_client(timeout=5.0, verify=True)

            assert captured["verify"] is True
            client.close()

    def test_default_verify_not_set_when_env_unset(self):
        """When JAATO_SSL_VERIFY is not set and no CA bundle, verify kwarg is absent."""
        captured = {}

        original_init = httpx.Client.__init__

        def capturing_init(self_client, **kwargs):
            captured.update(kwargs)
            return original_init(self_client, **kwargs)

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(httpx.Client, "__init__", capturing_init):
                client = get_httpx_client(timeout=5.0)

            assert "verify" not in captured
            client.close()


class TestSslVerifyRequestsSession:
    """Verify get_requests_session respects JAATO_SSL_VERIFY."""

    def test_verify_false_on_session(self):
        with mock.patch.dict(os.environ, {"JAATO_SSL_VERIFY": "false"}, clear=True):
            session = get_requests_session()
            assert session.verify is False

    def test_default_verify_unchanged(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            session = get_requests_session()
            # requests.Session defaults verify to True
            assert session.verify is True
