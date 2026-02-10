"""Tests for shared.http.proxy — NO_PROXY / JAATO_NO_PROXY handling."""

import os
from unittest import mock

import pytest

from shared.http.proxy import (
    _get_no_proxy_entries,
    _matches_no_proxy,
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
