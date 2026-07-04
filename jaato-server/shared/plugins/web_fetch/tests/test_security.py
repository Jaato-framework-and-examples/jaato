"""Unit tests for web_fetch security guards (host-bound secret expansion + SSRF).

These pin the interim application-layer defense for the web_fetch
credential-exfiltration / SSRF findings. All functions are pure (env and DNS
resolver injected), so the tests are hermetic — no real network or process env.
"""

from shared.plugins.web_fetch.security import (
    build_trusted_patterns,
    contains_var,
    expand_headers_bound,
    host_matches,
    validate_target_host,
)


# --- contains_var ---------------------------------------------------------

def test_contains_var_detects_placeholder():
    assert contains_var("Bearer ${TOKEN}")
    assert contains_var("${A}")
    assert not contains_var("Bearer static-token")
    assert not contains_var("price is $5")   # lone $ is not ${...}
    assert not contains_var("")


# --- host_matches ---------------------------------------------------------

def test_host_matches_exact_case_insensitive():
    assert host_matches("api.github.com", "api.github.com")
    assert host_matches("API.GitHub.com", "api.github.com")
    assert not host_matches("evil.com", "api.github.com")


def test_host_matches_wildcard_covers_apex_and_subdomain():
    assert host_matches("example.com", "*.example.com")       # apex
    assert host_matches("api.example.com", "*.example.com")   # subdomain
    assert host_matches("a.b.example.com", "*.example.com")   # nested
    assert not host_matches("example.com.evil.com", "*.example.com")
    assert not host_matches("notexample.com", "*.example.com")


def test_host_matches_empty_inputs():
    assert not host_matches("", "api.github.com")
    assert not host_matches("api.github.com", "")


# --- build_trusted_patterns ----------------------------------------------

def test_build_trusted_patterns_unions_and_dedupes():
    bindings = {"T": ["api.x.com", "shared.com"], "U": ["shared.com"]}
    internal = ["internal.local", "shared.com"]
    out = build_trusted_patterns(bindings, internal)
    assert out.count("shared.com") == 1
    assert set(out) == {"api.x.com", "shared.com", "internal.local"}


# --- expand_headers_bound -------------------------------------------------

def test_expand_headers_unbound_var_is_refused():
    headers = {"Authorization": "Bearer ${SECRET}"}
    out, err = expand_headers_bound(headers, "api.x.com", {}, {"SECRET": "s"})
    assert out is None
    assert "no host binding" in err.lower()


def test_expand_headers_bound_var_wrong_host_is_refused():
    headers = {"Authorization": "Bearer ${TOKEN}"}
    bindings = {"TOKEN": ["api.github.com"]}
    out, err = expand_headers_bound(headers, "attacker.example", bindings, {"TOKEN": "s"})
    assert out is None
    assert "not permitted for host" in err.lower()


def test_expand_headers_bound_var_right_host_expands():
    headers = {"Authorization": "Bearer ${TOKEN}", "X-Static": "keep"}
    bindings = {"TOKEN": ["api.github.com"]}
    out, err = expand_headers_bound(
        headers, "api.github.com", bindings, {"TOKEN": "real-secret"},
    )
    assert err is None
    assert out["Authorization"] == "Bearer real-secret"
    assert out["X-Static"] == "keep"


def test_expand_headers_bound_var_missing_in_env_is_refused():
    headers = {"Authorization": "Bearer ${TOKEN}"}
    bindings = {"TOKEN": ["api.github.com"]}
    out, err = expand_headers_bound(headers, "api.github.com", bindings, {})
    assert out is None
    assert "not set in the environment" in err.lower()


def test_expand_headers_wildcard_host_binding():
    headers = {"Authorization": "token ${GH}"}
    bindings = {"GH": ["*.github.com"]}
    out, err = expand_headers_bound(headers, "api.github.com", bindings, {"GH": "x"})
    assert err is None and out["Authorization"] == "token x"


def test_expand_headers_no_placeholder_passthrough():
    headers = {"Accept": "application/json"}
    out, err = expand_headers_bound(headers, "any.host", {}, {})
    assert err is None
    assert out == {"Accept": "application/json"}


def test_expand_headers_non_string_value_passthrough():
    headers = {"X-Num": 42}
    out, err = expand_headers_bound(headers, "any.host", {}, {})
    assert err is None
    assert out["X-Num"] == 42


def test_expand_headers_multiple_vars_one_unbound_refuses_all():
    headers = {"A": "${BOUND}", "B": "${UNBOUND}"}
    bindings = {"BOUND": ["h.com"]}
    out, err = expand_headers_bound(headers, "h.com", bindings, {"BOUND": "x"})
    assert out is None and err is not None


# --- validate_target_host (SSRF) -----------------------------------------

def _resolver(mapping):
    return lambda host: mapping[host]


def test_ssrf_disabled_allows_anything():
    assert validate_target_host(
        "169.254.169.254", [], block_private=False,
        resolver=_resolver({"169.254.169.254": ["169.254.169.254"]}),
    ) is None


def test_ssrf_blocks_cloud_metadata():
    err = validate_target_host(
        "metadata.evil", [], True,
        resolver=_resolver({"metadata.evil": ["169.254.169.254"]}),
    )
    assert err is not None and "non-public" in err.lower()


def test_ssrf_blocks_loopback():
    err = validate_target_host(
        "localhost", [], True, resolver=_resolver({"localhost": ["127.0.0.1"]}),
    )
    assert err is not None


def test_ssrf_blocks_private_rfc1918():
    err = validate_target_host(
        "intranet", [], True, resolver=_resolver({"intranet": ["10.1.2.3"]}),
    )
    assert err is not None


def test_ssrf_blocks_ipv4_mapped_ipv6():
    err = validate_target_host(
        "sneaky", [], True,
        resolver=_resolver({"sneaky": ["::ffff:169.254.169.254"]}),
    )
    assert err is not None, "IPv4-mapped IPv6 metadata address must be blocked"


def test_ssrf_allows_public_host():
    assert validate_target_host(
        "example.com", [], True, resolver=_resolver({"example.com": ["93.184.216.34"]}),
    ) is None


def test_ssrf_trusted_host_bypasses_even_if_private():
    # A declared internal host on a private IP is reachable — the resolver is
    # never even consulted for trusted hosts.
    def _boom(host):
        raise AssertionError("resolver must not be called for a trusted host")
    assert validate_target_host(
        "api.internal.company.com", ["api.internal.company.com"], True,
        resolver=_boom,
    ) is None


def test_ssrf_dns_failure_fails_open():
    # A hostname the resolver can't resolve fails OPEN: it can't be fetched
    # anyway, so nothing leaks, and we avoid a hard DNS dependency / over-block.
    def _fail(host):
        raise OSError("nxdomain")
    assert validate_target_host("nope.invalid", [], True, resolver=_fail) is None


def test_ssrf_ip_literal_metadata_blocked_without_resolver():
    # IP-literal targets are classified directly — no DNS needed.
    def _boom(host):
        raise AssertionError("resolver must not be called for an IP literal")
    assert validate_target_host("169.254.169.254", [], True, resolver=_boom) is not None
    assert validate_target_host("127.0.0.1", [], True, resolver=_boom) is not None
    assert validate_target_host("::1", [], True, resolver=_boom) is not None


def test_ssrf_mixed_ips_blocks_if_any_is_private():
    # DNS-rebinding style: one public + one private answer -> block.
    err = validate_target_host(
        "rebind", [], True,
        resolver=_resolver({"rebind": ["93.184.216.34", "127.0.0.1"]}),
    )
    assert err is not None
