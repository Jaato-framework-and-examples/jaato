"""AllowlistConfig schema validation + host/port matching (§5.11a)."""

from server.egress_proxy.config import (
    AllowlistConfig,
    host_matches,
    validate_allowlist,
    MAX_HOSTS,
)


# ---- host_matches --------------------------------------------------------

def test_exact_match_case_insensitive():
    assert host_matches("api.anthropic.com", "api.anthropic.com")
    assert host_matches("API.Anthropic.COM", "api.anthropic.com")
    assert not host_matches("evil.com", "api.anthropic.com")


def test_wildcard_matches_sublabels_not_apex():
    assert host_matches("bar.foo.com", "*.foo.com")
    assert host_matches("bar.baz.foo.com", "*.foo.com")
    assert not host_matches("foo.com", "*.foo.com")          # apex excluded
    assert not host_matches("foo.com.evil.com", "*.foo.com")


def test_trailing_dot_normalized():
    assert host_matches("api.anthropic.com.", "api.anthropic.com")


# ---- AllowlistConfig.allows ---------------------------------------------

def test_allows_requires_host_and_port():
    cfg = AllowlistConfig(allowed_hosts=["api.x.com"], allowed_ports=[443])
    assert cfg.allows("api.x.com", 443)
    assert not cfg.allows("api.x.com", 80)      # port not allowed
    assert not cfg.allows("evil.com", 443)      # host not allowed


def test_from_dict_lowercases_and_defaults_port():
    cfg = AllowlistConfig.from_dict({"allowed_hosts": ["API.X.COM"]})
    assert cfg.allowed_hosts == ["api.x.com"]
    assert cfg.allowed_ports == [443]


# ---- validate_allowlist --------------------------------------------------

def test_valid_config_has_no_errors():
    assert validate_allowlist({
        "allowed_hosts": ["api.anthropic.com", "*.googleapis.com"],
        "allowed_ports": [443, 80],
    }) == []


def test_empty_hosts_rejected():
    assert validate_allowlist({"allowed_hosts": []})


def test_not_an_object_rejected():
    assert validate_allowlist(["api.x.com"])


def test_invalid_host_chars_rejected():
    errs = validate_allowlist({"allowed_hosts": ["evil.com/../path"]})
    assert any("invalid characters" in e for e in errs)


def test_interior_wildcard_rejected():
    errs = validate_allowlist({"allowed_hosts": ["foo.*.com"]})
    assert any("wildcard" in e.lower() or "*" in e for e in errs)


def test_double_wildcard_rejected():
    errs = validate_allowlist({"allowed_hosts": ["*.*.com"]})
    assert errs


def test_bad_port_rejected():
    assert validate_allowlist({"allowed_hosts": ["x.com"], "allowed_ports": [70000]})
    assert validate_allowlist({"allowed_hosts": ["x.com"], "allowed_ports": ["443"]})
    # bool is not a valid port even though it is an int subclass
    assert validate_allowlist({"allowed_hosts": ["x.com"], "allowed_ports": [True]})


def test_too_many_hosts_rejected():
    errs = validate_allowlist({"allowed_hosts": [f"h{i}.com" for i in range(MAX_HOSTS + 1)]})
    assert any("max" in e for e in errs)


def test_non_bool_flag_rejected():
    assert validate_allowlist({"allowed_hosts": ["x.com"], "deny_on_default": "yes"})
