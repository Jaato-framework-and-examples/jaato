"""cgroup-nft egress: ruleset rendering + install/remove logic (§5.11d-v2).

Kernel-free: the pure renderer is asserted directly, and the manager's nft
shell-out is stubbed so install/remove logic is tested without touching
netfilter. The *enforcement* proof lives in scripts/verify_egress_nft.sh
(real-host, run as root).
"""

import subprocess

import pytest

from server.egress_proxy import nft
from server.egress_proxy.nft import (
    EgressNftManager,
    cgroup_rel_and_level,
    render_ruleset,
)


# ---- pure rendering ------------------------------------------------------

def test_cgroup_rel_and_level():
    assert cgroup_rel_and_level("/sys/fs/cgroup/jaato/jaato-ws-abc") == (
        "jaato/jaato-ws-abc", 2)
    assert cgroup_rel_and_level("/sys/fs/cgroup/session1") == ("session1", 1)


def test_render_contains_cgroup_match_and_proxy_port():
    rs = render_ruleset("sess-1", "/sys/fs/cgroup/jaato/jaato-ws-x", 45871)
    assert 'socket cgroupv2 level 2 "jaato/jaato-ws-x" jump gate' in rs
    assert "ip  daddr 127.0.0.1 tcp dport 45871 accept" in rs
    assert "counter reject" in rs
    # table name sanitized (no '-')
    assert "table inet jaato_egress_sess_1 {" in rs
    # host-safety: default-accept policy so non-cgroup traffic is untouched
    assert "policy accept;" in rs


# ---- #696: loopback is reachable ONLY at the proxy's address:port ---------

def test_render_never_opens_ipv6_loopback_wholesale():
    """Regression for #696.

    The gate used to carry an unconditional ``ip6 daddr ::1 accept`` beside the
    port-scoped IPv4 rule, which opened every service on IPv6 loopback to a
    confined session.  The proxy binds AF_INET, so no ip6 accept rule belongs
    in the ruleset at all — IPv6 must fall through to ``counter reject``.
    """
    rs = render_ruleset("sess-1", "/sys/fs/cgroup/jaato/jaato-ws-x", 45871)
    assert "::1" not in rs
    assert "ip6" not in rs
    # exactly one accept, and it is port-scoped
    accepts = [ln.strip() for ln in rs.splitlines() if ln.strip().endswith("accept")]
    assert accepts == ["ip  daddr 127.0.0.1 tcp dport 45871 accept"]


def test_render_accept_rule_follows_the_proxy_bind_family():
    """The family is derived from the bind address, not assumed.

    Today ``ConnectAllowlistProxy`` is AF_INET-only so this stays IPv4; if it
    ever binds ``::1`` the gate must narrow to that address AND port rather
    than opening the interface.
    """
    rs = render_ruleset("s", "/sys/fs/cgroup/s", 8443, proxy_host="::1")
    assert "ip6 daddr ::1 tcp dport 8443 accept" in rs
    assert "127.0.0.1" not in rs


def test_render_rejects_non_literal_proxy_host():
    """The ruleset is piped to ``nft -f -``; only IP literals may reach it."""
    for bad in ("localhost", "evil.example", "127.0.0.1 accept\n    ip daddr 0.0.0.0", ""):
        with pytest.raises(ValueError):
            render_ruleset("s", "/sys/fs/cgroup/s", 443, proxy_host=bad)


def test_render_rejects_out_of_range_port():
    for bad in (0, -1, 65536):
        with pytest.raises(ValueError):
            render_ruleset("s", "/sys/fs/cgroup/s", bad)


def test_render_denies_udp_by_omission():
    """No rule matches UDP, so QUIC / direct DNS hit ``counter reject``."""
    rs = render_ruleset("s", "/sys/fs/cgroup/s", 443)
    assert "udp" not in rs


def test_render_dns_closed_by_default_open_with_flag():
    assert "127.0.0.53" not in render_ruleset("s", "/sys/fs/cgroup/s", 443)
    rs = render_ruleset("s", "/sys/fs/cgroup/s", 443, allow_local_resolver=True)
    assert "127.0.0.53" in rs


def test_render_dns_carveout_is_scoped_to_port_53():
    """#696 follow-up: the stub-resolver carve-out had no port predicate, so it
    opened every port on 127.0.0.53 rather than DNS."""
    rs = render_ruleset("s", "/sys/fs/cgroup/s", 443, allow_local_resolver=True)
    stub = [ln.strip() for ln in rs.splitlines() if "127.0.0.53" in ln]
    assert stub == [
        "ip  daddr 127.0.0.53 udp dport 53 accept",
        "ip  daddr 127.0.0.53 tcp dport 53 accept",
    ]


# ---- manager install/remove (stubbed nft) --------------------------------

class _StubManager(EgressNftManager):
    def __init__(self, isdir=True, rc=0, stderr=""):
        super().__init__(use_sudo=False)
        self.calls = []
        self._isdir = isdir
        self._rc = rc
        self._stderr = stderr

    def _nft(self, args, stdin=None):
        self.calls.append((args, stdin))
        return subprocess.CompletedProcess(args, self._rc, "", self._stderr)


def _patch_isdir(monkeypatch, value):
    monkeypatch.setattr(nft.os.path, "isdir", lambda p: value)


def test_install_skips_when_cgroup_absent(monkeypatch):
    _patch_isdir(monkeypatch, False)
    m = _StubManager()
    ok = m.install("s1", "/sys/fs/cgroup/jaato/jaato-ws-s1", 443)
    assert ok is False
    assert m.calls == []            # no nft invoked


def test_install_loads_ruleset_when_cgroup_present(monkeypatch):
    _patch_isdir(monkeypatch, True)
    m = _StubManager()
    ok = m.install("s1", "/sys/fs/cgroup/jaato/jaato-ws-s1", 8080)
    assert ok is True
    # a delete (idempotent pre-clean) then the -f - load
    load = [c for c in m.calls if c[0] == ["-f", "-"]]
    assert load and "tcp dport 8080 accept" in load[0][1]
    assert m._installed["s1"] == "jaato_egress_s1"


def test_install_failure_returns_false(monkeypatch):
    _patch_isdir(monkeypatch, True)
    m = _StubManager(rc=1, stderr="boom")
    assert m.install("s1", "/sys/fs/cgroup/x", 443) is False
    assert "s1" not in m._installed


def test_remove_deletes_table(monkeypatch):
    _patch_isdir(monkeypatch, True)
    m = _StubManager()
    m.install("s1", "/sys/fs/cgroup/x", 443)
    m.calls.clear()
    m.remove("s1")
    assert any(c[0] == ["delete", "table", "inet", "jaato_egress_s1"] for c in m.calls)
    assert "s1" not in m._installed


def test_shutdown_removes_all(monkeypatch):
    _patch_isdir(monkeypatch, True)
    m = _StubManager()
    m.install("s1", "/sys/fs/cgroup/x", 443)
    m.install("s2", "/sys/fs/cgroup/y", 443)
    m.calls.clear()
    m.shutdown()
    deleted = {tuple(c[0]) for c in m.calls if c[0][0] == "delete"}
    assert ("delete", "table", "inet", "jaato_egress_s1") in deleted
    assert ("delete", "table", "inet", "jaato_egress_s2") in deleted
    assert m._installed == {}


def test_install_renders_the_proxy_bind_address(monkeypatch):
    """``install`` gates the address the proxy actually bound (#696)."""
    _patch_isdir(monkeypatch, True)
    m = _StubManager()
    assert m.install("s1", "/sys/fs/cgroup/x", 8080, proxy_host="::1") is True
    load = [c for c in m.calls if c[0] == ["-f", "-"]][0][1]
    assert "ip6 daddr ::1 tcp dport 8080 accept" in load


def test_install_refuses_non_literal_proxy_host(monkeypatch):
    """A host that cannot be rendered safely means no table, not a broad one."""
    _patch_isdir(monkeypatch, True)
    m = _StubManager()
    assert m.install("s1", "/sys/fs/cgroup/x", 443, proxy_host="localhost") is False
    assert [c for c in m.calls if c[0] == ["-f", "-"]] == []
    assert "s1" not in m._installed
