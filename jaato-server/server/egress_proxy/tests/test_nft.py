"""cgroup-nft egress: ruleset rendering + install/remove logic (§5.11d-v2).

Kernel-free: the pure renderer is asserted directly, and the manager's nft
shell-out is stubbed so install/remove logic is tested without touching
netfilter. The *enforcement* proof lives in scripts/verify_egress_nft.sh
(real-host, run as root).
"""

import subprocess
import types

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
    assert "ip6 daddr ::1 accept" in rs
    assert "counter reject" in rs
    # table name sanitized (no '-')
    assert "table inet jaato_egress_sess_1 {" in rs
    # host-safety: default-accept policy so non-cgroup traffic is untouched
    assert "policy accept;" in rs


def test_render_dns_closed_by_default_open_with_flag():
    assert "127.0.0.53" not in render_ruleset("s", "/sys/fs/cgroup/s", 443)
    assert "127.0.0.53" in render_ruleset(
        "s", "/sys/fs/cgroup/s", 443, allow_local_resolver=True)


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
