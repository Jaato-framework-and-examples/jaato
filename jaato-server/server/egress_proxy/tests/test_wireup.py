"""Wire-up glue: opt-in env injection + teardown (§5.11e)."""

import socket

import pytest

from server.egress_proxy import wireup
from server.egress_proxy.errors import EgressEnforcementError


def _reset():
    # Ensure a clean singleton per test.
    wireup.shutdown_all()
    wireup._manager = None


def test_no_allowlist_returns_empty_env_and_no_proxy():
    _reset()
    env, errs = wireup.egress_env_for_session("s1", None)
    assert env == {} and errs == []
    # nothing started
    assert wireup._manager is None or wireup.get_manager().get_proxy_url("s1") is None


def test_valid_allowlist_starts_proxy_and_sets_env():
    _reset()
    try:
        env, errs = wireup.egress_env_for_session(
            "s1", {"allowed_hosts": ["api.anthropic.com"], "allowed_ports": [443]})
        assert errs == []
        assert env["HTTPS_PROXY"].startswith("http://127.0.0.1:")
        assert env["HTTPS_PROXY"] == env["https_proxy"] == env["HTTP_PROXY"]
        assert "127.0.0.1" in env["NO_PROXY"]
        # proxy is actually listening
        port = int(env["HTTPS_PROXY"].rsplit(":", 1)[1])
        socket.create_connection(("127.0.0.1", port), 2).close()
    finally:
        _reset()


def test_invalid_allowlist_returns_errors_and_no_proxy():
    _reset()
    env, errs = wireup.egress_env_for_session("s1", {"allowed_hosts": []})
    assert env == {}
    assert errs  # non-empty
    assert wireup.get_manager().get_proxy_url("s1") is None
    _reset()


def test_teardown_stops_proxy():
    _reset()
    env, _ = wireup.egress_env_for_session(
        "s1", {"allowed_hosts": ["x.com"]})
    port = int(env["HTTPS_PROXY"].rsplit(":", 1)[1])
    wireup.egress_teardown("s1")
    assert wireup.get_manager().get_proxy_url("s1") is None
    try:
        socket.create_connection(("127.0.0.1", port), 1).close()
        raised = False
    except OSError:
        raised = True
    assert raised
    _reset()


def test_teardown_before_any_start_is_noop():
    _reset()
    wireup.egress_teardown("never")  # must not raise


# ---- nft hard-enforcement wire-up (opt-in) -------------------------------

class _CaptureNft:
    def __init__(self, ok=True):
        self.installed = []
        self.removed = []
        self._ok = ok
    def install(self, sid, cg, port, **kw):
        self.installed.append((sid, cg, port, kw.get("proxy_host"))); return self._ok
    def remove(self, sid):
        self.removed.append(sid)
    def shutdown(self):
        pass


def _reset_nft():
    _reset()
    wireup._nft_manager = None


def test_nft_not_touched_when_flag_off(monkeypatch):
    _reset_nft()
    monkeypatch.delenv("JAATO_EGRESS_NFT_ENFORCE", raising=False)
    cap = _CaptureNft()
    monkeypatch.setattr(wireup, "_nft_manager", cap)
    try:
        wireup.egress_env_for_session("s1", {"allowed_hosts": ["x.com"]})
        assert cap.installed == []          # flag off -> no enforcement
    finally:
        _reset_nft()


def test_nft_installed_when_flag_on_and_cgroup_present(monkeypatch):
    _reset_nft()
    monkeypatch.setenv("JAATO_EGRESS_NFT_ENFORCE", "1")
    monkeypatch.setenv("JAATO_CGROUPS_ROOT", "/sys/fs/cgroup/jaato")
    monkeypatch.setattr(wireup.EgressNftManager, "nft_available", staticmethod(lambda: True))
    cap = _CaptureNft()
    monkeypatch.setattr(wireup, "get_nft_manager", lambda: cap)
    try:
        env, _ = wireup.egress_env_for_session("s1", {"allowed_hosts": ["x.com"]})
        assert len(cap.installed) == 1
        sid, cg, port, host = cap.installed[0]
        assert sid == "s1"
        assert cg == "/sys/fs/cgroup/jaato/jaato-ws-s1"
        assert port == int(env["HTTPS_PROXY"].rsplit(":", 1)[1])
        # #696: the gate is told which address the proxy actually bound
        assert host == "127.0.0.1"
    finally:
        _reset_nft()


def test_nft_flag_on_but_nft_missing_degrades(monkeypatch):
    _reset_nft()
    monkeypatch.setenv("JAATO_EGRESS_NFT_ENFORCE", "1")
    monkeypatch.setattr(wireup.EgressNftManager, "nft_available", staticmethod(lambda: False))
    cap = _CaptureNft()
    monkeypatch.setattr(wireup, "get_nft_manager", lambda: cap)
    try:
        # must not raise, and no install attempted
        wireup.egress_env_for_session("s1", {"allowed_hosts": ["x.com"]})
        assert cap.installed == []
    finally:
        _reset_nft()


# ---- strict posture: deny when the gate cannot be installed --------------

def test_enforce_mode_parsing(monkeypatch):
    monkeypatch.delenv("JAATO_EGRESS_NFT_ENFORCE", raising=False)
    assert wireup._nft_enforce_mode() == "off"
    for val in ("1", "true", "YES", " on "):
        monkeypatch.setenv("JAATO_EGRESS_NFT_ENFORCE", val)
        assert wireup._nft_enforce_mode() == "best_effort"
    for val in ("strict", "Required", "fail-closed"):
        monkeypatch.setenv("JAATO_EGRESS_NFT_ENFORCE", val)
        assert wireup._nft_enforce_mode() == "strict"
    # an unrecognized value must not silently mean "enforced"
    monkeypatch.setenv("JAATO_EGRESS_NFT_ENFORCE", "maybe")
    assert wireup._nft_enforce_mode() == "off"


def test_strict_raises_and_tears_down_when_nft_missing(monkeypatch):
    _reset_nft()
    monkeypatch.setenv("JAATO_EGRESS_NFT_ENFORCE", "strict")
    monkeypatch.setattr(wireup.EgressNftManager, "nft_available",
                        staticmethod(lambda: False))
    try:
        with pytest.raises(EgressEnforcementError):
            wireup.egress_env_for_session("s1", {"allowed_hosts": ["x.com"]})
        # the proxy started for the denied session must not be left listening
        assert wireup.get_manager().get_proxy_url("s1") is None
    finally:
        _reset_nft()


def test_strict_raises_when_install_fails(monkeypatch):
    _reset_nft()
    monkeypatch.setenv("JAATO_EGRESS_NFT_ENFORCE", "strict")
    monkeypatch.setattr(wireup.EgressNftManager, "nft_available",
                        staticmethod(lambda: True))
    cap = _CaptureNft(ok=False)
    monkeypatch.setattr(wireup, "get_nft_manager", lambda: cap)
    try:
        with pytest.raises(EgressEnforcementError):
            wireup.egress_env_for_session("s1", {"allowed_hosts": ["x.com"]})
        assert wireup.get_manager().get_proxy_url("s1") is None
    finally:
        _reset_nft()


def test_best_effort_degrades_when_install_fails(monkeypatch):
    """Default posture is unchanged: a failed install still starts the session."""
    _reset_nft()
    monkeypatch.setenv("JAATO_EGRESS_NFT_ENFORCE", "1")
    monkeypatch.setattr(wireup.EgressNftManager, "nft_available",
                        staticmethod(lambda: True))
    cap = _CaptureNft(ok=False)
    monkeypatch.setattr(wireup, "get_nft_manager", lambda: cap)
    try:
        env, _ = wireup.egress_env_for_session("s1", {"allowed_hosts": ["x.com"]})
        assert env["HTTPS_PROXY"].startswith("http://127.0.0.1:")
    finally:
        _reset_nft()


def test_strict_succeeds_when_gate_installs(monkeypatch):
    _reset_nft()
    monkeypatch.setenv("JAATO_EGRESS_NFT_ENFORCE", "strict")
    monkeypatch.setattr(wireup.EgressNftManager, "nft_available",
                        staticmethod(lambda: True))
    cap = _CaptureNft()
    monkeypatch.setattr(wireup, "get_nft_manager", lambda: cap)
    try:
        env, errs = wireup.egress_env_for_session("s1", {"allowed_hosts": ["x.com"]})
        assert errs == [] and env["HTTPS_PROXY"]
        assert len(cap.installed) == 1
    finally:
        _reset_nft()
